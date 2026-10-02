"""M8.1: 合成样张集渲染器 (防噪声防线核心)。

对任意 checkpoint 一键渲染固定样张集, 输出 wav + 指标报告:

- 10 条固定文本 (短/长/快/慢/数字/中英混读/歌词) × 2 条固定参考音频 = 20 条 wav
- 可懂度: faster-whisper 回译 + CER (复用 training.callbacks.compute_cer)
- 音色相似度: 模型自身 TimbreEncoder embedding 余弦 (代理指标;
  resemblyzer 在 Windows 装不上 webrtcvad, 故用自洽的相对指标)

用法:
    python scripts/render_sample_pack.py <ckpt> [--out output/sample_pack] [--no-metrics]

一期教训: 指标绿 ≠ 能听。本脚本生成的 wav 必须人工听审, 指标只做粗筛。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ---- 固定样张集 (不随训练变化, 保证跨版本可比) ----
SAMPLE_TEXTS = [
    ("short", "你好"),
    ("greeting", "欢迎来到新一代语音克隆框架"),
    ("medium", "今天的天气真不错,我们一起去公园散步吧"),
    ("long", "人工智能技术正在以前所未有的速度发展,语音合成作为其中的重要分支,已经从实验室走进了千家万户的日常生活"),
    ("fast", "一二三四五六七八九十,快快快,再快一点"),
    ("slow", "夜 色 如 水 , 月 光 静 静 地 洒 在 窗 前"),
    ("digits", "我的电话号码是一三八零零八六零零零"),
    ("mixed", "这个 API 的 response time 只需要两百毫秒"),
    ("lyric", "远方的客人请你留下来,这里的山水等你来"),
    ("question", "你真的觉得这样做是对的吗?"),
]

SAMPLE_REFS = [
    ("ref_a", "data/opencpop/wavs/2100003756.wav"),
    ("ref_b", "data/opencpop/wavs/2044001628.wav"),
]


def render_pack(ckpt: str, out_dir: Path, with_metrics: bool = True) -> dict:
    import numpy as np
    import soundfile as sf
    import torch

    from adr.inference.pipeline import InferConfig, InferPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wavs").mkdir(exist_ok=True)

    pipe = InferPipeline.from_checkpoint(ckpt, config=InferConfig(n_timesteps=20))
    report = {
        "ckpt": ckpt,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(pipe.device),
        "samples": [],
        "summary": {},
    }

    # 预提取参考音色 embedding (用模型自身 TimbreEncoder, 代理指标)
    ref_embs = {}
    if with_metrics:
        with torch.no_grad():
            for ref_name, ref_path in SAMPLE_REFS:
                mel = torch.from_numpy(pipe.ref_audio_to_mel(ref_path)) \
                    .unsqueeze(0).float().to(pipe.device)
                ref_embs[ref_name] = pipe.backbone.timbre_encoder(mel)

    asr = None
    for text_name, text in SAMPLE_TEXTS:
        for ref_name, ref_path in SAMPLE_REFS:
            tag = f"{text_name}__{ref_name}"
            try:
                t0 = time.time()
                wav = pipe.synthesize(text, ref_path)
                dur = len(wav) / pipe.config.ref_sample_rate
                synth_ms = (time.time() - t0) * 1000
                wav_path = out_dir / "wavs" / f"{tag}.wav"
                sf.write(str(wav_path), wav, pipe.config.ref_sample_rate)

                item = {
                    "tag": tag, "text": text, "ref": ref_path,
                    "duration_sec": round(dur, 2),
                    "synth_ms": round(synth_ms, 0),
                    "rms": round(float(np.sqrt((wav ** 2).mean())), 4),
                    "wav": str(wav_path.relative_to(out_dir)),
                }

                if with_metrics:
                    # 音色相似度 (TimbreEncoder 代理)
                    from adr.utils.audio import compute_mel
                    with torch.no_grad():
                        mel = compute_mel(
                            wav, sample_rate=pipe.config.ref_sample_rate,
                            n_mels=pipe.config.n_mels,
                            hop_length=pipe.config.hop_length,
                        )
                        emb = pipe.backbone.timbre_encoder(
                            torch.from_numpy(mel).unsqueeze(0).float().to(pipe.device))
                        cos = torch.nn.functional.cosine_similarity(
                            emb, ref_embs[ref_name]).item()
                        item["timbre_cos"] = round(cos, 3)

                report["samples"].append(item)
                print(f"  [OK] {tag}: {dur:.2f}s ({synth_ms:.0f}ms) rms={item['rms']}")
            except Exception as e:
                report["samples"].append({"tag": tag, "text": text, "error": str(e)})
                print(f"  [FAIL] {tag}: {e}")

    # ASR 回译 CER (20 条全部合成完后统一跑, 模型只加载一次)
    if with_metrics:
        try:
            from adr.data.asr import ASR, ASRConfig
            from adr.training.callbacks import compute_cer

            # 强制 CPU: 本机 faster-whisper 走 CUDA 缺 cublas64_12.dll
            asr = ASR(ASRConfig(model_size="tiny", device="cpu", compute_type="int8"))
            for item in report["samples"]:
                if "error" in item:
                    continue
                try:
                    wav_path = out_dir / item["wav"]
                    res = asr.transcribe(str(wav_path))
                    item["asr_text"] = res.text
                    item["cer"] = round(compute_cer(item["text"], res.text), 3)
                except Exception as e:
                    item["cer_error"] = str(e)
        except Exception as e:
            report["summary"]["asr_unavailable"] = str(e)
            print(f"  [!] ASR 不可用, 跳过 CER: {e}")

    # 汇总
    ok = [s for s in report["samples"] if "error" not in s]
    cers = [s["cer"] for s in ok if "cer" in s]
    coss = [s["timbre_cos"] for s in ok if "timbre_cos" in s]
    report["summary"].update({
        "n_ok": len(ok), "n_fail": len(report["samples"]) - len(ok),
        "cer_mean": round(sum(cers) / len(cers), 3) if cers else None,
        "timbre_cos_mean": round(sum(coss) / len(coss), 3) if coss else None,
        "rms_mean": round(sum(s["rms"] for s in ok) / len(ok), 4) if ok else None,
    })

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def score_existing(out_dir: Path) -> dict:
    """对 wavs/ 目录重算指标 (不重合成) — 合成与打分解耦。

    从 SAMPLE_TEXTS × SAMPLE_REFS 重建条目, 不信任旧 report (可能是失败残留)。
    """
    import numpy as np
    import soundfile as sf

    report_path = out_dir / "report.json"
    old = json.load(open(report_path, encoding="utf-8")) if report_path.exists() else {}
    report = {"ckpt": old.get("ckpt", "?"), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "device": "score-only", "samples": [], "summary": {}}

    from adr.data.asr import ASR, ASRConfig
    from adr.training.callbacks import compute_cer

    # 强制 CPU: 本机 faster-whisper 走 CUDA 缺 cublas64_12.dll; CPU int8 足够快
    asr = ASR(ASRConfig(model_size="tiny", device="cpu", compute_type="int8"))
    for text_name, text in SAMPLE_TEXTS:
        for ref_name, ref_path in SAMPLE_REFS:
            tag = f"{text_name}__{ref_name}"
            wav_path = out_dir / "wavs" / f"{tag}.wav"
            if not wav_path.exists():
                continue
            wav, sr = sf.read(str(wav_path))
            item = {"tag": tag, "text": text, "ref": ref_path,
                    "duration_sec": round(len(wav) / sr, 2),
                    "rms": round(float(np.sqrt((wav ** 2).mean())), 4),
                    "wav": f"wavs/{tag}.wav"}
            try:
                res = asr.transcribe(str(wav_path))
                item["asr_text"] = res.text
                item["cer"] = round(compute_cer(text, res.text), 3)
                print(f"  [CER={item['cer']:.2f}] {tag}")
            except Exception as e:
                item["cer_error"] = str(e)
                print(f"  [CER-FAIL] {tag}: {e}")
            report["samples"].append(item)

    ok = report["samples"]
    cers = [s["cer"] for s in ok if "cer" in s]
    report["summary"].update({
        "n_ok": len(ok), "n_fail": 0,
        "cer_mean": round(sum(cers) / len(cers), 3) if cers else None,
        "rms_mean": round(sum(s["rms"] for s in ok) / len(ok), 4) if ok else None,
    })
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description="M8.1 合成样张集渲染器")
    ap.add_argument("ckpt", nargs="?", help="checkpoint 路径 (--score-only 时可省)")
    ap.add_argument("--out", default="output/sample_pack", help="输出目录")
    ap.add_argument("--no-metrics", action="store_true", help="只出 wav, 不算指标")
    ap.add_argument("--score-only", action="store_true",
                    help="不重合成, 只对已有 wavs 重算指标")
    args = ap.parse_args()

    if args.score_only:
        report = score_existing(Path(args.out))
    else:
        if not args.ckpt:
            ap.error("需要 ckpt 参数 (或 --score-only)")
        print("=" * 56)
        print("M8.1 合成样张集")
        print("=" * 56)
        print(f"  ckpt: {args.ckpt}")
        print(f"  out:  {args.out}")
        print(f"  规模: {len(SAMPLE_TEXTS)} 文本 × {len(SAMPLE_REFS)} 参考 = "
              f"{len(SAMPLE_TEXTS) * len(SAMPLE_REFS)} 条")
        print("=" * 56)
        report = render_pack(args.ckpt, Path(args.out), with_metrics=not args.no_metrics)

    s = report["summary"]
    print("\n" + "=" * 56)
    print("[Summary]")
    print("=" * 56)
    print(f"  成功/失败:  {s.get('n_ok')}/{s.get('n_fail')}")
    print(f"  CER 均值:   {s.get('cer_mean')}  (越小越好, ≤0.15 达标)")
    print(f"  音色相似度: {s.get('timbre_cos_mean')}  (越大越好)")
    print(f"  RMS 均值:   {s.get('rms_mean')}  (<0.01 疑似静音/噪声)")
    print(f"  报告: {args.out}/report.json")
    print("  注意: 指标只做粗筛, wav 必须人工听审!")


if __name__ == "__main__":
    main()
