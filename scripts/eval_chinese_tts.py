"""中文 TTS 多文本评估 (M2 方向 A 验证)。

用 1000 样本训练的 medium 模型, 跑多个不同类型中文文本:
- 短句 / 长句 / 数字 / 英文混合 / 问句
- 测量: 时长、采样率、RMS、spectral centroid
- 输出 wav + 报告

用法:
    python scripts/eval_chinese_tts.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from adr.inference.pipeline import InferConfig, InferPipeline
from adr.models.sovits import SoVITS, SoVITSConfig


PRESET_SIZES = {
    "small":  (256, 4, 4, 256, 128),
    "medium": (512, 6, 8, 512, 256),
    "x0p3b":  (1024, 24, 16, 1024, 512),
}


def build_model(preset: str = "medium") -> SoVITS:
    hd, nl, nh, cd, td = PRESET_SIZES[preset]
    cfg = SoVITSConfig(
        hidden_dim=hd, n_layers=nl, n_heads=nh, ffn_dim=hd * 4,
        vocab_size=607, content_dim=cd, timbre_dim=td,
        n_mels=80, sample_rate=22050, hop_length=256,
    )
    return SoVITS(cfg)


# 多样化中文测试集
TEST_TEXTS = [
    # 1. 短句
    ("短句-问候", "你好,很高兴认识你。"),
    # 2. 中等
    ("中等-介绍", "ADR 是一个低资源快速克隆训练框架,只需要 8GB 显存。"),
    # 3. 长句
    ("长句-说明", "在当今的人工智能领域,语音合成技术已经取得了巨大的突破,从早期的拼接合成到现在的端到端神经网络,质量已经达到了接近人类水平。"),
    # 4. 数字
    ("数字", "现在的时间是 2026 年 8 月 8 日 22 点 45 分,温度 26.5 度。"),
    # 5. 中英混合
    ("中英混合", "我们的模型基于 PyTorch 框架,使用了 BigVGAN 作为 vocoder。"),
    # 6. 问句
    ("问句", "你觉得未来的 AI 真的能够理解人类的情感吗?"),
    # 7. 古诗
    ("古诗", "床前明月光,疑是地上霜。举头望明月,低头思故乡。"),
    # 8. 拟声
    ("拟声", "雨声滴答滴答,风吹得树叶沙沙响。"),
    # 9. 数字电话
    ("数字电话", "我的电话号码是一三八零零一二三四五六七。"),
    # 10. 歌词 (OpenCpop 风格)
    ("歌词", "感受停在我发端的指尖,雨淋湿了天空灰得更讲究。"),
]


def compute_metrics(wav: np.ndarray, sr: int) -> dict:
    """计算 wav 质量指标。"""
    if len(wav) < 100:
        return {"valid": False, "reason": "too short"}

    # RMS
    rms = float(np.sqrt(np.mean(wav ** 2)))

    # Spectral centroid
    if rms > 0.001:  # 跳过纯零
        spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
        freqs = np.fft.rfftfreq(len(wav), 1 / sr)
        if spec.sum() > 0:
            centroid = float(np.sum(freqs * spec) / np.sum(spec))
        else:
            centroid = 0.0
    else:
        centroid = 0.0

    return {
        "valid": True,
        "duration_sec": round(len(wav) / sr, 2),
        "n_samples": int(len(wav)),
        "rms": round(rms, 4),
        "spectral_centroid_hz": round(centroid, 1),
        "max_abs": round(float(np.abs(wav).max()), 4),
        "is_silent": rms < 0.001,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(REPO / "examples" / "opencpop_train_1000" / "final.pt"))
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--n-timesteps", type=int, default=20)
    parser.add_argument("--ref", default=None, help="参考音频 (默认从 test set 取第一个)")
    parser.add_argument("--output-dir", default=str(REPO / "examples" / "tts_eval"))
    args = parser.parse_args()

    print("=" * 70)
    print("Chinese TTS Multi-Text Evaluation")
    print("=" * 70)
    print(f"  ckpt:    {args.ckpt}")
    print(f"  preset:  {args.preset}")
    print(f"  output:  {args.output_dir}")
    print(f"  texts:   {len(TEST_TEXTS)}")
    print("=" * 70)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    print("\n[1/3] Loading model ...")
    model = build_model(args.preset)
    ckpt = Path(args.ckpt)
    if ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        is_lora_ckpt = sd.get("use_lora", False) or "lora_state" in sd
        if is_lora_ckpt:
            from adr.training.lora import apply_lora, LoRAConfig, load_lora_state_dict
            apply_lora(model, LoRAConfig(rank=8, alpha=16,
                                         target_modules=["out_proj", "linear1", "linear2"]))
            lora_sd = sd.get("lora_state", sd.get("model_state", {}))
            load_lora_state_dict(model, lora_sd)
            print(f"  Loaded LoRA: {ckpt.name}")
        elif "model_state" in sd:
            model.load_state_dict(sd["model_state"], strict=False)
            print(f"  Loaded: {ckpt.name}")
    else:
        print(f"  [!] ckpt not found, using random weights")

    # 2. 加载 BigVGAN
    print("\n[2/3] Loading BigVGAN vocoder ...")
    vocoder = None
    try:
        from adr.vocoder.bigvgan import BigVGANVocoder, _find_bigvgan_dir
        bigvgan_dir = _find_bigvgan_dir()
        if bigvgan_dir:
            vocoder = BigVGANVocoder.from_pretrained(str(bigvgan_dir), device="auto")
            if not vocoder.is_loaded():
                vocoder = None
    except Exception as e:
        print(f"  BigVGAN load failed: {e}")
    print(f"  Vocoder: {'BigVGAN' if vocoder else 'placeholder'}")

    # 3. ref wav
    print("\n[3/3] Running inference on test texts ...")
    if args.ref:
        ref_wav = args.ref
    else:
        # 从 OpenCpop test set 取第一个 wav
        test_dir = REPO / "data" / "opencpop_npz" / "test"
        first = sorted(test_dir.glob("*.npz"))[0]
        d = np.load(first, allow_pickle=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_wav = f.name
        import soundfile as sf
        sf.write(ref_wav, d["waveform"], 22050)
        print(f"  Using ref: {first.stem}")

    # 4. Pipeline
    pipe = InferPipeline(
        model, config=InferConfig(n_timesteps=args.n_timesteps,
                                  use_placeholder_vocoder=(vocoder is None)),
        vocoder=vocoder,
    )

    # 5. 推理所有文本
    results = []
    print(f"\n  Running {len(TEST_TEXTS)} test texts ...")
    print(f"  {'#':<3} {'类别':<12} {'文本 (first 20)':<22} {'时长':<8} {'RMS':<8} {'谱质心':<10} {'状态'}")
    print("  " + "-" * 80)

    for i, (label, text) in enumerate(TEST_TEXTS):
        t0 = time.time()
        try:
            wav = pipe.synthesize(text, ref_wav, n_timesteps=args.n_timesteps)
            elapsed = time.time() - t0
            metrics = compute_metrics(wav, 22050)
            metrics["text"] = text
            metrics["label"] = label
            metrics["infer_time_sec"] = round(elapsed, 2)
            results.append(metrics)

            # 保存 wav
            wav_path = out_dir / f"utt_{i+1:02d}_{label}.wav"
            pipe.save_wav(wav, str(wav_path))

            # 打印
            valid = "✓" if metrics.get("valid", False) and not metrics.get("is_silent", False) else "✗"
            print(f"  {i+1:<3} {label:<12} {text[:20]:<22} "
                  f"{metrics.get('duration_sec', 0):<8.2f} {metrics.get('rms', 0):<8.4f} "
                  f"{metrics.get('spectral_centroid_hz', 0):<10.1f} {valid}")
        except Exception as e:
            print(f"  {i+1:<3} {label:<12} {text[:20]:<22} [FAIL] {e}")
            results.append({"label": label, "text": text, "error": str(e)})

    # 6. 统计
    valid_results = [r for r in results if r.get("valid", False) and not r.get("is_silent", False)]
    n_valid = len(valid_results)
    n_total = len(results)

    summary = {
        "ckpt": str(ckpt),
        "preset": args.preset,
        "vocoder": "BigVGAN" if vocoder else "placeholder",
        "n_timesteps": args.n_timesteps,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_silent": sum(1 for r in results if r.get("is_silent")),
        "n_fail": sum(1 for r in results if "error" in r),
        "metrics": results,
    }
    if n_valid > 0:
        durs = [r["duration_sec"] for r in valid_results]
        rmss = [r["rms"] for r in valid_results]
        cents = [r["spectral_centroid_hz"] for r in valid_results]
        summary["aggregate"] = {
            "duration": {
                "min": round(min(durs), 2),
                "max": round(max(durs), 2),
                "mean": round(sum(durs) / len(durs), 2),
            },
            "rms": {
                "min": round(min(rmss), 4),
                "max": round(max(rmss), 4),
                "mean": round(sum(rmss) / len(rmss), 4),
            },
            "spectral_centroid_hz": {
                "min": round(min(cents), 1),
                "max": round(max(cents), 1),
                "mean": round(sum(cents) / len(cents), 1),
            },
        }

    # 7. 写报告
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("[Summary] Chinese TTS Multi-Text Evaluation")
    print("=" * 70)
    print(f"  总数: {n_total} | 有效: {n_valid} | 静音: {summary['n_silent']} | 失败: {summary['n_fail']}")
    if n_valid > 0:
        agg = summary["aggregate"]
        print(f"  时长:    mean={agg['duration']['mean']}s  range=[{agg['duration']['min']}, {agg['duration']['max']}]")
        print(f"  RMS:     mean={agg['rms']['mean']}  range=[{agg['rms']['min']}, {agg['rms']['max']}]")
        print(f"  谱质心:  mean={agg['spectral_centroid_hz']['mean']}Hz  range=[{agg['spectral_centroid_hz']['min']}, {agg['spectral_centroid_hz']['max']}]")
    print(f"  wav 目录: {out_dir}")
    print(f"  报告:    {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
