"""对比 1000 vs 3550 训练样本的 TTS 质量。

对同一组测试文本,跑两个模型,统计指标差异。
"""
from __future__ import annotations

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


TEST_TEXTS = [
    "你好,很高兴认识你。",
    "ADR 是一个低资源快速克隆训练框架,只需要 8GB 显存。",
    "在当今的人工智能领域,语音合成技术已经取得了巨大的突破。",
    "现在的时间是 2026 年 8 月 8 日 22 点 45 分,温度 26.5 度。",
    "我们的模型基于 PyTorch 框架,使用了 BigVGAN 作为 vocoder。",
    "你觉得未来的 AI 真的能够理解人类的情感吗?",
    "床前明月光,疑是地上霜。举头望明月,低头思故乡。",
    "雨声滴答滴答,风吹得树叶沙沙响。",
    "我的电话号码是一三八零零一二三四五六七。",
    "感受停在我发端的指尖,雨淋湿了天空灰得更讲究。",
]


def compute_metrics(wav: np.ndarray, sr: int) -> dict:
    if len(wav) < 100:
        return {"valid": False, "reason": "too short"}
    rms = float(np.sqrt(np.mean(wav ** 2)))
    spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
    freqs = np.fft.rfftfreq(len(wav), 1 / sr)
    centroid = float(np.sum(freqs * spec) / np.sum(spec)) if spec.sum() > 0 else 0.0
    return {
        "valid": True,
        "duration_sec": round(len(wav) / sr, 2),
        "rms": round(rms, 4),
        "spectral_centroid_hz": round(centroid, 1),
        "is_silent": rms < 0.001,
    }


def run_model(ckpt_path: Path, preset: str, ref_wav: str, n_timesteps: int = 20):
    """跑一个模型,返回所有文本的指标。"""
    model = build_model(preset)
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        is_lora = sd.get("use_lora", False) or "lora_state" in sd
        if is_lora:
            from adr.training.lora import apply_lora, LoRAConfig, load_lora_state_dict
            apply_lora(model, LoRAConfig(rank=8, alpha=16,
                                         target_modules=["out_proj", "linear1", "linear2"]))
            load_lora_state_dict(model, sd.get("lora_state", sd.get("model_state", {})))
        elif "model_state" in sd:
            model.load_state_dict(sd["model_state"], strict=False)

    # 加载 BigVGAN
    vocoder = None
    try:
        from adr.vocoder.bigvgan import BigVGANVocoder, _find_bigvgan_dir
        bigvgan_dir = _find_bigvgan_dir()
        if bigvgan_dir:
            vocoder = BigVGANVocoder.from_pretrained(str(bigvgan_dir), device="auto")
            if not vocoder.is_loaded():
                vocoder = None
    except Exception:
        pass

    pipe = InferPipeline(
        model, config=InferConfig(n_timesteps=n_timesteps,
                                  use_placeholder_vocoder=(vocoder is None)),
        vocoder=vocoder,
    )

    results = []
    for text in TEST_TEXTS:
        t0 = time.time()
        try:
            wav = pipe.synthesize(text, ref_wav, n_timesteps=n_timesteps)
            elapsed = time.time() - t0
            m = compute_metrics(wav, 22050)
            m["text"] = text
            m["infer_time_sec"] = round(elapsed, 2)
            results.append(m)
        except Exception as e:
            results.append({"text": text, "error": str(e)})
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt1", required=True, help="模型 1 路径")
    parser.add_argument("--ckpt2", required=True, help="模型 2 路径")
    parser.add_argument("--label1", default="Model A")
    parser.add_argument("--label2", default="Model B")
    args = parser.parse_args()

    print("=" * 70)
    print(f"对比: {args.label1} vs {args.label2}")
    print("=" * 70)

    # ref
    test_dir = REPO / "data" / "opencpop_npz" / "test"
    first = sorted(test_dir.glob("*.npz"))[0]
    d = np.load(first, allow_pickle=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_wav = f.name
    import soundfile as sf
    sf.write(ref_wav, d["waveform"], 22050)

    out_dir = REPO / "examples" / f"compare_{Path(args.ckpt1).parent.name}_vs_{Path(args.ckpt2).parent.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 模型 1
    print(f"\n[1/2] {args.label1} ...")
    res_a = run_model(Path(args.ckpt1), "medium", ref_wav, n_timesteps=20)

    # 模型 2
    print(f"[2/2] {args.label2} ...")
    res_b = run_model(Path(args.ckpt2), "medium", ref_wav, n_timesteps=20)

    # 统计
    def agg(results, key):
        vals = [r[key] for r in results if r.get("valid") and key in r]
        if not vals:
            return None
        return {
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "mean": round(sum(vals) / len(vals), 4),
        }

    report = {
        "ckpt1": args.ckpt1, "ckpt2": args.ckpt2,
        "label1": args.label1, "label2": args.label2,
        "n_test": len(TEST_TEXTS),
        "results_a": res_a,
        "results_b": res_b,
        "agg_a": {
            "duration_sec": agg(res_a, "duration_sec"),
            "rms": agg(res_a, "rms"),
            "spectral_centroid_hz": agg(res_a, "spectral_centroid_hz"),
        },
        "agg_b": {
            "duration_sec": agg(res_b, "duration_sec"),
            "rms": agg(res_b, "rms"),
            "spectral_centroid_hz": agg(res_b, "spectral_centroid_hz"),
        },
    }

    # 打印对比
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {args.label1:<20} {args.label2:<20} {'Δ':<10}")
    print("-" * 70)
    for metric_name, key in [("Duration (sec)", "duration_sec"),
                             ("RMS", "rms"),
                             ("Spectral Centroid (Hz)", "spectral_centroid_hz")]:
        a = report["agg_a"][key]
        b = report["agg_b"][key]
        if a and b:
            delta = b["mean"] - a["mean"]
            delta_str = f"{delta:+.3f}" if abs(delta) < 10 else f"{delta:+.1f}"
            print(f"  {metric_name:<23} {a['mean']:<20.4f} {b['mean']:<20.4f} {delta_str:<10}")
    print("=" * 70)

    # 保存
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\nReport saved: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
