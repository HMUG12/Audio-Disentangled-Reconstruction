"""Smoke inference with REAL OpenCpop data (M2 方向 A 验证)。

验证:
- 真实数据训练后能推理
- ref + text → wav
- 端到端 pipeline 通

用法:
    python scripts/smoke_infer_opencpop.py
    python scripts/smoke_infer_opencpop.py --ckpt examples/opencpop_train/final.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

import numpy as np

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(REPO / "examples" / "opencpop_train" / "final.pt"))
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--test-dir", default=str(REPO / "data" / "opencpop_npz" / "test"))
    parser.add_argument("--output", default=str(REPO / "examples" / "opencpop_train" / "cloned.wav"))
    parser.add_argument("--n-test", type=int, default=3)
    parser.add_argument("--use-f0", action="store_true",
                        help="注入 GT f0 (SVS 旋律条件), 并对比输出 wav 的 f0 偏差")
    args = parser.parse_args()

    print("=" * 60)
    print("OpenCpop Smoke Inference")
    print("=" * 60)
    print(f"  ckpt:  {args.ckpt}")
    print(f"  test:  {args.test_dir}")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/3] Building model & loading ckpt ...")
    model = build_model(args.preset)
    ckpt = Path(args.ckpt)
    if ckpt.exists():
        import torch
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        is_lora_ckpt = sd.get("use_lora", False) or "lora_state" in sd
        if is_lora_ckpt:
            # LoRA ckpt (lora_state key OR use_lora flag)
            from adr.training.lora import apply_lora, LoRAConfig, load_lora_state_dict
            apply_lora(model, LoRAConfig(rank=8, alpha=16,
                                         target_modules=["out_proj", "linear1", "linear2"]))
            lora_sd = sd.get("lora_state", sd.get("model_state", {}))
            load_lora_state_dict(model, lora_sd)
            print(f"  Loaded LoRA: {ckpt.name} ({len(lora_sd)} keys)")
        elif "model_state" in sd:
            model.load_state_dict(sd["model_state"], strict=False)
            print(f"  Loaded: {ckpt.name}, step={sd.get('step', '?')}, "
                  f"loss={sd.get('best_loss', '?')}")
        else:
            # 可能是 raw state dict
            model.load_state_dict(sd, strict=False)
            print(f"  Loaded raw: {ckpt.name}")
    else:
        print(f"  [!] ckpt not found, using random weights")

    # 2. 尝试加载 BigVGAN
    print("\n[2/3] Loading BigVGAN vocoder ...")
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
    print(f"  Vocoder: {'BigVGAN' if vocoder else 'placeholder'}")

    # 3. 推理测试
    print(f"\n[3/3] Inference on {args.n_test} test samples ...")
    pipe = InferPipeline(
        model, config=InferConfig(n_timesteps=10, use_placeholder_vocoder=(vocoder is None)),
        vocoder=vocoder,
    )

    test_dir = Path(args.test_dir)
    test_files = sorted(test_dir.glob("*.npz"))[: args.n_test]

    # F0 提取器 (仅 --use-f0 时用于评估输出 wav 音高)
    f0_extractor = None
    if args.use_f0:
        from adr.data.f0 import F0Config, F0Extractor
        f0_extractor = F0Extractor(F0Config(sr=22050, hop_length=256))

    for i, npz_path in enumerate(test_files):
        d = np.load(npz_path, allow_pickle=True)
        text = str(d["text"][0])
        # 用自身 wav 作为 ref (因为我们用的是自己的 f0)
        wav_ref = d["waveform"]
        f0_in = d["f0"].astype(np.float32) if args.use_f0 else None
        # 临时保存到 tmp
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        sf.write(tmp_wav, wav_ref, 22050)
        try:
            t0 = time.time()
            wav = pipe.synthesize(text, tmp_wav, n_timesteps=10, f0=f0_in)
            t = time.time() - t0
            msg = (f"  [{i+1}] {npz_path.stem}: '{text[:20]}...' "
                   f"→ {len(wav)/22050:.2f}s wav in {t:.2f}s")
            if f0_in is not None:
                gt_voiced = f0_in[f0_in > 0]
                pred_f0 = f0_extractor(wav, sample_rate=22050)
                pred_voiced = pred_f0[pred_f0 > 0]
                if len(gt_voiced) and len(pred_voiced):
                    msg += (f" | f0: gt={gt_voiced.mean():.1f}Hz "
                            f"pred={pred_voiced.mean():.1f}Hz "
                            f"bias={abs(pred_voiced.mean()-gt_voiced.mean()):.1f}Hz")
            print(msg)
        finally:
            Path(tmp_wav).unlink(missing_ok=True)

    # 4. 保存一个完整示例
    print(f"\n[4/4] Saving example: {args.output} ...")
    if test_files:
        first = test_files[0]
        d = np.load(first, allow_pickle=True)
        text = str(d["text"][0])
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        sf.write(tmp_wav, d["waveform"], 22050)
        try:
            f0_in = d["f0"].astype(np.float32) if args.use_f0 else None
            wav = pipe.synthesize(text, tmp_wav, n_timesteps=10, f0=f0_in)
            pipe.save_wav(wav, args.output)
            print(f"  saved: {args.output} ({len(wav)/22050:.2f}s)")
        finally:
            Path(tmp_wav).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("[Summary] OpenCpop smoke inference")
    print("=" * 60)
    print(f"  Vocoder:  {'BigVGAN (real)' if vocoder else 'placeholder'}")
    print(f"  Tests:    {len(test_files)}")
    print(f"  Output:   {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
