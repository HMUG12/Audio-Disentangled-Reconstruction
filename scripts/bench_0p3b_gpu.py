"""Real GPU memory benchmark for 0.3B model on 8GB GPU.

Compares:
1. Full finetune (no LoRA, no quant): all params + grad + Adam state
2. LoRA only: only LoRA params trainable
3. QLoRA: 4-bit base + LoRA

Outputs actual measured peak GPU memory for each mode.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn

from adr.models.sovits import SoVITS, SoVITSConfig
from adr.training.lora import LoRAConfig, apply_lora
from adr.training.efficient import print_trainable_parameters, get_memory_stats, is_bnb_available


def reset_gpu():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def make_model(preset="x0p3b"):
    """Build model with given preset."""
    sizes = {
        "medium": (512, 6, 8, 512, 256),
        "x0p3b":  (1024, 24, 16, 1024, 512),
    }
    hd, nl, nh, cd, td = sizes[preset]
    cfg = SoVITSConfig(
        hidden_dim=hd, n_layers=nl, n_heads=nh, ffn_dim=hd*4,
        vocab_size=607, content_dim=cd, timbre_dim=td,
        n_mels=80, sample_rate=22050, hop_length=256,
    )
    return SoVITS(cfg).cuda()


def make_synthetic_batch(B=2, T=64, n_mels=80, T_mel=128, T_ref=200):
    """Make a synthetic batch on GPU for memory testing."""
    return {
        "phoneme_ids": torch.randint(1, 600, (B, T), device="cuda"),
        "phoneme_mask": torch.ones(B, T, dtype=torch.bool, device="cuda"),
        "ref_mel": torch.randn(B, n_mels, T_ref, device="cuda"),
        "target_mel": torch.randn(B, n_mels, T_mel, device="cuda"),
        "target_durations": torch.randint(1, 5, (B, T), device="cuda"),
    }


def bench_full_train(model, n_steps=5):
    """Benchmark full finetune memory."""
    import traceback
    reset_gpu()
    model.train()
    # All params trainable
    for p in model.parameters():
        p.requires_grad = True
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated() / 1024**2
    peak_mem = base_mem
    try:
        for i in range(n_steps):
            batch = make_synthetic_batch()
            out = model(batch)
            loss = out["loss"]
            loss.backward()
            optim.step()
            optim.zero_grad()
            torch.cuda.synchronize()
            cur = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            peak_mem = max(peak_mem, peak)
        success = True
        error = None
    except torch.cuda.OutOfMemoryError as e:
        success = False
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        error = f"OOM at peak {peak_mem:.0f}MB"
    except Exception as e:
        success = False
        error = f"{type(e).__name__}: {e}"
        print("bench_full_train ERROR:", error, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    return {
        "mode": "full_finetune",
        "success": success,
        "error": error,
        "base_mem_MB": base_mem,
        "peak_mem_MB": peak_mem,
    }


def bench_lora(model, rank=8, n_steps=5):
    """Benchmark LoRA finetune memory."""
    import traceback
    model = model.cuda()  # re-init if needed
    # Apply LoRA
    cfg = LoRAConfig(rank=rank, alpha=rank*2, target_modules=["out_proj", "linear1", "linear2"])
    apply_lora(model, cfg)
    # Optimizer only on trainable (LoRA) params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=1e-4)
    # 在 setup 完成后 reset: apply_lora 期间旧 Linear 与 LoRA 副本短暂共存,
    # 会虚高 peak (~2GB for 0.3B), 不属于训练显存
    reset_gpu()
    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated() / 1024**2
    peak_mem = base_mem
    error = None
    try:
        for i in range(n_steps):
            batch = make_synthetic_batch()
            out = model(batch)
            loss = out["loss"]
            loss.backward()
            optim.step()
            optim.zero_grad()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**2
            peak_mem = max(peak_mem, peak)
        success = True
    except torch.cuda.OutOfMemoryError as e:
        success = False
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        error = f"OOM at peak {peak_mem:.0f}MB"
    except Exception as e:
        success = False
        error = f"{type(e).__name__}: {e}"
        print("bench_lora ERROR:", error, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    return {
        "mode": f"lora_r{rank}",
        "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "success": success,
        "error": error,
        "base_mem_MB": base_mem,
        "peak_mem_MB": peak_mem,
    }


def bench_qlora(model, rank=8, n_steps=5):
    """Benchmark QLoRA (4-bit base + LoRA) memory.

    Note: 真实 QLoRA 需在 4bit weight 上加 LoRA 旁路 (PEFT 模式),
    这里用简化版: 4bit 基座 + 单独 LoRA 旁路。
    由于我们 zero-dep LoRA 套 Linear4bit 时会冲突, 改测:
    1. 仅 4bit 量化 (base 显存)
    2. 4bit 量化 + LoRA (在未量化部分)
    """
    import traceback
    if not is_bnb_available():
        return {"mode": "qlora", "success": False, "error": "bitsandbytes not installed"}
    from adr.training.efficient import quantize_4bit
    model = model.cuda()
    # 1. 先应用 LoRA (replacing 目标 Linear)
    cfg = LoRAConfig(rank=rank, alpha=rank*2, target_modules=["out_proj", "linear1", "linear2"])
    apply_lora(model, cfg)
    # 2. 再 4-bit 量化 (量化 LoRA 冻结基座 + MHA qkv + 其余 Linear)
    model = quantize_4bit(model, quant_type="nf4", compute_dtype=torch.float16)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return {"mode": "qlora_r4bit", "success": False, "error": "no trainable params"}
    optim = torch.optim.AdamW(trainable, lr=1e-4)
    # 在 setup 完成后 reset: apply_lora/quantize 的短暂副本不属于训练显存
    reset_gpu()
    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated() / 1024**2
    peak_mem = base_mem
    error = None
    try:
        for i in range(n_steps):
            batch = make_synthetic_batch()
            out = model(batch)
            loss = out["loss"]
            loss.backward()
            optim.step()
            optim.zero_grad()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**2
            peak_mem = max(peak_mem, peak)
        success = True
    except torch.cuda.OutOfMemoryError as e:
        success = False
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        error = f"OOM at peak {peak_mem:.0f}MB"
    except Exception as e:
        success = False
        error = f"{type(e).__name__}: {e}"
        print("bench_qlora ERROR:", error, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    return {
        "mode": f"qlora_r{rank}_4bit",
        "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "success": success,
        "error": error,
        "base_mem_MB": base_mem,
        "peak_mem_MB": peak_mem,
    }


def main():
    print("=" * 70)
    print("Real GPU Memory Benchmark: 0.3B Model on 8GB RTX A2000")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  bitsandbytes available: {is_bnb_available()}")
    print("=" * 70)

    presets = ["medium", "x0p3b"]
    all_results = []
    for preset in presets:
        print(f"\n{'='*70}\nPreset: {preset}\n{'='*70}")
        # Full
        model = make_model(preset)
        n_p = sum(p.numel() for p in model.parameters())
        print(f"  model: {n_p/1e6:.1f}M params")
        del model

        # Mode 1: full finetune
        model = make_model(preset)
        r1 = bench_full_train(model, n_steps=3)
        r1["preset"] = preset
        r1["model_params_M"] = n_p / 1e6
        all_results.append(r1)
        del model
        reset_gpu()

        # Mode 2: LoRA r=8
        model = make_model(preset)
        r2 = bench_lora(model, rank=8, n_steps=3)
        r2["preset"] = preset
        r2["model_params_M"] = n_p / 1e6
        all_results.append(r2)
        del model
        reset_gpu()

        # Mode 3: QLoRA r=8 (if bnb available)
        model = make_model(preset)
        r3 = bench_qlora(model, rank=8, n_steps=3)
        r3["preset"] = preset
        r3["model_params_M"] = n_p / 1e6
        all_results.append(r3)
        del model
        reset_gpu()

    # Summary
    print(f"\n\n{'='*70}\nBENCHMARK RESULTS\n{'='*70}")
    print(f"{'Preset':<10} {'Mode':<20} {'Params(M)':<10} {'Trainable':<12} {'Peak(MB)':<10} {'8GB':<5}")
    print("-" * 70)
    for r in all_results:
        n_tr = r.get("n_trainable", 0)
        tr_str = f"{n_tr/1e6:.4f}M" if n_tr else "ALL"
        peak = r["peak_mem_MB"]
        fits = "✓" if peak < 8*1024 else "✗"
        ok = "" if r["success"] else " (OOM)"
        print(f"{r['preset']:<10} {r['mode']:<20} {r['model_params_M']:<10.1f} {tr_str:<12} {peak:<10.0f} {fits}{ok}")
    print("=" * 70)
    return all_results


if __name__ == "__main__":
    results = main()
    import json
    out = REPO / "examples" / "03_lora_finetune" / "gpu_benchmark.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] saved to {out}")
