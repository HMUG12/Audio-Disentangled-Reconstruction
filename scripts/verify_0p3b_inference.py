"""A 轨 0.3B 全量 backbone 推理 OOM 测试。

目的：验证 8GB 显存能否跑 0.3B 模型 1500 帧 mel 推理。
如果 OOM，则 M1 必须上云；如果通过，则本地可做 LoRA 推理。
"""
import sys
from pathlib import Path

import torch

FISH_VENV_SITE = Path(__file__).resolve().parent.parent / "third_party" / "fish-speech" / ".venv" / "Lib" / "site-packages"
if FISH_VENV_SITE.exists():
    sys.path.insert(0, str(FISH_VENV_SITE))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adr.configs import ADRConfig, BackboneConfig, ContentConfig, MelodyConfig, TimbreConfig  # noqa: E402
from adr.model import ADRModel  # noqa: E402


def make_full_config() -> ADRConfig:
    """0.3B 全量配置（参照 UniVoice 28层 × 1024 维）。"""
    return ADRConfig(
        content=ContentConfig(vocab_size=607, embed_dim=512, n_layers=6, n_heads=8, ffn_dim=2048, max_phoneme_len=1024),
        melody=MelodyConfig(embed_dim=256, n_layers=3, n_heads=4),
        timbre=TimbreConfig(ref_mel_bins=80, embed_dim=512),
        backbone=BackboneConfig(
            hidden_dim=1024, n_layers=28, n_heads=16, ffn_dim=3072,
            mel_bins=80, max_mel_frames=1500,
        ),
    )


def estimate_params(cfg: ADRConfig) -> int:
    """粗略估算参数量。"""
    c = cfg.content
    m = cfg.melody
    b = cfg.backbone
    t = cfg.timbre
    p = (
        c.vocab_size * c.embed_dim
        + c.n_layers * (4 * c.embed_dim * c.ffn_dim + 4 * c.embed_dim * c.embed_dim)
        + m.embed_dim * (m.n_pitch_bins + m.n_velocity_bins) + m.embed_dim
        + m.n_layers * (4 * m.embed_dim * (4 * m.embed_dim) + 4 * m.embed_dim * m.embed_dim)
        + b.n_layers * (12 * b.hidden_dim * b.ffn_dim + 4 * b.hidden_dim * b.hidden_dim)
        + b.hidden_dim * b.mel_bins * 2  # in_proj + out_proj
        + 2 * 1024  # task_embed
        + t.embed_dim * 1000
    )
    return p


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = make_full_config()
    est = estimate_params(cfg)
    print(f"Estimated params: {est/1e6:.1f}M")

    model = ADRModel(cfg).to(device)
    actual = sum(p.numel() for p in model.parameters())
    print(f"Actual params: {actual/1e6:.1f}M")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory: free={free/1e9:.2f}GB, total={total/1e9:.2f}GB")

    # 训练一步：1500 帧 mel，batch=1
    print("\n=== 训练 forward + backward (T_mel=1500, B=1) ===")
    B = 1
    T_p = 256
    T_n = 128
    T_mel = 1500
    n_mels = 80
    batch = {
        "phoneme_ids": torch.randint(0, 607, (B, T_p), device=device, dtype=torch.long),
        "phoneme_mask": torch.ones(B, T_p, device=device, dtype=torch.bool),
        "pitch_ids": torch.randint(40, 80, (B, T_n), device=device, dtype=torch.long),
        "note_duration": torch.rand(B, T_n, device=device) * 0.5,
        "velocity": torch.randint(0, 32, (B, T_n), device=device, dtype=torch.long),
        "is_singing": torch.tensor([True], device=device, dtype=torch.bool),
        "ref_mel": torch.randn(B, n_mels, 256, device=device),
        "target_mel": torch.randn(B, n_mels, T_mel, device=device),
    }
    try:
        loss = model(batch)
        print(f"Loss: {loss.item():.4f}")
        loss.backward()
        print("Backward OK")
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"Peak GPU memory: {peak:.2f}GB / 8GB")
            if peak > 7.5:
                print("⚠️ 接近显存上限，M1 必须上云")
            else:
                print("✅ 显存余量充足，本地可做 LoRA 微调")
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ OOM: {e}")
        print("→ M1 必须上云")
    except Exception as e:
        print(f"❌ 错误: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
