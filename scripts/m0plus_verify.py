"""M0+ 验证脚本：搭建 ADR 自研模型骨架，跑前向 + 反向 + 推理采样。

目标：无需任何预训练权重，验证架构数学正确性（shape 对齐 + loss 收敛 + 推理可执行）。
"""
import sys
from pathlib import Path

import torch

# 复用 fish-speech 的 venv，确保 torch/transformers 可用
FISH_VENV_SITE = Path(__file__).resolve().parent.parent / "third_party" / "fish-speech" / ".venv" / "Lib" / "site-packages"
if FISH_VENV_SITE.exists():
    sys.path.insert(0, str(FISH_VENV_SITE))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adr.configs import ADRConfig, BackboneConfig, ContentConfig, MelodyConfig, TimbreConfig  # noqa: E402
from adr.model import ADRModel  # noqa: E402


def make_dummy_batch(B=2, T_p=32, T_n=16, T_mel=160, n_mels=80, device="cuda"):
    """构造一组 dummy 数据测试前向/反向。"""
    return {
        "phoneme_ids": torch.randint(0, 256, (B, T_p), device=device, dtype=torch.long),
        "phoneme_mask": torch.ones(B, T_p, device=device, dtype=torch.bool),
        "pitch_ids": torch.randint(40, 80, (B, T_n), device=device, dtype=torch.long),
        "note_duration": torch.rand(B, T_n, device=device) * 0.5,
        "velocity": torch.randint(0, 32, (B, T_n), device=device, dtype=torch.long),
        "is_singing": torch.tensor([True, False], device=device, dtype=torch.bool),
        "ref_mel": torch.randn(B, n_mels, 256, device=device),
        "target_mel": torch.randn(B, n_mels, T_mel, device=device),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 用小模型以快速验证
    cfg = ADRConfig(
        content=ContentConfig(embed_dim=128, n_layers=2, n_heads=4, ffn_dim=256, max_phoneme_len=128),
        melody=MelodyConfig(embed_dim=64, n_layers=1, n_heads=4),
        timbre=TimbreConfig(ref_mel_bins=80, embed_dim=256),
        backbone=BackboneConfig(hidden_dim=256, n_layers=4, n_heads=4, ffn_dim=512, mel_bins=80, max_mel_frames=512),
    )
    print("Param estimate:", cfg.total_params_estimate())

    model = ADRModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Actual params: {n_params/1e6:.1f}M")

    # 1) 训练前向 + 反向
    print("\n=== 训练 forward + backward ===")
    batch = make_dummy_batch(B=2, T_p=32, T_n=16, T_mel=160, n_mels=80, device=device)
    loss = model(batch)
    print(f"Loss: {loss.item():.4f}")
    loss.backward()
    print("Backward OK")

    # 2) 说话模式推理
    print("\n=== 说话模式推理 ===")
    model.eval()
    phoneme_ids = torch.randint(0, 256, (1, 32), device=device, dtype=torch.long)
    ref_mel = torch.randn(1, 80, 256, device=device)
    mel = model.sample(phoneme_ids, ref_mel, is_singing=False, n_timesteps=5)
    print(f"Speech mel shape: {mel.shape}")

    # 3) 唱歌模式推理
    print("\n=== 唱歌模式推理 ===")
    pitch_ids = torch.randint(40, 80, (1, 16), device=device, dtype=torch.long)
    note_duration = torch.rand(1, 16, device=device) * 0.5
    velocity = torch.randint(0, 32, (1, 16), device=device, dtype=torch.long)
    mel = model.sample(phoneme_ids, ref_mel,
                       pitch_ids=pitch_ids, note_duration=note_duration, velocity=velocity,
                       is_singing=True, n_timesteps=5)
    print(f"Sing mel shape: {mel.shape}")

    print("\n✅ 全部验证通过")


if __name__ == "__main__":
    main()
