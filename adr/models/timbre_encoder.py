"""音色编码器 (参考 mel → 音色向量)。

M1 简化: 3 层 CNN + AvgPool
M2 增强: WavLM / ECAPA-TDNN 预训练
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adr.core import register


@register("timbre_encoder", "mel_cnn")
class TimbreEncoder(nn.Module):
    """参考音频 → 音色嵌入。

    Args:
        mel: (B, n_mels, T_mel) log-mel
    Returns:
        (B, embed_dim) 音色向量
    """

    def __init__(self, n_mels: int = 80, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(256, embed_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv(mel)
        x = self.pool(x).squeeze(-1)
        return self.norm(x)
