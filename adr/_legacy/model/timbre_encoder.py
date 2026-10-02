"""音色编码器 — 参考音频 → 音色嵌入。

本期实现：mel + 简易 CNN 编码器（仅供骨架测试）。
M1 阶段：替换为 WavLM/ECAPA-TDNN/语音 codec 池化等预训练模型。
"""
import torch
import torch.nn as nn


class TimbreEncoder(nn.Module):
    """参考音频 → 音色嵌入（单向量）。

    输入：log-mel spectrogram (B, n_mels, T_mel)
    输出：音色嵌入 (B, embed_dim)
    """

    def __init__(self, n_mels: int = 80, embed_dim: int = 256):
        super().__init__()
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
        self.embed_dim = embed_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T_mel)
        Returns:
            (B, embed_dim)
        """
        x = self.conv(mel)
        x = self.pool(x).squeeze(-1)
        return self.norm(x)
