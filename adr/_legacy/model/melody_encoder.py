"""旋律编码器 — MIDI 音符 + 歌词对齐 → 音符级嵌入（含 null token 设计）。

关键设计（出自 UniVoice）：
- 唱歌模式：MIDI 音符序列（pitch + duration + velocity）→ 嵌入
- 说话模式：使用可学习 null_melody_token，等价于对旋律条件做边缘化，
  避免旋律约束干扰说话韵律
"""
from typing import Optional

import torch
import torch.nn as nn


class MelodyEncoder(nn.Module):
    """音符序列编码器。

    输入：
      pitch_ids:   (B, T_note) int64, MIDI 0..127
      duration:    (B, T_note) float, 秒
      velocity:    (B, T_note) int64, 0..31
      is_singing:  (B,) bool, True=唱歌模式, False=说话模式
    输出：
      (B, T_note, embed_dim) 音符级嵌入
    """

    NULL_TOKEN_ID = 0  # 内部保留

    def __init__(self, n_pitch_bins: int = 128, n_velocity_bins: int = 32,
                 embed_dim: int = 128, n_layers: int = 2, n_heads: int = 4,
                 ffn_dim: int = 512, max_notes: int = 1024):
        super().__init__()
        self.pitch_embed = nn.Embedding(n_pitch_bins, embed_dim, padding_idx=0)
        self.velocity_embed = nn.Embedding(n_velocity_bins, embed_dim, padding_idx=0)
        self.duration_proj = nn.Linear(1, embed_dim)
        # 关键：可学习 null melody token（说话模式使用）
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.null_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_notes, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, pitch_ids: torch.Tensor, duration: torch.Tensor,
                velocity: torch.Tensor, is_singing: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pitch_ids:  (B, T_note) int64
            duration:   (B, T_note) float
            velocity:   (B, T_note) int64
            is_singing: (B,) bool
        Returns:
            (B, T_note, embed_dim)
        """
        B, T = pitch_ids.shape
        x = (self.pitch_embed(pitch_ids)
             + self.velocity_embed(velocity)
             + self.duration_proj(duration.unsqueeze(-1)))

        # 说话模式：用 null token 替换全部旋律条件
        # mask: 1=唱歌保留原旋律, 0=说话替换为 null
        null = self.null_token.expand(B, T, self.embed_dim)
        mask = is_singing.view(B, 1, 1).to(x.dtype)  # 1=唱歌，0=说话
        x = x * mask + null * (1.0 - mask)

        x = x + self.pos_embed[:, :T]
        x = self.encoder(x)
        return self.norm(x)
