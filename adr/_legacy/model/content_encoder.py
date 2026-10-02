"""内容编码器 — 音素/歌词 → 上下文嵌入。

复用 DiffSinger 音素体系（中/英字典 + 全局音素 SP/AP + 静音）。
模块本身从零训练；M1 阶段可接入开源预训练音素器。
"""
from typing import Optional

import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    """音素序列编码器。

    输入：(B, T_phoneme) int64 token id
    输出：(B, T_phoneme, D) 上下文嵌入
    """

    def __init__(self, vocab_size: int = 512, embed_dim: int = 256,
                 n_layers: int = 4, n_heads: int = 4, ffn_dim: int = 1024,
                 max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, phoneme_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            phoneme_ids: (B, T) int64
            mask: (B, T) bool，True=有效，False=padding
        """
        B, T = phoneme_ids.shape
        x = self.embed(phoneme_ids) + self.pos_embed[:, :T]
        # nn.TransformerEncoder 用 src_key_padding_mask: True=padding 忽略
        kpm = None
        if mask is not None:
            kpm = ~mask  # 反转：mask=True(有效) → kpm=False(不忽略)
        x = self.encoder(x, src_key_padding_mask=kpm)
        return self.norm(x)
