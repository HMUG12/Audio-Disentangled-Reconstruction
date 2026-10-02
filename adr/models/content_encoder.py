"""内容编码器 (音素 → 上下文嵌入)。

简化版: 4 层 Transformer, 可被 LoRA 微调。
M2 集成: 替换为 WavLM / ContentVec 预训练。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from adr.core import register


@register("content_encoder", "transformer")
class ContentEncoder(nn.Module):
    """音素序列编码器。

    Args:
        phoneme_ids: (B, T) int64
        mask: (B, T) bool, True=有效
    Returns:
        (B, T, D) 上下文嵌入
    """

    def __init__(
        self,
        vocab_size: int = 607,
        embed_dim: int = 192,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        max_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers,
            enable_nested_tensor=False,  # norm_first=True 与 nested tensor 不兼容
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = phoneme_ids.shape
        x = self.embed(phoneme_ids) + self.pos_embed[:, :T]
        kpm = ~mask if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=kpm)
        return self.norm(x)
