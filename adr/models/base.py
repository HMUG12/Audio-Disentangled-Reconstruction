"""模型 Backend 抽象基类 (M8.6 接口定型)。

设计目标:
- 提供统一的 forward/sample 签名
- M8.6 扩展: capabilities 能力声明 (TTS/SVS/流式/F0 条件) + sample_stream 流式接口
  → 一期 SoVITS、二期 ADR-2、保底路线 (第三方底座套壳) 都实现同一接口
- 子类只需实现 _build_model 和 (可选) _build_optimizer
- 自动注册到 REGISTRY
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import IntFlag
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn

from adr.core import get_logger


class BackboneCapability(IntFlag):
    """backbone 能力声明 (位掩码, 可组合)。"""
    TTS = 1          # 文本转语音 (内置韵律)
    SVS = 2          # 歌声合成 (外部旋律/F0 条件)
    STREAM = 4       # 支持流式 sample_stream
    F0_COND = 8      # sample 接受 f0 条件
    LORA_READY = 16  # 大权重矩阵全是 Linear/MHA, 可被 LoRA/QLoRA 包裹


@dataclass
class BackboneConfig:
    """Backbone 配置基类。"""
    name: str = "base"
    hidden_dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    # 子类可扩展


class BaseBackbone(nn.Module, abc.ABC):
    """模型 Backend 抽象基类。"""

    config_class = BackboneConfig
    capabilities: BackboneCapability = BackboneCapability.TTS

    def __init__(self, config: Optional[BackboneConfig] = None):
        super().__init__()
        self.config = config or self.config_class()
        self.log = get_logger(f"adr.models.{self.__class__.__name__}")

    # ---- M8.6: 能力查询 ----
    def supports(self, cap: BackboneCapability) -> bool:
        """是否具备某项能力。"""
        return bool(self.capabilities & cap)

    @classmethod
    def supports_mode(cls, mode: str) -> bool:
        """是否支持某模式 ("tts"/"svs")。"""
        cap = BackboneCapability.TTS if mode == "tts" else BackboneCapability.SVS
        return bool(cls.capabilities & cap)

    # ---- M8.6: 流式推理 (默认回退整段, 子类可覆盖为真流式) ----
    @torch.inference_mode()
    def sample_stream(
        self,
        phoneme_ids: torch.Tensor,
        ref_mel: torch.Tensor,
        chunk_frames: int = 50,
        **kwargs,
    ) -> Iterator[torch.Tensor]:
        """流式生成 mel, 逐 chunk yield (n_mels, T_chunk)。

        默认实现: 整段 sample 后按 chunk_frames 切片 (接口对齐用,
        不算真流式)。ADR-2 将覆盖为 chunk 边界对齐音素的真流式。
        """
        mel = self.sample(phoneme_ids, ref_mel, **kwargs)
        for i in range(0, mel.size(-1), chunk_frames):
            yield mel[..., i:i + chunk_frames]

    @abc.abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """训练 forward。

        Args:
            batch: 输入数据 dict, 至少包含 phoneme_ids / ref_mel

        Returns:
            dict, 至少包含 'loss' (用于 backward)
        """
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def sample(
        self,
        phoneme_ids: torch.Tensor,
        ref_mel: torch.Tensor,
        n_timesteps: int = 20,
        **kwargs,
    ) -> torch.Tensor:
        """推理: 生成 mel。

        Args:
            phoneme_ids: (B, T_phoneme)
            ref_mel: (B, n_mels, T_ref)
            n_timesteps: CFM 步数
        """
        raise NotImplementedError

    def num_parameters(self) -> int:
        """总参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters_str(self) -> str:
        """人类可读的参数量。"""
        n = self.num_parameters()
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    def freeze_content(self) -> None:
        """冻结内容编码器 (常用于克隆微调)。"""
        if hasattr(self, "content_encoder"):
            for p in self.content_encoder.parameters():
                p.requires_grad = False
            self.log.info("  Content encoder frozen")

    def freeze_vocoder(self) -> None:
        """冻结声码器相关。"""
        for name, p in self.named_parameters():
            if "vocoder" in name or "bigvgan" in name:
                p.requires_grad = False
        self.log.info("  Vocoder frozen")

    def trainable_parameters(self) -> int:
        """可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_parameters_str(self) -> str:
        n = self.trainable_parameters()
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"params={self.num_parameters_str()}, "
            f"trainable={self.trainable_parameters_str()})"
        )
