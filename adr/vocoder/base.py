"""声码器抽象基类。"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from adr.core import get_logger


@dataclass
class VocoderConfig:
    """声码器配置。"""
    name: str = "base"
    sample_rate: int = 22050
    n_mels: int = 80
    hop_length: int = 256
    pretrained_path: Optional[str] = None


class BaseVocoder(nn.Module, abc.ABC):
    """声码器抽象基类 (mel → wav)。"""

    config_class = VocoderConfig

    def __init__(self, config: Optional[VocoderConfig] = None):
        super().__init__()
        self.config = config or self.config_class()
        self.log = get_logger(f"adr.vocoder.{self.__class__.__name__}")

    @abc.abstractmethod
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel → wav。

        Args:
            mel: (B, n_mels, T_mel) log-mel
        Returns:
            (B, T_wav) wav
        """
        raise NotImplementedError

    @torch.inference_mode()
    def infer(self, mel: torch.Tensor) -> torch.Tensor:
        """推理 (自动转 device + dtype)。"""
        was_training = self.training
        self.eval()
        try:
            mel = mel.to(next(self.parameters()).device)
            return self.forward(mel)
        finally:
            if was_training:
                self.train()

    def freeze(self) -> None:
        """冻结所有参数 (常用于克隆微调)。"""
        for p in self.parameters():
            p.requires_grad = False
        self.log.info("  Vocoder frozen")

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
