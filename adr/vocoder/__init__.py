"""ADR 声码器模块 (mel → wav)。

主要组件:
- base:    声码器抽象基类
- bigvgan: BigVGAN v2 集成 (默认)
- hifigan: HiFi-GAN 备选 (M2 集成)
"""

from adr.vocoder.base import BaseVocoder, VocoderConfig
from adr.vocoder.bigvgan import BigVGANVocoder

__all__ = [
    "BaseVocoder",
    "BigVGANVocoder",
    "VocoderConfig",
]
