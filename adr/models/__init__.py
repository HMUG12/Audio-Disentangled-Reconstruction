"""ADR 模型 Backend 模块 (可插拔架构)。

主要组件:
- base:           抽象基类
- sovits:         SoVITS 简化版 (默认 backend)
- content_encoder: 内容编码器 (WavLM / 简单 Transformer)
- timbre_encoder:  音色编码器 (256 维 adapter)
- hub:            预训练模型管理 (下载/缓存/版本)
"""

from adr.models.base import BaseBackbone, BackboneConfig
from adr.models.content_encoder import ContentEncoder
from adr.models.hub import PretrainedHub, get_hub
from adr.models.timbre_encoder import TimbreEncoder

# 注册 SoVITS
from adr.models import sovits  # noqa: F401  (触发注册)

__all__ = [
    "BackboneConfig",
    "BaseBackbone",
    "ContentEncoder",
    "PretrainedHub",
    "TimbreEncoder",
    "get_hub",
]
