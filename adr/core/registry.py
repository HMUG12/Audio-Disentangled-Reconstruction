"""插件注册表 (借鉴 Hugging Face transformers / timm 设计)。

用法:
    from adr.core.registry import REGISTRY, register

    @register("backbone", "my_backbone")
    class MyBackbone:
        ...

    # 获取
    cls = REGISTRY.get("backbone", "my_backbone")
    instance = cls()
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, Optional


class Registry:
    """通用注册表 (线程安全)。"""

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, key: str, obj: Any = None) -> Any:
        """注册对象。

        两种用法:
        1. @REGISTRY.register("key") 直接装饰
        2. REGISTRY.register("key", obj) 显式注册
        """
        # 用法 1: 装饰器
        if obj is None:

            def decorator(cls_or_fn: Any) -> Any:
                with self._lock:
                    self._registry[key] = cls_or_fn
                return cls_or_fn

            return decorator

        # 用法 2: 显式注册
        with self._lock:
            self._registry[key] = obj
        return obj

    def get(self, key: str, default: Any = None) -> Any:
        """获取已注册的对象。"""
        return self._registry.get(key, default)

    def has(self, key: str) -> bool:
        """检查 key 是否注册。"""
        return key in self._registry

    def keys(self) -> list[str]:
        """所有已注册的 key。"""
        return list(self._registry.keys())

    def items(self) -> list[tuple[str, Any]]:
        """所有 (key, obj) 对。"""
        return list(self._registry.items())

    def clear(self) -> None:
        """清空注册表 (主要用于测试)。"""
        with self._lock:
            self._registry.clear()

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        keys = ", ".join(self.keys())
        return f"Registry({self.name}, [{keys}])"


# ===== 框架全局注册表 =====
# Backbone:  TTS 主干 (SoVITS / MamTra / MAVE / ...)
BACKBONE_REGISTRY = Registry("backbone")

# Vocoder:   声码器 (BigVGAN / HiFi-GAN / ...)
VOCODER_REGISTRY = Registry("vocoder")

# ContentEncoder: 内容编码器 (WavLM / ContentVec / ...)
CONTENT_ENCODER_REGISTRY = Registry("content_encoder")

# TimbreEncoder:  音色编码器 (ECAPA / ResNet / ...)
TIMBRE_ENCODER_REGISTRY = Registry("timbre_encoder")

# ASR:        语音识别 (Faster-Whisper / Paraformer / ...)
ASR_REGISTRY = Registry("asr")

# G2P:        音素转换 (g2pW / CMUDict / ...)
G2P_REGISTRY = Registry("g2p")

# Separator:  人声分离 (UVR5 / MelBand-Roformer / ...)
SEPARATOR_REGISTRY = Registry("separator")

# F0:         音高提取 (RMVPE / FCPE / Harvest / ...)
F0_REGISTRY = Registry("f0")

# Quantizer:  量化方法 (GGUF / AWQ / GPTQ / ...)
QUANTIZER_REGISTRY = Registry("quantizer")


# ===== 顶层 REGISTRY 字典 (便于统一访问) =====
class _RegistryHub:
    """所有注册表的中心枢纽。"""

    @property
    def backbone(self) -> Registry:
        return BACKBONE_REGISTRY

    @property
    def vocoder(self) -> Registry:
        return VOCODER_REGISTRY

    @property
    def content_encoder(self) -> Registry:
        return CONTENT_ENCODER_REGISTRY

    @property
    def timbre_encoder(self) -> Registry:
        return TIMBRE_ENCODER_REGISTRY

    @property
    def asr(self) -> Registry:
        return ASR_REGISTRY

    @property
    def g2p(self) -> Registry:
        return G2P_REGISTRY

    @property
    def separator(self) -> Registry:
        return SEPARATOR_REGISTRY

    @property
    def f0(self) -> Registry:
        return F0_REGISTRY

    @property
    def quantizer(self) -> Registry:
        return QUANTIZER_REGISTRY

    def all(self) -> dict[str, Registry]:
        """所有注册表字典。"""
        return {
            "backbone": self.backbone,
            "vocoder": self.vocoder,
            "content_encoder": self.content_encoder,
            "timbre_encoder": self.timbre_encoder,
            "asr": self.asr,
            "g2p": self.g2p,
            "separator": self.separator,
            "f0": self.f0,
            "quantizer": self.quantizer,
        }


REGISTRY = _RegistryHub()


# ===== 顶层 register 函数 (便捷 API) =====
def register(category: str, key: str) -> Callable:
    """便捷注册函数。

    Args:
        category: 类别 (backbone/vocoder/...)
        key: 注册名

    Example:
        >>> from adr.core.registry import register
        >>> @register("backbone", "my_sovits")
        >>> class MySoVITS:
        ...     pass
    """
    if category not in REGISTRY.all():
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Available: {list(REGISTRY.all().keys())}"
        )
    return REGISTRY.all()[category].register(key)
