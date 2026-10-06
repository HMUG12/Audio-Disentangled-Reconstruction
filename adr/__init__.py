"""ADR — Audio Disentangled Reconstruction.

低资源快速克隆训练框架。目标:让 8GB 显存用户也能在 5 分钟内
克隆出高质量声音,无需高端 GPU。

主要入口:
    >>> import adr
    >>> adr.clone(ref_audio="ref.wav", text="你好世界", output="out.wav")

或者命令行:
    $ adr clone --ref ref.wav --text "你好世界" --output out.wav
    $ adr train --config configs/vram_6gb.yaml
    $ adr infer --ref ref.wav --text "你好世界"
    $ adr export --format gguf
    $ adr webui
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "ADR Team"
__license__ = "MIT"

# 触发插件注册 (各子模块 @register 装饰器在此执行)
# 只要用户 import adr,所有内置插件即可被 REGISTRY 检索到
from adr.core import REGISTRY  # noqa: F401

# Backbones
from adr.models import sovits  # noqa: F401
from adr.models import content_encoder  # noqa: F401
from adr.models import timbre_encoder  # noqa: F401

# Vocoders
from adr.vocoder import bigvgan  # noqa: F401


def get_version() -> str:
    """返回框架版本号。"""
    return __version__
