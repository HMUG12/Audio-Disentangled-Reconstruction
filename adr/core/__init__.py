"""ADR 框架核心模块。

- config:    全局配置 (dataclass + YAML)
- device:    设备管理 + 显存自适应 (RVC 范式)
- registry:  插件注册表 (Backbone/Vocoder/Dataset)
- logging:   统一日志 (rich 美化)
- exceptions: 自定义异常
"""

from adr.core.config import ADRConfig, get_config, load_config
from adr.core.device import DeviceConfig, get_device, setup_device
from adr.core.logging import get_logger, setup_logging
from adr.core.registry import REGISTRY, Registry, register

__all__ = [
    "ADRConfig",
    "DeviceConfig",
    "REGISTRY",
    "Registry",
    "get_config",
    "get_device",
    "get_logger",
    "load_config",
    "register",
    "setup_device",
    "setup_logging",
]
