"""统一配置 (dataclass + YAML)。

设计目标:
1. dataclass 提供类型安全和 IDE 提示
2. YAML 提供人类可读的配置
3. 三档显存配置 (4G/6G/8G) 自动适配 (RVC 范式)
4. 用户自定义配置优先级最高
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


def adr_data_dir() -> Path:
    """ADR 外部数据目录 (预训练权重/词典/日志等大文件, 不入仓库)。

    优先级: ADR_DATA_DIR 环境变量 > F:/ADR_data (历史约定, 存在即沿用) >
    平台用户目录 (Windows: %LOCALAPPDATA%/ADR/data; 其他: ~/.adr/data)。
    (批次26: 消除散落各处的盘符硬编码, 统一从这里推导)
    """
    env = os.environ.get("ADR_DATA_DIR", "").strip()
    if env:
        return Path(env)
    legacy = Path("F:/ADR_data")
    if legacy.is_dir():
        return legacy
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "ADR" / "data"
    return Path.home() / ".adr" / "data"


# ===== 数据配置 =====
@dataclass
class DataConfig:
    """数据流水线配置。"""
    sample_rate: int = 24000
    hop_length: int = 256
    n_mels: int = 80
    n_fft: int = 1024
    win_length: int = 1024
    fmin: float = 0.0
    fmax: float = 12000.0

    # 数据处理开关
    enable_separation: bool = True   # UVR5 人声分离
    enable_denoise: bool = True      # 降噪
    slice_min_sec: float = 3.0       # 最小切片时长
    slice_max_sec: float = 10.0      # 最大切片时长
    slice_min_silence_sec: float = 0.3  # 最小静音长度

    # ASR
    asr_model: str = "small"         # faster-whisper 模型大小

    # G2P
    g2p_backend: Literal["g2pW", "cmudict", "char"] = "g2pW"


# ===== 训练配置 =====
@dataclass
class TrainConfig:
    """训练超参。"""
    # 基础
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 10
    max_steps: int = -1  # -1 表示按 epoch 算

    # 优化
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"

    # 精度
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    use_gradient_checkpointing: bool = True

    # 加速
    use_flash_attn: bool = False  # M2 启用
    use_qlora: bool = False       # M2 启用
    qlora_r: int = 16
    qlora_alpha: int = 32
    qlora_dropout: float = 0.05
    qlora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 保存
    save_every_n_steps: int = 500
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    log_every_n_steps: int = 50


# ===== 推理配置 =====
@dataclass
class InferConfig:
    """推理配置。"""
    n_timesteps: int = 20            # CFM 推理步数
    speed: float = 1.0               # 语速
    pitch_shift: int = 0             # 音高偏移 (半音)
    energy_scale: float = 1.0        # 能量缩放
    noise_scale: float = 0.0         # 噪声 (增加随机性)
    top_k: int = 50                  # 采样 top-k
    top_p: float = 0.9               # 采样 top-p
    temperature: float = 1.0


# ===== 路径配置 =====
@dataclass
class PathConfig:
    """路径配置。"""
    cache_dir: str = "~/.cache/adr"
    pretrained_dir: str = "~/.cache/adr/pretrained"
    output_dir: str = "./output"
    log_dir: str = "./logs"
    data_dir: str = "./data"


# ===== 总配置 =====
@dataclass
class ADRConfig:
    """ADR 框架全局配置。"""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    path: PathConfig = field(default_factory=PathConfig)

    # 框架元信息
    framework_version: str = "0.1.0"
    config_preset: str = "default"  # default / vram_4gb / vram_6gb / vram_8gb

    def total_params_estimate(self) -> str:
        """估算模型参数量 (粗略)。"""
        return "~200M (SoVITS 简化版)"

    def vram_estimate(self) -> str:
        """估算显存占用。"""
        preset = self.config_preset
        if "4gb" in preset:
            return "3-4 GB"
        elif "6gb" in preset:
            return "4-6 GB"
        elif "8gb" in preset:
            return "6-8 GB"
        return "varies"

    def to_yaml(self, path: str | Path) -> None:
        """保存为 YAML。"""
        data = _to_dict(self)
        Path(path).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))

    def to_dict(self) -> dict[str, Any]:
        """转为 dict。"""
        return _to_dict(self)


# ===== 辅助函数 =====
def _to_dict(obj: Any) -> Any:
    """递归把 dataclass 转为 dict。"""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    elif isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


# 嵌套 dataclass 类型映射 (因 from __future__ import annotations 注解是 str)
_NESTED_TYPES = {
    "ADRConfig": None,  # 自身
    "DataConfig": DataConfig,
    "TrainConfig": TrainConfig,
    "InferConfig": InferConfig,
    "PathConfig": PathConfig,
}


def _resolve_type(type_str: str) -> type | None:
    """解析 dataclass 字段类型字符串。"""
    if type_str in _NESTED_TYPES:
        return _NESTED_TYPES[type_str]
    # 处理 Optional[X] / list[X] 等
    if "." in type_str:
        # 全限定名,尝试从 globals 找
        return globals().get(type_str.split(".")[-1])
    return None


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """递归把 dict 转为 dataclass。"""
    if not hasattr(cls, "__dataclass_fields__"):
        return data

    field_info = cls.__dataclass_fields__
    kwargs = {}

    for k, v in data.items():
        if k not in field_info:
            continue

        fld = field_info[k]

        # 解析类型
        target_type = None
        if isinstance(fld.type, str):
            target_type = _resolve_type(fld.type)
        else:
            target_type = fld.type

        # 嵌套 dataclass
        if (
            target_type is not None
            and hasattr(target_type, "__dataclass_fields__")
            and isinstance(v, dict)
        ):
            kwargs[k] = _from_dict(target_type, v)
        # list[dataclass] 不在 M1 范围内,跳过
        else:
            kwargs[k] = v

    return cls(**kwargs)


def load_config(
    config_path: Optional[str | Path] = None,
    preset: Literal["default", "vram_4gb", "vram_6gb", "vram_8gb"] = "default",
    override: Optional[dict[str, Any]] = None,
) -> ADRConfig:
    """加载配置。

    优先级: override > config_path > preset > default

    Args:
        config_path: YAML 配置文件路径 (可选)
        preset: 预设配置 (default/vram_4gb/vram_6gb/vram_8gb)
        override: 覆盖参数 (可选)
    """
    # 1. 加载预设
    config_dir = Path(__file__).parent.parent.parent / "configs"
    preset_path = config_dir / f"{preset}.yaml"

    if preset_path.exists():
        with open(preset_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # 2. 合并 config_path
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                user_data = yaml.safe_load(f) or {}
            _deep_merge(data, user_data)

    # 3. 应用 override
    if override:
        _deep_merge(data, override)

    # 4. 构造 dataclass
    config = _from_dict(ADRConfig, data)
    config.config_preset = preset

    # 5. 展开路径
    _expand_paths(config)

    return config


def get_config() -> ADRConfig:
    """获取当前全局配置 (lazy load)。

    通过环境变量 ADR_CONFIG 可指定配置文件。
    """
    if not hasattr(get_config, "_cache"):
        env_path = os.environ.get("ADR_CONFIG")
        preset = os.environ.get("ADR_PRESET", "default")
        get_config._cache = load_config(config_path=env_path, preset=preset)
    return get_config._cache


def _expand_paths(config: ADRConfig) -> None:
    """展开 ~ 路径。"""
    for fld in config.path.__dataclass_fields__:
        val = getattr(config.path, fld)
        if isinstance(val, str):
            setattr(config.path, fld, os.path.expanduser(val))


def _deep_merge(base: dict, override: dict) -> None:
    """深度合并字典 (override 优先)。"""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
