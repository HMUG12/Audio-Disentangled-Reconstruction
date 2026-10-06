"""设备管理 + 显存自适应 (RVC 范式)。

核心思想: 根据用户显存自动调整关键参数,
4G / 6G / 8G 三档配置,用户无需关心细节。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch


# ===== RVC 范式: 显存档位参数 =====
# 参考 RVC configs/config.py 的设计
VRAM_PRESETS = {
    "4gb": {
        "x_pad": 1,
        "x_query": 5,
        "x_center": 30,
        "x_max": 32,
        "batch_size": 1,
        "max_mel_frames": 512,
        "precision": "fp16",
        "use_gradient_checkpointing": True,
        "use_flash_attn": False,
    },
    "6gb": {
        "x_pad": 2,
        "x_query": 8,
        "x_center": 50,
        "x_max": 55,
        "batch_size": 2,
        "max_mel_frames": 800,
        "precision": "fp16",
        "use_gradient_checkpointing": True,
        "use_flash_attn": False,
    },
    "8gb": {
        "x_pad": 3,
        "x_query": 10,
        "x_center": 60,
        "x_max": 65,
        "batch_size": 4,
        "max_mel_frames": 1500,
        "precision": "fp16",
        "use_gradient_checkpointing": True,
        "use_flash_attn": True,
    },
    "12gb+": {
        "x_pad": 3,
        "x_query": 10,
        "x_center": 60,
        "x_max": 70,
        "batch_size": 8,
        "max_mel_frames": 2000,
        "precision": "bf16",
        "use_gradient_checkpointing": False,
        "use_flash_attn": True,
    },
}


@dataclass
class DeviceConfig:
    """设备配置。"""
    device: str = "cpu"              # cpu / cuda / mps
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    vram_preset: str = "default"     # 4gb/6gb/8gb/12gb+/default
    is_half: bool = False            # FP16
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"

    # RVC 范式: buffer 大小
    x_pad: int = 3
    x_query: int = 10
    x_center: int = 60
    x_max: int = 65

    # 训练参数
    batch_size: int = 4
    max_mel_frames: int = 1500

    # 优化
    use_gradient_checkpointing: bool = True
    use_flash_attn: bool = False

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    @property
    def is_mps(self) -> bool:
        return self.device == "mps"

    @property
    def vram_str(self) -> str:
        if self.gpu_vram_gb >= 1:
            return f"{self.gpu_vram_gb:.1f} GB"
        return "N/A (CPU)"

    @property
    def capability(self) -> str:
        """功能分级 (AMD/Intel/MPS 用户的明确预期)。"""
        if self.is_cuda:
            return "全功能 (NVIDIA CUDA: 训练 + GPU 推理)"
        if self.is_mps:
            return "实验性 (Apple MPS: 推理未验证, 训练不支持)"
        return "受限 (无 NVIDIA GPU: CPU 推理+声纹门禁可用, 训练不支持)"

    def summary(self) -> str:
        """人类可读的设备摘要。"""
        lines = [
            f"  Device      : {self.device}",
            f"  GPU         : {self.gpu_name or 'N/A'}",
            f"  VRAM        : {self.vram_str}",
            f"  Capability  : {self.capability}",
            f"  Preset      : {self.vram_preset}",
            f"  Precision   : {self.precision}",
            f"  Batch size  : {self.batch_size}",
            f"  Max frames  : {self.max_mel_frames}",
            f"  FlashAttn   : {self.use_flash_attn}",
            f"  GradCkpt    : {self.use_gradient_checkpointing}",
        ]
        return "\n".join(lines)


def detect_device() -> DeviceConfig:
    """自动检测设备 + 显存档位。"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        # PyTorch 2.10+ 改名为 total_memory
        props = torch.cuda.get_device_properties(0)
        gpu_vram_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        gpu_name = "Apple Silicon"
        gpu_vram_gb = 0  # MPS 共享内存
    else:
        device = "cpu"
        gpu_name = ""
        gpu_vram_gb = 0

    # 根据显存选 preset
    if gpu_vram_gb >= 12:
        preset = "12gb+"
    elif gpu_vram_gb >= 8:
        preset = "8gb"
    elif gpu_vram_gb >= 6:
        preset = "6gb"
    elif gpu_vram_gb >= 4:
        preset = "4gb"
    else:
        preset = "default"

    # 应用 preset
    preset_params = VRAM_PRESETS.get(preset, VRAM_PRESETS["8gb"])
    is_half = preset_params["precision"] in ("fp16", "bf16") and device != "cpu"

    return DeviceConfig(
        device=device,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        vram_preset=preset,
        is_half=is_half,
        precision=preset_params["precision"],
        x_pad=preset_params["x_pad"],
        x_query=preset_params["x_query"],
        x_center=preset_params["x_center"],
        x_max=preset_params["x_max"],
        batch_size=preset_params["batch_size"],
        max_mel_frames=preset_params["max_mel_frames"],
        use_gradient_checkpointing=preset_params["use_gradient_checkpointing"],
        use_flash_attn=preset_params["use_flash_attn"],
    )


def setup_device(
    device: Optional[str] = None,
    force_preset: Optional[str] = None,
    verbose: bool = True,
) -> DeviceConfig:
    """设置设备并打印信息。

    Args:
        device: 强制指定 ("cpu"/"cuda"/"mps"),None 则自动检测
        force_preset: 强制指定显存档 ("4gb"/"6gb"/"8gb"/"12gb+")
        verbose: 是否打印信息
    """
    cfg = detect_device()

    if device is not None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            # 批次41b: 请求 CUDA 但不可用时不再静默, 明确告警并回退 CPU
            from adr.core.logging import get_logger

            get_logger("adr.device").warning(
                f"请求设备 '{device}', 但 torch.cuda.is_available() 为 False "
                "(可能原因: 未安装 CUDA 版 PyTorch / 无 NVIDIA GPU / 驱动未就绪), "
                "已回退到 CPU"
            )
            cfg.device = "cpu"
        else:
            cfg.device = device

    if force_preset is not None:
        if force_preset in VRAM_PRESETS:
            params = VRAM_PRESETS[force_preset]
            cfg.vram_preset = force_preset
            cfg.x_pad = params["x_pad"]
            cfg.x_query = params["x_query"]
            cfg.x_center = params["x_center"]
            cfg.x_max = params["x_max"]
            cfg.batch_size = params["batch_size"]
            cfg.max_mel_frames = params["max_mel_frames"]
            cfg.precision = params["precision"]
            # CPU 不吃半精度 (批次28 复审 P6, 对齐 detect_device 的守卫)
            cfg.is_half = params["precision"] in ("fp16", "bf16") and cfg.device != "cpu"
            cfg.use_gradient_checkpointing = params["use_gradient_checkpointing"]
            cfg.use_flash_attn = params["use_flash_attn"]

    # 设置 PyTorch 后端
    if cfg.is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.9)  # 预留 10% 给系统

    # 缓存
    get_device._cache = cfg

    if verbose:
        from adr.core.logging import get_logger

        log = get_logger("adr.device")
        log.info(f"Device detected:\n{cfg.summary()}")

    return cfg


def get_device() -> DeviceConfig:
    """获取当前设备配置 (lazy setup)。"""
    if not hasattr(get_device, "_cache"):
        setup_device(verbose=False)
    return get_device._cache


def empty_cache() -> None:
    """清理显存 (RVC 范式)。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_free_vram_gb() -> float:
    """获取当前空闲显存 (GB)。"""
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return free / 1e9
    return 0.0


def print_memory_usage(prefix: str = "") -> None:
    """打印当前显存占用 (debug 用)。"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        from adr.core.logging import get_logger

        log = get_logger("adr.device")
        log.info(
            f"{prefix}GPU mem: alloc={alloc:.2f}GB, "
            f"reserved={reserved:.2f}GB, peak={peak:.2f}GB"
        )
