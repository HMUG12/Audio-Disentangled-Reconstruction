"""ADR 训练栈 (M1 Day 8 + M2 增强)。

主要组件:
- Trainer:          主训练循环 (AMP/GradCkpt/Callbacks/LoRA)
- TrainerConfig:    训练配置 (M2 新增 use_lora/lora_*)
- VoiceCloneDataset: PyTorch Dataset (npz → batch)
- OptimizerConfig:  优化器配置
- build_optimizer:  AdamW / 8-bit AdamW
- build_scheduler:  cosine / constant / warmup_cosine
- enable_gradient_checkpointing: 显存优化
- Callbacks:        Checkpoint / Logger / Progress / EarlyStop
- LoRA (M2):        零依赖 LoRA 包装 + PEFT 可选后端
- efficient (M2):   FlashAttn/8bit/4bit 量化 + 显存统计
"""

from adr.training.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggerCallback,
    ProgressCallback,
)
from adr.training.dataset import CollatedBatch, VoiceCloneDataset, collate_samples
from adr.training.grad_ckpt import (
    enable_gradient_checkpointing,
    estimate_memory_saved,
    is_gradient_checkpointing_enabled,
)
from adr.training.optimizer import (
    OptimizerConfig,
    build_optimizer,
    build_scheduler,
)
from adr.training.trainer import Trainer, TrainerConfig

# M2: LoRA
try:
    from adr.training.lora import (
        LoRAConfig,
        LoRALinear,
        apply_lora,
        freeze_non_lora,
        get_lora_state_dict,
        load_lora_state_dict,
        merge_lora,
    )
    _LORA_AVAILABLE = True
except ImportError:
    _LORA_AVAILABLE = False

# M2: efficient (FlashAttn/quant/memory)
try:
    from adr.training.efficient import (
        enable_flash_attention,
        estimate_inference_memory,
        estimate_model_memory,
        get_memory_stats,
        get_sdpa_backends,
        is_bnb_available,
        is_sdpa_available,
        print_memory_stats,
        print_trainable_parameters,
        quantize_4bit,
        quantize_8bit,
    )
    _EFFICIENT_AVAILABLE = True
except ImportError:
    _EFFICIENT_AVAILABLE = False

__all__ = [
    "Callback",
    "CheckpointCallback",
    "CollatedBatch",
    "EarlyStoppingCallback",
    "LoggerCallback",
    "OptimizerConfig",
    "ProgressCallback",
    "Trainer",
    "TrainerConfig",
    "VoiceCloneDataset",
    "build_optimizer",
    "build_scheduler",
    "collate_samples",
    "enable_gradient_checkpointing",
    "estimate_memory_saved",
    "is_gradient_checkpointing_enabled",
]

if _LORA_AVAILABLE:
    __all__ += [
        "LoRAConfig", "LoRALinear", "apply_lora", "freeze_non_lora",
        "get_lora_state_dict", "load_lora_state_dict", "merge_lora",
    ]
if _EFFICIENT_AVAILABLE:
    __all__ += [
        "enable_flash_attention", "estimate_inference_memory",
        "estimate_model_memory", "get_memory_stats", "get_sdpa_backends",
        "is_bnb_available", "is_sdpa_available", "print_memory_stats",
        "print_trainable_parameters", "quantize_4bit", "quantize_8bit",
    ]
