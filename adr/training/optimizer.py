"""优化器配置 (M1 Day 8)。

支持:
- AdamW (默认, 8GB 推荐)
- 8-bit AdamW (bitsandbytes, 可选, 进一步省显存)
- 简单 LR scheduler (warmup + cosine)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class OptimizerConfig:
    """优化器配置。"""
    name: str = "adamw"           # adamw / adamw_8bit / sgd
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"      # cosine / constant / warmup_cosine
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1

    # 8-bit
    use_8bit: bool = False         # bitsandbytes 8-bit AdamW (M2 默认)


def build_optimizer(
    model: nn.Module,
    config: Optional[OptimizerConfig] = None,
) -> torch.optim.Optimizer:
    """构造优化器。

    Args:
        model: nn.Module
        config: OptimizerConfig, None 用默认

    Returns:
        torch.optim.Optimizer
    """
    config = config or OptimizerConfig()

    # 拆参数: weight_decay 应用于非 bias/LayerNorm
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower() or "ln" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if config.use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
            )
        except ImportError:
            print("  [!] bitsandbytes not installed, falling back to AdamW")

    return torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Optional[OptimizerConfig] = None,
    total_steps: int = 1000,
):
    """构造 LR scheduler。

    Args:
        optimizer: 优化器
        config: OptimizerConfig
        total_steps: 总步数
    """
    config = config or OptimizerConfig()
    warmup = max(1, config.warmup_steps)

    if config.scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    if config.scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            # clamp 到 [0,1]: 越界时 cos 会回升导致 LR 反弹
            progress = min(max((step - warmup) / max(1, total_steps - warmup), 0.0), 1.0)
            return config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if config.scheduler == "warmup_cosine":
        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            # clamp 到 [0,1]: 越界时 cos 会回升导致 LR 反弹
            progress = min(max((step - warmup) / max(1, total_steps - warmup), 0.0), 1.0)
            return config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unknown scheduler: {config.scheduler}")
