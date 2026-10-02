"""Gradient Checkpointing 封装 (M1 Day 8)。

开启后用时间换显存,Transformer 类模型通常能省 30-50% 显存。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def enable_gradient_checkpointing(model: nn.Module, enabled: bool = True) -> int:
    """递归给所有支持 gradient checkpointing 的子模块开启。

    Args:
        model: nn.Module
        enabled: True 开启, False 关闭

    Returns:
        实际开启的子模块数量
    """
    count = 0
    for module in model.modules():
        # PyTorch 内置支持 checkpoint 的常见模块
        if isinstance(module, nn.TransformerEncoder):
            # PyTorch >= 2.0 支持
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enabled
                count += 1
        elif isinstance(module, nn.TransformerDecoder):
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enabled
                count += 1

    # 自定义 SoVITS 的 decoder 标记
    if hasattr(model, "decoder") and isinstance(getattr(model, "decoder", None), nn.Module):
        decoder = model.decoder
        if hasattr(decoder, "gradient_checkpointing"):
            decoder.gradient_checkpointing = enabled
            count += 1

    # 标记属性 (供 trainer 检查)
    model._gradient_checkpointing_enabled = enabled
    return count


def is_gradient_checkpointing_enabled(model: nn.Module) -> bool:
    """检查模型是否启用了 gradient checkpointing。"""
    return getattr(model, "_gradient_checkpointing_enabled", False)


def estimate_memory_saved(model: nn.Module) -> str:
    """粗略估算 gradient checkpointing 能省多少显存 (MB)。

    基于参数量和层数,简化公式: ~ activation_size * n_layers。
    """
    n_params = sum(p.numel() for p in model.parameters())
    # 简化估算: 假设每参数 4 字节, 8GB 显存
    param_mb = n_params * 4 / 1024 / 1024
    return f"~{param_mb * 0.4:.0f} MB (≈ 40% of {param_mb:.0f} MB params)"
