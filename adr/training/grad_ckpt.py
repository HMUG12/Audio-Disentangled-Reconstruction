"""Gradient Checkpointing 封装 (M1 Day 8)。

开启后用时间换显存,Transformer 类模型通常能省 30-50% 显存。

实现: 用 torch.utils.checkpoint 包装 nn.TransformerEncoderLayer.forward,
backward 时重计算激活以省去中间激活显存。
(HF 的 gradient_checkpointing_enable 只对 HF 模块生效, 对原生 nn.Transformer 无效)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _make_ckpt_forward(layer: nn.TransformerEncoderLayer):
    """给 TransformerEncoderLayer 生成带 checkpoint 的 forward。

    返回 (ckpt_forward, orig_forward); 推理 (无梯度) 时零开销直通。
    """
    orig_forward = layer.forward

    def ckpt_forward(src, *args, **kwargs):
        if not torch.is_grad_enabled():
            return orig_forward(src, *args, **kwargs)
        return checkpoint(orig_forward, src, *args, use_reentrant=False, **kwargs)

    return ckpt_forward, orig_forward


def enable_gradient_checkpointing(model: nn.Module, enabled: bool = True) -> int:
    """递归给模型中所有 nn.TransformerEncoderLayer 开/关 gradient checkpointing。

    Args:
        model: nn.Module
        enabled: True 开启, False 关闭

    Returns:
        实际变更的层数 (开启: 新包装层数; 关闭: 解包层数)
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            if enabled and not getattr(module, "_ckpt_wrapped", False):
                ckpt_forward, orig_forward = _make_ckpt_forward(module)
                module._orig_forward = orig_forward
                module.forward = ckpt_forward
                module._ckpt_wrapped = True
                count += 1
            elif not enabled and getattr(module, "_ckpt_wrapped", False):
                module.forward = module._orig_forward
                module._ckpt_wrapped = False
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
