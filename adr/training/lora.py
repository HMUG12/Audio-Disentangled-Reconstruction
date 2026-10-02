"""LoRA (Low-Rank Adaptation) 训练栈 (M2)。

设计目标:
- 零依赖 (只用 torch)
- 冻结原参数,只训低秩 A/B 矩阵
- 节省显存: 0.3B 模型 LoRA 训练只需 ~50-100MB 额外参数
- 提供 PEFT 兼容 API (可平滑切换)

用法:
    >>> from adr.training.lora import LoRAConfig, apply_lora, get_lora_state_dict
    >>> config = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    >>> model = apply_lora(model, config)
    >>> # 训练: 只 LoRA 参数 requires_grad=True
    >>> # 保存: 只保存 LoRA 权重
    >>> torch.save(get_lora_state_dict(model), "lora.pt")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from adr.core import get_logger


@dataclass
class LoRAConfig:
    """LoRA 配置。

    Args:
        rank: 低秩矩阵的秩 (越小越省显存,通常 4-64)
        alpha: 缩放因子 (通常 = rank 或 2*rank)
        dropout: LoRA dropout (默认 0.0)
        target_modules: 目标模块名 (子串匹配 Linear 层)
        exclude_modules: 排除模块名
    """
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    exclude_modules: List[str] = field(default_factory=list)


class LoRALinear(nn.Module):
    """LoRA 包装的 Linear 层。

    公式: y = W x + (B @ A @ x) * (alpha / rank)

    - W: 冻结 (in_features, out_features)
    - A: (rank, in_features) 可训练, Kaiming 初始化
    - B: (out_features, rank) 可训练, 0 初始化
    - dropout: 可选
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 原 linear (冻结)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        # 延迟初始化 (在 apply_lora 时填充)
        self._weight_initialized = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # LoRA A/B
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 保持 0,这样初始 BA=0,LoRA 不改变输出

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 标记 (供 apply_lora 识别)
        self.is_lora = True

    def init_from_linear(self, linear: nn.Linear) -> None:
        """从原 nn.Linear 初始化 weight/bias (一次性)。"""
        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if linear.bias is not None and self.bias is not None:
                self.bias.copy_(linear.bias)
            elif linear.bias is None and self.bias is not None:
                # 原 linear 无 bias,但 LoRA 有 bias
                self.bias.zero_()
        self._weight_initialized = True

    def quantize_base_4bit(
        self,
        compute_dtype: torch.dtype = torch.float16,
        quant_type: str = "nf4",
    ) -> None:
        """把冻结的基座 weight 量化为 4bit (真 QLoRA)。

        量化后 forward 用 Linear4bit 计算基座, 并释放 fp32 基座显存。
        幂等: 已量化则直接返回。
        """
        if getattr(self, "is_quantized", False):
            return
        from bitsandbytes.nn import Linear4bit

        base = Linear4bit(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
        )
        with torch.no_grad():
            base.weight.data = self.weight.data.to(dtype=compute_dtype)
            if self.bias is not None:
                base.bias.data = self.bias.data.to(dtype=compute_dtype)
        base = base.to(self.weight.device)

        self.base_4bit = base
        self.is_quantized = True
        # 释放 fp32 基座 (已由 4bit 表示替代)
        self.weight = None

    def dequantize_base_4bit(self) -> None:
        """把 4bit 基座反量化回 fp32 weight (用于 checkpoint 保存/CPU 推理)。"""
        if not getattr(self, "is_quantized", False):
            return
        from bitsandbytes.functional import dequantize_4bit as _deq

        w = self.base_4bit.weight.data
        qs = self.base_4bit.weight.quant_state
        w_fp = _deq(w, quant_state=qs)
        self.weight = nn.Parameter(w_fp.to(torch.float32).contiguous(), requires_grad=False)
        if self.bias is not None and self.base_4bit.bias is not None:
            self.bias = nn.Parameter(
                self.base_4bit.bias.data.to(torch.float32).contiguous(), requires_grad=False
            )
        self.base_4bit = None
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原 linear (真 QLoRA: 基座已 4bit 量化时走 Linear4bit)
        if getattr(self, "is_quantized", False):
            base_out = self.base_4bit(x)
        else:
            base_out = F.linear(x, self.weight, self.bias)

        # LoRA 路径 (输入可能来自 4bit attention 输出, dtype 与 lora_A 不一致, 需对齐)
        lora_x = self.lora_dropout(x)
        if lora_x.dtype != self.lora_A.dtype:
            lora_x = lora_x.to(self.lora_A.dtype)
        lora_out = F.linear(lora_x, self.lora_A)  # (..., rank)
        lora_out = F.linear(lora_out, self.lora_B)  # (..., out_features)
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


def _should_apply_lora(name: str, target_modules: List[str], exclude_modules: List[str]) -> bool:
    """判断某层是否应被 LoRA 替换。"""
    for exc in exclude_modules:
        if exc in name:
            return False
    for tgt in target_modules:
        if tgt in name:
            return True
    return False


def _is_quantized_linear(module: nn.Module) -> bool:
    """检查是否是 bitsandbytes 量化层 (4bit/8bit)。"""
    try:
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        return isinstance(module, (Linear4bit, Linear8bitLt))
    except ImportError:
        return False


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
    auto_init: bool = True,
) -> nn.Module:
    """给 model 注入 LoRA 包装 (原地修改)。

    Args:
        model: 任意 nn.Module
        config: LoRAConfig
        auto_init: True 时从原 Linear 初始化 weight/bias

    Returns:
        修改后的 model (同一对象)

    Note:
        跳过 bnb 量化层 (Linear4bit/Linear8bitLt) — 这些层有自己的优化路径。
    """
    log = get_logger("adr.training.lora")

    # 1. 找到所有目标 Linear 层
    targets: List[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not getattr(module, "is_lora", False):
            if _is_quantized_linear(module):
                # 跳过 bnb 量化层 (直接用 Linear4bit forward 即可, 没必要套 LoRA)
                continue
            if _should_apply_lora(name, config.target_modules, config.exclude_modules):
                targets.append((name, module))

    log.info(f"LoRA: 找到 {len(targets)} 个目标 Linear 层")
    if not targets:
        already_lora = any(isinstance(m, LoRALinear) for m in model.modules())
        if already_lora:
            # 模型已注入 LoRA (幂等调用), 跳过告警
            log.info("  模型已注入 LoRA,跳过重复应用")
        else:
            log.warning(f"  target_modules={config.target_modules} 没匹配到任何 Linear,检查名字")

    # 2. 替换
    for name, linear in targets:
        # 推断 device/dtype (与原 linear 保持一致)
        device = linear.weight.device
        dtype = linear.weight.dtype

        lora_layer = LoRALinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            bias=linear.bias is not None,
        )
        # 把 LoRA layer 移到正确 device (创建时是 CPU)
        lora_layer = lora_layer.to(device=device)
        if auto_init:
            lora_layer.init_from_linear(linear)

        # 替换 (用 setattr)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, lora_layer)

    # 3. 冻结所有非 LoRA 参数
    n_train = 0
    n_total = 0
    for p in model.parameters():
        n_total += p.numel()
        if getattr(p, "requires_grad", True):
            p.requires_grad = False
    # 解冻 LoRA 参数
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
            n_train += m.lora_A.numel() + m.lora_B.numel()

    n_frozen = n_total - n_train
    log.info(
        f"LoRA: 总参数 {n_total/1e6:.2f}M, "
        f"可训练 {n_train/1e6:.4f}M ({n_train/n_total*100:.3f}%), "
        f"冻结 {n_frozen/1e6:.2f}M"
    )
    return model


def freeze_non_lora(model: nn.Module) -> int:
    """冻结所有非 LoRA 参数 (只 LoRA 可训练)。

    Returns:
        可训练参数数量
    """
    n_train = 0
    n_total = 0
    for p in model.parameters():
        n_total += p.numel()
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
            n_train += m.lora_A.numel() + m.lora_B.numel()
    return n_train


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """提取 model 中所有 LoRA 参数 (state_dict 格式)。"""
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            state[f"{name}.lora_A"] = m.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = m.lora_B.detach().cpu()
    return state


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> tuple[int, int]:
    """加载 LoRA state_dict 到 model。

    Returns:
        (loaded, missing) - 成功加载数,缺失数
    """
    loaded = 0
    missing = 0
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state_dict and b_key in state_dict:
                with torch.no_grad():
                    m.lora_A.copy_(state_dict[a_key].to(m.lora_A.device))
                    m.lora_B.copy_(state_dict[b_key].to(m.lora_B.device))
                loaded += 2
            elif strict:
                missing += 2
    return loaded, missing


def merge_lora(model: nn.Module, alpha_scale: float = 1.0) -> nn.Module:
    """把 LoRA 权重 merge 回原 linear (推理优化)。

    Returns:
        新的 model (原 model 不变)
    """
    import copy

    new_model = copy.deepcopy(model)
    for name, m in new_model.named_modules():
        if isinstance(m, LoRALinear):
            if getattr(m, "is_quantized", False):
                # 4bit 基座无法直接 merge (需先反量化), 跳过
                continue
            with torch.no_grad():
                # W' = W + (alpha/rank) * B @ A
                delta = (m.lora_B @ m.lora_A) * m.scaling * alpha_scale
                m.weight.data += delta
                # 清空 LoRA (merge 后不再需要)
                m.lora_A.zero_()
                m.lora_B.zero_()
    return new_model


# ============================================================
# PEFT 兼容层 (可选)
# ============================================================
def try_apply_peft(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    target_modules: List[str] = None,
):
    """尝试用 HuggingFace PEFT 库 (若已安装)。

    返回 (peft_model, success)。
    """
    try:
        from peft import LoraConfig, get_peft_model

        target_modules = target_modules or ["q_proj", "v_proj"]
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        peft_model = get_peft_model(model, peft_config)
        return peft_model, True
    except ImportError:
        return model, False
