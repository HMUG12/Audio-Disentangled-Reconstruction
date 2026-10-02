"""M2 Training Stack 工具集 (高效训练 + 量化 + FlashAttn)。

组件:
1. enable_flash_attention: 强制 nn.MultiheadAttention 用 SDPA (PyTorch 2.0+ FlashAttn)
2. quantize_4bit: bitsandbytes NF4 量化 (QLoRA 基座)
3. quantize_8bit: bitsandbytes 8-bit 量化
4. get_memory_stats: GPU 显存统计
5. estimate_model_memory: 估算模型所需显存
6. print_trainable_parameters: 打印可训练参数统计

所有函数都做 safe-fallback: 依赖缺失时不报错,只警告。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from adr.core import get_logger


_log = get_logger("adr.training.efficient")


# ============================================================
# Flash Attention (SDPA)
# ============================================================
def is_sdpa_available() -> bool:
    """SDPA 是否可用 (PyTorch 2.0+)。"""
    return hasattr(torch.nn.functional, "scaled_dot_product_attention")


def get_sdpa_backends() -> dict:
    """获取当前 SDPA 后端状态。"""
    if not is_sdpa_available():
        return {"available": False}
    return {
        "available": True,
        "flash": torch.backends.cuda.flash_sdp_enabled() if torch.cuda.is_available() else None,
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled() if torch.cuda.is_available() else None,
        "math": torch.backends.cuda.math_sdp_enabled() if torch.cuda.is_available() else None,
    }


def enable_flash_attention(model: nn.Module) -> int:
    """强制 nn.MultiheadAttention 用 SDPA FlashAttn 后端。

    返回启用的 attention 层数。
    """
    if not is_sdpa_available():
        _log.warning("SDPA 不可用 (需要 PyTorch 2.0+)")
        return 0

    n = 0
    for m in model.modules():
        # nn.MultiheadAttention 在 batch_first=True 时,内部 forward 用 _native_sdp 或 _native_flash
        # PyTorch 2.0+ 默认走 SDPA,这里只确保 backends 开启
        if isinstance(m, nn.MultiheadAttention):
            n += 1

    if torch.cuda.is_available():
        # 确保 flash 和 mem_efficient 开启
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception as e:
            _log.warning(f"启用 SDPA 后端失败: {e}")

    _log.info(f"Flash Attention (SDPA): {n} 个 MHA 层, "
              f"后端: {get_sdpa_backends()}")
    return n


# ============================================================
# 4-bit / 8-bit 量化 (QLoRA)
# ============================================================
def is_bnb_available() -> bool:
    """bitsandbytes 是否可用。"""
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


class _QuantizedSelfAttention(nn.Module):
    """4bit 量化的自注意力 (替代 nn.MultiheadAttention 的 qkv 融合投影)。

    nn.MultiheadAttention 的 q/k/v 投影是裸 ``in_proj_weight`` Parameter,
    ``quantize_4bit`` 无法按 nn.Linear 处理, 因此单独用一个 4bit Linear 表示融合投影。

    仅覆盖 SoVITS decoder 使用场景: batch_first=True, q=k=v, 无 mask。
    """

    def __init__(
        self,
        mha: nn.MultiheadAttention,
        compute_dtype: Optional[torch.dtype] = None,
        quant_type: str = "nf4",
    ):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = mha.embed_dim // mha.num_heads
        self.dropout = mha.dropout
        self.batch_first = mha.batch_first
        # nn.TransformerEncoderLayer eval 时会探测 fast path 并访问
        # self_attn.in_proj_bias / _qkv_same_embed_dim 等私有属性。
        # 置 False 让它回退普通路径 (调我们的 forward)。
        self._qkv_same_embed_dim = False

        from bitsandbytes.nn import Linear4bit

        self.qkv = Linear4bit(
            self.embed_dim,
            3 * self.embed_dim,
            bias=mha.in_proj_bias is not None,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
        )
        _dtype = compute_dtype or torch.float16
        with torch.no_grad():
            self.qkv.weight.data = mha.in_proj_weight.detach().to(dtype=_dtype)
            if mha.in_proj_bias is not None:
                self.qkv.bias.data = mha.in_proj_bias.detach().to(dtype=_dtype)
        self.qkv = self.qkv.to(mha.in_proj_weight.device)

        # 保留原 out_proj (可能是 nn.Linear 或 LoRALinear)
        self.out_proj = mha.out_proj

    @property
    def in_proj_bias(self) -> Optional[torch.Tensor]:
        """兼容 nn.TransformerEncoderLayer fast path 探测 (映射到 qkv.bias)。"""
        return self.qkv.bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        B, L, E = query.shape
        q, k, v = self.qkv(query).chunk(3, dim=-1)  # 每个 (B, L, E)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA 只接受 attn_mask; 把 key_padding_mask 并进去 (SoVITS 实际不传 mask)
        if key_padding_mask is not None:
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_mask = kpm if attn_mask is None else (attn_mask | kpm)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, L, E)
        return self.out_proj(attn), None

    def to_multihead_attention(self) -> nn.MultiheadAttention:
        """反量化回 nn.MultiheadAttention (用于 checkpoint 保存 / CPU 推理)。

        把 4bit qkv 投影反量化成 fp ``in_proj_weight``, 重建标准 MHA。
        ``out_proj`` 原样保留 (可能是 nn.Linear / Linear4bit / LoRALinear,
        由 dequantize_4bit 后续统一处理)。
        """
        from bitsandbytes.functional import dequantize_4bit as _deq
        from bitsandbytes.nn import Linear4bit

        qkv = self.qkv
        if isinstance(qkv, Linear4bit):
            qs = getattr(qkv.weight, "quant_state", None)
            w = _deq(qkv.weight.data, quant_state=qs) if qs is not None else qkv.weight.data
        else:
            w = qkv.weight.data

        mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=qkv.bias is not None,
            batch_first=self.batch_first,
        )
        mha = mha.to(w.device)
        with torch.no_grad():
            mha.in_proj_weight.data = w.to(mha.in_proj_weight.dtype).contiguous()
            if qkv.bias is not None:
                mha.in_proj_bias.data = qkv.bias.data.to(mha.in_proj_bias.dtype).contiguous()
        mha.out_proj = self.out_proj
        return mha


def quantize_4bit(
    model: nn.Module,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """用 bitsandbytes 4-bit 量化模型 (QLoRA 基座)。

    量化三部分:
    1. nn.MultiheadAttention 的 qkv 融合投影 (in_proj_weight 是裸 Parameter)
    2. nn.Linear → Linear4bit
    3. LoRALinear 的冻结基座 → 4bit (真 QLoRA)

    Args:
        model: nn.Module (就地修改)
        quant_type: "nf4" (NormalFloat 4-bit, 推荐) / "fp4"
        compute_dtype: 计算时反量化到的 dtype

    Returns:
        量化后的 model
    """
    if not is_bnb_available():
        _log.warning("bitsandbytes 未装,跳过 4-bit 量化")
        return model

    try:
        from bitsandbytes.nn import Linear4bit

        n_mha = n_linear = n_lora = 0

        # 1. 量化 MultiheadAttention 的 qkv 融合投影
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.MultiheadAttention):
                qsa = _QuantizedSelfAttention(
                    module, compute_dtype=compute_dtype, quant_type=quant_type
                )
                parent_name, _, child_name = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, qsa)
                n_mha += 1

        # 2. 量化剩余 nn.Linear (跳过 LoRA 包装和已量化层)
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, Linear4bit):
                if getattr(module, "is_lora", False):
                    continue
                new_layer = Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                )
                target_device = module.weight.device
                with torch.no_grad():
                    new_layer.weight.data = module.weight.detach().to(dtype=compute_dtype)
                    if module.bias is not None:
                        new_layer.bias.data = module.bias.detach().to(dtype=compute_dtype)
                new_layer = new_layer.to(target_device)

                parent_name, _, child_name = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, new_layer)
                n_linear += 1

        # 3. 量化 LoRALinear 的冻结基座 (真 QLoRA)
        for name, module in list(model.named_modules()):
            if getattr(module, "is_lora", False) and hasattr(module, "quantize_base_4bit"):
                module.quantize_base_4bit(compute_dtype=compute_dtype, quant_type=quant_type)
                n_lora += 1

        _log.info(
            f"4-bit 量化: {n_mha} MHA + {n_linear} Linear + {n_lora} LoRA 基座 "
            f"(quant_type={quant_type})"
        )
        return model
    except Exception as e:
        _log.warning(f"4-bit 量化失败: {e}")
        return model


def quantize_8bit(model: nn.Module) -> nn.Module:
    """用 bitsandbytes 8-bit 量化 nn.Linear (节省 ~2x 权重显存)。"""
    if not is_bnb_available():
        _log.warning("bitsandbytes 未装,跳过 8-bit 量化")
        return model

    try:
        from bitsandbytes.nn import Linear8bitLt

        n_replaced = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, Linear8bitLt):
                if hasattr(module, "is_lora") and module.is_lora:
                    continue

                new_layer = Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                )
                with torch.no_grad():
                    new_layer.weight = module.weight
                    if module.bias is not None:
                        new_layer.bias = module.bias

                parent_name, _, child_name = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, new_layer)
                n_replaced += 1

        _log.info(f"8-bit 量化: 替换 {n_replaced} 个 Linear")
        return model
    except Exception as e:
        _log.warning(f"8-bit 量化失败: {e}")
        return model


def has_quantized_modules(model: nn.Module) -> bool:
    """判断模型是否含 4bit 量化层 (Linear4bit / 量化 LoRA 基座 / 量化 attention)。"""
    for m in model.modules():
        if isinstance(m, _QuantizedSelfAttention):
            return True
        if m.__class__.__name__ == "Linear4bit":
            return True
        if getattr(m, "is_quantized", False):
            return True
    return False


def dequantize_4bit(model: nn.Module) -> nn.Module:
    """把模型中的 4bit 量化层反量化回 fp16/fp32 (就地修改)。

    用于:
    - checkpoint 保存 (避免 4bit uint8 权重与未量化模型 shape/keys 不匹配)
    - CPU 推理 (bnb 4bit matmul 需要 CUDA)

    处理:
    - _QuantizedSelfAttention → nn.MultiheadAttention
    - Linear4bit → nn.Linear (fp16)
    - LoRALinear (is_quantized) → 反量化基座回 fp32
    """
    if not is_bnb_available():
        return model

    from bitsandbytes.functional import dequantize_4bit as _deq
    from bitsandbytes.nn import Linear4bit

    # 1. LoRALinear 基座反量化 (必须先做, 否则 base_4bit 会被当成普通 Linear4bit 处理)
    for name, module in list(model.named_modules()):
        if getattr(module, "is_lora", False) and hasattr(module, "dequantize_base_4bit"):
            module.dequantize_base_4bit()

    # 2. _QuantizedSelfAttention → nn.MultiheadAttention
    for name, module in list(model.named_modules()):
        if isinstance(module, _QuantizedSelfAttention):
            mha = module.to_multihead_attention()
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, mha)

    # 3. Linear4bit → nn.Linear
    for name, module in list(model.named_modules()):
        if isinstance(module, Linear4bit):
            qs = getattr(module.weight, "quant_state", None)
            if qs is None:
                continue
            w_fp = _deq(module.weight.data, quant_state=qs)
            new_linear = nn.Linear(
                module.in_features, module.out_features, bias=module.bias is not None
            )
            new_linear.weight.data = w_fp.to(new_linear.weight.dtype)
            if module.bias is not None:
                new_linear.bias.data = module.bias.data.to(new_linear.bias.dtype)
            new_linear = new_linear.to(module.weight.device)

            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_linear)

    return model


# ============================================================
# 显存统计
# ============================================================
def get_memory_stats(device: Optional[torch.device] = None) -> dict:
    """获取 GPU 显存使用统计。"""
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    if device is None:
        device = torch.device("cuda")

    stats = {
        "cuda_available": True,
        "device": str(device),
        "allocated_MB": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_MB": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_MB": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2
    stats["total_MB"] = total
    stats["free_MB"] = total - stats["reserved_MB"]
    stats["utilization_pct"] = (stats["reserved_MB"] / total) * 100
    return stats


def print_memory_stats(prefix: str = "", device: Optional[torch.device] = None):
    """打印显存统计。"""
    stats = get_memory_stats(device)
    if not stats.get("cuda_available"):
        print(f"{prefix}[memory] CUDA 不可用")
        return
    print(
        f"{prefix}[memory] "
        f"allocated={stats['allocated_MB']:.1f}MB, "
        f"reserved={stats['reserved_MB']:.1f}MB, "
        f"max={stats['max_allocated_MB']:.1f}MB, "
        f"total={stats['total_MB']:.0f}MB, "
        f"util={stats['utilization_pct']:.1f}%"
    )


def estimate_model_memory(
    n_params: int,
    dtype_bytes: int = 4,  # fp32=4, fp16/bf16=2, int8=1
    optimizer_factor: float = 8.0,  # Adam: 8 bytes/param (m, v in fp32)
    grad_factor: float = 4.0,  # gradient (同 dtype)
) -> dict:
    """估算训练一个模型需要的显存。

    Args:
        n_params: 模型参数量
        dtype_bytes: 参数 dtype 字节
        optimizer_factor: 优化器状态倍率 (Adam 8x, SGD 0x, 8-bit AdamW ~2x)
        grad_factor: 梯度倍率

    Returns:
        {param_MB, grad_MB, optim_MB, total_MB}
    """
    param_MB = n_params * dtype_bytes / 1024**2
    grad_MB = n_params * grad_factor / 1024**2
    optim_MB = n_params * optimizer_factor / 1024**2
    return {
        "n_params": n_params,
        "param_MB": param_MB,
        "grad_MB": grad_MB,
        "optim_MB": optim_MB,
        "total_MB": param_MB + grad_MB + optim_MB,
    }


def estimate_inference_memory(
    n_params: int,
    dtype_bytes: int = 2,  # fp16/bf16
    activation_MB: float = 500,  # 激活 (粗略)
) -> dict:
    """估算推理所需显存。"""
    param_MB = n_params * dtype_bytes / 1024**2
    return {
        "n_params": n_params,
        "param_MB": param_MB,
        "activation_MB": activation_MB,
        "total_MB": param_MB + activation_MB,
    }


# ============================================================
# 参数统计
# ============================================================
def print_trainable_parameters(model: nn.Module) -> dict:
    """打印可训练参数统计,返回数字 dict。"""
    n_train = 0
    n_total = 0
    for p in model.parameters():
        n_total += p.numel()
        if p.requires_grad:
            n_train += p.numel()
    n_frozen = n_total - n_train

    stats = {
        "total": n_total,
        "trainable": n_train,
        "frozen": n_frozen,
        "trainable_pct": (n_train / n_total * 100) if n_total > 0 else 0,
    }
    print(
        f"trainable params: {n_train/1e6:.4f}M || "
        f"frozen params: {n_frozen/1e6:.2f}M || "
        f"total: {n_total/1e6:.2f}M || "
        f"trainable%: {stats['trainable_pct']:.3f}%"
    )
    return stats
