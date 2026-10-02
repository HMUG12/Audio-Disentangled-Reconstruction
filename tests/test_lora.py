"""LoRA 模块单测 (M2 Day 11)。

测试:
- LoRALinear 基础前向
- apply_lora 注入 + 冻结
- 初始 BA=0 不改变输出
- 训练后 BA≠0 改变输出
- get/load state_dict 序列化
- merge_lora 正确
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ============================================================
# 基础
# ============================================================
def test_lora_linear_construction():
    """LoRALinear 构造。"""
    from adr.training.lora import LoRALinear

    layer = LoRALinear(in_features=16, out_features=32, rank=4, alpha=8)
    assert layer.in_features == 16
    assert layer.out_features == 32
    assert layer.rank == 4
    assert layer.alpha == 8
    assert layer.scaling == 2.0
    assert layer.lora_A.shape == (4, 16)
    assert layer.lora_B.shape == (32, 4)
    assert layer.weight.requires_grad is False
    assert layer.lora_A.requires_grad is True
    assert layer.lora_B.requires_grad is True


def test_lora_linear_forward():
    """LoRALinear 前向 (初始 BA=0 → 等于原 linear)。"""
    from adr.training.lora import LoRALinear

    torch.manual_seed(42)
    in_f, out_f = 8, 16
    original = nn.Linear(in_f, out_f, bias=True)

    lora = LoRALinear(in_f, out_f, rank=4, alpha=8, bias=True)
    lora.init_from_linear(original)

    x = torch.randn(2, 5, in_f)
    out_orig = original(x)
    out_lora = lora(x)

    # 初始 BA=0, 输出应该完全一致
    assert torch.allclose(out_orig, out_lora, atol=1e-6), \
        f"LoRA initial output should match original: max diff = {(out_orig - out_lora).abs().max()}"


def test_lora_linear_after_training():
    """训练后 LoRA 输出应偏离原 linear。"""
    from adr.training.lora import LoRALinear

    torch.manual_seed(42)
    in_f, out_f = 8, 16
    original = nn.Linear(in_f, out_f, bias=True)

    lora = LoRALinear(in_f, out_f, rank=4, alpha=8, bias=True)
    lora.init_from_linear(original)

    # 模拟训练: 给 lora_B 一些非零值
    with torch.no_grad():
        lora.lora_B.data = torch.randn_like(lora.lora_B) * 0.1

    x = torch.randn(2, 5, in_f)
    out_orig = original(x)
    out_lora = lora(x)

    diff = (out_orig - out_lora).abs().max()
    assert diff > 0.01, f"After LoRA training, output should differ (got diff={diff})"


# ============================================================
# apply_lora
# ============================================================
class TinyTransformer(nn.Module):
    """小型 transformer-like 模型,用于测试 LoRA。"""

    def __init__(self, dim=32, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(dim, dim),
                "v_proj": nn.Linear(dim, dim),
                "o_proj": nn.Linear(dim, dim),
                "ffn": nn.Linear(dim, dim),
            })
            for _ in range(n_layers)
        ])
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer["q_proj"](h) + layer["v_proj"](h) + layer["o_proj"](h) + layer["ffn"](h)
        return self.out(h)


def test_apply_lora_freezes_correctly():
    """apply_lora 后, 只有 LoRA 参数可训练。"""
    from adr.training.lora import LoRAConfig, apply_lora, LoRALinear

    model = TinyTransformer(dim=32, n_layers=2)
    n_total = sum(p.numel() for p in model.parameters())

    config = LoRAConfig(rank=4, alpha=8, target_modules=["q_proj", "v_proj"])
    apply_lora(model, config)

    # 1. 检查 q_proj/v_proj 被替换为 LoRALinear
    n_lora_layers = sum(
        1 for m in model.modules() if isinstance(m, LoRALinear)
    )
    assert n_lora_layers == 4  # 2 layers * (q_proj + v_proj)

    # 2. 检查只有 LoRA 参数 requires_grad=True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 4 个 LoRA 层, 每层 rank=4: A=(4,32), B=(32,4) = 128 + 128 = 256
    # 4 层 * 256 = 1024
    assert n_trainable == 4 * (4 * 32 + 32 * 4)  # = 1024
    # 注: TinyTransformer(32 dim) 总参 ~10K,LoRA 占比 ~10% (大模型上 < 0.1%)
    # 这里只验证机制正确,不验证精确比例
    assert n_trainable < 5000  # 远小于总参 10560

    # 3. 检查 o_proj/ffn 未被替换 (因为 target_modules 不包含)
    for layer in model.layers:
        assert not isinstance(layer["o_proj"], LoRALinear)
        assert not isinstance(layer["ffn"], LoRALinear)


def test_apply_lora_preserves_output_initially():
    """apply_lora 后, 初始输出应与原模型一致 (BA=0)。"""
    from adr.training.lora import LoRAConfig, apply_lora

    torch.manual_seed(42)
    model = TinyTransformer(dim=16, n_layers=2)
    model.eval()
    x = torch.randn(1, 5, 16)

    # 原输出
    with torch.no_grad():
        out_orig = model(x)

    # 应用 LoRA
    config = LoRAConfig(rank=4, alpha=8, target_modules=["q_proj"])
    apply_lora(model, config)
    model.eval()

    with torch.no_grad():
        out_lora = model(x)

    assert torch.allclose(out_orig, out_lora, atol=1e-5), \
        f"LoRA-applied model should match original initially: max diff = {(out_orig - out_lora).abs().max()}"


# ============================================================
# 序列化
# ============================================================
def test_lora_state_dict_roundtrip():
    """LoRA state_dict 保存/加载。

    注: 非 LoRA 层 (o_proj/ffn) 权重需要一致,否则 forward 输出会有差异。
    这里用两个 model 都用相同 seed 来验证。
    """
    from adr.training.lora import (
        LoRAConfig, apply_lora, get_lora_state_dict, load_lora_state_dict,
        LoRALinear,
    )

    def make_model():
        torch.manual_seed(42)  # 关键: 相同 seed
        return TinyTransformer(dim=32, n_layers=2)

    model = make_model()
    config = LoRAConfig(rank=4, alpha=8, target_modules=["q_proj", "v_proj"])
    apply_lora(model, config)

    # 改一下 LoRA B (模拟训练)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.data = torch.randn_like(m.lora_B) * 0.1

    # 1. 提取
    state = get_lora_state_dict(model)
    assert len(state) == 4 * 2  # 4 LoRA layers * (A, B)
    for k, v in state.items():
        assert v.requires_grad is False  # 已 detach

    # 2. 加载到新模型
    new_model = make_model()  # 同样 seed
    apply_lora(new_model, config)

    loaded, missing = load_lora_state_dict(new_model, state, strict=True)
    assert loaded == 4 * 2
    assert missing == 0

    # 3. 验证加载后 LoRA 参数完全一致
    for name, m1 in model.named_modules():
        if isinstance(m1, LoRALinear):
            name2 = name
            m2 = dict(new_model.named_modules())[name2]
            assert torch.allclose(m1.lora_A, m2.lora_A, atol=1e-6), \
                f"{name}.lora_A 不一致"
            assert torch.allclose(m1.lora_B, m2.lora_B, atol=1e-6), \
                f"{name}.lora_B 不一致"

    # 4. forward 一致 (因为 model 权重 seed 相同 + LoRA 加载相同)
    x = torch.randn(1, 5, 32)
    with torch.no_grad():
        out1 = model(x)
        out2 = new_model(x)
    # FP32 累计误差 < 1e-4
    assert torch.allclose(out1, out2, atol=1e-4)


# ============================================================
# Merge
# ============================================================
def test_merge_lora_consistent():
    """merge_lora 后, 输出应与 LoRA 版本一致。"""
    from adr.training.lora import (
        LoRAConfig, apply_lora, merge_lora, LoRALinear,
    )

    torch.manual_seed(42)
    model = TinyTransformer(dim=16, n_layers=2)
    config = LoRAConfig(rank=4, alpha=8, target_modules=["q_proj", "v_proj"])
    apply_lora(model, config)

    # 改 LoRA
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.data = torch.randn_like(m.lora_B) * 0.1

    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        out_lora = model(x)
        out_merged = merge_lora(model)(x)

    assert torch.allclose(out_lora, out_merged, atol=1e-5), \
        f"Merged model should match LoRA model: max diff = {(out_lora - out_merged).abs().max()}"


# ============================================================
# target_modules 匹配
# ============================================================
def test_target_modules_exclude():
    """exclude_modules 排除正确。"""
    from adr.training.lora import LoRAConfig, apply_lora, LoRALinear

    model = TinyTransformer(dim=16, n_layers=1)
    config = LoRAConfig(
        rank=4, alpha=8,
        target_modules=["q_proj", "v_proj"],
        exclude_modules=["layers.0.q_proj"],  # 排除第一个
    )
    apply_lora(model, config)

    n_lora = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    assert n_lora == 1  # 只有 layers.0.v_proj


# ============================================================
# 显存节省
# ============================================================
def test_lora_memory_savings():
    """LoRA 大幅减少可训练参数。"""
    from adr.training.lora import LoRAConfig, apply_lora

    # 0.3B-ish 模型 (实际 ~ 0.3M, 测试用)
    model = TinyTransformer(dim=512, n_layers=6)
    n_total = sum(p.numel() for p in model.parameters())

    config = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    apply_lora(model, config)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = n_trainable / n_total

    # 注意: TinyTransformer 故意小 (32 dim, 6 layers),总参数 ~200K。
    # 真实大模型 (0.3B+) 的 LoRA 占比 < 0.1%,这里放宽到 5% 验证机制。
    assert ratio < 0.05, f"LoRA should be < 5% of total params, got {ratio*100:.2f}%"
    print(f"  Total: {n_total/1e6:.2f}M, LoRA trainable: {n_trainable/1e6:.4f}M ({ratio*100:.3f}%)")


# ============================================================
# Trainer 集成 (简化版, 不依赖完整 Trainer)
# ============================================================
def test_lora_with_optimizer():
    """LoRA 模型 + Adam 优化器能正常 step。"""
    from adr.training.lora import LoRAConfig, apply_lora

    torch.manual_seed(42)
    model = TinyTransformer(dim=16, n_layers=2)
    config = LoRAConfig(rank=4, alpha=8, target_modules=["q_proj", "v_proj"])
    apply_lora(model, config)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    # 训练 3 步
    x = torch.randn(2, 5, 16)
    target = torch.randn(2, 5, 16)

    initial_loss = None
    for step in range(3):
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        if initial_loss is None:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < initial_loss * 1.1, \
        f"Loss should not increase: {initial_loss} -> {loss.item()}"


# ============================================================
# PEFT 可选集成
# ============================================================
def test_try_apply_peft():
    """PEFT 集成 (若未装返回 False)。"""
    from adr.training.lora import try_apply_peft

    model = TinyTransformer(dim=16, n_layers=1)
    peft_model, success = try_apply_peft(
        model, rank=4, alpha=8, target_modules=["q_proj"]
    )
    # 不论 PEFT 装没装,函数都应返回 (model, bool)
    assert peft_model is not None
    assert isinstance(success, bool)
