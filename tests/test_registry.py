"""测试 adr.core.registry 模块。"""

import pytest

from adr.core.registry import (
    BACKBONE_REGISTRY,
    REGISTRY,
    Registry,
    register,
)


def test_registry_decorator():
    """装饰器注册用法。"""
    reg = Registry("test_decorator")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.has("foo")
    assert reg.get("foo") is Foo


def test_registry_explicit():
    """显式注册用法。"""

    class Bar:
        pass

    reg = Registry("test_explicit")
    reg.register("bar", Bar)
    assert reg.get("bar") is Bar


def test_registry_contains():
    """in 操作符可用。"""
    reg = Registry("test_contains")

    @reg.register("x")
    class X:
        pass

    assert "x" in reg
    assert "y" not in reg


def test_registry_clear():
    """clear 清理所有注册。"""
    reg = Registry("test_clear")

    @reg.register("a")
    class A:
        pass

    assert len(reg) == 1
    reg.clear()
    assert len(reg) == 0


def test_top_level_register():
    """顶层 register 函数。"""

    @register("backbone", "test_backbone_for_unittest")
    class TestBackbone:
        pass

    assert BACKBONE_REGISTRY.has("test_backbone_for_unittest")
    assert BACKBONE_REGISTRY.get("test_backbone_for_unittest") is TestBackbone

    # 清理
    BACKBONE_REGISTRY._registry.pop("test_backbone_for_unittest", None)


def test_invalid_category():
    """无效 category 抛错。"""
    with pytest.raises(ValueError):
        register("invalid_category", "x")


def test_hub_all():
    """REGISTRY.all() 包含所有子注册表。"""
    all_regs = REGISTRY.all()
    assert "backbone" in all_regs
    assert "vocoder" in all_regs
    assert "asr" in all_regs
    assert "g2p" in all_regs
