"""测试 G2P 模块 (英文 token 保留 + g2pW 模型缓存/回退)。"""

from __future__ import annotations

import sys
import types

import pytest


def test_g2p_keeps_english_tokens():
    """英文 token 不应被过滤丢弃 (与推理端分布一致)。"""
    from adr.data.g2p import G2P, G2PConfig

    g2p = G2P(G2PConfig(backend="pypinyin"))
    phonemes = g2p("你好 hello world")
    assert "ni3" in phonemes
    assert "hao3" in phonemes
    assert "hello" in phonemes
    assert "world" in phonemes
    # 纯英文输入也不为空
    assert g2p("ok ok") == ["ok", "ok"]


def test_g2p_g2pw_model_cached(monkeypatch):
    """g2pW 模型应通过模块级缓存只构造一次。"""
    from adr.data import g2p as g2p_mod

    constructed = {"n": 0}

    class FakeG2PW:
        def __init__(self):
            constructed["n"] += 1

        def __call__(self, text):
            return [["ni3"], ["hao3"]]

    fake_mod = types.ModuleType("g2pW")
    fake_mod.G2PW = FakeG2PW
    monkeypatch.setitem(sys.modules, "g2pW", fake_mod)
    g2p_mod._load_g2pw.cache_clear()
    try:
        g = g2p_mod.G2P(g2p_mod.G2PConfig(backend="g2pW"))
        p1 = g("你好")
        p2 = g("你好")
        assert p1 == ["ni3", "hao3"]
        assert p2 == ["ni3", "hao3"]
        assert constructed["n"] == 1  # 两次调用共用同一模型实例
    finally:
        g2p_mod._load_g2pw.cache_clear()


def test_g2p_g2pw_fallback_on_import_error(monkeypatch):
    """g2pW 未安装时回退 pypinyin (ImportError 不被 lru_cache 缓存)。"""
    from adr.data import g2p as g2p_mod

    # sys.modules 置 None → 'from g2pW import G2PW' 抛 ImportError
    monkeypatch.setitem(sys.modules, "g2pW", None)
    g2p_mod._load_g2pw.cache_clear()
    try:
        g = g2p_mod.G2P(g2p_mod.G2PConfig(backend="g2pW"))
        phonemes = g("你好")
        assert g.config.backend == "pypinyin"  # 已回退
        assert "ni3" in phonemes and "hao3" in phonemes
    finally:
        g2p_mod._load_g2pw.cache_clear()
