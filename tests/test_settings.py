"""批次36: settings 单源读取函数口径测试。

ADR_ 前缀环境变量的解析集中在 adr/core/settings.py 后, 这里逐函数
验证默认值 / 解析 / 容错回退与被替换的原地读取**逐字等价** — 单源
收敛不允许悄悄改行为。函数式读取 (每次调用读 env) 保证 monkeypatch
setenv/delenv 即时生效 (设计备案见 settings 模块 docstring)。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from adr.core import settings

MB = 1024 * 1024


# ===== 服务面 =====
def test_api_keys_unset(monkeypatch):
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    assert settings.api_keys() == []
    assert settings.has_api_key() is False


def test_api_keys_parse(monkeypatch):
    monkeypatch.setenv("ADR_TTS_API_KEY", "k1, k2")
    assert settings.api_keys() == ["k1", "k2"]
    assert settings.has_api_key() is True


def test_api_keys_filter_empty(monkeypatch):
    monkeypatch.setenv("ADR_TTS_API_KEY", " k1 ,, ,k3 ")
    assert settings.api_keys() == ["k1", "k3"]


def test_has_api_key_blank(monkeypatch):
    monkeypatch.setenv("ADR_TTS_API_KEY", "   ")
    assert settings.api_keys() == []
    assert settings.has_api_key() is False


def test_default_profile(monkeypatch):
    monkeypatch.delenv("ADR_TTS_DEFAULT_PROFILE", raising=False)
    assert settings.default_profile() is None
    monkeypatch.setenv("ADR_TTS_DEFAULT_PROFILE", "")
    assert settings.default_profile() is None
    monkeypatch.setenv("ADR_TTS_DEFAULT_PROFILE", "demo")
    assert settings.default_profile() == "demo"


# ===== TTS 结果缓存 =====
def test_tts_cache_enabled(monkeypatch):
    monkeypatch.delenv("ADR_TTS_CACHE", raising=False)
    assert settings.tts_cache_enabled() is True
    monkeypatch.setenv("ADR_TTS_CACHE", "0")
    assert settings.tts_cache_enabled() is False
    monkeypatch.setenv("ADR_TTS_CACHE", "1")
    assert settings.tts_cache_enabled() is True


def test_tts_cache_dir(monkeypatch):
    monkeypatch.delenv("ADR_TTS_CACHE_DIR", raising=False)
    assert settings.tts_cache_dir() == settings.REPO_ROOT / "data" / "tts_cache"
    monkeypatch.setenv("ADR_TTS_CACHE_DIR", "E:/tmp/tc")
    assert settings.tts_cache_dir() == Path("E:/tmp/tc")


def test_tts_cache_max_entries(monkeypatch):
    monkeypatch.delenv("ADR_TTS_CACHE_MAX", raising=False)
    assert settings.tts_cache_max_entries() == 32
    monkeypatch.setenv("ADR_TTS_CACHE_MAX", "5")
    assert settings.tts_cache_max_entries() == 5
    monkeypatch.setenv("ADR_TTS_CACHE_MAX", "0")   # max(1, 0) 钳到 1
    assert settings.tts_cache_max_entries() == 1
    monkeypatch.setenv("ADR_TTS_CACHE_MAX", "abc")  # 非法回退默认
    assert settings.tts_cache_max_entries() == 32


def test_tts_cache_max_bytes(monkeypatch):
    monkeypatch.delenv("ADR_TTS_CACHE_MB", raising=False)
    assert settings.tts_cache_max_bytes() == 256 * MB   # 未设置 → 默认
    monkeypatch.setenv("ADR_TTS_CACHE_MB", "64")
    assert settings.tts_cache_max_bytes() == 64 * MB
    monkeypatch.setenv("ADR_TTS_CACHE_MB", "0")
    assert settings.tts_cache_max_bytes() == 256 * MB   # 0 → or 回退默认 (特殊口径)
    monkeypatch.setenv("ADR_TTS_CACHE_MB", "abc")
    assert settings.tts_cache_max_bytes() == 256 * MB


# ===== 流式句级缓存 =====
def test_seg_cache_enabled(monkeypatch):
    monkeypatch.delenv("ADR_SEG_CACHE", raising=False)
    assert settings.seg_cache_enabled() is True
    monkeypatch.setenv("ADR_SEG_CACHE", "0")
    assert settings.seg_cache_enabled() is False


def test_seg_cache_max_items(monkeypatch):
    monkeypatch.delenv("ADR_SEG_CACHE_MAX", raising=False)
    assert settings.seg_cache_max_items() == 128
    monkeypatch.setenv("ADR_SEG_CACHE_MAX", "2")
    assert settings.seg_cache_max_items() == 2
    monkeypatch.setenv("ADR_SEG_CACHE_MAX", "0")   # 0 = 禁用条数上限 (合法)
    assert settings.seg_cache_max_items() == 0
    monkeypatch.setenv("ADR_SEG_CACHE_MAX", "abc")
    assert settings.seg_cache_max_items() == 128


def test_seg_cache_max_bytes(monkeypatch):
    monkeypatch.delenv("ADR_SEG_CACHE_MB", raising=False)
    assert settings.seg_cache_max_bytes() == 256 * MB
    monkeypatch.setenv("ADR_SEG_CACHE_MB", "10")
    assert settings.seg_cache_max_bytes() == 10 * MB
    monkeypatch.setenv("ADR_SEG_CACHE_MB", "0")
    assert settings.seg_cache_max_bytes() == 0      # 0 = 禁用字节上限 (与 tts_cache 不同!)
    monkeypatch.setenv("ADR_SEG_CACHE_MB", "abc")
    assert settings.seg_cache_max_bytes() == 256 * MB


# ===== 引擎运行时开关 =====
def test_fragment_interval_default(monkeypatch):
    monkeypatch.delenv("ADR_TTS_FRAGMENT_INTERVAL", raising=False)
    assert settings.fragment_interval_default() == pytest.approx(0.3)
    monkeypatch.setenv("ADR_TTS_FRAGMENT_INTERVAL", "0.5")
    assert settings.fragment_interval_default() == pytest.approx(0.5)
    monkeypatch.setenv("ADR_TTS_FRAGMENT_INTERVAL", "")  # 空串 → or 回退
    assert settings.fragment_interval_default() == pytest.approx(0.3)
    monkeypatch.setenv("ADR_TTS_FRAGMENT_INTERVAL", "abc")  # 原口径 float() 直接崩
    with pytest.raises(ValueError):
        settings.fragment_interval_default()


def test_keep_switches(monkeypatch):
    for name, fn in (("ADR_TTS_KEEP_TQDM", settings.keep_tqdm),
                     ("ADR_TTS_KEEP_EMPTY_CACHE", settings.keep_empty_cache)):
        monkeypatch.delenv(name, raising=False)
        assert fn() is False
        monkeypatch.setenv(name, "1")
        assert fn() is True
        monkeypatch.setenv(name, "0")
        assert fn() is False
