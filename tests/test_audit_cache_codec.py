"""批次66 回归: tts_cache 惰性扫描/逐出容错/.tmp 清理 + audio_codec ffmpeg 失败信号。"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from adr.server import audio_codec, tts_cache


@pytest.fixture()
def tc(monkeypatch, tmp_path):
    monkeypatch.setenv("ADR_TTS_CACHE", "1")
    monkeypatch.setenv("ADR_TTS_CACHE_MAX", "2")
    monkeypatch.setenv("ADR_TTS_CACHE_DIR", str(tmp_path / "tts_cache"))
    tts_cache.clear()
    yield tts_cache
    tts_cache.clear()


def _names(tc):
    return sorted(p.name for p in tc._dir().iterdir() if p.is_file())


def _restart(tc):
    with tc._lock:
        tc._cache.clear()
        tc._total_bytes = 0
        tc._loaded = False


def test_put_at_capacity_keeps_live_disk_files(tc):
    """P3/#28: 满载后 put 不能删掉仍在内存里的条目的盘上副本。"""
    tc.put("A", b"AAAA", "wav")
    tc.put("B", b"BBBB", "wav")
    tc.put("C", b"CCCC", "wav")  # 逐出 A
    assert _names(tc) == ["B.wav", "C.wav"]
    _restart(tc)
    assert tc.get("B") == b"BBBB"
    assert tc.get("C") == b"CCCC"


def test_get_at_capacity_does_not_delete(tc):
    """P4: 纯读不能破坏持久化副本, 记账也不能被重复累加。"""
    tc.put("k1", b"one", "bin")
    tc.put("k2", b"two", "bin")
    for _ in range(3):
        assert tc.get("k1") == b"one"
    assert _names(tc) == ["k1.bin", "k2.bin"]
    assert tc._total_bytes == 6


def test_rescan_after_restart_counts_once(tc):
    """重启后扫描只发生一次, 后续 get 不再重扫/删盘。"""
    tc.put("k1", b"one", "bin")
    tc.put("k2", b"two", "bin")
    _restart(tc)
    assert tc.get("k1") == b"one"
    assert tc.get("k2") == b"two"
    assert tc._total_bytes == 6
    assert _names(tc) == ["k1.bin", "k2.bin"]


def test_evict_unlink_permission_error_is_tolerated(tc, monkeypatch):
    """#21: 逐出时盘上文件被占用 (PermissionError) 不能让 put 抛错。"""
    tc.put("A", b"AAAA", "wav")
    tc.put("B", b"BBBB", "wav")
    real_unlink = Path.unlink

    def locked_unlink(self, missing_ok=False):
        if self.name == "A.wav":
            raise PermissionError(32, "file in use")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    tc.put("C", b"CCCC", "wav")  # 不抛
    assert list(tc._cache) == ["B", "C"]
    assert tc._total_bytes == 8
    assert tc.get("C") == b"CCCC"


def test_scan_unlink_permission_error_is_tolerated(tc, monkeypatch):
    """#21: 启动扫描删超限旧文件被占用时, 扫描不中断, LRU 照常重建。"""
    d = tc._dir()
    d.mkdir(parents=True)
    for i, name in enumerate(["old", "mid", "new"]):
        f = d / f"{name}.wav"
        f.write_bytes(b"xx")
        os.utime(f, (1000 + i, 1000 + i))
    real_unlink = Path.unlink

    def locked_unlink(self, missing_ok=False):
        if self.name == "old.wav":
            raise PermissionError(32, "file in use")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    assert tc.get("new") == b"xx"
    assert list(tc._cache) == ["mid", "new"]


def test_failed_write_removes_tmp(tc, monkeypatch):
    """#22: os.replace 失败时 .tmp 不残留, 退化为纯内存条目。"""
    def boom(src, dst):
        raise PermissionError(32, "target locked")

    monkeypatch.setattr(tts_cache.os, "replace", boom)
    tc.put("K", b"data", "wav")
    assert not any(p.name.endswith(".tmp") for p in tc._dir().iterdir())
    assert tc.get("K") == b"data"
    assert tc._cache["K"][1] is None


def test_scan_cleans_leftover_tmp(tc):
    """#22: 历史残留的 .tmp 在启动扫描时清掉, 不进 LRU。"""
    d = tc._dir()
    d.mkdir(parents=True)
    (d / "stale.tmp").write_bytes(b"half")
    (d / "ok.wav").write_bytes(b"full")
    assert tc.get("ok") == b"full"
    assert _names(tc) == ["ok.wav"]


class _FakeProc:
    def __init__(self, rc, out, err):
        self.returncode, self._out, self._err = rc, out, err

    def communicate(self, input=None):
        return self._out, self._err


@pytest.mark.parametrize("packer", ["pack_ogg", "pack_mp3", "pack_aac"])
def test_ffmpeg_failure_raises(monkeypatch, packer):
    """P5: ffmpeg 非 0 退出必须抛错, 错误里带 stderr 便于排查。"""
    monkeypatch.setattr(audio_codec, "_resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(audio_codec.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(1, b"", b"Unknown encoder 'x'\n"))
    with pytest.raises(RuntimeError, match="Unknown encoder"):
        getattr(audio_codec, packer)(BytesIO(), np.zeros(160, np.float32), 32000)


@pytest.mark.parametrize("packer", ["pack_ogg", "pack_mp3", "pack_aac"])
def test_ffmpeg_success_writes_output(monkeypatch, packer):
    """成功路径不变: rc=0 时输出原样写入 buffer。"""
    monkeypatch.setattr(audio_codec, "_resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(audio_codec.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(0, b"ENCODED", b"warn"))
    buf = getattr(audio_codec, packer)(BytesIO(), np.zeros(160, np.float32), 32000)
    assert buf.getvalue() == b"ENCODED"
