"""批次66 回归: tts_cache 惰性扫描/逐出容错/.tmp 清理 + audio_codec ffmpeg 失败信号。"""
from __future__ import annotations

import os
import subprocess
import threading
import time
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


def _disk(tc, key, ext="wav"):
    """批次69: 盘上文件名带版本前缀, 断言一律走 _fname 生成, 不硬编码。"""
    return tc._fname(key, ext)


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
    assert _names(tc) == [_disk(tc, "B"), _disk(tc, "C")]
    _restart(tc)
    assert tc.get("B") == b"BBBB"
    assert tc.get("C") == b"CCCC"


def test_get_at_capacity_does_not_delete(tc):
    """P4: 纯读不能破坏持久化副本, 记账也不能被重复累加。"""
    tc.put("k1", b"one", "bin")
    tc.put("k2", b"two", "bin")
    for _ in range(3):
        assert tc.get("k1") == b"one"
    assert _names(tc) == [_disk(tc, "k1", "bin"), _disk(tc, "k2", "bin")]
    assert tc._total_bytes == 6


def test_rescan_after_restart_counts_once(tc):
    """重启后扫描只发生一次, 后续 get 不再重扫/删盘。"""
    tc.put("k1", b"one", "bin")
    tc.put("k2", b"two", "bin")
    _restart(tc)
    assert tc.get("k1") == b"one"
    assert tc.get("k2") == b"two"
    assert tc._total_bytes == 6
    assert _names(tc) == [_disk(tc, "k1", "bin"), _disk(tc, "k2", "bin")]


def test_scan_discards_legacy_and_old_version_files(tc):
    """批次69 (Q5): 无前缀旧格式 / 旧缓存版本文件扫描时丢弃, 不再命中。"""
    d = tc._dir()
    d.mkdir(parents=True)
    (d / "v0-deadbeef.wav").write_bytes(b"old")          # 旧缓存版本
    (d / ("a" * 40 + ".wav")).write_bytes(b"legacy")     # 版本前缀引入前的格式
    (d / _disk(tc, "fresh")).write_bytes(b"new")
    assert tc.get("deadbeef") is None
    assert tc.get("a" * 40) is None
    assert tc.get("fresh") == b"new"
    assert _names(tc) == [_disk(tc, "fresh")]


def test_evict_unlink_permission_error_is_tolerated(tc, monkeypatch):
    """#21: 逐出时盘上文件被占用 (PermissionError) 不能让 put 抛错。"""
    tc.put("A", b"AAAA", "wav")
    tc.put("B", b"BBBB", "wav")
    real_unlink = Path.unlink

    def locked_unlink(self, missing_ok=False):
        if self.name == _disk(tc, "A"):
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
    now = time.time()
    for i, name in enumerate(["old", "mid", "new"]):
        f = d / _disk(tc, name)
        f.write_bytes(b"xx")
        # S11 TTL 只看 mtime, 这里用"刚刚"的时间戳, 只验证超条数清理不被占用打断
        os.utime(f, (now - 30 + i, now - 30 + i))
    real_unlink = Path.unlink

    def locked_unlink(self, missing_ok=False):
        if self.name == _disk(tc, "old"):
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
    (d / _disk(tc, "ok")).write_bytes(b"full")
    assert tc.get("ok") == b"full"
    assert _names(tc) == [_disk(tc, "ok")]


# ─── 批次79 (#5): 磁盘 IO 移出全局锁 ───

def test_get_reads_disk_outside_lock(tc, monkeypatch):
    """#5: 磁盘条目首次读入在锁外 — 单条最大 256MB, 持锁读盘会冻住所有请求。"""
    tc.put("big", b"payload", "wav")
    _restart(tc)
    locked_at_read = []
    real_read = Path.read_bytes

    def spy(self):
        locked_at_read.append(tc._lock.locked())
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    assert tc.get("big") == b"payload"
    assert locked_at_read == [False]


def test_put_writes_disk_outside_lock(tc, monkeypatch):
    """#5: 落盘 (write_bytes) 在锁外, 写盘期间其它请求仍可读写缓存。"""
    locked_at_write = []
    real_write = Path.write_bytes

    def spy(self, data):
        locked_at_write.append(tc._lock.locked())
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", spy)
    tc.put("k", b"data", "wav")
    assert locked_at_write and not any(locked_at_write)


def test_evict_unlink_outside_lock(tc, monkeypatch):
    """#5: LRU 逐出的删盘也在锁外 (占用/杀软下 unlink 可能长时间阻塞)。"""
    tc.put("A", b"AAAA", "wav")
    tc.put("B", b"BBBB", "wav")
    locked_at_unlink = []
    real_unlink = Path.unlink

    def spy(self, missing_ok=False):
        locked_at_unlink.append(tc._lock.locked())
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", spy)
    tc.put("C", b"CCCC", "wav")  # 逐出 A
    assert locked_at_unlink and not any(locked_at_unlink)
    assert list(tc._cache) == ["B", "C"]


def test_get_backfill_skips_after_clear(tc, monkeypatch):
    """#5: 读盘期间缓存被 clear — 不回填旧条目, 记账归零 (字节仍可返回)。"""
    tc.put("k", b"hello", "wav")
    _restart(tc)
    real_read = Path.read_bytes

    def read_then_clear(self):
        data = real_read(self)
        tts_cache.clear()
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_clear)
    assert tc.get("k") == b"hello"
    assert "k" not in tc._cache
    assert tc._total_bytes == 0


def test_get_missing_disk_file_drops_entry(tc):
    """盘上文件被外部删除 → 未命中且按记录值回减, 账目不漂移。"""
    tc.put("keep", b"kk", "wav")
    tc.put("gone", b"gggg", "wav")
    _restart(tc)
    assert tc.get("keep") == b"kk"      # 触发扫描: gone 只登记文件名, 不读盘
    (tc._dir() / _disk(tc, "gone")).unlink()
    assert tc.get("gone") is None
    assert "gone" not in tc._cache
    assert tc._total_bytes == 2


def test_concurrent_get_put_keeps_accounting(tc):
    """#5: 并发 get/put/逐出不死锁, 记账始终等于条目大小之和。"""
    tc.put("seed", b"1234567890", "wav")
    errors = []

    def worker(i):
        try:
            for n in range(20):
                key = f"k{i}-{n % 3}"
                tc.put(key, b"x" * 8, "wav")
                tc.get("seed")
                tc.get(key)
        except Exception as e:  # 线程内异常带回主线程断言
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    with tc._lock:
        assert tc._total_bytes == sum(v[2] for v in tc._cache.values())


class _FakeProc:
    def __init__(self, rc, out, err):
        self.returncode, self._out, self._err = rc, out, err
        self.killed = False

    def communicate(self, input=None, timeout=None):
        return self._out, self._err

    def kill(self):
        self.killed = True


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


class _HangProc:
    """批次78 (#14): 首次 communicate 超时, 回收调用正常返回。"""

    returncode = None

    def __init__(self):
        self.timeouts = []
        self.killed = False

    def communicate(self, input=None, timeout=None):
        self.timeouts.append(timeout)
        if len(self.timeouts) == 1:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout or 0)
        return b"", b""

    def kill(self):
        self.killed = True


@pytest.mark.parametrize("packer", ["pack_ogg", "pack_mp3", "pack_aac"])
def test_ffmpeg_timeout_kills_and_raises(monkeypatch, packer):
    """#14: ffmpeg 卡死必须带超时调用, 超时 kill 子进程并抛可读错误。"""
    monkeypatch.setattr(audio_codec, "_resolve_ffmpeg", lambda: "ffmpeg")
    proc = _HangProc()
    monkeypatch.setattr(audio_codec.subprocess, "Popen", lambda *a, **k: proc)
    with pytest.raises(RuntimeError, match="超时"):
        getattr(audio_codec, packer)(BytesIO(), np.zeros(160, np.float32), 32000)
    assert proc.killed is True
    assert len(proc.timeouts) == 2          # 第一次带超时, 第二次回收
    assert proc.timeouts[0] == audio_codec._encode_timeout()


def test_encode_timeout_env(monkeypatch):
    """#14: 超时秒数可由 ADR_FFMPEG_TIMEOUT_SEC 调整, 非法值回落默认。"""
    monkeypatch.delenv("ADR_FFMPEG_TIMEOUT_SEC", raising=False)
    assert audio_codec._encode_timeout() == 120.0
    monkeypatch.setenv("ADR_FFMPEG_TIMEOUT_SEC", "7.5")
    assert audio_codec._encode_timeout() == 7.5
    monkeypatch.setenv("ADR_FFMPEG_TIMEOUT_SEC", "-3")
    assert audio_codec._encode_timeout() == 120.0

