"""批次65: 流式合成跨线程迭代回归 (StreamingResponse 逐 next() 可能换线程)。

旧实现持 _CHDIR_LOCK(RLock) 跨 yield, 换线程恢复时 release 抛错并永久卡锁。
新实现: 持锁合成跑在单一 worker 线程, 生成器只排空队列。断言:
- 每个 next() 换一个线程迭代, 结束后 _CHDIR_LOCK 可被其它线程立即获取
- 消费端中途 close (断连) → worker 线程退出, 不泄漏
- worker 内异常转发到消费端重抛
"""
import threading
import time

import numpy as np
import pytest

from adr.models import gsv_engine
from adr.models.gsv_engine import GSVEngine


class FakeTTS:
    def __init__(self, blocks=3, delay=0.0, fail_at=None):
        self.blocks, self.delay, self.fail_at = blocks, delay, fail_at

    def run(self, inputs):
        for i in range(self.blocks):
            if self.fail_at == i:
                raise RuntimeError("boom")
            if self.delay:
                time.sleep(self.delay)
            yield 32000, np.full(160, 0.1, dtype=np.float32)


@pytest.fixture(autouse=True)
def _no_seg_cache(monkeypatch):
    monkeypatch.setenv("ADR_SEG_CACHE", "0")


@pytest.fixture
def eng(monkeypatch):
    """真 _gsv_context 会 chdir 到 gsv_dir; 换成同样持 _CHDIR_LOCK 的替身。"""
    import contextlib
    e = GSVEngine()
    monkeypatch.setattr(e, "_lazy_init", lambda: None)
    monkeypatch.setattr(e, "_ensure_weights", lambda v, t: None)

    @contextlib.contextmanager
    def _ctx():
        with gsv_engine._CHDIR_LOCK:
            yield
    monkeypatch.setattr(e, "_gsv_context", _ctx)
    return e


def _next_in_new_thread(gen):
    box = {}

    def run():
        try:
            box["v"] = next(gen)
        except StopIteration:
            box["stop"] = True
        except BaseException as ex:   # noqa: BLE001
            box["err"] = ex
    t = threading.Thread(target=run)
    t.start()
    t.join(10)
    assert not t.is_alive(), "next() 挂死"
    return box


def _lock_free():
    ok = gsv_engine._CHDIR_LOCK.acquire(timeout=2)
    if ok:
        gsv_engine._CHDIR_LOCK.release()
    return ok


def _stream_workers():
    return [t for t in threading.enumerate() if t.name == "adr-gsv-stream"]


def test_cross_thread_iteration_releases_lock(eng):
    eng._tts = FakeTTS(blocks=3)
    gen = eng.synthesize_stream("你好世界。", ref_audio="ref.wav")
    got = 0
    while True:
        box = _next_in_new_thread(gen)
        assert "err" not in box, box.get("err")
        if box.get("stop"):
            break
        got += 1
    assert got == 3
    assert _lock_free()
    assert not _stream_workers()
    assert eng._stalled is False   # 批次82 (#1): 正常收尾不降级


class HalfOpenTTS:
    """先产出一块, 随后卡在 run() 里不返回 (模拟上游卡死/半开连接)。"""

    def __init__(self, release):
        self.release = release

    def run(self, inputs):
        yield 32000, np.full(160, 0.1, dtype=np.float32)
        self.release.wait(30)


def test_abandoned_consumer_does_not_block_engine(eng):
    """批次82 (#1): 消费端停止迭代也不让 producer 卡在 put 里持锁。

    旧实现队列 maxsize=8, worker 写满后阻塞在 q.put (持引擎 _lock),
    其它合成请求全部饿死; 无界队列后 worker 能跑完并释放锁。
    """
    eng._tts = FakeTTS(blocks=50, delay=0.002)
    gen = eng.synthesize_stream("你好世界。", ref_audio="ref.wav")
    try:
        next(gen)   # 只取一块, 之后既不消费也不关闭 (模拟半开连接)
        deadline = time.time() + 5
        while _stream_workers() and time.time() < deadline:
            time.sleep(0.02)
        assert not _stream_workers(), "worker 卡在 put 上未退出"
        assert _lock_free()
    finally:
        gen.close()


def test_stalled_worker_marks_engine_for_rebuild(eng, monkeypatch):
    """批次82 (#1): worker 超时未退出 → 清理有界返回 + 引擎降级重建。"""
    release = threading.Event()
    eng._tts = HalfOpenTTS(release)
    monkeypatch.setattr(gsv_engine, "_STREAM_DRAIN_SEC", 0.2)
    monkeypatch.setattr(gsv_engine, "_STREAM_JOIN_SEC", 0.2)

    gen = eng.synthesize_stream("你好世界。", ref_audio="ref.wav")
    try:
        next(gen)     # 拿到第一块后 producer 卡在 run() 里, 仍持着 _lock
        gen.close()   # 旧实现这里会永久阻塞 (q.get 无超时)
    finally:
        release.set()

    assert eng._stalled is True
    deadline = time.time() + 5
    while _stream_workers() and time.time() < deadline:
        time.sleep(0.02)
    assert not _stream_workers()
    assert _lock_free()

    # 降级标记 → 下次 _lazy_init 丢弃模型与权重指纹, 强制重建
    assert eng.is_ready is False
    assert eng._drop_stalled_model() is True
    assert eng._tts is None
    assert eng._loaded == {"t2s": None, "vits": None}
    assert eng._stalled is False
    assert eng.is_ready is False


def test_module_is_ready_respects_stalled(monkeypatch):
    """批次88 (BUG-05): 模块级 is_ready 不再把降级引擎报成就绪。"""

    class _Fake:
        def __init__(self, stalled):
            self._tts = object()
            self._stalled = stalled

    monkeypatch.setattr(gsv_engine, "_ENGINE", _Fake(True))
    assert gsv_engine.is_ready() is False

    monkeypatch.setattr(gsv_engine, "_ENGINE", _Fake(False))
    assert gsv_engine.is_ready() is True

    monkeypatch.setattr(gsv_engine, "_ENGINE", None)
    assert gsv_engine.is_ready() is False


def test_client_disconnect_stops_worker(eng):
    eng._tts = FakeTTS(blocks=50, delay=0.01)
    gen = eng.synthesize_stream("你好世界。", ref_audio="ref.wav")
    next(gen)
    gen.close()   # 模拟 StreamingResponse 断连关闭
    deadline = time.time() + 5
    while _stream_workers() and time.time() < deadline:
        time.sleep(0.02)
    assert not _stream_workers()
    assert _lock_free()


def test_worker_error_reraised_to_consumer(eng):
    eng._tts = FakeTTS(blocks=3, fail_at=1)
    gen = eng.synthesize_stream("你好世界。", ref_audio="ref.wav")
    with pytest.raises(RuntimeError, match="boom"):
        list(gen)
    assert _lock_free()


# ─── 批次86 (CON-01): 上游永卡 → 有界返回 503 语义, 进程不假死 ───

class NeverYieldsTTS:
    """run() 既不产出也不返回 (模拟上游死锁 / 半开连接永久挂起)。"""

    def __init__(self, release):
        self.release = release

    def run(self, inputs):
        self.release.wait(30)
        if False:   # 保持生成器语义
            yield 32000, np.zeros(1, dtype=np.float32)


def test_idle_stream_marks_stalled_and_next_request_bounded(eng, monkeypatch):
    """上游永不产出: 消费端静默超时即降级, 后续请求有界返回 503 而非永久排队。

    旧实现 q.get() 无超时 + 调度准入无超时 → 卡死 worker 永久占住槽位与
    引擎锁, 第二个请求永不返回 (全站假死)。
    """
    from adr.core.exceptions import EngineStalledError

    release = threading.Event()
    eng._tts = NeverYieldsTTS(release)
    monkeypatch.setattr(gsv_engine, "_STREAM_IDLE_SEC", 0.2)
    monkeypatch.setattr(gsv_engine, "_STREAM_JOIN_SEC", 0.1)
    monkeypatch.setattr(gsv_engine, "_LOCK_WAIT_SEC", 0.2)

    gen = eng.synthesize_stream("你好世界。", ref_audio="ref.wav")
    try:
        t0 = time.monotonic()
        with pytest.raises(EngineStalledError):
            next(gen)
        assert time.monotonic() - t0 < 5
    finally:
        gen.close()

    assert eng._stalled is True
    # 第二个请求: worker 仍占着调度槽/引擎锁 → 有界返回 EngineStalledError (503)
    t0 = time.monotonic()
    with pytest.raises(EngineStalledError):
        eng.synthesize("你好世界。", ref_audio="ref.wav")
    assert time.monotonic() - t0 < 5
    release.set()


def test_engine_lock_wait_timeout_bounded(eng, monkeypatch):
    """引擎锁被长期占用: 流式与非流式都在有限时间内抛 EngineStalledError。"""
    from adr.core.exceptions import EngineStalledError

    monkeypatch.setattr(gsv_engine, "_LOCK_WAIT_SEC", 0.2)
    eng._tts = FakeTTS(blocks=1)

    held, release = threading.Event(), threading.Event()

    def _hold():
        eng._lock.acquire()
        held.set()
        release.wait(10)
        eng._lock.release()

    th = threading.Thread(target=_hold, daemon=True)
    th.start()
    assert held.wait(2)
    try:
        t0 = time.monotonic()
        with pytest.raises(EngineStalledError):
            list(eng.synthesize_stream("你好世界。", ref_audio="ref.wav"))
        assert time.monotonic() - t0 < 5

        t0 = time.monotonic()
        with pytest.raises(EngineStalledError):
            eng.synthesize("你好世界。", ref_audio="ref.wav")
        assert time.monotonic() - t0 < 5
    finally:
        release.set()
        th.join(5)
    assert _lock_free()
