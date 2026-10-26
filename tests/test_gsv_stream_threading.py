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
