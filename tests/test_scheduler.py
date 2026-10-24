"""SynthesisScheduler 测试 (批次63 第三子批, m10 ④ N=1 硬预算)。

- N=1 互斥与排队可观测 (waiting/busy/admitted)
- N=2 双槽并发 (预算可调, 为 1.3.x 固定份额预留)
- stats 快照字段
- GSVEngine 接线: queue_depth/synth_busy 委托 + 请求级准入 (批次23 语义保持)
"""
import contextlib
import threading
import time

import numpy as np
import pytest

from adr.core.scheduler import SynthesisScheduler
from adr.models import gsv_engine
from adr.models.gsv_engine import GSVEngine


def _wait_until(pred, timeout=2.0):
    """轮询到条件成立 (线程时序不确定, 自带 deadline)。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_slot_serializes_n1():
    """N=1: 第二请求排队等待, 释放后准入; 计数全程一致。"""
    s = SynthesisScheduler(1)
    tickets = []
    with s.slot() as tk0:
        assert tk0.busy == 1 and tk0.waiting == 0 and tk0.wait_ms == 0
        assert s.busy == 1 and s.waiting == 0

        def _take():
            with s.slot() as tk:
                tickets.append(tk)

        th = threading.Thread(target=_take)
        th.start()
        assert _wait_until(lambda: s.waiting == 1)
        assert s.busy == 1            # 预算满, 第二请求未准入
        assert tickets == []
    th.join(timeout=2)
    assert not th.is_alive()
    assert len(tickets) == 1
    tk = tickets[0]
    assert tk.busy == 1 and tk.waiting == 0
    st = s.stats()
    assert st["busy"] == 0 and st["waiting"] == 0
    assert st["admitted_total"] == 2
    assert st["max_waiting_observed"] == 1


def test_n2_allows_two_concurrent():
    """N=2: 预算内两槽并发 (1.3.x 固定份额插槽位)。"""
    s = SynthesisScheduler(2)
    with s.slot():
        with s.slot() as tk2:
            assert s.busy == 2
            assert tk2.busy == 2
        assert s.stats()["admitted_total"] == 2
    assert s.busy == 0


def test_stats_snapshot():
    """stats 字段齐全且随准入累计。"""
    s = SynthesisScheduler(1)
    st = s.stats()
    assert st["max_concurrent"] == 1
    assert st["admitted_total"] == 0 and st["busy"] == 0
    with s.slot():
        pass
    st = s.stats()
    assert st["admitted_total"] == 1
    assert st["last_wait_ms"] >= 0 and st["max_wait_ms"] >= 0
    assert st["max_waiting_observed"] == 0   # 无竞争准入不计排队


def test_env_bad_value_falls_back(monkeypatch):
    """非数字预算回落 1 (settings 单源解析)。"""
    from adr.core import settings
    monkeypatch.setenv("ADR_SCHED_MAX_CONCURRENCY", "abc")
    assert settings.scheduler_max_concurrency() == 1
    monkeypatch.setenv("ADR_SCHED_MAX_CONCURRENCY", "0")
    assert settings.scheduler_max_concurrency() == 1
    monkeypatch.setenv("ADR_SCHED_MAX_CONCURRENCY", "3")
    assert settings.scheduler_max_concurrency() == 3


# ── GSVEngine 接线 ──

class FakeTTS:
    def __init__(self):
        self.sr = 32000
        self.calls = []

    def run(self, inputs):
        self.calls.append(inputs.get("text", ""))
        for i in range(2):
            yield self.sr, np.full(160, 0.1, dtype=np.float32)


@pytest.fixture(autouse=True)
def _clean_seg_cache():
    gsv_engine._seg_cache_clear()
    yield
    gsv_engine._seg_cache_clear()


@pytest.fixture
def eng(monkeypatch) -> GSVEngine:
    """轻量 GSVEngine: _lazy_init / _gsv_context / _ensure_weights 全旁路。"""
    e = GSVEngine()
    monkeypatch.setattr(e, "_lazy_init", lambda: None)
    monkeypatch.setattr(e, "_gsv_context", contextlib.nullcontext)
    monkeypatch.setattr(e, "_ensure_weights", lambda v, t: None)
    return e


def test_engine_admission_counts(eng):
    """非流式: 合成后 admitted 计 1, busy/queue 归零 (批次23 语义保持)。"""
    eng._tts = FakeTTS()
    audio, sr = eng.synthesize("你好", "ref.wav")
    assert audio.shape == (160,) and sr == 32000   # 非流式取最后一次 yield 的块
    st = eng.sched_stats()
    assert st["admitted_total"] == 1
    assert st["busy"] == 0
    assert eng.queue_depth() == 0 and eng.synth_busy() == 0


def test_engine_stream_request_level_admission(eng):
    """流式: 请求级准入 — 多段共享一个槽, admitted 只计 1。"""
    eng._tts = FakeTTS()
    chunks = [(c, sr) for c, sr in
              eng.synthesize_stream("你好。世界。", ref_audio="ref.wav")]
    assert chunks
    st = eng.sched_stats()
    assert st["admitted_total"] == 1
    assert eng.synth_busy() == 0 and eng.queue_depth() == 0


def test_engine_env_budget(monkeypatch):
    """ADR_SCHED_MAX_CONCURRENCY 贯通到引擎调度器。"""
    monkeypatch.setenv("ADR_SCHED_MAX_CONCURRENCY", "2")
    assert GSVEngine()._sched.max_concurrent == 2
    monkeypatch.delenv("ADR_SCHED_MAX_CONCURRENCY", raising=False)
    assert GSVEngine()._sched.max_concurrent == 1
