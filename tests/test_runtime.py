"""v1.3.x 次子批: 调度层 Runtime 闭环 (纯逻辑, 不依赖真模型)。"""
from __future__ import annotations

import pytest

from adr.core import policy, runtime


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    # 模块状态与分块键隔离由 conftest._isolate_runtime 负责; 这里打开微调
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADR_SAFETY_AUDIT", "0")
    monkeypatch.delenv("ADR_RUNTIME_AUTOTUNE")
    monkeypatch.setattr(policy, "_applied", {})


def _feed(n, rtf=1.5, ok=True):
    for _ in range(n):
        runtime.record(ok=ok, elapsed_s=rtf * 2.0, audio_s=2.0 if ok else 0.0)


def test_slow_window_tunes_stream_chunks(monkeypatch):
    import os
    _feed(runtime.MIN_SAMPLES - 1)
    assert "ADR_TTS_STREAM_HEAD" not in os.environ        # 样本不足不动
    _feed(1)
    assert os.environ["ADR_TTS_STREAM_HEAD"] == "10"
    assert os.environ["ADR_TTS_SPLIT_MAX"] == "30"
    rep = runtime.report()
    assert rep["events"][-1]["kind"] == "tune"
    assert rep["known_good"] == {"ADR_TTS_STREAM_HEAD": None, "ADR_TTS_SPLIT_MAX": None}
    assert rep["window_size"] == 0                        # 微调后窗口重计
    for k in runtime.TUNE_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_fast_window_no_tune():
    import os
    _feed(10, rtf=0.4)
    assert "ADR_TTS_STREAM_HEAD" not in os.environ
    assert runtime.report()["events"] == []


def test_explicit_env_not_touched(monkeypatch):
    import os
    monkeypatch.setenv("ADR_TTS_STREAM_HEAD", "20")
    _feed(runtime.MIN_SAMPLES)
    assert os.environ["ADR_TTS_STREAM_HEAD"] == "20"      # 显式 env 优先
    assert os.environ["ADR_TTS_SPLIT_MAX"] == "30"
    monkeypatch.delenv("ADR_TTS_SPLIT_MAX", raising=False)


def test_policy_value_overridable_and_restored(monkeypatch):
    """policy 写入的值不算显式 env: runtime 可再收, 回退恢复到 policy 值。"""
    import os
    monkeypatch.setenv("ADR_TTS_STREAM_HEAD", "12")
    policy._applied["ADR_TTS_STREAM_HEAD"] = "12"
    _feed(runtime.MIN_SAMPLES)
    assert os.environ["ADR_TTS_STREAM_HEAD"] == "10"
    _feed(runtime.MIN_SAMPLES, ok=False)                  # 微调后持续失败
    assert os.environ["ADR_TTS_STREAM_HEAD"] == "12"      # 回退已知良好
    assert "ADR_TTS_SPLIT_MAX" not in os.environ
    rep = runtime.report()
    assert rep["frozen"] and rep["events"][-1]["kind"] == "fallback"
    assert rep["runtime_applied"] == {}


def test_frozen_after_fallback_no_retune():
    import os
    _feed(runtime.MIN_SAMPLES)
    _feed(runtime.MIN_SAMPLES, ok=False)
    _feed(10)                                             # 回退后再慢也不微调
    assert "ADR_TTS_STREAM_HEAD" not in os.environ
    assert [e["kind"] for e in runtime.report()["events"]] == ["tune", "fallback"]


def test_failures_without_tune_do_not_fallback():
    _feed(10, ok=False)
    assert runtime.report()["events"] == []               # 无微调可回退


def test_autotune_off_monitor_only(monkeypatch):
    import os
    monkeypatch.setenv("ADR_RUNTIME_AUTOTUNE", "0")
    _feed(10)
    assert "ADR_TTS_STREAM_HEAD" not in os.environ
    rep = runtime.report()
    assert rep["autotune"] is False and rep["window_size"] == 10


def test_snapshot_reports_runtime_source(monkeypatch):
    _feed(runtime.MIN_SAMPLES)
    eff = policy.export_snapshot()["effective"]
    assert eff["ADR_TTS_STREAM_HEAD"] == {"value": "10", "source": "runtime"}
    for k in runtime.TUNE_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_record_never_raises(monkeypatch):
    monkeypatch.setattr(runtime, "_evaluate", lambda: 1 / 0)
    runtime.record(ok=True, elapsed_s=1.0, audio_s=1.0)   # 吞异常
    assert runtime.report()["window_size"] == 1
