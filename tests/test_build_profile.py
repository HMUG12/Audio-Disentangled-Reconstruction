"""v1.3.x 建档后触发采样: 首启档案库为空 → 建档成功当场补跑 profile + policy。"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf


class _SyncThread:
    """同步替身: start() 直接在当前线程跑 target, 便于断言结果。"""

    def __init__(self, target=None, args=(), name=None, daemon=None):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


FAKE = {"profile_id": "build_mock", "timestamp": 1.0,
        "device_info": {"type": "cpu"},
        "profiles": [{"name": "default", "params": {}, "rtf": 1.5,
                      "mem_mb": 0, "notes": "mock"}]}


@pytest.fixture
def env(monkeypatch, tmp_path, caplog):
    import adr.models.voice_library as vl
    from adr.models import gsv_engine

    caplog.set_level(logging.INFO, logger="adr.models.gsv_engine")
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path))
    voices = tmp_path / "voices"
    voices.mkdir()
    monkeypatch.setattr(vl, "VOICES_DIR", voices)
    monkeypatch.setattr(gsv_engine, "_PROFILE_IDLE_DELAY", 0.0)
    monkeypatch.setattr(gsv_engine, "_PROFILE_BUILD_DELAY", 0.0)
    monkeypatch.setattr(gsv_engine, "_PROFILE_STATE", "idle")
    idle = MagicMock()
    idle.queue_depth.return_value = 0
    idle.synth_busy.return_value = 0
    monkeypatch.setattr(gsv_engine, "get_gsv_engine", lambda: idle)
    monkeypatch.setattr(gsv_engine.threading, "Thread", _SyncThread)
    return tmp_path


def _make_voice(tmp_path, name="v1"):
    from adr.models.voice_library import save_voice

    ref = tmp_path / "ref.wav"
    t = np.arange(32000 * 4) / 32000
    sf.write(str(ref), (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 32000)
    save_voice(name, str(ref), "test text")


def test_empty_library_then_build_samples(env, caplog):
    """首启档案库为空 → 退回 idle; 建档后 try_schedule_profile 当场采样落盘。"""
    from adr.core import policy, profiler
    from adr.models import gsv_engine

    gsv_engine._schedule_idle_profile()
    assert "档案库为空" in caplog.text
    assert gsv_engine._PROFILE_STATE == "idle"      # 未锁死, 可再调度

    _make_voice(env)
    with patch.object(profiler, "run_profile", return_value=FAKE), \
         patch.object(policy, "apply") as mock_apply:
        gsv_engine.try_schedule_profile()
    assert "采样完成, policy 已生成并应用" in caplog.text
    mock_apply.assert_called_once()
    assert (env / "profile_build_mock_cpu.json").exists()
    assert (env / "policy.json").exists()
    assert gsv_engine._PROFILE_STATE == "done"      # 测完不再调度


def test_existing_profile_marks_done(env, caplog):
    """已有 profile → 跳过且置 done, 后续建档不再起线程。"""
    from adr.core import profiler
    from adr.models import gsv_engine

    profiler.save_profile({"profile_id": profiler._machine_id(), "timestamp": 1.0,
                           "device_info": {"type": "cpu"}, "profiles": []})
    with patch.object(profiler, "current_device_type", return_value="cpu"), \
         patch("adr.core.policy.apply"):
        gsv_engine._schedule_idle_profile()
    assert "本机已有 profile" in caplog.text
    assert gsv_engine._PROFILE_STATE == "done"
    assert gsv_engine._start_profile(0.0) is False


def test_running_blocks_duplicate(env):
    """采样中 (running) 再建档不重复启动。"""
    from adr.models import gsv_engine

    gsv_engine._PROFILE_STATE = "running"
    assert gsv_engine._start_profile(0.0) is False


def test_sample_failure_does_not_retry(env, caplog):
    """采样异常 → 置 done, 本进程不反复重试 (避免每次建档都卡一遍)。"""
    from adr.core import profiler
    from adr.models import gsv_engine

    _make_voice(env)
    with patch.object(profiler, "run_profile", side_effect=RuntimeError("boom")):
        gsv_engine.try_schedule_profile()
    assert "后台采样失败" in caplog.text
    assert gsv_engine._PROFILE_STATE == "done"


def test_try_schedule_swallows_errors(env, monkeypatch):
    """调度本身出错也不抛给建档调用方。"""
    from adr.models import gsv_engine

    monkeypatch.setattr(gsv_engine, "_start_profile",
                        MagicMock(side_effect=RuntimeError("x")))
    gsv_engine.try_schedule_profile()               # 不抛即通过


def test_console_worker_triggers_on_success_only(monkeypatch):
    """控制台建档 worker: 成功后调度采样, 失败不调度。"""
    from adr.models import gsv_engine, voice_library
    from adr.server import console

    hook = MagicMock()
    monkeypatch.setattr(gsv_engine, "try_schedule_profile", hook)
    monkeypatch.setattr(voice_library, "save_voice_auto",
                        lambda *a, **k: ("/tmp/v", {"mode": "whole"}))
    console._profile_worker("n", "a.wav", "t", "")
    hook.assert_called_once()

    hook.reset_mock()
    monkeypatch.setattr(voice_library, "save_voice_auto",
                        MagicMock(side_effect=ValueError("bad")))
    console._profile_worker("n", "a.wav", "t", "")
    hook.assert_not_called()
