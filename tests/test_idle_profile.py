"""批次63 v1.3.x 壳首启接线: 预热 ready 后空闲时后台跑 profile + policy。"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """临时 data_dir + 隔离档案库 (避免全局档案干扰)。"""
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path))
    # 隔离档案库: 在 tmp_path 下创建空档案库目录
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    # 直接 patch voice_library.VOICES_DIR
    import adr.models.voice_library
    from adr.models import gsv_engine
    monkeypatch.setattr(adr.models.voice_library, "VOICES_DIR", voices_dir)
    # 测试不等空闲期; 引擎换成空闲桩, 不触发真实加载
    monkeypatch.setattr(gsv_engine, "_PROFILE_IDLE_DELAY", 0.0)
    idle = MagicMock()
    idle.queue_depth.return_value = 0
    idle.synth_busy.return_value = 0
    monkeypatch.setattr(gsv_engine, "get_gsv_engine", lambda: idle)
    yield tmp_path


@pytest.fixture(autouse=True)
def _info_logs(caplog):
    caplog.set_level(logging.INFO, logger="adr.models.gsv_engine")


def test_idle_profile_scheduled_once(isolated_env):
    """空闲调度每进程至多触发一次 (重复调 _schedule_idle_profile 不重复启线程)。"""
    from adr.models import gsv_engine

    # 重置状态 (测试间隔离)
    gsv_engine._PROFILE_SCHEDULED = False

    with patch("adr.models.gsv_engine.threading.Thread") as mock_thread:
        gsv_engine._schedule_idle_profile()
        assert mock_thread.call_count == 1
        gsv_engine._schedule_idle_profile()
        assert mock_thread.call_count == 1  # 第二次调用不应再启线程


def test_idle_profile_skip_when_profile_exists(isolated_env, caplog):
    """本机已有 profile 时跳过后台采样。"""
    from adr.core import profiler
    from adr.models import gsv_engine

    # 伪造一个已有 profile
    fake_prof = {
        "profile_id": profiler._machine_id(),
        "timestamp": 1234567890.0,
        "device_info": {"type": "cpu"},
        "profiles": [],
    }
    profiler.save_profile(fake_prof)

    gsv_engine._idle_profile_worker()
    assert "本机已有 profile, 跳过后台采样" in caplog.text


def test_idle_profile_skip_when_no_voices(isolated_env, caplog):
    """档案库为空时跳过后台采样 (建档后下次启动再跑)。"""
    from adr.models import gsv_engine
    from adr.models.voice_library import list_voices

    assert not list_voices()  # 确认档案库为空
    gsv_engine._idle_profile_worker()
    assert "档案库为空, 跳过后台采样" in caplog.text


def test_idle_profile_samples_and_applies(isolated_env, tmp_path, caplog):
    """档案库非空 + 无 profile → 采样完成后生成 policy 并 apply。"""
    from adr.core import policy, profiler
    from adr.models import gsv_engine
    from adr.models.voice_library import save_voice

    # 创建一个音色档案 (ref_audio 用临时音频文件占位)
    import numpy as np
    import soundfile as sf

    ref_audio = tmp_path / "ref.wav"
    sr = 32000
    t = np.arange(sr * 4) / sr
    sf.write(str(ref_audio), (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    save_voice("test_voice", str(ref_audio), "test text")

    # mock profiler.run_profile 返回假数据 (避免真的跑 20s)
    fake_result = {
        "profile_id": "test_mock",
        "timestamp": 1234567890.0,
        "device_info": {"type": "cpu"},
        "profiles": [{"name": "default", "params": {}, "rtf": 1.5, "mem_mb": 0, "notes": "mock"}],
    }
    with patch.object(profiler, "run_profile", return_value=fake_result):
        with patch.object(policy, "apply") as mock_apply:
            gsv_engine._idle_profile_worker()

    # 日志验证
    assert "服务端空闲, 开始后台采样" in caplog.text
    assert "采样完成, policy 已生成并应用" in caplog.text
    # policy.apply 应被调用
    mock_apply.assert_called_once()
    # policy.json 应已落盘
    policy_path = tmp_path / "policy.json"
    assert policy_path.exists()
