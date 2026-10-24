"""v1.3.x 首子批: 调度层 Policy conservative-v1 (纯逻辑, 不依赖真模型)。"""
from __future__ import annotations

import json
import os

import pytest

from adr.core import policy, settings

_KEYS = (*policy.AUTO_KEYS, *policy.GATED_KEYS)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path))
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(policy, "_applied", {})


def _prof(dev: dict, rtf: float) -> dict:
    return {"profile_id": "abc", "device_info": dev,
            "profiles": [{"name": "default", "rtf": rtf, "mem_mb": 0}]}


CPU_SLOW = _prof({"type": "cpu"}, 1.7)                     # E5-2667 v3 锚点
GPU_FAST = _prof({"type": "cuda", "total_mem_gb": 16.0}, 0.3)
GPU_SMALL = _prof({"type": "cuda", "total_mem_gb": 4.0}, 0.6)


def test_cpu_slow_auto_overrides():
    p = policy.infer(CPU_SLOW, physical_cores=8)
    assert p["policy_version"] == policy.POLICY_VERSION
    assert p["rules"] == "conservative-v1"
    assert p["overrides"] == {"ADR_TTS_STREAM_HEAD": "10",
                              "ADR_TTS_SPLIT_MAX": "30",
                              "ADR_TORCH_THREADS": "7"}
    # CPU 不推荐 GPU 加速开关
    assert not set(p["recommendations"]) & set(policy.GATED_KEYS)


def test_cpu_few_cores_no_reserve():
    p = policy.infer(_prof({"type": "cpu"}, 0.8), physical_cores=2)
    assert "ADR_TORCH_THREADS" not in p["overrides"]
    assert "ADR_TTS_STREAM_HEAD" not in p["overrides"]   # 快机不收短


def test_gpu_gated_only_recommended():
    p = policy.infer(GPU_FAST)
    assert p["overrides"] == {}                           # 音质敏感项绝不进 overrides
    recs = p["recommendations"]
    assert recs["ADR_T2S_STATIC_KV"]["value"] == "1"
    assert recs["ADR_T2S_CUDAGRAPH"]["value"] == "1"
    assert recs["int8"]["value"] == "off"
    small = policy.infer(GPU_SMALL)["recommendations"]
    assert "ADR_T2S_CUDAGRAPH" not in small and "ADR_T2S_STATIC_KV" in small


def test_thresholds_configurable():
    p = policy.infer(GPU_SMALL, thresholds={"slow_rtf": 0.5, "graph_min_mem_gb": 2})
    assert p["overrides"]["ADR_TTS_STREAM_HEAD"] == "10"
    assert "ADR_T2S_CUDAGRAPH" in p["recommendations"]
    assert p["thresholds"]["slow_rtf"] == 0.5


def test_generate_keeps_user_thresholds(tmp_path):
    path = tmp_path / "policy.json"
    first = policy.generate(GPU_SMALL, path)
    first["thresholds"]["slow_rtf"] = 0.5               # 用户手改阈值
    path.write_text(json.dumps(first), encoding="utf-8")
    again = policy.generate(GPU_SMALL, path)
    assert again["thresholds"]["slow_rtf"] == 0.5
    assert again["overrides"]["ADR_TTS_STREAM_HEAD"] == "10"


def test_apply_whitelist_and_env_priority(monkeypatch):
    monkeypatch.setenv("ADR_TORCH_THREADS", "3")          # 显式 env
    pol = {"policy_version": 1, "overrides": {
        "ADR_TORCH_THREADS": "7", "ADR_TTS_STREAM_HEAD": "10",
        "ADR_T2S_STATIC_KV": "1"}}                       # 手写违规项
    done = policy.apply(pol)
    assert done == {"ADR_TTS_STREAM_HEAD": "10"}
    assert os.environ["ADR_TORCH_THREADS"] == "3"
    assert "ADR_T2S_STATIC_KV" not in os.environ
    assert settings.stream_head_chars() == 10
    # 重新应用 (policy 改值) 可覆盖自己写过的键
    policy.apply({"policy_version": 1, "overrides": {"ADR_TTS_STREAM_HEAD": "12"}})
    assert settings.stream_head_chars() == 12


def test_apply_rejects_newer_version():
    assert policy.apply({"policy_version": 99,
                         "overrides": {"ADR_TTS_SPLIT_MAX": "20"}}) == {}
    assert "ADR_TTS_SPLIT_MAX" not in os.environ


def test_load_missing_or_corrupt(tmp_path):
    assert policy.load() is None and policy.apply() == {}
    policy.policy_path().write_text("{oops", encoding="utf-8")
    assert policy.load() is None


def test_snapshot_sources(monkeypatch):
    policy.save(policy.infer(CPU_SLOW, physical_cores=8))
    policy.apply()
    monkeypatch.setenv("ADR_T2S_STATIC_KV", "1")
    eff = policy.export_snapshot()["effective"]
    assert eff["ADR_TTS_STREAM_HEAD"] == {"value": "10", "source": "policy"}
    assert eff["ADR_T2S_STATIC_KV"] == {"value": "1", "source": "env"}
    assert eff["ADR_T2S_CUDAGRAPH"]["source"] == "default"


def test_settings_chunk_defaults(monkeypatch):
    assert settings.stream_head_chars() == 14 and settings.split_max_chars() == 40
    monkeypatch.setenv("ADR_TTS_SPLIT_MAX", "0")
    assert settings.split_max_chars() == 40              # 非正回退默认
    monkeypatch.setenv("ADR_TTS_SPLIT_MAX", "x")
    assert settings.split_max_chars() == 40
