"""批次63 第四子批: 调度层 Profiler (假引擎, 不依赖真模型/GPU)。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from adr.core import profiler


class _FakeEngine:
    """每次合成返回 1 秒音频 (sr=32000), 记录调用参数。"""

    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    def synthesize(self, text, ref_audio, **kw):
        self.calls.append({"text": text, "ref": ref_audio, **kw})
        if self.fail and len(self.calls) > 1:   # warmup 过, 测量失败
            raise RuntimeError("boom")
        return np.zeros(32000, dtype=np.float32), 32000


@pytest.fixture
def fake_engine(monkeypatch):
    import torch
    from adr.models import gsv_engine

    eng = _FakeEngine()
    monkeypatch.delenv("ADR_T2S_CUDAGRAPH", raising=False)
    monkeypatch.delenv("ADR_T2S_STATIC_KV", raising=False)
    monkeypatch.setattr(gsv_engine, "get_gsv_engine", lambda: eng)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return eng


def test_run_profile_schema(fake_engine):
    data = profiler.run_profile(text="你好", ref_audio="ref.wav", n_runs=3)
    assert set(data) == {"profile_id", "timestamp", "device_info", "profiles"}
    assert len(data["profile_id"]) == 16
    assert data["device_info"] == {"type": "cpu"}
    (p,) = data["profiles"]
    assert p["name"] == "default"
    assert p["params"] == {"top_k": 15, "cudagraph": 0, "static_kv": 0}
    assert 0 <= p["rtf"] < 1 and p["mem_mb"] == 0   # 假引擎瞬时返回 1s 音频
    # warmup 1 次 + 测量 3 次, 均透传 top_k
    assert len(fake_engine.calls) == 4
    assert all(c["top_k"] == 15 and c["ref"] == "ref.wav" and c["seed"] == 1234
               for c in fake_engine.calls)


def test_voice_profile_conditions_passed(fake_engine, monkeypatch):
    """默认档案: prompt_text / 微调权重透传到合成 (测的是真实合成路径)。"""
    from adr.models import voice_library

    monkeypatch.setattr(voice_library, "list_voices", lambda: ["v1"])
    monkeypatch.setattr(voice_library, "load_voice", lambda n: {
        "ref_audio": "v1.wav", "prompt_text": "参考", "t2s_weights": "a.ckpt",
        "vits_weights": ""})
    profiler.run_profile(text="你好", n_runs=1)
    c = fake_engine.calls[-1]
    assert c["ref"] == "v1.wav" and c["prompt_text"] == "参考"
    assert c["t2s_weights"] == "a.ckpt" and "vits_weights" not in c


def test_measure_failure_returns_zero(monkeypatch):
    import torch
    from adr.models import gsv_engine

    eng = _FakeEngine(fail=True)
    monkeypatch.setattr(gsv_engine, "get_gsv_engine", lambda: eng)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert profiler._measure_rtf_mem("t", "r.wav", n_runs=3) == (0.0, 0)


def test_no_voice_raises(fake_engine, monkeypatch):
    from adr.models import voice_library

    monkeypatch.setattr(voice_library, "list_voices", lambda: [])
    with pytest.raises(ValueError):
        profiler.run_profile(ref_audio=None)


def test_save_load_roundtrip(fake_engine, tmp_path, monkeypatch):
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path))
    data = profiler.run_profile(text="你好", ref_audio="ref.wav", n_runs=1)
    path = profiler.save_profile(data)
    assert path == tmp_path / f"profile_{data['profile_id']}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == data
    assert profiler.load_profile() == data


def test_load_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path))
    assert profiler.load_profile() is None


def test_machine_id_stable():
    assert profiler._machine_id() == profiler._machine_id()


def test_module_sched_stats_none_without_engine(monkeypatch):
    from adr.models import gsv_engine

    monkeypatch.setattr(gsv_engine, "_ENGINE", None)
    assert gsv_engine.sched_stats() is None
