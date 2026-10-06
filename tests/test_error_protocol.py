"""批次35: 错误协议测试 — 类型化异常错误码 + 四面统一映射。

覆盖:
- core.exceptions: 全家族 code 类属性 + error_code() (非框架异常 → internal_error)
- v2 HTTP: 错误响应 {"message", "code"} (code 增量字段, 既有 message 逐字保留)
- native HTTP: /profiles/{name}/ref 404 带 code
- v3 WS: init 类型化上报 (profile_invalid / profile_not_found) + error 帧 code
- webui 面: _err dict 为 UI 形态, 无 wire 协议, 不加 code (设计备案, 无需测试)
"""
import json

import pytest
from fastapi.testclient import TestClient

from adr.core.exceptions import (
    ADRException,
    ConfigError,
    DataPipelineError,
    DependencyMissingError,
    DeviceError,
    InferenceError,
    ModelNotFoundError,
    ProfileInvalidError,
    ProfileNotFoundError,
    SynthesisError,
    SynthesisParamsError,
    TrainingError,
    VocabError,
    error_code,
)
from adr.server import create_app


# ─── core.exceptions 错误码 ───

@pytest.mark.parametrize("exc, code", [
    (ConfigError, "config_error"),
    (ModelNotFoundError, "model_not_found"),
    (VocabError, "vocab_error"),
    (DataPipelineError, "data_pipeline_error"),
    (TrainingError, "training_error"),
    (InferenceError, "inference_error"),
    (DeviceError, "device_error"),
    (DependencyMissingError, "dependency_missing"),
    (SynthesisError, "synthesis_error"),
    (ProfileInvalidError, "profile_invalid"),
    (ProfileNotFoundError, "profile_not_found"),
    (SynthesisParamsError, "invalid_params"),
])
def test_exception_codes(exc, code):
    assert issubclass(exc, ADRException)
    assert exc("x").code == code


def test_error_code_unknown_exception():
    """非框架异常 → internal_error; 框架异常 → 自身 code。"""
    assert error_code(RuntimeError("boom")) == "internal_error"
    assert error_code(Exception()) == "internal_error"
    assert error_code(ProfileNotFoundError("nope")) == "profile_not_found"


# ─── fixtures (模式对齐 test_server_api) ───

class _FakeEngine:
    """恒抛错的占位引擎: 错误协议只测失败面。"""

    sr = 32000

    def synthesize(self, *args, **kwargs):
        raise RuntimeError("boom")

    def synthesize_stream(self, *args, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover


@pytest.fixture
def engine():
    return _FakeEngine()


@pytest.fixture
def client(engine, monkeypatch):
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    return TestClient(create_app(engine=engine))


@pytest.fixture
def voice_dir(tmp_path, monkeypatch):
    """假档案库: 仅 demo 档案 (ref.wav + meta.json)。"""
    from adr.models import voice_library
    vd = tmp_path / "voices"
    (vd / "demo").mkdir(parents=True)
    (vd / "demo" / "ref.wav").write_bytes(b"RIFF0000WAVEfmt ")
    (vd / "demo" / "meta.json").write_text(json.dumps({
        "name": "demo", "prompt_text": "参考文本",
        "t2s_weights": None, "vits_weights": None,
        "created_at": "2026-01-01 00:00:00"}), encoding="utf-8")
    monkeypatch.setattr(voice_library, "VOICES_DIR", vd)
    return vd


# ─── v2 HTTP 面 ───

def test_v2_profile_invalid_code(client, voice_dir):
    r = client.post("/api/v2/tts", json={"text": "你好", "ref_audio_path": "a.wav",
                                         "profile": "../evil"})
    assert r.status_code == 400
    assert r.json() == {"message": "invalid profile name: ../evil",
                        "code": "profile_invalid"}


def test_v2_profile_not_found_code(client, voice_dir):
    r = client.post("/api/v2/tts", json={"text": "你好", "ref_audio_path": "a.wav",
                                         "profile": "ghost"})
    assert r.status_code == 400
    assert r.json() == {"message": "unknown profile: ghost",
                        "code": "profile_not_found"}


def test_v2_missing_text_code(client):
    r = client.post("/api/v2/tts", json={"ref_audio_path": "a.wav"})
    assert r.status_code == 400
    assert r.json() == {"message": "text is required", "code": "invalid_params"}


def test_v2_media_type_code(client):
    r = client.post("/api/v2/tts", json={"text": "你好", "ref_audio_path": "a.wav",
                                         "media_type": "flac"})
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "invalid_params"
    assert body["message"].startswith("unsupported media_type: flac")


def test_v2_streaming_mode_code(client):
    r = client.post("/api/v2/tts", json={"text": "你好", "ref_audio_path": "a.wav",
                                         "streaming_mode": 9})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_params"


def test_v2_engine_fail_code(client, voice_dir):
    """引擎抛非框架异常 → 消息逐字保留 + internal_error。"""
    r = client.post("/api/v2/tts", json={"text": "你好", "ref_audio_path": "a.wav"})
    assert r.status_code == 400
    body = r.json()
    assert body["message"] == "tts failed"
    assert body["Exception"] == "boom"
    assert body["code"] == "internal_error"


@pytest.mark.parametrize("ep, field", [
    ("set_gpt_weights", "gpt weight path is required"),
    ("set_sovits_weights", "sovits weight path is required"),
])
def test_v2_weights_required_code(client, ep, field):
    r = client.get(f"/api/v2/{ep}")
    assert r.status_code == 400
    assert r.json() == {"message": field, "code": "invalid_params"}


# ─── native HTTP 面 ───

def test_native_profile_ref_code(client):
    r = client.get("/api/adr/v1/profiles/ghost/ref")
    assert r.status_code == 404
    assert r.json() == {"message": "profile not found: ghost",
                        "code": "profile_not_found"}


# ─── v3 WS 面 ───

def _connect(client):
    return client.websocket_connect("/api/v3/tts/stream-input")


def test_v3_init_profile_invalid(client, voice_dir):
    """非法名 (../ 穿越) → 精确 message + profile_invalid (原为误导性 not found)。"""
    with _connect(client) as ws:
        ws.send_json({"cmd": "init", "voice_id": "../evil"})
        msg = ws.receive_json()
        assert msg == {"type": "error", "message": "invalid profile name: ../evil",
                       "code": "profile_invalid"}


def test_v3_init_profile_not_found(client, voice_dir):
    """不存在 → not found 消息逐字保留 (wire 兼容) + profile_not_found。"""
    with _connect(client) as ws:
        ws.send_json({"cmd": "init", "voice_id": "ghost"})
        msg = ws.receive_json()
        assert msg == {"type": "error", "message": "voice_id 'ghost' not found",
                       "code": "profile_not_found"}


def test_v3_init_default_no_profiles(client, tmp_path, monkeypatch):
    """_default 且无任何档案 → 静默 None 路径, not found 消息保留。"""
    from adr.models import voice_library
    vd = tmp_path / "voices"
    vd.mkdir()
    monkeypatch.setattr(voice_library, "VOICES_DIR", vd)
    with _connect(client) as ws:
        ws.send_json({"cmd": "init", "voice_id": "_default"})
        msg = ws.receive_json()
        assert msg == {"type": "error", "message": "voice_id '_default' not found",
                       "code": "profile_not_found"}


def test_v3_init_ready(client, voice_dir):
    with _connect(client) as ws:
        ws.send_json({"cmd": "init", "voice_id": "demo"})
        assert ws.receive_json() == {"type": "ready", "voice_id": "demo"}


def test_v3_synth_error_code(client, voice_dir):
    """合成中非框架异常 → error 帧 code=internal_error。"""
    with _connect(client) as ws:
        ws.send_json({"cmd": "init", "voice_id": "demo"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"cmd": "text", "data": "你好"})
        err = None
        for _ in range(5):
            msg = ws.receive_json()
            if msg["type"] == "error":
                err = msg
                break
        assert err is not None
        assert err["message"] == "boom"
        assert err["code"] == "internal_error"
