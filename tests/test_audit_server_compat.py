"""批次66: server_compat 审计修复回归 (#20 #23 #24 #25 #26 #27 #37 #38 #39 #40 #41 #42)。

FakeEngine 不加载真模型; 档案库 / 缓存目录全部 tmp_path 隔离。
"""
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adr.server import create_app, tts_cache


class FakeEngine:
    def __init__(self):
        self.fail = None          # 待抛异常 (None = 正常)
        self.waiting = 0          # 模拟排队数
        self.synth_calls = 0

    def queue_depth(self):
        return self.waiting

    def synthesize(self, text, ref_audio, **kw):
        self.synth_calls += 1
        if self.fail:
            raise self.fail
        return np.full(320, 0.1, np.float32), 32000

    def synthesize_stream(self, text, ref_audio, **kw):
        yield np.full(160, 0.1, np.float32), 32000
        if self.fail:
            raise self.fail
        yield np.full(160, 0.1, np.float32), 32000

    def warmup(self, vits_weights=None, t2s_weights=None):
        if self.fail:
            raise self.fail


@pytest.fixture(autouse=True)
def _iso(monkeypatch, tmp_path):
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    monkeypatch.setenv("ADR_TTS_CACHE", "0")
    monkeypatch.setenv("ADR_TTS_CACHE_DIR", str(tmp_path / "tts_cache"))
    tts_cache.clear()
    yield
    tts_cache.clear()


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine=engine))


@pytest.fixture
def voice_dir(tmp_path, monkeypatch):
    from adr.models import voice_library
    vd = tmp_path / "voices"
    (vd / "demo").mkdir(parents=True)
    (vd / "demo" / "ref.wav").write_bytes(b"RIFF0000WAVEfmt ")
    (vd / "demo" / "meta.json").write_text(json.dumps({
        "name": "demo", "prompt_text": "参考", "t2s_weights": None,
        "vits_weights": None}), encoding="utf-8")
    (vd / "bad").mkdir()
    (vd / "bad" / "meta.json").write_text("{not json", encoding="utf-8")
    (vd / "bad" / "ref.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(voice_library, "VOICES_DIR", vd)
    return vd


def _oom():
    import torch
    return torch.cuda.OutOfMemoryError("CUDA out of memory")


BODY = {"text": "hi", "ref_audio_path": "a.wav"}


# ─── #23 / #25 服务端日志 (wire 不变) ───

def test_v2_failure_logs_traceback_keeps_400(client, engine, caplog):
    engine.fail = RuntimeError("boom")
    with caplog.at_level(logging.ERROR, logger="adr.server.v2_compat"):
        r = client.post("/api/v2/tts", json=BODY)
    assert r.status_code == 400 and r.json()["code"] == "internal_error"
    assert any(rec.exc_info for rec in caplog.records)


def test_v2_stream_oom_logged(client, engine, caplog):
    engine.fail = _oom()
    with caplog.at_level(logging.WARNING, logger="adr.server.v2_compat"):
        r = client.post("/api/v2/tts", json=dict(BODY, streaming_mode=1))
    assert r.status_code == 200          # GSV parity: 截断不改状态码
    assert any("OOM" in rec.getMessage() for rec in caplog.records
               if rec.name == "adr.server.v2_compat")


# ─── #24 热换权重 OOM ───

@pytest.mark.parametrize("ep", ["set_gpt_weights", "set_sovits_weights"])
def test_set_weights_oom_503(client, engine, tmp_path, ep, monkeypatch):
    from adr.server import v2_compat
    monkeypatch.setattr(v2_compat, "_validate_weights_file", lambda p: None)
    engine.fail = _oom()
    r = client.get(f"/api/v2/{ep}", params={"weights_path": str(tmp_path / "w.pth")})
    assert r.status_code == 503 and r.json()["code"] == "cuda_oom"


# ─── #40 文本长度 / 数值边界 ───

def test_text_too_long_413(client, engine, monkeypatch):
    monkeypatch.setenv("ADR_TTS_MAX_TEXT_CHARS", "10")
    r = client.post("/api/v2/tts", json=dict(BODY, text="a" * 11))
    assert r.status_code == 413 and r.json()["code"] == "payload_too_large"
    assert engine.synth_calls == 0
    assert client.post("/api/v2/tts", json=dict(BODY, text="a" * 10)).status_code == 200


@pytest.mark.parametrize("field, val", [
    ("speed_factor", 0), ("speed_factor", -1), ("speed_factor", 100),
    ("top_k", 0), ("top_p", 1.5), ("temperature", -0.1),
])
def test_numeric_bounds_400(client, engine, field, val):
    r = client.post("/api/v2/tts", json=dict(BODY, **{field: val}))
    assert r.status_code == 400 and r.json()["code"] == "invalid_params"
    assert engine.synth_calls == 0


def test_openai_speed_bound(client, voice_dir):
    r = client.post("/v1/audio/speech", json={"input": "hi", "model": "demo",
                                              "speed": 0, "stream": False})
    assert r.status_code == 400
    assert "speed_factor out of range" in r.json()["error"]["message"]


def test_normal_params_pass(client):
    r = client.post("/api/v2/tts", json=dict(
        BODY, speed_factor=1.5, top_k=15, top_p=1, temperature=1))
    assert r.status_code == 200


# ─── #42 ref_audio_path 普通文件 + 大小上限 ───

def test_ref_audio_dir_rejected(client, engine, tmp_path):
    r = client.post("/api/v2/tts", json=dict(BODY, ref_audio_path=str(tmp_path)))
    assert r.status_code == 400 and r.json()["code"] == "invalid_params"
    assert engine.synth_calls == 0


def test_ref_audio_size_cap(client, engine, tmp_path, monkeypatch):
    monkeypatch.setenv("ADR_TTS_REF_MAX_MB", "1")
    big = tmp_path / "big.wav"
    big.write_bytes(b"\0" * (1024 * 1024 + 1))
    r = client.post("/api/v2/tts", json=dict(BODY, ref_audio_path=str(big)))
    assert r.status_code == 400 and r.json()["code"] == "invalid_params"
    small = tmp_path / "s.wav"
    small.write_bytes(b"RIFF")
    assert client.post("/api/v2/tts", json=dict(
        BODY, ref_audio_path=str(small))).status_code == 200


# ─── #41 排队上限 → 503 busy ───

@pytest.mark.parametrize("mode", [0, 1])
def test_queue_cap_busy(client, engine, monkeypatch, mode):
    monkeypatch.setenv("ADR_TTS_MAX_QUEUE", "2")
    engine.waiting = 2
    r = client.post("/api/v2/tts", json=dict(BODY, streaming_mode=mode))
    assert r.status_code == 503 and r.json()["code"] == "busy"
    engine.waiting = 1
    assert client.post("/api/v2/tts", json=dict(BODY, streaming_mode=mode)).status_code == 200


# ─── #26 / #39 / #41 v3 WS ───

def _ws(client):
    return client.websocket_connect("/api/v3/tts/stream-input")


@pytest.mark.parametrize("frame", ["not json", "[1, 2]", "42"])
def test_v3_bad_frame_keeps_session(client, voice_dir, frame):
    with _ws(client) as ws:
        ws.send_text(frame)
        msg = ws.receive_json()
        assert msg["type"] == "error" and msg["code"] == "invalid_params"
        ws.send_json({"cmd": "init", "voice_id": "demo"})   # 会话仍可用
        assert ws.receive_json() == {"type": "ready", "voice_id": "demo"}


def test_v3_append_buffer_cap(client, voice_dir, monkeypatch):
    monkeypatch.setenv("ADR_TTS_MAX_TEXT_CHARS", "10")
    with _ws(client) as ws:
        ws.send_json({"cmd": "init", "voice_id": "demo"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"cmd": "append", "data": "a" * 6})
        ws.send_json({"cmd": "append", "data": "b" * 6})   # 累计 12 > 10
        msg = ws.receive_json()
        assert msg["code"] == "payload_too_large"
        ws.send_json({"cmd": "text", "data": "c" * 11})
        assert ws.receive_json()["code"] == "payload_too_large"


def test_v3_busy_when_queue_full(client, engine, voice_dir, monkeypatch):
    monkeypatch.setenv("ADR_TTS_MAX_QUEUE", "1")
    engine.waiting = 1
    with _ws(client) as ws:
        ws.send_json({"cmd": "init", "voice_id": "demo"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"cmd": "text", "data": "你好"})
        assert ws.receive_json() == {"type": "error", "code": "busy",
                                     "message": "busy: synthesis queue full"}


def test_v3_voices_logs_bad_profile(client, voice_dir, caplog):
    with caplog.at_level(logging.WARNING, logger="adr.server.v3_compat"):
        ids = [v["id"] for v in client.get("/api/v3/voices").json()]
    assert "bad" in ids                     # 行为不变: 坏档案仍列出
    assert any("bad" in rec.getMessage() for rec in caplog.records)


# ─── #27 / #37 / #20 / #38 native ───

def test_native_profiles_logs_bad_profile(client, voice_dir, caplog):
    with caplog.at_level(logging.WARNING, logger="adr.server.native"):
        names = [p["name"] for p in client.get("/api/adr/v1/profiles").json()["profiles"]]
    assert names == ["demo"]
    assert any("bad" in rec.getMessage() for rec in caplog.records)


def test_health_no_absolute_paths(client, tmp_path):
    w = str(tmp_path / "secret_user" / "GPT_weights_v2" / "x.ckpt")
    client.app.state.t2s_weights = w
    client.app.state.vits_weights = w
    body = client.get("/api/adr/v1/health").json()
    assert body["weights"] == {"t2s": "x.ckpt", "vits": "x.ckpt"}
    assert "secret_user" not in json.dumps(body)
    for k in ("engine_ready", "engine_stage", "engine_stage_text",
              "queue_depth", "synth_busy"):
        assert k in body


def test_export_missing_profile_404(client, voice_dir):
    r = client.get("/api/adr/v1/profiles/ghost/export")
    assert r.status_code == 404 and r.json()["code"] == "export_failed"


def test_export_other_valueerror_400(client, voice_dir, monkeypatch):
    from adr.models import voice_package

    def _raise(name, out):
        raise ValueError("权重文件不存在: x.pth")   # 文案含"不存在"也不再误判 404
    monkeypatch.setattr(voice_package, "export_voice", _raise)
    r = client.get("/api/adr/v1/profiles/demo/export")
    assert r.status_code == 400 and r.json()["code"] == "export_failed"


def test_import_upload_cap_413(client, monkeypatch, tmp_path):
    from adr.models import voice_package
    monkeypatch.setenv("ADR_PROFILE_IMPORT_MAX_MB", "1")
    called = []
    monkeypatch.setattr(voice_package, "import_voice",
                        lambda *a, **k: called.append(a))
    r = client.post("/api/adr/v1/profiles/import",
                    files={"package": ("x.adrvoice", b"\0" * (2 << 20))})
    assert r.status_code == 413 and r.json()["code"] == "payload_too_large"
    assert called == []


@pytest.fixture
def parse_spy(monkeypatch):
    # 批次66: 记录 multipart 解析器是否真正吃进了文件数据
    import python_multipart
    fed = []
    real_write = python_multipart.MultipartParser.write

    def spy(self, data):
        fed.append(len(data))
        return real_write(self, data)
    monkeypatch.setattr(python_multipart.MultipartParser, "write", spy)
    return fed


def test_import_cap_rejects_before_parse_content_length(client, monkeypatch,
                                                        parse_spy):
    # #38: Content-Length 超限 → 解析前 413, 一字节都不进解析器/临时盘
    monkeypatch.setenv("ADR_PROFILE_IMPORT_MAX_MB", "1")
    r = client.post("/api/adr/v1/profiles/import",
                    files={"package": ("x.adrvoice", b"\0" * (2 << 20))})
    assert r.status_code == 413 and r.json()["code"] == "payload_too_large"
    assert parse_spy == []


def test_import_cap_rejects_chunked_without_length(client, monkeypatch,
                                                   parse_spy):
    # #38: 无 Content-Length (chunked) 也在计数越限时中断, 不喂给解析器
    monkeypatch.setenv("ADR_PROFILE_IMPORT_MAX_MB", "1")
    b = "XBOUND"
    head = (f"--{b}\r\nContent-Disposition: form-data; name=\"package\"; "
            f"filename=\"x.adrvoice\"\r\n\r\n").encode()

    def gen():
        yield head
        for _ in range(4):
            yield b"\0" * (1 << 20)
        yield f"\r\n--{b}--\r\n".encode()
    r = client.post("/api/adr/v1/profiles/import", content=gen(),
                    headers={"content-type": f"multipart/form-data; boundary={b}"})
    assert r.status_code == 413 and r.json()["code"] == "payload_too_large"
    assert sum(parse_spy) <= (1 << 20) + len(head)


def test_import_under_cap_still_works(client, monkeypatch):
    from adr.models import gsv_engine, voice_package
    monkeypatch.setenv("ADR_PROFILE_IMPORT_MAX_MB", "1")
    got = []
    monkeypatch.setattr(voice_package, "import_voice",
                        lambda tmp, name, ow: got.append((name, ow)) or {"name": "v"})
    monkeypatch.setattr(gsv_engine, "try_schedule_profile", lambda: None)
    r = client.post("/api/adr/v1/profiles/import",
                    files={"package": ("x.adrvoice", b"PK" * 1000)},
                    data={"name": "v", "overwrite": "true"})
    assert r.status_code == 200 and r.json() == {"imported": {"name": "v"}}
    assert got == [("v", True)]


# ─── P0-2 OpenAI 兼容层与官方 SDK 对齐 ───

def test_openai_standard_model_voice_uses_default_profile(client, voice_dir):
    # tts-1 / alloy 是 OpenAI 标准名, 不当作 ADR 档案, 回退默认档案
    client.app.state.default_profile = "demo"
    r = client.post("/v1/audio/speech", json={
        "model": "tts-1", "voice": "alloy", "input": "hi",
        "response_format": "wav"})
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"


def test_openai_stream_default_false(client, voice_dir):
    # 官方 SDK 默认非流式: 整段音频 (有 Content-Length); stream=true 才流式
    body = {"input": "hi", "model": "demo", "response_format": "wav"}
    r = client.post("/v1/audio/speech", json=body)
    assert r.status_code == 200 and "content-length" in r.headers
    r2 = client.post("/v1/audio/speech", json=dict(body, stream=True))
    assert r2.status_code == 200
    assert "content-length" not in {k.lower() for k in r2.headers}


def test_openai_custom_profile_still_resolves(client, voice_dir):
    r = client.post("/v1/audio/speech", json={
        "model": "demo", "input": "hi", "response_format": "wav"})
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"


def test_openai_unknown_profile_keeps_error(client, voice_dir):
    # 自定义档案名笔误不能被静默换成默认人声
    client.app.state.default_profile = "demo"
    r = client.post("/v1/audio/speech", json={
        "model": "typo-voice", "input": "hi", "response_format": "wav"})
    assert r.status_code == 400
    assert "error" in r.json()
