"""批次11: adr.server 服务层测试 (FakeEngine, 不加载真模型)。

覆盖:
- /api/v2/tts GET+POST: 参数校验 400 / 非流式 WAV / 流式首块 WAV 头+裸 PCM
- streaming_mode 分支语义 (含 bool True == 1)
- set_gpt/set_sovits_weights 端点与错误格式
- profile/voice 档案解析 + ADR_TTS_DEFAULT_PROFILE 回退
- /api/adr/v1 原生端点 (health/profiles/ref/tts)
"""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adr.server import create_app


class FakeEngine:
    """记录调用参数的假引擎 (签名对齐 GSVEngine)。"""

    def __init__(self, sr=32000):
        self.sr = sr
        self.synth_calls = []
        self.stream_calls = []
        self.warmup_calls = []
        self.fail = False

    def synthesize(self, text, ref_audio, prompt_text="", text_lang="zh",
                   prompt_lang="zh", speed_factor=1.0, seed=-1,
                   t2s_weights=None, vits_weights=None, split_method="cut1",
                   top_k=15, top_p=1.0, temperature=1.0):
        self.synth_calls.append(dict(
            text=text, ref_audio=ref_audio, prompt_text=prompt_text,
            text_lang=text_lang, prompt_lang=prompt_lang,
            speed_factor=speed_factor, seed=seed, t2s_weights=t2s_weights,
            vits_weights=vits_weights, split_method=split_method))
        if self.fail:
            raise RuntimeError("boom")
        return np.full(self.sr // 10, 0.1, np.float32), self.sr

    def synthesize_stream(self, text, ref_audio, prompt_text="", text_lang="zh",
                          prompt_lang="zh", t2s_weights=None, vits_weights=None,
                          split_method="cut3", head_seed=-1):
        self.stream_calls.append(dict(
            text=text, ref_audio=ref_audio, prompt_text=prompt_text,
            t2s_weights=t2s_weights, vits_weights=vits_weights,
            split_method=split_method, head_seed=head_seed))
        if self.fail:
            raise RuntimeError("boom")
        for _ in range(3):
            yield np.full(160, 0.1, np.float32), self.sr

    def warmup(self, vits_weights=None, t2s_weights=None):
        self.warmup_calls.append((vits_weights, t2s_weights))


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine=engine))


@pytest.fixture
def voice_dir(tmp_path, monkeypatch):
    """假档案库: demo 档案 (ref.wav + meta.json), monkeypatch VOICES_DIR。"""
    from adr.models import voice_library
    vd = tmp_path / "voices"
    (vd / "demo").mkdir(parents=True)
    (vd / "demo" / "ref.wav").write_bytes(b"RIFF0000WAVEfmt ")  # 内容不校验
    (vd / "demo" / "meta.json").write_text(json.dumps({
        "name": "demo", "prompt_text": "参考文本",
        "t2s_weights": None, "vits_weights": "SoVITS_weights_v2/demo.pth",
        "rvc_weights": None, "rvc_index": None, "style": "平静",
        "created_at": "2026-01-01 00:00:00"}), encoding="utf-8")
    monkeypatch.setattr(voice_library, "VOICES_DIR", vd)
    return vd


# ─── /api/v2/tts 校验 ───

def test_v2_missing_text_400(client):
    r = client.post("/api/v2/tts", json={"ref_audio_path": "a.wav"})
    assert r.status_code == 400
    assert r.json()["message"] == "text is required"


def test_v2_missing_ref_400(client):
    r = client.post("/api/v2/tts", json={"text": "hi"})
    assert r.status_code == 400
    assert "ref_audio_path" in r.json()["message"]


def test_v2_bad_media_type_400(client):
    r = client.post("/api/v2/tts",
                    json={"text": "hi", "ref_audio_path": "a.wav", "media_type": "mp3"})
    assert r.status_code == 400
    assert "media_type" in r.json()["message"]


def test_v2_bad_streaming_mode_400(client):
    r = client.post("/api/v2/tts",
                    json={"text": "hi", "ref_audio_path": "a.wav", "streaming_mode": 9})
    assert r.status_code == 400
    assert "streaming_mode" in r.json()["message"]


# ─── /api/v2/tts 非流式 ───

def test_v2_post_nonstream_wav(client, engine):
    r = client.post("/api/v2/tts", json={
        "text": "你好", "ref_audio_path": "a.wav", "text_lang": "ZH"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"
    call = engine.synth_calls[0]
    assert call["text"] == "你好"
    assert call["text_lang"] == "zh"          # GET/POST 均转小写
    # 3200 样本 * 2 字节 + 44 头
    assert len(r.content) == 44 + 3200 * 2


def test_v2_get_nonstream(client, engine):
    r = client.get("/api/v2/tts",
                   params={"text": "hi", "ref_audio_path": "a.wav",
                           "prompt_lang": "EN", "speed_factor": 1.2})
    assert r.status_code == 200
    call = engine.synth_calls[0]
    assert call["prompt_lang"] == "en"         # lower() 生效
    assert call["speed_factor"] == 1.2


def test_v2_nonstream_raw(client, engine):
    r = client.post("/api/v2/tts",
                    json={"text": "hi", "ref_audio_path": "a.wav", "media_type": "raw"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/raw")
    assert len(r.content) == 3200 * 2          # 裸 s16le, 无头
    assert r.content[:4] != b"RIFF"


def test_v2_tts_failed_400(client, engine):
    engine.fail = True
    r = client.post("/api/v2/tts", json={"text": "hi", "ref_audio_path": "a.wav"})
    assert r.status_code == 400
    body = r.json()
    assert body["message"] == "tts failed"
    assert "boom" in body["Exception"]


# ─── /api/v2/tts 流式 ───

def _assert_stream_wav(body: bytes):
    """首块 44B WAV 头 + 3 块 * 160 样本 * 2 字节裸 PCM。"""
    assert body[:4] == b"RIFF"
    assert b"WAVE" in body[:44]
    assert len(body) == 44 + 3 * 160 * 2


def test_v2_stream_mode1(client, engine):
    r = client.post("/api/v2/tts", json={
        "text": "hi", "ref_audio_path": "a.wav", "streaming_mode": 1})
    assert r.status_code == 200
    _assert_stream_wav(r.content)
    assert len(engine.stream_calls) == 1
    assert engine.synth_calls == []


def test_v2_stream_bool_true_is_branch1(client, engine):
    """bool True == 1 → 分支 1 (分段流) — 与 GSV api_v2 行为一致。"""
    r = client.post("/api/v2/tts", json={
        "text": "hi", "ref_audio_path": "a.wav", "streaming_mode": True})
    assert r.status_code == 200
    _assert_stream_wav(r.content)


def test_v2_stream_mode2_and_3(client, engine):
    for mode in (2, 3):
        r = client.post("/api/v2/tts", json={
            "text": "hi", "ref_audio_path": "a.wav", "streaming_mode": mode})
        assert r.status_code == 200
        _assert_stream_wav(r.content)


# ─── 权重切换端点 ───

def test_v2_set_gpt_weights(client, engine):
    r = client.get("/api/v2/set_gpt_weights", params={"weights_path": "g.ckpt"})
    assert r.status_code == 200
    assert r.json() == {"message": "success"}
    assert engine.warmup_calls == [(None, "g.ckpt")]   # (vits, t2s)


def test_v2_set_sovits_weights(client, engine):
    r = client.get("/api/v2/set_sovits_weights", params={"weights_path": "s.pth"})
    assert r.status_code == 200
    assert engine.warmup_calls == [("s.pth", None)]


def test_v2_set_weights_missing_path(client):
    for ep, msg in [("set_gpt_weights", "gpt weight path is required"),
                    ("set_sovits_weights", "sovits weight path is required")]:
        r = client.get(f"/api/v2/{ep}")
        assert r.status_code == 400
        assert r.json()["message"] == msg


def test_v2_set_weights_load_fail():
    class _Boom:
        def warmup(self, vits_weights=None, t2s_weights=None):
            raise RuntimeError("bad ckpt")

    c2 = TestClient(create_app(engine=_Boom()))
    r = c2.get("/api/v2/set_gpt_weights", params={"weights_path": "x.ckpt"})
    assert r.status_code == 400
    assert r.json()["message"] == "change gpt weight failed"


# ─── profile 档案扩展 ───

def test_v2_profile_resolution(client, engine, voice_dir):
    r = client.post("/api/v2/tts", json={"text": "hi", "profile": "demo"})
    assert r.status_code == 200
    call = engine.synth_calls[0]
    assert call["ref_audio"].endswith("demo\\ref.wav") or \
           call["ref_audio"].endswith("demo/ref.wav")
    assert call["prompt_text"] == "参考文本"
    assert call["vits_weights"] == "SoVITS_weights_v2/demo.pth"
    assert call["t2s_weights"] is None


def test_v2_voice_alias_and_explicit_override(client, engine, voice_dir):
    r = client.post("/api/v2/tts", json={
        "text": "hi", "voice": "demo",
        "prompt_text": "显式覆盖", "vits_weights": "显式.pth"})
    assert r.status_code == 200
    call = engine.synth_calls[0]
    assert call["prompt_text"] == "显式覆盖"       # 显式值优先于档案
    assert call["vits_weights"] == "显式.pth"


def test_v2_unknown_profile_400(client, engine, voice_dir):
    r = client.post("/api/v2/tts", json={"text": "hi", "profile": "ghost"})
    assert r.status_code == 400
    assert r.json()["message"] == "unknown profile: ghost"


def test_v2_default_profile_env(engine, voice_dir, monkeypatch):
    monkeypatch.setenv("ADR_TTS_DEFAULT_PROFILE", "demo")
    client = TestClient(create_app(engine=engine))
    r = client.post("/api/v2/tts", json={"text": "hi"})
    assert r.status_code == 200
    assert engine.synth_calls[0]["ref_audio"].endswith(("demo\\ref.wav", "demo/ref.wav"))


def test_v2_profile_stream(client, engine, voice_dir):
    r = client.post("/api/v2/tts", json={
        "text": "hi", "profile": "demo", "streaming_mode": 1})
    assert r.status_code == 200
    _assert_stream_wav(r.content)
    assert engine.stream_calls[0]["ref_audio"].endswith(("demo\\ref.wav", "demo/ref.wav"))


# ─── /api/adr/v1 原生 API ───

def test_native_health(client):
    r = client.get("/api/adr/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "adr-tts"


def test_native_profiles(client, voice_dir):
    r = client.get("/api/adr/v1/profiles")
    assert r.status_code == 200
    names = [p["name"] for p in r.json()["profiles"]]
    assert names == ["demo"]


def test_native_profile_ref(client, voice_dir):
    r = client.get("/api/adr/v1/profiles/demo/ref")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content.startswith(b"RIFF")


def test_native_profile_ref_404(client, voice_dir):
    r = client.get("/api/adr/v1/profiles/ghost/ref")
    assert r.status_code == 404


def test_native_tts_post(client, engine, voice_dir):
    r = client.post("/api/adr/v1/tts",
                    json={"text": "hi", "profile": "demo", "seed": 7})
    assert r.status_code == 200
    call = engine.synth_calls[0]
    assert call["ref_audio"].endswith(("demo\\ref.wav", "demo/ref.wav"))
    assert call["seed"] == 7


def test_native_tts_get_and_stream(client, engine, voice_dir):
    r = client.get("/api/adr/v1/tts",
                   params={"text": "hi", "profile": "demo", "streaming_mode": "true"})
    assert r.status_code == 200
    _assert_stream_wav(r.content)
    assert len(engine.stream_calls) == 1
