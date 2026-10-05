"""批次11: adr.server 服务层测试 (FakeEngine, 不加载真模型)。

覆盖:
- /api/v2/tts GET+POST: 参数校验 400 / 非流式 WAV / 流式首块 WAV 头+裸 PCM
- streaming_mode 分支语义 (含 bool True == 1)
- set_gpt/set_sovits_weights 端点与错误格式
- profile/voice 档案解析 + ADR_TTS_DEFAULT_PROFILE 回退
- /api/adr/v1 原生端点 (health/profiles/ref/tts)
- ADR_TTS_API_KEY 鉴权 (Bearer/X-API-Key/query 三通道 + 豁免 health + 默认关闭)
- /v1/audio/speech + /v1/models OpenAI 兼容面 (批次21, N.E.KO OpenAI provider)
- TTS LRU 缓存: 命中/关闭/media_type 入 key/seed 不入 key (批次21)
"""
import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adr.server import create_app
from adr.server import tts_cache


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
                          split_method="cut3", head_seed=-1,
                          top_k=15, top_p=1.0, temperature=1.0,
                          speed_factor=1.0):
        self.stream_calls.append(dict(
            text=text, ref_audio=ref_audio, prompt_text=prompt_text,
            t2s_weights=t2s_weights, vits_weights=vits_weights,
            split_method=split_method, head_seed=head_seed,
            top_k=top_k, top_p=top_p, temperature=temperature,
            speed_factor=speed_factor))
        if self.fail:
            raise RuntimeError("boom")
        for _ in range(3):
            yield np.full(160, 0.1, np.float32), self.sr

    def warmup(self, vits_weights=None, t2s_weights=None):
        self.warmup_calls.append((vits_weights, t2s_weights))


@pytest.fixture(autouse=True)
def _clean_tts_cache(monkeypatch):
    """tts_cache 是模块级全局: 每用例前清空防跨用例泄漏 (批次21)。"""
    monkeypatch.delenv("ADR_TTS_CACHE", raising=False)
    tts_cache.clear()
    yield
    tts_cache.clear()


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine, monkeypatch):
    # 防鉴权用例的环境变量泄漏到默认 client
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
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
                    json={"text": "hi", "ref_audio_path": "a.wav", "media_type": "flac"})
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


# ─── 鉴权 (ADR_TTS_API_KEY) ───

@pytest.fixture
def auth_client(engine, monkeypatch):
    """双 key 配置: k1 / k2 (逗号分隔 + 空格, 验证解析健壮性)。

    缓存关闭: 三通道用例同 body 连发 3 次, 命中缓存会吞掉第 2/3 次合成。
    """
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    monkeypatch.setenv("ADR_TTS_API_KEY", "k1, k2")
    monkeypatch.setenv("ADR_TTS_CACHE", "0")
    return TestClient(create_app(engine=engine))


def test_auth_open_by_default(engine, monkeypatch):
    """未配置 key → 完全放行 (N.E.K.O 零改造兼容)。"""
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    c = TestClient(create_app(engine=engine))
    assert c.get("/api/adr/v1/health").status_code == 200
    r = c.post("/api/v2/tts", json={"text": "hi", "ref_audio_path": "a.wav"})
    assert r.status_code == 200


def test_auth_missing_key_401(auth_client):
    r = auth_client.post("/api/v2/tts", json={"text": "hi", "ref_audio_path": "a.wav"})
    assert r.status_code == 401
    assert r.json() == {"message": "unauthorized"}


def test_auth_wrong_key_401(auth_client):
    r = auth_client.get("/api/v2/tts",
                        params={"text": "hi", "ref_audio_path": "a.wav"},
                        headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_auth_three_credential_channels(auth_client, engine):
    """Bearer 头 / X-API-Key 头 / 查询参数 三通道等价。"""
    for kwargs in (
        {"headers": {"Authorization": "Bearer k1"}},
        {"headers": {"X-API-Key": "k2"}},
        {"params": {"api_key": "k1"}},
    ):
        r = auth_client.post("/api/v2/tts",
                             json={"text": "hi", "ref_audio_path": "a.wav"}, **kwargs)
        assert r.status_code == 200, kwargs
    assert len(engine.synth_calls) == 3


def test_auth_keys_exact_match(auth_client):
    """key 大小写敏感; 逗号+空格分隔解析出的 k1/k2 都可用。"""
    assert auth_client.get("/api/adr/v1/profiles",
                           headers={"X-API-Key": "K1"}).status_code == 401
    assert auth_client.get("/api/adr/v1/profiles",
                           headers={"X-API-Key": "k2"}).status_code == 200


def test_auth_health_exempt(auth_client):
    """监控探活免凭据。"""
    assert auth_client.get("/api/adr/v1/health").status_code == 200


def test_auth_empty_value_means_off(engine, monkeypatch):
    """ADR_TTS_API_KEY 置空串 = 未配置 → 放行。"""
    monkeypatch.setenv("ADR_TTS_API_KEY", "")
    c = TestClient(create_app(engine=engine))
    assert c.post("/api/v2/tts",
                  json={"text": "hi", "ref_audio_path": "a.wav"}).status_code == 200


# ─── OpenAI 兼容面 (/v1/audio/speech + /v1/models, 批次21) ───

def test_openai_speech_wav_nonstream(client, engine, voice_dir):
    """model=档案名 + stream=false → 与 /api/v2 非流式完全同路径。"""
    r = client.post("/v1/audio/speech", json={
        "model": "demo", "input": "你好", "response_format": "wav",
        "stream": False})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"
    call = engine.synth_calls[0]
    assert call["ref_audio"].endswith(("demo\\ref.wav", "demo/ref.wav"))
    assert call["prompt_text"] == "参考文本"          # 档案回填 prompt_text
    assert call["speed_factor"] == 1.0


def test_openai_voice_alias_fallback(client, engine, voice_dir):
    """voice 字段 (OpenAI 语义) 无 model 时作档案别名回退。"""
    r = client.post("/v1/audio/speech", json={
        "input": "hi", "voice": "demo", "response_format": "wav", "stream": False})
    assert r.status_code == 200
    assert engine.synth_calls[0]["ref_audio"].endswith(
        ("demo\\ref.wav", "demo/ref.wav"))


def test_openai_speed_passthrough(client, engine, voice_dir):
    r = client.post("/v1/audio/speech", json={
        "model": "demo", "input": "hi", "response_format": "wav",
        "stream": False, "speed": 1.3})
    assert r.status_code == 200
    assert engine.synth_calls[0]["speed_factor"] == 1.3


def test_openai_stream_wav(client, engine, voice_dir):
    """stream=true (默认) → 流式 WAV, 首块 44B 头 + 裸 PCM。"""
    r = client.post("/v1/audio/speech", json={
        "model": "demo", "input": "hi", "response_format": "wav", "stream": True})
    assert r.status_code == 200
    _assert_stream_wav(r.content)
    assert len(engine.stream_calls) == 1
    assert engine.synth_calls == []


def test_openai_stream_speed_passthrough(client, engine, voice_dir):
    """流式 speed 透传到引擎 (批次21 补齐)。"""
    r = client.post("/v1/audio/speech", json={
        "model": "demo", "input": "hi", "response_format": "wav",
        "stream": True, "speed": 1.2})
    assert r.status_code == 200
    assert engine.stream_calls[0]["speed_factor"] == 1.2


def test_openai_unknown_model_openai_error(client, engine, voice_dir):
    """未知档案 → 400 + OpenAI 风格错误壳 (error.type / error.message)。"""
    r = client.post("/v1/audio/speech", json={
        "model": "ghost", "input": "hi", "response_format": "wav", "stream": False})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "unknown profile: ghost" in err["message"]


def test_openai_bad_response_format_400(client):
    r = client.post("/v1/audio/speech", json={
        "input": "hi", "response_format": "flac"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "response_format" in err["message"]


def test_openai_models_list(client, voice_dir):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["demo"]
    assert body["data"][0]["object"] == "model"
    assert body["data"][0]["owned_by"] == "adr"


def test_openai_default_format_is_mp3(client, engine, voice_dir):
    """不传 response_format → mp3 (OpenAI 官方默认); 需 ffmpeg。"""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    r = client.post("/v1/audio/speech", json={
        "model": "demo", "input": "hi", "stream": False})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/mp3")


def test_openai_explicit_null_model_uses_voice(client, engine, voice_dir):
    """显式 null model (N.E.KO 可能发) + voice 别名 → 不炸。"""
    r = client.post("/v1/audio/speech", json={
        "model": None, "voice": "demo", "input": "hi",
        "response_format": "wav", "stream": False})
    assert r.status_code == 200
    assert engine.synth_calls[0]["ref_audio"].endswith(
        ("demo\\ref.wav", "demo/ref.wav"))


# ─── TTS LRU 缓存 (批次21) ───

def test_cache_hit_skips_synth(client, engine):
    """同 body 二次 POST → 命中缓存, 引擎只跑一次。"""
    body = {"text": "hi", "ref_audio_path": "a.wav"}
    r1 = client.post("/api/v2/tts", json=body)
    r2 = client.post("/api/v2/tts", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.content == r2.content
    assert len(engine.synth_calls) == 1


def test_cache_disabled_via_env(client, engine, monkeypatch):
    """ADR_TTS_CACHE=0 → 完全旁路。"""
    monkeypatch.setenv("ADR_TTS_CACHE", "0")
    body = {"text": "hi", "ref_audio_path": "a.wav"}
    client.post("/api/v2/tts", json=body)
    client.post("/api/v2/tts", json=body)
    assert len(engine.synth_calls) == 2


def test_cache_key_includes_media_type(client, engine):
    """wav 与 raw 缓存不同条目 (key 含 media_type)。"""
    client.post("/api/v2/tts", json={"text": "hi", "ref_audio_path": "a.wav",
                                     "media_type": "wav"})
    client.post("/api/v2/tts", json={"text": "hi", "ref_audio_path": "a.wav",
                                     "media_type": "raw"})
    assert len(engine.synth_calls) == 2


def test_cache_seed_not_in_key(client, engine):
    """seed 不入 key: 播报一致性优先, 仅 seed 不同 → 复用同一次合成。"""
    client.post("/api/v2/tts", json={"text": "hi", "ref_audio_path": "a.wav",
                                     "seed": 1})
    client.post("/api/v2/tts", json={"text": "hi", "ref_audio_path": "a.wav",
                                     "seed": 2})
    assert len(engine.synth_calls) == 1


def test_cache_stream_bypassed(client, engine):
    """流式路径不缓存: 同 body 两次流式 → 引擎跑两次。"""
    body = {"text": "hi", "ref_audio_path": "a.wav", "streaming_mode": 1}
    client.post("/api/v2/tts", json=body)
    client.post("/api/v2/tts", json=body)
    assert len(engine.stream_calls) == 2


def test_cache_failed_synth_not_stored(client, engine):
    """合成失败不写缓存: 失败后修好, 同 body 重试会真正再合成 (共 2 次调用)。"""
    body = {"text": "hi", "ref_audio_path": "a.wav"}
    engine.fail = True
    assert client.post("/api/v2/tts", json=body).status_code == 400
    engine.fail = False
    assert client.post("/api/v2/tts", json=body).status_code == 200
    assert len(engine.synth_calls) == 2


def test_v2_stream_speed_factor_passthrough(client, engine):
    """v2 流式 speed_factor 透传 (批次21 补齐; ==1.0 也照传, 引擎自行短路)。"""
    r = client.post("/api/v2/tts", json={
        "text": "hi", "ref_audio_path": "a.wav",
        "streaming_mode": 1, "speed_factor": 1.2})
    assert r.status_code == 200
    _assert_stream_wav(r.content)
    assert engine.stream_calls[0]["speed_factor"] == 1.2
