"""Track B 服务安全测试: pathsafe 单元 + WS 鉴权中间件。

覆盖:
- resolve_within: 相对/绝对合法路径放行, ../ 穿越 / 换盘符越界拒绝
- is_safe_name: 单段纯名校验 (路径分隔符 / .. / 空字节)
- WS 鉴权 (ADR_TTS_API_KEY): 无凭据握手拒绝 (4401) / ?token= 放行 /
  ?api_key= 放行 / Sec-WebSocket-Protocol 子协议放行且回显 /
  未配置 key 默认放行 (NEKO 零改造兼容红线)
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from adr.server import create_app
from adr.server.pathsafe import is_safe_name, resolve_within


# ─── pathsafe 单元 ───

def test_resolve_within_relative_joins_root(tmp_path):
    """相对 candidate 基于 root 解析 (曾因基于 cwd 解析误判越界)。"""
    (tmp_path / "demo").mkdir()
    got = resolve_within(tmp_path, Path("demo") / "ref.wav")
    assert got is not None
    assert got == tmp_path.resolve() / "demo" / "ref.wav"


def test_resolve_within_absolute_inside_ok(tmp_path):
    assert resolve_within(tmp_path, tmp_path / "demo" / "ref.wav") is not None


def test_resolve_within_rejects_escape(tmp_path):
    assert resolve_within(tmp_path, "../outside") is None
    assert resolve_within(tmp_path, "demo/../../outside") is None
    assert resolve_within(tmp_path, tmp_path.parent / "outside") is None


def test_resolve_within_rejects_other_drive(tmp_path):
    """越界候选 (含换盘符): relative_to 抛 ValueError → None。"""
    assert resolve_within(tmp_path, "C:/Windows") is None


def test_is_safe_name():
    assert is_safe_name("demo")
    assert is_safe_name("demo_v2")
    for bad in ("", "..", "../demo", "demo/..", "a/b", "a\\b", "a\x00b"):
        assert not is_safe_name(bad), bad


def test_is_safe_name_rejects_absolute_paths():
    """批次41a: 绝对路径 / 盘符名 / 反斜杠绝对路径一律拒绝。"""
    for bad in ("/abs/evil", "\\evil", "C:\\evil", "C:/evil"):
        assert not is_safe_name(bad), bad


# ─── 批次41a: 鉴权常量时间比较 ───

def test_auth_uses_compare_digest(monkeypatch):
    """_key_ok 必须走 hmac.compare_digest (恒定时间), 不允许退化成 ==。"""
    import hmac as _hmac

    from adr.server.auth import APIKeyMiddleware

    called = []
    orig = _hmac.compare_digest

    def _spy(a, b):
        called.append((a, b))
        return orig(a, b)

    monkeypatch.setattr(_hmac, "compare_digest", _spy)
    mw = APIKeyMiddleware(app=None, api_keys=["k1"])
    assert mw._key_ok("k1")
    assert called, "key 比较必须经 hmac.compare_digest"


def test_auth_key_ok_bytes_safe(engine, monkeypatch):
    """批次41a: 两侧 bytes 编码比较 — 正确/错误/非 ASCII 凭据行为正确,
    非 ASCII 凭据不匹配时返回 False 而非抛 TypeError。"""
    from adr.server.auth import APIKeyMiddleware

    mw = APIKeyMiddleware(app=None, api_keys=["sécret", "k2"])
    assert mw._key_ok("k2")
    assert not mw._key_ok("k3")
    assert mw._key_ok("sécret")       # 非 ASCII key 命中
    assert not mw._key_ok("sécrét")   # 非 ASCII 不匹配 → False, 不抛错
    assert not mw._key_ok("")
    assert not mw._key_ok(None)


# ─── 批次41a: console 档案名穿越 + HTTP 错误响应结构化 code ───

def test_console_profile_create_rejects_traversal(security_client):
    """console 建档恶意档案名 → 400 + code=profile_invalid
    (名字校验先于音频存在性检查, 无需真实文件即可触发)。"""
    r = security_client.post("/api/adr/v1/profiles/create", json={
        "name": "../evil", "audio_path": "nonexistent.wav"})
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "profile_invalid"
    assert body["detail"]  # 既有 detail 字段逐字保留 (wire 兼容)


def test_console_profile_create_rejects_dotdot(security_client):
    """纯 ".." 名 (清洗不改变它) 同样拒绝。"""
    r = security_client.post("/api/adr/v1/profiles/create", json={
        "name": "..", "audio_path": "nonexistent.wav"})
    assert r.status_code == 400
    assert r.json()["code"] == "profile_invalid"


def test_console_profile_create_default_name_ok(security_client):
    """缺省中文名走清洗兜底后合法, 不因名字校验误拒 (报音频不存在而非名字)。"""
    r = security_client.post("/api/adr/v1/profiles/create", json={
        "name": "", "audio_path": "nonexistent.wav"})  # 空名 → 清洗兜底「我的声音」
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "bad_request"
    assert "音频不存在" in body["detail"]


def test_http_error_responses_carry_code(security_client):
    """HTTPException 响应增量补 code, detail 保留 (wire 兼容)。"""
    r = security_client.post("/api/adr/v1/train/stop")  # 无训练 → 409
    assert r.status_code == 409
    body = r.json()
    assert body["code"] == "conflict"
    assert "当前没有进行中的训练" in body["detail"]


# ─── 批次41a: voice_library 档案名安全 + meta.json 原子写 ───

def _mk_ref_wav(tmp_path):
    """4s 正弦波参考音频 (建档防护要求 3~10s)。"""
    import numpy as np
    import soundfile as sf

    p = tmp_path / "ref.wav"
    t = np.linspace(0, 4, 64000, endpoint=False, dtype="float32")
    sf.write(str(p), (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"),
             16000)
    return str(p)


def test_voice_library_rejects_malicious_names(tmp_path, monkeypatch):
    """save_voice / load_voice 拒绝穿越名; ".." 不把文件写进档案库上级。"""
    import adr.models.voice_library as vl

    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    for bad in ("..", "../evil", "C:\\evil", ""):
        with pytest.raises(ValueError):
            vl.save_voice(bad, "whatever.wav"), bad
        with pytest.raises(ValueError):
            vl.load_voice(bad), bad
    # 穿越 == 未得逞: 档案库上级没有落任何 ref.wav
    assert not (tmp_path / "ref.wav").exists()
    assert not (tmp_path / "meta.json").exists()


def test_voice_meta_json_atomic_replace(tmp_path, monkeypatch):
    """meta.json 原子写: .tmp 中转 + os.replace, 落盘后不残留 .tmp。"""
    import adr.models.voice_library as vl

    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    vl.save_voice("demo", _mk_ref_wav(tmp_path))
    vdir = tmp_path / "voices" / "demo"
    assert (vdir / "meta.json").exists()
    assert not (vdir / "meta.json.tmp").exists()
    import json

    assert json.loads(
        (vdir / "meta.json").read_text(encoding="utf-8"))["name"] == "demo"


# ─── WS 鉴权 (握手层, 不触达合成) ───

WS_PATH = "/api/v3/tts/stream-input"


class _FakeEngine:
    """占位引擎: 鉴权用例只验证握手层; 批次38 起 synthesize 供权重预检用例。"""

    sr = 32000

    def __init__(self):
        self.synth_calls = []

    def synthesize(self, text, ref_audio, prompt_text="", text_lang="zh",
                   prompt_lang="zh", speed_factor=1.0, seed=-1,
                   t2s_weights=None, vits_weights=None, split_method="cut1",
                   top_k=15, top_p=1.0, temperature=1.0,
                   fragment_interval=None):
        import numpy as np

        self.synth_calls.append(dict(
            text=text, ref_audio=ref_audio, t2s_weights=t2s_weights,
            vits_weights=vits_weights))
        return np.full(self.sr // 10, 0.1, np.float32), self.sr


@pytest.fixture
def engine():
    return _FakeEngine()


@pytest.fixture
def auth_client(engine, monkeypatch):
    """与 test_server_api.auth_client 同配置 (k1/k2), 面向 WS 握手。"""
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    monkeypatch.setenv("ADR_TTS_API_KEY", "k1, k2")
    return TestClient(create_app(engine=engine))


def test_ws_rejected_without_credential(auth_client):
    """有 key 配置 + 无凭据 → accept 前 close 4401。"""
    with pytest.raises(WebSocketDisconnect) as ei:
        with auth_client.websocket_connect(WS_PATH):
            pass
    assert ei.value.code == 4401


def test_ws_query_token_allows(auth_client):
    with auth_client.websocket_connect(f"{WS_PATH}?token=k1"):
        pass


def test_ws_query_api_key_allows(auth_client):
    with auth_client.websocket_connect(f"{WS_PATH}?api_key=k2"):
        pass


def test_ws_subprotocol_allows_and_echoes(auth_client):
    """Sec-WebSocket-Protocol 首元素作 key; 端点 accept 回显选中子协议
    (浏览器不回显会拒握手)。"""
    with auth_client.websocket_connect(WS_PATH, subprotocols=["k2"]) as ws:
        assert ws.accepted_subprotocol == "k2"


def test_ws_open_by_default(engine, monkeypatch):
    """未配置 key → WS 完全放行 (NEKO 零改造兼容红线)。"""
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    c = TestClient(create_app(engine=engine))
    with c.websocket_connect(WS_PATH):
        pass


# ─── 批次38: 请求级权重预检 (pickle RCE 早检) ───

@pytest.fixture
def security_client(engine, monkeypatch):
    """无鉴权环境 (权重预检用例不关心鉴权)。"""
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    return TestClient(create_app(engine=engine))


def test_tts_request_weights_reject_malicious_pickle(security_client, tmp_path):
    """非张量 pickle 对象 → /api/v2/tts 早检 400 (invalid_params), 不触达引擎。"""
    import pickle

    class _Boom:
        def __reduce__(self):
            return (print, ("pwned",))

    p = tmp_path / "evil.pth"
    p.write_bytes(pickle.dumps(_Boom()))
    r = security_client.post("/api/v2/tts", json={
        "text": "hi", "ref_audio_path": "a.wav", "t2s_weights": str(p)})
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "invalid_params"
    assert "t2s_weights" in body["message"]


def test_tts_request_weights_reject_nondict_pickle(security_client, tmp_path):
    """合法 pickle 但顶层不是 dict (如裸 tensor) → 400。"""
    import pickle

    import torch

    p = tmp_path / "bare.pth"
    p.write_bytes(pickle.dumps(torch.zeros(3)))
    r = security_client.post("/api/v2/tts", json={
        "text": "hi", "ref_audio_path": "a.wav", "vits_weights": str(p)})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_params"


def test_tts_request_weights_allow_tensor_dict(security_client, engine, tmp_path):
    """合法纯张量 dict 权重 (torch.save 格式) → 早检放行, 引擎收到请求级权重路径。"""
    import torch

    p = tmp_path / "good.pth"
    # 注意: torch.load(weights_only=True) 只放行 torch.save 产物,
    # 裸 pickle.dumps 的 protocol-4 流会被拒 (与真实 .pth 权重格式一致)。
    torch.save({"w": torch.zeros(3)}, p)
    r = security_client.post("/api/v2/tts", json={
        "text": "hi", "ref_audio_path": "a.wav", "vits_weights": str(p)})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert engine.synth_calls and engine.synth_calls[0]["vits_weights"] == str(p)
