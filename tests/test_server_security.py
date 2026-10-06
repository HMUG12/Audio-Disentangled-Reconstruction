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
