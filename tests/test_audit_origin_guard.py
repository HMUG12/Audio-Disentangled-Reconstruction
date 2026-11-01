"""批次70: 来源守卫 (OriginGuard) 边界回归 — 审计 §5 测试缺口。

三层覆盖:
1. 纯函数: host_allowed / origin_allowed 的边界 (缺 Host / IP / 单段名 /
   带点域名 / 白名单 / 同源 / "null")
2. 中间件 scope 级: 直接驱动纯 ASGI, 覆盖 TestClient 不方便构造的场景
   (缺 Host 头、WebSocket close 4403、enabled=False 放行)
3. TestClient 集成: 真实 create_app 中间件栈顺序下 403 / 200
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from adr.server import create_app
from adr.server.origin_guard import (
    OriginGuardMiddleware,
    host_allowed,
    origin_allowed,
    split_host,
)

TAURI_ORIGIN = "tauri://localhost"


def test_split_host_variants():
    assert split_host("127.0.0.1:9881") == "127.0.0.1"
    assert split_host("LocalHost:9881") == "localhost"
    assert split_host("[::1]:9881") == "::1"
    assert split_host("mypc") == "mypc"


@pytest.mark.parametrize("host, ok", [
    (None, True),                      # 非浏览器 (HTTP/1.0) 无 Host 头
    ("127.0.0.1:9881", True),          # 回环 IP
    ("192.168.1.10:9881", True),       # 局域网 IP 直连
    ("[::1]:9881", True),              # IPv6 回环
    ("localhost:9881", True),          # 单段主机名
    ("mypc", True),                    # 局域网机器名 (无点)
    ("evil.example.com", False),       # DNS 重绑定典型形态
    ("evil.example.com.", False),      # 尾部点不能绕过
])
def test_host_allowed(host, ok):
    assert host_allowed(host, set()) is ok


def test_host_allowed_whitelist():
    extra = {"mypc.local"}
    assert host_allowed("mypc.local:9881", extra)
    assert host_allowed("MYPC.local", extra)        # 大小写不敏感
    assert not host_allowed("other.local", extra)


@pytest.mark.parametrize("origin, host, ok", [
    (TAURI_ORIGIN, "127.0.0.1:9881", True),              # 白名单
    ("http://localhost:3000", "127.0.0.1:9881", True),    # 本机页面 (任意端口)
    ("http://[::1]:8080", "127.0.0.1:9881", True),
    ("http://127.0.0.1:9881", "127.0.0.1:9881", True),   # 同源 (控制台自身)
    ("http://127.0.0.1:9999", "127.0.0.1:9881", True),   # 本机页面任意端口均放行
    ("https://evil.example.com", "127.0.0.1:9881", False),
    ("null", "127.0.0.1:9881", False),                    # 沙箱 iframe / file://
    ("http://evil.example.com", None, False),             # 无 Host 可比对
])
def test_origin_allowed(origin, host, ok):
    assert origin_allowed(origin, host, {TAURI_ORIGIN}) is ok


# ─── 中间件 scope 级 (纯 ASGI, 不经 TestClient) ───

class _Recorder:
    """内层应用: 记录是否被放行。"""

    def __init__(self):
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        await send({"type": "http.response.start", "status": 200,
                    "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _http_scope(headers):
    return {"type": "http", "method": "GET", "path": "/", "headers": headers}


async def _drive(app, scope):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    await app(scope, receive, send)
    return sent


def _status(sent):
    return next(m for m in sent if m["type"] == "http.response.start")["status"]


def test_middleware_missing_host_passes():
    """缺 Host 头 (非浏览器客户端) 放行 — N.E.K.O / curl 零改造红线。"""
    inner = _Recorder()
    app = OriginGuardMiddleware(inner)
    sent = asyncio.run(_drive(app, _http_scope([])))
    assert _status(sent) == 200 and inner.calls == 1


def test_middleware_rebinding_host_403():
    inner = _Recorder()
    app = OriginGuardMiddleware(inner)
    sent = asyncio.run(_drive(app, _http_scope(
        [(b"host", b"evil.example.com")])))
    assert _status(sent) == 403 and inner.calls == 0
    assert b"host not allowed" in sent[-1]["body"]


def test_middleware_null_origin_403():
    inner = _Recorder()
    app = OriginGuardMiddleware(inner)
    sent = asyncio.run(_drive(app, _http_scope(
        [(b"host", b"127.0.0.1:9881"), (b"origin", b"null")])))
    assert _status(sent) == 403 and b"origin not allowed" in sent[-1]["body"]


def test_middleware_websocket_close_4403():
    inner = _Recorder()
    app = OriginGuardMiddleware(inner)
    scope = {"type": "websocket", "path": "/api/v3/tts/stream-input",
             "headers": [(b"host", b"127.0.0.1:9881"),
                         (b"origin", b"https://evil.example.com")]}
    sent = asyncio.run(_drive(app, scope))
    assert sent == [{"type": "websocket.close", "code": 4403}]
    assert inner.calls == 0


def test_middleware_disabled_bypasses():
    inner = _Recorder()
    app = OriginGuardMiddleware(inner, enabled=False)
    sent = asyncio.run(_drive(app, _http_scope(
        [(b"host", b"evil.example.com")])))
    assert _status(sent) == 200 and inner.calls == 1


def test_middleware_allow_any_host_valve():
    """ADR_ALLOWED_HOSTS=* — 关闭 Host 检查 (Origin 检查仍独立生效)。"""
    inner = _Recorder()
    app = OriginGuardMiddleware(inner, allow_any_host=True)
    sent = asyncio.run(_drive(app, _http_scope(
        [(b"host", b"evil.example.com")])))
    assert _status(sent) == 200 and inner.calls == 1


def test_middleware_allow_any_origin_valve():
    """ADR_CORS_ORIGINS=* — 关闭 Origin 检查 (Host 检查仍独立生效)。"""
    inner = _Recorder()
    app = OriginGuardMiddleware(inner, allow_any_origin=True)
    sent = asyncio.run(_drive(app, _http_scope(
        [(b"host", b"127.0.0.1:9881"),
         (b"origin", b"https://evil.example.com")])))
    assert _status(sent) == 200 and inner.calls == 1


# ─── TestClient 集成 (真实中间件栈: APIKey → OriginGuard → CORS) ───

class _DummyEngine:
    def queue_depth(self):
        return 0


@pytest.fixture
def client(monkeypatch):
    # 守卫仅在未配置 API Key 时生效; 同时清掉逃生阀环境变量
    for name in ("ADR_TTS_API_KEY", "ADR_CORS_ORIGINS", "ADR_ALLOWED_HOSTS"):
        monkeypatch.delenv(name, raising=False)
    return TestClient(create_app(engine=_DummyEngine()))


def test_integration_rebinding_host_403(client):
    r = client.get("/v1/models", headers={"host": "evil.example.com"})
    assert r.status_code == 403
    assert r.json()["message"] == "forbidden: host not allowed"


def test_integration_cross_site_origin_403(client):
    r = client.get("/v1/models", headers={"host": "127.0.0.1:9881",
                                          "origin": "https://evil.example.com"})
    assert r.status_code == 403
    assert r.json()["message"] == "forbidden: origin not allowed"


def test_integration_same_origin_ok(client):
    r = client.get("/v1/models", headers={"host": "127.0.0.1:9881",
                                          "origin": "http://127.0.0.1:9881"})
    assert r.status_code == 200


def test_integration_non_browser_ok(client):
    """不带 Origin (N.E.K.O 后端 / curl) 且 Host 为回环 → 放行。"""
    assert client.get("/v1/models").status_code == 200
