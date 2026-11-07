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
    cross_site_allowed,
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


# ─── 批次77 (#4): 有副作用接口的浏览器跨站防护 ───

DOCTOR = "/api/adr/v1/system/doctor"


@pytest.mark.parametrize("site, origin, ok", [
    (None, None, True),                                  # 非浏览器 (N.E.K.O/curl/SDK)
    ("", None, True),                                    # 空头视同缺头
    ("same-origin", None, True),                         # 控制台页自身
    ("same-site", None, True),                           # 本机其它端口页面
    ("none", None, True),                                # 地址栏直达
    ("cross-site", None, False),                         # <img>/导航/no-cors fetch
    ("cross-site", "https://evil.example.com", False),   # 白名单外 Origin
    ("cross-site", "http://127.0.0.1:9999", True),       # 本机页面
    ("cross-site", TAURI_ORIGIN, True),                  # Tauri 壳 → 127.0.0.1
])
def test_cross_site_allowed(site, origin, ok):
    allowed = {TAURI_ORIGIN}
    assert cross_site_allowed(site, origin, "127.0.0.1:9881", allowed) is ok


def test_cross_site_allowed_valve():
    """ADR_CORS_ORIGINS=* 逃生阀 — 恢复旧的全开语义。"""
    assert cross_site_allowed("cross-site", None, "127.0.0.1:9881", set(), True)


@pytest.mark.parametrize("path", [
    "/api/v2/set_gpt_weights",
    "/api/v2/set_sovits_weights",
    DOCTOR,
])
def test_integration_cross_site_side_effect_403(client, path):
    """浏览器跨站 (无 Origin) 触发换权 / doctor 修复 → 403, 不进业务逻辑。"""
    r = client.get(path, headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403
    assert "cross-site" in r.json()["detail"]


def test_integration_cross_site_allowed_origin_passes(client):
    """Tauri 壳的合法跨站 (带白名单 Origin) 不被依赖拦。"""
    r = client.get("/api/v2/set_gpt_weights",
                   headers={"sec-fetch-site": "cross-site",
                            "origin": TAURI_ORIGIN})
    assert r.status_code == 400
    assert r.json()["message"] == "gpt weight path is required"


@pytest.mark.parametrize("ep, kind", [("set_gpt_weights", "gpt"),
                                      ("set_sovits_weights", "sovits")])
def test_post_alias_reaches_handler(client, ep, kind):
    """批次77: 有副作用接口新增 POST 别名 (N.E.K.O 的 GET 仍保留)。"""
    r = client.post(f"/api/v2/{ep}")
    assert r.status_code == 400
    assert r.json()["message"] == f"{kind} weight path is required"


def test_doctor_post_alias(client):
    r = client.post(DOCTOR)
    assert r.status_code == 200 and "checks" in r.json()


def test_cross_site_allowed_when_api_key_set(monkeypatch):
    """配了 API Key 时由鉴权兜底 — 依赖放行, 反代/自定义域名部署不被误伤。"""
    monkeypatch.setenv("ADR_TTS_API_KEY", "k1")
    c = TestClient(create_app(engine=_DummyEngine()))
    r = c.get("/api/v2/set_gpt_weights",
              headers={"sec-fetch-site": "cross-site", "x-api-key": "k1"})
    assert r.status_code == 400
    assert r.json()["message"] == "gpt weight path is required"
