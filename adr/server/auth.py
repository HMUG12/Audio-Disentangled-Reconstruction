"""批次12: API Key 鉴权中间件 (纯 ASGI, 零框架依赖)。

用法:
    app.add_middleware(APIKeyMiddleware, api_keys=["k1", "k2"])

规则:
- api_keys 为空列表 → 完全放行 (默认关闭, N.E.K.O 零改造兼容)
- 豁免路径: /api/adr/v1/health (监控探活不带凭据)
  + /, /pro, /easy (控制台静态页, 页面 JS 会把 URL 上的 api_key 透传给后续请求)
- HTTP 凭据三选一: ``Authorization: Bearer <key>`` / ``X-API-Key: <key>``
  / 查询参数 ``?api_key=<key>``
- WS 凭据三选一 (Track B 收口): 查询参数 ``?token=<key>`` (NEKO http→ws 转换
  会透传 query, 故兼容 ``?api_key=``) / 查询参数 ``?api_key=<key>``
  / ``Sec-WebSocket-Protocol`` 子协议首元素。首条消息通道不采用 —
  v3 协议首条必须是 init, 加鉴权帧会破坏协议兼容
- key 精确匹配 (大小写敏感, 恒定时间比较); 未通过 HTTP 统一
  ``401 {"message": "unauthorized"}``, WS 在握手前 close (4401)
"""
from __future__ import annotations

import json
import secrets
from urllib.parse import unquote

# 监控探活豁免 (无敏感数据, 供 N.E.K.O / 负载均衡探活)
# 控制台静态页豁免 (批次26 补 /call): 页面本身无数据, 启用 API key 时页面才能
# 加载; 页面 JS 发起的 API 调用仍走鉴权 — 与 /pro /easy 语义一致。
EXEMPT_PATHS = {"/api/adr/v1/health", "/", "/pro", "/easy", "/call"}


class APIKeyMiddleware:
    def __init__(self, app, api_keys=None):
        self.app = app
        # 空集 = 鉴权关闭 (不设 ADR_TTS_API_KEY 即开放)
        self.api_keys = {k for k in (api_keys or ()) if k}

    def _key_ok(self, key) -> bool:
        """恒定时间比较 (Track B): 逐个 compare_digest, 防时序侧信道枚举。"""
        if not key:
            return False
        return any(secrets.compare_digest(key, k) for k in self.api_keys)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            # Track B 收口: WS 连接此前完全绕过鉴权。未配置 key 时维持
            # 完全放行 (本地开放语义 / NEKO 零改造兼容红线)。
            if (not self.api_keys or scope.get("path") in EXEMPT_PATHS):
                await self.app(scope, receive, send)
                return
            message = await receive()  # 取 websocket.connect 再裁决
            if (message.get("type") == "websocket.connect"
                    and self._key_ok(self._extract_ws_key(scope))):
                # connect 消息已被消费, 须向下游重放一次 — starlette 的
                # websocket.accept() 自身会 receive() 等待 connect 消息,
                # 不重放会拿到后续数据帧并抛 "Expected websocket.connect"。
                replayed = {"done": False}

                async def _replay_receive():
                    if not replayed["done"]:
                        replayed["done"] = True
                        return message
                    return await receive()

                await self.app(scope, _replay_receive, send)
                return
            # 未 accept 前直接 close → 服务器拒绝握手 (uvicorn 回 403)
            await send({"type": "websocket.close", "code": 4401})
            return

        if (scope["type"] != "http" or not self.api_keys
                or scope.get("path") in EXEMPT_PATHS):
            await self.app(scope, receive, send)
            return

        if self._key_ok(self._extract_key(scope)):
            await self.app(scope, receive, send)
            return

        body = json.dumps({"message": "unauthorized"}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json; charset=utf-8")],
        })
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _extract_key(scope):
        """按 Bearer 头 → X-API-Key 头 → 查询参数的顺序取 key。"""
        headers = dict(scope.get("headers") or [])  # ASGI 头键为小写字节
        auth = headers.get(b"authorization")
        if auth:
            parts = auth.decode("latin-1").split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
        api_key = headers.get(b"x-api-key")
        if api_key:
            return api_key.decode("latin-1").strip()
        return _query_key(scope, b"api_key")

    @staticmethod
    def _extract_ws_key(scope):
        """WS 凭据 (Track B): query token= → query api_key= → 子协议首元素。

        子协议通道: 客户端 ``WebSocket(url, ["<key>"])``, 取首元素为 key,
        由 v3 端点在 accept 时回显 (浏览器要求回显选中子协议, 否则握手失败)。
        """
        key = _query_key(scope, b"token")
        if key:
            return key
        key = _query_key(scope, b"api_key")
        if key:
            return key
        headers = dict(scope.get("headers") or [])
        proto = headers.get(b"sec-websocket-protocol")
        if proto:
            parts = [p.strip() for p in proto.decode("latin-1").split(",")]
            if parts and parts[0]:
                return parts[0]
        return None


def _query_key(scope, name: bytes):
    """从 query_string 取 ``name=`` 的值 (unquote); 不存在返回 None。"""
    prefix = name + b"="
    for pair in (scope.get("query_string") or b"").split(b"&"):
        if pair.startswith(prefix):
            return unquote(pair[len(prefix):].decode("latin-1"))
    return None
