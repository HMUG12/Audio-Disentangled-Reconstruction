"""批次12: API Key 鉴权中间件 (纯 ASGI, 零框架依赖)。

用法:
    app.add_middleware(APIKeyMiddleware, api_keys=["k1", "k2"])

规则:
- api_keys 为空列表 → 完全放行 (默认关闭, N.E.K.O 零改造兼容)
- 豁免路径: /api/adr/v1/health (监控探活不带凭据)
  + /, /pro, /easy (控制台静态页, 页面 JS 会把 URL 上的 api_key 透传给后续请求)
- 凭据三选一: ``Authorization: Bearer <key>`` / ``X-API-Key: <key>``
  / 查询参数 ``?api_key=<key>``
- key 精确匹配 (大小写敏感); 未通过统一 ``401 {"message": "unauthorized"}``
"""
from __future__ import annotations

import json
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

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or not self.api_keys
                or scope.get("path") in EXEMPT_PATHS):
            await self.app(scope, receive, send)
            return

        if self._extract_key(scope) in self.api_keys:
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
        for pair in (scope.get("query_string") or b"").split(b"&"):
            if pair.startswith(b"api_key="):
                return unquote(pair[len(b"api_key="):].decode("latin-1"))
        return None
