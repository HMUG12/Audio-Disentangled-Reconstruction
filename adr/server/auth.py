"""批次12: API Key 鉴权中间件 (纯 ASGI, 零框架依赖)。

用法:
    app.add_middleware(APIKeyMiddleware, api_keys=["k1", "k2"])

规则:
- api_keys 为空列表 → 完全放行 (默认关闭, N.E.K.O 零改造兼容)
- 豁免路径: /api/adr/v1/health、/api/adr/v1/readyz (监控探活不带凭据)
  + /, /pro, /easy, /call, /settings (控制台静态页, 页面 JS 首屏读取 URL 上的
  api_key 后转存 sessionStorage 并抹掉地址栏, 后续请求改带 X-API-Key 头)
- HTTP 凭据: ``Authorization: Bearer <key>`` / ``X-API-Key: <key>``;
  查询参数 ``?api_key=<key>`` **默认关闭**, 仅在 ``ADR_ALLOW_QUERY_API_KEY=1``
  时作为兜底 (查询串会泄漏进浏览器历史 / Referer / 反代日志)
- WS 凭据三选一 (Track B 收口): 查询参数 ``?token=<key>`` (NEKO http→ws 转换
  会透传 query, 是 v3 协议固有握手通道) / ``?api_key=<key>`` (与 HTTP 同政策,
  仅 ``ADR_ALLOW_QUERY_API_KEY=1`` 时接受) / ``Sec-WebSocket-Protocol``
  子协议首元素。首条消息通道不采用 — v3 协议首条必须是 init, 加鉴权帧会破坏
  协议兼容。``?token=`` 会进入反代/访问日志, 部署方需自行脱敏
- key 精确匹配 (大小写敏感, 恒定时间比较); 未通过 HTTP 统一
  ``401 {"detail": ..., "message": ..., "code": "unauthorized"}`` (批次84:
  与全站错误契约对齐, 只增字段), WS 在握手前 close (4401)
"""
from __future__ import annotations

import hmac
import json
import re
from urllib.parse import unquote

from adr.core.settings import allow_query_api_key
from adr.server.exceptions import UNAUTHORIZED

# 监控探活豁免 (无敏感数据, 供 N.E.K.O / 负载均衡 / 容器 HEALTHCHECK 探活)
# 控制台静态页豁免 (批次26 补 /call): 页面本身无数据, 启用 API key 时页面才能
# 加载; 页面 JS 发起的 API 调用仍走鉴权 — 与 /pro /easy 语义一致。
# 批次90: /readyz 与 /health 同档 — 容器 HEALTHCHECK 不带 key, 只回就绪态。
EXEMPT_PATHS = {"/api/adr/v1/health", "/api/adr/v1/readyz",
                "/", "/pro", "/easy", "/call", "/settings"}

# 各页语言包 /i18n-<page>.js: 严格只放行 console.py 真正提供的白名单形态,
# 不接受 /i18n-../x.js、/i18n-a/b.js 之类的任意后缀 (深审 A9)。
_I18N_PAGE_JS = re.compile(r"/i18n-(index|easy|pro|call|settings)\.js")


def _is_exempt(path) -> bool:
    """鉴权豁免判定: 固定豁免路径 + i18n/theme 运行时脚本。

    i18n 脚本 (/i18n.js, /i18n-<page>.js) 是纯静态语言包 (console.py 以
    FileResponse 提供), 本身无数据。各控制台页面用 ``<script src="/i18n.js">``
    加载它们, 而浏览器对 <script src> 不会附带 Authorization/X-API-Key 头,
    页面 URL 上的 ?api_key 也不会带进脚本子请求。故启用 API key 后若不豁免,
    这些脚本一律 401 → 全局 ``I`` 未定义 → 页面主脚本 ``I.init`` 抛错 →
    四个控制台整体瘫痪。放行它们不泄露任何业务数据。

    批次66: /theme.js 同理 — 控制台页面 (call/easy/index/pro.html) 均以
    ``<script src="/theme.js">`` 加载它 (console.py 以 FileResponse 提供),
    纯静态、无业务数据。启用 API key 后若不豁免, 脚本 401 → ``window.adrTheme``
    未定义 → 主题开关消失、data-theme 不生效, 页面退回默认配色。放行无数据风险。
    """
    if path in EXEMPT_PATHS:
        return True
    if path == "/i18n.js" or _I18N_PAGE_JS.fullmatch(path or ""):
        return True
    if path == "/theme.js":
        return True
    # 统一外壳层的 CSS/JS (app-shell.css|js): 以 <link>/<script> 子请求加载,
    # 浏览器不带 key; 纯静态无业务数据, 与 theme.js 同档豁免。
    if path in ("/app-shell.css", "/app-shell.js"):
        return True
    return False


class APIKeyMiddleware:
    def __init__(self, app, api_keys=None):
        self.app = app
        # 空集 = 鉴权关闭 (不设 ADR_TTS_API_KEY 即开放)
        self.api_keys = {k for k in (api_keys or ()) if k}

    def _key_ok(self, key) -> bool:
        """恒定时间比较 (Track B): 逐个 compare_digest, 防时序侧信道枚举。

        批次41a: 两侧统一 encode 成 bytes 再比 — compare_digest 对 str
        遇非 ASCII 会抛 TypeError (异常路径同时泄漏比较失败信息),
        bytes 口径对任意凭据输入都恒定时间且不抛错。
        """
        if not key:
            return False
        kb = key.encode("utf-8", "surrogateescape")
        return any(hmac.compare_digest(kb, k.encode("utf-8", "surrogateescape"))
                   for k in self.api_keys)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            # Track B 收口: WS 连接此前完全绕过鉴权。未配置 key 时维持
            # 完全放行 (本地开放语义 / NEKO 零改造兼容红线)。
            if (not self.api_keys or _is_exempt(scope.get("path"))):
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
                or _is_exempt(scope.get("path"))):
            await self.app(scope, receive, send)
            return

        if self._key_ok(self._extract_key(scope)):
            await self.app(scope, receive, send)
            return

        # 批次84 (#13): 与 HTTPException / v2/v3 统一错误契约 — 三个字段齐全,
        # 客户端按 message 或 detail 取文案都不会拿到空值。
        body = json.dumps({"detail": "unauthorized", "message": "unauthorized",
                           "code": UNAUTHORIZED}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json; charset=utf-8")],
        })
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _extract_key(scope):
        """按 Bearer 头 → X-API-Key 头 → (默认关闭的) 查询参数取 key。

        查询参数通道默认关闭: ``?api_key=`` 会随 Referer、浏览器历史、
        反代/容器 access log 长期留存 (凭据易被翻出)。控制台页面已改为
        首屏从 URL 读一次后转存 sessionStorage + ``history.replaceState``
        抹掉地址栏, 后续请求一律走 ``X-API-Key`` 头。仅当部署方显式设置
        ``ADR_ALLOW_QUERY_API_KEY=1`` 时才回退到查询参数 (兼容外部调用方)。
        WS 不走这里 — 见 ``_extract_ws_key`` (浏览器 WS 握手无法带自定义头)。
        """
        headers = dict(scope.get("headers") or [])  # ASGI 头键为小写字节
        auth = headers.get(b"authorization")
        if auth:
            parts = auth.decode("latin-1").split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
        api_key = headers.get(b"x-api-key")
        if api_key:
            return api_key.decode("latin-1").strip()
        if allow_query_api_key():
            return _query_key(scope, b"api_key")
        return None

    @staticmethod
    def _extract_ws_key(scope):
        """WS 凭据 (Track B): query token= → (受开关约束的) query api_key= → 子协议首元素。

        子协议通道: 客户端 ``WebSocket(url, ["<key>"])``, 取首元素为 key,
        由 v3 端点在 accept 时回显 (浏览器要求回显选中子协议, 否则握手失败)。

        ``token=`` 是 v3 协议固有的握手通道 (N.E.K.O 的 http→ws 转换会透传 query),
        无法改为请求头, 保留但会进入访问日志 —— 部署方需在反代侧脱敏 query。
        ``api_key=`` 与 HTTP 侧同政策: 仅 ``ADR_ALLOW_QUERY_API_KEY=1`` 时接受,
        避免两套入口对同一凭据给出相反的暴露策略。
        """
        key = _query_key(scope, b"token")
        if key:
            return key
        if allow_query_api_key():
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
