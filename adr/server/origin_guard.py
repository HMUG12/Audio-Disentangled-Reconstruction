"""S2: 浏览器来源守卫 (纯 ASGI) — 关上「任意网页偷调本机服务」的门。

背景: 默认不设 ADR_TTS_API_KEY 时服务不鉴权 (N.E.K.O 零改造兼容红线)。
此时浏览器里任何网页都能对 127.0.0.1:9881 发请求:
- 跨站请求伪造: 表单 POST / multipart 上传属于"简单请求", 不经 CORS 预检
  就直达服务端 (CORS 只管浏览器能否读响应, 不阻止请求本身生效);
  WebSocket 握手也完全不受 CORS 约束。
- DNS 重绑定: 恶意域名先解析到攻击者服务器, 再改解析到 127.0.0.1,
  浏览器视其为同源, 可读全部响应 — 此时请求的 Host 头是恶意域名。

规则 (仅在**未配置 API Key** 时生效; 配置 key 后由鉴权兜底, 不额外设限,
以免破坏反向代理/自定义域名部署):
1. Host 检查: Host 必须是 IP 字面量 (127.0.0.1 / 192.168.x.x / ::1 ...)、
   单段主机名 (localhost / 局域网机器名 / Docker 服务名, 不含点), 或
   ``ADR_ALLOWED_HOSTS`` 列出的主机名 (如 mypc.local)。DNS 重绑定必须用
   攻击者可控的公网域名 (含点), 上述三类无法被操控 — 局域网 IP 直连、
   Docker 端口映射 (Host=127.0.0.1) 不受影响。缺 Host 头 (非浏览器) 放行。
2. Origin 检查: 请求带 Origin 头 (浏览器才带) 时, 必须是同源 (Origin 的
   host:port 与 Host 头一致, 即控制台页面自身) 或在白名单内
   (见 settings.cors_origins: Tauri 壳 / 本机 localhost 页 / ADR_CORS_ORIGINS)。
   不带 Origin 的请求 (N.E.K.O 后端 / curl / Python 客户端) 一律放行。
3. 有副作用接口 (批次77): `<img>` / 顶层导航 / no-cors fetch 等浏览器发起
   的跨站 GET 不带 Origin, 上面两条拦不住 — 这类路由额外挂
   ``require_browser_safe`` 依赖, 按 Sec-Fetch-Site 判定 (见该函数)。

逃生阀: ``ADR_ALLOWED_HOSTS=*`` 关闭 Host 检查; ``ADR_CORS_ORIGINS=*``
恢复旧的 CORS 全开并关闭 Origin 检查。
"""
from __future__ import annotations

import ipaddress
import re

from fastapi import HTTPException, Request

# 本机来源: 任意端口的 localhost / 127.0.0.1 / [::1] 页面 (本机软件, 如
# N.E.K.O 前端若改为浏览器直连)。Tauri 壳来源见 settings.TAURI_ORIGINS。
LOCAL_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"
_LOCAL_ORIGIN_RE = re.compile(LOCAL_ORIGIN_REGEX)


def split_host(host_header: str) -> str:
    """``host[:port]`` → 小写 host (IPv6 去方括号)。"""
    h = host_header.strip().lower()
    if h.startswith("["):
        end = h.find("]")
        return h[1:end] if end > 0 else h
    if h.count(":") == 1:
        return h.split(":", 1)[0]
    return h


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def host_allowed(host_header: str | None, extra_hosts: set[str]) -> bool:
    if not host_header:
        return True  # 非浏览器 (HTTP/1.0 等); 浏览器恒带 Host
    host = split_host(host_header).rstrip(".")
    # 单段主机名 (无点: localhost / 局域网机器名 / Docker 服务名) 只能由本地
    # 解析, 公网 DNS 无法重绑定到它们 — 放行; 带点域名需显式列入白名单
    return ("." not in host or _is_ip(host) or host in extra_hosts)


def origin_allowed(origin: str, host_header: str | None,
                   allowed_origins: set[str]) -> bool:
    origin = origin.strip()
    low = origin.lower()
    if low in allowed_origins or _LOCAL_ORIGIN_RE.fullmatch(low):
        return True
    # 同源: Origin = scheme://host[:port], 与 Host 头逐字比较 netloc
    if host_header and "://" in low:
        netloc = low.split("://", 1)[1]
        return netloc == host_header.strip().lower()
    return False  # 含 "null" (沙箱 iframe / file://) 等


def configured_allowed_origins() -> set[str]:
    """当前静态白名单 (Tauri 壳 + ADR_CORS_ORIGINS), 小写化。"""
    from adr.core import settings
    return {o.strip().lower()
            for o in (*settings.TAURI_ORIGINS, *settings.cors_origins())
            if o.strip() and o.strip() != "*"}


def any_origin_allowed() -> bool:
    """``ADR_CORS_ORIGINS=*`` 逃生阀 — 恢复旧的全开语义。"""
    from adr.core import settings
    return "*" in settings.cors_origins()


def cross_site_allowed(sec_fetch_site: str | None, origin: str | None,
                       host_header: str | None, allowed: set[str],
                       allow_any_origin: bool = False) -> bool:
    """有副作用接口的浏览器跨站判定 (批次77 / CODE_REVIEW #4)。

    中间件的 Origin 检查只能看见"带 Origin 的跨站"; 浏览器 `<img>` /
    顶层导航 / ``{mode: "no-cors"}`` fetch 的跨站 GET 一律不带 Origin,
    可直接触发换权 / doctor 修复这类有副作用的接口。这里改用
    Sec-Fetch-Site (Chrome 76+ / Firefox 90+ / Safari 16.4+ 均发送):

    - 缺 Sec-Fetch-Site = 非浏览器客户端 (N.E.K.O / curl / SDK) → 放行,
      保住零改造兼容红线;
    - same-origin / same-site / none → 放行 (控制台自身 / 本机页面 / 地址栏);
    - cross-site → 必须带白名单内的 Origin 才放行 (Tauri 壳
      ``tauri.localhost`` → ``127.0.0.1`` 的合法跨站 fetch), 否则拒。
    """
    site = (sec_fetch_site or "").strip().lower()
    if not site:
        return True
    if site != "cross-site":
        return True
    if allow_any_origin:
        return True
    return bool(origin) and origin_allowed(origin, host_header, allowed)


async def require_browser_safe(request: Request) -> None:
    """路由依赖: 有副作用接口拒绝浏览器跨站触发 (批次77)。

    与中间件 enabled 语义对齐: 配了 ``ADR_TTS_API_KEY`` 时由鉴权兜底
    (跨站请求带不上 key), 这里直接放行 — 反代 / 自定义域名部署不被误伤。
    """
    from adr.core import settings
    if settings.api_keys():
        return
    headers = request.headers
    if not cross_site_allowed(headers.get("sec-fetch-site"),
                              headers.get("origin"),
                              headers.get("host"),
                              configured_allowed_origins(),
                              any_origin_allowed()):
        raise HTTPException(
            status_code=403,
            detail="forbidden: cross-site request to side-effecting endpoint")


class OriginGuardMiddleware:
    def __init__(self, app, enabled=True, allowed_origins=(),
                 allowed_hosts=(), allow_any_origin=False,
                 allow_any_host=False):
        self.app = app
        self.enabled = enabled
        self.allowed_origins = {o.strip().lower() for o in allowed_origins if o}
        self.allowed_hosts = {h.strip().lower() for h in allowed_hosts if h}
        self.allow_any_origin = allow_any_origin
        self.allow_any_host = allow_any_host

    async def __call__(self, scope, receive, send):
        if not self.enabled or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        host = headers.get(b"host")
        host = host.decode("latin-1") if host else None
        origin = headers.get(b"origin")
        reason = None
        if not self.allow_any_host and not host_allowed(host, self.allowed_hosts):
            reason = "host not allowed"
        elif (origin is not None and not self.allow_any_origin
              and not origin_allowed(origin.decode("latin-1"), host,
                                     self.allowed_origins)):
            reason = "origin not allowed"
        if reason is None:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4403})
            return
        body = ('{"message": "forbidden: %s"}' % reason).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [(b"content-type", b"application/json; charset=utf-8")],
        })
        await send({"type": "http.response.body", "body": body})
