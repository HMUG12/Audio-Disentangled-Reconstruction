"""应用工厂 (批次11): 组装 GSV api_v2 兼容层 + ADR 原生 API。

用法:
    from adr.server import create_app
    app = create_app()                    # 生产: 启动即后台预热 GSV 引擎
    app = create_app(engine=fake_engine)  # 测试注入

环境变量:
    ADR_TTS_DEFAULT_PROFILE  默认音色档案名 (ref_audio_path 缺失时回退)
    ADR_TTS_API_KEY          API Key, 逗号分隔多个; 未设置则不鉴权 (默认)
    ADR_CORS_ORIGINS         额外允许的浏览器来源, 逗号分隔; * = 全开 (S2)
    ADR_ALLOWED_HOSTS        额外允许的 Host 主机名, 逗号分隔; * = 不检查 (S2)
    (解析单源: adr/core/settings.py, 批次36 — 本文件不再裸读 os.environ)
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from adr.core import settings
from adr.server import console, native, openai_compat, v2_compat, v3_compat
from adr.server.auth import APIKeyMiddleware
from adr.server.exceptions import (
    PAYLOAD_TOO_LARGE,
    VALIDATION_ERROR,
    http_error_code,
    json_error,
)
from adr.server.origin_guard import LOCAL_ORIGIN_REGEX, OriginGuardMiddleware


def _prewarm_gsv(default_profile: str | None) -> None:
    """后台预热 GSV 引擎 — 委托 gsv_engine.prewarm (批次26)。

    没有预热时, 桌面壳拉起服务后的**首次合成**要现场加载全套模型
    (实测 ~1 分钟, 用户感知为"合成巨慢/显卡没动")。预热后服务起来
    几十秒内引擎就绪, 用户打开页面输入文字期间加载已完成。
    阶段状态机 (_LOADING/_STAGE) 内聚在 gsv_engine 模块内部管理。
    """
    from adr.models import gsv_engine

    gsv_engine.prewarm(default_profile)


def _detail_message(detail) -> str:
    """HTTPException.detail → 统一 ``message`` 文本 (批次84 / CODE_REVIEW #13)。"""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, (list, tuple)):
        return "; ".join(str(x) for x in detail)
    return str(detail)


def _validation_message(errors: list) -> str:
    """RequestValidationError.errors() → 人类可读摘要 (最多取前 3 条)。"""
    parts = []
    for err in errors[:3]:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        msg = err.get("msg") or "invalid value"
        parts.append(f"{loc}: {msg}" if loc else str(msg))
    text = "; ".join(parts) or "invalid request"
    if len(errors) > 3:
        text += f"; ... ({len(errors) - 3} more)"
    return text


class _BodyTooLarge(BaseException):
    """ASGI receive 累计超限信号 (批次88 / SEC-06)。

    继承 BaseException 而非 Exception: FastAPI 的 body 解析会用
    ``except Exception`` 兜底转成 400 "error parsing the body", 普通异常
    会被吞掉; BaseException 才能穿透到本中间件改写为 413。
    """


class BodySizeLimitMiddleware:
    """非 multipart 请求体上限 (批次88 / SEC-06)。

    ``await request.json()`` 在参数校验之前完成, 而 Starlette 本身没有 body
    上限 — 单个数百 MB 的 JSON 即可打爆内存。这里在 ASGI receive 层:
    1. Content-Length 已声明且超限 → 直接 413 (不读 body);
    2. 无 Content-Length (chunked) → 逐块累计, 越限抛 ``_BodyTooLarge``。

    multipart 上传 (档案包) 由 ``native._CappedImportRoute`` 按
    ``ADR_PROFILE_IMPORT_MAX_MB`` 单独限制, 本中间件放行。
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in (scope.get("headers") or [])}
        ctype = headers.get(b"content-type", b"").lower()
        if b"multipart/form-data" in ctype:
            return await self.app(scope, receive, send)

        declared = headers.get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            return await self._reject(send)

        seen = 0
        started = False

        async def _limited_receive():
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.max_bytes:
                    raise _BodyTooLarge
            return msg

        async def _send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, _limited_receive, _send)
        except _BodyTooLarge:
            if not started:   # 响应已开始 (极少见) 时无法再改状态码
                await self._reject(send)

    async def _reject(self, send):
        resp = json_error(
            413,
            f"请求体超过 {self.max_bytes // (1 << 20)}MB 上限 (ADR_MAX_BODY_MB)",
            PAYLOAD_TOO_LARGE,
            extra={"detail": "request body too large"})
        await resp({"type": "http"}, None, send)


class JSONCharsetMiddleware:
    """JSON 响应统一声明 ``charset=utf-8`` (批次96)。

    FastAPI 默认的 ``application/json`` 不带 charset; Windows PowerShell 5.1
    的 ``Invoke-RestMethod`` 在缺 charset 时按 ISO-8859-1 解码, 中文直接乱码
    (实测 ``/system/stats`` 的 sched.runtime reason)。这里只改响应头、不动
    body: 非 JSON (音频 / HTML / 纯文本) 与已声明 charset 的响应零影响。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def _send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                for i, (k, v) in enumerate(headers):
                    if k.lower() != b"content-type":
                        continue
                    ctype = v.decode("latin-1")
                    base = ctype.split(";", 1)[0].strip().lower()
                    if base == "application/json" and "charset=" not in ctype.lower():
                        headers[i] = (k, f"{ctype}; charset=utf-8".encode("latin-1"))
                    break
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, _send)


class SecurityHeadersMiddleware:
    """基础安全响应头 (批次112 / 审计 R5-SEC-01)。

    服务面向本机浏览器与壳内 WebView, 缺这三类头时: MIME 嗅探可把上传的
    ``.txt`` 当脚本执行、外链 Referer 会带出本机控制台地址、控制台页可被
    第三方站点用 iframe 套壳点击劫持。只加头不改 body, 已存在同名头的响应
    (路由自己设的 CSP) 一律不覆盖。
    """

    _HEADERS = (
        (b"x-content-type-options", b"nosniff"),
        (b"referrer-policy", b"no-referrer"),
        (b"x-frame-options", b"DENY"),
        (b"content-security-policy", b"frame-ancestors 'none'"),
    )

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def _send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                existing = {k.lower() for k, _ in headers}
                headers.extend((k, v) for k, v in self._HEADERS
                               if k not in existing)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, _send)


def create_app(engine=None) -> FastAPI:
    default_profile = settings.default_profile()

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # 引擎注入 (测试) 时跳过预热; 生产启动即后台加载, 合成请求随时可进
        # (引擎锁会阻塞到预热完成, 保证一致性)
        if engine is None:
            threading.Thread(target=_prewarm_gsv, args=(default_profile,),
                             daemon=True, name="adr-prewarm").start()
            # 批次90 (R3-LIFE-01): 兜底 — 非 uvicorn 优雅退出路径 (异常/直接
            # 退出) 也要回收训练/门禁/下载子进程, 否则孤儿进程继续占 GPU。
            atexit.register(console.stop_background_procs)
        try:
            yield
        finally:
            # 批次90 (R3-LIFE-01): 控制台起的训练/门禁/下载子进程都是
            # start_new_session 独立进程组, 父进程退出不传导 — 必须显式回收。
            try:
                await asyncio.to_thread(console.stop_background_procs)
            except Exception:
                logging.getLogger("adr").warning(
                    "shutdown: 回收后台子进程失败", exc_info=True)

    app = FastAPI(
        title="ADR TTS Server",
        version="1.0",
        description="ADR 低资源声音克隆 — GSV api_v2 兼容层 + ADR 原生档案 API",
        lifespan=_lifespan,
    )
    # 批次88 (SEC-06): 非 multipart 请求体上限, 最先加 = 最内层 —
    # CORS/守卫在外层, 413 也带 CORS 头; 大 body 不会先被读进内存。
    app.add_middleware(BodySizeLimitMiddleware,
                       max_bytes=settings.max_body_bytes())
    # 鉴权 (批次12): 先加 = 内层; 未设 ADR_TTS_API_KEY 时完全放行。
    # CORS 需在外层: 预检 OPTIONS 免 key, 401 响应也带 CORS 头 (浏览器可读)。
    api_keys = settings.api_keys()
    app.add_middleware(APIKeyMiddleware, api_keys=api_keys)
    # S2: CORS 白名单 (原为 "*", 任意网页可跨站读写本机服务)。
    # 允许: Tauri 壳 + 本机 localhost 页面 + ADR_CORS_ORIGINS; "*" 恢复全开。
    # 控制台页同源, 不经 CORS。N.E.K.O 后端 / curl 不带 Origin, 不受影响。
    extra_origins = settings.cors_origins()
    any_origin = "*" in extra_origins
    if any_origin:
        cors_kwargs = {"allow_origins": ["*"]}
    else:
        cors_kwargs = {
            "allow_origins": [*settings.TAURI_ORIGINS, *extra_origins],
            "allow_origin_regex": LOCAL_ORIGIN_REGEX,
        }
    # 来源守卫: 未设 API Key 时拦截跨站请求 (CORS 拦不住表单/WS) 与 DNS 重绑定;
    # 设了 key 由鉴权兜底, 守卫关闭 (反代/自定义域名部署不受限)。
    # 加在 CORS 之前 = 位于 CORS 内层: 403 也带 CORS 头, 白名单来源可读。
    extra_hosts = settings.allowed_hosts()
    app.add_middleware(
        OriginGuardMiddleware,
        enabled=not api_keys,
        allowed_origins=[*settings.TAURI_ORIGINS, *extra_origins],
        allowed_hosts=extra_hosts,
        allow_any_origin=any_origin,
        allow_any_host="*" in extra_hosts,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_methods=["*"],
        allow_headers=["*"],
        **cors_kwargs,
    )
    # 批次96: JSON 响应补 charset=utf-8 — 放最外层, 鉴权/守卫/413 等
    # 中间件直接产出的 JSON 也一并覆盖 (它们本就带 charset, 命中即跳过)。
    app.add_middleware(JSONCharsetMiddleware)
    # 批次112 (R5-SEC-01): 基础安全响应头 — 放最外层, 401/403/413/404 等
    # 中间件与异常处理器直接产出的响应也一并覆盖。
    app.add_middleware(SecurityHeadersMiddleware)
    # 批次41a: HTTPException 响应增量补结构化 code — 既有 detail 字段与
    # 文本逐字保留 (wire 兼容)。
    # 批次84 (#13): 再增量补 ``message``, 与 v2/v3 的 {"message","code"} 契约对齐;
    # 同时接管 422 (RequestValidationError), 补 message + code,
    # detail 仍是 FastAPI 原有 errors 列表 (只增不改)。
    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request, exc: StarletteHTTPException):
        detail = exc.detail
        return JSONResponse(
            status_code=exc.status_code, headers=exc.headers,
            content={"detail": detail, "message": _detail_message(detail),
                     "code": http_error_code(exc)})

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request, exc: RequestValidationError):
        errors = exc.errors()
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(errors),
                     "message": _validation_message(errors),
                     "code": VALIDATION_ERROR})

    # v2/v3_compat 的请求期引擎缓存 (批次26 注明): 生产模式下初始为 None,
    # 首次合成时由 v2_compat/v3_compat 惰性赋值 get_gsv_engine() 并复用。
    # /health 就绪判据已改用 gsv_engine 模块级三态, 不读此字段 (批次22)。
    app.state.engine = engine
    app.state.default_profile = default_profile
    app.include_router(v2_compat.router)
    app.include_router(v3_compat.router)  # 批次19: N.E.K.O v3 面 (voices + stream-input WS)
    app.include_router(openai_compat.router)  # 批次21: OpenAI 兼容面 (/v1/audio/speech)
    app.include_router(native.router)
    app.include_router(console.router)  # 批次14: 控制台 API (状态/训练/模型)
    console.register_pages(app)         # 批次14: 静态控制台页 (/, /pro, /easy)
    assert_unique_routes(app)           # M3-b: native/console 共用前缀, 冲突即启动失败
    return app


def assert_unique_routes(app: FastAPI) -> None:
    """启动期路由冲突断言 (M3-b)。

    ``native`` 与 ``console`` 共用 ``/api/adr/v1`` 前缀。FastAPI 对同
    method+path 的重复注册不报错, 而是静默取先注册的那条 — 表现为"新接口
    莫名不生效"。装配末尾显式扫描, 冲突直接抛 ``RuntimeError`` 让服务起不来。
    """
    seen: dict[tuple[str, str], str] = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or ()
        for method in methods:
            key = (method.upper(), path)
            if key in seen:
                raise RuntimeError(
                    "duplicate route registration: "
                    f"{key[0]} {key[1]} ({seen[key]} vs {route.name})")
            seen[key] = str(route.name)
