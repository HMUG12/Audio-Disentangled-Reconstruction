"""应用工厂 (批次11): 组装 GSV api_v2 兼容层 + ADR 原生 API。

用法:
    from adr.server import create_app
    app = create_app()                    # 生产: 首请求时惰性建 GSV 引擎
    app = create_app(engine=fake_engine)  # 测试注入

环境变量:
    ADR_TTS_DEFAULT_PROFILE  默认音色档案名 (ref_audio_path 缺失时回退)
    ADR_TTS_API_KEY          API Key, 逗号分隔多个; 未设置则不鉴权 (默认)
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adr.server import native, v2_compat
from adr.server.auth import APIKeyMiddleware


def create_app(engine=None) -> FastAPI:
    app = FastAPI(
        title="ADR TTS Server",
        version="1.0",
        description="ADR 低资源声音克隆 — GSV api_v2 兼容层 + ADR 原生档案 API",
    )
    # 鉴权 (批次12): 先加 = 内层; 未设 ADR_TTS_API_KEY 时完全放行。
    # CORS 需在外层: 预检 OPTIONS 免 key, 401 响应也带 CORS 头 (浏览器可读)。
    api_keys = [k.strip() for k in os.environ.get("ADR_TTS_API_KEY", "").split(",")]
    app.add_middleware(APIKeyMiddleware, api_keys=api_keys)
    # CORS 全开 (与 api_neko 一致): N.E.K.O 前端 / 浏览器插件直连
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine  # None → 请求期惰性 get_gsv_engine()
    app.state.default_profile = os.environ.get("ADR_TTS_DEFAULT_PROFILE") or None
    app.include_router(v2_compat.router)
    app.include_router(native.router)
    return app
