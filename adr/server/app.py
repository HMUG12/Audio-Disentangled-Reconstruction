"""应用工厂 (批次11): 组装 GSV api_v2 兼容层 + ADR 原生 API。

用法:
    from adr.server import create_app
    app = create_app()                    # 生产: 启动即后台预热 GSV 引擎
    app = create_app(engine=fake_engine)  # 测试注入

环境变量:
    ADR_TTS_DEFAULT_PROFILE  默认音色档案名 (ref_audio_path 缺失时回退)
    ADR_TTS_API_KEY          API Key, 逗号分隔多个; 未设置则不鉴权 (默认)
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adr.server import console, native, openai_compat, v2_compat, v3_compat
from adr.server.auth import APIKeyMiddleware


def _prewarm_gsv(default_profile: str | None) -> None:
    """后台预热 GSV 引擎 — 委托 gsv_engine.prewarm (批次26)。

    没有预热时, 桌面壳拉起服务后的**首次合成**要现场加载全套模型
    (实测 ~1 分钟, 用户感知为"合成巨慢/显卡没动")。预热后服务起来
    几十秒内引擎就绪, 用户打开页面输入文字期间加载已完成。
    阶段状态机 (_LOADING/_STAGE) 内聚在 gsv_engine 模块内部管理。
    """
    from adr.models import gsv_engine

    gsv_engine.prewarm(default_profile)


def create_app(engine=None) -> FastAPI:
    default_profile = os.environ.get("ADR_TTS_DEFAULT_PROFILE") or None

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # 引擎注入 (测试) 时跳过预热; 生产启动即后台加载, 合成请求随时可进
        # (引擎锁会阻塞到预热完成, 保证一致性)
        if engine is None:
            threading.Thread(target=_prewarm_gsv, args=(default_profile,),
                             daemon=True, name="adr-prewarm").start()
        yield

    app = FastAPI(
        title="ADR TTS Server",
        version="1.0",
        description="ADR 低资源声音克隆 — GSV api_v2 兼容层 + ADR 原生档案 API",
        lifespan=_lifespan,
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
    return app
