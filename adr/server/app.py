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

from adr.server import console, native, v2_compat
from adr.server.auth import APIKeyMiddleware


def _prewarm_gsv(default_profile: str | None) -> None:
    """后台预热 GSV 引擎 (控制台/裸服务与 webui._prewarm_engines 同思路)。

    没有预热时, 桌面壳拉起服务后的**首次合成**要现场加载全套模型
    (实测 ~1 分钟, 用户感知为"合成巨慢/显卡没动")。预热后服务起来
    几十秒内引擎就绪, 用户打开页面输入文字期间加载已完成。
    失败不致命 (首次合成时仍会按需加载), 只打日志。
    """
    import logging

    from adr.models import gsv_engine

    log = logging.getLogger("adr.server.prewarm")
    gsv_engine._LOADING = True
    gsv_engine._STAGE = "queued"
    try:
        gsv_engine._STAGE = "importing"  # import torch + GSV 模块 (最耗时可达 20s)
        from adr.models.voice_library import list_voices, load_voice

        prof_name = default_profile
        if not prof_name:
            voices = list_voices()
            prof_name = voices[0] if voices else None
        kw = {}
        prof = None
        if prof_name:
            try:
                prof = load_voice(prof_name)
                kw = {"vits_weights": prof.get("vits_weights"),
                      "t2s_weights": prof.get("t2s_weights")}
                log.info("[prewarm] 用音色档案「%s」的权重预热", prof_name)
            except Exception:
                prof = None  # 档案损坏 → 退回纯预训练权重
        gsv_engine._STAGE = "loading"    # 权重加载进显存
        gsv_engine.get_gsv_engine().warmup(**{k: v for k, v in kw.items() if v})
        log.info("[prewarm] GPT-SoVITS 引擎就绪")
        # kernel JIT 预热: 否则首次合成再付 ~14s CUDA 编译 (只取首块)
        try:
            gsv_engine._STAGE = "kernel"
            if prof:
                for _ in gsv_engine.get_gsv_engine().synthesize_stream(
                        "引擎预热。", prof["ref_audio"],
                        vits_weights=kw.get("vits_weights"),
                        split_method="cut0"):
                    break
                log.info("[prewarm] kernel 预热完成, 首次合成即秒级")
        except Exception as e:
            log.warning("[prewarm] kernel 预热失败 (不影响功能): %s", e)
        gsv_engine._STAGE = "ready"
    except Exception as e:
        gsv_engine._STAGE = "failed"
        log.warning("[prewarm] GSV 预热失败 (首次合成时将现场加载): %s", e)
    finally:
        gsv_engine._LOADING = False


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
    app.state.engine = engine  # None → 请求期惰性 get_gsv_engine()
    app.state.default_profile = default_profile
    app.include_router(v2_compat.router)
    app.include_router(native.router)
    app.include_router(console.router)  # 批次14: 控制台 API (状态/训练/模型)
    console.register_pages(app)         # 批次14: 静态控制台页 (/, /pro, /easy)
    return app
