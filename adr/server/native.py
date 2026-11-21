"""ADR 原生 API (批次11): /api/adr/v1 — 音色档案一等公民的对外接口。

面向新接入方 (以及 ADR 自己的插件/WebUI):
- /health                服务与引擎状态
- /readyz                就绪检查 (引擎可用且未永久降级; 供容器/编排器探活)
- /profiles              音色档案清单
- /profiles/{name}/ref   下载档案参考音频
- /tts  POST+GET         按档案名合成 (内部复用 v2 兼容层的核心处理)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel

from adr.models import voice_library
from adr.server import pathsafe
from adr.core.exceptions import EngineStalledError
from adr.server.exceptions import (
    EXPORT_FAILED,
    IMPORT_FAILED,
    INVALID_NAME,
    PAYLOAD_TOO_LARGE,
    PROFILE_NOT_FOUND,
    ROLLBACK_FAILED,
    json_error,
)
from adr.server.origin_guard import require_browser_safe
from adr.server.v2_compat import tts_handle

logger = logging.getLogger(__name__)

# 路由归属 (M3-b): 本模块只拥有 /api/adr/v1 下"音色档案一等公民"的接口
# (/health /readyz /profiles* /tts); 同前缀的 /system/* /train/* /models 等
# 属于 console.py。两个 router 由 app.py 的 assert_unique_routes 在启动期
# 断言不存在同 method+path 的重复注册。
router = APIRouter(prefix="/api/adr/v1", tags=["adr-native"])

_API_VERSION = "1.0"


def _basename(path):
    return Path(path).name if path else None


@router.get("/health", summary="服务状态")
async def health(request: Request):
    state = request.app.state
    # 真实就绪判据 (批次22): 原先判 app.state.engine is not None, 生产模式
    # 恒 false — N.E.KO 健康检查会误判服务不可用。改用引擎模块级状态
    # (不触发加载), 与 console 的 engine_stage 三态同源。
    from adr.models import gsv_engine as _gsv
    return {
        "status": "ok",
        "service": "adr-tts",
        "api_version": _API_VERSION,
        "engine": "adr",
        "engine_ready": _gsv.is_ready(),
        "engine_loading": _gsv.is_loading(),
        "engine_stage": _gsv.stage(),
        "engine_stage_text": _gsv.stage_text(),
        # 批次73: failed 阶段的原因摘要 (traceback 末几行), 桌面壳预热页展开显示
        "engine_error": _gsv.stage_error(),
        # 批次23: 并发排队可视化 (排队等待数 / 合成中数)
        "queue_depth": _gsv.queue_depth(),
        "synth_busy": _gsv.synth_busy(),
        "default_profile": getattr(state, "default_profile", None),
        # 批次66: /health 免鉴权 — 权重只回文件名, 不泄露服务器绝对路径/用户名
        "weights": {
            "t2s": _basename(getattr(state, "t2s_weights", None)),
            "vits": _basename(getattr(state, "vits_weights", None)),
        },
    }


@router.get("/readyz", summary="就绪检查 (引擎可用且未永久降级)")
async def readyz():
    """批次90 (R3-OBS-01): 真正反映"服务还能不能干活"。

    /health 是存活探针 (进程活着就是 200, 桌面壳启动探活依赖它), 卡死降级后
    仍会 200; 容器 HEALTHCHECK 只看状态码, 于是"永久 503 但始终健康"。
    这里 stalled / 未就绪一律 503, 让 Docker healthcheck 与编排器能识别
    "不可恢复"降级, 配合进程级看门狗 (ADR_ENGINE_WATCHDOG_SEC) 重启进程。

    与 /health 一样免鉴权 (容器内探针没有 API key), 不含任何路径信息。
    """
    from adr.models import gsv_engine as _gsv

    ready = _gsv.is_ready()
    stalled = _gsv.is_stalled()
    ok = ready and not stalled
    return JSONResponse({
        "status": "ok" if ok else "unavailable",
        "service": "adr-tts",
        "api_version": _API_VERSION,
        "engine_ready": ready,
        "engine_stalled": stalled,
        "engine_loading": _gsv.is_loading(),
        "engine_stage": _gsv.stage(),
        "degraded_seconds": round(_gsv.degraded_seconds(), 1),
    }, status_code=200 if ok else 503)


@router.post("/engine/restart", summary="重启合成引擎 (释放显存并重建模型)")
async def engine_restart(request: Request):
    """手动重启引擎: 丢弃已加载模型 + 释放显存, 后台重新预热。

    与进程级看门狗 (降级超时直接退出进程) 不同, 这里不杀进程 — 控制台轮询、
    训练子进程与前端连接都不中断, 只重建推理模型。引擎锁被卡死 worker 占着
    时返回 503, 由看门狗/Docker 重启兜底 (那种情况进程内已无法恢复)。
    """
    from adr.models import gsv_engine as _gsv

    if _gsv.is_loading():
        return json_error(409, "引擎正在加载或预热, 请稍后再试", "engine_busy")
    try:
        reset = await asyncio.to_thread(_gsv.reset_engine, "manual")
    except EngineStalledError as exc:
        return json_error(503, str(exc), "engine_stalled")
    threading.Thread(
        target=_gsv.prewarm,
        args=(getattr(request.app.state, "default_profile", None),),
        daemon=True,
        name="adr-engine-reload",
    ).start()
    logger.info("[engine] 手动重启: 已丢弃模型并开始重新预热")
    return {"ok": True, "reset": bool(reset), "reloading": True,
            "engine_stage": _gsv.stage()}


@router.get("/profiles", summary="音色档案清单")
async def profiles():
    out = []
    for name in voice_library.list_voices():
        try:
            meta = voice_library.load_voice(name)
        except Exception:
            # 批次66: 坏档案仍跳过 (行为不变), 但留日志便于定位
            logger.warning("profiles: load_voice(%r) failed, skipped", name,
                           exc_info=True)
            continue
        out.append({
            "name": name,
            "ref_audio": str(Path(voice_library.VOICES_DIR) / name / "ref.wav"),
            "prompt_text": meta.get("prompt_text", ""),
            "style": meta.get("style", ""),
            "language": meta.get("language", "zh"),   # 批次53: 语种标记
            "t2s_weights": meta.get("t2s_weights"),
            "vits_weights": meta.get("vits_weights"),
            "created_at": meta.get("created_at", ""),
            # 批次50: 版本化 — 现行版次 = 历史数 + 1; 退化标记供 UI 警示
            "last_sim": meta.get("last_sim"),
            "degraded": meta.get("degraded"),
            "versions": len(meta.get("weights_history") or []) + 1,
        })
    return {"profiles": out}


@router.get("/profiles/{name}/history", summary="档案权重版本历史 (批次50)")
async def profile_history(name: str):
    # 旧→新; 现行绑定不在 history 里 (见 /profiles 的 vits_weights)
    if not pathsafe.is_safe_name(name):
        return json_error(400, f"invalid profile name: {name}", INVALID_NAME)
    try:
        hist = voice_library.voice_history(name)
    except ValueError as e:
        return json_error(400, str(e), INVALID_NAME)
    return {"name": name, "history": hist}


@router.post("/profiles/{name}/rollback", summary="回滚到上一版权重 (批次50)")
async def profile_rollback(name: str):
    # 现行绑定入史, history 末版恢复现行; 连续调用 = 逐版回退
    if not pathsafe.is_safe_name(name):
        return json_error(400, f"invalid profile name: {name}", INVALID_NAME)
    try:
        meta = voice_library.rollback_voice(name)
    except ValueError as e:
        return json_error(400, str(e), ROLLBACK_FAILED)
    return {"name": name, "profile": meta}


@router.get("/profiles/{name}/ref", summary="下载档案参考音频")
async def profile_ref(name: str):
    # Track B 收口: name 先清洗再拼路径, ../ 穿越越出 VOICES_DIR 一律 404
    ref = pathsafe.resolve_within(voice_library.VOICES_DIR, Path(name) / "ref.wav")
    if ref is None or not ref.exists():
        # 批次35: code 增量字段 (与错误协议统一)
        return json_error(404, f"profile not found: {name}", PROFILE_NOT_FOUND)
    return FileResponse(str(ref), media_type="audio/wav", filename="ref.wav")


@router.get("/profiles/{name}/export", summary="导出音色档案包 (.adrvoice, 含权重)")
async def profile_export(name: str, request: Request):
    """导出 .adrvoice 包 (参考音频 + 绑定微调权重)。

    注意: 权重即声音本体, 包内含生物特征性质数据, 分享时自行斟酌。
    """
    import asyncio
    import os
    import tempfile

    from adr.models import voice_package
    from starlette.background import BackgroundTask

    if not pathsafe.is_safe_name(name):
        return json_error(400, f"invalid profile name: {name}", INVALID_NAME)
    # 批次66: 404 由档案存在性直接判定, 不再匹配异常文案 "不存在"
    if name not in voice_library.list_voices():
        return json_error(404, f"档案不存在: {name}", EXPORT_FAILED)
    # P1: 导出模型包二次验证 (每次导出都验, 不在会话内免验)
    from adr.core import settings_store
    from adr.server import local_account
    if settings_store.get_settings()["security"]["protect_export"]:
        pwd = request.headers.get("x-adr-password", "")
        if not pwd:
            return json_error(403, "导出模型包需要本机账号密码", "password_required")
        if not local_account.verify_password(pwd):
            return json_error(403, "本机账号密码不正确", "password_invalid")
    fd, tmp = tempfile.mkstemp(suffix=".adrvoice")
    os.close(fd)
    try:
        await asyncio.to_thread(voice_package.export_voice, name, tmp)
    except ValueError as e:
        os.unlink(tmp)
        return json_error(400, str(e), EXPORT_FAILED)
    except BaseException:
        # 批次65 导出回滚: 非 ValueError (OSError/取消等) 也删掉含声纹权重的临时包
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return FileResponse(tmp, media_type="application/zip",
                        filename=f"{name}.adrvoice",
                        background=BackgroundTask(os.unlink, tmp))


def _import_max_bytes() -> int:
    """ADR_PROFILE_IMPORT_MAX_MB — 档案包上传上限 (批次66, 默认 4096MB; 含权重的正常包远小于此)。"""
    try:
        mb = int(os.environ.get("ADR_PROFILE_IMPORT_MAX_MB", 4096))
    except ValueError:
        mb = 4096
    return (mb if mb > 0 else 4096) * (1 << 20)


def _import_too_large(cap: int) -> JSONResponse:
    return json_error(
        413,
        f"档案包超过 {cap // (1 << 20)}MB 上限 (ADR_PROFILE_IMPORT_MAX_MB)",
        PAYLOAD_TOO_LARGE)


class _ImportTooLarge(HTTPException):
    """批次66: 继承 HTTPException — FastAPI 解析 body 时原样上抛 (不被包成 400)。"""


class _CappedImportRoute(APIRoute):
    """批次66: 在 multipart 解析 (会先整包落临时盘) 之前卡上传大小。

    Content-Length 超限直接 413; 无 Content-Length (chunked) 则逐块计数,
    越限即中断解析 — 临时盘占用不超过上限 + 一块。
    """

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def capped(request: Request):
            cap = _import_max_bytes()
            cl = request.headers.get("content-length", "")
            if cl.isdigit() and int(cl) > cap:
                return _import_too_large(cap)
            receive, seen = request.receive, 0

            async def limited_receive():
                nonlocal seen
                msg = await receive()
                if msg["type"] == "http.request":
                    seen += len(msg.get("body", b""))
                    if seen > cap:
                        raise _ImportTooLarge(status_code=413)
                return msg

            try:
                return await handler(Request(request.scope, limited_receive))
            except _ImportTooLarge:
                return _import_too_large(cap)

        return capped


async def profile_import(package: UploadFile = File(...),
                         name: str = Form(None),
                         overwrite: bool = Form(False),
                         force: bool = Form(False)):
    """导入 .adrvoice 包; 权重落位 GSV/RVC 标准目录并过 validate_weights_file。"""
    import asyncio
    import os
    import tempfile
    import zipfile as _zipfile

    from adr.models import voice_package
    from starlette.background import BackgroundTask

    suffix = Path(package.filename or "import.adrvoice").suffix or ".adrvoice"
    cap = _import_max_bytes()
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        size = 0
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await package.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > cap:
                    # 批次66: 二道防线 (主防线在 _CappedImportRoute, 解析前即拦)
                    return _import_too_large(cap)
                out.write(chunk)
        # 批次88 (SEC-05): force 仅在覆盖会被其它档案共用时必需
        r = await asyncio.to_thread(voice_package.import_voice, tmp,
                                    name, overwrite, None, force)
    except (ValueError, _zipfile.BadZipFile) as e:
        return json_error(400, str(e), IMPORT_FAILED)
    finally:
        os.unlink(tmp)
    # v1.3.x: 导入首个档案即补跑本机采样, 新用户不必等下次启动
    from adr.models.gsv_engine import try_schedule_profile
    try_schedule_profile()
    from adr.core.audit import log_event, text_fp
    # 审计脱敏: 档案名可能含人名, 只记指纹 (与 voice_enroll 同口径)
    log_event("voice_import",
              name_fp=text_fp(name) if name else None, overwrite=overwrite)
    return {"imported": r}


# 批次66: 用 add_api_route 挂自定义 route 类 (装饰器不支持 route_class_override)
router.add_api_route("/profiles/import", profile_import, methods=["POST"],
                     summary="导入音色档案包 (.adrvoice)",
                     route_class_override=_CappedImportRoute)


class NativeTTS_Request(BaseModel):
    """原生合成请求: 档案名必填, 其余可选覆盖档案默认值。"""
    text: str
    profile: str
    prompt_text: Optional[str] = None
    # 批次53: 语言可省 → 默认联动档案语种 (无档案时兜底 zh, 见 _resolve_profile)
    text_lang: Optional[str] = None
    prompt_lang: Optional[str] = None
    speed_factor: float = 1.0
    seed: int = -1
    media_type: str = "wav"        # wav/raw/ogg/aac
    streaming_mode: bool = False   # True → 分段流 (首块 WAV 头 + 裸 PCM)
    t2s_weights: Optional[str] = None
    vits_weights: Optional[str] = None
    # 采样参数 (批次14E 控制台高级面板透传)
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut1"  # cut0 不切 / cut1 凑四句 / cut3 按句 / cut5 按标点


def _native_to_v2(body: NativeTTS_Request) -> dict:
    """原生请求 → v2 核心 req (streaming_mode 映射: True → 分支 1 分段流)。

    批次53: 语言 None 保留为 None — _resolve_profile 按档案语种回填 (默认 zh)。
    """
    return {
        "text": body.text,
        "text_lang": body.text_lang.lower() if body.text_lang else None,
        "ref_audio_path": None,        # 由 profile 解析
        "prompt_text": body.prompt_text,
        "prompt_lang": body.prompt_lang.lower() if body.prompt_lang else None,
        "speed_factor": body.speed_factor,
        "seed": body.seed,
        "media_type": body.media_type,
        "streaming_mode": 1 if body.streaming_mode else 0,
        "profile": body.profile,
        "t2s_weights": body.t2s_weights,
        "vits_weights": body.vits_weights,
        "top_k": body.top_k,
        "top_p": body.top_p,
        "temperature": body.temperature,
        "text_split_method": body.text_split_method,
    }


@router.post("/tts", summary="按档案合成 (POST)")
async def tts_post(request: Request, body: NativeTTS_Request):
    return await tts_handle(_native_to_v2(body), request)


@router.get("/tts", summary="按档案合成 (GET)")
async def tts_get(
    request: Request,
    text: str,
    profile: str,
    prompt_text: str = "",
    text_lang: str = "zh",
    prompt_lang: str = "zh",
    speed_factor: float = 1.0,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    top_k: int = 15,
    top_p: float = 1.0,
    temperature: float = 1.0,
    text_split_method: str = "cut1",
    _guard: None = Depends(require_browser_safe),
):
    body = NativeTTS_Request(
        text=text, profile=profile, prompt_text=prompt_text or None,
        text_lang=text_lang, prompt_lang=prompt_lang,
        speed_factor=speed_factor, seed=seed, media_type=media_type,
        streaming_mode=streaming_mode, top_k=top_k, top_p=top_p,
        temperature=temperature, text_split_method=text_split_method)
    return await tts_handle(_native_to_v2(body), request)
