"""ADR 原生 API (批次11): /api/adr/v1 — 音色档案一等公民的对外接口。

面向新接入方 (以及 ADR 自己的插件/WebUI):
- /health                服务与引擎状态
- /profiles              音色档案清单
- /profiles/{name}/ref   下载档案参考音频
- /tts  POST+GET         按档案名合成 (内部复用 v2 兼容层的核心处理)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from adr.models import voice_library
from adr.server import pathsafe
from adr.server.v2_compat import tts_handle

router = APIRouter(prefix="/api/adr/v1", tags=["adr-native"])

_API_VERSION = "1.0"


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
        # 批次23: 并发排队可视化 (排队等待数 / 合成中数)
        "queue_depth": _gsv.queue_depth(),
        "synth_busy": _gsv.synth_busy(),
        "default_profile": getattr(state, "default_profile", None),
        "weights": {
            "t2s": getattr(state, "t2s_weights", None),
            "vits": getattr(state, "vits_weights", None),
        },
    }


@router.get("/profiles", summary="音色档案清单")
async def profiles():
    out = []
    for name in voice_library.list_voices():
        try:
            meta = voice_library.load_voice(name)
        except Exception:
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
        return JSONResponse(status_code=400,
                            content={"message": f"invalid profile name: {name}",
                                     "code": "invalid_name"})
    try:
        hist = voice_library.voice_history(name)
    except ValueError as e:
        return JSONResponse(status_code=400,
                            content={"message": str(e), "code": "invalid_name"})
    return {"name": name, "history": hist}


@router.post("/profiles/{name}/rollback", summary="回滚到上一版权重 (批次50)")
async def profile_rollback(name: str):
    # 现行绑定入史, history 末版恢复现行; 连续调用 = 逐版回退
    if not pathsafe.is_safe_name(name):
        return JSONResponse(status_code=400,
                            content={"message": f"invalid profile name: {name}",
                                     "code": "invalid_name"})
    try:
        meta = voice_library.rollback_voice(name)
    except ValueError as e:
        return JSONResponse(status_code=400,
                            content={"message": str(e), "code": "rollback_failed"})
    return {"name": name, "profile": meta}


@router.get("/profiles/{name}/ref", summary="下载档案参考音频")
async def profile_ref(name: str):
    # Track B 收口: name 先清洗再拼路径, ../ 穿越越出 VOICES_DIR 一律 404
    ref = pathsafe.resolve_within(voice_library.VOICES_DIR, Path(name) / "ref.wav")
    if ref is None or not ref.exists():
        # 批次35: code 增量字段 (与错误协议统一)
        return JSONResponse(status_code=404,
                            content={"message": f"profile not found: {name}",
                                     "code": "profile_not_found"})
    return FileResponse(str(ref), media_type="audio/wav", filename="ref.wav")


@router.get("/profiles/{name}/export", summary="导出音色档案包 (.adrvoice, 含权重)")
async def profile_export(name: str):
    """导出 .adrvoice 包 (参考音频 + 绑定微调权重)。

    注意: 权重即声音本体, 包内含生物特征性质数据, 分享时自行斟酌。
    """
    import asyncio
    import os
    import tempfile

    from adr.models import voice_package
    from starlette.background import BackgroundTask

    if not pathsafe.is_safe_name(name):
        return JSONResponse(status_code=400,
                            content={"message": f"invalid profile name: {name}",
                                     "code": "invalid_name"})
    fd, tmp = tempfile.mkstemp(suffix=".adrvoice")
    os.close(fd)
    try:
        await asyncio.to_thread(voice_package.export_voice, name, tmp)
    except ValueError as e:
        os.unlink(tmp)
        return JSONResponse(
            status_code=404 if "不存在" in str(e) else 400,
            content={"message": str(e), "code": "export_failed"})
    return FileResponse(tmp, media_type="application/zip",
                        filename=f"{name}.adrvoice",
                        background=BackgroundTask(os.unlink, tmp))


@router.post("/profiles/import", summary="导入音色档案包 (.adrvoice)")
async def profile_import(package: UploadFile = File(...),
                         name: str = Form(None),
                         overwrite: bool = Form(False)):
    """导入 .adrvoice 包; 权重落位 GSV/RVC 标准目录并过 validate_weights_file。"""
    import asyncio
    import os
    import tempfile
    import zipfile as _zipfile

    from adr.models import voice_package
    from starlette.background import BackgroundTask

    suffix = Path(package.filename or "import.adrvoice").suffix or ".adrvoice"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await package.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        r = await asyncio.to_thread(voice_package.import_voice, tmp,
                                    name, overwrite)
    except (ValueError, _zipfile.BadZipFile) as e:
        return JSONResponse(status_code=400,
                            content={"message": str(e), "code": "import_failed"})
    finally:
        os.unlink(tmp)
    from adr.core.audit import log_event
    log_event("voice_import", name=name or "", overwrite=overwrite)
    return {"imported": r}


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
):
    body = NativeTTS_Request(
        text=text, profile=profile, prompt_text=prompt_text or None,
        text_lang=text_lang, prompt_lang=prompt_lang,
        speed_factor=speed_factor, seed=seed, media_type=media_type,
        streaming_mode=streaming_mode, top_k=top_k, top_p=top_p,
        temperature=temperature, text_split_method=text_split_method)
    return await tts_handle(_native_to_v2(body), request)
