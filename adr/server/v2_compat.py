"""GSV api_v2 兼容层 (批次11): N.E.K.O / GPT-SoVITS 生态零改造接入 ADR。

协议面对齐 api_neko (GPT-SoVITS API v3) 的 /api/v2 实现:
- /tts GET+POST、/set_gpt_weights、/set_sovits_weights
- streaming_mode 分支语义逐字复制 (含 Python 中 bool True == 1 落分支 1 的行为)
- 流式 WAV 字节格式: 首块 44B WAV 头 + 后续裸 s16le PCM
- 错误格式: JSONResponse(400, {"message": ..., "code": ...})
  (批次35: code 为稳定错误码, 增量字段; 既有 message 逐字保留)

ADR 扩展 (GSV 协议外, 向后兼容):
- profile / voice: 引用 ADR 音色档案 (data/voices/<name>), 自动解析
  参考音频 / prompt_text / 微调权重
- t2s_weights / vits_weights: 请求级热换, 优先于档案
- 引擎不支持的采样参数 (batch_size/sample_steps/...) 接受但忽略,
  差异清单见 docs/tts-api-spec.md
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Union

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from adr.core.exceptions import (
    ProfileInvalidError,
    ProfileNotFoundError,
    SynthesisParamsError,
    error_code,
)
from adr.models.gsv_engine import handle_cuda_oom, is_cuda_oom
from adr.server import tts_cache
from adr.server.audio_codec import pack_audio, to_int16
from adr.services import SynthesisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["tts"])


class TTS_Request(BaseModel):
    """与 GSV api_v2 同名同义; 末 4 个字段为 ADR 扩展。"""
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float | None = None  # 批次33: None → 引擎侧 env (默认 0.3 = GSV 原行为)
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    # ── ADR 扩展 ── (显式 null 视同未传, 部分客户端会序列化全字段)
    profile: str | None = None        # 音色档案名 (data/voices/<name>)
    voice: str | None = None          # profile 别名
    t2s_weights: str | None = None    # 请求级热换 (GPT/语义, 优先于档案)
    vits_weights: str | None = None   # 请求级热换 (SoVITS/声学, 优先于档案)


def _get_engine(request: Request):
    """惰性取引擎: 测试注入 FakeEngine; 生产首次请求时建 GSV 单例。"""
    if getattr(request.app.state, "engine", None) is None:
        from adr.models.gsv_engine import get_gsv_engine
        request.app.state.engine = get_gsv_engine()
    return request.app.state.engine


def _resolve_profile(request: Request, req: dict):
    """profile 扩展参数 → 档案解析 (mutates req)。返回错误 Response 或 None。"""
    profile = req.get("profile") or req.get("voice")
    if not profile:
        profile = getattr(request.app.state, "default_profile", None)
    if profile:
        # 批次34: 名称安全 (../ 穿越一律 400) 与档案回填语义收口至 SynthesisService
        try:
            meta = SynthesisService.load_profile(profile)
        except (ProfileInvalidError, ProfileNotFoundError) as e:
            # 批次35: code 增量字段 (既有 message 逐字保留, wire 兼容)
            return JSONResponse(status_code=400,
                                content={"message": str(e), "code": error_code(e)})
        req.update(SynthesisService.profile_fill(
            meta,
            ref_audio_path=req.get("ref_audio_path"),
            prompt_text=req.get("prompt_text"),
            t2s_weights=req.get("t2s_weights"),
            vits_weights=req.get("vits_weights"),
        ))
        # 批次53: 语言未显式传入 (None) 时按档案语种兜底 — 参考音频是档案语种,
        # 合成文本默认同语种最稳; 显式传入 (含 easy 面板 auto/en/...) 不覆盖。
        if req.get("text_lang") is None:
            req["text_lang"] = meta.get("language") or "zh"
        if req.get("prompt_lang") is None:
            req["prompt_lang"] = meta.get("language") or "zh"
    return None


def _env_num(name: str, default, cast=int):
    """批次66: 资源上限 env 读取; 非法/非正值回落默认 (正常使用碰不到的宽松值)。"""
    try:
        v = cast(os.environ.get(name, default))
    except ValueError:
        return default
    return v if v > 0 and math.isfinite(v) else default


def max_text_chars() -> int:
    """ADR_TTS_MAX_TEXT_CHARS — 单次合成文本字数上限 (批次66, 默认 20000; v3 WS 共用)。"""
    return _env_num("ADR_TTS_MAX_TEXT_CHARS", 20000)


def max_queue() -> int:
    """ADR_TTS_MAX_QUEUE — 引擎排队数上限, 超出回 503 busy (批次66, 默认 32; v3 WS 共用)。"""
    return _env_num("ADR_TTS_MAX_QUEUE", 32)


def engine_busy(engine) -> bool:
    """批次66: 软准入 — 引擎排队数达上限即拒绝 (测试假引擎无 queue_depth 视为不忙)。"""
    qd = getattr(engine, "queue_depth", None)
    return callable(qd) and qd() >= max_queue()


# 批次66: 数值参数合法闭区间 (None = 未传不校验; speed 区间可 env 调)
_FIXED_BOUNDS = {
    "top_k": (1, 1000),
    "top_p": (0.0, 1.0),
    "temperature": (0.0, 10.0),
    "fragment_interval": (0.0, 10.0),
}


def _bad_number(req: dict):
    """返回第一个越界/非数值参数的错误文案, 全部合法返回 None。"""
    bounds = dict(_FIXED_BOUNDS, speed_factor=(
        _env_num("ADR_TTS_SPEED_MIN", 0.1, float),
        _env_num("ADR_TTS_SPEED_MAX", 10.0, float)))
    for key, (lo, hi) in bounds.items():
        v = req.get(key)
        if v is None:
            continue
        try:
            ok = lo <= float(v) <= hi   # NaN 比较恒 False → 拒绝
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return f"{key} out of range: {v!r}, must be within [{lo}, {hi}]"
    return None


def _check_ref_audio(path: str):
    """批次66: ref_audio_path 若存在须为普通文件且不超大小上限 (挡设备文件/巨型文件)。

    路径不存在时放行, 维持既有行为 (由引擎报错); 返回错误文案或 None。
    """
    try:
        p = Path(path)
        if not p.exists():
            return None
        if not p.is_file():
            return f"ref_audio_path is not a regular file: {path}"
        cap_mb = _env_num("ADR_TTS_REF_MAX_MB", 100)
        if p.stat().st_size > cap_mb * 1024 * 1024:
            return f"ref_audio_path exceeds {cap_mb}MB limit: {path}"
    except (OSError, ValueError) as e:
        return f"invalid ref_audio_path: {e}"
    return None


def _check_params(req: dict):
    """必填项 + media_type 白名单 (批次34 收口至 SynthesisService)。

    批次66: 追加文本长度上限 (413) 与数值参数区间校验 (400)。
    返回错误 Response 或 None。
    """
    try:
        req["media_type"] = SynthesisService.validate_params(
            req.get("text"), req.get("ref_audio_path"), req.get("media_type"))
    except SynthesisParamsError as e:
        return JSONResponse(status_code=400,
                            content={"message": str(e), "code": error_code(e)})
    cap = max_text_chars()
    if len(req["text"]) > cap:
        return JSONResponse(
            status_code=413,
            content={"message": f"text too long: {len(req['text'])} chars, "
                                f"limit {cap} (ADR_TTS_MAX_TEXT_CHARS)",
                     "code": "payload_too_large"})
    msg = _bad_number(req)
    if msg:
        return JSONResponse(status_code=400,
                            content={"message": msg, "code": "invalid_params"})
    return None


def _busy_response() -> JSONResponse:
    return JSONResponse(status_code=503,
                        content={"message": "busy: synthesis queue full",
                                 "code": "busy"})


def _stream_generator(engine, req: dict, media_type: str):
    """同步生成器: StreamingResponse 在线程池内迭代 (不堵事件循环)。

    字节契约 (GSV api_v2, 批次34 收口至 SynthesisService.stream_bytes):
    media_type=wav 时首块发 44B WAV 头, 之后全部为裸 s16le PCM 块;
    ogg/aac 逐块独立编码。
    """
    try:
        yield from SynthesisService.stream_bytes(
            engine, req["text"], req["ref_audio_path"], media_type=media_type,
            prompt_text=req.get("prompt_text") or "",
            text_lang=req.get("text_lang") or "zh",
            prompt_lang=req.get("prompt_lang") or "zh",
            t2s_weights=req.get("t2s_weights"),
            vits_weights=req.get("vits_weights"),
            split_method=req.get("text_split_method"),
            head_seed=req.get("seed", -1),
            top_k=req.get("top_k", 15),
            top_p=req.get("top_p", 1.0),
            temperature=req.get("temperature", 1.0),
            speed_factor=req.get("speed_factor", 1.0),
            fragment_interval=req.get("fragment_interval"),
        )
    except Exception as e:
        # 流已开始, 无法改写状态码 — 记录后终止 (与 GSV api_v2 行为一致)
        if is_cuda_oom(e):
            handle_cuda_oom(e)  # 释放碎片显存 + 友好警告日志 (批次57)
            # 批次66: OOM 截断同样记一条可定位的服务端日志 (wire 行为不变)
            logger.warning("streaming tts truncated by CUDA OOM")
        else:
            logger.exception("streaming tts failed: %s", e)


async def tts_handle(req: dict, request: Request) -> Response:
    """TTS 核心处理 (GET/POST 共用; native 层也复用)。

    分支语义逐字对齐 api_neko tts.py:
      0 → 非流式 | 1 → 分段流 (return_fragment) | 2 → 真流式 | 3 → 真流式+定长块
      其余 → 400。注: bool True == 1, 自然落分支 1。
    """
    # 批次38 (Track B 补口): 请求显式传入的权重文件先过安全预检, 给出干净的 400
    # (流式路径响应头先行, 引擎校验发生在流开始后只能断流)。仅校验请求级热换值 —
    # 档案回填的权重由引擎 _ensure_weights 换权分支强收口, 不在此处重复。
    for _wk in ("t2s_weights", "vits_weights"):
        _wp = req.get(_wk)
        if _wp:
            try:
                await asyncio.to_thread(_validate_weights_file, _wp)
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"message": f"invalid {_wk}: {e}",
                             "code": "invalid_params"})

    err = _resolve_profile(request, req)
    if err is not None:
        return err
    err = _check_params(req)
    if err is not None:
        return err
    msg = await asyncio.to_thread(_check_ref_audio, req["ref_audio_path"])
    if msg:
        return JSONResponse(status_code=400,
                            content={"message": msg, "code": "invalid_params"})

    streaming_mode = req.get("streaming_mode", False)
    media_type = req["media_type"]

    if streaming_mode == 0:
        streaming, return_fragment = False, False
    elif streaming_mode == 1:
        streaming, return_fragment = False, True
    elif streaming_mode == 2:
        streaming, return_fragment = True, False
    elif streaming_mode == 3:
        streaming, return_fragment = True, False
    else:
        return JSONResponse(
            status_code=400,
            content={"message": "the value of streaming_mode must be 0, 1, 2, 3(int) "
                                "or true/false(bool)",
                     "code": "invalid_params"})
    streaming = streaming or return_fragment
    engine = _get_engine(request)

    try:
        if streaming:
            # 批次66: 排队达上限即 503, 不再无界占用线程池
            if engine_busy(engine):
                return _busy_response()
            return StreamingResponse(
                _stream_generator(engine, req, media_type),
                media_type=f"audio/{media_type}")
        cache_key = tts_cache.make_key(req)
        cached = tts_cache.get(cache_key)
        if cached is not None:
            return Response(cached, media_type=f"audio/{media_type}")
        if engine_busy(engine):   # 批次66: 缓存命中不占引擎, 故放在命中之后
            return _busy_response()
        audio, sr = await asyncio.to_thread(
            SynthesisService.synthesize_once,
            engine, req["text"], req["ref_audio_path"],
            prompt_text=req.get("prompt_text") or "",
            text_lang=req.get("text_lang") or "zh",
            prompt_lang=req.get("prompt_lang") or "zh",
            speed_factor=req.get("speed_factor", 1.0),
            seed=req.get("seed", -1),
            t2s_weights=req.get("t2s_weights"),
            vits_weights=req.get("vits_weights"),
            split_method=req.get("text_split_method"),
            top_k=req.get("top_k", 15),
            top_p=req.get("top_p", 1.0),
            temperature=req.get("temperature", 1.0),
            fragment_interval=req.get("fragment_interval"),
        )
        buf = pack_audio(BytesIO(), to_int16(audio), sr, media_type)
        tts_cache.put(cache_key, buf.getvalue(), ext=media_type)
        return Response(buf.getvalue(), media_type=f"audio/{media_type}")
    except Exception as e:
        if is_cuda_oom(e):
            # 批次57: OOM 独立分流 — 释放碎片显存 + 503 友好提示 (此前是裸 400 堆栈原文)
            return JSONResponse(status_code=503,
                                content={"message": handle_cuda_oom(e),
                                         "code": "cuda_oom"})
        # 批次66: 服务端留堆栈; 400 状态码为 GSV wire 兼容保留 (code 区分 internal_error)
        logger.exception("tts failed")
        return JSONResponse(status_code=400,
                            content={"message": "tts failed", "Exception": str(e),
                                     "code": error_code(e)})


# ─── TTS 端点 ───

@router.get("/tts", summary="TTS 推理 (GET)")
async def tts_get_endpoint(
    request: Request,
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 15,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float | None = None,
    seed: int = -1,
    media_type: str = "wav",
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
    streaming_mode: Union[bool, int] = False,
    overlap_length: int = 2,
    min_chunk_length: int = 16,
    profile: str = None,
    voice: str = None,
    t2s_weights: str = None,
    vits_weights: str = None,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower() if text_lang else text_lang,
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower() if prompt_lang else prompt_lang,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
        "overlap_length": int(overlap_length),
        "min_chunk_length": int(min_chunk_length),
        "profile": profile,
        "voice": voice,
        "t2s_weights": t2s_weights,
        "vits_weights": vits_weights,
    }
    return await tts_handle(req, request)


@router.post("/tts", summary="TTS 推理 (POST)")
async def tts_post_endpoint(request: Request, body: TTS_Request):
    req = body.model_dump()
    if req.get("text_lang"):
        req["text_lang"] = req["text_lang"].lower()
    if req.get("prompt_lang"):
        req["prompt_lang"] = req["prompt_lang"].lower()
    return await tts_handle(req, request)


# ─── 模型切换端点 (全局语义: 热换引擎默认权重) ───

def _validate_weights_file(weights_path: str) -> None:
    """权重文件安全预检 (Track B 收口): 防恶意 pickle 反序列化 RCE。

    批次38: 实现收编为引擎单点 gsv_engine.validate_weights_file —
    _ensure_weights 换权分支同样预检, 覆盖 v3 档案等不经本层的入口;
    此处委托保持既有访问面 (tests/早期引用) 不变。
    """
    from adr.models.gsv_engine import validate_weights_file
    validate_weights_file(weights_path)


@router.get("/set_gpt_weights", summary="切换 GPT (t2s) 权重")
async def set_gpt_weights(request: Request, weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400,
                                content={"message": "gpt weight path is required",
                                         "code": "invalid_params"})
        # Track B: 安全预检, 恶意 pickle 在进入引擎加载前即被拒 (400)
        await asyncio.to_thread(_validate_weights_file, weights_path)
        engine = _get_engine(request)
        await asyncio.to_thread(engine.warmup, None, weights_path)
    except Exception as e:
        if is_cuda_oom(e):
            # 批次66: 热换权重 OOM 与 tts_handle 同口径 — 释放碎片显存 + 503
            return JSONResponse(status_code=503,
                                content={"message": handle_cuda_oom(e),
                                         "code": "cuda_oom"})
        logger.exception("change gpt weight failed")  # 批次66
        return JSONResponse(status_code=400,
                            content={"message": "change gpt weight failed",
                                     "Exception": str(e),
                                     "code": error_code(e)})
    request.app.state.t2s_weights = weights_path
    return JSONResponse(status_code=200, content={"message": "success"})


@router.get("/set_sovits_weights", summary="切换 SoVITS (vits) 权重")
async def set_sovits_weights(request: Request, weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400,
                                content={"message": "sovits weight path is required",
                                         "code": "invalid_params"})
        # Track B: 安全预检, 恶意 pickle 在进入引擎加载前即被拒 (400)
        await asyncio.to_thread(_validate_weights_file, weights_path)
        engine = _get_engine(request)
        await asyncio.to_thread(engine.warmup, weights_path, None)
    except Exception as e:
        if is_cuda_oom(e):
            # 批次66: 热换权重 OOM 与 tts_handle 同口径 — 释放碎片显存 + 503
            return JSONResponse(status_code=503,
                                content={"message": handle_cuda_oom(e),
                                         "code": "cuda_oom"})
        logger.exception("change sovits weight failed")  # 批次66
        return JSONResponse(status_code=400,
                            content={"message": "change sovits weight failed",
                                     "Exception": str(e),
                                     "code": error_code(e)})
    request.app.state.vits_weights = weights_path
    return JSONResponse(status_code=200, content={"message": "success"})
