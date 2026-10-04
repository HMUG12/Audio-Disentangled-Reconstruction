"""GSV api_v2 兼容层 (批次11): N.E.K.O / GPT-SoVITS 生态零改造接入 ADR。

协议面对齐 api_neko (GPT-SoVITS API v3) 的 /api/v2 实现:
- /tts GET+POST、/set_gpt_weights、/set_sovits_weights
- streaming_mode 分支语义逐字复制 (含 Python 中 bool True == 1 落分支 1 的行为)
- 流式 WAV 字节格式: 首块 44B WAV 头 + 后续裸 s16le PCM
- 错误格式: JSONResponse(400, {"message": ...})

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
from io import BytesIO
from typing import Union

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from adr.models import voice_library
from adr.server.audio_codec import pack_audio, to_int16, wave_header_chunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["tts"])

MEDIA_TYPES = {"wav", "raw", "ogg", "aac"}


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
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    # ── ADR 扩展 ──
    profile: str = None        # 音色档案名 (data/voices/<name>)
    voice: str = None          # profile 别名
    t2s_weights: str = None    # 请求级热换 (GPT/语义, 优先于档案)
    vits_weights: str = None   # 请求级热换 (SoVITS/声学, 优先于档案)


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
        try:
            meta = voice_library.load_voice(profile)
        except Exception:
            return JSONResponse(status_code=400,
                                content={"message": f"unknown profile: {profile}"})
        # 请求显式值优先; 档案值兜底
        if not req.get("ref_audio_path"):
            req["ref_audio_path"] = meta["ref_audio"]
        if not req.get("prompt_text"):
            req["prompt_text"] = meta.get("prompt_text") or ""
        if req.get("vits_weights") is None:
            req["vits_weights"] = meta.get("vits_weights")
        if req.get("t2s_weights") is None:
            req["t2s_weights"] = meta.get("t2s_weights")
    return None


def _check_params(req: dict):
    """必填项 + media_type 白名单。返回错误 Response 或 None。"""
    if not req.get("text"):
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if not req.get("ref_audio_path"):
        return JSONResponse(status_code=400,
                            content={"message": "ref_audio_path is required "
                                                "(or pass profile / set ADR_TTS_DEFAULT_PROFILE)"})
    media_type = (req.get("media_type") or "wav").lower()
    if media_type not in MEDIA_TYPES:
        return JSONResponse(status_code=400,
                            content={"message": f"unsupported media_type: {media_type}, "
                                                f"must be one of wav/raw/ogg/aac"})
    req["media_type"] = media_type
    return None


def _stream_generator(engine, req: dict, media_type: str):
    """同步生成器: StreamingResponse 在线程池内迭代 (不堵事件循环)。

    字节契约 (GSV api_v2): media_type=wav 时首块发 44B WAV 头,
    之后全部为裸 s16le PCM 块; ogg/aac 逐块独立编码。
    """
    first = True
    mt = media_type
    try:
        # 注意: ADR 引擎 yield (chunk, sr) — 与 GSV pipeline 的 (sr, chunk) 相反
        for chunk, sr in engine.synthesize_stream(
            req["text"], req["ref_audio_path"],
            prompt_text=req.get("prompt_text") or "",
            text_lang=req.get("text_lang") or "zh",
            prompt_lang=req.get("prompt_lang") or "zh",
            t2s_weights=req.get("t2s_weights"),
            vits_weights=req.get("vits_weights"),
            split_method=req.get("text_split_method") or "cut3",
            head_seed=req.get("seed", -1),
        ):
            if first and mt == "wav":
                yield wave_header_chunk(sample_rate=sr)
                mt = "raw"
                first = False
            yield pack_audio(BytesIO(), to_int16(chunk), sr, mt).getvalue()
    except Exception as e:
        # 流已开始, 无法改写状态码 — 记录后终止 (与 GSV api_v2 行为一致)
        logger.exception("streaming tts failed: %s", e)


async def tts_handle(req: dict, request: Request) -> Response:
    """TTS 核心处理 (GET/POST 共用; native 层也复用)。

    分支语义逐字对齐 api_neko tts.py:
      0 → 非流式 | 1 → 分段流 (return_fragment) | 2 → 真流式 | 3 → 真流式+定长块
      其余 → 400。注: bool True == 1, 自然落分支 1。
    """
    err = _resolve_profile(request, req)
    if err is not None:
        return err
    err = _check_params(req)
    if err is not None:
        return err

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
                                "or true/false(bool)"})
    streaming = streaming or return_fragment
    engine = _get_engine(request)

    try:
        if streaming:
            return StreamingResponse(
                _stream_generator(engine, req, media_type),
                media_type=f"audio/{media_type}")
        audio, sr = await asyncio.to_thread(
            engine.synthesize,
            req["text"], req["ref_audio_path"],
            prompt_text=req.get("prompt_text") or "",
            text_lang=req.get("text_lang") or "zh",
            prompt_lang=req.get("prompt_lang") or "zh",
            speed_factor=req.get("speed_factor", 1.0),
            seed=req.get("seed", -1),
            t2s_weights=req.get("t2s_weights"),
            vits_weights=req.get("vits_weights"),
            split_method=req.get("text_split_method") or "cut1",
            top_k=req.get("top_k", 15),
            top_p=req.get("top_p", 1.0),
            temperature=req.get("temperature", 1.0),
        )
        buf = pack_audio(BytesIO(), to_int16(audio), sr, media_type)
        return Response(buf.getvalue(), media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400,
                            content={"message": "tts failed", "Exception": str(e)})


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
    fragment_interval: float = 0.3,
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

@router.get("/set_gpt_weights", summary="切换 GPT (t2s) 权重")
async def set_gpt_weights(request: Request, weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400,
                                content={"message": "gpt weight path is required"})
        engine = _get_engine(request)
        await asyncio.to_thread(engine.warmup, None, weights_path)
    except Exception as e:
        return JSONResponse(status_code=400,
                            content={"message": "change gpt weight failed",
                                     "Exception": str(e)})
    request.app.state.t2s_weights = weights_path
    return JSONResponse(status_code=200, content={"message": "success"})


@router.get("/set_sovits_weights", summary="切换 SoVITS (vits) 权重")
async def set_sovits_weights(request: Request, weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400,
                                content={"message": "sovits weight path is required"})
        engine = _get_engine(request)
        await asyncio.to_thread(engine.warmup, weights_path, None)
    except Exception as e:
        return JSONResponse(status_code=400,
                            content={"message": "change sovits weight failed",
                                     "Exception": str(e)})
    request.app.state.vits_weights = weights_path
    return JSONResponse(status_code=200, content={"message": "success"})
