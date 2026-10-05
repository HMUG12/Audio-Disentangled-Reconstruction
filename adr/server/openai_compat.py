"""OpenAI 兼容层 (批次21): N.E.KO "OpenAI 兼容" provider 零改造接入 ADR。

对齐 OpenAI Audio API 规范:
- POST /v1/audio/speech  body: {model, input, voice?, response_format?, speed?, stream?}
  · model / voice → ADR 音色档案 (data/voices/<name>), 与 /api/v2 的 profile 同源
  · response_format ∈ wav/mp3/aac/ogg/pcm, 默认 mp3 (与 OpenAI 官方一致)
  · speed 透传语速; stream=true 时流式音频响应 (逐块可播, 默认 true,
    对应 N.E.KO 界面"完整文本请求、流式音频响应")
- GET /v1/models  列出可用音色档案, 供客户端模型下拉选择
- 错误响应统一改写为 OpenAI 风格 {"error": {"message", "type"}}

实现上构造与 v2 同形的 req dict 后直接复用 v2_compat.tts_handle,
档案解析 / 参数校验 / 流式分支行为与 /api/v2/tts 完全一致, 仅错误外壳不同。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from adr.models import voice_library
from adr.server import tts_cache, v2_compat

router = APIRouter(prefix="/v1", tags=["openai"])

# OpenAI response_format → v2 media_type (pcm = 裸 s16le)
FORMATS = {"wav": "wav", "mp3": "mp3", "aac": "aac", "ogg": "ogg", "pcm": "raw"}


class SpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech 请求体; 末段为 ADR 扩展 (可选)。"""
    model: str | None = None          # 音色档案名 (N.E.KO"模型ID")
    input: str = ""                   # 要合成的完整文本
    voice: str | None = None          # OpenAI 语义的音色 → 档案别名回退
    response_format: str = "mp3"      # wav/mp3/aac/ogg/pcm
    speed: float = 1.0
    stream: bool = True
    # ── ADR 扩展 ── (OpenAI 协议外, 可选)
    text_lang: str | None = None
    prompt_lang: str | None = None
    seed: int = -1
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str | None = None
    t2s_weights: str | None = None
    vits_weights: str | None = None


def _openai_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"message": message,
                                           "type": "invalid_request_error"}})


def _rewrite(resp: Response) -> Response:
    """v2 风格错误 ({"message": ...}) → OpenAI 风格; 成功响应原样透传。"""
    if isinstance(resp, JSONResponse) and resp.status_code >= 400:
        try:
            detail = json.loads(resp.body)
        except Exception:
            detail = {}
        msg = detail.get("message") or "request failed"
        if detail.get("Exception"):
            msg = f"{msg}: {detail['Exception']}"
        return _openai_error(resp.status_code, msg)
    return resp


@router.post("/audio/speech", summary="OpenAI 兼容 TTS (N.E.KO OpenAI provider)")
async def audio_speech(request: Request, body: SpeechRequest):
    fmt = (body.response_format or "mp3").lower()
    if fmt not in FORMATS:
        return _openai_error(400, f"unsupported response_format: {body.response_format}, "
                                  f"must be one of wav/mp3/aac/ogg/pcm")
    req = {
        "text": body.input,
        "text_lang": body.text_lang.lower() if body.text_lang else body.text_lang,
        "prompt_lang": body.prompt_lang.lower() if body.prompt_lang else body.prompt_lang,
        "prompt_text": "",              # 档案解析时回填
        "top_k": body.top_k,
        "top_p": body.top_p,
        "temperature": body.temperature,
        "speed_factor": body.speed,
        "seed": body.seed,
        "media_type": FORMATS[fmt],
        "streaming_mode": 1 if body.stream else 0,
        "profile": body.model,
        "voice": body.voice,
        "t2s_weights": body.t2s_weights,
        "vits_weights": body.vits_weights,
    }
    resp = await v2_compat.tts_handle(req, request)
    return _rewrite(resp)


@router.get("/models", summary="OpenAI 兼容模型列表 (即音色档案)")
async def list_models():
    data = [{"id": name, "object": "model", "created": 0, "owned_by": "adr"}
            for name in voice_library.list_voices()]
    return {"object": "list", "data": data}
