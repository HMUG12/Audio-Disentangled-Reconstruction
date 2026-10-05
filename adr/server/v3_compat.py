"""N.E.K.O API v3 兼容层 (批次19): N.E.K.O 主程序 GPT-SoVITS provider 直连。

N.E.K.O 的 gptsovits worker (main_logic/tts_client/workers/gptsovits.py) 消费两个面:
1. GET  /api/v3/voices            → 音色列表 [{id, name, description, version}]
   (N.E.K.O 为每项加 "gsv:" 前缀作为 voice_id)
2. WS   /api/v3/tts/stream-input  → 双向流式: init/ready → text|append|flush → end
   (N.E.K.O 把 http base_url 转 ws://{host}/api/v3/tts/stream-input)

WS 帧契约 (N.E.K.O _extract_pcm_from_wav): 每个 binary 帧是**完整 WAV**
(44B 头, 采样率在 24:28), 客户端自行抽 PCM 并重采样到 48k — 与 media_type
无关, binary 帧恒为 WAV。

服务端 JSON 消息: ready/sentence/sentence_done/flushed/done/error
(字段与 api_neko routers/tts_v3.py 逐字对齐; task_id 为会话内自增计数)。

差异 (与 api_neko v3): ADR 无推理任务队列, 逐句串行合成 (engine 单例本就
串行); overrides 的 media_type 被忽略 (帧恒为 WAV)。
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import threading
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from adr.models import voice_library
from adr.server.audio_codec import to_int16, wave_header_chunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["tts-v3"])

# 句子边界标点 (与 api_neko tts_v3._SENTENCE_SPLITS 一致)
_SENTENCE_SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}


class _TextBuffer:
    """文本缓冲: 累积碎片 → 按标点边界提取完整句子 (照搬 api_neko 实现)。"""

    def __init__(self):
        self._buf = ""

    def append(self, text: str):
        self._buf += text

    def extract_sentences(self) -> list[str]:
        sentences: list[str] = []
        last_split = -1
        for i, ch in enumerate(self._buf):
            if ch in _SENTENCE_SPLITS:
                last_split = i
        if last_split < 0:
            return sentences
        completed = self._buf[: last_split + 1]
        self._buf = self._buf[last_split + 1:]
        current = ""
        for ch in completed:
            current += ch
            if ch in _SENTENCE_SPLITS:
                s = current.strip()
                if s:
                    sentences.append(s)
                current = ""
        if current.strip():
            sentences.append(current.strip())
        return sentences

    def flush(self) -> str:
        text = self._buf.strip()
        self._buf = ""
        return text

    @property
    def content(self) -> str:
        return self._buf


def _resolve_voice(voice_id: str, app) -> Optional[dict]:
    """voice_id → 档案 dict {ref_audio, prompt_text, t2s_weights, vits_weights}。

    "_default" → app.state.default_profile 或第一个档案; 找不到返回 None。
    """
    name = (voice_id or "").strip() or "_default"
    if name == "_default":
        name = (getattr(app.state, "default_profile", None)
                or (voice_library.list_voices() or [None])[0])
    if not name:
        return None
    try:
        return voice_library.load_voice(name)
    except Exception:
        return None


@router.get("/voices", summary="音色列表 (N.E.K.O 自定义音色源)")
async def voices_list(request: Request):
    out = []
    default_name = (getattr(request.app.state, "default_profile", None)
                    or (voice_library.list_voices() or ["_default"])[0])
    out.append({"id": "_default", "name": "默认声音",
                "description": f"ADR 默认档案 ({default_name})", "version": "v2"})
    for name in voice_library.list_voices():
        try:
            meta = voice_library.load_voice(name)
        except Exception:
            meta = {}
        out.append({
            "id": name,
            "name": name,
            "description": meta.get("prompt_text") or meta.get("style") or "",
            "version": "v2",
        })
    return out


@router.websocket("/tts/stream-input")
async def tts_ws_stream_input(websocket: WebSocket):
    """双向流式: 文本流入 + 音频流出 (N.E.K.O gptsovits worker 消费面)。"""
    await websocket.accept()

    app = websocket.app
    voice: Optional[dict] = None
    session_voice_id = "_default"
    overrides: dict = {}
    text_buffer = _TextBuffer()
    task_ids = itertools.count(1)
    send_lock = asyncio.Lock()

    async def _safe_send_json(data: dict):
        async with send_lock:
            await websocket.send_json(data)

    async def _safe_send_bytes(data: bytes):
        async with send_lock:
            await websocket.send_bytes(data)

    def _engine():
        if getattr(app.state, "engine", None) is None:
            from adr.models.gsv_engine import get_gsv_engine
            app.state.engine = get_gsv_engine()
        return app.state.engine

    async def _push_task(text: str):
        """合成一句: sentence → 逐帧完整 WAV → sentence_done / error。"""
        task_id = next(task_ids)
        await _safe_send_json({"type": "sentence", "text": text, "task_id": task_id})

        loop = asyncio.get_running_loop()
        chan: asyncio.Queue = asyncio.Queue(maxsize=64)

        def _produce():
            try:
                samp = voice.get("sampling") or {}
                for chunk, sr in _engine().synthesize_stream(
                    text, voice["ref_audio"],
                    prompt_text=voice.get("prompt_text") or "",
                    text_lang=overrides.get("text_lang") or "zh",
                    prompt_lang="zh",
                    t2s_weights=voice.get("t2s_weights"),
                    vits_weights=voice.get("vits_weights"),
                    split_method=overrides.get("text_split_method") or "cut3",
                    head_seed=overrides.get("seed", -1),
                    top_k=samp.get("top_k", 15),
                    top_p=samp.get("top_p", 1.0),
                    temperature=samp.get("temperature", 1.0),
                ):
                    frame = wave_header_chunk(sample_rate=sr) + \
                        to_int16(chunk).tobytes()
                    loop.call_soon_threadsafe(chan.put_nowait, frame)
            except Exception as e:
                logger.exception("v3 stream synth failed")
                loop.call_soon_threadsafe(chan.put_nowait, ("__err__", str(e)))
            loop.call_soon_threadsafe(chan.put_nowait, None)

        threading.Thread(target=_produce, daemon=True,
                         name=f"adr-v3-tts-{task_id}").start()

        n, err = 0, None
        while True:
            item = await chan.get()
            if item is None:
                break
            if isinstance(item, tuple):
                err = item[1]
                break
            await _safe_send_bytes(item)
            n += 1
        logger.debug("v3 task#%s done frames=%s err=%s", task_id, n, err)
        if err:
            await _safe_send_json({"type": "error", "message": err})
        else:
            await _safe_send_json({"type": "sentence_done", "task_id": task_id,
                                   "chunks_sent": n})

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=300.0)
            except asyncio.TimeoutError:
                await _safe_send_json({"type": "error", "message": "session timeout"})
                break

            cmd = data.get("cmd", "")

            if cmd == "init":
                vid = data.get("voice_id", "_default")
                v = _resolve_voice(vid, app)
                if v is None:
                    await _safe_send_json(
                        {"type": "error", "message": f"voice_id '{vid}' not found"})
                    break
                voice = v
                session_voice_id = vid
                overrides = {k: data.get(k) for k in (
                    "text_lang", "speed_factor", "temperature", "top_k", "top_p",
                    "seed", "batch_size", "text_split_method", "media_type",
                ) if data.get(k) is not None}
                await _safe_send_json({"type": "ready", "voice_id": vid})

            elif cmd == "text":
                if voice is None:
                    await _safe_send_json(
                        {"type": "error", "message": "not initialized, send init first"})
                    continue
                text_data = (data.get("data") or "").strip()
                if text_data:
                    await _push_task(text_data)

            elif cmd == "append":
                if voice is None:
                    await _safe_send_json(
                        {"type": "error", "message": "not initialized, send init first"})
                    continue
                text_buffer.append(data.get("data") or "")
                for sentence in text_buffer.extract_sentences():
                    await _push_task(sentence)

            elif cmd == "flush":
                if voice is not None:
                    remaining = text_buffer.flush()
                    if remaining:
                        await _push_task(remaining)
                await _safe_send_json({"type": "flushed"})

            elif cmd == "end":
                if voice is not None:
                    remaining = text_buffer.flush()
                    if remaining:
                        await _push_task(remaining)
                await _safe_send_json({"type": "done"})
                break

            else:
                await _safe_send_json({"type": "error", "message": f"unknown cmd: {cmd}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("v3 stream-input error")
        try:
            await _safe_send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
