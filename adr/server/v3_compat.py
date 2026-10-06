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
(字段与 api_neko routers/tts_v3.py 逐字对齐; task_id 为会话内自增计数;
批次35: error 帧统一携带 "code" 稳定错误码, 增量字段, 既有 message 保留)。

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

from adr.core.exceptions import (
    ProfileInvalidError,
    ProfileNotFoundError,
    error_code,
)
from adr.models import voice_library
from adr.services import SynthesisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["tts-v3"])

# 句子边界标点 (与 api_neko tts_v3._SENTENCE_SPLITS 一致)
_SENTENCE_SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}

# 句级帧队列容量 (Track B 收口: 提为常量供测试缩小触发 QueueFull 路径)
_QUEUE_MAXSIZE = 64


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

    "_default" → app.state.default_profile 或第一个档案; 无档案可用返回 None。
    批次35: 非法名/不存在改类型化上报 — ProfileInvalidError/ProfileNotFoundError
    原样抛出, init 分支捕获回 error 帧; 仅"默认解析无档案"仍静默 None。
    """
    name = (voice_id or "").strip() or "_default"
    if name == "_default":
        name = (getattr(app.state, "default_profile", None)
                or (voice_library.list_voices() or [None])[0])
        if not name:
            return None
    return SynthesisService.load_profile(name)


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
    """双向流式: 文本流入 + 音频流出 (N.E.K.O gptsovits worker 消费面)。

    鉴权 (Track B): APIKeyMiddleware 在握手前校验 ?token= / ?api_key= /
    Sec-WebSocket-Protocol; 未配置 ADR_TTS_API_KEY 时完全放行 (协议兼容)。
    浏览器以子协议携带 key 时必须回显所选子协议, 否则浏览器会断开握手。
    """
    proto = websocket.headers.get("sec-websocket-protocol", "")
    chosen = proto.split(",")[0].strip() or None
    if chosen:
        await websocket.accept(subprotocol=chosen)
    else:
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
        """合成一句: sentence → 逐帧完整 WAV → sentence_done / error。

        Track B 收口:
        - 任何退出路径 (断连 / 发送失败 / 队列满 / 完成) 都在 finally 置位
          cancel, 生产线程随即 break 并 close 合成生成器 → 引擎锁经其内部
          finally 立即释放, 不再"断连后持锁合成到结束"饿死后续请求;
        - 帧队列满时不再堆积/吞异常: 消费侧立即回 busy 错误帧并结束本句,
          生产线程感知 cancel 后停止。
        """
        task_id = next(task_ids)
        await _safe_send_json({"type": "sentence", "text": text, "task_id": task_id})

        loop = asyncio.get_running_loop()
        chan: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        cancel = threading.Event()     # 置位 → 生产线程停止合成并释放引擎锁
        chan_full = threading.Event()  # 队列满 → 消费侧立即回 busy 错误帧

        def _put(item):
            """生产线程 → 事件循环 入队; 满时不阻塞、不吞异常 (置标志上报)。"""
            try:
                chan.put_nowait(item)
            except asyncio.QueueFull:
                chan_full.set()

        def _produce():
            gen = None
            try:
                samp = voice.get("sampling") or {}
                gen = SynthesisService.stream_chunks(
                    _engine(), text, voice["ref_audio"],
                    prompt_text=voice.get("prompt_text") or "",
                    text_lang=overrides.get("text_lang") or "zh",
                    prompt_lang="zh",
                    t2s_weights=voice.get("t2s_weights"),
                    vits_weights=voice.get("vits_weights"),
                    split_method=overrides.get("text_split_method"),
                    head_seed=overrides.get("seed", -1),
                    top_k=samp.get("top_k", 15),
                    top_p=samp.get("top_p", 1.0),
                    temperature=samp.get("temperature", 1.0),
                    fragment_interval=overrides.get("fragment_interval"),
                )
                for chunk, sr in gen:
                    if cancel.is_set():
                        break
                    frame = SynthesisService.frame_bytes(chunk, sr)
                    loop.call_soon_threadsafe(_put, frame)
            except Exception as e:
                if not cancel.is_set():
                    logger.exception("v3 stream synth failed")
                    loop.call_soon_threadsafe(
                        _put, ("__err__", str(e), error_code(e)))
            finally:
                if gen is not None:
                    gen.close()  # 触发生成器 finally → 引擎锁释放 (线程内安全)
                loop.call_soon_threadsafe(_put, None)

        threading.Thread(target=_produce, daemon=True,
                         name=f"adr-v3-tts-{task_id}").start()

        n, err, code = 0, None, None
        try:
            while True:
                if chan_full.is_set():
                    # 队列曾满: 立即报 busy, 不阻塞不堆积 (生产者随后自行停止)
                    await _safe_send_json(
                        {"type": "error", "message": "busy: synthesis queue full",
                         "code": "busy"})
                    return
                item = await chan.get()
                if item is None:
                    break
                if isinstance(item, tuple):
                    err, code = item[1], item[2]
                    break
                await _safe_send_bytes(item)
                n += 1
            logger.debug("v3 task#%s done frames=%s err=%s", task_id, n, err)
            if err:
                await _safe_send_json({"type": "error", "message": err, "code": code})
            else:
                await _safe_send_json({"type": "sentence_done", "task_id": task_id,
                                       "chunks_sent": n})
        finally:
            cancel.set()  # 断连/异常/完成 统一取消生产者并释放引擎锁

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=300.0)
            except asyncio.TimeoutError:
                await _safe_send_json({"type": "error", "message": "session timeout",
                                       "code": "session_timeout"})
                break

            cmd = data.get("cmd", "")

            if cmd == "init":
                vid = data.get("voice_id", "_default")
                # 批次35: 类型化上报 — 非法名回精确 message; not found 消息逐字
                # 保留 (N.E.K.O 消费); code 均为增量字段
                try:
                    v = _resolve_voice(vid, app)
                except ProfileInvalidError as e:
                    await _safe_send_json(
                        {"type": "error", "message": str(e),
                         "code": error_code(e)})
                    break
                except ProfileNotFoundError:
                    await _safe_send_json(
                        {"type": "error", "message": f"voice_id '{vid}' not found",
                         "code": "profile_not_found"})
                    break
                if v is None:
                    await _safe_send_json(
                        {"type": "error", "message": f"voice_id '{vid}' not found",
                         "code": "profile_not_found"})
                    break
                voice = v
                session_voice_id = vid
                overrides = {k: data.get(k) for k in (
                    "text_lang", "speed_factor", "temperature", "top_k", "top_p",
                    "seed", "batch_size", "text_split_method", "media_type",
                    "fragment_interval",
                ) if data.get(k) is not None}
                await _safe_send_json({"type": "ready", "voice_id": vid})

            elif cmd == "text":
                if voice is None:
                    await _safe_send_json(
                        {"type": "error", "message": "not initialized, send init first",
                         "code": "not_initialized"})
                    continue
                text_data = (data.get("data") or "").strip()
                if text_data:
                    await _push_task(text_data)

            elif cmd == "append":
                if voice is None:
                    await _safe_send_json(
                        {"type": "error", "message": "not initialized, send init first",
                         "code": "not_initialized"})
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
                await _safe_send_json({"type": "error", "message": f"unknown cmd: {cmd}",
                                       "code": "unknown_cmd"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("v3 stream-input error")
        try:
            await _safe_send_json({"type": "error", "message": str(e),
                                   "code": error_code(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
