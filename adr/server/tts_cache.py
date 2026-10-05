"""TTS 合成结果 LRU 缓存 (批次21): 重复文本秒回、省 GPU。

播报 / LLM 对话场景里同一文本常被反复请求 (重试、多端播放、调试),
每次都跑完整 GSV 前向 (~数秒) 是纯浪费。这里在 v2 非流式路径上
加一层进程内 LRU:

- key: 请求参数指纹 (sha1); seed 不入 key — 播报一致性优先,
  同文本复用同一次合成, 避免"重试一次换一个人声"的观感
- 上限: 32 条 / 256MB (ADR_TTS_CACHE_MAX / ADR_TTS_CACHE_MB 可调)
- 开关: ADR_TTS_CACHE=0 关闭 (默认开)
- 流式响应不缓存 (分块字节契约, 复用价值低)

纯标准库, 线程安全 (tts_handle 的查/填跑在事件循环线程, 但合成
在线程池, 用锁保护避免并发交错下的 dict 竞态)。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict

DEFAULT_MAX_ENTRIES = 32
DEFAULT_MAX_BYTES = 256 * 1024 * 1024

# 参与 key 的请求字段 (seed 有意排除, 见模块 docstring)
_KEY_FIELDS = (
    "text", "ref_audio_path", "prompt_text", "text_lang", "prompt_lang",
    "top_k", "top_p", "temperature", "speed_factor",
    "t2s_weights", "vits_weights", "media_type", "text_split_method",
)

_lock = threading.Lock()
_cache: "OrderedDict[str, bytes]" = OrderedDict()
_total_bytes = 0


def enabled() -> bool:
    return os.environ.get("ADR_TTS_CACHE", "1") != "0"


def _max_entries() -> int:
    try:
        return max(1, int(os.environ.get("ADR_TTS_CACHE_MAX", DEFAULT_MAX_ENTRIES)))
    except ValueError:
        return DEFAULT_MAX_ENTRIES


def _max_bytes() -> int:
    try:
        return max(0, int(os.environ.get("ADR_TTS_CACHE_MB", 0)) * 1024 * 1024) \
            or DEFAULT_MAX_BYTES
    except ValueError:
        return DEFAULT_MAX_BYTES


def make_key(req: dict) -> str:
    payload = {f: req.get(f) for f in _KEY_FIELDS}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def get(key: str) -> bytes | None:
    if not enabled():
        return None
    with _lock:
        if key not in _cache:
            return None
        _cache.move_to_end(key)  # 命中即续期
        return _cache[key]


def put(key: str, data: bytes) -> None:
    global _total_bytes
    if not enabled() or not data:
        return
    if len(data) > _max_bytes():  # 单条超上限: 不缓存, 免逐出死循环
        return
    with _lock:
        if key in _cache:
            _total_bytes -= len(_cache[key])
            del _cache[key]
        _cache[key] = data
        _total_bytes += len(data)
        while _cache and (_total_bytes > _max_bytes()
                          or len(_cache) > _max_entries()):
            _, old = _cache.popitem(last=False)  # LRU 逐出最旧
            _total_bytes -= len(old)


def clear() -> None:
    global _total_bytes
    with _lock:
        _cache.clear()
        _total_bytes = 0
