"""GSV 运行时支撑 (批次26 自 gsv_engine 拆出): 流式句级缓存。

只放无引擎依赖的纯运行时状态与工具; 引擎本体/单例/预热状态机仍在
gsv_engine (单向依赖本模块)。gsv_engine 顶部 re-export, 既有访问面
(tests 直接用 gsv_engine._SEG_CACHE / _seg_cache_clear 等) 不变。

批次23: 流式句级缓存 (进程内内存 LRU, 不落盘)
服务层整体缓存 (tts_cache, 批次22) 兜底跨重启; 这里按"段"缓存 int16 PCM —
同文本重复播报/跨请求重发时已合成句子直接回放, 跳过 GPU 前向。PCM 体积大
且 GPU 前向才是瓶颈, 内存 LRU 性价比最高。key 不含 seed (播报一致性, 同
文本不同采样轮次返回相同音频, 与批次21整体缓存语义一致)。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from typing import Optional

_SEG_CACHE: "OrderedDict[str, tuple[bytes, int]]" = OrderedDict()
_SEG_LOCK = threading.Lock()
_SEG_BYTES = 0   # 当前缓存占用量 (int16 PCM 字节)


def _seg_cache_enabled() -> bool:
    return os.environ.get("ADR_SEG_CACHE", "1") != "0"


def _seg_cache_key(seg: str, ref_audio: str, prompt_text: str,
                   text_lang: str, prompt_lang: str, split_method: str,
                   top_k: int, top_p: float, temperature: float,
                   speed_factor: float,
                   t2s_weights: Optional[str], vits_weights: Optional[str]) -> str:
    payload = json.dumps(
        [seg, ref_audio, prompt_text, text_lang, prompt_lang, split_method,
         top_k, top_p, temperature, speed_factor, t2s_weights, vits_weights],
        ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _seg_cache_get(key: str):
    with _SEG_LOCK:
        item = _SEG_CACHE.get(key)
        if item is not None:
            _SEG_CACHE.move_to_end(key)   # LRU 触碰
        return item


def _seg_cache_put(key: str, pcm: bytes, sr: int):
    global _SEG_BYTES
    try:
        max_items = max(0, int(os.environ.get("ADR_SEG_CACHE_MAX", "128")))
    except ValueError:
        max_items = 128
    try:
        max_bytes = max(0, int(os.environ.get("ADR_SEG_CACHE_MB", "256"))) * 1024 * 1024
    except ValueError:
        max_bytes = 256 * 1024 * 1024
    if max_items <= 0 or max_bytes <= 0:
        return
    with _SEG_LOCK:
        _SEG_CACHE[key] = (pcm, sr)
        _SEG_BYTES += len(pcm)
        # 双上限逐出 (条数/字节); 单段超上限时把自己弹空, 自然不缓存
        while _SEG_CACHE and (len(_SEG_CACHE) > max_items
                              or _SEG_BYTES > max_bytes):
            _, (old_pcm, _) = _SEG_CACHE.popitem(last=False)
            _SEG_BYTES -= len(old_pcm)


def _seg_cache_clear():
    global _SEG_BYTES
    with _SEG_LOCK:
        _SEG_CACHE.clear()
        _SEG_BYTES = 0
