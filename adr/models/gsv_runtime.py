"""GSV 运行时支撑 (批次26 自 gsv_engine 拆出): 流式句级缓存。

只放无引擎依赖的纯运行时状态与工具; 引擎本体/单例/预热状态机仍在
gsv_engine (单向依赖本模块)。gsv_engine 顶部 re-export, 既有访问面
(tests 直接用 gsv_engine._SEG_CACHE / _seg_cache_clear 等) 不变。

批次23: 流式句级缓存 (进程内内存 LRU, 不落盘)
服务层整体缓存 (tts_cache, 批次22) 兜底跨重启; 这里按"段"缓存 int16 PCM —
同文本重复播报/跨请求重发时已合成句子直接回放, 跳过 GPU 前向。PCM 体积大
且 GPU 前向才是瓶颈, 内存 LRU 性价比最高。key 只含段实际生效的 seed
(批次85 / CODE_REVIEW #18: 首段 head_seed, 其余段 -1), 未指定 seed 时仍
与历史一致 — 同文本不同采样轮次返回相同音频 (播报一致性)。

批次85 (#18): key 增加参考音频版本指纹 (路径 + mtime_ns + size)。只看
路径时, 同名档案重建 ref.wav / 覆盖写入后仍会回放旧缓存的人声。

批次36: env 读取单源化 — ADR_SEG_CACHE* 委托 adr/core/settings.py。
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Optional

from adr.core import settings

_SEG_CACHE: "OrderedDict[str, tuple[bytes, int]]" = OrderedDict()
_SEG_LOCK = threading.Lock()
_SEG_BYTES = 0   # 当前缓存占用量 (int16 PCM 字节)


def _seg_cache_enabled() -> bool:
    return settings.seg_cache_enabled()


def _seg_cache_key(seg: str, ref_audio: str, prompt_text: str,
                   text_lang: str, prompt_lang: str, split_method: str,
                   top_k: int, top_p: float, temperature: float,
                   speed_factor: float, fragment_interval: float,
                   t2s_weights: Optional[str], vits_weights: Optional[str],
                   ref_stamp: tuple = (), seed: int = -1,
                   t2s_stamp: tuple = (), vits_stamp: tuple = ()) -> str:
    """段缓存 key。ref_stamp 为参考音频文件指纹 (路径, mtime_ns, size),
    seed 为该段实际生效的采样种子 (首段 head_seed, 其余段 -1)。

    批次88 (CACHE-01): 权重也带 (路径, mtime_ns, size) 指纹 — 训练产物
    覆盖同路径权重 / 音色包 overwrite 后 mtime/size 变化即换 key,
    不再回放旧人声。
    """
    payload = json.dumps(
        [seg, ref_audio, prompt_text, text_lang, prompt_lang, split_method,
         top_k, top_p, temperature, speed_factor, fragment_interval,
         t2s_weights, vits_weights, list(ref_stamp), seed,
         list(t2s_stamp), list(vits_stamp)],
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
    max_items = settings.seg_cache_max_items()
    max_bytes = settings.seg_cache_max_bytes()
    if max_items <= 0 or max_bytes <= 0:
        return
    with _SEG_LOCK:
        # 批次85 (#26): 同 key 覆盖必须先回减旧值字节, 否则 _SEG_BYTES 虚高,
        # 字节上限会提前逐出仍然有效的缓存条目。
        old = _SEG_CACHE.pop(key, None)
        if old is not None:
            _SEG_BYTES -= len(old[0])
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
