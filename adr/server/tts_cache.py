"""TTS 合成结果 LRU 缓存 (批次21): 重复文本秒回、省 GPU。

播报 / LLM 对话场景里同一文本常被反复请求 (重试、多端播放、调试),
每次都跑完整 GSV 前向 (~数秒) 是纯浪费。这里在 v2 非流式路径上
加一层 LRU:

- key: 请求参数指纹 (sha1); seed 不入 key — 播报一致性优先,
  同文本复用同一次合成, 避免"重试一次换一个人声"的观感
- 上限: 32 条 / 256MB (ADR_TTS_CACHE_MAX / ADR_TTS_CACHE_MB 可调)
- 开关: ADR_TTS_CACHE=0 关闭 (默认开)
- 流式响应不缓存 (分块字节契约, 复用价值低)

持久化 (批次22): 结果同步落盘 data/tts_cache/<key>.<ext> — 服务重启
后惰性扫描目录按 mtime 新→旧重建 LRU, 重启不清缓存。单文件读入
内存延迟到首次命中 (盘上只记文件名), 启动零读盘成本。
目录可用 ADR_TTS_CACHE_DIR 覆盖 (测试隔离用)。

纯标准库, 线程安全 (tts_handle 的查/填跑在事件循环线程, 但合成
在线程池, 用锁保护避免并发交错下的 dict 竞态)。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MAX_ENTRIES = 32
DEFAULT_MAX_BYTES = 256 * 1024 * 1024

# 参与 key 的请求字段 (seed 有意排除, 见模块 docstring)
# fragment_interval 烤进输出音频 (句末静音长度), 必须参与 key (批次33)
_KEY_FIELDS = (
    "text", "ref_audio_path", "prompt_text", "text_lang", "prompt_lang",
    "top_k", "top_p", "temperature", "speed_factor", "fragment_interval",
    "t2s_weights", "vits_weights", "media_type", "text_split_method",
)

_lock = threading.Lock()
# key -> (bytes, 盘上文件名 | None)。bytes 为 b"" 且有文件名 = 磁盘条目未读入
_cache: "OrderedDict[str, tuple[bytes, str | None]]" = OrderedDict()
_total_bytes = 0
_loaded = False  # 缓存目录是否已扫描重建 (惰性, 首次 get/put 触发)


def enabled() -> bool:
    return os.environ.get("ADR_TTS_CACHE", "1") != "0"


def _dir() -> Path:
    env = os.environ.get("ADR_TTS_CACHE_DIR")
    return Path(env) if env else REPO_ROOT / "data" / "tts_cache"


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


def _disk_load_locked() -> None:
    """惰性扫描缓存目录重建内存 LRU (服务重启后缓存仍可用)。

    按 mtime 新→旧回填: 盘上只登记 (b"", 文件名), 字节延迟到首次
    get 再读 — 启动零读盘; 超出条数/字节上限的旧文件直接删。
    盘上文件就是 put 时写下的完整字节, st_size == len(data), 统计口径一致。
    """
    global _loaded, _total_bytes
    _loaded = True
    d = _dir()
    if not d.is_dir():
        return
    try:
        files = [p for p in d.iterdir()
                 if p.is_file() and not p.name.endswith(".tmp")]
    except OSError:
        return
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    max_b, max_n = _max_bytes(), _max_entries()
    for p in files:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if len(_cache) >= max_n or _total_bytes + size > max_b:
            p.unlink(missing_ok=True)  # 超限旧文件: 删盘上副本
            continue
        if p.stem in _cache:
            continue
        _cache[p.stem] = (b"", p.name)
        _total_bytes += size


def get(key: str) -> bytes | None:
    if not enabled():
        return None
    with _lock:
        _disk_load_locked()
        item = _cache.get(key)
        if item is None:
            return None
        data, fname = item
        if not data and fname is not None:  # 磁盘条目: 首次命中读入
            try:
                data = (_dir() / fname).read_bytes()
            except OSError:
                del _cache[key]  # 盘上文件已丢: 视为未命中
                return None
            _cache[key] = (data, fname)
        _cache.move_to_end(key)  # 命中即续期
        return data


def put(key: str, data: bytes, ext: str = "bin") -> None:
    """写缓存: 内存 LRU + 落盘 (临时文件 + 原子替换防并发读半截)。

    ext: 文件后缀 (media_type), 仅用于落盘可辨识, key 已含 media_type 指纹。
    """
    global _total_bytes
    if not enabled() or not data:
        return
    if len(data) > _max_bytes():  # 单条超上限: 不缓存, 免逐出死循环
        return
    with _lock:
        _disk_load_locked()
        # 落盘 (盘不可写则退化为纯内存, 不影响服务)
        fname: str | None = f"{key}.{ext}"
        try:
            d = _dir()
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / (key + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, d / fname)
        except OSError:
            fname = None
        old = _cache.get(key)
        if old is not None:
            _total_bytes -= len(old[0])
            del _cache[key]
        _cache[key] = (data, fname)
        _total_bytes += len(data)
        while _cache and (_total_bytes > _max_bytes()
                          or len(_cache) > _max_entries()):
            _, (old_data, old_name) = _cache.popitem(last=False)  # LRU 逐出最旧
            _total_bytes -= len(old_data)
            if old_name is not None:
                (_dir() / old_name).unlink(missing_ok=True)


def clear() -> None:
    """全清: 内存 + 盘上缓存文件 (测试隔离 / 手动清空)。"""
    global _total_bytes, _loaded
    with _lock:
        _cache.clear()
        _total_bytes = 0
        _loaded = False
        d = _dir()
        if d.is_dir():
            for p in d.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                except OSError:
                    pass
