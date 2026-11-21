"""TTS 合成结果 LRU 缓存 (批次21): 重复文本秒回、省 GPU。

播报 / LLM 对话场景里同一文本常被反复请求 (重试、多端播放、调试),
每次都跑完整 GSV 前向 (~数秒) 是纯浪费。这里在 v2 非流式路径上
加一层 LRU:

- key: 请求参数指纹 (sha1)。批次88 (API-02): seed 入 key — 与段缓存
  (``_seg_cache_key``) 语义对齐; 显式 seed 可复现、不同 seed 不再互相
  命中。默认 seed=-1 的播报场景 key 不变, 同文本仍复用同一次合成。
- 上限: 32 条 / 256MB (ADR_TTS_CACHE_MAX / ADR_TTS_CACHE_MB 可调)
- 开关: ADR_TTS_CACHE=0 关闭 (默认开)
- 流式响应不缓存 (分块字节契约, 复用价值低)

持久化 (批次22): 结果同步落盘 data/tts_cache/<key>.<ext> — 服务重启
后惰性扫描目录按 mtime 新→旧重建 LRU, 重启不清缓存。单文件读入
内存延迟到首次命中 (盘上只记文件名), 启动零读盘成本。
目录可用 ADR_TTS_CACHE_DIR 覆盖 (测试隔离用)。

批次36: env 读取单源化 — 开关/目录/上限委托 adr/core/settings.py,
本模块只保留缓存数据结构与算法。

纯标准库, 线程安全 (tts_handle 的查/填跑在事件循环线程, 但合成
在线程池, 用锁保护避免并发交错下的 dict 竞态)。

批次79 (#5): ``_lock`` 只保护 ``_cache`` / ``_total_bytes``; 目录扫描、
读盘、写盘、删盘全部移到锁外 (单文件可达 256MB, 持锁 IO 会把并发请求
一起冻住), v2 非流式路径的 get 也改走 ``asyncio.to_thread``。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

from adr.core import settings

# 缓存格式版本 (批次69): 音频编码参数 (pack_audio 码率/重采样) 或 _KEY_FIELDS
# 集合变化时递增 — make_key 参与哈希, 且磁盘文件名带 v<版本>- 前缀,
# 启动扫描丢弃旧版本条目, 避免升级后仍命中旧格式音频。
CACHE_VERSION = 3
_PREFIX = f"v{CACHE_VERSION}-"

# 参与 key 的请求字段 (批次88: seed 入 key, 见模块 docstring)
# fragment_interval 烤进输出音频 (句末静音长度), 必须参与 key (批次33)
_KEY_FIELDS = (
    "text", "ref_audio_path", "prompt_text", "text_lang", "prompt_lang",
    "top_k", "top_p", "temperature", "speed_factor", "fragment_interval",
    "t2s_weights", "vits_weights", "media_type", "text_split_method", "seed",
)

_lock = threading.Lock()        # 只保护下面三个结构, 不做任何磁盘 IO (批次79)
_scan_lock = threading.Lock()   # 串行化首次磁盘扫描, 避免两次扫描重复登记
# key -> (bytes, 盘上文件名 | None, 记账字节数)。bytes 为 b"" 且有文件名 = 磁盘条目未读入
# 记账字节数为条目真实大小: 磁盘条目按落盘文件 size (os.path.getsize) 记,
# 内存条目按 len(data) 记; 逐出/剔除一律按记录值回减, 防 _total_bytes 漂移
_cache: "OrderedDict[str, tuple[bytes, str | None, int]]" = OrderedDict()
_total_bytes = 0
_loaded = False  # 缓存目录是否已扫描重建 (惰性, 首次 get/put 触发)


def enabled() -> bool:
    return settings.tts_cache_enabled()


def _dir() -> Path:
    return settings.tts_cache_dir()


def _max_entries() -> int:
    return settings.tts_cache_max_entries()


def _max_bytes() -> int:
    return settings.tts_cache_max_bytes()


def _max_age_sec() -> float:
    return settings.tts_cache_ttl_days() * 86400.0


def _file_stamp(path) -> list | None:
    """权重文件指纹 [mtime_ns, size] (批次99 / P3-01)。

    只把路径放进 key 时, 训练/音色包在同名路径上换了权重仍会命中旧音频。
    """
    if not path:
        return None
    try:
        st = os.stat(str(path))
    except OSError:
        return None
    return [st.st_mtime_ns, st.st_size]


def make_key(req: dict) -> str:
    payload = {f: req.get(f) for f in _KEY_FIELDS}
    payload["_cache_version"] = CACHE_VERSION
    # 批次99 (P3-01): 权重内容指纹入 key — 换权后不再命中旧结果
    payload["_weights_stamp"] = [_file_stamp(req.get("t2s_weights")),
                                 _file_stamp(req.get("vits_weights"))]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _fname(key: str, ext: str) -> str:
    """盘上文件名: v<版本>-<key>.<ext> (版本前缀供扫描识别/丢弃旧格式)。"""
    return f"{_PREFIX}{key}.{ext}"


def _unlink_quiet(p: Path) -> None:
    """删盘上文件, 吞 OSError (批次66: Windows 上杀软/播放器占用会抛
    PermissionError; 删不掉只是多占点盘, 不能让已合成的音频跟着失败)。"""
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def _load_disk_once() -> None:
    """惰性扫描缓存目录重建内存 LRU (服务重启后缓存仍可用)。

    批次79 (#5): 枚举目录 / stat / 删旧文件 / 删超限副本全在 ``_lock`` 外,
    只有 LRU 登记与字节记账进锁 — 一次重启扫描不再长时间冻住所有请求。
    按 mtime 新→旧回填: 盘上只登记 (b"", 文件名, size), 字节延迟到首次
    get 再读 — 启动零读盘; 超出条数/字节上限的旧文件直接删。
    盘上文件就是 put 时写下的完整字节, st_size == len(data), 统计口径一致。
    """
    global _loaded, _total_bytes
    with _lock:
        if _loaded:  # 批次66: 惰性守卫 — 只扫一次, 否则满载后重扫会删掉活条目的盘上副本
            return
    with _scan_lock:
        with _lock:
            if _loaded:  # 等锁期间别的线程已扫完
                return
        d = _dir()
        entries: list[Path] = []
        if d.is_dir():
            try:
                entries = [p for p in d.iterdir() if p.is_file()]
            except OSError:
                entries = []
        files = []
        for p in entries:
            if p.name.endswith(".tmp"):
                _unlink_quiet(p)  # 批次66: 上次写盘失败残留的半截临时文件, 扫描时清掉
            elif not p.name.startswith(_PREFIX):
                _unlink_quiet(p)  # 批次69: 旧缓存版本/未知格式 → 丢弃, 不再命中
            else:
                files.append(p)
        # 一次 stat 同时取 mtime + size; 并发下文件可能刚被逐出删除
        # (stat 抛 FileNotFoundError), 容错跳过当作不存在
        statted: list[tuple[float, int, Path]] = []
        for p in files:
            try:
                st = p.stat()
            except OSError:
                continue
            statted.append((st.st_mtime, st.st_size, p))
        statted.sort(key=lambda t: t[0], reverse=True)
        max_b, max_n = _max_bytes(), _max_entries()
        # S11: 合成音频含用户文本语音, 盘上缓存按 TTL 过期 (默认 7 天, 0=不过期)
        ttl = _max_age_sec()
        now = time.time()
        doomed: list[Path] = []
        with _lock:
            for mtime, size, p in statted:
                if ttl > 0 and now - mtime > ttl:
                    doomed.append(p)
                    continue
                key = p.stem[len(_PREFIX):]   # 批次69: 去掉版本前缀得到缓存键
                if key in _cache:  # 批次66: 已在内存的条目先跳过, 不能按超限误删 / 重复记账
                    continue
                if len(_cache) >= max_n or _total_bytes + size > max_b:
                    doomed.append(p)  # 超限旧文件: 锁外删盘
                    continue
                _cache[key] = (b"", p.name, size)
                _total_bytes += size
            _loaded = True
        for p in doomed:
            _unlink_quiet(p)


def get(key: str) -> bytes | None:
    if not enabled():
        return None
    global _total_bytes
    _load_disk_once()
    with _lock:
        item = _cache.get(key)
        if item is None:
            return None
        data, fname, size = item
        _cache.move_to_end(key)  # 命中即续期
        if data or fname is None:
            return data
    # 批次79 (#5): 磁盘条目首次命中 — 读盘移出全局锁 (最大 256MB),
    # 读盘期间其它 key 的 get/put 不再跟着排队。
    try:
        loaded = (_dir() / fname).read_bytes()
    except OSError:
        loaded = None
    with _lock:
        cur = _cache.get(key)
        # 只回填未被替换/清除的同一磁盘条目: 并发 put 或 clear 后不复活旧数据
        if cur is not None and cur[1] == fname and not cur[0]:
            if loaded is None:
                del _cache[key]        # 盘上文件已丢: 视为未命中
                _total_bytes -= cur[2]  # 按记录值回减, 防账目漂移
            else:
                _cache[key] = (loaded, fname, cur[2])
    return loaded


def put(key: str, data: bytes, ext: str = "bin") -> None:
    """写缓存: 内存 LRU + 落盘 (临时文件 + 原子替换防并发读半截)。

    ext: 文件后缀 (media_type), 仅用于落盘可辨识, key 已含 media_type 指纹。
    """
    global _total_bytes
    if not enabled() or not data:
        return
    if len(data) > _max_bytes():  # 单条超上限: 不缓存, 免逐出死循环
        return
    _load_disk_once()
    # 批次79 (#5): 落盘 IO (write/replace/getsize) 全在锁外。同 key 并发写用
    # 唯一 .tmp 名避免互相踩到半截文件; 后写者覆盖先写者, 内容等价无需细粒度锁。
    base = _dir()
    fname: str | None = _fname(key, ext)
    size = len(data)
    tmp: Path | None = None
    try:
        base.mkdir(parents=True, exist_ok=True)
        tmp = base / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, base / fname)
        tmp = None  # 已原子落到正式文件名, 后续失败无需再清 .tmp
        # 记录真实落盘大小作为记账值 (与磁盘重建扫描口径一致);
        # 极端并发下文件刚被清理会 FileNotFoundError, 退回内存字节数
        try:
            size = os.path.getsize(base / fname)
        except OSError:
            size = len(data)
    except OSError:
        fname = None
        if tmp is not None:
            _unlink_quiet(tmp)  # 批次66: 写一半/替换失败时清掉 .tmp, 免残留占盘
    doomed: list[Path] = []
    with _lock:
        old = _cache.get(key)
        if old is not None:
            _total_bytes -= old[2]  # 按旧条目记录值回减
            del _cache[key]
        _cache[key] = (data, fname, size)
        _total_bytes += size
        while _cache and (_total_bytes > _max_bytes()
                          or len(_cache) > _max_entries()):
            _, (_, old_name, old_size) = _cache.popitem(last=False)  # LRU 逐出最旧
            _total_bytes -= old_size  # 按记录值回减 (磁盘条目 data=b"", 不能用 len)
            if old_name is not None:
                doomed.append(base / old_name)
    for p in doomed:  # 批次79 (#5): 逐出删盘在锁外 (批次66 起: 删失败不能冒泡)
        _unlink_quiet(p)


def clear() -> None:
    """全清: 内存 + 盘上缓存文件 (测试隔离 / 手动清空)。

    批次79 (#5): 先锁内清结构 (后续读写立刻视为空/未扫描), 再锁外删盘 —
    删盘可能被占用或杀软拖慢, 不再阻塞其它请求的缓存读写。
    """
    global _total_bytes, _loaded
    with _lock:
        _cache.clear()
        _total_bytes = 0
        _loaded = False
    d = _dir()
    if not d.is_dir():
        return
    try:
        entries = list(d.iterdir())
    except OSError:
        return
    for p in entries:
        if p.is_file():
            _unlink_quiet(p)
