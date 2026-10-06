"""服务与运行时开关配置单源 (批次36)。

所有 ``ADR_`` 前缀环境变量的**变量名、默认值、解析口径**只在本文件出现
一次; 各面 (server / models) 的裸 ``os.environ`` 读取统一改调这里的具名
函数。排查配置问题时只看这一个文件。

刻意做成**函数式读取**而非启动快照:
1. tests 大量 monkeypatch.setenv 后即时调 create_app / 缓存函数
   (test_server_api / test_server_security / test_gsv_seg_cache),
   快照会让 setenv 静默失效;
2. 运行中切腿语义有意保留 — gsv_engine 的 empty_cache 节流实时读 env
   支持同进程 A/B 基准 (批次33)。

不在此收编:
- ADR_DATA_DIR / ADR_CONFIG / ADR_PRESET — 已有各自单点 (core/config.py);
- rvc_engine 的 weight_root 等 / HF_HUB_CACHE — 第三方库桥接变量, 非配置面;
- ADR_T2S_CUDAGRAPH — 是写默认值 (setdefault) 的引擎启动行为, 留在 gsv_engine。

各函数解析口径与被替换的原地读取**逐字等价** (含容错回退), 不统一不重构。
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_log = logging.getLogger("adr.settings")

# ===== 默认值 (原散布点的字面常量, 就地收编) =====
DEFAULT_TTS_CACHE_MAX_ENTRIES = 32
DEFAULT_TTS_CACHE_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_SEG_CACHE_MAX_ITEMS = 128
DEFAULT_SEG_CACHE_MAX_MB = 256
DEFAULT_FRAGMENT_INTERVAL = 0.3


# ===== 服务面 (server/app.py + server/__main__.py) =====
def api_keys() -> list[str]:
    """ADR_TTS_API_KEY — 逗号分隔; strip 后过滤空串 (auth 中间件本就
    只认非空 key, 行为等价)。未设置返回空列表 = 鉴权关闭。"""
    raw = os.environ.get("ADR_TTS_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def has_api_key() -> bool:
    """ADR_TTS_API_KEY 是否设置 (strip 非空) — __main__ 启动警告判据。"""
    return bool(os.environ.get("ADR_TTS_API_KEY", "").strip())


def default_profile() -> str | None:
    """ADR_TTS_DEFAULT_PROFILE — 空串/未设置归一为 None。"""
    return os.environ.get("ADR_TTS_DEFAULT_PROFILE") or None


# ===== TTS 结果缓存 (server/tts_cache.py) =====
def tts_cache_enabled() -> bool:
    """ADR_TTS_CACHE=0 关闭, 其余 (含未设置) 开。"""
    return os.environ.get("ADR_TTS_CACHE", "1") != "0"


def tts_cache_dir() -> Path:
    """ADR_TTS_CACHE_DIR 覆盖落盘目录, 默认 <repo>/data/tts_cache。"""
    env = os.environ.get("ADR_TTS_CACHE_DIR")
    return Path(env) if env else REPO_ROOT / "data" / "tts_cache"


def tts_cache_max_entries() -> int:
    try:
        return max(1, int(os.environ.get(
            "ADR_TTS_CACHE_MAX", DEFAULT_TTS_CACHE_MAX_ENTRIES)))
    except ValueError:
        return DEFAULT_TTS_CACHE_MAX_ENTRIES


def tts_cache_max_bytes() -> int:
    """注意口径: MB=0 (含非法值) 回退默认上限, 与条数上限语义不同。"""
    try:
        return max(0, int(os.environ.get("ADR_TTS_CACHE_MB", 0)) * 1024 * 1024) \
            or DEFAULT_TTS_CACHE_MAX_BYTES
    except ValueError:
        return DEFAULT_TTS_CACHE_MAX_BYTES


# ===== 流式句级缓存 (models/gsv_runtime.py) =====
def seg_cache_enabled() -> bool:
    """ADR_SEG_CACHE=0 关闭, 其余 (含未设置) 开。"""
    return os.environ.get("ADR_SEG_CACHE", "1") != "0"


def seg_cache_max_items() -> int:
    """0 = 禁用条数上限 (仅字节上限逐出)。"""
    try:
        return max(0, int(os.environ.get(
            "ADR_SEG_CACHE_MAX", DEFAULT_SEG_CACHE_MAX_ITEMS)))
    except ValueError:
        return DEFAULT_SEG_CACHE_MAX_ITEMS


def seg_cache_max_bytes() -> int:
    """MB=0 = 禁用字节上限 (与 tts_cache 的 0→默认 语义不同, 勿"统一")。"""
    try:
        return max(0, int(os.environ.get(
            "ADR_SEG_CACHE_MB", DEFAULT_SEG_CACHE_MAX_MB))) * 1024 * 1024
    except ValueError:
        return DEFAULT_SEG_CACHE_MAX_MB * 1024 * 1024


# ===== 引擎运行时开关 (models/gsv_engine.py) =====
def fragment_interval_default() -> float:
    """ADR_TTS_FRAGMENT_INTERVAL — None 显式传参时的句末静音默认秒数。

    批次41b: 非法值 (非数值/负数/NaN/Inf) 不再让 float() 直接崩或直接采用,
    改为打 warning 并回退默认值 (服务不因一个环境变量写错而起不来)。
    """
    raw = os.environ.get("ADR_TTS_FRAGMENT_INTERVAL", "").strip()
    if not raw:
        return DEFAULT_FRAGMENT_INTERVAL
    try:
        val = float(raw)
    except ValueError:
        _log.warning(
            f"ADR_TTS_FRAGMENT_INTERVAL={raw!r} 不是合法数值, "
            f"回退默认 {DEFAULT_FRAGMENT_INTERVAL}"
        )
        return DEFAULT_FRAGMENT_INTERVAL
    if not math.isfinite(val) or val < 0:
        _log.warning(
            f"ADR_TTS_FRAGMENT_INTERVAL={raw!r} 非法 (需为非负有限数), "
            f"回退默认 {DEFAULT_FRAGMENT_INTERVAL}"
        )
        return DEFAULT_FRAGMENT_INTERVAL
    return val


def keep_tqdm() -> bool:
    """ADR_TTS_KEEP_TQDM=1 保留 AR 解码 tqdm 渲染 (默认静音, 批次33)。"""
    return os.environ.get("ADR_TTS_KEEP_TQDM", "") == "1"


def keep_empty_cache() -> bool:
    """ADR_TTS_KEEP_EMPTY_CACHE=1 回退每句 empty_cache (默认节流, 批次33)。"""
    return os.environ.get("ADR_TTS_KEEP_EMPTY_CACHE", "") == "1"
