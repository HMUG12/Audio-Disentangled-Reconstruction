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
- ADR_T2S_CUDAGRAPH / ADR_T2S_STATIC_KV / ADR_T2S_CUDAGRAPH_BUCKETS — 是写默认值
  (setdefault) 的引擎启动行为, 留在 gsv_engine。

Policy (core/policy.py, v1.3.x) 只会往 env 写 ADR_TORCH_THREADS /
ADR_TTS_STREAM_HEAD / ADR_TTS_SPLIT_MAX, 且不覆盖显式 env; 本文件读取口径不变。

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


# ===== 低配加速 (批次49, m10 C 节) =====
def tts_device() -> str:
    """ADR_TTS_DEVICE — 引擎设备走向: auto (默认) / cuda / cpu。

    仅在 EngineConfig.device=="auto" 时生效 (config 显式传参优先);
    非法值回退 auto。低配/核显机器强制 CPU 排障用, 壳设置中心注入。
    """
    raw = (os.environ.get("ADR_TTS_DEVICE") or "").strip().lower()
    return raw if raw in ("auto", "cuda", "cpu") else "auto"


def _pos_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, default))
        return v if v > 0 else default
    except ValueError:
        return default


def stream_head_chars() -> int:
    """ADR_TTS_STREAM_HEAD — 流式首段最大字数 (默认 14, 批次9)。

    Policy conservative-v1 慢机可自动收短; 只改切分位置, 不改模型分布。"""
    return _pos_int("ADR_TTS_STREAM_HEAD", 14)


def split_max_chars() -> int:
    """ADR_TTS_SPLIT_MAX — 无句读长段细分字数 (默认 40, 批次20)。"""
    return _pos_int("ADR_TTS_SPLIT_MAX", 40)


def runtime_autotune() -> bool:
    """ADR_RUNTIME_AUTOTUNE — Runtime 闭环是否自动微调 (默认开; 0/false/no/off 只监控)。
    微调范围仅限流式分块 (不影响音质), 显式 env 永远优先。"""
    return os.environ.get("ADR_RUNTIME_AUTOTUNE", "1").strip().lower() not in (
        "0", "false", "no", "off")


def torch_threads() -> int:
    """ADR_TORCH_THREADS — torch.set_num_threads 线程数; 0/未设置/非法 = 默认
    (PyTorch 自按物理核)。低配核少机器 (如 4 核) 跑满会抢系统资源, 手动
    压线程可减争抢; >0 时引擎 _lazy_init 早期设置一次。"""
    try:
        return max(0, int(os.environ.get("ADR_TORCH_THREADS", 0)))
    except ValueError:
        return 0


# ===== 声音安全 (批次62, m10 方案 C) =====
def agree_terms() -> bool:
    """ADR_AGREE_TERMS — 无 UI 形态条款 env 同意入口 (1/true/yes/on)。"""
    return os.environ.get("ADR_AGREE_TERMS", "").strip().lower() in ("1", "true", "yes", "on")


def safety_audit_enabled() -> bool:
    """ADR_SAFETY_AUDIT=0 关闭审计日志, 其余 (含未设置) 开。"""
    return os.environ.get("ADR_SAFETY_AUDIT", "1") != "0"


def wav_metadata_enabled() -> bool:
    """ADR_WAV_METADATA=0 关闭分发侧 WAV AI 生成标识注入, 其余 (含未设置) 开。"""
    return os.environ.get("ADR_WAV_METADATA", "1") != "0"


def scheduler_max_concurrency() -> int:
    """调度层并发预算 (ADR_SCHED_MAX_CONCURRENCY, 批次63 第三子批 m10 ④)。

    v1.3.0 交付 N=1 硬预算 (默认); >=2 为 N=2 固定份额 (1.3.x) 预留 —
    真并行需等引擎内连续批处理内核, 当前 >1 仅放宽准入, 合成仍被引擎锁串行。
    非数字 / 小于 1 一律回落 1。
    """
    try:
        return max(1, int(os.environ.get("ADR_SCHED_MAX_CONCURRENCY", "1")))
    except ValueError:
        return 1
