"""GPT-SoVITS 官方预训练套壳引擎 (保底路线 - 说话克隆腿)。

定位: 零样本音色克隆 TTS 基线 (5 秒参考音频 → 克隆朗读)。
与 DiffSinger 引擎互补: 那个管唱歌, 这个管说话。

权重 (HF lj1995/GPT-SoVITS):
- s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt (GPT 语义)
- s2G488k.pth / s2D488k.pth (VITS 声学)
- chinese-hubert-base / chinese-roberta-wwm-ext-large (特征)

设计注记 (不能完全抄, 快速克隆优势不能丢):
- 引擎只做推理保底; 数据预处理 / QLoRA 微调 / WebUI 流程仍是 ADR 自己的
- GPT-SoVITS 官方支持 s1/s2 微调, 后续可由 ADR 训练管线接管定制音色
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import pickle
import sys
import threading
import time
import traceback
import zipfile
from collections import OrderedDict
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 批次26: 段缓存拆至 gsv_runtime, 此处 re-export 保持既有访问面不变
# (tests 直接用 gsv_engine._SEG_CACHE / _seg_cache_clear 等)
from adr.models.gsv_runtime import (  # noqa: F401
    _SEG_CACHE,
    _seg_cache_clear,
    _seg_cache_enabled,
    _seg_cache_get,
    _seg_cache_key,
    _seg_cache_put,
)
from adr.core import settings
from adr.core.audit import log_event
from adr.core.exceptions import EngineStalledError
from adr.core.scheduler import SynthesisScheduler
from adr.models.gsv_coalescer import _Coalescer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GSV_DIR = REPO_ROOT / "third_party" / "gpt_sovits"

# Track B 收口: chdir 是进程级全局状态, GSV third_party 代码用相对路径
# 定位权重, chdir 无法消除, 只能串行化整个 chdir 窗口 — 否则线程 A 刚
# 进入 GSV 上下文, 线程 B 的 finally 恢复 cwd 会把 A 踩进错误目录。
# RLock: synthesize_stream 的 _gsv_context 内嵌套 _ensure_weights 再开
# _gsv_context, 必须可重入。
_CHDIR_LOCK = threading.RLock()

# 批次82 (#1): 流式消费端断连后的清理预算。producer 可能卡在上游 run() 里
# 迟迟不退出, 消费端不能无限期等 _DONE (旧实现 q.get() 无超时 + join(30)
# 超时后直接返回, worker 仍持引擎锁存活)。超时即标记引擎降级, 下次重建模型。
_STREAM_DRAIN_SEC = 30.0
_STREAM_JOIN_SEC = 5.0

# 批次86 (CON-01): 引擎锁等待上限。卡死的 worker 会长期持有 _lock, 旧实现用
# 无超时 ``with self._lock`` → 后续请求全部永久排队, 表现为全站假死。
# 超时统一转 EngineStalledError (503 可重试), 进程保持可响应。
_LOCK_WAIT_SEC = 55.0
# 批次86 (CON-01): 流式消费端静默上限 — 上游 run() 这么久没有产出即判定卡死,
# 主动取消 + 降级, 让 finally 里的 _mark_stalled 能真正执行到。
_STREAM_IDLE_SEC = 120.0

# 批次90 (R3-CON-01): 合批 future 等待上限。合批 worker 卡在上游 run() 时
# future 永不完成, 旧实现 ``.result()`` 无超时 → 请求线程与准入槽永久挂起
# (anyio 线程池被逐个耗尽)。这里兜"future 永不完成"这一半; 单纯慢请求不会
# 走到 — 整轮拿不到 _engine_lock 时 55s 内已经 503。
_COALESCE_WAIT_SEC = 600.0

# 批次94 (R3-CON-02): 非流式 solo 合成静默上限。上游 run() 卡死时同步调用会让
# 请求线程永久停在 run() 里, 既不置降级标记, 看门狗与 /readyz 也全盲。这里把
# 调用挪到 daemon 线程 + 有界等待; 上限与合批 future 对齐, 长文本正常合成不受
# 影响 (超过该时长仍无返回视为上游卡死)。
_NONSTREAM_IDLE_SEC = 600.0

# 批次90 (R3-OBS-01 / CON-01): 进程级看门狗。卡死 worker 同时持有 _lock 与
# _CHDIR_LOCK, Python 线程无法安全强杀 → 进程内重建永远拿不到锁, 全站永久
# 503。唯一可靠收口是交给外部守护重启 (Docker restart policy / 桌面壳
# supervise 检测到退出后自动拉起)。ADR_ENGINE_WATCHDOG_SEC 为降级容忍秒数,
# 0 = 关闭; 看门狗只由服务入口启动, 库/测试调用不启用。
_WATCHDOG_POLL_SEC = 15.0
_WATCHDOG_DEFAULT_SEC = 300.0
_WATCHDOG_EXIT_CODE = 75          # EX_TEMPFAIL: 与崩溃退出码区分, 便于排障
_WATCHDOG_LOCK = threading.Lock()
_WATCHDOG_STARTED = False

# 批次87 (SEC-02): 权重安全校验结果缓存。旧实现每个带 t2s_weights/vits_weights
# 的请求都在线程池里整文件 torch.load + 慢路径 f.read() (峰值 2× 文件体积),
# 稳态调用/并发几次即 OOM。与 _ensure_weights 同一指纹 (路径 + mtime_ns +
# size): 文件被换掉指纹变化 → 自动重校验, 不会漏检。
_WEIGHTS_CHECKED: "OrderedDict[tuple[str, int, int], None]" = OrderedDict()
_WEIGHTS_CHECKED_LOCK = threading.Lock()
_WEIGHTS_CHECKED_MAX = 256


def is_cuda_oom(e: BaseException) -> bool:
    """异常是否 torch CUDA OOM (批次57 兼容性第二批)。不引 torch —
    按类型名 + 模块前缀判定, 供 API 层零成本分流 (引擎进程 torch 必已加载)。"""
    return (type(e).__name__ == "OutOfMemoryError"
            and getattr(type(e), "__module__", "").startswith("torch"))


def resolve_runtime_device(device: str) -> str:
    """设备可用性优先于用户选择 (批次122): 要求 cuda 但 CUDA 不可用时回落 cpu。

    CPU 包 + 设置中心里选了 "CUDA" 是必崩组合: GSV 的 ``init_t2s_weights`` 用
    ``torch.load(..., map_location=self.configs.device)`` 反序列化权重, CUDA
    不可用时 torch 直接抛 "Attempting to deserialize object on a CUDA device"
    (2026-11-14 实机首启报错)。这里只降不升: cpu 设置永不被改成 cuda。
    """
    if device != "cuda":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        logging.getLogger("adr").warning(
            "[gpu] 配置要求 CUDA, 但本轮子 (torch %s) 的 CUDA 不可用 — 已回落 CPU"
            " (CPU 包或驱动缺失时不要在设置里选 CUDA)", torch.__version__)
    except Exception:
        logging.getLogger("adr").warning("[gpu] torch 不可用 — 已回落 CPU")
    return "cpu"


def handle_cuda_oom(e: BaseException) -> str:
    """CUDA OOM 统一善后 (单点): 立即 empty_cache 释放碎片显存 — 否则显存
    碎片化后后续请求持续 OOM; 落警告日志; 返回用户可读提示 (API 层包 503/cuda_oom)。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logging.getLogger("adr").warning(
        "[oom] CUDA 显存不足: %s — 已自动释放缓存",
        (str(e).splitlines() or [""])[0])
    return ("GPU 显存不足 (CUDA out of memory): 已自动释放缓存。"
            "建议缩短合成文本、避免并发请求, 或在 设置 → 高级 → 设备 改为 cpu 后重试。")


# 批次74: 英文 G2P 离线依赖 (build_runtime 4c 随包到 python/nltk_data)
_NLTK_RESOURCES = ("corpora/cmudict", "taggers/averaged_perceptron_tagger_eng")

NLTK_MISSING_MESSAGE = (
    "英文合成依赖的 NLTK 语料缺失 (cmudict / averaged_perceptron_tagger): "
    "安装包未随附且当前离线、无法自动下载。请联网后重启应用自动补齐, "
    "或改用完整版安装包; 中文 / 日文 / 粤语合成不受影响。")


def nltk_missing_reason() -> str:
    """英文 G2P 依赖的 nltk 语料是否随包 (批次74)。齐全返回 "", 否则返回缺失清单。
    只查本地 nltk_data — 不联网、不抛异常 (供启动期体检与 API 层复用)。"""
    try:
        import nltk
    except Exception:
        return "nltk 未安装"
    missing = []
    for res in _NLTK_RESOURCES:
        try:
            nltk.data.find(res)
        except Exception:
            missing.append(res)
    return ", ".join(missing)


def is_nltk_missing(e: BaseException) -> bool:
    """异常是否由 nltk 语料缺失导致 (批次74)。按类型名 + 资源名判定,
    不 import nltk — 供 API 层零成本分流。"""
    if type(e).__name__ != "LookupError":
        return False
    msg = str(e).lower()
    return any(k in msg for k in ("cmudict", "averaged_perceptron", "nltk"))


def _nvidia_driver_version() -> Optional[str]:
    """nvidia-smi 查驱动版本 (批次57, 零依赖): 驱动未装/不在 PATH → None。
    排障关键: "驱动已装但 CUDA 不可用" = 驱动版本低于 torch 轮子所需。"""
    import subprocess
    try:
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, **kw)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip() or None
    except Exception:
        pass
    return None


def _weights_max_bytes() -> int:
    """权重文件大小上限 (批次86 / SEC-01): ADR_TTS_WEIGHTS_MAX_MB, 默认 2048MB。
    GSV 权重常见 500MB~1.5GB, 留足余量; 超限只可能是误传/攻击。"""
    try:
        mb = int(os.environ.get("ADR_TTS_WEIGHTS_MAX_MB", "2048"))
    except ValueError:
        mb = 2048
    return (mb if mb > 0 else 2048) * 1024 * 1024


def resolve_weights_path(weights_path: str) -> str:
    """权重路径 → 绝对路径 (批次99 / R4-API-01)。

    相对路径统一按 GSV_DIR 解析。此前预检按 GSV_DIR、实际加载按进程 cwd
    解析, 服务进程 cwd ≠ GSV 目录时 ``GPT_weights_v2/xxx.ckpt`` 这类文档化
    写法会"预检通过、加载 not found"。预检、换权、合成入口必须共用本函数。
    """
    path = Path(str(weights_path))
    if not path.is_absolute():
        path = GSV_DIR / path
    return str(path.resolve())


def _precheck_weights_path(weights_path: str) -> str:
    """整文件读入前的本地路径预检, 返回解析后的绝对路径 (批次86 / SEC-01)。

    v2 的 GET /tts 默认无鉴权, 旧实现把 query 里的路径直接丢给 torch.load /
    f.read(): 任意本地大文件可打爆内存, UNC 路径 (\\\\host\\share\\x) 还会让
    服务主动发起 SMB 连接。这里先拒绝 UNC/设备路径 + stat 校验 (存在、普通
    文件、大小上限), 通过后才进入 pickle 预检。

    相对路径解析统一走 resolve_weights_path (按 GSV 目录), 与 _ensure_weights
    实际加载时同一基准。
    """
    p = str(weights_path or "")
    if not p or "\x00" in p:
        raise ValueError("weights path is empty or invalid")
    if p.startswith(("\\\\", "//", "\\\\?\\", "\\\\.\\")):
        raise ValueError("UNC/device weights path is not allowed")
    path = Path(resolve_weights_path(p))
    if not path.is_file():
        raise ValueError(f"weights file not found or not a regular file: {path}")
    size = path.stat().st_size
    limit = _weights_max_bytes()
    if size > limit:
        raise ValueError(
            f"weights file too large: {size} bytes (limit {limit} bytes, "
            f"see ADR_TTS_WEIGHTS_MAX_MB)")
    return str(path)


def _weights_check_cached(stamp: tuple[str, int, int]) -> bool:
    """批次87 (SEC-02): 校验结果缓存命中? (路径 + mtime_ns + size 指纹)"""
    with _WEIGHTS_CHECKED_LOCK:
        return stamp in _WEIGHTS_CHECKED


def _weights_check_remember(stamp: tuple[str, int, int]) -> None:
    """记一次校验通过 (FIFO 上限 256, 防无界增长)。"""
    with _WEIGHTS_CHECKED_LOCK:
        _WEIGHTS_CHECKED[stamp] = None
        _WEIGHTS_CHECKED.move_to_end(stamp)
        while len(_WEIGHTS_CHECKED) > _WEIGHTS_CHECKED_MAX:
            _WEIGHTS_CHECKED.popitem(last=False)


def validate_weights_file(weights_path: str) -> None:
    """权重文件安全预检 (Track B 收口; 批次38 收编为引擎单点): 防恶意 pickle RCE。

    引擎热换最终走 GSV third_party 内部的 torch.load (无 weights_only,
    third_party 不可改), 恶意 pickle 文件会在其加载时执行任意代码。
    两段式校验 (批次43):

    1) 快路径 torch.load(weights_only=True) — 官方预训练等纯 torch 权重
       直接通过;
    2) 慢路径受限 pickle 扫描 — GPT-SoVITS 训练产物把 config 存成第三方
       utils.HParams 对象, torch>=2.6 的 weights_only=True 默认拒绝
       (官方预训练的 config 是纯 dict, 不受影响), 导致安装态 prewarm /
       换权全挂 ("tts failed")。慢路径按 GLOBAL 白名单扫描 pickle 流:
       白名单外一律拒绝 (os.system 等 RCE 载荷进不来), 白名单内对象
       stub 化 — 不 import 模块、不执行类体, storage 一律 stub; 顶层
       仍须为 dict。安全性不降, 训练产物放行。

    唯一强收口点在 _ensure_weights 换权分支内 (覆盖 v2 请求级热换 /
    v3 档案 / native / openai / set_*_weights 全部入口); v2 层另有
    请求入口早检, 仅为了流式请求能在响应头发出前给出干净的 400。

    批次86 (SEC-01): 进入反序列化前先过 _precheck_weights_path —
    UNC/设备路径拒绝、非普通文件拒绝、超过 ADR_TTS_WEIGHTS_MAX_MB 拒绝。
    批次87 (SEC-02): 同一指纹 (路径+mtime_ns+size) 只校验一次, 结果缓存。
    批次90 (SEC-01 残留): 快路径加 ``mmap=True`` — 仅校验时不把张量整体读进
    RAM (此前快路径本身就是一次整文件加载)。
    """
    weights_path = _precheck_weights_path(weights_path)
    stamp = weights_stamp(weights_path)
    if _weights_check_cached(stamp):
        return
    import torch
    try:
        try:
            obj = torch.load(weights_path, map_location="cpu",
                             weights_only=True, mmap=True)
        except TypeError:  # 老版本 torch 无 mmap= 参数
            obj = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            _restricted_weights_scan(weights_path)
        except Exception as slow_err:
            raise ValueError(
                f"weights file rejected by safe loader: {type(slow_err).__name__}: "
                f"{slow_err} ({weights_path})") from slow_err
        _weights_check_remember(stamp)
        return  # 慢路径通过: 含 HParams 的 GSV 训练产物
    if not isinstance(obj, dict):
        raise ValueError("weights file top-level must be a dict of tensors")
    _weights_check_remember(stamp)


def safe_torch_load(path, map_location="cpu",
                    allow_globals: tuple[tuple[str, str], ...] = ()):
    """受限反序列化 (批次87 / SEC-03): 优先 weights_only=True, 需要时按白名单回落。

    ADR 自己的训练产物把 ``TrainerConfig`` 等本仓 dataclass 存进了 checkpoint,
    ``weights_only=True`` 会拒; 直接 ``weights_only=False`` 又是 RCE 面。这里
    先受限扫描 pickle GLOBAL (白名单外一律拒绝), 再以 weights_only=False 真正
    加载 —— 能落地的只有白名单内的类型。

    allow_globals: 额外放行的 (module, name), 只应列本仓自有类。
    """
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        _restricted_weights_scan(str(path), allow_globals=allow_globals)
        return torch.load(path, map_location=map_location, weights_only=False)


def _is_trusted_weights_global(module: str, name: str,
                               extra: tuple[tuple[str, str], ...] = ()) -> bool:
    """慢路径 GLOBAL 白名单 (批次43, 探针实测真实权重文件 GLOBAL 并集)。

    - collections.OrderedDict: torch.save 顶层容器 (真类, 保顶层 dict 校验)
    - torch.*Storage: zip 权重 persistent_id 元组里的 storage 类型
    - torch._utils._rebuild_*: torch 张量重建函数
    - utils.HParams: GPT-SoVITS 训练产物 config 的第三方类型 (唯一放行的
      非 torch GLOBAL; stub 化, 不 import third_party)
    - extra: 调用方显式放行的 (module, name) — 仅本仓训练产物用
      (批次87 / SEC-03)
    """
    if (module, name) in extra:
        return True
    if module == "collections" and name == "OrderedDict":
        return True
    if module == "torch" and name.endswith("Storage"):
        return True
    if module == "torch._utils" and name in (
        "_rebuild_tensor_v2",
        "_rebuild_tensor",
        "_rebuild_parameter",
        "_rebuild_device_tensor_from_numpy",
    ):
        return True
    return module == "utils" and name == "HParams"


class _WeightsScanStub:
    """GLOBAL 白名单内对象的 no-op 替身: 不 import 模块、不执行任何代码,
    只让 pickle 流走完 (REDUCE / BUILD / SETITEMS / APPENDS 全部吞掉)。"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _WeightsScanStub()

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return None

    def __setstate__(self, state):
        pass

    def append(self, value):
        pass

    def __len__(self):
        return 0


class _WeightsScanUnpickler(pickle.Unpickler):
    """受限 Unpickler: 白名单外 GLOBAL 直接抛 UnpicklingError。"""

    def __init__(self, *args, allow_globals: tuple[tuple[str, str], ...] = ()):
        super().__init__(*args)
        self._allow = allow_globals

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict  # 标准库纯容器, 真类以保顶层 dict 校验
        if _is_trusted_weights_global(module, name, self._allow):
            return _WeightsScanStub
        raise pickle.UnpicklingError(
            f"forbidden global in weights pickle: {module}.{name}")

    def persistent_load(self, pid):
        # 只验结构不读 storage: zip 权重的张量数据不进入内存
        return _WeightsScanStub()


# 批次90 (R3 SEC-01 残留): 慢路径不再整文件读入。data.pkl 是唯一进内存的成员,
# 给它设解压后大小上限 — 压缩炸弹可以把 zf.read() 撑爆。
_WEIGHTS_PKL_MAX = 256 * 1024 * 1024
# 截头 zip 自愈必须整读文件补 "PK", 只对足够小的文件启用: 真实权重 GB 级,
# 大文件还缺 zip 头只能是损坏或构造的攻击样本。
_WEIGHTS_PATCH_MAX = 64 * 1024 * 1024


@contextlib.contextmanager
def _open_weights_zip(weights_path: str):
    """打开 torch zip 权重; 截头文件仅在足够小时自愈 (批次90)。"""
    with open(weights_path, "rb") as f:
        head = f.read(2)
    if head == b"PK":
        zf = zipfile.ZipFile(weights_path)
    else:
        # GSV 生态常见的截头 zip 自愈 (与 third_party process_ckpt.load_sovits_new
        # 同款补头); 真 legacy 非 zip 格式会在 BadZipFile 处被拒绝。
        size = os.path.getsize(weights_path)
        if size > _WEIGHTS_PATCH_MAX:
            raise pickle.UnpicklingError(
                "weights file is not a zip archive (header missing, "
                f"{size} bytes too large to patch)")
        with open(weights_path, "rb") as f:
            zf = zipfile.ZipFile(io.BytesIO(b"PK" + f.read()))
    try:
        yield zf
    finally:
        zf.close()


def _restricted_weights_scan(weights_path: str,
                             allow_globals: tuple[tuple[str, str], ...] = ()) -> None:
    """慢路径: 受限 Unpickler 扫描 torch zip 权重的 data.pkl (不加载张量)。

    批次90 (SEC-01 残留): 旧实现 ``f.read()`` 整文件 + BytesIO → 单请求峰值约
    2× 文件体积 (默认上限 2GB → 4GB+)。现在 zip 走磁盘流式中央目录, 只有
    data.pkl 一个成员进内存, 且解压后大小有硬上限。
    """
    try:
        with _open_weights_zip(weights_path) as zf:
            pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
            if len(pkl_names) != 1:
                raise pickle.UnpicklingError(
                    f"expected exactly one data.pkl in weights zip, got {len(pkl_names)}")
            info = zf.getinfo(pkl_names[0])
            if info.file_size > _WEIGHTS_PKL_MAX:
                raise pickle.UnpicklingError(
                    f"data.pkl too large: {info.file_size} bytes "
                    f"(limit {_WEIGHTS_PKL_MAX})")
            data = zf.read(info)
    except zipfile.BadZipFile as e:
        raise pickle.UnpicklingError(
            "not a torch zip weights file (legacy non-zip format not accepted)") from e
    obj = _WeightsScanUnpickler(
        io.BytesIO(data), allow_globals=allow_globals).load()
    if not isinstance(obj, dict):
        raise pickle.UnpicklingError(
            "weights file top-level must be a dict of tensors")


def weights_stamp(path: str) -> tuple[str, int, int]:
    """权重文件指纹 (批次77 / CODE_REVIEW #6): 路径 + mtime_ns + size。

    只用路径做幂等键时, 音色包导入 / 训练产物用 ``os.replace`` 覆盖同一路径
    后仍会被判为"已加载", 既跳过 ``validate_weights_file`` 的安全预检,
    也继续用旧权重合成。mtime_ns + size 能感知文件被换掉。文件不可 stat
    (不存在 / 无权限) 时返回零值 — 后续 validate/加载会给出真正的错误。
    """
    try:
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return (path, 0, 0)


def _resolve_fragment_interval(fragment_interval: Optional[float]) -> float:
    """Track E (批次33): 句末静音秒数解析 — None → env ADR_TTS_FRAGMENT_INTERVAL
    (默认 0.3 = GSV 原行为); 显式传入钳到 [0, 2] (负数会使下游 np.zeros 崩)。
    (批次36: env 读取委托 settings 单源)"""
    fi = fragment_interval
    if fi is None:
        fi = settings.fragment_interval_default()
    return max(0.0, min(float(fi), 2.0))


@dataclass
class GSVEngineConfig:
    """GPT-SoVITS 套壳引擎配置。"""
    gsv_dir: Path = GSV_DIR
    version: str = "v2"           # v1/v2 权重已下载; v3/v4 需另下
    device: str = "auto"
    half: bool = True             # 8GB 卡半精度
    bert_onnx: str = "auto"       # auto=仅 CPU 设备启用 / on / off


class _OnnxBertFeat:
    """批次5: ORT 版 BERT 特征提取 (纯 CPU 档 ~1.4x, 实测 bench_bert_onnx.py)。

    模仿 BertForMaskedLM 调用面: res = model(**inputs, output_hidden_states=True)
    GSV 消费面只有 res["hidden_states"][-3] (倒数第 3 层, TextPreprocessor
    get_bert_feature)。其余层不导出 — 一次前向只算一层。
    惰性: 首次调用建 ORT 会话; onnx 模型不存在时自动导出并缓存。
    """

    def __init__(self, bert_dir: str, cache_path: Path, fallback=None):
        self.bert_dir = str(bert_dir)
        self.cache_path = Path(cache_path)
        self._sess = None
        # 批次66: ORT 不可用时的 torch BERT 回退; 传入 GSV 已加载的原模型免重复占内存
        self._fallback = fallback
        self._torch = None
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._sess is not None or self._torch is not None:
            return
        with self._init_lock:   # 批次66: 惰性建会话串行化, 防并发重复导出写坏缓存
            if self._sess is not None or self._torch is not None:
                return
            from transformers import AutoModel
            created = not self.cache_path.exists()
            m = None
            try:
                if created:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    m = AutoModel.from_pretrained(self.bert_dir).eval()
                    self._export(m)
                from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                                         SessionOptions)
                opts = SessionOptions()
                opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
                self._sess = InferenceSession(str(self.cache_path), opts,
                                              providers=["CPUExecutionProvider"])
                self._fallback = None   # 批次66: ORT 就绪即释放 torch BERT 内存
            except Exception:
                # 批次66: 导出需 onnx/onnxscript (all/m1 extra 未声明)、ORT 缺失或
                # 缓存损坏 → 回退 torch BERT, 不让首次合成崩溃
                logging.getLogger("adr").warning(
                    "[gsv] ONNX BERT 不可用, 回退 torch 版", exc_info=True)
                if created:   # 半截导出文件会让后续每次都失败
                    self.cache_path.unlink(missing_ok=True)
                self._torch = (self._fallback if self._fallback is not None
                               else m if m is not None
                               else AutoModel.from_pretrained(self.bert_dir).eval())

    def _export(self, m) -> None:
        import torch

        class _Feat(torch.nn.Module):
            def __init__(self, mm):
                super().__init__()
                self.m = mm

            def forward(self, input_ids, attention_mask, token_type_ids):
                out = self.m(input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             output_hidden_states=True)
                return out.hidden_states[-3]

        dummy = torch.ones(1, 8, dtype=torch.long)
        torch.onnx.export(
            _Feat(m), (dummy, dummy, dummy), str(self.cache_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["feat"],
            dynamic_axes={n: {0: "b", 1: "s"} for n in
                          ["input_ids", "attention_mask", "token_type_ids",
                           "feat"]},
            opset_version=17, do_constant_folding=True)

    def __call__(self, *, input_ids=None, attention_mask=None,
                 token_type_ids=None, **_kw):
        import torch
        self._ensure_session()
        if self._sess is None:
            # 批次66: torch 回退, 输入已由 GSV 放到原模型设备上, 不搬运;
            # 输出面与 ORT 版一致 (只取倒数第 3 层)
            with torch.no_grad():
                out = self._torch(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  output_hidden_states=True)
            return {"hidden_states": [out.hidden_states[-3], None, None]}
        feed = {"input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
                "token_type_ids": token_type_ids.cpu().numpy()}
        feat = self._sess.run(["feat"], feed)[0]
        # 模仿 hidden_states 元组: GSV 取 [-3:-2] (倒数第 3 层), 3 元素列表
        # 的首位即 [-3]; 其余占位 (只有这一层被消费)
        return {"hidden_states": [torch.from_numpy(feat), None, None]}


class _RuntimeProbe:
    """v1.3.x Runtime 闭环探针: 持槽后计时, 退出时把 RTF / 首包 / 显存峰值
    交给 runtime.record。客户端断连 (GeneratorExit) 不算失败也不计样本;
    record 内部吞异常, 探针永不影响合成。"""

    def __init__(self, stream: bool):
        self.stream = stream
        self.samples = 0
        self.sr = 0
        self.ttfb_ms: Optional[int] = None

    def __enter__(self):
        self.start()
        return self

    def start(self) -> None:
        """(重新) 起表 + 清显存峰值; 模型就绪后再调一次, 排除懒加载耗时。"""
        import time
        try:   # sys.modules 查: 不为探针主动导入 torch
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self._t0 = time.perf_counter()

    def chunk(self, n: int, sr: int) -> None:
        import time
        if self.ttfb_ms is None:
            self.ttfb_ms = int((time.perf_counter() - self._t0) * 1000)
        self.samples += int(n)
        self.sr = int(sr) or self.sr

    def __exit__(self, exc_type, exc, tb):
        import time
        if exc_type is GeneratorExit:      # 客户端断连: 不计样本
            return False
        try:
            from adr.core import runtime
            mem_mb = None
            try:
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    mem_mb = int(torch.cuda.max_memory_allocated() / 1048576)
            except Exception:
                pass
            audio_s = self.samples / self.sr if self.sr else 0.0
            runtime.record(ok=exc_type is None and self.samples > 0,
                           elapsed_s=time.perf_counter() - self._t0,
                           audio_s=audio_s, ttfb_ms=self.ttfb_ms,
                           mem_mb=mem_mb, stream=self.stream)
        except Exception:
            pass
        return False


class GSVEngine:
    """GPT-SoVITS 推理引擎 (懒加载 + 锁 + cwd 上下文, 同 DiffSingerEngine 模式)。"""

    def __init__(self, config: Optional[GSVEngineConfig] = None):
        self.config = config or GSVEngineConfig()
        self._tts = None
        self._lock = threading.Lock()
        self._loaded = {"t2s": None, "vits": None}   # 值 = weights_stamp 指纹 (批次9幂等跳过)
        # 批次82 (#1): 流式 worker 超时未退出 → 置位, 下次 _lazy_init 重建模型
        self._stalled = False
        # 批次90 (R3-OBS-01): 降级起点 (monotonic)。只有模型真正重建成功才清零 —
        # 卡死 worker 持锁时 _stalled 会先被 _drop_stalled_model 清掉, 但重建会
        # 卡在 _lock/_CHDIR_LOCK 上, 单看 _stalled 会漏判"永久不可用"。
        self._degraded_since: Optional[float] = None
        # 批次63 第三子批 (m10 ④): N=1 硬预算调度 — 排队/预算/调度报告收口到
        # SynthesisScheduler; _lock 仍是引擎态硬闸 (旁路调度者也被串行化)。
        # 批次64 第一子批: N=2 固定份额调度层, 放宽准入到 2 槽; _lock 仍串行化,
        # 真并行需第二子批 t2s 连续批处理内核 (bs>1, per-seq RNG, 音质门)
        self._sched = SynthesisScheduler(settings.scheduler_max_concurrency())
        # 批次 67 (m10 N=2 第三子批): 跨请求合批 worker — 同签名请求合并
        # 段池走 t2s 批解码; 实际生效还需 ADR_T2S_COALESCE=1 (默认开) +
        # 行种子桥 ADR_T2S_BATCH_RNG=1 + 调度并发 >=2 (见 synthesize 门控),
        # 默认配置下三条件不同时满足 → 合成行为不变
        self._coalescer = _Coalescer(self)
        # v1.3.x Policy (conservative-v1): 只应用不影响音质的 overrides
        # (线程数 / 流式分块), 显式 env 优先; 构造时应用, 首个请求的切分即生效
        try:
            from adr.core import policy
            policy.apply()
        except Exception as e:
            print(f"[ADR] policy 应用失败, 走默认值: {e}")

    def queue_depth(self) -> int:
        """正在排队等待合成的请求数 (批次23 引入, 批次63 委托调度器)。"""
        return self._sched.waiting

    def synth_busy(self) -> int:
        """正在合成 (持预算槽) 的请求数 (批次23 引入, 批次63 委托调度器)。"""
        return self._sched.busy

    def sched_stats(self) -> dict:
        """调度报告 (批次63 第三子批): 预算/排队/等待时延统计快照。
        v1.3.x 起附 runtime 闭环报告 (窗口 RTF / 微调 / 回退事件)。"""
        st = self._sched.stats()
        try:
            from adr.core import runtime
            st["runtime"] = runtime.report()
        except Exception:
            pass
        return st

    @contextlib.contextmanager
    def _admit(self):
        """调度准入 (批次63 第三子批): N=1 硬预算排队 + 调度报告落审计。

        批次86 (CON-01): 排队也带上限 — 卡死的流式 worker 会一直占着槽位,
        无界排队会让后续请求全部永久挂起 (全站假死)。超时转 503 可重试。
        log_event 内部吞异常降级, 审计失败不影响合成。
        """
        admitted = False
        try:
            with self._sched.slot(timeout=_LOCK_WAIT_SEC) as tk:
                admitted = True
                log_event("schedule", wait_ms=tk.wait_ms,
                          waiting=tk.waiting, busy=tk.busy)
                yield
        except TimeoutError as e:
            if admitted:
                raise   # 合成体自己抛的 TimeoutError, 不当作准入超时
            log_event("engine_admit_timeout", wait_s=_LOCK_WAIT_SEC)
            raise EngineStalledError(
                f"合成引擎忙或已卡死 ({_LOCK_WAIT_SEC:.0f}s 未获得调度准入), "
                f"请稍后重试") from e

    @property
    def is_ready(self) -> bool:
        # 批次82 (#1): 卡死的流式 worker 可能还占着模型 → 对外报不可用,
        # 下次 _lazy_init 会重建
        return self._tts is not None and not self._stalled

    @contextlib.contextmanager
    def _gsv_context(self):
        """进入 GSV 代码上下文 (sys.path + cwd, 其代码用相对路径找权重)。"""
        code_dir = str(self.config.gsv_dir)
        inner = str(self.config.gsv_dir / "GPT_SoVITS")
        # Track B: 全程持锁覆盖 chdir 窗口, 进出后 cwd 恒等于原值
        with _CHDIR_LOCK:
            old_cwd = os.getcwd()
            for p in (code_dir, inner):
                if p not in sys.path:
                    sys.path.insert(0, p)
            os.chdir(code_dir)
            try:
                yield
            finally:
                os.chdir(old_cwd)

    @contextlib.contextmanager
    def _engine_lock(self):
        """批次86 (CON-01): 带超时获取引擎锁。

        卡死的流式 worker 可能正持着 ``_lock``; 旧实现的无超时 ``with
        self._lock`` 会让后续请求永久排队。这里超时即转 503 (engine_stalled),
        由调用方重试, 不再拖住整个进程。
        """
        if not self._lock.acquire(timeout=_LOCK_WAIT_SEC):
            logging.getLogger("adr").error(
                "[gsv] 等待引擎锁 %.0fs 超时: 可能有流式 worker 卡死",
                _LOCK_WAIT_SEC)
            log_event("engine_lock_timeout", wait_s=_LOCK_WAIT_SEC)
            raise EngineStalledError(
                f"合成引擎忙或已卡死 ({_LOCK_WAIT_SEC:.0f}s 未获得锁), 请稍后重试")
        try:
            yield
        finally:
            self._lock.release()

    def _mark_stalled(self, worker: Optional[threading.Thread],
                      where: str = "流式") -> None:
        """批次82 (#1): 流式 worker 超时未退出 — 标记引擎不可复用。

        这里不抢 ``_lock`` (卡死的 worker 可能正持着它), 只置降级标记;
        下一次 ``_lazy_init()`` 在持有 ``_lock`` 后丢弃模型并重建。

        批次90: 同时记录降级起点, 供进程级看门狗判定"永久不可恢复" —
        ``_drop_stalled_model`` 会提前清掉 ``_stalled``, 但重建会卡在
        ``_lock`` / ``_CHDIR_LOCK`` 上, 计时必须保留到 TTS 真正重建成功。

        批次94 (R3-CON-02): ``where`` 区分卡死路径 (流式 / 非流式 / 合批),
        非流式路径此前同步调用 run(), 卡死时永远不会走到这里。
        """
        name = worker.name if worker is not None else "?"
        self._stalled = True
        if self._degraded_since is None:
            self._degraded_since = time.monotonic()
        logging.getLogger("adr").error(
            "[gsv] %s合成 worker %s 超时未退出, 引擎降级: 下次请求重建模型",
            where, name)
        log_event("stream_stall", worker=name, where=where)

    def _drop_stalled_model(self) -> bool:
        """批次82 (#1): 上次流式卡死 → 丢弃模型与权重指纹, 本次强制重建。

        调用方须持有 ``_lock``。
        """
        if not self._stalled:
            return False
        self._stalled = False
        self._tts = None
        self._loaded = {"t2s": None, "vits": None}
        return True

    def _lazy_init(self):
        if self._drop_stalled_model():
            logging.getLogger("adr").warning(
                "[gsv] 上次流式合成 worker 超时未退出, 强制重建模型")
        if self._tts is not None:
            return
        # 批次10: t2s CUDA Graph 桥 — AR 提速 3.77x 但官方 SDPA 实现下 AR 分布
        # 漂移 (同输入 45→254 token, 音频时长翻倍), 默认关闭; 需上游修复后
        # 可设 ADR_T2S_CUDAGRAPH=1 启用 (诊断见 docs/m9-phase3-plan.md 批次10)
        os.environ.setdefault("ADR_T2S_CUDAGRAPH", "0")
        # 批次63: t2s 静态 KV Cache 桥 — decode 每步 cat 重建 KV (O(L^2) 拷贝)
        # 换预分配 buffer + 原地写, 与原路径逐 token 位等; 默认关闭
        os.environ.setdefault("ADR_T2S_STATIC_KV", "0")
        # 批次63: 分桶 CUDA Graph — 仅在 ADR_T2S_CUDAGRAPH=1 时生效; 非空非 "0"
        # 时桥改用分桶 runner ("1"/"default" = 默认 9 档, 或逗号分隔档位)
        os.environ.setdefault("ADR_T2S_CUDAGRAPH_BUCKETS", "")
        # 兼容补丁: NLTK (英文 G2P 依赖) 离线环境下载默认超时 72s/次,
        # 收紧到 5s 快速失败走内置回退发音
        import socket
        socket.setdefaulttimeout(5)
        # 批次74: 英文 G2P 离线依赖体检 — 缺 cmudict/标注器时英文文本会在
        # g2p_en 抛 LookupError (无外网机器无法自动补下载), 这里提前落日志 +
        # 审计事件, 便于排障; 不影响启动, 中/日/粤合成照常。
        _nltk_missing = nltk_missing_reason()
        if _nltk_missing:
            logging.getLogger("adr").warning(
                "[nltk] 英文合成依赖缺失: %s — 英文文本将合成失败 "
                "(中文 / 日文 / 粤语不受影响)", _nltk_missing)
            log_event("nltk", ready=False, missing=_nltk_missing)
        else:
            log_event("nltk", ready=True)
        with self._gsv_context():
            import torch
            # Track E (批次33): AR 解码 tqdm 每句 1500 迭代渲染 stderr,
            # TQDM_DISABLE 对 tqdm 4.67 实测无效 — monkeypatch __init__ 注入
            # disable=True。第三方 `from tqdm import tqdm` 绑定同一类对象,
            # 补类属性即全局生效; 必须在 TTS 导入前打。ADR_TTS_KEEP_TQDM=1 保留。
            if not settings.keep_tqdm():
                try:
                    import tqdm as _tqdm

                    _tqdm_init = _tqdm.tqdm.__init__

                    def _silent_tqdm_init(_tq, *a, **k):
                        k.setdefault("disable", True)
                        _tqdm_init(_tq, *a, **k)

                    _tqdm.tqdm.__init__ = _silent_tqdm_init
                except Exception:
                    pass
            from TTS_infer_pack.TTS import TTS, TTS_Config

            # 批次49: 低配机器线程上限 (核少机器跑满会抢系统资源), 越早设越好
            _th = settings.torch_threads()
            if _th > 0:
                torch.set_num_threads(_th)
            device = self.config.device
            _forced = device != "auto"   # config 显式指定 = 用户主动选择 (批次57 review)
            if device == "auto":
                # 批次49: ADR_TTS_DEVICE 决定 auto 的走向 (壳设置中心/低配机器
                # 强制 CPU); config 显式传参不受影响
                env_dev = settings.tts_device()
                if env_dev != "auto":
                    device = env_dev
                    _forced = True       # env 强制 CPU 同理
                else:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
            # 批次122: 强制 cuda 但 CUDA 不可用 (CPU 包 / 驱动缺失) 必须回落 ——
            # 否则 GSV 用 map_location=cfg.device 反序列化权重时直接抛
            # "Attempting to deserialize object on a CUDA device"。
            device = resolve_runtime_device(device)
            # 批次57: GPU 档案落日志 (排障关键: N 卡型号/算力/显存/驱动代际一目了然)
            if device == "cuda":
                # 批次102 首启向导 ⑤⑥: 单进程显存上限 (占总显存比例)。
                # 超限时 torch 直接抛 OOM (走既有 cuda_oom 善后), 而不是把
                # 整机拖进交换。未设置 = 不限制。
                _frac = settings.gpu_mem_fraction()
                if _frac is not None:
                    try:
                        torch.cuda.set_per_process_memory_fraction(_frac, 0)
                        logging.getLogger("adr").info(
                            "[gpu] 显存上限 %.0f%% (ADR_GPU_MEM_FRACTION)", _frac * 100)
                    except Exception:
                        logging.getLogger("adr").warning(
                            "[gpu] 显存上限设置失败 (ADR_GPU_MEM_FRACTION)", exc_info=True)
                try:
                    _cap = torch.cuda.get_device_capability(0)
                    _props = torch.cuda.get_device_properties(0)
                    _cc = f"{_cap[0]}.{_cap[1]}"
                    _drv = _nvidia_driver_version()
                    logging.getLogger("adr").info(
                        "[gpu] %s, 算力 %s, 显存 %.1fGB, CUDA runtime %s, 驱动 %s",
                        _props.name, _cc, _props.total_memory / 1024 ** 3,
                        torch.version.cuda, _drv or "未知")
                    if _cap < (7, 0):
                        logging.getLogger("adr").warning(
                            "[gpu] 算力 %s 低于 7.0, 本 CUDA 轮子可能不含该架构内核"
                            " (报 no kernel image available 即此因), 建议用 CPU 版", _cc)
                except Exception:
                    logging.getLogger("adr").debug("[gpu] 诊断信息获取失败", exc_info=True)
            elif torch.version.cuda:
                _drv = _nvidia_driver_version()
                if _drv and _forced:
                    # 批次57 review: 强制 CPU 是主动选择, 驱动正常时落 info 而非
                    # 误导性"请升级驱动"警告 (该警告只针对 auto 判定回退场景)
                    logging.getLogger("adr").info(
                        "[gpu] 按配置强制 CPU 运行 (N 卡驱动 %s 正常)", _drv)
                elif _drv:
                    # 经典故障: 装了 CUDA 轮子但驱动过旧 → is_available()=False 静默回退 CPU
                    logging.getLogger("adr").warning(
                        "[gpu] N 卡驱动 %s 已装但 CUDA 不可用 — 本轮子 (CUDA %s) 通常"
                        "要求更高驱动, 请升级 NVIDIA 驱动或改用 CPU 版",
                        _drv, torch.version.cuda)
                else:
                    logging.getLogger("adr").info(
                        "[gpu] 未走 CUDA (不可用或强制 CPU); 未检出 N 卡驱动")
            else:
                logging.getLogger("adr").info("[gpu] CPU 版 torch (无 CUDA 轮子), 以 CPU 运行")
            # 批次4 yaml 漂移修复: GSV 的 init_vits_weights 会把热换权重写回
            # tts_infer.yaml 的 custom 段 (实测两次进程启动基线 0.812→0.759 漂移,
            # 任何引擎重启都会静默换权重)。对策: 只取 yaml 的 v2 原始段 (预训练
            # 权重) 构造 dict 传入 — 每次进程启动都从干净状态出发; 并把写回
            # 路径重定向到私有 scratch 文件, 共享 yaml 永不被改。
            import yaml as _yaml
            _cfgp = self.config.gsv_dir / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
            _raw = _yaml.safe_load(_cfgp.read_text(encoding="utf-8"))
            cfg = TTS_Config({"v2": _raw["v2"]})
            cfg.device = device
            cfg.is_half = self.config.half and device == "cuda"
            cfg.version = self.config.version
            self._tts = TTS(cfg)
            # 批次90: 模型重建成功 = 降级结束 (看门狗计时清零)
            self._degraded_since = None
            # 热换权重触发的 save_configs 写到私有文件, 不再污染共享配置
            _scratch = self.config.gsv_dir / "TEMP" / "tts_infer_adr_scratch.yaml"
            _scratch.parent.mkdir(exist_ok=True)
            self._tts.configs.configs_path = str(_scratch)
            # Track E (批次33): run() finally 每句 empty_cache (gc.collect +
            # cuda.empty_cache) — 句间停顿来源之一, 且清缓存后下一步分配要重新
            # cudaMalloc。改为每 16 次调用真清理一次 (防显存碎片兜底保留)。
            # ADR_TTS_KEEP_EMPTY_CACHE=1 回退每句清理。
            if not settings.keep_empty_cache():
                _orig_empty_cache = self._tts.empty_cache
                _ec_calls = {"n": 0}

                def _throttled_empty_cache():
                    # 实时读 env (经 settings): A/B 基准可在同进程切腿
                    # (KEEP_EMPTY_CACHE=1 即旧每句清)
                    if settings.keep_empty_cache():
                        _orig_empty_cache()
                        return
                    _ec_calls["n"] += 1
                    if _ec_calls["n"] % 16 == 0:
                        _orig_empty_cache()

                self._tts.empty_cache = _throttled_empty_cache

            # 批次5: 纯 CPU 档用 ORT 版 BERT (组件级 ~1.4x, 端到端 ~10%)
            if ((device == "cpu" and self.config.bert_onnx == "auto")
                    or self.config.bert_onnx == "on"):
                try:
                    self._tts.bert_model = _OnnxBertFeat(
                        self._tts.bert_model.config._name_or_path,
                        self.config.gsv_dir / "TEMP" / "bert_feat_cpu.onnx",
                        fallback=self._tts.bert_model)
                except Exception:  # 批次66: 导出/ORT 失败在首调时回退, 此处只兜构造
                    pass

            # 兼容补丁: 新版 torchaudio.load 依赖 torchcodec (无 Windows 轮子),
            # 用 soundfile shim 替换 (返回 (C,T) float tensor, 与原版一致)
            import soundfile as _sf
            import torch as _torch
            import torchaudio as _ta

            def _sf_load(path, *a, **k):
                data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
                return _torch.from_numpy(data.T), sr

            _ta.load = _sf_load

    def warmup(self, vits_weights: Optional[str] = None,
               t2s_weights: Optional[str] = None):
        """预热: 后台线程调用, 把权重加载从首次合成挪到启动期 (A1)。

        传权重时额外热换到目标权重 — 否则首次流式调用要付 ~12s 的
        init_vits_weights 罚金 (实测 bench_stream_variance)。
        """
        with self._engine_lock():
            self._lazy_init()
            self._ensure_weights(vits_weights, t2s_weights)

    def _ensure_weights(self, vits_weights: Optional[str],
                        t2s_weights: Optional[str]):
        """热换到目标权重, 同指纹幂等跳过 (批次 9; 批次77 改指纹)。

        上游 init_vits_weights/init_t2s_weights 无条件完整重载 — 稳态下
        连续合成同一音色每次白付 1~3s 权重重载。幂等键含 (路径, mtime, size):
        同路径文件被覆盖时能感知变化, 不会跳过安全预检与重载。
        调用方须持有 self._lock。
        """
        if not (vits_weights or t2s_weights):
            return
        # 批次99 (R4-API-01): 与预检同一基准 (GSV_DIR), 不再按进程 cwd 解析
        vw = resolve_weights_path(vits_weights) if vits_weights else None
        tw = resolve_weights_path(t2s_weights) if t2s_weights else None
        changed = False
        with self._gsv_context():
            if vw:
                vstamp = weights_stamp(vw)
                if self._loaded.get("vits") != vstamp:
                    # 批次38: 安全预检收编到唯一换权点 — 恶意 pickle 在 third_party
                    # torch.load 前拒绝, 覆盖 v2 请求级 / v3 档案 / set_* 全部入口
                    validate_weights_file(vw)
                    self._tts.init_vits_weights(vw)
                    self._loaded["vits"] = vstamp
                    changed = True
            if tw:
                tstamp = weights_stamp(tw)
                if self._loaded.get("t2s") != tstamp:
                    validate_weights_file(tw)
                    self._tts.init_t2s_weights(tw)
                    self._loaded["t2s"] = tstamp
                    changed = True
        if changed:
            # 批次88 (CACHE-01): 换权后段缓存里的 PCM 已属于上一套权重;
            # 旧实现只清 tts_cache, 段缓存默认 128 条/256MB 会继续回放旧人声。
            _seg_cache_clear()

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """前端符号清洗: '-'/'—'/'~' 在非英文单词内 -> ','(GSV 会把 '-' 读成"减");
        换行 -> ','。英文词内连字符 (GPT-SoVITS) 保留。"""
        import re
        # 任一侧是中文字符即视为分隔符; 两侧皆英文/数字 (GPT-SoVITS) 保留
        text = re.sub(r"(?<=[一-鿿])[-—–~]|[-—–~](?=[一-鿿])", "，", text)
        text = text.replace("\n", "，").replace("\r", "")
        return re.sub(r"，{2,}", "，", text).strip("，")

    @staticmethod
    def _stream_head_split(text: str, max_head: int = 14) -> "tuple":
        """W1 首包优化: 从长文本切出短首段 (句末标点优先, 逗号次之)。

        实测 (output/bench_stream.json): 首块延迟 ∝ 首段长度, 45 字长句
        cut1 首包 5.3s / cut0 25.3s。批次 9 再收紧 30→14 字: 首段 AR 生成
        token 数减半 (首包 4.4s→~2.5s), 逗号层 min 6 字早生效 — 2s 级短句
        (16~20 字) 也能在逗号处切出短首段。返回 (首段, 剩余); 无需切分时
        剩余为空串。
        """
        if len(text) <= max_head:
            return text, ""
        win = text[:max_head]
        cut = max((i for i, ch in enumerate(win) if ch in "。！？!?；;"), default=-1)
        if cut >= 6:
            return text[: cut + 1], text[cut + 1:]
        cut = next((i for i, ch in enumerate(win) if ch in "，,、：" and i >= 6), -1)
        if cut >= 6:
            return text[: cut + 1], text[cut + 1:]
        return text, ""

    @staticmethod
    def _split_long(text: str, max_len: int = 40) -> list:
        """批次20 长句优化: 无句读超长段细分, 救 GSV cut 切不开的最坏情况。

        GSV 内部 split (cut1/3/5) 依赖标点 — 一整段无句号的"单长句"会整段
        一次 AR decode: 耗时线性膨胀且发散 (电音/加字) 概率随长度陡增。
        有句读的段交给 GSV 自切 (保留其凑句/分段优化); 仅对无句读且超
        max_len 的段按停顿标点细分, 完全无标点按 max_len+15 硬切兜底。
        """
        if len(text) <= max_len or any(c in text for c in "。！？!?；;"):
            return [text]
        segs, buf = [], ""
        for ch in text:
            buf += ch
            if len(buf) >= max_len and ch in "，,、：:—…":
                segs.append(buf)
                buf = ""
            elif len(buf) >= max_len + 15:  # 无标点硬切兜底
                segs.append(buf)
                buf = ""
        if buf:
            segs.append(buf)
        return segs

    def synthesize_stream(
        self,
        text: str,
        ref_audio: str,
        prompt_text: str = "",
        text_lang: str = "zh",
        prompt_lang: str = "zh",
        t2s_weights: Optional[str] = None,
        vits_weights: Optional[str] = None,
        split_method: str = "cut3",
        head_seed: int = -1,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        speed_factor: float = 1.0,
        fragment_interval: Optional[float] = None,
    ):
        """流式合成: 逐块 yield (wav_chunk float32 [-1,1], sr)。

        官方 streaming_mode: 语义 token 分段解码, 首块延迟远小于整段。
        首包优化: 先合成预切的 ≤30 字短首段, 再合成剩余 (W1, 实测首包
        25.3s→2.7s 级)。split_method: cut3 按句号切(默认, 首包最优)
              / cut1 凑四句一切 / cut0 不切 / cut5 按标点切
        head_seed: ≥0 时固定首段采样种子 — 首包延迟从 2.7~4.9s 方差
              收敛到确定值 (后续段仍随机, 不伤韵律多样性)
        top_k / top_p / temperature: GPT 采样参数。默认 (15, 1.0, 1.0)
              为上游流式历史行为; 偏发散, 档案可配保守值压电音 (批次20)
        speed_factor: 语速 (批次21)。仅 ≠1.0 时注入 GSV inputs —
              GSV 流式路径对语速支持不稳, ==1.0 保持历史行为字节级一致
        fragment_interval: 句末静音秒数 (批次33)。None → env
              ADR_TTS_FRAGMENT_INTERVAL (默认 0.3 = GSV 原行为); 置 0 可
              消除句间停顿 (Track E)
        """
        import numpy as np
        import queue as _queue

        text = self._sanitize_text(text)
        ref_audio = str(Path(ref_audio).resolve())
        # 批次99 (R4-API-01): 权重按 GSV_DIR 解析, 与 _ensure_weights 指纹一致
        t2s_weights = resolve_weights_path(t2s_weights) if t2s_weights else None
        vits_weights = resolve_weights_path(vits_weights) if vits_weights else None
        head, rest = self._stream_head_split(text, settings.stream_head_chars())
        segments = ([head, *self._split_long(rest, settings.split_max_chars())]
                    if rest else [text])

        # 批次65 死锁修复: StreamingResponse 在 anyio 线程池逐 next() 迭代本
        # 生成器, 每次 next 可能换线程。旧实现持 _CHDIR_LOCK(RLock)跨 yield,
        # 恢复线程 != 加锁线程 → release 抛 "cannot release un-acquired lock",
        # _CHDIR_LOCK 永久卡死 (批次64 放宽到 N=2 并发后必现)。改法: 持锁的
        # 合成体全程跑在单一 worker 线程 (chdir 窗口仍串行化, RLock 同线程
        # acquire/release), 音频块入队; 本生成器只在锁外取队列并 yield。
        # _admit/_RuntimeProbe/seg 缓存语义不变, 只是归属 worker 线程。
        # 批次82 (#1): 无界队列 — 旧 maxsize=8 时消费端停止迭代 (半开连接/
        # 浏览器取消后 generator 迟迟不被回收) 会让 worker 卡在 q.put 里,
        # 而它正持着引擎 _lock, 后续所有合成被永久阻塞。无界后 put 永不阻塞,
        # worker 总能跑完并释放锁; 代价是消费端慢时音频在内存里堆积
        # (单请求级别, 可接受), cancel 检查仍会让断连请求尽早停。
        q: "_queue.Queue" = _queue.Queue()
        cancel = threading.Event()
        _DONE = object()

        def _produce():
            try:
                with self._admit(), _RuntimeProbe(stream=True) as probe:
                    with self._engine_lock():
                        self._lazy_init()
                        probe.start()   # 计时从模型就绪起: 懒加载不计入 RTF
                        inputs = {
                            "text": head,
                            "text_lang": text_lang,
                            "ref_audio_path": ref_audio,
                            "prompt_text": prompt_text,
                            "prompt_lang": prompt_lang,
                            "top_k": top_k,
                            "top_p": top_p,
                            "temperature": temperature,
                            "text_split_method": split_method,
                            "streaming_mode": True,
                            "parallel_infer": True,
                        }
                        if speed_factor != 1.0:
                            inputs["speed_factor"] = speed_factor
                        fi = _resolve_fragment_interval(fragment_interval)
                        inputs["fragment_interval"] = fi
                        with self._gsv_context():
                            self._ensure_weights(vits_weights, t2s_weights)
                            # 批次85 (#18): 参考音频版本指纹 — 同名档案重建
                            # ref.wav 后 mtime/size 变化, 不再回放旧缓存人声
                            ref_stamp = weights_stamp(ref_audio)
                            # 批次88 (CACHE-01): 权重同样带指纹入 key
                            t2s_stamp = (weights_stamp(t2s_weights)
                                         if t2s_weights else ())
                            vits_stamp = (weights_stamp(vits_weights)
                                          if vits_weights else ())
                            for i, seg in enumerate(segments):
                                if cancel.is_set():   # 断连: 与旧语义一致, 探针不计样本
                                    raise GeneratorExit
                                inputs["text"] = seg
                                seg_seed = (head_seed
                                            if (i == 0 and head_seed >= 0)
                                            else -1)
                                inputs["seed"] = seg_seed
                                ckey = None
                                if _seg_cache_enabled():
                                    ckey = _seg_cache_key(
                                        seg, ref_audio, prompt_text, text_lang,
                                        prompt_lang, split_method, top_k, top_p,
                                        temperature, speed_factor, fi,
                                        t2s_weights, vits_weights,
                                        ref_stamp, seg_seed,
                                        t2s_stamp=t2s_stamp,
                                        vits_stamp=vits_stamp)
                                    hit = _seg_cache_get(ckey)
                                    if hit is not None:
                                        pcm, hit_sr = hit
                                        buf = np.frombuffer(pcm, dtype=np.int16)
                                        probe.chunk(len(buf), int(hit_sr))
                                        q.put((buf.astype(np.float32) / 32767.0,
                                               int(hit_sr)))
                                        continue
                                seg_chunks: list = []
                                seg_sr = 32000
                                for sr, chunk in self._tts.run(inputs):
                                    if cancel.is_set():
                                        raise GeneratorExit
                                    chunk = np.asarray(chunk, dtype=np.float32)
                                    if np.abs(chunk).max() > 1.5:
                                        chunk = chunk / 32768.0
                                    seg_chunks.append(chunk)
                                    seg_sr = int(sr)
                                    probe.chunk(len(chunk), seg_sr)
                                    q.put((chunk, int(sr)))
                                if ckey is not None and seg_chunks:
                                    full = (seg_chunks[0] if len(seg_chunks) == 1
                                            else np.concatenate(seg_chunks))
                                    _seg_cache_put(
                                        ckey,
                                        (np.clip(full, -1.0, 1.0) * 32767)
                                        .astype(np.int16).tobytes(),
                                        seg_sr)
            except GeneratorExit:
                pass   # 消费端已断连, 无人读错误
            except BaseException as e:   # noqa: BLE001 — 转发到消费线程重抛
                q.put(("__error__", e))
            finally:
                q.put(_DONE)

        worker = threading.Thread(target=_produce, name="adr-gsv-stream",
                                  daemon=True)
        worker.start()
        done_seen = False
        idle_stalled = False
        try:
            while True:
                try:
                    item = q.get(timeout=_STREAM_IDLE_SEC)
                except _queue.Empty:
                    # 批次86 (CON-01): 上游 run() 卡死时旧实现的 q.get() 无超时,
                    # 消费端永远停在循环里, finally 里的 _mark_stalled 根本执行
                    # 不到 → 引擎锁被永久占住, 后续请求全部排队 (全站假死)。
                    # 静默超时即判定卡死: 主动取消 + 降级, 并把 503 交给调用方。
                    idle_stalled = True
                    raise EngineStalledError(
                        f"合成上游 {_STREAM_IDLE_SEC:.0f}s 无输出, 已判定卡死")
                if item is _DONE:
                    done_seen = True
                    break
                if isinstance(item[0], str) and item[0] == "__error__":
                    raise item[1]
                yield item[0], item[1]
        finally:
            # 批次82 (#1): 清理带 deadline。旧实现 q.get() 无超时 + join(30)
            # 超时后直接返回: worker 若卡在上游 run() 里就永久持锁, 而且调用
            # 方永远拿不回控制权。现在排空到 deadline 为止 (无界队列下正常
            # 路径不会阻塞), join 有界; 仍未退出则判定引擎卡死并降级。
            cancel.set()
            # 批次86 (CON-01): 已因静默超时判定卡死时跳过 30s 排空 — worker
            # 不会再产出, 直接 join + 降级, 让调用方尽快拿到 503。
            deadline = time.monotonic() + (0.0 if idle_stalled
                                           else _STREAM_DRAIN_SEC)
            while not done_seen:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    # 短超时轮询: 不能用 remaining 一把等死, 要留出
                    # deadline 到点后走 join + 降级判定的机会
                    if q.get(timeout=min(remaining, 0.5)) is _DONE:
                        done_seen = True
                except _queue.Empty:
                    continue
            worker.join(timeout=_STREAM_JOIN_SEC)
            # worker 已退出 = 没有卡死 (deadline 最后一刻完成的也算);
            # 仍存活说明卡在上游 run() 或持锁逻辑里, 不能再复用它。
            if worker.is_alive():
                self._mark_stalled(worker)

    def _run_nonstream(self, inputs: dict) -> tuple:
        """非流式 ``_tts.run`` 的有界等待 (批次94, R3-CON-02)。

        上游卡死时同步调用会让请求线程永久停在 ``run()`` 里: 既不置降级标记,
        看门狗与 ``/readyz`` 也无从判定 → 进程只能等外部超时。这里把调用挪到
        daemon 线程, 调用线程按 ``_NONSTREAM_IDLE_SEC`` 有界等待; 超时即
        ``_mark_stalled`` 并转 ``EngineStalledError`` (503 可重试)。
        """
        import queue as _queue

        q: "_queue.Queue" = _queue.Queue()
        box: dict = {}

        def _produce():
            try:
                sr, audio = None, None
                for sr, audio in self._tts.run(inputs):
                    pass  # 非流式只 yield 一次完整音频
                box["value"] = (sr, audio)
            except BaseException as e:   # noqa: BLE001 — 转发到调用线程重抛
                box["error"] = e
            finally:
                q.put(None)   # 哨兵: 生产线程已结束

        worker = threading.Thread(target=_produce, name="adr-gsv-nonstream",
                                  daemon=True)
        worker.start()
        try:
            q.get(timeout=_NONSTREAM_IDLE_SEC)
        except _queue.Empty:
            self._mark_stalled(worker, where="非流式")
            raise EngineStalledError(
                f"合成上游 {_NONSTREAM_IDLE_SEC:.0f}s 未返回, 已判定卡死")
        worker.join(timeout=_STREAM_JOIN_SEC)
        if worker.is_alive():
            self._mark_stalled(worker, where="非流式")
            raise EngineStalledError("合成 worker 未正常退出, 已判定卡死")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def synthesize(
        self,
        text: str,
        ref_audio: str,
        prompt_text: str = "",
        text_lang: str = "zh",
        prompt_lang: str = "zh",
        speed_factor: float = 1.0,
        seed: int = -1,
        t2s_weights: Optional[str] = None,   # C2: 微调后的 s1 ckpt
        vits_weights: Optional[str] = None,  # C2: 微调后的 s2 pth
        split_method: str = "cut1",          # 切句方式, 见 synthesize_stream
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        fragment_interval: Optional[float] = None,
    ) -> "tuple":
        """零样本克隆朗读: (text, 参考音频) → (wav, sr)。

        Args:
            text: 要朗读的文本
            ref_audio: 参考音频路径 (3-10 秒干净人声最佳)
            prompt_text: 参考音频的文本 (可空, 空则弱化文本条件)
            text_lang / prompt_lang: zh/en/ja/ko/yue 等
            speed_factor: 语速 (1.0 原速)
            seed: -1 随机
            top_k / top_p / temperature: GPT 采样参数 (短句可收 top_k 降跑偏)
        """
        import numpy as np

        text = self._sanitize_text(text)
        # 绝对路径: _gsv_context 会 chdir, 相对路径会丢
        ref_audio = str(Path(ref_audio).resolve())
        # 批次99 (R4-API-01): 权重按 GSV_DIR 解析, 与 _ensure_weights 指纹一致
        t2s_weights = resolve_weights_path(t2s_weights) if t2s_weights else None
        vits_weights = resolve_weights_path(vits_weights) if vits_weights else None
        # 批次63 第三子批: 调度准入 (N=1 硬预算), 计数/报告由调度器持有
        # 批次 67: 跨请求合批门控 — 总开关 (默认开) × 行种子桥 (行隔离
        # 内核, 默认关) × 调度并发 >=2 (单并发合批无意义) — 默认配置走 solo
        coalesce_on = (settings.t2s_coalesce_enabled()
                       and settings.t2s_batch_rng()
                       and settings.scheduler_max_concurrency() >= 2)

        def _build_inputs(with_weights: bool) -> dict:
            d = {
                "text": text,
                "text_lang": text_lang,
                "ref_audio_path": str(ref_audio),
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "speed_factor": speed_factor,
                "seed": seed,
                "text_split_method": split_method,
                "return_fragment": False,
                "streaming_mode": False,
                "parallel_infer": True,
                "fragment_interval": _resolve_fragment_interval(fragment_interval),
            }
            if with_weights:
                # 合批 worker 按 (t2s_weights, vits_weights) 热换/分组
                d["t2s_weights"] = t2s_weights
                d["vits_weights"] = vits_weights
            return d

        with self._admit(), _RuntimeProbe(stream=False) as probe:
            if coalesce_on:
                # 请求线程不懒加载 (锁语义与 solo 相反): 懒加载由 worker
                # 轮内持 _lock 完成 — 冷启动 burst 若 12 个请求线程各自
                # 抢锁, 会与 prewarm 争锁串行成数十秒停顿 (实测 50s);
                # 直接排队, 由 worker 单点等待, 请求侧零阻塞。
                probe.start()
                inputs = _build_inputs(with_weights=True)
                # 批次 65 第二子批: 段合批大小 (ADR_TTS_BATCH_SIZE, >1 时
                # to_batch 把多段合成 bs>1 批; 默认 1 不注入 = stock 字节级一致)
                bs = settings.tts_batch_size()
                if bs > 1:
                    inputs["batch_size"] = bs
                # 批次90 (R3-CON-01): 有界等待 — 合批 worker 卡在上游时 future
                # 永不完成, 旧实现无超时 → 请求线程 + 准入槽永久挂起。
                fut = self._coalescer.submit(inputs)
                try:
                    sr, audio = fut.result(timeout=_COALESCE_WAIT_SEC)
                except FuturesTimeout as e:
                    # 批次94 (R3-CON-02): future 永不完成 = worker 多半卡在上游
                    # run() 里; 旧实现只抛 503 不置降级 → 看门狗与 /readyz 全盲
                    self._mark_stalled(None, where="合批")
                    log_event("coalesce_wait_timeout",
                              wait_s=_COALESCE_WAIT_SEC)
                    raise EngineStalledError(
                        f"合批合成等待超时 ({_COALESCE_WAIT_SEC:.0f}s), "
                        f"引擎可能卡住, 请稍后重试") from e
            else:
                with self._engine_lock():
                    self._lazy_init()
                    probe.start()
                    inputs = _build_inputs(with_weights=False)
                    # 批次 65 第二子批: 段合批大小 (ADR_TTS_BATCH_SIZE, >1 时
                    # to_batch 把多段合成 bs>1 批; 默认 1 不注入 = stock 字节级一致)
                    bs = settings.tts_batch_size()
                    if bs > 1:
                        inputs["batch_size"] = bs
                    self._ensure_weights(vits_weights, t2s_weights)
                    with self._gsv_context():
                        # 批次94 (R3-CON-02): 去程有界 — 上游 run() 卡死时不再
                        # 永久占住请求线程, 超时置降级交给看门狗收口
                        sr, audio = self._run_nonstream(inputs)
            if audio is not None:
                probe.chunk(len(audio), int(sr))
        if audio is None:
            raise RuntimeError("GPT-SoVITS 未返回音频")
        audio = np.asarray(audio, dtype=np.float32)
        # 官方返回 int16 量程 (rms ~1600), 归一化到 [-1,1]
        if np.abs(audio).max() > 1.5:
            audio = audio / 32768.0
        return audio, int(sr)


_ENGINE: Optional[GSVEngine] = None
_ENGINE_LOCK = threading.Lock()  # 批次41a: 单例创建锁 (多线程首次并发防多实例)
_LOADING = False   # 后台预热进行中 (服务启动期, 供控制台显示引擎三态)
_STAGE = ""        # 预热细分阶段: queued/importing/loading/kernel/failed (供前端实时显示)
# 批次73: 预热失败原因摘要 (traceback 末几行, 最多 1200 字)。原实现 failed 只
# log.warning, 前端与壳只见"失败"二字, 用户无从判断是缺权重 / 显存不足 / 依赖坏。
_STAGE_ERROR = ""


def is_loading() -> bool:
    """引擎后台预热是否进行中 (进程级, 不触发加载)。"""
    return _LOADING


def is_ready() -> bool:
    """引擎是否已就绪可合成 (_tts 权重已加载, 不触发加载)。

    批次88 (BUG-05): 与实例属性 `is_ready` 对齐, 降级标记置位后不再报 ready,
    否则 CON-01 卡死后 /health 与桌面壳探活仍持续放流量进来。
    """
    return (_ENGINE is not None and _ENGINE._tts is not None
            and not _ENGINE._stalled)


def is_stalled() -> bool:
    """引擎是否处于卡死降级态 (批次90, 供 /readyz 与看门狗; 不触发加载)。"""
    return _ENGINE is not None and bool(_ENGINE._stalled)


def degraded_seconds() -> float:
    """引擎连续降级时长 (秒), 0 = 未降级 (批次90, 供 /readyz 与看门狗)。

    只认 ``_degraded_since``: 卡死 worker 持锁时 ``_stalled`` 会被
    ``_drop_stalled_model`` 提前清掉, 但重建会卡在 ``_lock`` / ``_CHDIR_LOCK``
    上, 引擎实际仍不可用 — 计时起点必须保留到 TTS 真正重建成功。
    """
    eng = _ENGINE
    if eng is None or eng._degraded_since is None:
        return 0.0
    return max(0.0, time.monotonic() - eng._degraded_since)


def watchdog_grace_sec() -> float:
    """看门狗降级容忍秒数 (ADR_ENGINE_WATCHDOG_SEC, 默认 300; 0 = 关闭)。"""
    raw = os.environ.get("ADR_ENGINE_WATCHDOG_SEC", "")
    try:
        return max(0.0, float(raw) if raw.strip() else _WATCHDOG_DEFAULT_SEC)
    except ValueError:
        return _WATCHDOG_DEFAULT_SEC


def start_watchdog(on_exit=None) -> bool:
    """启动进程级看门狗 (批次90, 幂等; 已在跑返回 False)。

    只应由服务入口 (``python -m adr.server``) 调用 — 库/测试不启动, 避免测试
    进程被误杀。``on_exit`` 是退出前清理回调 (回收训练/门禁/下载子进程)。
    """
    global _WATCHDOG_STARTED
    grace = watchdog_grace_sec()
    if grace <= 0:
        logging.getLogger("adr").info(
            "[watchdog] 已关闭 (ADR_ENGINE_WATCHDOG_SEC=0)")
        return False
    with _WATCHDOG_LOCK:
        if _WATCHDOG_STARTED:
            return False
        _WATCHDOG_STARTED = True
    threading.Thread(target=_watchdog_loop, args=(grace, on_exit),
                     name="adr-engine-watchdog", daemon=True).start()
    logging.getLogger("adr").info(
        "[watchdog] 已启动: 引擎降级超过 %.0fs 未恢复即退出进程 (exit %d, "
        "交由 Docker restart policy / 桌面壳 supervise 重启)",
        grace, _WATCHDOG_EXIT_CODE)
    return True


def _watchdog_loop(grace: float, on_exit) -> None:
    while True:
        time.sleep(_WATCHDOG_POLL_SEC)
        waited = degraded_seconds()
        if waited < grace:
            continue
        logging.getLogger("adr").critical(
            "[watchdog] 引擎已降级 %.0fs (阈值 %.0fs): 卡死 worker 无法在进程内"
            "回收, 退出进程交由外部守护重启", waited, grace)
        log_event("engine_watchdog", degraded_s=round(waited, 1),
                  threshold_s=grace, action="exit")
        if on_exit is not None:
            try:
                on_exit()
            except Exception:
                logging.getLogger("adr").warning(
                    "[watchdog] 退出前清理子进程失败", exc_info=True)
        try:
            logging.shutdown()
        except Exception:
            pass
        os._exit(_WATCHDOG_EXIT_CODE)


def stage() -> str:
    """预热细分阶段 (人读字符串, 不触发加载): ready 优先, 其余见 _STAGE。"""
    if is_ready():
        return "ready"
    return _STAGE


def stage_error() -> str:
    """预热失败原因摘要 (批次73; 非 failed 阶段返回空串, 不触发加载)。

    取 traceback 末几行 (异常类型 + 消息 + 最后两帧), 截断到 1200 字:
    过长的栈在预热页展开区反而看不清, 用户要的是"哪一步因为什么炸了"。
    """
    return _STAGE_ERROR if _STAGE == "failed" else ""


def _error_summary(exc: BaseException) -> str:
    """把异常整理成人读摘要 (纯函数, 单测覆盖)。"""
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    # 末 6 行足以覆盖 "xxxError: msg" + 关键两帧, 首行(File 链)也留
    tail = [ln.rstrip() for ln in lines[-6:] if ln.strip()]
    txt = "\n".join(tail) if tail else f"{type(exc).__name__}: {exc}"
    if len(txt) > 1200:
        txt = "…" + txt[-1200:]
    return txt


def stage_text() -> str:
    """downloading 阶段的实时进度文案 (批次44 下载提示), 其余阶段返回空串。

    由 server stats 透传 → 桌面壳轮询 → prewarm 页显示下载进度。
    """
    if _STAGE != "downloading":
        return ""
    try:
        from adr.models import gsv_bootstrap

        st = gsv_bootstrap.status()
    except Exception:
        return "正在下载预训练模型…"
    b = st.get("stage")
    if b == "downloading":
        done, total = st.get("downloaded", 0), st.get("total", 0)
        mb = done / 1048576
        if total > 0:
            return (f"正在下载 {st.get('file', '')} "
                    f"{mb:.1f}/{total / 1048576:.1f} MB ({int(done * 100 / total)}%)")
        return f"正在下载 {st.get('file', '')} {mb:.1f} MB"
    if b == "extracting":
        return f"正在解压 {st.get('file', '')}…"
    return "正在下载预训练模型…"


def queue_depth() -> int:
    """进程级排队数 (批次23, 引擎未初始化时 0)。"""
    return _ENGINE.queue_depth() if _ENGINE is not None else 0


def synth_busy() -> int:
    """进程级合成中请求数 (批次23, 引擎未初始化时 0)。"""
    return _ENGINE.synth_busy() if _ENGINE is not None else 0


def sched_stats() -> Optional[dict]:
    """进程级调度报告 (批次63, 引擎未初始化时 None, 不触发加载)。"""
    return _ENGINE.sched_stats() if _ENGINE is not None else None


def get_gsv_engine() -> GSVEngine:
    """进程级单例 (批次41a: 双重检查锁 — 已初始化后走无锁快路径,
    多线程首次并发调用只创建一个实例)。"""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = GSVEngine()
    return _ENGINE


def prewarm(default_profile: str | None = None) -> None:
    """后台预热入口 (批次26): 阻塞调用, 放后台线程跑。

    状态机 queued→importing→loading→kernel→ready/failed 由本函数独占管理
    (_LOADING/_STAGE 只在模块内部读写, 外部经 is_loading()/stage() 只读观察)。
    服务端 (server/app.py) 启动线程调用; 桌面壳 prewarm 页面轮询 stage() 显示。
    失败不致命 (首次合成时仍会按需加载), 只打日志。
    """
    global _LOADING, _STAGE, _STAGE_ERROR
    log = logging.getLogger("adr.models.gsv_engine")
    _LOADING = True
    _STAGE_ERROR = ""  # 批次73: 重试时清掉上一轮失败原因
    _STAGE = "queued"
    try:
        _STAGE = "importing"  # import torch + GSV 模块 (最耗时可达 20s)
        from adr.models.voice_library import list_voices, load_voice

        prof_name = default_profile
        if not prof_name:
            voices = list_voices()
            prof_name = voices[0] if voices else None
        kw: dict = {}
        prof = None
        if prof_name:
            try:
                prof = load_voice(prof_name)
                kw = {"vits_weights": prof.get("vits_weights"),
                      "t2s_weights": prof.get("t2s_weights")}
                log.info("[prewarm] 用音色档案「%s」的权重预热", prof_name)
            except Exception:
                prof = None  # 档案损坏 → 退回纯预训练权重
        # 批次44: 预训练资源 bootstrap — 便携包不含底模, 首启按需下载+解压
        # (已就位时 pending() 为空, 零开销直接过)
        from adr.models import gsv_bootstrap

        if gsv_bootstrap.pending():
            _STAGE = "downloading"  # 进度经 stage_text() → stats → 壳透传
            gsv_bootstrap.ensure()
        _STAGE = "loading"    # 权重加载进显存
        eng = get_gsv_engine()
        eng.warmup(**{k: v for k, v in kw.items() if v})
        log.info("[prewarm] GPT-SoVITS 引擎就绪")
        # kernel JIT 预热: 否则首次合成再付 ~14s CUDA 编译 (只取首块)
        try:
            _STAGE = "kernel"
            if prof:
                for _ in eng.synthesize_stream(
                        "引擎预热。", prof["ref_audio"],
                        vits_weights=kw.get("vits_weights"),
                        split_method="cut0"):
                    break
                log.info("[prewarm] kernel 预热完成, 首次合成即秒级")
        except Exception as e:
            log.warning("[prewarm] kernel 预热失败 (不影响功能): %s", e)
        _STAGE = "ready"
        # 空闲期后台 profile + policy (仅 ready 态调度, failed 不触发)
        _schedule_idle_profile()
    except Exception as e:
        _STAGE = "failed"
        # 批次73: 保留失败摘要并经 stage_error() → health → 预热页展示,
        # 不再只落一行 warning 让用户对着"预热失败"猜原因。
        _STAGE_ERROR = _error_summary(e)
        log.warning("[prewarm] GSV 预热失败 (首次合成时将现场加载): %s\n%s",
                    e, _STAGE_ERROR)
    finally:
        _LOADING = False


_PROFILE_IDLE_DELAY = 10.0  # 秒: 预热 ready 后的空闲期, 之后再启动后台采样
_PROFILE_BUILD_DELAY = 3.0  # 秒: 建档后的短缓冲 (让建档请求的响应先回)
_PROFILE_LOCK = threading.Lock()
# idle=可调度 / running=采样中 / done=本进程不再调度 (已有 profile、已测完或失败)
_PROFILE_STATE = "idle"


def _start_profile(delay: float) -> bool:
    """调度后台 profile + policy; 非 idle 态不重复启动。返回是否启动了线程。"""
    global _PROFILE_STATE
    with _PROFILE_LOCK:
        if _PROFILE_STATE != "idle":
            return False
        _PROFILE_STATE = "running"
    threading.Thread(target=_idle_profile_worker, args=(delay,),
                     name="adr-idle-profile", daemon=True).start()
    return True


def _schedule_idle_profile() -> None:
    """预热 ready 后调度后台 profile + policy。

    daemon 线程: 先等空闲期, 再等引擎无排队/无合成。本机已有 profile 则
    本进程不再调度; 档案库为空 (profiler 至少需要一个档案) 则退回 idle,
    建档后由 try_schedule_profile 当场补跑, 不必等下次启动。
    采样经引擎调度器排队, 期间到来的用户请求排队等本次测完。
    """
    _start_profile(_PROFILE_IDLE_DELAY)


def try_schedule_profile() -> None:
    """建档 / 导入档案成功后调用: 本机尚无 profile 则当场调度后台采样。

    调用方 (save_voice_auto / import_voice) 不应因此失败, 故吞掉一切异常。
    """
    try:
        _start_profile(_PROFILE_BUILD_DELAY)
    except Exception as e:
        logging.getLogger("adr.models.gsv_engine").warning(
            "[profile] 建档后调度采样失败 (不影响建档): %s", e)


def _idle_profile_worker(delay: float = _PROFILE_IDLE_DELAY) -> None:
    import time
    global _PROFILE_STATE
    log = logging.getLogger("adr.models.gsv_engine")
    final = "done"
    try:
        time.sleep(delay)
        from adr.core import policy, profiler

        existing = profiler.load_profile()
        if existing is not None:
            log.info("[profile] 本机已有 profile, 跳过后台采样")
            # 换包后当前设备画像已在, 但 policy.json 是另一设备生成的: 就地重新生成
            pol = policy.load()
            if pol is None or not policy._device_matches(pol):
                policy.generate(existing)
                policy.apply()
                log.info("[profile] policy 与当前设备不一致, 已按本设备画像重新生成")
            return
        from adr.models.voice_library import list_voices

        if not list_voices():
            final = "idle"   # 建档后可再调度
            log.info("[profile] 档案库为空, 跳过后台采样 (建档后自动补跑)")
            return
        eng = get_gsv_engine()
        while eng.queue_depth() > 0 or eng.synth_busy() > 0:
            time.sleep(1.0)
        log.info("[profile] 服务端空闲, 开始后台采样")
        result = profiler.run_profile()
        # 先落盘 profile: 否则下次启动 load_profile 仍为空, 每次都重采
        profiler.save_profile(result)
        # 采样完成立即生成 policy 并应用, 后续请求即受益
        policy.generate(result)
        policy.apply()
        log.info("[profile] 采样完成, policy 已生成并应用")
    except Exception as e:
        log.warning("[profile] 后台采样失败 (不影响功能): %s", e)
    finally:
        with _PROFILE_LOCK:
            _PROFILE_STATE = final
