"""调度层 Policy: Profiler 画像 → 配置档 (m10 ④ 第二级, v1.3.x 首子批)。

规则集 conservative-v1:
- 音质敏感开关 (STATIC_KV / CUDA_GRAPH / INT8) 只推荐, 永不自动开;
  音质门通过前默认值不变 (gsv_engine 的 setdefault 链不动)。
- 不影响音质的开关 (流式分块 / torch 线程数) 可自动调, 写入 overrides。
- 不查设备表: 只用 profile 实测的 RTF / 显存 / 核数推断。

policy.json = overrides + 元数据 (policy_version / rules / thresholds /
recommendations / 来源 profile)。完整生效快照走 export_snapshot (诊断导出)。
生效优先级: 显式 env > policy.json overrides > 代码默认值。
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

POLICY_VERSION = 1          # 规则或 schema 变化时递增
RULES = "conservative-v1"

# 可自动生效的键 (不影响音质)。其余键即使出现在 overrides 也不应用。
AUTO_KEYS = ("ADR_TORCH_THREADS", "ADR_TTS_STREAM_HEAD", "ADR_TTS_SPLIT_MAX")
# 音质敏感键: 只推荐, 用户须显式设 env 才生效
GATED_KEYS = ("ADR_T2S_STATIC_KV", "ADR_T2S_CUDAGRAPH", "ADR_T2S_CUDAGRAPH_BUCKETS")

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "slow_rtf": 1.0,              # RTF 高于此值视为慢机, 收短流式分块
    "slow_stream_head": 10,       # 慢机首段字数 (默认 14)
    "slow_split_max": 30,         # 慢机长句细分字数 (默认 40)
    "cpu_reserve_cores": 1,       # CPU 推理时给系统留的物理核数
    "cpu_min_cores_reserve": 4,   # 物理核数达到此值才留核
    "graph_min_mem_gb": 6.0,      # 推荐分桶 CUDA Graph 的显存下限
}


def policy_path() -> Path:
    from adr.core.config import adr_data_dir
    return adr_data_dir() / "policy.json"


def _physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def infer(profile: dict | None, thresholds: dict | None = None,
          physical_cores: int | None = None) -> dict[str, Any]:
    """profile → policy dict (纯函数, 不写盘、不改 env)。"""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    overrides: dict[str, str] = {}
    recs: dict[str, dict] = {}

    dev = (profile or {}).get("device_info", {}) or {}
    points = (profile or {}).get("profiles", []) or []
    default = next((p for p in points if p.get("name") == "default"), None)
    rtf = float(default["rtf"]) if default and default.get("rtf") else None
    is_cuda = dev.get("type") == "cuda"

    # 自动项 1: 慢机收短流式分块 (首包更快, 不改模型输出分布)
    if rtf is not None and rtf > th["slow_rtf"]:
        overrides["ADR_TTS_STREAM_HEAD"] = str(int(th["slow_stream_head"]))
        overrides["ADR_TTS_SPLIT_MAX"] = str(int(th["slow_split_max"]))

    # 自动项 2: CPU 推理留核给系统
    if dev.get("type") == "cpu":    # unknown 设备不猜
        cores = physical_cores if physical_cores is not None else _physical_cores()
        if cores >= int(th["cpu_min_cores_reserve"]):
            overrides["ADR_TORCH_THREADS"] = str(cores - int(th["cpu_reserve_cores"]))

    # 推荐项 (不生效): 音质门通过前一律保持默认关
    gate = "音质门未通过, 需显式设 env 才生效"
    if is_cuda:
        recs["ADR_T2S_STATIC_KV"] = {"value": "1", "reason": f"与原路径逐 token 位等; {gate}"}
        mem = float(dev.get("total_mem_gb") or 0)
        if mem >= float(th["graph_min_mem_gb"]):
            recs["ADR_T2S_CUDAGRAPH"] = {"value": "1", "reason": f"配合分桶; {gate}"}
            recs["ADR_T2S_CUDAGRAPH_BUCKETS"] = {"value": "default", "reason": gate}
    # INT8 推理侧尚无开关 (无对应 env), 只记推荐结论
    recs["int8"] = {"value": "off", "reason": "推理侧未实现 INT8, 保持关"}

    return {
        "policy_version": POLICY_VERSION,
        "rules": RULES,
        "generated_at": time.time(),
        "profile_id": (profile or {}).get("profile_id"),
        "basis": {"device": dev.get("type"), "rtf": rtf},
        "thresholds": th,
        "overrides": overrides,
        "recommendations": recs,
    }


def save(policy: dict, path: Path | None = None) -> Path:
    path = path or policy_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load(path: Path | None = None) -> dict | None:
    path = path or policy_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("[Policy] policy.json 无法解析, 忽略: %s", e)
        return None


def generate(profile: dict | None, path: Path | None = None) -> dict:
    """重新推断并写盘; 沿用已有 policy.json 里用户改过的 thresholds。"""
    prev = load(path)
    th = (prev or {}).get("thresholds") if prev else None
    pol = infer(profile, th if isinstance(th, dict) else None)
    save(pol, path)
    return pol


# 本进程由 policy 写入的 env 键 (区分来源用; 显式 env 不在此表)
_applied: dict[str, str] = {}


def apply(policy: dict | None = None) -> dict[str, str]:
    """把 overrides 中可自动生效的键写入 env (显式 env 优先, 不覆盖)。

    只应用 AUTO_KEYS; 音质敏感键即使被手写进 overrides 也忽略并告警。
    policy_version 高于本代码认识的版本时整体不应用 (防降级读错)。
    """
    pol = policy if policy is not None else load()
    if not pol:
        return {}
    ver = pol.get("policy_version")
    if not isinstance(ver, int) or ver > POLICY_VERSION:
        log.warning("[Policy] policy_version=%r 不受支持 (当前 %d), 忽略", ver, POLICY_VERSION)
        return {}
    done: dict[str, str] = {}
    for k, v in (pol.get("overrides") or {}).items():
        if k not in AUTO_KEYS:
            log.warning("[Policy] %s 不在自动生效白名单 (conservative-v1), 忽略", k)
            continue
        if k in os.environ and _applied.get(k) != os.environ[k]:
            continue   # 显式 env 优先
        os.environ[k] = str(v)
        _applied[k] = str(v)
        done[k] = str(v)
    if done:
        log.info("[Policy] 已应用 overrides: %s", done)
    return done


def export_snapshot() -> dict[str, Any]:
    """诊断导出: 全部调度相关键的当前生效值 + 来源 (env / policy / default)。"""
    pol = load()
    ver = (pol or {}).get("policy_version")
    pending = ((pol or {}).get("overrides") or {}) if isinstance(ver, int) and ver <= POLICY_VERSION else {}
    keys = (*AUTO_KEYS, *GATED_KEYS, "ADR_SCHED_MAX_CONCURRENCY")
    eff = {}
    for k in keys:
        if k in os.environ:
            src = "policy" if _applied.get(k) == os.environ[k] else "env"
            eff[k] = {"value": os.environ[k], "source": src}
        elif k in AUTO_KEYS and k in pending:
            # 本进程未 apply (如 CLI 诊断): 显示引擎启动时将生效的值
            eff[k] = {"value": str(pending[k]), "source": "policy"}
        else:
            eff[k] = {"value": None, "source": "default"}
    return {
        "policy_version": POLICY_VERSION,
        "rules": RULES,
        "policy_file": str(policy_path()),
        "policy_loaded": pol is not None,
        "effective": eff,
        "recommendations": (pol or {}).get("recommendations", {}),
    }
