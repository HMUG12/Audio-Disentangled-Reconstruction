"""调度层 Runtime 闭环 (m10 ④ 第三级, v1.3.x 次子批)。

纪要: 按初始配置运行, 监控 RTF / 首包延迟 / 显存水位, 在安全边界内微调;
异常时回退到最近一个已知良好配置, 并生成人可读的调度报告。

conservative-v1 边界 (与 Policy 一致):
- 只微调不影响音质的键 (policy.AUTO_KEYS 中的流式分块); 加速开关一律不碰。
- 显式 env 永远优先: 用户设过的键 runtime 不改。
- 微调: 滚动窗口 RTF 中位数 > slow_rtf → 收短流式分块 (与 Policy 慢机同值)。
- 回退: 微调后窗口失败率超阈 → 恢复微调前快照 (已知良好), 并冷却不再微调。
ADR_RUNTIME_AUTOTUNE=0 只监控不微调。
"""
from __future__ import annotations

import logging
import os
import statistics
import threading
import time
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

WINDOW = 20            # 滚动窗口样本数
MIN_SAMPLES = 5        # 少于此数不做判断
FAIL_RATE_FALLBACK = 0.3
TUNE_KEYS = ("ADR_TTS_STREAM_HEAD", "ADR_TTS_SPLIT_MAX")

_lock = threading.Lock()
_samples: deque = deque(maxlen=WINDOW)
_events: deque = deque(maxlen=20)
_applied: dict[str, str] = {}          # runtime 写入的 env 值 (区分来源)
_known_good: dict[str, str | None] | None = None
_frozen = False                        # 回退后冷却: 本进程不再微调


def _thresholds() -> dict:
    from adr.core import policy
    th = dict(policy.DEFAULT_THRESHOLDS)
    pol = policy.load()
    if pol and isinstance(pol.get("thresholds"), dict):
        th.update(pol["thresholds"])
    return th


def _is_explicit(k: str) -> bool:
    """env 里的值既非 policy 也非 runtime 写入 → 用户显式设置。"""
    from adr.core import policy
    if k not in os.environ:
        return False
    v = os.environ[k]
    return policy._applied.get(k) != v and _applied.get(k) != v


def applied_value(k: str) -> str | None:
    return _applied.get(k)


def _event(kind: str, reason: str, changes: dict) -> None:
    _events.append({"at": time.time(), "kind": kind, "reason": reason,
                    "changes": changes})
    log.info("[Runtime] %s: %s %s", kind, reason, changes)
    try:
        from adr.core.audit import log_event
        log_event("runtime_" + kind, reason=reason, **{k.lower(): v for k, v in changes.items()})
    except Exception:
        pass


def record(*, ok: bool, elapsed_s: float, audio_s: float = 0.0,
           ttfb_ms: int | None = None, mem_mb: int | None = None,
           stream: bool = False) -> None:
    """记录一次合成 (引擎在持槽后调用, 不含排队时间)。"""
    rtf = round(elapsed_s / audio_s, 3) if ok and audio_s > 0 else None
    with _lock:
        _samples.append({"ok": ok, "rtf": rtf, "ttfb_ms": ttfb_ms,
                         "mem_mb": mem_mb, "stream": stream})
        try:
            _evaluate()
        except Exception as e:      # 闭环失败不影响合成
            log.warning("[Runtime] 评估失败: %s", e)


def _evaluate() -> None:
    global _known_good, _frozen
    from adr.core import settings
    if not settings.runtime_autotune() or len(_samples) < MIN_SAMPLES:
        return
    fails = sum(1 for s in _samples if not s["ok"])
    fail_rate = fails / len(_samples)
    # 回退: 只回退 runtime 自己做过的微调
    if _applied and fail_rate >= FAIL_RATE_FALLBACK and _known_good is not None:
        restored = {}
        for k in list(_applied):
            if os.environ.get(k) == _applied[k]:     # 期间未被显式改过
                prev = _known_good.get(k)
                if prev is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev
                restored[k] = prev
            _applied.pop(k)
        _frozen = True
        _samples.clear()
        _event("fallback", f"失败率 {fail_rate:.0%} ≥ {FAIL_RATE_FALLBACK:.0%}, 回退已知良好配置", restored)
        return
    if _frozen or _applied:
        return
    rtfs = [s["rtf"] for s in _samples if s["rtf"] is not None]
    if len(rtfs) < MIN_SAMPLES or fails:
        return
    th = _thresholds()
    med = statistics.median(rtfs)
    if med <= float(th["slow_rtf"]):
        return
    target = {"ADR_TTS_STREAM_HEAD": str(int(th["slow_stream_head"])),
              "ADR_TTS_SPLIT_MAX": str(int(th["slow_split_max"]))}
    changes = {}
    snapshot = {}
    for k, v in target.items():
        if _is_explicit(k) or os.environ.get(k) == v:
            continue
        snapshot[k] = os.environ.get(k)
        os.environ[k] = v
        _applied[k] = v
        changes[k] = v
    if changes:
        _known_good = snapshot
        _samples.clear()        # 微调后窗口重新计, 回退判断只看新配置下的样本
        _event("tune", f"窗口 RTF 中位数 {med:.2f} > {th['slow_rtf']}, 收短流式分块", changes)


def report() -> dict[str, Any]:
    """导出调度报告 (配合 scheduler.stats / policy.export_snapshot)。

    批次64 第一子批: 附带调度预算 sched_budget (N=2 固定份额时为 2)。
    """
    from adr.core import settings
    with _lock:
        win = [dict(s) for s in _samples]
        evts = [dict(e) for e in _events]
    return {
        "sched_budget": settings.scheduler_max_concurrency(),
        "autotune": settings.runtime_autotune(),
        "window_size": len(win),
        "window_ok": sum(1 for s in win if s["ok"]),
        "window_rtf": [s["rtf"] for s in win if s["rtf"] is not None],
        "runtime_applied": dict(_applied),
        "known_good": dict(_known_good) if _known_good else None,
        "frozen": _frozen,
        "events": evts,
    }

