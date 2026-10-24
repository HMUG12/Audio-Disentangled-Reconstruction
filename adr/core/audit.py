"""安全审计日志 (批次62)。

JSONL 追加写 ``<adr_data_dir()>/logs/audit.log``, 每行一个事件::

    {"ts": "2026-10-24T12:00:00+08:00", "event": "synthesis", ...}

隐私边界:
- **不记明文** — 合成文本只记 sha256 前 16 位指纹 + 长度, 声纹来源声明
  同样只记指纹; 文本/声纹属个人敏感信息, 审计日志要能长期留存。
- 写失败**静默降级** — 审计是旁路设施, 不能阻断业务主流程。

``ADR_SAFETY_AUDIT=0`` 全局关闭 (settings.safety_audit_enabled)。
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime

_log = logging.getLogger("adr.audit")


def text_fp(text: str) -> str:
    """文本指纹 — sha256 前 16 位。审计日志中代替明文定位同文本。"""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def log_event(event: str, **fields) -> None:
    """追加一条审计事件。任何异常都吞掉 (降级不阻断业务)。"""
    try:
        from adr.core import settings  # 延迟导入: 避免 core 间循环
        if not settings.safety_audit_enabled():
            return
        from adr.core.config import adr_data_dir
        log_dir = adr_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        rec = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
               "event": event, **fields}
        with open(log_dir / "audit.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        _log.warning("audit 写入失败 (已降级跳过)")
