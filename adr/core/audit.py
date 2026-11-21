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
import hmac
import json
import logging
import os
import threading
from datetime import datetime

_log = logging.getLogger("adr.audit")

# OPS-01 (批次89): 进程内串行 + 按大小轮转, 避免并发写交错与日志无限增长。
_lock = threading.Lock()
MAX_BYTES = 5 * 1024 * 1024   # 单文件上限, 超过即轮转
BACKUPS = 3                   # 保留 audit.log.1~.3


def _rotate(log_path) -> None:
    """audit.log → .1, 旧档依次后移, 超出 BACKUPS 的最旧档被覆盖。"""
    for i in range(BACKUPS - 1, 0, -1):
        src = log_path.with_name(f"{log_path.name}.{i}")
        if src.exists():
            src.replace(log_path.with_name(f"{log_path.name}.{i + 1}"))
    log_path.replace(log_path.with_name(f"{log_path.name}.1"))


_FP_SECRET: bytes | None = None


def _fp_secret() -> bytes:
    """S7: 每台机器一份随机盐 — 无盐 sha256 可被字典反推短文本 (如人名/句子)。"""
    global _FP_SECRET
    if _FP_SECRET:
        return _FP_SECRET
    try:
        from adr.core.config import adr_data_dir
        p = adr_data_dir() / ".audit_fp_key"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(os.urandom(32).hex(), encoding="ascii")
        _FP_SECRET = p.read_text(encoding="ascii").strip().encode("ascii")
    except Exception:
        # 数据目录不可写时降级为固定盐, 仍优于无盐 (且保证进程内确定性)
        _FP_SECRET = b"adr-audit-fp-v1"
    return _FP_SECRET


def text_fp(text: str) -> str:
    """文本指纹 — 带本机随机盐的 HMAC-SHA256 前 16 位 (S7)。"""
    return hmac.new(_fp_secret(), text.encode("utf-8", errors="replace"),
                    hashlib.sha256).hexdigest()[:16]


def log_event(event: str, **fields) -> None:
    """追加一条审计事件。任何异常都吞掉 (降级不阻断业务)。"""
    try:
        from adr.core import settings  # 延迟导入: 避免 core 间循环
        if not settings.safety_audit_enabled():
            return
        from adr.core.config import adr_data_dir
        log_dir = adr_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "audit.log"
        rec = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
               "event": event, **fields}
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with _lock:
            try:
                if log_path.stat().st_size >= MAX_BYTES:
                    _rotate(log_path)
            except OSError:
                pass  # 首次写入 / 文件被占用时跳过轮转, 继续追加
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        _log.warning("audit 写入失败 (已降级跳过)")
