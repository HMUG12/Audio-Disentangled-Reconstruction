"""使用条款同意标记 (批次62)。

无 UI 形态 (CLI / server 直跑) 的条款 gate 落点: 同意后写
``<adr_data_dir()>/.terms_agreed_v1`` 标记文件, 后续启动检测到即放行。
壳 (Tauri) 形态的同意由 launcher 条款视图担保, 不走本文件。

``ADR_AGREE_TERMS`` env 与 ``--agree-terms`` 参数是两种显式同意入口
(解析在 server/__main__.py, settings.agree_terms 只读 env)。
"""
from __future__ import annotations

FLAG_NAME = ".terms_agreed_v1"


def flag_path():
    from adr.core.config import adr_data_dir
    return adr_data_dir() / FLAG_NAME


def agreed() -> bool:
    """是否已有同意标记。读失败 (权限等) 视为未同意 — 保守拒绝。"""
    try:
        return flag_path().exists()
    except OSError:
        return False


def mark() -> bool:
    """写入同意标记 (含毫秒时间戳)。已存在直接成功; 写失败返回 False 不抛。"""
    import time
    try:
        p = flag_path()
        if p.exists():
            return True
        p.parent.mkdir(parents=True, exist_ok=True)  # 数据目录可能尚未创建
        p.write_text(f"agreed_at={int(time.time() * 1000)}\n", encoding="utf-8")
        return True
    except OSError:
        return False
