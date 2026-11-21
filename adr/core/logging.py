"""统一日志 (基于 rich + loguru 风格 API)。"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


# 全局 console
_console: Optional[Console] = None

# 文件日志保留策略: 按日切割, 最多 14 份; 所有 adr.log* 合计不超过 200MB。
LOG_KEEP_DAYS = 14
LOG_MAX_TOTAL_BYTES = 200 * 1024 * 1024


def get_console() -> Console:
    """获取 rich console (单例)。"""
    global _console
    if _console is None:
        _console = Console(
            stderr=True,
            force_terminal=sys.stderr.isatty(),
            width=120,
        )
    return _console


def trim_rotated_logs(base_path: str, max_total_bytes: int) -> int:
    """按总量上限清理已轮转日志 (纯函数式副作用, 返回删除字节数)。

    ``TimedRotatingFileHandler`` 只按份数保留, 不控制总量: 单日日志暴涨
    (排障时 DEBUG + 循环重试) 时 14 份仍可能占数 GB。这里从最旧的文件开始
    删除, 直到 adr.log* 合计不超过上限。只处理 ``<base>.*`` 命中的轮转文件,
    当前写入中的 ``base`` 本体不删。
    """
    base = Path(base_path)
    texts = [base] if base.exists() else []
    try:
        texts += sorted(base.parent.glob(base.name + ".*"),
                        key=lambda p: p.stat().st_mtime)
    except OSError:
        return 0
    total = 0
    for p in texts:
        try:
            total += p.stat().st_size
        except OSError:
            continue
    freed = 0
    for p in texts:
        if total <= max_total_bytes:
            break
        if p == base:      # 正在写入的文件永不删
            continue
        try:
            size = p.stat().st_size
            p.unlink()
        except OSError:
            continue
        total -= size
        freed += size
    return freed


class DailyRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """按日轮转的文件 handler: 保留 14 份 + 总量 200MB 上限。

    以 ``when="midnight"`` 切割, 轮转文件命名为 ``<name>.YYYY-MM-DD``
    (与标准库一致)。每次切换后顺带按总量清理最旧的文件。
    """

    def __init__(self, filename: str, max_total_bytes: int = LOG_MAX_TOTAL_BYTES,
                 **kwargs):
        super().__init__(
            filename,
            when="midnight",
            interval=1,
            backupCount=LOG_KEEP_DAYS,
            encoding="utf-8",
            **kwargs,
        )
        self.max_total_bytes = max_total_bytes
        trim_rotated_logs(self.baseFilename, self.max_total_bytes)

    def doRollover(self) -> None:
        super().doRollover()
        trim_rotated_logs(self.baseFilename, self.max_total_bytes)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    show_path: bool = False,
) -> None:
    """配置全局日志。

    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 日志文件路径 (None 则只输出到 stderr); 文件按日轮转,
            保留 14 份且总量不超过 200MB
        show_path: 是否显示代码路径
    """
    # 根 logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 清空已有 handler
    root.handlers.clear()

    # rich handler (stderr)
    rich_handler = RichHandler(
        console=get_console(),
        show_path=show_path,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(rich_handler)

    # 文件 handler (按日轮转)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = DailyRotatingFileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)

    # 调整第三方库的日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def get_logger(name: str = "adr") -> logging.Logger:
    """获取 logger 实例。

    Args:
        name: logger 名称 (通常以 'adr.xxx' 开头)
    """
    return logging.getLogger(name)


# ===== 便捷 API =====
def info(msg: str, *args, **kwargs) -> None:
    get_logger("adr").info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    get_logger("adr").warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    get_logger("adr").error(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs) -> None:
    get_logger("adr").debug(msg, *args, **kwargs)


def success(msg: str) -> None:
    """打印成功消息 (绿色 ✓)。"""
    get_console().print(f"[bold green]✓[/bold green] {msg}")


def fail(msg: str) -> None:
    """打印失败消息 (红色 ✗)。"""
    get_console().print(f"[bold red]✗[/bold red] {msg}")


def section(title: str) -> None:
    """打印分隔段落标题。"""
    get_console().print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    get_console().print(f"[bold cyan]{title}[/bold cyan]")
    get_console().print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")
