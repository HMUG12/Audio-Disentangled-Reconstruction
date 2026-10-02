"""统一日志 (基于 rich + loguru 风格 API)。"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


# 全局 console
_console: Optional[Console] = None


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


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    show_path: bool = False,
) -> None:
    """配置全局日志。

    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 日志文件路径 (None 则只输出到 stderr)
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

    # 文件 handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
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
