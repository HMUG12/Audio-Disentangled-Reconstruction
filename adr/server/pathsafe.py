"""路径安全 helper (Track B 收口): 用户可控路径点统一清洗。

规则:
- resolve_within(root, candidate): candidate 解析后必须落在 root 目录内,
  否则返回 None — 用于"目录内取文件"场景 (档案参考音频等);
- is_safe_name(name): 名称类参数必须是单段纯名 (无路径分隔符/..),
  用于档案名等直接拼路径的字符串。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from adr.core.naming import is_windows_reserved


def resolve_within(root: Union[str, Path],
                   candidate: Union[str, Path]) -> Optional[Path]:
    """candidate 解析为绝对路径, 必须位于 root 内; 越界 (../ 穿越 / 外链
    符号链接 / 换盘符) 返回 None, 合法返回解析后的绝对 Path。

    resolve() 同时规范化 ../ 与符号链接, 两类穿越一并拦截。
    """
    try:
        root_p = Path(root).resolve()
        cand_p = Path(candidate)
        if not cand_p.is_absolute():
            # 相对 candidate 基于 root 解析 (resolve() 默认基于 cwd, 会误判越界)
            cand_p = root_p / cand_p
        cand_p = cand_p.resolve()
        cand_p.relative_to(root_p)
    except (ValueError, OSError):
        return None
    return cand_p


def is_safe_name(name: str) -> bool:
    """名称类参数 (档案名等) 必须是单段纯名, 拒绝任何路径成分穿越,
    并拒绝 Windows 设备保留名 (CON/NUL 等会致落盘失败或重定向为设备)。"""
    if not name:
        return False
    # ":" 拦 Windows 盘符相对名 (如 "D:evil"): pathlib join 丢基路径换盘符 (批次62)。
    if any(ch in name for ch in ("/", "\\", "\x00", ":")):
        return False
    if ".." in name:
        return False
    return not is_windows_reserved(name)
