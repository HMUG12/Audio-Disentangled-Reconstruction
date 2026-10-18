"""实验/档案名称清洗 (批次58): Windows 落盘安全。

训练 exp 名最终会成为 logs/<exp>/ 目录名。Windows 设备保留名
(CON/PRN/AUX/NUL/COM1-9/LPT1-9, 大小写不敏感, 含 CON.txt 带扩展名形式)
与尾部点/空格会导致 mkdir 失败或被重定向为设备, 必须清洗。
"""
from __future__ import annotations

import re
from unicodedata import category as _ucat

# Windows 设备保留名 (含带扩展名形式; 判断对尾部点/空格截断后的基名生效)
_WINDOWS_RESERVED = re.compile(
    r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$", re.IGNORECASE)

_INVALID_CHARS = '\\/:*?"<>|'


def is_windows_reserved(name: str) -> bool:
    """是否命中 Windows 设备保留名 (按尾部点/空格截断后的基名判断)。"""
    return bool(_WINDOWS_RESERVED.match(name.rstrip(". ")))


def sanitize_exp_name(name: str, fallback: str = "my_voice") -> str:
    """清洗为合法目录名: 去控制字符/非法字符/尾部点空格, 保留名加下划线前缀。

    清空时回退 fallback; 输出保证在 Windows/Linux 均可直接 mkdir。
    """
    out = "".join(
        ch for ch in str(name or "")
        if _ucat(ch) != "Cc" and ch not in _INVALID_CHARS)
    out = out.strip().rstrip(". ") or fallback
    if is_windows_reserved(out):
        out = f"_{out}"
    return out
