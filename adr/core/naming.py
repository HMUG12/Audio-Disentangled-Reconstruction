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

# 批次65 安全收口: exp/档案名会流入 gsv_finetune.py / rvc_train.py 的
# subprocess.run(shell=True) 命令串 (作为 slicer_opt/<exp> 等路径段), Linux
# /bin/sh 在双引号内仍会做命令替换/变量展开, 构成 RCE。这些字符在合法的
# 目录/档案名中本就无意义, 在清洗咽喉处一并剔除, 一处堵住所有训练端点
# (/train/start, webui) 的注入面。与 _INVALID_CHARS 合并后统一过滤。
_SHELL_META = "$`;&(){}[]!~'"


def is_windows_reserved(name: str) -> bool:
    """是否命中 Windows 设备保留名 (按尾部点/空格截断后的基名判断)。"""
    return bool(_WINDOWS_RESERVED.match(name.rstrip(". ")))


def sanitize_exp_name(name: str, fallback: str = "my_voice") -> str:
    """清洗为合法目录名: 去控制字符/非法字符/尾部点空格, 保留名加下划线前缀。

    清空时回退 fallback; 输出保证在 Windows/Linux 均可直接 mkdir,
    且可安全拼入 shell 命令串(已剔除 shell 元字符)。
    """
    _keep = _INVALID_CHARS + _SHELL_META
    out = "".join(
        ch for ch in str(name or "")
        if _ucat(ch) != "Cc" and ch not in _keep)
    out = out.strip().rstrip(". ") or fallback
    if is_windows_reserved(out):
        out = f"_{out}"
    return out
