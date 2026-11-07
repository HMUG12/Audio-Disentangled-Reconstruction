"""实验/档案名称清洗 (批次58): Windows 落盘安全。

训练 exp 名最终会成为 logs/<exp>/ 目录名。Windows 设备保留名
(CON/PRN/AUX/NUL/COM1-9/LPT1-9, 大小写不敏感, 含 CON.txt 带扩展名形式)
与尾部点/空格会导致 mkdir 失败或被重定向为设备, 必须清洗。

本模块同时是路径安全判定的唯一实现 (批次89): `is_safe_name` /
`resolve_within` 原先在 adr/models/voice_library.py 与
adr/server/pathsafe.py 各有一份, 现统一于此, 两处只做再导出,
避免规则漂移导致"服务端拦住、落盘端放行"。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union
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
