"""本机桌面账号口令校验 (只读, 供「导出模型包」二次验证)。

账号文件由桌面壳维护: ``<数据目录>/.local_account``, 内容为
``{"username","salt","hash","iterations","updated_at"}``, 口令按
PBKDF2-HMAC-SHA256 派生。服务端只在需要二次验证的接口里读它,
不改写、不缓存明文。
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

from adr.core.config import adr_data_dir


def _path() -> Path:
    return adr_data_dir() / ".local_account"


def verify_password(password: str) -> bool:
    """口令是否匹配本机账号 (无账号文件 / 坏文件一律 False)。"""
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        salt = bytes.fromhex(str(data["salt"]))
        stored = bytes.fromhex(str(data["hash"]))
        iterations = int(data["iterations"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not password or iterations <= 0:
        return False
    calc = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(calc, stored)
