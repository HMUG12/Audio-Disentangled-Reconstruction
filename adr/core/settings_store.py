"""控制台全局设置存储 (settings.json)。

定位: 这里是**控制台前端推理参数的默认值**, 由设置页写入全局; 控制台里
临时调整的值只留在前端, 不落盘。本模块**绝不参与** /api/adr/v1/tts 的
默认参数推导 —— 未显式传参的外部 API 调用方行为保持不变
(native.py / param_norm.py 的默认值逻辑不动)。

落盘: ``adr_data_dir()/settings.json``
并发: 模块级 RLock 串行化读改写; 原子写 (同目录唯一 ``.tmp`` + os.replace)
容错: JSON 损坏 / 顶层非对象 / 已知字段非法 → 备份为
      ``settings.json.bad.<时间戳>`` 并回落默认值, 不静默夹取。
兼容: 文件保留 ``schema_version``; 读取时缺失/更小则用默认值深合并迁移,
      未知字段忽略 (为未来版本留余地)。
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from adr.core.config import adr_data_dir

_log = logging.getLogger("adr.core.settings_store")

SCHEMA_VERSION = 1

# 默认值 (字段集合即契约, 只增不改)
DEFAULT_SETTINGS: dict[str, Any] = {
    "synth": {
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "speed_factor": 1.0,
        "seed": -1,
        "repetition_penalty": 1.35,
        "media_type": "wav",
        "streaming_mode": "off",
    },
    "voice": {
        "default_profile": "",
        "language": "zh",
    },
    "ui": {
        "theme": "default",
        "lang": "zh",
        "motion": True,
        "onboarding_done": False,
    },
    "security": {
        "protect_export": False,
    },
}

_LOCK = threading.RLock()


# ===== 校验器 =====
def _v_int(name: str, lo: int, hi: int) -> Callable[[Any], int]:
    def _check(value: Any) -> int:
        # 校验层统一抛 ValueError: API 只需捕获一种异常映射成 400
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(  # noqa: TRY004
                f"{name} 必须是 {lo}~{hi} 的整数 (收到 {value!r})")
        if not lo <= value <= hi:
            raise ValueError(f"{name} 超出范围, 需在 {lo}~{hi} 之间 (收到 {value})")
        return value

    return _check


def _v_float(name: str, lo: float, hi: float) -> Callable[[Any], float]:
    def _check(value: Any) -> float:
        # bool 是 int 子类, True/False 不能混进数值字段
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(  # noqa: TRY004
                f"{name} 必须是 {lo}~{hi} 的数字 (收到 {value!r})")
        val = float(value)
        # NaN 走 not(lo <= val <= hi) 同样被拒
        if not lo <= val <= hi:
            raise ValueError(f"{name} 超出范围, 需在 {lo}~{hi} 之间 (收到 {value})")
        return val

    return _check


def _v_choice(name: str, allowed: tuple[str, ...]) -> Callable[[Any], str]:
    def _check(value: Any) -> str:
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(
                f"{name} 必须是 {'/'.join(allowed)} 之一 (收到 {value!r})")
        return value

    return _check


def _v_bool(name: str) -> Callable[[Any], bool]:
    def _check(value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError(  # noqa: TRY004
                f"{name} 必须是布尔值 true/false (收到 {value!r})")
        return value

    return _check


def _v_str(name: str) -> Callable[[Any], str]:
    def _check(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(  # noqa: TRY004
                f"{name} 必须是字符串 (收到 {value!r})")
        return value

    return _check


_MAX_SEED = 2**31 - 1

_VALIDATORS: dict[str, dict[str, Callable[[Any], Any]]] = {
    "synth": {
        "top_k": _v_int("synth.top_k", 1, 100),
        "top_p": _v_float("synth.top_p", 0.05, 1.0),
        "temperature": _v_float("synth.temperature", 0.05, 2.0),
        "speed_factor": _v_float("synth.speed_factor", 0.5, 2.0),
        "seed": _v_int("synth.seed", -1, _MAX_SEED),
        "repetition_penalty": _v_float("synth.repetition_penalty", 1.0, 2.0),
        "media_type": _v_choice("synth.media_type", ("wav", "mp3", "ogg")),
        "streaming_mode": _v_choice("synth.streaming_mode", ("off", "on", "auto")),
    },
    "voice": {
        "default_profile": _v_str("voice.default_profile"),
        "language": _v_choice("voice.language", ("zh", "en")),
    },
    "ui": {
        "theme": _v_choice("ui.theme", ("default", "classic")),
        "lang": _v_choice("ui.lang", ("zh", "en")),
        "motion": _v_bool("ui.motion"),
        "onboarding_done": _v_bool("ui.onboarding_done"),
    },
    "security": {
        "protect_export": _v_bool("security.protect_export"),
    },
}


def settings_path() -> Path:
    """settings.json 落盘路径 (每次现算, 便于测试替换 adr_data_dir)。"""
    return adr_data_dir() / "settings.json"


def _defaults() -> dict[str, Any]:
    out = copy.deepcopy(DEFAULT_SETTINGS)
    # P0: 首启/全局设置选择的控制台语言经 ADR_UI_LANG 注入, 无 settings.json 时即默认值
    lang = (os.environ.get("ADR_UI_LANG") or "").strip().lower()
    if lang in ("zh", "en"):
        out["ui"]["lang"] = lang
    return out


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    """文件内容 → 完整设置: 已知字段校验, 缺失补默认, 未知字段忽略。"""
    out: dict[str, Any] = {}
    for group, fields in _VALIDATORS.items():
        raw_group = data.get(group)
        if raw_group is None:
            out[group] = copy.deepcopy(DEFAULT_SETTINGS[group])
            continue
        if not isinstance(raw_group, dict):
            raise ValueError(  # noqa: TRY004
                f"{group} 必须是对象 (收到 {type(raw_group).__name__})")
        merged = copy.deepcopy(DEFAULT_SETTINGS[group])
        for key, check in fields.items():
            if key in raw_group:
                merged[key] = check(raw_group[key])
        out[group] = merged
    return out


def _merge_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """部分嵌套对象 → 完整设置: 深合并已知字段, 未知键忽略。"""
    out = copy.deepcopy(base)
    for group, fields in _VALIDATORS.items():
        if group not in patch:
            continue
        raw = patch[group]
        if raw is None:   # 显式 null = 该组恢复默认
            out[group] = copy.deepcopy(DEFAULT_SETTINGS[group])
            continue
        if not isinstance(raw, dict):
            raise ValueError(  # noqa: TRY004
                f"{group} 必须是对象 (收到 {type(raw).__name__})")
        for key, check in fields.items():
            if key in raw:
                out[group][key] = check(raw[key])
    return out


def _backup_bad(path: Path, reason: object) -> None:
    """损坏文件改名备份 (settings.json.bad.<时间戳>), 失败也不阻断回落。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bad = path.with_name(f"{path.name}.bad.{stamp}-{uuid.uuid4().hex[:6]}")
    try:
        os.replace(str(path), str(bad))
        _log.warning("settings.json 不可用 (%s), 已备份为 %s, 回落默认值",
                     reason, bad.name)
    except OSError:
        _log.warning("settings.json 不可用 (%s), 备份失败, 本次使用默认值",
                     reason, exc_info=True)


def _load_unlocked() -> dict[str, Any]:
    """读文件并校验 (调用方持锁)。无文件/损坏 → 默认值。"""
    path = settings_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _defaults()
    except OSError as exc:
        _log.warning("读取 settings.json 失败, 本次使用默认值: %s", exc)
        return _defaults()

    try:
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("顶层必须是 JSON 对象")  # noqa: TRY004
        # schema_version 缺失/更小: _normalize 以默认值做底, 即合并迁移;
        # 未来版本 (更大) 也按已知字段解析, 未知字段忽略。
        return _normalize(raw)
    except ValueError as exc:
        _backup_bad(path, exc)
        return _defaults()


def _write_unlocked(settings: dict[str, Any]) -> None:
    """原子写 (调用方持锁): 同目录唯一 .tmp + os.replace。"""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION}
    payload.update(copy.deepcopy(settings))
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        for attempt in range(10):
            try:
                os.replace(str(tmp), str(path))
                return
            except PermissionError:
                # Windows: 目标被杀毒/索引短暂占用, 短重试
                if attempt == 9:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)


# ===== 公开接口 =====
def get_settings() -> dict[str, Any]:
    """完整设置 (深拷贝, 调用方随意改不会污染内部状态)。"""
    with _LOCK:
        return copy.deepcopy(_load_unlocked())


def update_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """部分更新并落盘, 返回更新后的完整设置; 非法值抛 ValueError。"""
    if not isinstance(patch, dict):
        raise ValueError("settings 更新必须是 JSON 对象")  # noqa: TRY004
    with _LOCK:
        updated = _merge_patch(_load_unlocked(), patch)
        _write_unlocked(updated)
    return copy.deepcopy(updated)


def reset_settings() -> dict[str, Any]:
    """恢复默认并落盘, 返回默认设置。"""
    with _LOCK:
        defaults = _defaults()
        _write_unlocked(defaults)
    return copy.deepcopy(defaults)


def updated_at() -> str | None:
    """settings.json 的 mtime (本地时区 ISO-8601); 无文件为 None。"""
    try:
        mtime = settings_path().stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds")
