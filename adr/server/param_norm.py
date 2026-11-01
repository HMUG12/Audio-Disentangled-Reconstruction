"""v2/v3 参数归一单点实现 (审计 D5): 协议层只取值, 规则集中在此。

历史: v2 有 ``_bad_number`` 区间校验, v3 的 overrides 另写一套 (缺 NaN/Inf
拦截与语言白名单), 两处语义分叉。本模块把「类型转换 → 区间/有限性 →
语言白名单」固定成一条路径, v2 (HTTP) 与 v3 (WS init) 只做协议适配。

错误文案属于 wire 契约 (v2 的 400 message / v3 的 error frame), 改动需同步测试。
"""
from __future__ import annotations

import math
import os

# GSV v2_languages 全集 (third_party/gpt_sovits TTS.py), v1 模型为其子集
SUPPORTED_LANGS = frozenset({
    "auto", "auto_yue", "en", "zh", "ja", "yue", "ko",
    "all_zh", "all_ja", "all_yue", "all_ko",
})

# 出现即归一的字段 (None/未传不校验)
_INT_FIELDS = ("top_k", "seed", "batch_size")
_FLOAT_FIELDS = ("top_p", "temperature", "speed_factor", "fragment_interval")
_LANG_FIELDS = ("text_lang", "prompt_lang")

# 数值合法闭区间 (speed 区间可 env 调)
_FIXED_BOUNDS = {
    "top_k": (1, 1000),
    "top_p": (0.0, 1.0),
    "temperature": (0.0, 10.0),
    "fragment_interval": (0.0, 10.0),
}


def env_num(name: str, default, cast=int):
    """资源上限 env 读取; 非法/非正值回落默认 (正常使用碰不到的宽松值)。"""
    try:
        v = cast(os.environ.get(name, default))
    except ValueError:
        return default
    return v if v > 0 and math.isfinite(v) else default


def num_bounds() -> dict:
    """当前生效的数值区间 (speed 每次读 env, 便于测试与线上调参)。"""
    return dict(_FIXED_BOUNDS, speed_factor=(
        env_num("ADR_TTS_SPEED_MIN", 0.1, float),
        env_num("ADR_TTS_SPEED_MAX", 10.0, float)))


def bad_number(values: dict):
    """返回第一个越界/非有限值参数的错误文案, 全部合法返回 None。"""
    for key, (lo, hi) in num_bounds().items():
        v = values.get(key)
        if v is None:
            continue
        try:
            ok = lo <= float(v) <= hi   # NaN 比较恒 False → 拒绝
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return f"{key} out of range: {v!r}, must be within [{lo}, {hi}]"
    return None


def normalize_params(values: dict):
    """按字段归一: 类型 → 区间/有限性 → 语言小写 + 白名单。

    入参只需包含显式传入的键 (None 一并传入也会被忽略);
    返回 (归一结果, 错误文案|None)。归一结果只含合法键,
    调用方在通过后 update 回请求 dict 即可。
    """
    out = {k: v for k, v in values.items() if v is not None}
    for k in _INT_FIELDS:
        if k in out:
            try:
                out[k] = int(out[k])
            except (TypeError, ValueError):
                return out, f"{k} must be an integer: {out[k]!r}"
    for k in _FLOAT_FIELDS:
        if k in out:
            try:
                out[k] = float(out[k])
            except (TypeError, ValueError):
                lo, hi = num_bounds()[k]
                return (out, f"{k} out of range: {out[k]!r}, "
                             f"must be within [{lo}, {hi}]")
    msg = bad_number(out)
    if msg:
        return out, msg
    for k in _LANG_FIELDS:
        if k in out:
            out[k] = str(out[k]).lower()
            if out[k] not in SUPPORTED_LANGS:
                return (out, f"{k} unsupported: {out[k]!r}, "
                             f"must be one of {sorted(SUPPORTED_LANGS)}")
    return out, None
