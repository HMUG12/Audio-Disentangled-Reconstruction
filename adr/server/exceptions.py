"""批次41a: HTTP 面错误码表 + 档案名非法异常。

与 adr/core/exceptions.py (域异常码表, ``error_code()`` 单点) 分工:
- 本模块服务 console / HTTPException 面: 错误响应在既有 ``detail`` 字段
  之外**增量补** ``code`` 字段 (wire 兼容: 既有字段与文本逐字保留, 只增不改);
- ProfileNameInvalid: console 档案名非法 (含路径穿越成分) 的专用 400,
  code 复用 core 码表既有稳定码 "profile_invalid"。
"""
from __future__ import annotations

from fastapi import HTTPException

from adr.core.exceptions import error_code


class ProfileNameInvalid(HTTPException):
    """档案名非法 (含路径穿越/盘符成分), HTTP 400, code=profile_invalid。"""

    def __init__(self, name: str):
        super().__init__(400, f"档案名含非法字符或路径成分: {name!r}")
        self.code = "profile_invalid"


# status_code → 稳定错误码 (console 面增量字段; 域异常码表见 core/exceptions.py)
_HTTP_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    500: "internal_error",
    503: "service_unavailable",
}


def http_error_code(exc: HTTPException) -> str:
    """HTTPException → 稳定错误码: 异常自带 ``code`` 属性优先, 否则按状态码
    映射, 未知状态码回落 "http_error"。"""
    custom = getattr(exc, "code", None)
    if isinstance(custom, str):
        return custom
    return _HTTP_STATUS_CODES.get(exc.status_code, "http_error")


# 批次84 (#12): 合成兜底 except 的状态码分类 — 此前所有内部异常统一 400,
# 客户端会把"服务端坏了"当参数错误反复重试, 放大负载并掩盖真因。
#   503 = 服务暂时不可用 (依赖缺失/设备/模型/配置), 客户端可稍后重试;
#   400 = 请求数据本身有问题 (引擎深处抛的 ValueError/TypeError,
#         如"参考音频超出 3~10 秒"), 保留既有 400 语义;
#   500 = 其余未预期内部故障。
_UNAVAILABLE_CODES = {
    "config_error", "model_not_found", "device_error", "dependency_missing",
}


def error_status_code(exc: BaseException, default: int = 500) -> int:
    """域异常 → HTTP 状态码 (批次84 / CODE_REVIEW #12)。

    映射依据 ``core.exceptions.error_code()``, 与错误码单源保持一致;
    非框架异常按类型判定 (ValueError/TypeError 视为请求数据错误 → 400)。
    """
    if error_code(exc) in _UNAVAILABLE_CODES:
        return 503
    if isinstance(exc, (ValueError, TypeError)):
        return 400
    return default
