"""批次41a: HTTP 面错误码表 + 档案名非法异常。

与 adr/core/exceptions.py (域异常码表, ``error_code()`` 单点) 分工:
- 本模块服务 console / HTTPException 面: 错误响应在既有 ``detail`` 字段
  之外**增量补** ``code`` 字段 (wire 兼容: 既有字段与文本逐字保留, 只增不改);
- ProfileNameInvalid: console 档案名非法 (含路径穿越成分) 的专用 400,
  code 复用 core 码表既有稳定码 "profile_invalid"。
"""
from __future__ import annotations

from fastapi import HTTPException


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
