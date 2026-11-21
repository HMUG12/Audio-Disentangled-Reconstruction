"""批次41a: HTTP 面错误码表 + 档案名非法异常。

与 adr/core/exceptions.py (域异常码表, ``error_code()`` 单点) 分工:
- 本模块服务 console / HTTPException 面: 错误响应在既有 ``detail`` 字段
  之外**增量补** ``code`` 字段 (wire 兼容: 既有字段与文本逐字保留, 只增不改);
- ProfileNameInvalid: console 档案名非法 (含路径穿越成分) 的专用 400,
  code 复用 core 码表既有稳定码 "profile_invalid"。
"""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from adr.core.exceptions import error_code


# ── wire 错误码表 (M3-a) ──────────────────────────────────────────────────
# 五套 HTTP/WS 面 (v2 / v3 / OpenAI / native / console) 的错误外壳只做字段
# 形态转换, 码值一律引用本表, 协议文件不再写字面量。
# 码值是 wire 契约 (调用方按字符串匹配), 不得改名, 只增不改。
INVALID_PARAMS = "invalid_params"
# 媒体类型不在白名单: 历史 wire 值就是 invalid_params (调用方已按该值匹配),
# 语义键单独登记、值复用, 不引入新的 wire 值。
UNSUPPORTED_MEDIA = INVALID_PARAMS
PROFILE_NOT_FOUND = "profile_not_found"
PAYLOAD_TOO_LARGE = "payload_too_large"
BUSY = "busy"
# 以下为各面已在用的增量码, 一并收表 (同样只增不改)
CUDA_OOM = "cuda_oom"
NLTK_MISSING = "nltk_missing"
INVALID_NAME = "invalid_name"
ROLLBACK_FAILED = "rollback_failed"
EXPORT_FAILED = "export_failed"
IMPORT_FAILED = "import_failed"
NOT_INITIALIZED = "not_initialized"
UNKNOWN_CMD = "unknown_cmd"
SESSION_TIMEOUT = "session_timeout"
VALIDATION_ERROR = "validation_error"
UNAUTHORIZED = "unauthorized"
# F7: 预热未完成 (与 engine_stalled 区分, 客户端应稍后重试而非判定卡死)
ENGINE_LOADING = "engine_loading"

# 门禁测试断言: 协议文件里出现的 code 字面量必须在本集合内
WIRE_CODES = frozenset({
    INVALID_PARAMS, PROFILE_NOT_FOUND, PAYLOAD_TOO_LARGE, BUSY,
    CUDA_OOM, NLTK_MISSING, INVALID_NAME, ROLLBACK_FAILED, EXPORT_FAILED,
    IMPORT_FAILED, NOT_INITIALIZED, UNKNOWN_CMD, SESSION_TIMEOUT,
    VALIDATION_ERROR, UNAUTHORIZED, ENGINE_LOADING,
})


def error_body(message: str, code: str) -> dict:
    """协议无关的错误核心 body: 各面在此之外只加自己的增量字段。"""
    return {"message": message, "code": code}


def json_error(status_code: int, message: str, code: str, *,
               extra: dict | None = None, headers=None) -> JSONResponse:
    """HTTP 错误外壳 (v2 / native / 中间件共用): ``{"message", "code"}``。"""
    body = error_body(message, code)
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status_code, content=body, headers=headers)


def ws_error(message: str, code: str) -> dict:
    """v3 WebSocket error 帧外壳: ``{"type": "error", "message", "code"}``。"""
    return {"type": "error", "message": message, "code": code}


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
    # 批次86 (CON-01): 引擎卡死 / 锁等待超时 → 503, 客户端可重试
    "engine_stalled",
    # F7: 预热未完成 → 503 (Retry-After), 不是故障
    "engine_loading",
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
