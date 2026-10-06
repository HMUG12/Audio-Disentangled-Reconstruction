"""ADR 框架自定义异常 + 错误协议 (批次35)。

错误协议: 每个 ADR 异常携带稳定错误码 ``code`` (wire 字段, 不得改名);
四面 (v2/native HTTP, v3 WS, webui) 各自按 wire 形态序列化, 唯一映射点是
``error_code()``。code 为增量字段, 既有消息 (``message`` 等) 逐字保留。
"""


class ADRException(Exception):
    """ADR 框架基类异常。"""

    code = "adr_error"


class ConfigError(ADRException):
    """配置错误。"""

    code = "config_error"


class ModelNotFoundError(ADRException):
    """模型未找到。"""

    code = "model_not_found"


class VocabError(ADRException):
    """词表/音素错误。"""

    code = "vocab_error"


class DataPipelineError(ADRException):
    """数据流水线错误。"""

    code = "data_pipeline_error"


class TrainingError(ADRException):
    """训练错误。"""

    code = "training_error"


class InferenceError(ADRException):
    """推理错误。"""

    code = "inference_error"


class DeviceError(ADRException):
    """设备错误。"""

    code = "device_error"


class DependencyMissingError(ADRException):
    """依赖缺失。"""

    code = "dependency_missing"


class SynthesisError(ADRException):
    """合成服务层错误基类 (批次34)。"""

    code = "synthesis_error"


class ProfileInvalidError(SynthesisError):
    """音色档案名非法 (含路径穿越)。"""

    code = "profile_invalid"


class ProfileNotFoundError(SynthesisError):
    """音色档案不存在。"""

    code = "profile_not_found"


class SynthesisParamsError(SynthesisError):
    """合成请求参数缺失/非法。"""

    code = "invalid_params"


def error_code(e: BaseException) -> str:
    """异常 → 稳定错误码 (批次35 四面统一映射的单点)。

    ADR 框架异常返回其 ``code`` 类属性; 非框架异常一律 "internal_error"。
    """
    return getattr(e, "code", "internal_error")
