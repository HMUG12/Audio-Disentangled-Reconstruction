"""ADR 框架自定义异常。"""


class ADRException(Exception):
    """ADR 框架基类异常。"""


class ConfigError(ADRException):
    """配置错误。"""


class ModelNotFoundError(ADRException):
    """模型未找到。"""


class VocabError(ADRException):
    """词表/音素错误。"""


class DataPipelineError(ADRException):
    """数据流水线错误。"""


class TrainingError(ADRException):
    """训练错误。"""


class InferenceError(ADRException):
    """推理错误。"""


class DeviceError(ADRException):
    """设备错误。"""


class DependencyMissingError(ADRException):
    """依赖缺失。"""


class SynthesisError(ADRException):
    """合成服务层错误基类 (批次34)。"""


class ProfileInvalidError(SynthesisError):
    """音色档案名非法 (含路径穿越)。"""


class ProfileNotFoundError(SynthesisError):
    """音色档案不存在。"""


class SynthesisParamsError(SynthesisError):
    """合成请求参数缺失/非法。"""
