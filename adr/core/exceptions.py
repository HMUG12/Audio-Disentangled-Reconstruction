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
