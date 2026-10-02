"""ADR 数据流水线模块。

完整流水线 (参考 GPT-SoVITS):
    原始音频 → [可选:人声分离] → 切片 → ASR → G2P → F0 → 训练数据

主要组件:
    - separate: 人声/伴奏分离 (UVR5)
    - slice:    音频切片 (VAD + 时长约束)
    - asr:      Faster-Whisper 自动标注
    - g2p:      文本转音素 (g2pW / pypinyin / char)
    - f0:       音高提取 (RMVPE / pyin)
    - pipeline: 编排整个流水线
"""

from adr.data.pipeline import DataPipeline, PipelineResult

__all__ = [
    "DataPipeline",
    "PipelineResult",
]
