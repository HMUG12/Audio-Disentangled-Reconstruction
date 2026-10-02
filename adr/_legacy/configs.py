"""核心配置 — ADR 架构全局参数。

与设计文档 [docs/architecture-design.md](../../docs/architecture-design.md) 对应。
所有模块维度、训练超参都从此处读取，便于 M1/M2 统一调整。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContentConfig:
    """音素/歌词内容编码器配置。"""
    vocab_size: int = 512          # 音素数（DiffSinger 字典: 中/英 + SP/AP + 静音）
    embed_dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 1024
    max_phoneme_len: int = 1024
    dropout: float = 0.1


@dataclass
class MelodyConfig:
    """旋律编码器配置（含 null melody token 设计）。"""
    n_pitch_bins: int = 128        # MIDI 0–127 + 扩展
    n_velocity_bins: int = 32
    embed_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    use_null_token: bool = True    # 说话模式用可学习 null token 替代
    null_token_id: int = 0         # 内部保留 token id


@dataclass
class TimbreConfig:
    """音色编码器配置。"""
    ref_audio_sr: int = 24000
    ref_mel_bins: int = 80
    ref_mel_frames: int = 256      # 约 3-5 秒参考音频
    embed_dim: int = 256
    encoder: str = "wavlm"         # 复用预训练，wavlm | mel-cnn | codec-pool


@dataclass
class BackboneConfig:
    """共享 DiT + CFM backbone 配置。

    设计参照 UniVoice (0.3B)：
    - 28 层 DiT
    - 1024 hidden
    - 条件流匹配目标
    """
    hidden_dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    ffn_dim: int = 3072
    mel_bins: int = 80             # 声学特征维度
    max_mel_frames: int = 1500
    dropout: float = 0.1
    use_task_modulation: bool = True
    # 条件流匹配 (CFM) 超参
    cfm_sigma_min: float = 1e-5
    cfm_n_timesteps: int = 20      # 推理步数


@dataclass
class ADRConfig:
    content: ContentConfig = field(default_factory=ContentConfig)
    melody: MelodyConfig = field(default_factory=MelodyConfig)
    timbre: TimbreConfig = field(default_factory=TimbreConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    # 任务位：speech / sing 二选一
    task: Optional[str] = None

    def total_params_estimate(self) -> str:
        """估算参数量（粗略）。"""
        c = self.content
        m = self.melody
        b = self.backbone
        t = self.timbre
        # 简化估算：embed + transformer
        # melody ffn 默认 4×embed_dim
        m_ffn = 4 * m.embed_dim
        params_m = (
            c.vocab_size * c.embed_dim / 1e6
            + c.n_layers * (4 * c.embed_dim * c.ffn_dim) / 1e6
            + m.embed_dim * (m.n_pitch_bins + m.n_velocity_bins) / 1e6
            + m.n_layers * (4 * m.embed_dim * m_ffn) / 1e6
            + b.n_layers * (12 * b.hidden_dim * b.ffn_dim) / 1e6
            + t.embed_dim * 1000 / 1e6
        )
        return f"~{params_m:.0f}M params"
