"""F0 音高提取。

支持后端:
- pyin: librosa 内置,纯 Python (慢,无依赖)
- rmvpe: 高精度,M2 集成
- torchcrepe: 深度学习,M2 集成
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from adr.core import get_logger


@dataclass
class F0Config:
    """F0 配置。"""
    backend: str = "pyin"          # pyin / rmvpe / torchcrepe
    fmin: float = 50.0             # 最低频率 (Hz)
    fmax: float = 1100.0           # 最高频率 (Hz)
    hop_length: int = 256          # 帧移 (samples)
    sr: int = 24000                # 采样率


class F0Extractor:
    """F0 提取器 (懒加载)。"""

    def __init__(self, config: Optional[F0Config] = None):
        self.config = config or F0Config()
        self.log = get_logger("adr.data.f0")

    def __call__(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """提取 F0 序列。

        Args:
            audio: 音频路径 / 波形数组
            sample_rate: 波形采样率

        Returns:
            F0 序列 (T,), 0 表示无声帧
        """
        if self.config.backend == "pyin":
            return self._pyin_extract(audio, sample_rate)
        else:
            self.log.warning(
                f"F0 backend '{self.config.backend}' not in M1, "
                f"falling back to pyin"
            )
            return self._pyin_extract(audio, sample_rate)

    def _pyin_extract(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """使用 librosa.pyin 提取 F0。"""
        try:
            import librosa
        except ImportError as e:
            raise ImportError("librosa not installed. Run: pip install librosa") from e

        if isinstance(audio, (str, Path)):
            y, sr = librosa.load(str(audio), sr=None)
        else:
            y = audio
            sr = sample_rate or self.config.sr
            if y.ndim > 1:
                y = y.mean(axis=0)

        # pyin
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            sr=sr,
            hop_length=self.config.hop_length,
        )

        # NaN -> 0
        f0 = np.nan_to_num(f0, nan=0.0)

        return f0.astype(np.float32)

    @staticmethod
    def f0_to_midi(f0: np.ndarray) -> np.ndarray:
        """F0 -> MIDI (log 频率, 0 表示无声)。"""
        midi = np.zeros_like(f0)
        mask = f0 > 0
        midi[mask] = 12 * np.log2(f0[mask] / 440.0) + 69
        return midi

    @staticmethod
    def f0_statistics(f0: np.ndarray) -> dict:
        """F0 统计信息。"""
        voiced = f0[f0 > 0]
        if len(voiced) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "voiced_ratio": 0}

        return {
            "mean": float(voiced.mean()),
            "std": float(voiced.std()),
            "min": float(voiced.min()),
            "max": float(voiced.max()),
            "voiced_ratio": float(len(voiced) / len(f0)),
        }
