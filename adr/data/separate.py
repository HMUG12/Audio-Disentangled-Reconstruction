"""人声分离 (UVR5 集成, M1 提供简单 VAD 替代)。

参考 GPT-SoVITS tools/uvr5/webui.py 设计。
M1 阶段: 提供轻量 VAD-based fallback,不强制依赖 UVR5。
M2+: 集成 UVR5 (audio-separator 库)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from adr.core import get_logger
from adr.utils.audio import AudioData, load_audio, save_audio


@dataclass
class SeparateConfig:
    """分离配置。"""
    method: str = "vad"             # vad / uvr5
    uvr5_model: str = "MDX-Net"     # MDX-Net / Demucs / VR Arch
    output_vocals: bool = True      # 输出人声
    output_instrumental: bool = True  # 输出伴奏


class Separator:
    """人声分离器。"""

    def __init__(self, config: Optional[SeparateConfig] = None):
        self.config = config or SeparateConfig()
        self.log = get_logger("adr.data.separator")
        self._model = None

    def _ensure_model(self):
        if self.config.method == "vad":
            return  # VAD 不需要模型
        if self._model is not None:
            return

        try:
            from audio_separator.separator import Separator as AudioSeparator
        except ImportError:
            self.log.warning(
                "audio-separator not installed. "
                "M1 will use VAD fallback. "
                "Install: pip install audio-separator[gpu]"
            )
            self.config.method = "vad"
            return

        self._model = AudioSeparator(
            model_file_dir=Path.home() / ".cache/audio-separator",
            output_dir=Path("/tmp/adr_separated"),
        )

    def __call__(
        self,
        audio: Union[str, Path, AudioData],
        output_dir: Optional[Path] = None,
    ) -> tuple[AudioData, Optional[AudioData]]:
        """分离人声与伴奏。

        Returns:
            (vocals, instrumental) - instrumental 可能为 None
        """
        self._ensure_model()

        if isinstance(audio, (str, Path)):
            audio_data = load_audio(audio)
        else:
            audio_data = audio

        if self.config.method == "vad":
            return self._vad_separate(audio_data)
        else:
            return self._uvr5_separate(audio_data, output_dir)

    def _vad_separate(
        self,
        audio: AudioData,
    ) -> tuple[AudioData, None]:
        """基于 VAD 的简单分离 (M1 fallback)。

        直接返回原音频,instrumental = None。
        因为纯 VAD 无法真正分离人声和伴奏,只是为流水线提供统一接口。
        """
        self.log.info(
            "Using VAD fallback (no actual separation). "
            "Install audio-separator for real vocal/instrumental split."
        )
        return audio, None

    def _uvr5_separate(
        self,
        audio: AudioData,
        output_dir: Optional[Path],
    ) -> tuple[AudioData, AudioData]:
        """UVR5 真实人声分离。"""
        import tempfile
        import os

        if output_dir is None:
            output_dir = Path("/tmp/adr_separated")
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = Path(f.name)
            save_audio(audio, tmp_path)

        try:
            self.log.info(f"Running UVR5 separation on {tmp_path}")
            output_files = self._model.separate(tmp_path)
            # output_files: [vocals_path, instrumental_path]
            vocals = load_audio(output_files[0], sample_rate=audio.sample_rate)
            instr = load_audio(output_files[1], sample_rate=audio.sample_rate)
            return vocals, instr
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)
