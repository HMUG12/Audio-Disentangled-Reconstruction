"""ASR 自动标注 (Faster-Whisper 封装)。

参考 GPT-SoVITS tools/asr/fasterwhisper_asr.py 设计。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from adr.core import get_logger


@dataclass
class ASRConfig:
    """ASR 配置。"""
    model_size: str = "small"        # tiny/base/small/medium/large-v3
    device: str = "auto"             # auto/cpu/cuda
    compute_type: str = "auto"       # auto/int8/float16/float32
    language: Optional[str] = None   # None=自动检测, "zh"/"en"...
    beam_size: int = 5
    vad_filter: bool = True          # 启用 VAD
    min_silence_duration_ms: int = 500


@dataclass
class ASRResult:
    """ASR 单条结果。"""
    text: str
    language: str
    confidence: float = 0.0
    segments: List[dict] = None

    def __repr__(self) -> str:
        return f"ASRResult(text='{self.text[:30]}...', lang={self.language}, conf={self.confidence:.2f})"


class ASR:
    """Faster-Whisper 封装 (懒加载)。"""

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self._model = None
        self.log = get_logger("adr.data.asr")

    def _ensure_model(self):
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            ) from e

        device = self.config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = self.config.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self.log.info(
            f"Loading Faster-Whisper: model={self.config.model_size}, "
            f"device={device}, compute_type={compute_type}"
        )

        # 设置模型缓存目录
        cache_dir = os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
            "~/.cache/huggingface/hub"
        )
        os.makedirs(cache_dir, exist_ok=True)

        self._model = WhisperModel(
            self.config.model_size,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
        self.log.info("  ASR model loaded")

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> ASRResult:
        """转录音频。

        Args:
            audio: 音频路径 / 波形数组
            sample_rate: 波形采样率 (仅当传数组时需要)
        """
        self._ensure_model()

        if isinstance(audio, (str, Path)):
            segments, info = self._model.transcribe(
                str(audio),
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                vad_parameters={"min_silence_duration_ms": self.config.min_silence_duration_ms}
                if self.config.vad_filter else None,
            )
        else:
            # numpy array
            import tempfile
            import soundfile as sf

            sr = sample_rate or 16000  # whisper 默认 16k
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                sf.write(tmp_path, audio, sr)
            try:
                segments, info = self._model.transcribe(
                    tmp_path,
                    language=self.config.language,
                    beam_size=self.config.beam_size,
                    vad_filter=self.config.vad_filter,
                )
            finally:
                os.unlink(tmp_path)

        # 收集结果
        seg_list = []
        full_text = []
        total_conf = 0.0
        n_segs = 0

        for seg in segments:
            seg_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_logprob": seg.avg_logprob,
            })
            full_text.append(seg.text.strip())
            # 粗略置信度 (avg_logprob 范围 ~ [-1, 0])
            conf = max(0.0, min(1.0, 1.0 + seg.avg_logprob))
            total_conf += conf
            n_segs += 1

        text = " ".join(full_text).strip()
        avg_conf = total_conf / max(n_segs, 1)

        return ASRResult(
            text=text,
            language=info.language if hasattr(info, "language") else "auto",
            confidence=avg_conf,
            segments=seg_list,
        )

    def transcribe_slices(
        self,
        slices: List,
        progress: bool = True,
    ) -> List[ASRResult]:
        """批量转录切片。"""
        results = []
        for i, s in enumerate(slices):
            try:
                result = self.transcribe(s.waveform, sample_rate=s.sample_rate)
                s.text = result.text
                results.append(result)
                if progress and (i + 1) % 10 == 0:
                    self.log.info(f"  ASR progress: {i+1}/{len(slices)}")
            except Exception as e:
                self.log.error(f"  ASR failed on slice {s.index}: {e}")
                results.append(ASRResult(text="", language="unknown", confidence=0.0))
        return results
