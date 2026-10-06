"""数据流水线编排器 (Day 2 核心)。

完整流程:
    原始 wav
        ↓ [可选] UVR5 人声分离
    纯人声 wav
        ↓ 静音检测切片 (3-10 秒)
    多个短切片
        ↓ Faster-Whisper ASR
    带文本标注的切片
        ↓ G2P 文本→音素
    带音素标注的切片
        ↓ F0 提取
    训练样本 (waveform + text + phonemes + f0 + mel)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from adr.core import get_logger
from adr.data.asr import ASR, ASRConfig
from adr.data.f0 import F0Config, F0Extractor
from adr.data.g2p import G2P, G2PConfig
from adr.data.separate import SeparateConfig, Separator
from adr.data.slice import SliceConfig, slice_audio
from adr.utils.audio import AudioData, compute_mel, load_audio


@dataclass
class PipelineConfig:
    """流水线总配置。"""
    # 各阶段开关
    enable_separation: bool = True
    enable_slicing: bool = True
    enable_asr: bool = True
    enable_g2p: bool = True
    enable_f0: bool = True
    enable_mel: bool = True

    # 子配置
    separate: SeparateConfig = field(default_factory=SeparateConfig)
    slice: SliceConfig = field(default_factory=SliceConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    g2p: G2PConfig = field(default_factory=G2PConfig)
    f0: F0Config = field(default_factory=F0Config)

    # 输出
    output_dir: str = "./output/processed"
    save_format: str = "npz"  # npz / wav


@dataclass
class TrainSample:
    """单个训练样本。"""
    sample_id: str
    waveform: np.ndarray
    sample_rate: int
    text: str
    phonemes: List[str]
    f0: np.ndarray
    mel: Optional[np.ndarray] = None
    start_sec: float = 0.0
    end_sec: float = 0.0

    def save(self, path: Union[str, Path]) -> None:
        """保存到 npz。"""
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            sample_id=self.sample_id,
            waveform=self.waveform,
            sample_rate=self.sample_rate,
            text=np.array([self.text], dtype=object),
            phonemes=np.array([self.phonemes], dtype=object),
            f0=self.f0,
            mel=self.mel if self.mel is not None else np.array([]),
            start_sec=self.start_sec,
            end_sec=self.end_sec,
        )

    @staticmethod
    def load(path: Union[str, Path]) -> "TrainSample":
        """从 npz 加载。"""
        data = np.load(path, allow_pickle=True)
        return TrainSample(
            sample_id=str(data["sample_id"]),
            waveform=data["waveform"],
            sample_rate=int(data["sample_rate"]),
            text=str(data["text"][0]) if data["text"].size > 0 else "",
            phonemes=list(data["phonemes"][0]) if data["phonemes"].size > 0 else [],
            f0=data["f0"],
            mel=data["mel"] if data["mel"].size > 0 else None,
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
        )


@dataclass
class PipelineResult:
    """流水线结果。"""
    samples: List[TrainSample] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def summary(self) -> str:
        return (
            f"PipelineResult: {len(self.samples)} samples, "
            f"{self.skipped} skipped, {len(self.errors)} errors"
        )


class DataPipeline:
    """数据流水线编排器。"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.log = get_logger("adr.data.pipeline")

        # 懒加载各组件
        self._separator = None
        self._asr = None
        self._g2p = None
        self._f0 = None

    def _ensure_components(self):
        if self.config.enable_separation and self._separator is None:
            self._separator = Separator(self.config.separate)
        if self.config.enable_asr and self._asr is None:
            self._asr = ASR(self.config.asr)
        if self.config.enable_g2p and self._g2p is None:
            self._g2p = G2P(self.config.g2p)
        if self.config.enable_f0 and self._f0 is None:
            self._f0 = F0Extractor(self.config.f0)

    def run(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PipelineResult:
        """运行完整流水线。

        Args:
            input_path: 输入音频路径
            output_dir: 输出目录

        Returns:
            PipelineResult
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_components()

        result = PipelineResult()
        result.metadata["input"] = str(input_path)
        result.metadata["config"] = asdict(self.config)

        # Step 1: Load audio
        self.log.info(f"[1/6] Loading audio: {input_path}")
        audio = load_audio(input_path, sample_rate=self.config.slice.target_sr)
        result.metadata["duration"] = audio.duration
        result.metadata["sample_rate"] = audio.sample_rate
        self.log.info(f"  Duration: {audio.duration:.2f}s, SR: {audio.sample_rate}")

        # Step 2: Separation
        if self.config.enable_separation and self._separator:
            self.log.info("[2/6] Separating vocals (may be VAD fallback in M1)")
            vocals, _ = self._separator(audio)
            audio = vocals

        # Step 3: Slicing
        if not self.config.enable_slicing:
            self.log.info("[3/6] Slicing skipped")
            slices = []
        else:
            self.log.info("[3/6] Slicing audio into 3-10s segments")
            from adr.data.slice import AudioSlice
            slices = slice_audio(audio, self.config.slice)
            self.log.info(f"  Got {len(slices)} slices")
            if not slices:
                self.log.warning("  No slices generated, audio too short or all silence")
                return result

        # Step 4: ASR
        if self.config.enable_asr and self._asr and slices:
            self.log.info(f"[4/6] ASR on {len(slices)} slices (model={self.config.asr.model_size})")
            for i, s in enumerate(slices):
                try:
                    asr_result = self._asr.transcribe(s.waveform, sample_rate=s.sample_rate)
                    s.text = asr_result.text
                    if not s.text.strip():
                        result.skipped += 1
                        self.log.warning(f"  Slice {s.index}: empty transcription, skipping")
                except Exception as e:
                    result.errors.append(f"ASR slice {s.index}: {e}")
                    s.text = ""
            slices = [s for s in slices if s.text and s.text.strip()]
            self.log.info(f"  {len(slices)} slices with valid transcription")

        # Step 5: G2P
        if self.config.enable_g2p and self._g2p and slices:
            self.log.info(f"[5/6] G2P on {len(slices)} texts")
            for s in slices:
                try:
                    s.phonemes = self._g2p(s.text)
                except Exception as e:
                    result.errors.append(f"G2P slice {s.index}: {e}")
                    s.phonemes = []

        # Step 6: F0 + Mel
        # 批次41b: 单样本提取失败只跳过该样本 (记入 errors), 不让整条数据集构建中断
        f0_failed: set = set()
        if self.config.enable_f0 and self._f0 and slices:
            self.log.info(f"[6/6] F0 + Mel extraction on {len(slices)} slices")
            for s in slices:
                try:
                    s.f0 = self._f0(s.waveform, sample_rate=s.sample_rate)
                    if self.config.enable_mel:
                        s.mel = compute_mel(
                            s.waveform, s.sample_rate,
                            n_mels=80, hop_length=self.config.f0.hop_length,
                        )
                except Exception as e:
                    result.errors.append(f"F0/Mel slice {s.index}: {e}")
                    self.log.warning(
                        f"  F0/Mel 提取失败, 跳过切片 {s.index}: {e}")
                    f0_failed.add(s.index)

        # Build TrainSample list
        # 批次41b: sample_id 加入源文件路径哈希 — 不同目录同名文件
        # (如多说话人各自的 voice.wav) 不再互相覆盖 npz
        path_hash = hashlib.md5(
            str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
        for s in slices:
            if not s.text or not s.phonemes:
                continue
            if s.index in f0_failed:
                # F0 失败的切片缺关键训练信号, 跳过而非带空 f0 入库
                continue
            sample = TrainSample(
                sample_id=f"{input_path.stem}_{path_hash}_{s.index:04d}",
                waveform=s.waveform,
                sample_rate=s.sample_rate,
                text=s.text,
                phonemes=s.phonemes,
                f0=s.f0 if s.f0 is not None else np.array([]),
                # AudioSlice 无 mel 字段, 仅 enable_mel=True 时才有动态属性
                mel=getattr(s, "mel", None),
                start_sec=s.start_sec,
                end_sec=s.end_sec,
            )
            result.samples.append(sample)

        # Save samples
        if self.config.save_format == "npz" and result.samples:
            samples_dir = output_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for sample in result.samples:
                sample.save(samples_dir / f"{sample.sample_id}.npz")

            # Save metadata
            meta_path = output_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "n_samples": len(result.samples),
                    "input": str(input_path),
                    "duration": result.metadata.get("duration"),
                    "skipped": result.skipped,
                    "errors": result.errors[:10],  # 最多保存 10 条
                }, f, ensure_ascii=False, indent=2)
            self.log.info(f"  Saved {len(result.samples)} samples to {samples_dir}")

        # Final summary
        self.log.info("=" * 60)
        self.log.info(f"Pipeline finished: {result.summary()}")
        return result
