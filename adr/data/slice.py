"""音频切片 (基于 VAD + 时长约束)。

参考 GPT-SoVITS tools/slice_audio.py 设计:
- 静音检测 (RMS / WebRTC VAD)
- 切片长度 3-10 秒 (RVC/GPT-SoVITS 标准)
- 切点避开静音
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from adr.core import get_logger
from adr.utils.audio import AudioData, load_audio, save_audio


@dataclass
class SliceConfig:
    """切片配置。"""
    min_sec: float = 3.0
    max_sec: float = 10.0
    min_silence_sec: float = 0.3   # 静音最小长度
    silence_threshold_db: float = -40.0  # 静音阈值 (dB)
    merge_short: bool = True        # 短切片是否合并
    target_sr: int = 24000


@dataclass
class AudioSlice:
    """单个音频切片。"""
    index: int
    start_sec: float
    end_sec: float
    waveform: np.ndarray
    sample_rate: int
    text: Optional[str] = None      # ASR 后填充
    phonemes: Optional[List[str]] = None  # G2P 后填充
    f0: Optional[np.ndarray] = None  # F0 后填充

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def to_audio_data(self) -> AudioData:
        return AudioData(self.waveform, self.sample_rate)


def detect_silence_segments(
    waveform: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_silence_sec: float = 0.3,
    frame_length_ms: int = 30,
) -> List[tuple[float, float]]:
    """检测静音段 (返回 [(start_sec, end_sec), ...])。

    Args:
        waveform: (T,) 单声道
        sample_rate: 采样率
        threshold_db: 低于此 dB 视为静音
        min_silence_sec: 短于此长度的"静音"忽略
        frame_length_ms: 帧长 (ms)
    """
    frame_length = int(sample_rate * frame_length_ms / 1000)
    n_frames = len(waveform) // frame_length

    if n_frames == 0:
        return []

    # 计算每帧的 dB
    frames = np.array_split(waveform[: n_frames * frame_length], n_frames)
    rms_values = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    db_values = 20 * np.log10(np.clip(rms_values, a_min=1e-8, a_max=None))

    # 标记静音帧
    is_silent = db_values < threshold_db

    # 找连续静音段
    segments = []
    in_silence = False
    silence_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not silent and in_silence:
            silence_end = i
            silence_dur = (silence_end - silence_start) * frame_length_ms / 1000
            if silence_dur >= min_silence_sec:
                start_sec = silence_start * frame_length_ms / 1000
                end_sec = silence_end * frame_length_ms / 1000
                segments.append((start_sec, end_sec))
            in_silence = False

    # 结尾的静音
    if in_silence:
        silence_end = n_frames
        silence_dur = (silence_end - silence_start) * frame_length_ms / 1000
        if silence_dur >= min_silence_sec:
            start_sec = silence_start * frame_length_ms / 1000
            end_sec = silence_end * frame_length_ms / 1000
            segments.append((start_sec, end_sec))

    return segments


def slice_audio(
    audio: AudioData,
    config: Optional[SliceConfig] = None,
    progress: bool = True,
) -> List[AudioSlice]:
    """将长音频切分为短片段。

    Args:
        audio: 音频数据
        config: 切片配置
        progress: 是否打印进度
    """
    log = get_logger("adr.data.slice")
    config = config or SliceConfig()

    waveform = audio.waveform
    sr = audio.sample_rate
    duration = audio.duration

    if progress:
        log.info(f"Slicing audio: duration={duration:.2f}s, sr={sr}, "
                 f"target=[{config.min_sec}, {config.max_sec}]s")

    # 短音频直接整段返回
    if duration <= config.max_sec and duration >= config.min_sec:
        return [AudioSlice(
            index=0,
            start_sec=0.0,
            end_sec=duration,
            waveform=waveform,
            sample_rate=sr,
        )]

    # 检测静音段
    silence_segments = detect_silence_segments(
        waveform, sr,
        threshold_db=config.silence_threshold_db,
        min_silence_sec=config.min_silence_sec,
    )

    if progress:
        log.info(f"  Found {len(silence_segments)} silence segments")

    # 切点候选 = 静音中点
    cut_points = [(s + e) / 2 for s, e in silence_segments]

    # 如果没检测到静音,均匀切
    if not cut_points:
        if progress:
            log.warning("  No silence detected, using uniform slicing")
        n_slices = max(1, int(duration / config.max_sec))
        slice_dur = duration / n_slices
        cut_points = [i * slice_dur for i in range(1, n_slices)]

    # 在切点附近生成 [min_sec, max_sec] 范围的切片
    slices = []
    slice_idx = 0
    cur_start = 0.0

    for cut in cut_points + [duration]:
        segment_dur = cut - cur_start
        if segment_dur < config.min_sec:
            continue
        elif segment_dur <= config.max_sec:
            # 整段作为一个切片
            start_sample = int(cur_start * sr)
            end_sample = int(cut * sr)
            slices.append(AudioSlice(
                index=slice_idx,
                start_sec=cur_start,
                end_sec=cut,
                waveform=waveform[start_sample:end_sample],
                sample_rate=sr,
            ))
            slice_idx += 1
            cur_start = cut
        else:
            # 超过 max_sec,强制切分
            n_sub = int(np.ceil(segment_dur / config.max_sec))
            sub_dur = segment_dur / n_sub
            for i in range(n_sub):
                sub_start = cur_start + i * sub_dur
                sub_end = sub_start + sub_dur
                start_sample = int(sub_start * sr)
                end_sample = int(sub_end * sr)
                if end_sample - start_sample < sr * config.min_sec:
                    continue
                slices.append(AudioSlice(
                    index=slice_idx,
                    start_sec=sub_start,
                    end_sec=sub_end,
                    waveform=waveform[start_sample:end_sample],
                    sample_rate=sr,
                ))
                slice_idx += 1
            cur_start = cut

    if progress:
        log.info(f"  Generated {len(slices)} slices")
        if slices:
            avg_dur = np.mean([s.duration for s in slices])
            log.info(f"  Avg duration: {avg_dur:.2f}s")

    return slices


def save_slices(
    slices: List[AudioSlice],
    output_dir: Union[str, Path],
    prefix: str = "slice",
) -> List[Path]:
    """保存切片到目录,返回文件路径列表。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for s in slices:
        path = output_dir / f"{prefix}_{s.index:04d}.wav"
        save_audio(s.to_audio_data(), path)
        paths.append(path)

    return paths
