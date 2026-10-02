"""音频 I/O 工具 (基于 soundfile + torchaudio + librosa)。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio


PathLike = Union[str, Path]


@dataclass
class AudioData:
    """音频数据封装。"""
    waveform: np.ndarray    # (T,) 或 (C, T)
    sample_rate: int
    path: Optional[str] = None

    @property
    def duration(self) -> float:
        """时长 (秒)。"""
        if self.waveform.ndim == 1:
            return len(self.waveform) / self.sample_rate
        return self.waveform.shape[-1] / self.sample_rate

    @property
    def num_channels(self) -> int:
        return 1 if self.waveform.ndim == 1 else self.waveform.shape[0]

    @property
    def num_samples(self) -> int:
        return self.waveform.shape[-1]

    def to_mono(self) -> "AudioData":
        """转为单声道。"""
        if self.waveform.ndim == 1:
            return self
        mono = self.waveform.mean(axis=0)
        return AudioData(mono, self.sample_rate, self.path)

    def to_tensor(self) -> torch.Tensor:
        """转 torch tensor。"""
        if self.waveform.ndim == 1:
            return torch.from_numpy(self.waveform).float().unsqueeze(0)
        return torch.from_numpy(self.waveform).float()

    def __repr__(self) -> str:
        return (
            f"AudioData(duration={self.duration:.2f}s, "
            f"sr={self.sample_rate}, channels={self.num_channels}, "
            f"samples={self.num_samples})"
        )


def load_audio(
    path: PathLike,
    sample_rate: Optional[int] = None,
    mono: bool = True,
) -> AudioData:
    """加载音频文件。

    Args:
        path: 音频文件路径
        sample_rate: 目标采样率 (None 保持原始)
        mono: 是否转为单声道
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    # 用 soundfile (支持 wav/flac/ogg) - 优先
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        # soundfile 返回 (T,) 或 (T, C)
        if data.ndim == 1:
            waveform = data
        else:
            waveform = data.T  # (C, T)
    except Exception:
        # fallback scipy
        try:
            from scipy.io import wavfile

            sr, data = wavfile.read(path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype != np.float32:
                data = data.astype(np.float32)
            if data.ndim == 1:
                waveform = data
            else:
                waveform = data.T
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load audio with soundfile and scipy: {e2}"
            ) from e2

    # 转单声道
    if mono and waveform.ndim > 1:
        waveform = waveform.mean(axis=0)

    # Resample
    if sample_rate is not None and sr != sample_rate:
        waveform = resample_audio(waveform, sr, sample_rate)
        sr = sample_rate

    return AudioData(waveform=waveform.astype(np.float32), sample_rate=sr, path=path)


def save_audio(
    audio: AudioData,
    path: PathLike,
    subtype: str = "PCM_16",
) -> None:
    """保存音频到文件。

    Args:
        audio: AudioData 实例
        path: 输出路径
        subtype: soundfile 子类型 (PCM_16/FLOAT/VORBIS/...)
    """
    import soundfile as sf

    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    waveform = audio.waveform
    # soundfile 期望 (T,) 或 (T, C)
    if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
        # (C, T) -> (T, C)
        waveform = waveform.T

    sf.write(path, waveform, audio.sample_rate, subtype=subtype)


def resample_audio(
    waveform: np.ndarray,
    src_sr: int,
    tgt_sr: int,
) -> np.ndarray:
    """重采样音频。

    Args:
        waveform: (T,) 或 (C, T)
        src_sr: 源采样率
        tgt_sr: 目标采样率
    """
    if src_sr == tgt_sr:
        return waveform

    # 用 torchaudio (高质量)
    if waveform.ndim == 1:
        tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    else:
        tensor = torch.from_numpy(waveform).float()

    resampler = torchaudio.transforms.Resample(src_sr, tgt_sr)
    resampled = resampler(tensor).numpy()

    if waveform.ndim == 1:
        resampled = resampled[0]

    return resampled


def normalize_audio(
    waveform: np.ndarray,
    target_db: float = -23.0,
    peak: float = 0.95,
) -> np.ndarray:
    """归一化音频到目标响度。

    Args:
        waveform: (T,) 或 (C, T)
        target_db: 目标 dBFS (默认 -23 LUFS 近似)
        peak: 峰值上限
    """
    # 简单 RMS 归一化
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < 1e-8:
        return waveform

    target_rms = 10 ** (target_db / 20)
    waveform = waveform * (target_rms / rms)

    # 防止削波
    max_val = np.abs(waveform).max()
    if max_val > peak:
        waveform = waveform * (peak / max_val)

    return waveform


def compute_mel(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """计算 mel 频谱。

    Args:
        waveform: (T,) 单声道
        sample_rate: 采样率
        ... mel 参数
        device: 'cuda' / 'cpu' (None 自动)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if fmax is None:
        fmax = sample_rate / 2.0

    # PyTorch 实现 (GPU 加速)
    if waveform.ndim == 1:
        tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    else:
        tensor = torch.from_numpy(waveform).float()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=1.0,
        normalized=False,
    ).to(device)

    tensor = tensor.to(device)
    mel = mel_transform(tensor)  # (B, n_mels, T)
    mel = mel.cpu().numpy()[0]  # (n_mels, T)

    # log mel
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

    return mel


def compute_mel_bigvgan(
    waveform: np.ndarray,
    sample_rate: int = 22050,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: float = 0.0,
    fmax: float = 8000.0,
    device: Optional[str] = None,
) -> np.ndarray:
    """与 BigVGAN 训练严格一致的 log-mel (v2)。

    与 compute_mel (v1) 的差异 (v1 与 BigVGAN 不匹配 → 声码器全域失真,
    GT mel 直送也乱语, M9 探针实测):
    - fmax=8000 (BigVGAN 22k config; v1 用 sr/2=11025)
    - librosa slaney-norm 滤波器组 (v1 torchaudio 无 norm)
    - center=False + reflect pad (n_fft-hop)//2 (v1 center=True)
    - log(clamp(x, 1e-5)) 动态压缩 (与 BigVGAN dynamic_range_compression 相同)

    训练数据 (npz) 与推理必须用本函数, 否则训练/声码器域错位。
    """
    import librosa

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    y = torch.from_numpy(waveform).float()
    if y.ndim == 1:
        y = y.unsqueeze(0)          # (1, T)
    y = y.to(device)

    mel_basis = torch.from_numpy(librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
        fmin=fmin, fmax=fmax,
    )).float().to(device)
    hann = torch.hann_window(win_length).to(device)

    padding = (n_fft - hop_length) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding),
                                mode="reflect").squeeze(1)
    spec = torch.stft(
        y, n_fft, hop_length=hop_length, win_length=win_length,
        window=hann, center=False, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel = torch.matmul(mel_basis, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel[0].cpu().numpy()      # (n_mels, T)


def get_audio_info(path: PathLike) -> dict:
    """获取音频文件元信息。"""
    import soundfile as sf

    info = sf.info(str(path))
    return {
        "path": str(path),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration": info.duration,
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
    }


def slice_audio_by_time(
    waveform: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """按时间切片。"""
    start = int(start_sec * sample_rate)
    end = int(end_sec * sample_rate)
    return waveform[..., start:end]


def pad_or_trim(
    waveform: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """填充或裁剪到目标长度。"""
    cur_length = waveform.shape[-1]
    if cur_length == target_length:
        return waveform
    if cur_length > target_length:
        return waveform[..., :target_length]

    pad_width = target_length - cur_length
    pad_shape = list(waveform.shape)
    pad_shape[-1] = pad_width
    pad = np.full(pad_shape, pad_value, dtype=waveform.dtype)
    return np.concatenate([waveform, pad], axis=-1)
