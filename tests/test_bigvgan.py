"""BigVGAN 单元测试 (M2 B 步骤验收)。

依赖:
- F:/ADR_data/bigvgan/ 完整源码 + 权重

跳过条件:
- BigVGAN 路径不存在 (用 skip)
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


# BigVGAN 路径 (可在 conftest 或环境变量覆盖)
BIGVGAN_PATH = Path(r"F:/ADR_data/bigvgan")


def make_synthetic_wav(
    path: str, duration_sec: float = 2.0, sample_rate: int = 22050,
) -> str:
    """生成合成 wav。"""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 220 * t)
    audio += 0.05 * np.random.randn(len(t)).astype(np.float32)
    audio = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return path


@pytest.fixture(scope="module")
def bigvgan_vocoder():
    """加载 BigVGAN (skip if not found)。"""
    if not BIGVGAN_PATH.exists():
        pytest.skip(f"BigVGAN not found at {BIGVGAN_PATH}")
    if not (BIGVGAN_PATH / "bigvgan_generator.pt").exists():
        pytest.skip(f"BigVGAN weights not found at {BIGVGAN_PATH}")

    from adr.vocoder.bigvgan import BigVGANVocoder
    vocoder = BigVGANVocoder.from_pretrained(BIGVGAN_PATH, device="cpu")
    return vocoder


def test_bigvgan_vocoder_loads(bigvgan_vocoder):
    """测试 BigVGAN 真实加载。"""
    assert bigvgan_vocoder.is_loaded(), "BigVGAN not loaded"
    assert bigvgan_vocoder._model is not None
    assert bigvgan_vocoder._native_sr == 22050
    assert bigvgan_vocoder.config.n_mels == 80


def test_bigvgan_mel_to_wav(bigvgan_vocoder):
    """测试 mel → wav (真实音频,不是常数)。"""
    import torch

    # 64 帧 mel (对应约 0.74s 音频, 64 * 256 hop = 16384 / 22050)
    mel = torch.randn(1, 80, 64)
    wav = bigvgan_vocoder.infer(mel)

    # wav 形状: (B, T_wav) = (1, 16384)
    assert wav.ndim == 2
    assert wav.shape[0] == 1
    assert wav.shape[1] > 1000, f"wav too short: {wav.shape[1]}"

    # wav 范围: [-1, 1]
    assert wav.abs().max() <= 1.0, f"wav out of range: {wav.abs().max()}"

    # wav 非平凡 (std > 0.05)
    assert wav.std() > 0.05, f"wav std too low: {wav.std()}"

    # FFT 有多个非平凡频谱分量
    wav_np = wav.squeeze().numpy()
    fft_mag = np.abs(np.fft.rfft(wav_np[: min(8000, len(wav_np))]))
    n_bins = (fft_mag > fft_mag.max() * 0.01).sum()
    assert n_bins > 10, f"too few frequency components: {n_bins}"


def test_bigvgan_in_pipeline():
    """测试 BigVGAN 集成到 InferPipeline。"""
    if not BIGVGAN_PATH.exists():
        pytest.skip(f"BigVGAN not found at {BIGVGAN_PATH}")

    import tempfile

    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS, SoVITSConfig
    from adr.vocoder.bigvgan import BigVGANVocoder

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_path = f.name
    try:
        make_synthetic_wav(ref_path, duration_sec=2.0, sample_rate=22050)

        vocoder = BigVGANVocoder.from_pretrained(BIGVGAN_PATH, device="cpu")
        cfg = SoVITSConfig(
            hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
            vocab_size=607, content_dim=16, timbre_dim=16,
        )
        model = SoVITS(cfg)
        pipe = InferPipeline(
            model, config=InferConfig(n_timesteps=3), vocoder=vocoder,
        )

        wav = pipe.synthesize("你好", ref_path, n_timesteps=3)

        # BigVGAN 输出: wav.shape[0] 应该是 ref mel 帧数 × 256 (4×4×2×2×2×2)
        # 而非 placeholder 的少数几个样本
        assert wav.shape[0] > 100, f"wav too short, vocoder not used: {wav.shape}"
        assert wav.std() > 0.05, f"wav std too low: {wav.std()}"
    finally:
        Path(ref_path).unlink(missing_ok=True)


def test_bigvgan_placeholder_fallback():
    """测试无 vocoder 时 placeholder 仍工作。"""
    import tempfile

    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS, SoVITSConfig

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_path = f.name
    try:
        make_synthetic_wav(ref_path, duration_sec=2.0, sample_rate=22050)
        cfg = SoVITSConfig(
            hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
            vocab_size=607, content_dim=16, timbre_dim=16,
        )
        model = SoVITS(cfg)
        # use_placeholder_vocoder=True 强制走 placeholder
        pipe = InferPipeline(
            model, config=InferConfig(n_timesteps=3, use_placeholder_vocoder=True),
        )
        wav = pipe.synthesize("你好", ref_path, n_timesteps=3)
        # placeholder 输出短 (一帧 mel ≈ 80 sample)
        assert wav.shape[0] < 100, f"placeholder should be short: {wav.shape}"
        # placeholder 是归一化线性 mel 第一个 channel,值在 [-1, 1]
        # 不期望真实音频特征 (FFT 多分量等)
        assert wav.shape[0] < 5 or np.abs(wav).max() <= 1.0, \
            f"placeholder wav out of range: max={np.abs(wav).max()}"
    finally:
        Path(ref_path).unlink(missing_ok=True)


def test_bigvgan_vocoder_registry():
    """测试 BigVGAN 在 VOCODER registry 中可查找。"""
    from adr.core import REGISTRY

    vocoders = REGISTRY.vocoder.keys()
    assert "bigvgan" in vocoders, f"BigVGAN not registered: {vocoders}"


def test_bigvgan_resample():
    """测试 BigVGAN 自动 resample (target_sr != native_sr)。"""
    if not BIGVGAN_PATH.exists():
        pytest.skip(f"BigVGAN not found at {BIGVGAN_PATH}")

    import torch

    from adr.vocoder.bigvgan import BigVGANVocoder

    vocoder = BigVGANVocoder.from_pretrained(
        BIGVGAN_PATH, device="cpu", target_sr=16000,  # 强制 16kHz
    )
    mel = torch.randn(1, 80, 64)
    wav = vocoder.infer(mel)

    # 期望: 16kHz × (64 帧 × 256 hop × 256 upsample) / 22050 ≈ 一样
    # resample 后 wav 长度按 16/22.05 比例缩短
    assert vocoder.config.sample_rate == 16000
    assert wav.shape[1] > 1000  # 仍然有内容
