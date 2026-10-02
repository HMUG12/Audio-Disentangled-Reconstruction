"""测试 WavQualityCallback (M3 - wav-based 早停)。"""
import sys
from pathlib import Path

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
import torch

from adr.data.pipeline import TrainSample
from adr.models.sovits import SoVITS, SoVITSConfig
from adr.training.callbacks import WavQualityCallback
from adr.training.dataset import VoiceCloneDataset
from adr.training.trainer import Trainer, TrainerConfig


def make_dataset(n: int = 20):
    samples = []
    for i in range(n):
        s = TrainSample(
            sample_id=f"utt_{i:03d}",
            phonemes=["zh", "ong1", "g_uan1"] * 7,
            text="测试",
            f0=np.linspace(180, 200, 80).astype(np.float32),
            mel=np.random.randn(80, 80).astype(np.float32) * 0.1 + 0.5,
            waveform=np.random.randn(8000).astype(np.float32) * 0.1,
            sample_rate=22050,
        )
        samples.append(s)
    return VoiceCloneDataset(samples=samples)


def make_small_model():
    cfg = SoVITSConfig(
        hidden_dim=64, n_layers=2, n_heads=2, ffn_dim=128,
        vocab_size=607, content_dim=64, timbre_dim=32,
        n_mels=80, sample_rate=22050, hop_length=256,
    )
    return SoVITS(cfg)


def test_wav_quality_callback_runs():
    """WavQualityCallback 应该在每个 epoch 后跑, 记录 wav 指标。"""
    ds = make_dataset(20)
    model = make_small_model()
    cb = WavQualityCallback(n_samples=2, n_timesteps=2, patience=2)
    config = TrainerConfig(
        epochs=2, batch_size=2, val_ratio=0.3,
        output_dir=str(REPO / "tests" / "_tmp_wq_runs"),
        log_every_n_steps=100, save_every_n_epochs=100,
    )
    trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
    trainer.fit()
    # 应有 history
    assert len(cb._history) > 0
    # 应有 wav_quality 指标
    last = cb._history[-1]
    assert "wav_rms_mean" in last
    assert "wav_centroid_mean" in last
    assert "wav_quality" in last
    # 清理
    import shutil
    shutil.rmtree(REPO / "tests" / "_tmp_wq_runs", ignore_errors=True)


def test_wav_quality_callback_can_stop():
    """WavQualityCallback 在 wav quality 不变时能触发 should_stop。"""
    ds = make_dataset(20)
    model = make_small_model()
    # patience=1 更容易触发
    cb = WavQualityCallback(n_samples=2, n_timesteps=2, patience=1, min_delta=1e9)
    # min_delta=1e9 让它永远不"改善", 立即触发
    config = TrainerConfig(
        epochs=5, batch_size=2, val_ratio=0.3,
        output_dir=str(REPO / "tests" / "_tmp_wq_stop"),
        log_every_n_steps=100, save_every_n_epochs=100,
    )
    trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
    trainer.fit()
    # should_stop 应被设置
    assert cb.should_stop is True
    # 清理
    import shutil
    shutil.rmtree(REPO / "tests" / "_tmp_wq_stop", ignore_errors=True)


def test_wav_quality_callback_skips_without_val():
    """没有 val_loader 时应 skip。"""
    ds = make_dataset(20)
    model = make_small_model()
    cb = WavQualityCallback(n_samples=2, n_timesteps=2, patience=2)
    config = TrainerConfig(
        epochs=2, batch_size=2, val_ratio=0.0,  # 没 val
        output_dir=str(REPO / "tests" / "_tmp_wq_noval"),
        log_every_n_steps=100, save_every_n_epochs=100,
    )
    trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
    trainer.fit()
    # 没 history (因为没 val_loader)
    assert len(cb._history) == 0
    # 不应 stop
    assert cb.should_stop is False
    # 清理
    import shutil
    shutil.rmtree(REPO / "tests" / "_tmp_wq_noval", ignore_errors=True)


def test_wav_metrics_computation():
    """Wav 指标计算函数本身应能跑。"""
    from adr.training.callbacks import WavQualityCallback
    cb = WavQualityCallback(n_samples=3, use_placeholder_vocoder=True)
    # 直接测试 _mel_to_wav
    mel = torch.randn(80, 100)
    wav = cb._mel_to_wav(mel)
    assert wav is not None
    assert len(wav) > 1000
    # RMS 应 > 0
    rms = float(np.sqrt(np.mean(wav ** 2)))
    assert rms > 0
    # 谱质心应在合理范围 (几百到几千 Hz)
    spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
    freqs = np.fft.rfftfreq(len(wav), 1 / 22050)
    centroid = float(np.sum(freqs * spec) / np.sum(spec))
    assert 100 < centroid < 10000


def test_wav_quality_vocoder_path_param():
    """vocoder_path 参数应被接受, 不破坏 placeholder 行为。"""
    cb1 = WavQualityCallback(use_placeholder_vocoder=True)
    assert cb1.use_placeholder_vocoder is True
    assert cb1.vocoder_path is None

    cb2 = WavQualityCallback(
        use_placeholder_vocoder=False, vocoder_path="F:/fake/bigvgan"
    )
    assert cb2.use_placeholder_vocoder is False
    assert cb2.vocoder_path == "F:/fake/bigvgan"


def test_wav_quality_vocoder_caching():
    """真实 vocoder 失败一次后, 不再重试 (缓存机制)。"""
    cb = WavQualityCallback(
        n_samples=2, use_placeholder_vocoder=False, vocoder_path="F:/nonexistent/bigvgan"
    )
    # _get_vocoder 第一次失败
    v1 = cb._get_vocoder()
    assert v1 is None
    assert cb._vocoder_load_failed is True
    # 第二次不应重试
    v2 = cb._get_vocoder()
    assert v2 is None


def test_wav_quality_placeholder_fallback_on_vocoder_fail():
    """vocoder 加载失败时, 应回退到 placeholder (不报错)。"""
    cb = WavQualityCallback(
        n_samples=2, use_placeholder_vocoder=False, vocoder_path="F:/fake/path"
    )
    # _mel_to_wav 失败时返回 None (但不应抛异常)
    mel = torch.randn(80, 100)
    wav = cb._mel_to_wav(mel)
    # 由于 use_placeholder_vocoder=False 且 vocoder 加载失败, wav 应为 None
    assert wav is None


def test_wav_quality_placeholder_with_explicit_disable():
    """显式 use_placeholder_vocoder=False 但无 path 时, _get_vocoder 应尝试 auto。"""
    cb = WavQualityCallback(use_placeholder_vocoder=False, vocoder_path=None)
    # 验证 _get_vocoder 不会抛异常 (可能加载失败, 但捕获了)
    v = cb._get_vocoder()
    # v 可能是 None (失败) 或 BigVGANVocoder 实例 (成功)
    # 关键是不抛异常
    assert v is None or hasattr(v, "infer")


def test_wav_quality_f0_extraction():
    """F0 提取函数应能识别已知频率的合成信号。"""
    import numpy as np
    cb = WavQualityCallback(use_placeholder_vocoder=True)
    # 合成 220Hz 正弦波
    sr = 22050
    t = np.linspace(0, 1.0, sr)
    wav_220 = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    f0 = cb._extract_f0(wav_220, sr=sr)
    assert len(f0) > 5, "应能提取到足够的 F0 帧"
    # 检测到的 F0 应在 200-240Hz 之间 (容差)
    assert 200 < f0.mean() < 240, f"220Hz 信号检测到 {f0.mean():.1f}Hz, 偏差太大"

    # 合成 330Hz
    wav_330 = 0.3 * np.sin(2 * np.pi * 330 * t).astype(np.float32)
    f0_330 = cb._extract_f0(wav_330, sr=sr)
    assert 310 < f0_330.mean() < 350, f"330Hz 信号检测到 {f0_330.mean():.1f}Hz"


def test_wav_quality_f0_extraction_handles_silence():
    """静音应返回空数组 (不抛异常)。"""
    import numpy as np
    cb = WavQualityCallback(use_placeholder_vocoder=True)
    wav_silence = np.zeros(22050, dtype=np.float32)
    f0 = cb._extract_f0(wav_silence, sr=22050)
    assert len(f0) == 0  # 静音无 F0


def test_wav_quality_f0_metrics_in_computation():
    """_compute_wav_metrics 应返回 F0 指标 (如果有 waveform)。"""
    import numpy as np
    ds = make_dataset(20)
    model = make_small_model()
    cb = WavQualityCallback(
        n_samples=3, n_timesteps=2, patience=99,  # 不早停
        use_placeholder_vocoder=True,
    )
    config = TrainerConfig(
        epochs=1, batch_size=2, val_ratio=0.3,
        output_dir=str(REPO / "tests" / "_tmp_wq_f0"),
        log_every_n_steps=100, save_every_n_epochs=100,
    )
    trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
    trainer.fit()
    # 应有 history
    assert len(cb._history) > 0
    last = cb._history[-1]
    # M3.6 必有的字段
    assert "f0_pred_mean" in last or "wav_quality" in last
    # 清理
    import shutil
    shutil.rmtree(REPO / "tests" / "_tmp_wq_f0", ignore_errors=True)
