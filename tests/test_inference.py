"""测试推理流水线 (M1 Day 9)。

不需要真实 ref 音频,使用合成 wav。
"""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest


def make_synthetic_wav(
    path: str,
    duration_sec: float = 3.0,
    sample_rate: int = 22050,
) -> str:
    """生成合成 wav (用于推理测试)。"""
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


def test_infer_config_defaults():
    """测试默认配置。"""
    from adr.inference.pipeline import InferConfig

    cfg = InferConfig()
    assert cfg.g2p_backend == "pypinyin"
    assert cfg.g2p_with_tone is True
    assert cfg.ref_sample_rate == 22050
    assert cfg.n_mels == 80
    assert cfg.n_timesteps == 20


def test_pipeline_construction():
    """测试 pipeline 构造。"""
    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS, SoVITSConfig

    cfg = SoVITSConfig(
        hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,  # DiffSinger 607
    )
    model = SoVITS(cfg)
    pipe = InferPipeline(model, config=InferConfig(n_timesteps=3))

    assert pipe.backbone is model
    assert pipe.g2p is not None
    assert pipe.vocoder is None


def test_text_to_phoneme_ids_chinese():
    """测试中文转音素 id (用真实 PhonemeDict)。"""
    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS, SoVITSConfig

    cfg = SoVITSConfig(
        hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    )
    model = SoVITS(cfg)
    pipe = InferPipeline(model, config=InferConfig())

    ids = pipe.text_to_phoneme_ids("你好")
    assert len(ids) == 2
    # 真实音素 id (zhong=582, guo=187, ni=?, hao=?), 应该都在 [0, 606]
    assert all(0 <= i < 607 for i in ids)


def test_text_to_phoneme_ids_english_fallback():
    """测试英文 fallback (pypinyin 失败时用 char)。"""
    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS, SoVITSConfig

    cfg = SoVITSConfig(
        hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    )
    model = SoVITS(cfg)
    pipe = InferPipeline(model, config=InferConfig())

    ids = pipe.text_to_phoneme_ids("hello world")
    # 11 chars (含空格),但 space 会被过滤,实际应该是 10
    assert len(ids) >= 2
    assert all(0 <= i < 607 for i in ids)


def test_text_to_phoneme_ids_empty():
    """测试空文本报错。"""
    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS, SoVITSConfig

    cfg = SoVITSConfig(
        hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    )
    model = SoVITS(cfg)
    pipe = InferPipeline(model, config=InferConfig())

    with pytest.raises(ValueError, match="Empty text"):
        pipe.text_to_phoneme_ids("")


def test_ref_audio_to_mel_shape():
    """测试 ref 音频 → mel。"""
    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS, SoVITSConfig

    cfg = SoVITSConfig(
        hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    )
    model = SoVITS(cfg)
    pipe = InferPipeline(model, config=InferConfig(n_mels=80, ref_sample_rate=22050))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        make_synthetic_wav(tmp, duration_sec=2.0, sample_rate=22050)
        mel = pipe.ref_audio_to_mel(tmp)
        assert mel.shape[0] == 80  # n_mels
        assert mel.shape[1] > 0    # T_frames
        assert mel.dtype == np.float32
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_synthesize_end_to_end():
    """端到端: ref + text → wav。"""
    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS, SoVITSConfig

    cfg = SoVITSConfig(
        hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    )
    model = SoVITS(cfg)
    pipe = InferPipeline(model, config=InferConfig(n_timesteps=3))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_path = f.name
    try:
        make_synthetic_wav(ref_path, duration_sec=3.0, sample_rate=22050)
        wav = pipe.synthesize("测试一下", ref_path, n_timesteps=3)
        assert wav.ndim == 1
        assert wav.shape[0] > 0
        assert wav.dtype == np.float32
        assert np.abs(wav).max() <= 1.0
    finally:
        Path(ref_path).unlink(missing_ok=True)


def test_save_wav_roundtrip():
    """测试 wav 保存+读回。"""
    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS, SoVITSConfig

    cfg = SoVITSConfig(
        hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    )
    model = SoVITS(cfg)
    pipe = InferPipeline(model, config=InferConfig(n_timesteps=3, ref_sample_rate=22050))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    try:
        make_synthetic_wav(ref_path, duration_sec=3.0, sample_rate=22050)
        wav = pipe.synthesize("你好", ref_path, n_timesteps=3)
        pipe.save_wav(wav, out_path)

        # 读回
        import soundfile as sf
        wav_back, sr_back = sf.read(out_path)
        assert sr_back == 22050
        assert wav_back.shape[0] > 0
    finally:
        Path(ref_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def test_from_checkpoint_nonexistent():
    """测试 checkpoint 不存在报错。"""
    from adr.inference.pipeline import InferPipeline

    with pytest.raises(FileNotFoundError):
        InferPipeline.from_checkpoint("/nonexistent/path/ckpt.pt")


# ---------------------------------------------------------------------------
# 批次41b: from_checkpoint 权重错配校验
# (unexpected keys / 形状不符 → raise; missing keys → warning + strict=False)
# ---------------------------------------------------------------------------


def _make_ckpt(tmp_path, mutate=None) -> str:
    """构造合法 SoVITS checkpoint; mutate(state_dict) 可注入错配。"""
    import dataclasses

    import torch

    from adr.models.sovits import SoVITS, SoVITSConfig

    cfg = SoVITSConfig(
        hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    )
    model = SoVITS(cfg)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    if mutate is not None:
        mutate(sd)
    path = tmp_path / "ckpt.pt"
    torch.save({
        "model_state": sd,
        "model_class": "SoVITS",
        "backbone_config": dataclasses.asdict(cfg),
    }, path)
    return str(path)


def test_from_checkpoint_roundtrip(tmp_path):
    """正常 checkpoint → 加载成功 (批次41b 基线)。"""
    from adr.inference.pipeline import InferConfig, InferPipeline

    ckpt = _make_ckpt(tmp_path)
    pipe = InferPipeline.from_checkpoint(
        ckpt, config=InferConfig(n_timesteps=3, device="cpu")
    )
    assert pipe.backbone is not None


def test_from_checkpoint_unexpected_keys_raise(tmp_path):
    """checkpoint 含模型不存在的权重键 → RuntimeError, 不再静默丢弃 (批次41b)。"""
    from adr.inference.pipeline import InferPipeline

    def add_bogus(sd):
        import torch
        sd["ghost_layer.weight"] = torch.zeros(4, 4)

    ckpt = _make_ckpt(tmp_path, mutate=add_bogus)
    with pytest.raises(RuntimeError, match="不存在的权重"):
        InferPipeline.from_checkpoint(ckpt)


def test_from_checkpoint_shape_mismatch_raise(tmp_path):
    """权重形状与模型不符 → RuntimeError, 不再静默 strict=False (批次41b)。"""
    from adr.inference.pipeline import InferPipeline

    def wrong_shape(sd):
        import torch
        k = next(iter(sd))
        sd[k] = torch.zeros(7, 3)

    ckpt = _make_ckpt(tmp_path, mutate=wrong_shape)
    with pytest.raises(RuntimeError, match="形状"):
        InferPipeline.from_checkpoint(ckpt)


def test_from_checkpoint_missing_keys_warns(tmp_path, caplog):
    """checkpoint 缺权重键 → warning + strict=False 加载 (LoRA/部分微调场景)。"""
    import logging

    from adr.inference.pipeline import InferPipeline

    def drop_one(sd):
        del sd[sorted(sd.keys())[-1]]

    ckpt = _make_ckpt(tmp_path, mutate=drop_one)
    with caplog.at_level(logging.WARNING, logger="adr.inference"):
        pipe = InferPipeline.from_checkpoint(ckpt)
    assert pipe.backbone is not None
    assert any("缺少" in r.message for r in caplog.records)
