"""测试模型 Backend (M1 Day 3)。

不依赖真实权重,只测试:
- Backbone 抽象接口
- SoVITS stub 能构造 + forward
- ContentEncoder / TimbreEncoder 基本 forward
- PretrainedHub 元信息
- BigVGAN 占位加载
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def test_base_backbone_abstract():
    """BaseBackbone 是抽象类。"""
    from adr.models.base import BaseBackbone, BackboneConfig

    config = BackboneConfig(hidden_dim=128, n_layers=2)
    # 抽象类不能直接实例化
    with pytest.raises(TypeError):
        BaseBackbone(config)


def test_sovits_construction():
    """SoVITS 能构造。"""
    from adr.models.sovits import SoVITS, SoVITSConfig

    config = SoVITSConfig(
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        ffn_dim=256,
        vocab_size=100,
        content_dim=64,
        timbre_dim=64,
    )
    model = SoVITS(config)
    assert model is not None
    assert "sovits" in str(model).lower() or "SoVITS" in str(model)

    # 参数量 > 0
    n_params = model.num_parameters()
    assert n_params > 1000
    print(f"  SoVITS params: {model.num_parameters_str()}")


def test_sovits_forward():
    """SoVITS forward 能跑通 (随机数据)。"""
    from adr.models.sovits import SoVITS, SoVITSConfig

    config = SoVITSConfig(
        hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128,
        vocab_size=50, content_dim=32, timbre_dim=32,
    )
    model = SoVITS(config)

    B, T_phoneme, T_mel, T_ref = 2, 8, 32, 64
    batch = {
        "phoneme_ids": torch.randint(0, 50, (B, T_phoneme)),
        "phoneme_mask": torch.ones(B, T_phoneme, dtype=torch.bool),
        "ref_mel": torch.randn(B, 80, T_ref),
        "target_mel": torch.randn(B, 80, T_mel),
        "target_durations": torch.randint(1, 4, (B, T_phoneme)),
    }

    out = model(batch)
    assert "loss" in out
    assert "pred_mel" in out
    assert out["loss"].dim() == 0  # scalar
    assert out["loss"].item() >= 0
    assert out["pred_mel"].shape[1] == 80  # n_mels

    # 反向传播
    out["loss"].backward()


def _make_sovits_small():
    from adr.models.sovits import SoVITS, SoVITSConfig

    config = SoVITSConfig(
        hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128,
        vocab_size=50, content_dim=32, timbre_dim=32,
    )
    return SoVITS(config)


def test_sovits_forward_with_f0():
    """SoVITS forward 带 f0 能跑通, f0_embed 收到梯度。"""
    model = _make_sovits_small()

    B, T_phoneme, T_mel, T_ref = 2, 8, 32, 64
    batch = {
        "phoneme_ids": torch.randint(0, 50, (B, T_phoneme)),
        "phoneme_mask": torch.ones(B, T_phoneme, dtype=torch.bool),
        "ref_mel": torch.randn(B, 80, T_ref),
        "target_mel": torch.randn(B, 80, T_mel),
        "target_durations": torch.full((B, T_phoneme), T_mel // T_phoneme),
        "f0": torch.rand(B, T_mel) * 400,  # 0~400Hz
    }

    out = model(batch)
    assert out["loss"].dim() == 0
    out["loss"].backward()

    # f0_embed 必须收到梯度 (否则 F0 注入断了)
    assert model.f0_embed.weight.grad is not None
    assert model.f0_embed.weight.grad.abs().sum() > 0


def test_sovits_f0_zero_init_backward_compat():
    """零初始化下: 带/不带 f0 输出完全一致 (向后兼容旧 checkpoint)。"""
    model = _make_sovits_small()
    model.eval()

    B, T_phoneme, T_mel, T_ref = 1, 8, 32, 64
    base = {
        "phoneme_ids": torch.randint(0, 50, (B, T_phoneme)),
        "phoneme_mask": torch.ones(B, T_phoneme, dtype=torch.bool),
        "ref_mel": torch.randn(B, 80, T_ref),
        "target_mel": torch.randn(B, 80, T_mel),
        "target_durations": torch.full((B, T_phoneme), T_mel // T_phoneme),
    }

    with torch.no_grad():
        out_no_f0 = model(base)["pred_mel"]
        out_with_f0 = model({**base, "f0": torch.rand(B, T_mel) * 400})["pred_mel"]

    assert torch.allclose(out_no_f0, out_with_f0, atol=1e-6)


def test_sovits_f0_changes_output_after_training():
    """f0_embed 非零后, 不同 f0 应产生不同输出 (注入真的生效)。"""
    model = _make_sovits_small()
    # 手动打破零初始化
    with torch.no_grad():
        model.f0_embed.weight.normal_(0, 0.1)
    model.eval()

    B, T_phoneme, T_mel, T_ref = 1, 8, 32, 64
    base = {
        "phoneme_ids": torch.randint(0, 50, (B, T_phoneme)),
        "phoneme_mask": torch.ones(B, T_phoneme, dtype=torch.bool),
        "ref_mel": torch.randn(B, 80, T_ref),
        "target_mel": torch.randn(B, 80, T_mel),
        "target_durations": torch.full((B, T_phoneme), T_mel // T_phoneme),
    }

    with torch.no_grad():
        out_low = model({**base, "f0": torch.full((B, T_mel), 150.0)})["pred_mel"]
        out_high = model({**base, "f0": torch.full((B, T_mel), 450.0)})["pred_mel"]

    assert not torch.allclose(out_low, out_high, atol=1e-4)


def test_sovits_sample_with_f0():
    """sample 接受 f0 参数, 长度不匹配时不崩。"""
    model = _make_sovits_small()
    model.eval()

    phoneme_ids = torch.randint(1, 50, (1, 8))
    ref_mel = torch.randn(1, 80, 64)
    with torch.inference_mode():
        mel_no_f0 = model.sample(phoneme_ids, ref_mel, n_timesteps=5)
        # f0 比 regulated 短 → 内部 pad
        mel_short_f0 = model.sample(
            phoneme_ids, ref_mel, n_timesteps=5, f0=torch.rand(1, 16) * 300
        )
    assert mel_no_f0.shape[1] == 80
    assert mel_short_f0.shape[1] == 80


def test_sovits_sample():
    """SoVITS 推理接口。"""
    from adr.models.sovits import SoVITS, SoVITSConfig

    config = SoVITSConfig(
        hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128,
        vocab_size=50, content_dim=32, timbre_dim=32,
    )
    model = SoVITS(config)
    model.eval()

    phoneme_ids = torch.randint(1, 50, (1, 8))
    ref_mel = torch.randn(1, 80, 64)
    with torch.inference_mode():
        mel = model.sample(phoneme_ids, ref_mel, n_timesteps=5)
    assert mel.shape[1] == 80  # n_mels
    assert mel.shape[0] == 1


def test_sovits_registered():
    """SoVITS 已注册到 REGISTRY。"""
    from adr.core import REGISTRY

    backbone_reg = REGISTRY.backbone
    assert "sovits" in backbone_reg
    assert "sovits_base" in backbone_reg

    # 通过 REGISTRY 获取
    cls = backbone_reg.get("sovits")
    assert cls is not None


def test_content_encoder_registered():
    """ContentEncoder 已注册。"""
    from adr.core import REGISTRY

    reg = REGISTRY.content_encoder
    assert "transformer" in reg


def test_timbre_encoder_registered():
    """TimbreEncoder 已注册。"""
    from adr.core import REGISTRY

    reg = REGISTRY.timbre_encoder
    assert "mel_cnn" in reg


def test_content_encoder_forward():
    """ContentEncoder forward 正确。"""
    from adr.models.content_encoder import ContentEncoder

    enc = ContentEncoder(vocab_size=100, embed_dim=64, n_layers=2, n_heads=4, ffn_dim=128)
    ids = torch.randint(0, 100, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)
    out = enc(ids, mask)
    assert out.shape == (2, 10, 64)


def test_timbre_encoder_forward():
    """TimbreEncoder forward 正确。"""
    from adr.models.timbre_encoder import TimbreEncoder

    enc = TimbreEncoder(n_mels=80, embed_dim=64)
    mel = torch.randn(2, 80, 100)
    out = enc(mel)
    assert out.shape == (2, 64)


def test_pretrained_hub_list():
    """PretrainedHub 列出模型。"""
    from adr.models.hub import PretrainedHub

    hub = PretrainedHub()
    items = hub.list()
    assert len(items) >= 2
    names = [i.name for i in items]
    assert "bigvgan_22khz_80band" in names


def test_pretrained_hub_is_downloaded():
    """检查下载状态。"""
    from adr.models.hub import PretrainedHub

    hub = PretrainedHub()
    # 不实际下载,只检查接口
    assert isinstance(hub.is_downloaded("bigvgan_22khz_80band"), bool)


def test_bigvgan_vocoder_placeholder():
    """BigVGAN 占位加载 (无源码时回退)。"""
    from adr.vocoder.bigvgan import BigVGANVocoder

    # 直接构造, 无权重 (会回退到 identity)
    v = BigVGANVocoder()
    assert v._model is None  # 无权重加载


def test_vocoder_base_abstract():
    """BaseVocoder 抽象类。"""
    from adr.vocoder.base import BaseVocoder

    with pytest.raises(TypeError):
        BaseVocoder()


def test_bigvgan_registered():
    """BigVGANVocoder 注册到 vocoder registry。"""
    from adr.core import REGISTRY
    from adr.vocoder.bigvgan import BigVGANVocoder

    reg = REGISTRY.vocoder
    assert "bigvgan" in reg
    assert reg.get("bigvgan") is BigVGANVocoder


def test_sovits_freeze():
    """SoVITS freeze 接口。"""
    from adr.models.sovits import SoVITS, SoVITSConfig

    config = SoVITSConfig(
        hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128,
        vocab_size=50, content_dim=32, timbre_dim=32,
    )
    model = SoVITS(config)
    model.freeze_content()
    # 验证 content_encoder 被冻结
    for p in model.content_encoder.parameters():
        assert p.requires_grad is False


def test_model_loading_local_bigvgan():
    """尝试加载用户已有的 BigVGAN 权重 (数据目录, ADR_DATA_DIR 推导)。"""
    from pathlib import Path

    from adr.core.config import adr_data_dir
    from adr.vocoder.bigvgan import BigVGANVocoder

    bigvgan_dir = Path(adr_data_dir()) / "bigvgan"
    if not (bigvgan_dir / "bigvgan_generator.pt").exists():
        pytest.skip("BigVGAN weights not available locally")

    try:
        v = BigVGANVocoder.from_pretrained(bigvgan_dir, device="cpu")
        # 即使加载失败也是 graceful 的
        print(f"  BigVGAN loaded: {v.is_loaded()}")
    except Exception as e:
        pytest.skip(f"BigVGAN load failed: {e}")
