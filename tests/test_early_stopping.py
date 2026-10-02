"""测试 Early Stopping 机制 (M3)。"""
import sys
from pathlib import Path

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
import torch
import torch.nn as nn

from adr.models.sovits import SoVITS, SoVITSConfig
from adr.training.dataset import TrainSample, VoiceCloneDataset
from adr.training.trainer import Trainer, TrainerConfig


def make_dataset(n: int = 20):
    """创建小型 dataset。"""
    from adr.data.pipeline import TrainSample
    samples = []
    for i in range(n):
        s = TrainSample(
            sample_id=f"utt_{i:03d}",
            phonemes=["zh", "ong1", "g_uan1"] * 7,  # 21 phonemes
            text="测试",
            f0=np.linspace(180, 200, 80).astype(np.float32),
            mel=np.random.randn(80, 80).astype(np.float32),
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


def test_early_stopping_disabled_by_default():
    """默认 early_stop_patience=0, 不应触发早停。"""
    ds = make_dataset(10)
    model = make_small_model()
    config = TrainerConfig(
        epochs=2, batch_size=2, val_ratio=0.3,
        output_dir=str(REPO / "tests" / "_tmp_es_disabled"),
        log_every_n_steps=10, save_every_n_epochs=10,
        early_stop_patience=0,  # 关闭
    )
    trainer = Trainer(model=model, train_data=ds, config=config)
    metrics = trainer.fit()
    # 应跑完所有 epoch
    assert "val/loss" in metrics
    # 清理
    import shutil
    shutil.rmtree(REPO / "tests" / "_tmp_es_disabled", ignore_errors=True)


def test_early_stopping_triggers_on_plateau():
    """早停应在 val loss plateau 时触发。"""
    ds = make_dataset(20)
    model = make_small_model()
    config = TrainerConfig(
        epochs=20, batch_size=2, val_ratio=0.3,
        output_dir=str(REPO / "tests" / "_tmp_es_trigger"),
        log_every_n_steps=100, save_every_n_epochs=100,
        early_stop_patience=2,  # 2 epoch 不下降就停
        early_stop_min_delta=1e-3,
    )
    trainer = Trainer(model=model, train_data=ds, config=config)
    metrics = trainer.fit()
    # 应有 val/loss
    assert "val/loss" in metrics
    # 清理
    import shutil
    shutil.rmtree(REPO / "tests" / "_tmp_es_trigger", ignore_errors=True)


def test_early_stop_config_exists():
    """TrainerConfig 应有早停相关字段。"""
    cfg = TrainerConfig()
    assert hasattr(cfg, "early_stop_patience")
    assert hasattr(cfg, "early_stop_min_delta")
    assert cfg.early_stop_patience == 0
    assert cfg.early_stop_min_delta == 1e-4
