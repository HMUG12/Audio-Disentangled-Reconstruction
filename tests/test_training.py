"""测试训练模块 (M1 Day 8)。

测试:
- Dataset 加载 + collate
- Optimizer / scheduler 构造
- Gradient checkpointing 开关
- Trainer 5-step smoke
- Checkpoint save/load
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


def _make_dummy_sample(idx: int) -> "TrainSample":
    """构造一个虚拟训练样本。"""
    from adr.data.pipeline import TrainSample
    np.random.seed(idx)
    return TrainSample(
        sample_id=f"dummy_{idx:04d}",
        waveform=np.random.randn(24000).astype(np.float32) * 0.1,
        sample_rate=24000,
        text=f"你好世界 第{idx}句",
        phonemes=["ni3", "hao3", "shi4", "jie4"],
        f0=np.random.uniform(80, 300, 100).astype(np.float32),
        mel=np.random.randn(80, 100).astype(np.float32) * 0.1,
        start_sec=0.0,
        end_sec=1.0,
    )


def test_dataset_construct_from_samples():
    """从内存样本构造 Dataset。"""
    from adr.training.dataset import VoiceCloneDataset

    samples = [_make_dummy_sample(i) for i in range(5)]
    ds = VoiceCloneDataset(samples=samples)
    assert len(ds) == 5
    assert len(ds.phoneme_to_id) > 4  # 至少有 4 个音素 + pad
    print(f"  Dataset: {ds}")


def test_dataset_collate():
    """测试 collate 把变长样本整理为 batch。"""
    from adr.training.dataset import VoiceCloneDataset

    samples = [_make_dummy_sample(i) for i in range(3)]
    ds = VoiceCloneDataset(samples=samples)
    batch = ds.collate([ds[0], ds[1], ds[2]])

    B = 3
    assert batch.phoneme_ids.shape[0] == B
    assert batch.phoneme_mask.shape == batch.phoneme_ids.shape
    assert batch.target_mel.dim() == 3
    assert batch.target_mel.shape[0] == B
    assert batch.target_mel.shape[1] == 80  # n_mels
    assert batch.phoneme_mask.dtype == torch.bool
    print(f"  phoneme_ids: {batch.phoneme_ids.shape}")
    print(f"  target_mel:  {batch.target_mel.shape}")
    print(f"  ref_mel:     {batch.ref_mel.shape}")


def test_dataset_split():
    """测试 train/val split。"""
    from adr.training.dataset import VoiceCloneDataset

    samples = [_make_dummy_sample(i) for i in range(10)]
    ds = VoiceCloneDataset(samples=samples)
    train_ds, val_ds = ds.split(val_ratio=0.2, seed=42)
    assert len(train_ds) + len(val_ds) == 10
    assert len(val_ds) >= 1


def test_dataset_from_npz_dir(tmp_path):
    """从 npz 目录加载。"""
    from adr.data.pipeline import TrainSample
    from adr.training.dataset import VoiceCloneDataset

    # 写 3 个虚拟 npz
    for i in range(3):
        sample = _make_dummy_sample(i)
        sample.save(tmp_path / f"sample_{i}.npz")

    ds = VoiceCloneDataset(npz_dir=tmp_path)
    assert len(ds) == 3
    # 音素会 _strip_tone 去掉声调, 'ni3' -> 'ni'
    # 用 DiffSinger PhonemeDict 时 'ni' 应在 dict 中
    assert "ni" in ds.phoneme_to_id


def test_optimizer_build():
    """测试优化器构造。"""
    from adr.training.optimizer import build_optimizer, OptimizerConfig

    model = torch.nn.Linear(10, 2)
    opt = build_optimizer(model, OptimizerConfig(lr=1e-3))
    assert opt is not None
    # 检查参数组: decay vs no_decay
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.01
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_scheduler_construction():
    """测试 LR scheduler 构造。"""
    from adr.training.optimizer import build_optimizer, build_scheduler, OptimizerConfig

    model = torch.nn.Linear(10, 2)
    opt = build_optimizer(model, OptimizerConfig(lr=1e-3))
    sched = build_scheduler(opt, OptimizerConfig(warmup_steps=5, scheduler="cosine"), total_steps=100)
    assert sched is not None
    # 走一步 (先 optimizer 后 scheduler, 符合 PyTorch 约定)
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] > 0


def test_gradient_checkpointing():
    """测试 gradient checkpointing 开关。"""
    from adr.models.sovits import SoVITS, SoVITSConfig
    from adr.training.grad_ckpt import (
        enable_gradient_checkpointing,
        is_gradient_checkpointing_enabled,
    )

    model = SoVITS(SoVITSConfig(
        hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128,
        vocab_size=607, content_dim=32, timbre_dim=32,
    ))
    n = enable_gradient_checkpointing(model, enabled=True)
    assert is_gradient_checkpointing_enabled(model)
    assert n >= 0
    # 关掉
    enable_gradient_checkpointing(model, enabled=False)
    assert not is_gradient_checkpointing_enabled(model)


def test_trainer_smoke():
    """Trainer 5-step smoke test。"""
    from adr.models.sovits import SoVITS, SoVITSConfig
    from adr.training import Trainer, TrainerConfig
    from adr.training.dataset import VoiceCloneDataset

    # 小模型
    model = SoVITS(SoVITSConfig(
        hidden_dim=32, n_layers=2, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
        n_mels=80,
    ))

    # 小数据集
    samples = [_make_dummy_sample(i) for i in range(8)]
    ds = VoiceCloneDataset(samples=samples)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TrainerConfig(
            epochs=2,
            batch_size=2,
            use_amp=False,           # CPU 模式
            use_gradient_checkpointing=False,
            output_dir=tmp_dir,
            val_ratio=0.0,           # 不划分验证集
            warmup_steps=2,
        )
        trainer = Trainer(model=model, train_data=ds, config=config)
        metrics = trainer.fit()

        assert "train/loss" in metrics
        assert metrics["train/loss"] >= 0

        # 检查输出文件
        ckpt_dir = Path(tmp_dir) / "checkpoints"
        assert ckpt_dir.exists()
        log_file = Path(tmp_dir) / "train_log.jsonl"
        assert log_file.exists()
        print(f"  Final train loss: {metrics['train/loss']:.4f}")


def test_trainer_checkpoint_save_load():
    """checkpoint 保存/加载。"""
    from adr.models.sovits import SoVITS, SoVITSConfig
    from adr.training import Trainer, TrainerConfig
    from adr.training.dataset import VoiceCloneDataset

    model = SoVITS(SoVITSConfig(
        hidden_dim=32, n_layers=2, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    ))
    samples = [_make_dummy_sample(i) for i in range(4)]
    ds = VoiceCloneDataset(samples=samples)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TrainerConfig(
            epochs=1, batch_size=2, use_amp=False,
            use_gradient_checkpointing=False,
            output_dir=tmp_dir, val_ratio=0.0,
        )
        trainer = Trainer(model=model, train_data=ds, config=config)
        # 保存
        ckpt_path = Path(tmp_dir) / "test.pt"
        trainer.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()
        # 加载 (用新 trainer)
        new_model = SoVITS(SoVITSConfig(
            hidden_dim=32, n_layers=2, n_heads=2, ffn_dim=64,
            vocab_size=607, content_dim=16, timbre_dim=16,
        ))
        new_trainer = Trainer(model=new_model, train_data=ds, config=config)
        new_trainer.load_checkpoint(ckpt_path)
        print(f"  Checkpoint load OK ({ckpt_path.stat().st_size} bytes)")


def test_trainer_with_val():
    """带验证集的 trainer。"""
    from adr.models.sovits import SoVITS, SoVITSConfig
    from adr.training import Trainer, TrainerConfig
    from adr.training.dataset import VoiceCloneDataset

    model = SoVITS(SoVITSConfig(
        hidden_dim=32, n_layers=2, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    ))
    samples = [_make_dummy_sample(i) for i in range(10)]
    ds = VoiceCloneDataset(samples=samples)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TrainerConfig(
            epochs=2, batch_size=2, use_amp=False,
            use_gradient_checkpointing=False,
            output_dir=tmp_dir, val_ratio=0.2,  # 自动划分
        )
        trainer = Trainer(model=model, train_data=ds, config=config)
        assert trainer.val_loader is not None
        metrics = trainer.fit()
        assert "val/loss" in metrics
        print(f"  val/loss: {metrics['val/loss']:.4f}")


def test_callbacks_invoke():
    """测试 callback 调度。"""
    from adr.models.sovits import SoVITS, SoVITSConfig
    from adr.training import Trainer, TrainerConfig
    from adr.training.callbacks import Callback
    from adr.training.dataset import VoiceCloneDataset

    state = {"on_train_start": 0, "on_epoch_start": 0, "on_epoch_end": 0, "on_train_end": 0}

    class TrackingCallback(Callback):
        def on_train_start(self, trainer, **kw):
            state["on_train_start"] += 1
        def on_epoch_start(self, trainer, epoch, **kw):
            state["on_epoch_start"] += 1
        def on_epoch_end(self, trainer, epoch, metrics, **kw):
            state["on_epoch_end"] += 1
        def on_train_end(self, trainer, **kw):
            state["on_train_end"] += 1

    cb = TrackingCallback()
    model = SoVITS(SoVITSConfig(
        hidden_dim=32, n_layers=2, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    ))
    samples = [_make_dummy_sample(i) for i in range(4)]
    ds = VoiceCloneDataset(samples=samples)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TrainerConfig(
            epochs=3, batch_size=2, use_amp=False,
            use_gradient_checkpointing=False,
            output_dir=tmp_dir, val_ratio=0.0,
        )
        trainer = Trainer(
            model=model, train_data=ds, config=config,
            callbacks=[cb],  # 不传默认,只跟踪
        )
        trainer.fit()

    assert state["on_train_start"] == 1
    assert state["on_epoch_start"] == 3
    assert state["on_epoch_end"] == 3
    assert state["on_train_end"] == 1
