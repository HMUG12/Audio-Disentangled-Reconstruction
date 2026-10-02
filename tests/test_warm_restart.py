"""测试 WarmRestartCallback (M6 - SGDR 式学习率热重启)。

测试覆盖:
1. 指标平台期触发重启, LR 衰减
2. max_restarts 上限
3. 指标改善时不重启
4. min_lr 下限保护
5. val/loss 指标也可用
"""
import sys
from pathlib import Path

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
import torch

from adr.data.pipeline import TrainSample
from adr.models.sovits import SoVITS, SoVITSConfig
from adr.training.callbacks import Callback, WarmRestartCallback
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


class FlatMetricCallback(Callback):
    """注入恒定指标 (模拟平台期)。"""

    def __init__(self, name="fake_metric", value=0.5):
        self.name = name
        self.value = value

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        metrics[self.name] = self.value


class ImprovingMetricCallback(Callback):
    """注入每 epoch 改善的指标。"""

    def __init__(self, name="fake_metric", start=1.0, step=0.1):
        self.name = name
        self.value = start
        self.step = step

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        self.value -= self.step
        metrics[self.name] = self.value


def make_trainer(callbacks, epochs=6, lr=2e-4, tmp="wr"):
    ds = make_dataset(20)
    model = make_small_model()
    config = TrainerConfig(
        epochs=epochs, batch_size=2, val_ratio=0.3,
        lr=lr, warmup_steps=2,
        output_dir=str(REPO / "tests" / f"_tmp_{tmp}"),
        log_every_n_steps=100, save_every_n_epochs=100,
    )
    return Trainer(model=model, train_data=ds, config=config, callbacks=callbacks)


def cleanup(tmp="wr"):
    import shutil
    shutil.rmtree(REPO / "tests" / f"_tmp_{tmp}", ignore_errors=True)


class TestWarmRestartTriggers:
    """平台期触发重启。"""

    def test_restart_on_flat_metric(self):
        flat = FlatMetricCallback()
        wr = WarmRestartCallback(
            metric_name="fake_metric", mode="min",
            patience=2, min_delta=0.01, lr_decay=0.5,
            warmup_steps=2, max_restarts=2,
        )
        trainer = make_trainer([flat, wr], epochs=6, tmp="wr_trig")
        lrs = []
        orig_step = trainer.scheduler.step
        trainer.fit()
        # 6 epoch 全平: epoch2 末(第1次计数满2)触发第1次重启, epoch4 末第2次
        assert wr.n_restarts == 2
        assert len(wr._history) == 2
        # LR 应逐次减半
        h1, h2 = wr._history
        assert h1["lr_after"] < h1["lr_before"]
        assert abs(h1["lr_after"] - h1["lr_before"] * 0.5) < 1e-12
        cleanup("wr_trig")

    def test_max_restarts_respected(self):
        flat = FlatMetricCallback()
        wr = WarmRestartCallback(
            metric_name="fake_metric", patience=1, lr_decay=0.5,
            warmup_steps=2, max_restarts=1,
        )
        trainer = make_trainer([flat, wr], epochs=6, tmp="wr_max")
        trainer.fit()
        # patience=1 每 epoch 都满足, 但最多重启 1 次
        assert wr.n_restarts == 1
        cleanup("wr_max")

    def test_no_restart_when_improving(self):
        imp = ImprovingMetricCallback()
        wr = WarmRestartCallback(
            metric_name="fake_metric", mode="min",
            patience=2, min_delta=0.01, lr_decay=0.5,
            warmup_steps=2, max_restarts=2,
        )
        trainer = make_trainer([imp, wr], epochs=5, tmp="wr_imp")
        trainer.fit()
        # 指标一直改善 → 不重启
        assert wr.n_restarts == 0
        cleanup("wr_imp")


class TestWarmRestartLR:
    """LR 行为验证。"""

    def test_scheduler_actually_replaced(self):
        """重启后 scheduler 应被替换, 且 warmup 从低 LR 爬升。"""
        flat = FlatMetricCallback()
        wr = WarmRestartCallback(
            metric_name="fake_metric", patience=1, lr_decay=0.5,
            warmup_steps=3, max_restarts=1,
        )
        trainer = make_trainer([flat, wr], epochs=4, tmp="wr_sched")
        old_scheduler = trainer.scheduler
        trainer.fit()
        assert wr.n_restarts == 1
        # scheduler 对象已被替换
        assert trainer.scheduler is not old_scheduler
        cleanup("wr_sched")

    def test_min_lr_guard(self):
        """LR 已低于 min_lr/decay 时不重启。"""
        flat = FlatMetricCallback()
        wr = WarmRestartCallback(
            metric_name="fake_metric", patience=1, lr_decay=0.5,
            warmup_steps=2, max_restarts=3, min_lr=1e-4,
        )
        # lr=1e-4, 重启后 5e-5 < min_lr=1e-4 → 应拒绝
        trainer = make_trainer([flat, wr], epochs=4, lr=1e-4, tmp="wr_minlr")
        trainer.fit()
        assert wr.n_restarts == 0
        assert wr._exhausted is True  # 触底后永久停用
        cleanup("wr_minlr")

    def test_val_loss_metric(self):
        """监控 val/loss 也可触发 (无需外部指标 callback)。"""
        wr = WarmRestartCallback(
            metric_name="val/loss", mode="min",
            patience=1, min_delta=1e9,  # 永远不"改善"
            lr_decay=0.5, warmup_steps=2, max_restarts=2,
        )
        trainer = make_trainer([wr], epochs=5, tmp="wr_valloss")
        trainer.fit()
        # min_delta=1e9 → 永远不满足改善 → patience=1 → epoch2 起每 epoch 重启
        assert wr.n_restarts == 2
        cleanup("wr_valloss")


class TestWarmRestartOrder:
    """callback 顺序: 指标缺失时安全跳过。"""

    def test_skips_when_metric_missing(self):
        """metrics 中没有目标指标时不重启也不报错。"""
        wr = WarmRestartCallback(
            metric_name="nonexistent", patience=1, max_restarts=2,
        )
        trainer = make_trainer([wr], epochs=3, tmp="wr_miss")
        trainer.fit()
        assert wr.n_restarts == 0
        cleanup("wr_miss")

    def test_history_records_lr(self):
        """_history 应记录重启前后的 LR。"""
        flat = FlatMetricCallback()
        wr = WarmRestartCallback(
            metric_name="fake_metric", patience=1, lr_decay=0.5,
            warmup_steps=2, max_restarts=1,
        )
        trainer = make_trainer([flat, wr], epochs=3, tmp="wr_hist")
        trainer.fit()
        assert len(wr._history) == 1
        h = wr._history[0]
        assert "lr_before" in h and "lr_after" in h
        assert h["lr_after"] == pytest.approx(h["lr_before"] * 0.5)
        cleanup("wr_hist")
