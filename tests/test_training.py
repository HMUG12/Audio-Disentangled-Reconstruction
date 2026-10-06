"""测试训练模块 (M1 Day 8)。

测试:
- Dataset 加载 + collate
- Optimizer / scheduler 构造
- Gradient checkpointing 开关
- Trainer 5-step smoke
- Checkpoint save/load
"""

from __future__ import annotations

import random
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
    # speaker_ids 与 ref 池随 split 重建
    assert len(train_ds.speaker_ids) == len(train_ds)
    assert len(train_ds._ref_pools) == len(train_ds)


def test_dataset_ref_mel_pool_avoids_shortcut():
    """ref_mel 应取同说话人异样本, 切断 ref→target 的抄写捷径。"""
    from adr.training.dataset import VoiceCloneDataset

    samples = [_make_dummy_sample(i) for i in range(4)]
    ds = VoiceCloneDataset(samples=samples)

    # 池不含自身
    for i in range(4):
        assert i not in ds._ref_pools[i]
        assert len(ds._ref_pools[i]) == 3

    # __getitem__ 挂 _ref_mel (浅拷贝), 原样本对象不被修改
    s = ds[0]
    assert s._ref_mel is not None
    assert s._ref_mel is not samples[0].mel
    assert not hasattr(samples[0], "_ref_mel")

    # collate 后 ref_mel 不等于 target_mel (异样本)
    batch = ds.collate([ds[i] for i in range(4)])
    rT = batch.ref_mel.shape[-1]
    assert not torch.allclose(batch.ref_mel[0], batch.target_mel[0, :, :rT])


def test_dataset_multi_speaker_pools():
    """多说话人分组: 池只含同说话人的异样本。"""
    from adr.training.dataset import VoiceCloneDataset

    samples = [_make_dummy_sample(i) for i in range(4)]
    ds = VoiceCloneDataset(samples=samples, speaker_ids=["a", "a", "b", "b"])
    assert ds._ref_pools[0] == [1]
    assert ds._ref_pools[1] == [0]
    assert ds._ref_pools[2] == [3]
    assert ds._ref_pools[3] == [2]


def test_dataset_ref_mel_single_sample_fallback(caplog):
    """单样本说话人: 无异样本可用 → 回退同样本 + warning。"""
    from adr.training.dataset import VoiceCloneDataset

    samples = [_make_dummy_sample(0)]
    with caplog.at_level("WARNING", logger="adr.training.dataset"):
        ds = VoiceCloneDataset(samples=samples)
    assert ds._ref_pools[0] == []
    assert any("捷径" in r.message for r in caplog.records)

    # 回退: ref 取自身前半段
    batch = ds.collate([ds[0]])
    rT = batch.ref_mel.shape[-1]
    assert torch.allclose(batch.ref_mel[0], batch.target_mel[0, :, :rT])


def test_dataset_speaker_ids_length_mismatch():
    """speaker_ids 长度与样本数不符应报错。"""
    from adr.training.dataset import VoiceCloneDataset

    samples = [_make_dummy_sample(i) for i in range(3)]
    with pytest.raises(ValueError, match="不一致"):
        VoiceCloneDataset(samples=samples, speaker_ids=["a", "b"])


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
    """测试 gradient checkpointing 开关: 包装所有 TransformerEncoderLayer。"""
    from adr.models.sovits import SoVITS, SoVITSConfig
    from adr.training.grad_ckpt import (
        enable_gradient_checkpointing,
        is_gradient_checkpointing_enabled,
    )

    model = SoVITS(SoVITSConfig(
        hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128,
        vocab_size=607, content_dim=32, timbre_dim=32,
    ))
    # decoder(2) + content_encoder(4) = 6 个 TransformerEncoderLayer
    n_layers = sum(
        1 for m in model.modules()
        if isinstance(m, torch.nn.TransformerEncoderLayer)
    )
    n = enable_gradient_checkpointing(model, enabled=True)
    assert is_gradient_checkpointing_enabled(model)
    assert n == n_layers
    # 幂等: 重复开启不重复包装
    n2 = enable_gradient_checkpointing(model, enabled=True)
    assert n2 == 0
    # 关掉
    n3 = enable_gradient_checkpointing(model, enabled=False)
    assert n3 == n_layers
    assert not is_gradient_checkpointing_enabled(model)


def _sovits_forward_backward_hook_count(model) -> int:
    """跑一次 forward+backward, 返回 decoder 第一层 linear1 的 forward 调用次数。"""
    counts = {"n": 0}
    linear1 = model.decoder.layers[0].linear1
    hook = linear1.register_forward_hook(
        lambda m, i, o: counts.__setitem__("n", counts["n"] + 1)
    )
    try:
        B, T_p, T_mel = 2, 8, 16
        batch = {
            "phoneme_ids": torch.randint(4, 100, (B, T_p)),
            "phoneme_mask": torch.ones(B, T_p, dtype=torch.bool),
            "ref_mel": torch.randn(B, 80, 8) * 0.1,
            "target_mel": torch.randn(B, 80, T_mel) * 0.1,
            "target_durations": torch.full((B, T_p), 2, dtype=torch.long),
            "f0": torch.rand(B, T_mel) * 200 + 80,
        }
        model.zero_grad(set_to_none=True)
        out = model(batch)
        out["loss"].backward()
    finally:
        hook.remove()
    return counts["n"]


def test_gradient_checkpointing_recompute():
    """证明 checkpointing 真生效: 开启后 backward 期间重计算 → linear1 forward 被调用 2 次。"""
    from adr.models.sovits import SoVITS, SoVITSConfig
    from adr.training.grad_ckpt import enable_gradient_checkpointing

    model = SoVITS(SoVITSConfig(
        hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128,
        vocab_size=607, content_dim=32, timbre_dim=32,
    ))
    model.train()

    # 关闭: 只有主 forward 1 次
    enable_gradient_checkpointing(model, enabled=False)
    assert _sovits_forward_backward_hook_count(model) == 1

    # 开启: 主 forward 1 次 + backward 重计算 1 次 = 2 次
    enable_gradient_checkpointing(model, enabled=True)
    assert _sovits_forward_backward_hook_count(model) == 2


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


def test_sovits_mel_loss_padding_mask():
    """mel loss 应按 target_mel_mask 只对有效帧计算, padding 帧不贡献损失。"""
    from adr.models.sovits import SoVITS, SoVITSConfig

    model = SoVITS(SoVITSConfig(
        hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128,
        vocab_size=607, content_dim=32, timbre_dim=32,
    ))
    model.eval()  # 固定 dropout, 保证两次 forward 可比

    B, T_p, T_mel = 2, 8, 16
    base = {
        "phoneme_ids": torch.randint(4, 100, (B, T_p)),
        "phoneme_mask": torch.ones(B, T_p, dtype=torch.bool),
        "ref_mel": torch.randn(B, 80, 8) * 0.1,
        "target_durations": torch.full((B, T_p), 2, dtype=torch.long),
        "f0": torch.rand(B, T_mel) * 200 + 80,
    }
    target = torch.randn(B, 80, T_mel) * 0.1
    mask = torch.ones(B, T_mel, dtype=torch.bool)
    mask[1, 8:] = False  # 第二个样本后半是 padding

    with torch.no_grad():
        out1 = model({**base, "target_mel": target, "target_mel_mask": mask})
        # padding 区改成极端值: 有 mask 时 loss 不应变化
        polluted = target.clone()
        polluted[1, :, 8:] = 100.0
        out2 = model({**base, "target_mel": polluted, "target_mel_mask": mask})
    assert torch.allclose(out1["loss"], out2["loss"], rtol=1e-5)

    # 对照: 无 mask 时走 F.l1_loss, 污染帧会推高 loss
    with torch.no_grad():
        out3 = model({**base, "target_mel": polluted})
    assert out3["loss"] > out2["loss"]


def test_yaml_train_section_mapping(configs_dir, caplog):
    """vram_*.yaml 的 train 嵌套键显式映射到 TrainerConfig; 未映射键 warning。"""
    import yaml
    from adr.training.trainer import TrainerConfig, apply_yaml_to_trainer_config

    with open(configs_dir / "vram_8gb.yaml", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)

    cfg = TrainerConfig()
    with caplog.at_level("WARNING", logger="adr.training"):
        applied = apply_yaml_to_trainer_config(yaml_cfg, cfg)

    # 旧 hasattr 循环会静默丢弃的嵌套键, 现在正确映射
    assert cfg.batch_size == 4
    assert cfg.grad_accum_steps == 4
    assert cfg.lr == 2.0e-4
    assert cfg.epochs == 10
    assert cfg.amp_dtype == "fp16"
    assert cfg.use_gradient_checkpointing is True
    assert "gradient_accumulation_steps->grad_accum_steps" in applied

    # 未映射键汇总 warning (use_qlora 故意不映射, 由 CLI 标志控制)
    all_msgs = " ".join(r.message for r in caplog.records)
    assert "未映射" in all_msgs
    for key in ("use_flash_attn", "use_qlora", "save_every_n_steps"):
        assert key in all_msgs
    assert cfg.weight_decay == 0.01  # yaml 中无该键, 不被触碰


def test_scheduler_total_steps_divides_grad_accum():
    """total_steps 按 optimizer 更新次数计 (除以 grad_accum, 向上取整)。"""
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
            epochs=2, batch_size=2, grad_accum_steps=2,
            lr=2e-4, warmup_steps=1,
            use_amp=False, use_gradient_checkpointing=False,
            output_dir=tmp_dir, val_ratio=0.0,
        )
        trainer = Trainer(model=model, train_data=ds, config=config)
        # 4 样本 / batch 2 → 2 batches/epoch; grad_accum=2 → 1 次优化/epoch → 总 2 步
        metrics = trainer.fit()

    base_lr = 2e-4
    final_lr = metrics["train/lr"]
    # 2 个优化步 = total_steps → cosine 走完 → lr = base * min_lr_ratio (0.1)
    # 旧实现 total_steps=4 (未除以 grad_accum) 会停在 ~0.775*base
    assert abs(final_lr - base_lr * 0.1) <= 0.02 * base_lr * 0.1


def test_trainer_seed_full_and_deterministic():
    """Trainer 构造时补全 random/numpy/cuda 种子 + cudnn deterministic 开关。"""
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
            seed=123, deterministic=True,
        )
        trainer = Trainer(model=model, train_data=ds, config=config)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    # numpy/random 种子生效: 构造后 RNG 状态与重设 seed 后首个取值一致
    # (说明 seed 被正确设定且构造过程中无隐式消耗)
    v_np = np.random.rand()
    np.random.seed(123)
    assert v_np == np.random.rand()

    v_py = random.random()
    random.seed(123)
    assert v_py == random.random()
