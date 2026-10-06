"""主训练器 (M1 Day 8)。

设计目标:
- 8GB 显存可训练 SoVITS 简化版 (~50M 参数)
- 5 分钟 5-秒克隆 demo 跑通
- 支持 AMP (FP16/BF16) + Gradient Checkpointing
- 5 行启动训练: trainer = Trainer.from_config(...); trainer.fit()
- 插件式 callback (save/log/earlystop)

用法:
    >>> from adr.training import Trainer, TrainerConfig
    >>> from adr.models import SoVITS
    >>> model = SoVITS(SoVITSConfig(...))
    >>> trainer = Trainer(model=model, train_data=ds, config=TrainerConfig(epochs=10))
    >>> trainer.fit()
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adr.core import get_logger
from adr.models.base import BaseBackbone
from adr.training.callbacks import Callback, CheckpointCallback, LoggerCallback, ProgressCallback
from adr.training.dataset import VoiceCloneDataset, collate_samples
from adr.training.grad_ckpt import enable_gradient_checkpointing
from adr.training.optimizer import OptimizerConfig, build_optimizer, build_scheduler


@dataclass
class TrainerConfig:
    """训练器配置。"""
    # 基础
    epochs: int = 10
    batch_size: int = 4
    grad_accum_steps: int = 1       # 梯度累积 (显存不够时调大)
    num_workers: int = 0            # DataLoader workers
    val_ratio: float = 0.1

    # 优化
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip: float = 1.0          # 梯度裁剪 (-1 关闭)

    # 显存优化
    use_amp: bool = True            # 自动混合精度
    amp_dtype: str = "fp16"         # "fp16" / "bf16"
    use_gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = False

    # LoRA (M2)
    use_lora: bool = False          # 启用 LoRA 微调
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple = ("out_proj", "linear1", "linear2")  # 目标模块名 (SoVITS 实际名)

    # 输出
    output_dir: str = "output/train"
    save_every_n_epochs: int = 1
    log_every_n_steps: int = 10

    # Early Stopping (M3)
    early_stop_patience: int = 0      # 0 = 关闭, N = val loss 连续 N epoch 不下降则停
    early_stop_min_delta: float = 1e-4  # 视为"下降"的最小幅度

    # 设备
    device: str = "auto"            # auto / cpu / cuda
    seed: int = 42
    deterministic: bool = False     # cudnn 确定性模式 (牺牲速度换可复现)


class Trainer:
    """主训练器。

    负责:
    1. 构造 dataloader (train/val split)
    2. 构造 optimizer + scheduler
    3. 训练循环 (forward → loss → backward → step)
    4. 验证 (每 epoch)
    5. callback 调度
    6. checkpoint save/load
    """

    def __init__(
        self,
        model: BaseBackbone,
        train_data: Union[VoiceCloneDataset, str, Path],
        val_data: Optional[VoiceCloneDataset] = None,
        config: Optional[TrainerConfig] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.config = config or TrainerConfig()
        self.log = get_logger("adr.training")

        # 设备
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # 完整种子设定 (torch / random / numpy / cuda) + cudnn 确定性开关
        seed = self.config.seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 模型
        self.model = model.to(self.device)
        self.log.info(f"Model: {model.__class__.__name__} -> {self.device}")
        self.log.info(f"  params: {model.num_parameters_str()}")

        # LoRA 注入 (M2)
        if self.config.use_lora:
            from adr.training.lora import LoRAConfig, apply_lora
            lora_cfg = LoRAConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                target_modules=list(self.config.lora_target_modules),
            )
            self.log.info(f"LoRA: 应用 rank={lora_cfg.rank} alpha={lora_cfg.alpha} "
                         f"target={lora_cfg.target_modules}")
            apply_lora(self.model, lora_cfg)
            n_lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.log.info(f"  LoRA trainable: {n_lora_params/1e6:.4f}M")

        # 数据
        if isinstance(train_data, (str, Path)):
            train_data = VoiceCloneDataset(npz_dir=train_data)
        if val_data is None and self.config.val_ratio > 0:
            train_data, val_data = train_data.split(
                val_ratio=self.config.val_ratio,
                seed=self.config.seed,
            )
        self.train_data = train_data
        self.val_data = val_data
        self.log.info(
            f"Data: train={len(self.train_data)}, "
            f"val={len(self.val_data) if self.val_data else 0}"
        )

        # DataLoader
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=lambda b: self.train_data.collate(b),
            drop_last=True,
        )
        self.val_loader = (
            DataLoader(
                self.val_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=lambda b: self.val_data.collate(b),
            )
            if self.val_data is not None else None
        )

        # 优化器
        opt_config = OptimizerConfig(
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            use_8bit=self.config.use_8bit_optimizer,
        )
        self.optimizer = build_optimizer(self.model, opt_config)
        # optimizer 只在梯度累积边界步进, scheduler 总步数按有效更新次数计 (向上取整)
        total_steps = math.ceil(len(self.train_loader) / self.config.grad_accum_steps) * self.config.epochs
        self.scheduler = build_scheduler(self.optimizer, opt_config, total_steps=total_steps)

        # AMP
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        if self.config.amp_dtype == "bf16":
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp and self.amp_dtype == torch.float16 else None

        # Gradient Checkpointing
        if self.config.use_gradient_checkpointing:
            n = enable_gradient_checkpointing(self.model, enabled=True)
            self.log.info(f"Gradient checkpointing: enabled ({n} modules)")

        # Callbacks (默认)
        self.callbacks = callbacks or []
        if not any(isinstance(cb, ProgressCallback) for cb in self.callbacks):
            self.callbacks.append(ProgressCallback(print_every_n_steps=self.config.log_every_n_steps))
        if not any(isinstance(cb, CheckpointCallback) for cb in self.callbacks):
            self.callbacks.append(CheckpointCallback(
                save_dir=self.config.output_dir + "/checkpoints",
                save_every_n_epochs=self.config.save_every_n_epochs,
            ))
        if not any(isinstance(cb, LoggerCallback) for cb in self.callbacks):
            self.callbacks.append(LoggerCallback(
                log_path=self.config.output_dir + "/train_log.jsonl",
            ))

    def fit(self) -> dict:
        """主训练循环。"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log.info("=" * 60)
        self.log.info("Training start")
        self.log.info(f"  Output: {output_dir}")
        self.log.info(f"  Epochs: {self.config.epochs}")
        self.log.info(f"  Batch:  {self.config.batch_size} (grad_accum={self.config.grad_accum_steps})")
        self.log.info(f"  Steps/epoch: {len(self.train_loader)}")
        self.log.info(f"  AMP: {self.use_amp} ({self.amp_dtype})")
        self.log.info("=" * 60)

        # 触发 on_train_start
        for cb in self.callbacks:
            cb.on_train_start(self)

        best_metrics = {}
        global_step = 0
        t_start = time.time()

        # Early Stopping 状态 (M3)
        best_val_loss = float("inf")
        epochs_no_improve = 0
        early_stopped = False

        for epoch in range(self.config.epochs):
            # on_epoch_start
            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            # 训练
            train_metrics = self._train_epoch(epoch, global_step)
            global_step = train_metrics["global_step"]

            # 验证
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._val_epoch(epoch)
                val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}

            # 合并
            metrics = {**train_metrics, **val_metrics}
            metrics.pop("global_step", None)

            # on_epoch_end
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, metrics)

            best_metrics.update(metrics)

            # 检查 callback 是否要求停止 (M3 - WavQualityCallback)
            if any(getattr(cb, "should_stop", False) for cb in self.callbacks):
                self.log.info(
                    f"  [CallbackStop] epoch {epoch+1}: "
                    f"callback requested stop"
                )
                early_stopped = True
                break

            # Early Stopping 检查 (M3)
            if self.config.early_stop_patience > 0 and "val/loss" in metrics:
                val_loss = metrics["val/loss"]
                if val_loss < best_val_loss - self.config.early_stop_min_delta:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    self.log.info(
                        f"  [EarlyStop] val/loss improved to {val_loss:.4f}"
                    )
                else:
                    epochs_no_improve += 1
                    self.log.info(
                        f"  [EarlyStop] no improve ({epochs_no_improve}/"
                        f"{self.config.early_stop_patience})"
                    )
                    if epochs_no_improve >= self.config.early_stop_patience:
                        self.log.info(
                            f"  [EarlyStop] STOP at epoch {epoch+1}, "
                            f"best val/loss = {best_val_loss:.4f}"
                        )
                        early_stopped = True
                        break

        # on_train_end
        for cb in self.callbacks:
            cb.on_train_end(self)

        elapsed = time.time() - t_start
        self.log.info(f"Training done in {elapsed:.1f}s ({elapsed / 60:.2f} min)")
        if early_stopped:
            self.log.info(f"  Best val/loss = {best_val_loss:.4f}")
        return best_metrics

    def _train_epoch(self, epoch: int, global_step: int) -> dict:
        """一个训练 epoch。"""
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            # 转到 device
            batch_dict = {
                "phoneme_ids": batch.phoneme_ids.to(self.device),
                "phoneme_mask": batch.phoneme_mask.to(self.device),
                "ref_mel": batch.ref_mel.to(self.device),
                "target_mel": batch.target_mel.to(self.device),
                "target_mel_mask": batch.target_mel_mask.to(self.device),
                "target_durations": batch.target_durations.to(self.device),
                "f0": batch.f0.to(self.device),
            }

            # forward
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    out = self.model(batch_dict)
                    loss = out["loss"] / self.config.grad_accum_steps
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                out = self.model(batch_dict)
                loss = out["loss"] / self.config.grad_accum_steps
                loss.backward()

            epoch_loss += loss.item() * self.config.grad_accum_steps
            n_batches += 1
            global_step += 1

            # 累积到一定步数才更新
            if (step + 1) % self.config.grad_accum_steps == 0:
                # 梯度裁剪
                if self.config.grad_clip > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )

                # 更新
                if self.scaler is not None:
                    # AMP: inf 梯度时 scaler 会跳过 step (scale 下降)
                    prev_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    opt_stepped = self.scaler.get_scale() >= prev_scale
                else:
                    self.optimizer.step()
                    opt_stepped = True
                self.optimizer.zero_grad()
                # 只有 optimizer 真正更新后才步进 scheduler
                # (避免 "lr_scheduler.step() before optimizer.step()" 警告)
                if opt_stepped:
                    self.scheduler.step()

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t_start
        return {
            "train/loss": avg_loss,
            "train/epoch_time_sec": elapsed,
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "global_step": global_step,
        }

    @torch.inference_mode()
    def _val_epoch(self, epoch: int) -> dict:
        """一个验证 epoch。"""
        self.model.eval()
        epoch_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch_dict = {
                "phoneme_ids": batch.phoneme_ids.to(self.device),
                "phoneme_mask": batch.phoneme_mask.to(self.device),
                "ref_mel": batch.ref_mel.to(self.device),
                "target_mel": batch.target_mel.to(self.device),
                "target_mel_mask": batch.target_mel_mask.to(self.device),
                "target_durations": batch.target_durations.to(self.device),
                "f0": batch.f0.to(self.device),
            }

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    out = self.model(batch_dict)
            else:
                out = self.model(batch_dict)
            epoch_loss += out["loss"].item()
            n_batches += 1

        return {"loss": epoch_loss / max(1, n_batches)}

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """保存 checkpoint (model + optimizer + config)。

        LoRA 模式: 只保存 LoRA 参数,大幅减小 checkpoint 体积。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 尝试保存 backbone config (如果有)
        backbone_config = None
        if hasattr(self.model, "config") and self.model.config is not None:
            try:
                from dataclasses import asdict, is_dataclass
                if is_dataclass(self.model.config):
                    backbone_config = asdict(self.model.config)
            except Exception:
                pass

        # LoRA 模式: 只保存 LoRA 参数
        if self.config.use_lora:
            from adr.training.lora import get_lora_state_dict
            model_state_to_save = get_lora_state_dict(self.model)
            self.log.info(f"  checkpoint: saving LoRA-only "
                         f"({sum(v.numel() for v in model_state_to_save.values())/1e3:.1f}K params)")
        else:
            # 全量保存前反量化 4bit 层, 避免 uint8 权重与未量化模型 shape/keys 不匹配
            from adr.training.efficient import has_quantized_modules, dequantize_4bit
            if has_quantized_modules(self.model):
                self.log.info("  checkpoint: 4bit layers detected, dequantizing before save")
                dequantize_4bit(self.model)
            model_state_to_save = self.model.state_dict()

        torch.save({
            "model_state": model_state_to_save,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
            "model_class": self.model.__class__.__name__,
            "backbone_config": backbone_config,
            "use_lora": self.config.use_lora,
        }, path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """加载 checkpoint。

        LoRA 模式: 用 load_lora_state_dict 加载。
        非 LoRA 模式: 用 load_state_dict。
        """
        path = Path(path)
        # weights_only=False 是因为我们保存了 dataclass config
        # ADR checkpoint 只来自我们自己的训练,trusted
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("use_lora", False) or self.config.use_lora:
            from adr.training.lora import load_lora_state_dict
            loaded, missing = load_lora_state_dict(self.model, ckpt["model_state"])
            self.log.info(f"  LoRA loaded: {loaded} params, {missing} missing")
        else:
            # strict 失败时 (如旧基座缺新增模块, 例 f0_embed) 降级 strict=False
            try:
                self.model.load_state_dict(ckpt["model_state"])
            except RuntimeError as e:
                self.log.warning(
                    f"  strict load failed ({e}), retrying with strict=False"
                )
                result = self.model.load_state_dict(ckpt["model_state"], strict=False)
                if result.missing_keys:
                    self.log.info(f"  new params (random init): {result.missing_keys}")
                if result.unexpected_keys:
                    self.log.warning(f"  ignored ckpt params: {result.unexpected_keys}")
        # optimizer/scheduler: 结构不匹配时跳过 (finetune 场景从基座热启动)
        if "optimizer_state" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                self.log.warning(f"  optimizer state skipped: {e}")
        if "scheduler_state" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception as e:
                self.log.warning(f"  scheduler state skipped: {e}")

    @classmethod
    def from_config(
        cls,
        model: BaseBackbone,
        train_data: Union[VoiceCloneDataset, str, Path],
        config_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> "Trainer":
        """从 YAML 配置构造 (M2 完整实现)。"""
        config = TrainerConfig(**kwargs)
        return cls(model=model, train_data=train_data, config=config)


# ============================================================
# YAML → TrainerConfig 显式映射 (vram_*.yaml 的 train 节)
# ============================================================
YAML_TRAIN_KEY_MAP: dict = {
    "batch_size": "batch_size",
    "gradient_accumulation_steps": "grad_accum_steps",
    "learning_rate": "lr",
    "num_epochs": "epochs",
    "weight_decay": "weight_decay",
    "warmup_steps": "warmup_steps",
    "max_grad_norm": "grad_clip",
    "use_gradient_checkpointing": "use_gradient_checkpointing",
    "precision": "amp_dtype",
    "save_every_n_epochs": "save_every_n_epochs",
    "log_every_n_steps": "log_every_n_steps",
}

_AMP_DTYPE_ALIASES = {"fp16": "fp16", "float16": "fp16", "bf16": "bf16", "bfloat16": "bf16"}


def apply_yaml_to_trainer_config(yaml_cfg: dict, cfg: "TrainerConfig", log=None) -> List[str]:
    """把 YAML 配置的 train 节显式映射到 TrainerConfig。

    - 仅处理顶层 'train' 子节 (若不存在则视 yaml_cfg 本身为 train 节)
    - 映射表之外的键不静默丢弃: 汇总后单条 warning
    - 不映射 use_qlora (由 CLI --lora/--qlora 标志控制, 避免 yaml 覆盖 CLI)

    Returns:
        实际应用的 "yaml键->配置字段" 列表
    """
    log = log or get_logger("adr.training")
    train_cfg = yaml_cfg.get("train", yaml_cfg) if isinstance(yaml_cfg, dict) else {}
    applied: List[str] = []
    unmapped: List[str] = []
    for key, value in train_cfg.items():
        field_name = YAML_TRAIN_KEY_MAP.get(key)
        if field_name is None:
            unmapped.append(key)
            continue
        if field_name == "amp_dtype":
            norm = _AMP_DTYPE_ALIASES.get(str(value).lower())
            if norm is None:
                unmapped.append(key)
                continue
            value = norm
        setattr(cfg, field_name, value)
        applied.append(f"{key}->{field_name}")
    if unmapped:
        log.warning(f"  YAML 训练配置中未映射的键已忽略: {unmapped}")
    return applied
