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
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from adr.core import get_logger
from adr.models.base import BaseBackbone
from adr.training.callbacks import Callback, CheckpointCallback, LoggerCallback, ProgressCallback
from adr.training.dataset import VoiceCloneDataset
from adr.training.grad_ckpt import enable_gradient_checkpointing
from adr.training.optimizer import OptimizerConfig, build_optimizer, build_scheduler


def _replace_with_retry(src: Path, dst: Path, attempts: int = 5,
                        delay: float = 0.2) -> None:
    """os.replace + 退避重试 (批次99 / R4-TRAIN-01)。

    Windows 上目标文件被杀毒/索引/另一个读取者短暂占用时 os.replace 抛
    WinError 5/32; 旧实现一次失败就让整轮训练中止。这里按 0.2s 指数退避
    重试, 仍失败才抛出 (最后一次异常保留原始错误信息)。
    """
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(delay * (2 ** i))


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

    def __post_init__(self) -> None:
        # 批次93: 设置中心「训练数据加载进程数」注入面 (ADR_S2_NUM_WORKERS)。
        # 显式传参优先; 未传时由环境变量覆盖默认 0 (主进程加载, Windows 最稳)。
        raw = os.environ.get("ADR_S2_NUM_WORKERS", "").strip()
        if not raw or self.num_workers != 0:
            return
        try:
            n = int(raw)
        except ValueError:
            return
        if 0 <= n <= 32:
            self.num_workers = n


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
        # 批次81 (#8): collate_fn 用 dataset 的绑定方法而不是 lambda ——
        # lambda 在 Windows spawn 模式下不可 pickle, num_workers>0 直接
        # PicklingError; 绑定方法可被 pickle, 且仍携带 phoneme_to_id。
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.train_data.collate,
            drop_last=True,
        )
        self.val_loader = (
            DataLoader(
                self.val_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=self.val_data.collate,
            )
            if self.val_data is not None else None
        )

        # 批次87 (BUG-01): drop_last=True 时"样本数 < batch_size"会得到 0 个 step,
        # 训练循环静默空转 (scheduler total_steps=0, 跑完不报错)。提前失败。
        if len(self.train_loader) == 0:
            raise ValueError(
                f"训练集不足以构成 1 个 batch: samples={len(self.train_data)}, "
                f"batch_size={self.config.batch_size} (drop_last=True); "
                "请减小 batch_size 或补充训练数据")

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

        # 断点续训/进度状态 (load_checkpoint 恢复; save_checkpoint 写入)
        self._start_epoch = 0          # fit 起始 epoch (resume 时 > 0)
        self._start_global_step = 0    # fit 起始全局步
        # Early Stopping 状态 (批次99 / R4-TRAIN-02: 随 checkpoint 持久化)
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.current_epoch = 0         # 已完成的 epoch 数 (下一个待跑 epoch)
        self.global_step = 0           # 累计 batch 步数

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
        # 从断点恢复 (load_checkpoint 已写入 _start_epoch/_start_global_step)
        self.global_step = self._start_global_step
        t_start = time.time()

        if self._start_epoch > 0:
            self.log.info(
                f"Resuming from epoch {self._start_epoch} "
                f"(global_step={self.global_step})"
            )

        # Early Stopping 状态 (M3)
        # 批次99 (R4-TRAIN-02): resume 时沿用 checkpoint 里的 best/耐心计数,
        # 否则重启后 best 归零, 早停永不触发且 best.pt 会被更差的权重覆盖。
        if self._start_epoch <= 0:
            self.best_val_loss = float("inf")
            self.epochs_no_improve = 0
        early_stopped = False

        for epoch in range(self._start_epoch, self.config.epochs):
            self.current_epoch = epoch  # 进行中的 epoch
            # on_epoch_start
            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            # 训练
            train_metrics = self._train_epoch(epoch, self.global_step)
            self.global_step = train_metrics["global_step"]

            # 验证
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._val_epoch(epoch)
                val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}

            # 合并
            metrics = {**train_metrics, **val_metrics}
            metrics.pop("global_step", None)

            # 本 epoch 训练+验证已完成 (供 checkpoint 记录进度)
            self.current_epoch = epoch + 1

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
                if val_loss < self.best_val_loss - self.config.early_stop_min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    self.log.info(
                        f"  [EarlyStop] val/loss improved to {val_loss:.4f}"
                    )
                else:
                    self.epochs_no_improve += 1
                    self.log.info(
                        f"  [EarlyStop] no improve ({self.epochs_no_improve}/"
                        f"{self.config.early_stop_patience})"
                    )
                    if self.epochs_no_improve >= self.config.early_stop_patience:
                        self.log.info(
                            f"  [EarlyStop] STOP at epoch {epoch+1}, "
                            f"best val/loss = {self.best_val_loss:.4f}"
                        )
                        early_stopped = True
                        break

        # on_train_end
        for cb in self.callbacks:
            cb.on_train_end(self)

        elapsed = time.time() - t_start
        self.log.info(f"Training done in {elapsed:.1f}s ({elapsed / 60:.2f} min)")
        if early_stopped:
            self.log.info(f"  Best val/loss = {self.best_val_loss:.4f}")
        return best_metrics

    def _step_optimizer(self) -> None:
        """梯度裁剪 + optimizer.step + scheduler.step (含 AMP scaler)。

        批次81 (#7): 从 ``_train_epoch`` 抽出, 供累积边界与 epoch 末尾
        余数 flush 共用, 避免两处逻辑漂移。
        """
        if self.config.grad_clip > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )

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
                self._step_optimizer()

        # 批次81 (#7): epoch 末尾不足一组的余数梯度原先被下一个 epoch 开头的
        # zero_grad() 静默丢弃 (epoch_loss 却统计了它), 这里补一次
        # clip + step + scheduler.step; 口径与 total_steps 的 ceil() 一致。
        if n_batches and n_batches % self.config.grad_accum_steps != 0:
            self.log.debug(
                f"  [GradAccum] flush {n_batches % self.config.grad_accum_steps}"
                f" batch(es) at epoch {epoch} end"
            )
            self._step_optimizer()

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
            # 批次66 #2: dequantize 需要复制模型, 不能就地破坏正在训练的 self.model
            # 全量保存前反量化 4bit 层, 避免 uint8 权重与未量化模型 shape/keys 不匹配
            from adr.training.efficient import has_quantized_modules, dequantize_4bit
            if has_quantized_modules(self.model):
                self.log.info("  checkpoint: 4bit layers detected, dequantizing a copy for save")
                import copy
                model_copy = copy.deepcopy(self.model)
                dequantize_4bit(model_copy)
                model_state_to_save = model_copy.state_dict()
                del model_copy
            else:
                model_state_to_save = self.model.state_dict()

        ckpt = {
            "model_state": model_state_to_save,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config,
            "model_class": self.model.__class__.__name__,
            "backbone_config": backbone_config,
            "use_lora": self.config.use_lora,
            # 训练进度 (resume 时恢复 epoch/global_step)
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            # 批次99 (R4-TRAIN-02): 早停状态 + 各 callback 的 best 状态,
            # 否则 resume 后 best 归零 (早停失效 / best.pt 被更差权重覆盖)
            "trainer_state": {
                "best_val_loss": self.best_val_loss,
                "epochs_no_improve": self.epochs_no_improve,
            },
            "callback_state": [
                cb.state_dict() if hasattr(cb, "state_dict") else {}
                for cb in self.callbacks
            ],
        }
        # AMP GradScaler 状态 (scale/增长追踪器), 保证 resume 后梯度缩放稳定
        if self.scaler is not None:
            ckpt["scaler"] = self.scaler.state_dict()
        # 批次90 (R3-DATA-01): 原子落盘。
        # 直写目标文件时, 训练机掉电 / 进程被杀 / 磁盘写满都会留下半截 zip,
        # 下次 torch.load 直接抛错, 整个实验不可恢复; callbacks 每 epoch 覆盖
        # best.pt 更会把上一份好文件一起毁掉。改为同目录写 .tmp → 旧文件转 .bak
        # (best-effort) → os.replace 原子换名。os.replace 在同卷上是原子的:
        # 要么看到完整新文件, 要么仍是完整的旧文件。
        tmp = path.with_name(path.name + ".tmp")
        bak = path.with_name(path.name + ".bak")
        try:
            torch.save(ckpt, tmp)
            if path.exists():
                try:
                    _replace_with_retry(path, bak)
                except OSError:
                    # 备份失败不阻断保存: 换名失败时旧文件还在, 仍是完整的
                    self.log.warning(f"  checkpoint: 覆盖前备份失败 ({bak.name})")
            _replace_with_retry(tmp, path)
        except BaseException:
            # 换名失败且旧文件已被挪走 → 立刻还原, 保证异常后仍有一份可加载的
            if not path.exists() and bak.exists():
                try:
                    os.replace(bak, path)
                except OSError:
                    pass
            raise
        finally:
            # 失败留下的半截 .tmp 必须清掉, 否则会被误当 checkpoint 加载
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """加载 checkpoint。

        LoRA 模式: 用 load_lora_state_dict 加载。
        非 LoRA 模式: 用 load_state_dict。
        """
        path = Path(path)
        # 批次87 (SEC-03): weights_only=True 优先, 需要 dataclass config 时
        # 回落到白名单扫描后的加载 (不再无条件 weights_only=False)
        # 批次65 显存双重加载: map CPU 后再由 load_state_dict 自动迁到模型设备,
        # 避免反序列化与参数复制两份 GPU 驻留
        from adr.models.gsv_engine import safe_torch_load
        ckpt = safe_torch_load(
            path, map_location="cpu",
            allow_globals=(("adr.training.trainer", "TrainerConfig"),))
        if ckpt.get("use_lora", False) or self.config.use_lora:
            from adr.training.lora import load_lora_state_dict
            loaded, missing = load_lora_state_dict(self.model, ckpt["model_state"])
            self.log.info(f"  LoRA loaded: {loaded} params, {missing} missing")
        else:
            # shape 预检查: strict=False 也无法容忍 size mismatch (PyTorch 必然 raise),
            # 提前抛出含 key/期望/实际 shape 的明确错误
            model_state = self.model.state_dict()
            for key, tensor in ckpt["model_state"].items():
                if key in model_state and model_state[key].shape != tensor.shape:
                    raise RuntimeError(
                        f"checkpoint shape mismatch: {key}: "
                        f"expected {tuple(model_state[key].shape)}, "
                        f"got {tuple(tensor.shape)}"
                    )
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
        # GradScaler 状态恢复 (仅 AMP+fp16 启用时有 scaler 实例)
        if "scaler" in ckpt:
            if self.scaler is not None:
                try:
                    self.scaler.load_state_dict(ckpt["scaler"])
                except Exception as e:
                    self.log.warning(f"  scaler state skipped: {e}")
            else:
                self.log.warning("  ckpt 含 scaler 状态但当前未启用 AMP+fp16, 已忽略")
        # 恢复训练进度 (旧格式 ckpt 缺键默认 0, 从头继续)
        if "epoch" not in ckpt or "global_step" not in ckpt:
            self.log.warning("  ckpt 缺少 epoch/global_step, 按从头训练恢复 (epoch=0)")
        self._start_epoch = int(ckpt.get("epoch", 0) or 0)
        self._start_global_step = int(ckpt.get("global_step", 0) or 0)
        self.current_epoch = self._start_epoch
        self.global_step = self._start_global_step
        # 批次99 (R4-TRAIN-02): 恢复早停状态 (旧 ckpt 缺键则保持默认)
        state = ckpt.get("trainer_state") or {}
        if state:
            self.best_val_loss = float(state.get("best_val_loss", float("inf")))
            self.epochs_no_improve = int(state.get("epochs_no_improve", 0) or 0)
        cb_states = ckpt.get("callback_state") or []
        for cb, st in zip(self.callbacks, cb_states):
            if isinstance(st, dict) and st and hasattr(cb, "load_state_dict"):
                try:
                    cb.load_state_dict(st)
                except Exception as e:
                    self.log.warning(
                        f"  callback state skipped ({type(cb).__name__}): {e}")

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
