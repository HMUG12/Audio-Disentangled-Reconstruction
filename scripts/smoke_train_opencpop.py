"""Smoke training with REAL OpenCpop data (M2 方向 A 验证)。

验证:
- 真实数据加载
- Trainer.fit() 完整流程
- 真实数据上能跑通

用法:
    python scripts/smoke_train_opencpop.py
    python scripts/smoke_train_opencpop.py --preset medium --epochs 1 --n-samples 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from adr.core import get_logger
from adr.data.phoneme_dict import encode_phonemes, load_default_phoneme_dict
from adr.models.sovits import SoVITS, SoVITSConfig
from adr.training import Trainer, TrainerConfig


PRESET_SIZES = {
    "small":  (256, 4, 4, 256, 128),
    "medium": (512, 6, 8, 512, 256),
    "x0p3b":  (1024, 24, 16, 1024, 512),
}


def build_model(preset: str = "medium", vocab_size: int = 607,
                model_type: str = "sovits"):
    hd, nl, nh, cd, td = PRESET_SIZES[preset]
    if model_type == "adr2":
        # ADR-2 骨架: z_dim 固定 64, flow 4 层; preset 只调主干维度
        from adr.models.adr2 import ADR2, ADR2Config
        cfg = ADR2Config(
            hidden_dim=hd, content_dim=cd, timbre_dim=td,
            vocab_size=vocab_size, n_mels=80,
            sample_rate=22050, hop_length=256,
        )
        return ADR2(cfg)
    cfg = SoVITSConfig(
        hidden_dim=hd, n_layers=nl, n_heads=nh, ffn_dim=hd * 4,
        vocab_size=vocab_size, content_dim=cd, timbre_dim=td,
        n_mels=80, sample_rate=22050, hop_length=256,
    )
    return SoVITS(cfg)


def _make_dataset(data_dir, n_samples: int):
    """构建限制样本数的 dataset (Trainer 默认加载目录全部 npz)。"""
    from adr.training import VoiceCloneDataset
    return VoiceCloneDataset(npz_dir=data_dir, max_samples=n_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(REPO / "data" / "opencpop_npz" / "train"))
    parser.add_argument("--preset", default="medium", choices=list(PRESET_SIZES.keys()))
    parser.add_argument("--model", default="sovits", choices=["sovits", "adr2"],
                        help="声学模型: sovits=一期, adr2=M9.2 VAE+MAS 骨架")
    parser.add_argument("--n-samples", type=int, default=300,
                        help="取前 N 个样本训练 (smoke)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", default=None,
                        help="从基座 checkpoint 热启动 (finetune), 新增参数随机初始化")
    parser.add_argument("--output", default=str(REPO / "examples" / "opencpop_train"))
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--quantize-4bit", action="store_true",
                        help="真 QLoRA: 4-bit 量化基座 + LoRA (需 CUDA + bitsandbytes)")
    parser.add_argument("--use-amp", action="store_true", help="启用混合精度 (GPU)")
    parser.add_argument("--grad-ckpt", action="store_true", help="启用梯度检查点")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="早停 patience (0=关闭, N=val loss N epoch 不降则停)")
    parser.add_argument("--wav-quality-patience", type=int, default=0,
                        help="wav 早停 patience (0=关闭, N=wav quality N epoch 不升则停)")
    parser.add_argument("--asr-wer-patience", type=int, default=0,
                        help="ASR WER/CER 早停 patience (0=关闭, N=CER N epoch 不降则停)")
    parser.add_argument("--asr-model-size", default="tiny",
                        choices=["tiny", "base", "small", "medium"],
                        help="ASR 模型大小 (默认 tiny, 越大越准但越慢)")
    parser.add_argument("--asr-language", default="zh", help="ASR 语言 (zh/en/...)")
    parser.add_argument("--asr-ensemble", default=None,
                        help="M7 多 ASR 集成, 逗号分隔 (如 'tiny,small'), 覆盖 --asr-model-size")
    parser.add_argument("--warm-restart-patience", type=int, default=0,
                        help="M6 热重启 patience (0=关闭, N=指标 N epoch 不改善则重启 LR)")
    parser.add_argument("--warm-restart-metric", default="asr_cer",
                        choices=["asr_cer", "wav_quality", "val/loss"],
                        help="M6 热重启监控指标 (默认 asr_cer)")
    parser.add_argument("--warm-restart-decay", type=float, default=0.5,
                        help="M6 重启时 LR 衰减系数 (默认 0.5)")
    parser.add_argument("--max-restarts", type=int, default=2,
                        help="M6 最大重启次数 (默认 2)")
    parser.add_argument("--vocoder-path", default=None,
                        help="真实 vocoder 路径 (默认 None=placeholder, 设路径启用 BigVGAN)")
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="验证集比例 (默认 0, 启用早停时建议 0.1)")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    log = get_logger("smoke_opencpop")

    print("=" * 60)
    print(f"OpenCpop Smoke Training ({args.preset})")
    print("=" * 60)
    print(f"  data:     {args.data_dir}")
    print(f"  samples:  {args.n_samples}")
    print(f"  epochs:   {args.epochs}")
    print(f"  LoRA:     {args.use_lora}")
    print(f"  QLoRA:    {args.quantize_4bit}")
    print("=" * 60)

    # 1. 模型
    print(f"\n[1/4] Building model ({args.preset}) ...")
    model = build_model(args.preset, model_type=args.model)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_p/1e6:.1f}M")

    # 1.5 可选 QLoRA (先 LoRA 后量化; Trainer 内部会幂等跳过二次注入)
    use_lora = args.use_lora or args.quantize_4bit
    if args.quantize_4bit:
        import torch
        from adr.training.lora import LoRAConfig, apply_lora
        from adr.training.efficient import is_bnb_available, quantize_4bit
        if not is_bnb_available():
            raise RuntimeError("--quantize-4bit 需要 bitsandbytes")
        if not torch.cuda.is_available():
            raise RuntimeError("--quantize-4bit 需要 CUDA (bnb 4bit 不支持 CPU)")
        apply_lora(model, LoRAConfig(
            rank=8, alpha=16, target_modules=["out_proj", "linear1", "linear2"]))
        model = model.cuda()
        model = quantize_4bit(model)
        print("  [QLoRA] 4-bit 量化基座已应用")

    # 2. 加载真实数据
    print(f"\n[2/4] Loading real OpenCpop data ...")
    pd = load_default_phoneme_dict()

    data_dir = Path(args.data_dir)
    npz_files = sorted(data_dir.glob("*.npz"))[: args.n_samples]
    print(f"  Found {len(npz_files)} npz files")

    # 加载到内存 (smoke 用)
    t0 = time.time()
    train_samples = []
    oov_count = 0
    for npz in npz_files:
        d = np.load(npz, allow_pickle=True)
        # 注意: npz 的 phonemes 是原始音素 (y/v/in/...), 不在音节字典里。
        # 必须用中文 text 走 G2P 音节路径, 否则编码全为空 → 训练无效!
        text = str(d["text"][0])
        ids = encode_phonemes(text, phoneme_dict=pd, strip_tone=True)
        if not ids:
            ids = [pd.encode("<unk>")]
            oov_count += 1
        train_samples.append({
            "sample_id": str(d["sample_id"][0]),
            "phoneme_ids": np.array(ids, dtype=np.int64),
            "mel": d["mel"].astype(np.float32),
            "f0": d["f0"].astype(np.float32),
            "text": text,
        })
    print(f"  Loaded {len(train_samples)} samples in {time.time()-t0:.1f}s")
    print(f"  OOV phonemes: {oov_count}")

    # 打印一个样本
    s = train_samples[0]
    print(f"  sample 0: text='{s['text']}', phonemes={s['phoneme_ids'].shape}, "
          f"mel={s['mel'].shape}, f0 mean={s['f0'][s['f0']>0].mean():.1f}Hz")

    # 3. 训练
    print(f"\n[3/4] Training ({args.epochs} epochs) ...")
    cfg = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=0.0,
        warmup_steps=10,
        grad_clip=1.0,
        use_amp=args.use_amp,
        use_gradient_checkpointing=args.grad_ckpt,
        use_lora=use_lora,
        lora_rank=8,
        lora_alpha=16,
        lora_target_modules=("out_proj", "linear1", "linear2"),
        output_dir=str(output / "train"),
        val_ratio=args.val_ratio,
        early_stop_patience=args.early_stop_patience,
        device="auto",
    )
    trainer = Trainer(model=model, train_data=_make_dataset(data_dir, args.n_samples), config=cfg)

    t0 = time.time()
    # 替换 dataloader 使用我们已经加载的样本 (而不是再走 npz 加载)
    # 可选: 加入 WavQualityCallback 用于 wav-based 早停
    if args.wav_quality_patience > 0:
        from adr.training.callbacks import WavQualityCallback, CheckpointCallback
        # 根据 vocoder_path 决定 placeholder 或真实
        use_placeholder = args.vocoder_path is None
        wav_cb = WavQualityCallback(
            n_samples=5, n_timesteps=10, patience=args.wav_quality_patience,
            use_placeholder_vocoder=use_placeholder,
            vocoder_path=args.vocoder_path,
        )
        # 同时加一个 best-by-wav_quality 的 checkpoint callback
        wq_ckpt = CheckpointCallback(
            save_dir=cfg.output_dir + "/checkpoints",
            save_every_n_epochs=99,  # 不存普通 epoch (避免重复)
            save_best=True,
            metric_name="wav_quality",
            mode="max",
        )
        # 重命名避免和 default 的 best.pt 冲突
        # 这里我们用 best_wav.pt 作区分
        wq_ckpt.best_path_name = "best_wav.pt"
        # Patch _save_best 用 best_wav.pt
        original = wq_ckpt.on_epoch_end
        def _on_epoch_end_save_wav(trainer, epoch, metrics, **kw):
            import pathlib
            if "wav_quality" in metrics and (
                wq_ckpt.best_value is None
                or metrics["wav_quality"] > wq_ckpt.best_value + 0.001
            ):
                wq_ckpt.best_value = metrics["wav_quality"]
                best_path = pathlib.Path(cfg.output_dir) / "checkpoints" / "best_wav.pt"
                trainer.save_checkpoint(best_path)
                print(f"  [Best-Wav] wav_quality={metrics['wav_quality']:.4f} -> best_wav.pt")
        wq_ckpt.on_epoch_end = _on_epoch_end_save_wav
        trainer.callbacks = [wav_cb, wq_ckpt]
        print(f"  [WavQuality] enabled, patience={args.wav_quality_patience}")

    # M4: 可选 ASR WER 早停
    if args.asr_wer_patience > 0:
        from adr.training.callbacks import ASRCallback, CheckpointCallback
        # ASR 需要 val set, 因此若没设, 强制最小 0.1
        if args.val_ratio < 0.05:
            print(f"  [ASR] WARNING: --asr-wer-patience={args.asr_wer_patience} requires val set, "
                  f"自动设置 val_ratio=0.1")
            # 重新构建 trainer with val_ratio
            cfg.val_ratio = 0.1
            trainer = Trainer(model=model, train_data=_make_dataset(data_dir, args.n_samples), config=cfg)
            # 重新设置 wav callbacks (如果之前设过)
            if args.wav_quality_patience > 0:
                trainer.callbacks = [wav_cb, wq_ckpt]
        asr_cb = ASRCallback(
            n_samples=5, n_timesteps=10,
            patience=args.asr_wer_patience,
            asr_model_size=args.asr_model_size,
            asr_model_sizes=(
                [s.strip() for s in args.asr_ensemble.split(",") if s.strip()]
                if args.asr_ensemble else None
            ),
            asr_language=args.asr_language,
            use_real_vocoder=(args.vocoder_path is not None),
            vocoder_path=args.vocoder_path,
        )
        # 同时加一个 best-by-asr_cer 的 checkpoint callback
        asr_ckpt = CheckpointCallback(
            save_dir=cfg.output_dir + "/checkpoints",
            save_every_n_epochs=99,
            save_best=True,
            metric_name="asr_cer",
            mode="min",
        )
        asr_ckpt.best_path_name = "best_asr.pt"

        def _on_epoch_end_save_asr(trainer, epoch, metrics, **kw):
            import pathlib
            if "asr_cer" in metrics and (
                asr_ckpt.best_value is None
                or metrics["asr_cer"] < asr_ckpt.best_value - 0.001
            ):
                asr_ckpt.best_value = metrics["asr_cer"]
                best_path = pathlib.Path(cfg.output_dir) / "checkpoints" / "best_asr.pt"
                trainer.save_checkpoint(best_path)
                print(f"  [Best-ASR] asr_cer={metrics['asr_cer']:.4f} -> best_asr.pt")
        asr_ckpt.on_epoch_end = _on_epoch_end_save_asr
        # 合并到 callbacks
        existing = list(trainer.callbacks) if trainer.callbacks else []
        trainer.callbacks = existing + [asr_cb, asr_ckpt]
        print(f"  [ASR] enabled, patience={args.asr_wer_patience}, "
              f"model={args.asr_model_size}, lang={args.asr_language}"
              + (f", ensemble={args.asr_ensemble}" if args.asr_ensemble else ""))

    # M6: 可选 LR 热重启 (需排在产生指标的 callback 之后)
    if args.warm_restart_patience > 0:
        from adr.training.callbacks import WarmRestartCallback
        # 校验: 指标来源必须已启用
        metric_src = {
            "asr_cer": args.asr_wer_patience > 0,
            "wav_quality": args.wav_quality_patience > 0,
            "val/loss": True,  # val set 有就始终有
        }[args.warm_restart_metric]
        if not metric_src:
            print(f"  [WarmRestart] WARNING: metric '{args.warm_restart_metric}' 的来源 callback 未启用, "
                  f"热重启不会触发 (请先启用对应的 --*-patience)")
        wr_mode = "max" if args.warm_restart_metric == "wav_quality" else "min"
        wr_cb = WarmRestartCallback(
            metric_name=args.warm_restart_metric,
            mode=wr_mode,
            patience=args.warm_restart_patience,
            lr_decay=args.warm_restart_decay,
            max_restarts=args.max_restarts,
        )
        existing = list(trainer.callbacks) if trainer.callbacks else []
        trainer.callbacks = existing + [wr_cb]
        print(f"  [WarmRestart] enabled, metric={args.warm_restart_metric}, "
              f"patience={args.warm_restart_patience}, decay={args.warm_restart_decay}, "
              f"max={args.max_restarts}")

    # Finetune: 从基座 checkpoint 热启动
    if args.resume:
        print(f"  [Resume] loading base: {args.resume}")
        trainer.load_checkpoint(args.resume)

    metrics = trainer.fit()
    train_time = time.time() - t0
    print(f"  [OK] trained in {train_time:.1f}s")
    print(f"  final loss: {metrics.get('train/loss', 'N/A')}")

    # 4. 保存
    print(f"\n[4/4] Saving artifacts ...")
    ckpt_path = output / "final.pt"
    # 优先级: best_asr.pt > best_wav.pt > final.pt (避免过拟合)
    best_asr_ckpt = output / "train" / "checkpoints" / "best_asr.pt"
    best_wav_ckpt = output / "train" / "checkpoints" / "best_wav.pt"
    if args.asr_wer_patience > 0 and best_asr_ckpt.exists():
        import shutil
        shutil.copy(best_asr_ckpt, ckpt_path)
        print(f"  [Note] Using best_asr.pt (ASR CER peak)")
    elif args.wav_quality_patience > 0 and best_wav_ckpt.exists():
        import shutil
        shutil.copy(best_wav_ckpt, ckpt_path)
        print(f"  [Note] Using best_wav.pt (wav quality peak)")
    elif args.use_lora:
        from adr.training.lora import get_lora_state_dict
        state = get_lora_state_dict(model)
        torch.save({
            "lora_state": state,
            "lora_config": {"rank": 8, "alpha": 16},
            "model_class": "SoVITS",
        }, ckpt_path)
    else:
        trainer.save_checkpoint(ckpt_path)

    meta = {
        "preset": args.preset,
        "n_samples": len(train_samples),
        "epochs": args.epochs,
        "train_time_sec": train_time,
        "oov_phonemes": oov_count,
        "model_params_M": n_p / 1e6,
        "use_lora": args.use_lora,
        "final_loss": metrics.get("train/loss"),
        "checkpoint": str(ckpt_path),
    }
    (output / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  saved: {ckpt_path}")
    print(f"  metadata: {output/'metadata.json'}")

    print("\n" + "=" * 60)
    print("[Summary] OpenCpop smoke training")
    print("=" * 60)
    print(f"  Model:     {args.preset} ({n_p/1e6:.1f}M)")
    print(f"  Samples:   {len(train_samples)}")
    print(f"  Time:      {train_time:.1f}s ({train_time/60:.2f} min)")
    print(f"  Throughput: {len(train_samples)/train_time:.1f} samples/sec")
    print(f"  Loss:      {metrics.get('train/loss', 'N/A')}")
    print(f"  LoRA:      {args.use_lora}")
    if args.wav_quality_patience > 0:
        print(f"  Wav ES:    enabled (patience={args.wav_quality_patience})")
    if args.asr_wer_patience > 0:
        print(f"  ASR ES:    enabled (patience={args.asr_wer_patience}, "
              f"model={args.asr_model_size}, lang={args.asr_language})")
    if args.warm_restart_patience > 0:
        print(f"  WarmRest:  enabled (metric={args.warm_restart_metric}, "
              f"patience={args.warm_restart_patience}, max={args.max_restarts})")
    print("=" * 60)


if __name__ == "__main__":
    main()
