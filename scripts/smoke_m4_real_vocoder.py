"""M4 真实验证: BigVGAN vocoder + ASR 早停 (真实 CER 曲线)。

100 samples, small model, 5 epochs. 后台运行, 结果写 m4_real_result.txt。
"""
import sys, time, json
from pathlib import Path

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

import torch
from adr.data.phoneme_dict import load_default_phoneme_dict
from adr.models.sovits import SoVITS, SoVITSConfig
from adr.training import Trainer, TrainerConfig
from adr.training.callbacks import ASRCallback, WavQualityCallback

OUT = REPO / "scripts" / "m4_real_result.txt"

def log(s):
    print(s, flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(s + "\n")

OUT.write_text("", encoding="utf-8")
log("=" * 60)
log("M4 Real Validation: BigVGAN + ASR early stop")
log("=" * 60)

t0 = time.time()
data_dir = REPO / "data" / "opencpop_npz" / "train"

cfg = SoVITSConfig(
    hidden_dim=128, n_layers=2, n_heads=4, ffn_dim=512,
    vocab_size=607, content_dim=128, timbre_dim=64,
    n_mels=80, sample_rate=22050, hop_length=256,
)
model = SoVITS(cfg)
log(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

trcfg = TrainerConfig(
    epochs=5, batch_size=4, lr=2e-4,
    val_ratio=0.2, output_dir=str(REPO/"examples"/"m4_real"/"train"),
    log_every_n_steps=25, save_every_n_epochs=99,
    device="auto",
)
trainer = Trainer(model=model, train_data=data_dir, config=trcfg)

# 只取前 100 个样本 (通过 npz 加载器已自动处理, 这里依靠 trainer 默认加载全部)
# 为控制时长, 直接截断 dataset
try:
    trainer.train_data.samples = trainer.train_data.samples[:80]
    if trainer.val_data is not None:
        trainer.val_data.samples = trainer.val_data.samples[:20]
    log(f"Truncated: train={len(trainer.train_data.samples)}, val={len(trainer.val_data.samples)}")
except Exception as e:
    log(f"Truncate failed ({e}), use full dataset")

wav_cb = WavQualityCallback(
    n_samples=3, n_timesteps=5, patience=3,
    use_placeholder_vocoder=False,
    vocoder_path=r"F:\ADR_data\bigvgan",
)
asr_cb = ASRCallback(
    n_samples=3, n_timesteps=5,
    patience=2, asr_model_size="tiny", asr_language="zh",
    use_real_vocoder=True,
    vocoder_path=r"F:\ADR_data\bigvgan",
)
trainer.callbacks = [wav_cb, asr_cb]

log("[Start training]")
metrics = trainer.fit()
log(f"[Done in {(time.time()-t0)/60:.1f} min]")
log(f"Final train loss: {metrics.get('train/loss')}")

log("\n--- WavQuality history ---")
for h in wav_cb._history:
    log(json.dumps(h, ensure_ascii=False))
log("\n--- ASR history ---")
for h in asr_cb._history:
    log(json.dumps(h, ensure_ascii=False))
log(f"\nasr should_stop={asr_cb.should_stop}, best={asr_cb.best_value}")
log(f"wav should_stop={wav_cb.should_stop}, best={wav_cb.best_value}")
log("DONE")
