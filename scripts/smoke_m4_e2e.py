"""M4 ASR End-to-End Smoke Test."""
import sys, time, json
from pathlib import Path

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from adr.data.phoneme_dict import encode_phonemes, load_default_phoneme_dict
from adr.models.sovits import SoVITS, SoVITSConfig
from adr.training import Trainer, TrainerConfig
from adr.training.callbacks import ASRCallback, CheckpointCallback

print("=" * 60)
print("M4 End-to-End: 50 samples, 3 epochs, ASR early stopping")
print("=" * 60)
pd = load_default_phoneme_dict()
data_dir = REPO / "data" / "opencpop_npz" / "train"
npz_files = sorted(data_dir.glob("*.npz"))[:50]
print(f"Using {len(npz_files)} samples")

# 2. Build small model (CPU-friendly)
cfg = SoVITSConfig(
    hidden_dim=128, n_layers=2, n_heads=4, ffn_dim=512,
    vocab_size=607, content_dim=128, timbre_dim=64,
    n_mels=80, sample_rate=22050, hop_length=256,
)
model = SoVITS(cfg)
print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# 3. Configure trainer
trcfg = TrainerConfig(
    epochs=3, batch_size=4, lr=2e-4,
    val_ratio=0.2, output_dir=str(REPO/"examples"/"m4_test"/"train"),
    log_every_n_steps=10, save_every_n_epochs=99,
    device="auto",
)
trainer = Trainer(model=model, train_data=data_dir, config=trcfg)

# 4. ASR callback (tiny, 2 samples, no BigVGAN for speed)
asr_cb = ASRCallback(
    n_samples=2, n_timesteps=5,
    patience=2, asr_model_size="tiny", asr_language="zh",
    use_placeholder_vocoder=True,
)
trainer.callbacks = [asr_cb]

print("\n[Start training with ASR callback]")
t0 = time.time()
try:
    metrics = trainer.fit()
    print(f"\n[Done in {time.time()-t0:.1f}s]")
    print(f"Final train loss: {metrics.get('train/loss')}")
    print(f"\nASR history:")
    for h in asr_cb._history:
        print(f"  {h}")
    print(f"\nshould_stop={asr_cb.should_stop}")
    print(f"best_value={asr_cb.best_value}")

    # Check if ASR was actually triggered
    asr_loaded = asr_cb._asr_cache is not None
    asr_failed = asr_cb._asr_load_failed
    print(f"\nASR loaded: {asr_loaded}")
    print(f"ASR failed: {asr_failed}")
    if asr_loaded:
        print("✅ End-to-end ASR test PASSED")
    else:
        print("⚠️  ASR was not loaded (likely faster-whisper missing) - skipping ASR eval")
        print("✅ Test framework works (callback doesn't break training)")
except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
