"""Debug WavQualityCallback error."""
import sys
sys.path.insert(0, r"E:\新创意构思\新建文件夹\ADR")
import tempfile
import numpy as np
import torch

from tests.test_wav_quality_callback import make_dataset, make_small_model
from adr.training.trainer import Trainer, TrainerConfig

ds = make_dataset(20)
m = make_small_model()
cfg = TrainerConfig(epochs=1, batch_size=2, val_ratio=0.3,
                    output_dir=tempfile.mkdtemp(),
                    log_every_n_steps=100, save_every_n_epochs=100)
t = Trainer(model=m, train_data=ds, config=cfg)

# 直接调用 forward 看错误
m.eval()
for batch in t.val_loader:
    print("Trying forward...")
    batch_dict = {
        "phoneme_ids": batch.phoneme_ids,
        "phoneme_mask": batch.phoneme_mask,
        "ref_mel": batch.ref_mel,
    }
    print(f"  phoneme_ids shape: {batch.phoneme_ids.shape}")
    print(f"  phoneme_mask shape: {batch.phoneme_mask.shape}, dtype={batch.phoneme_mask.dtype}")
    try:
        out = m(batch_dict)
        print(f"  OK: {out.keys()}")
    except Exception as e:
        import traceback
        print(f"  FAIL: {e}")
        traceback.print_exc()
    break
