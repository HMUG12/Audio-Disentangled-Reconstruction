"""Debug WavQualityCallback val_loader."""
import sys
sys.path.insert(0, r"E:\新创意构思\新建文件夹\ADR")
import tempfile
from tests.test_wav_quality_callback import make_dataset, make_small_model
from adr.training.trainer import Trainer, TrainerConfig

ds = make_dataset(20)
m = make_small_model()
cfg = TrainerConfig(epochs=1, batch_size=2, val_ratio=0.3,
                    output_dir=tempfile.mkdtemp(),
                    log_every_n_steps=100, save_every_n_epochs=100)
t = Trainer(model=m, train_data=ds, config=cfg)
print(f"val_loader is None: {t.val_loader is None}")
if t.val_loader is not None:
    print(f"val batches: {len(t.val_loader)}")
    for batch in t.val_loader:
        print(f"  batch waveform shape: {batch.waveform.shape}")
        print(f"  has mel attr: {hasattr(batch, 'mel')}")
        break
