"""预测 mel 仿射校准探针: per-bin 线性校正 (a*pred+b) 拟合到 GT mel。

用法: python scripts/calibrate_mel.py <ckpt> [--n 300] [--out adr/vocoder/mel_calib.npz]
产出: per-bin 增益/偏置 + 校准前后 ASR 对照。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adr.inference.pipeline import InferPipeline
from adr.training import VoiceCloneDataset


def fit_calibration(ckpt: str, n: int = 300, out: str = "adr/vocoder/mel_calib.npz"):
    pipe = InferPipeline.from_checkpoint(ckpt)
    model = pipe.backbone.eval()
    dev = pipe.device
    ds = VoiceCloneDataset(npz_dir="data/opencpop_npz/train", max_samples=n)

    preds, gts = [], []
    bs = 8
    for i in range(0, n, bs):
        batch = ds.collate(ds.samples[i:i + bs])
        bd = {
            "phoneme_ids": batch.phoneme_ids.to(dev),
            "phoneme_mask": batch.phoneme_mask.to(dev),
            "ref_mel": batch.ref_mel.to(dev),
            "target_mel": batch.target_mel.to(dev),
            "target_mel_mask": batch.target_mel_mask.to(dev),
            "target_durations": batch.target_durations.to(dev),
            "f0": batch.f0.to(dev),
        }
        with torch.no_grad():
            res = model(bd)
        pred = res["pred_mel"].cpu().numpy()
        gt = batch.target_mel.numpy()
        mask = batch.target_mel_mask.numpy()
        for j in range(len(pred)):
            T = int(mask[j].sum())
            preds.append(pred[j][:, :T])
            gts.append(gt[j][:, :T])
        if (i // bs) % 10 == 0:
            print(f"  {i}/{n}")

    P = np.concatenate(preds, axis=1)  # (80, T_total)
    G = np.concatenate(gts, axis=1)
    # per-bin 最小二乘: g ≈ a*p + b
    a = np.zeros(80, dtype=np.float32)
    b = np.zeros(80, dtype=np.float32)
    for k in range(80):
        p_k, g_k = P[k], G[k]
        A = np.stack([p_k, np.ones_like(p_k)], axis=1)
        sol, *_ = np.linalg.lstsq(A, g_k, rcond=None)
        a[k], b[k] = sol
    np.savez(out, gain=a, bias=b)
    print(f"[OK] calibration saved: {out}")
    print(f"  gain mean={a.mean():.3f} std={a.std():.3f} "
          f"bias mean={b.mean():.3f} std={b.std():.3f}")
    # 校准前后 L1
    l1_before = np.abs(P - G).mean()
    l1_after = np.abs(a[:, None] * P + b[:, None] - G).mean()
    print(f"  L1 before={l1_before:.4f} after={l1_after:.4f}")
    return a, b


def probe_asr(ckpt: str, calib: str):
    """校准前后: test 样本 TF pred mel → vocoder → ASR 对照。"""
    import soundfile as sf
    from adr.data.asr import ASR, ASRConfig

    pipe = InferPipeline.from_checkpoint(ckpt)
    model = pipe.backbone.eval()
    dev = pipe.device
    c = np.load(calib)
    gain = torch.from_numpy(c["gain"]).to(dev).view(1, 80, 1)
    bias = torch.from_numpy(c["bias"]).to(dev).view(1, 80, 1)

    ds = VoiceCloneDataset(npz_dir="data/opencpop_npz/test", max_samples=4)
    batch = ds.collate(ds.samples[:4])
    bd = {
        "phoneme_ids": batch.phoneme_ids.to(dev),
        "phoneme_mask": batch.phoneme_mask.to(dev),
        "ref_mel": batch.ref_mel.to(dev),
        "target_mel": batch.target_mel.to(dev),
        "target_mel_mask": batch.target_mel_mask.to(dev),
        "target_durations": batch.target_durations.to(dev),
        "f0": batch.f0.to(dev),
    }
    asr = ASR(ASRConfig(model_size="tiny", device="cpu", compute_type="int8"))
    with torch.no_grad():
        res = model(bd)
        # 自由采样 (非 TF) 更贴近真实使用
        mel_free = model.sample(bd["phoneme_ids"][:1], bd["ref_mel"][:1],
                                f0=bd["f0"][:1])

    i = 0
    T = int(batch.target_mel_mask[0].sum())
    variants = {
        "free_raw": mel_free,
        "free_calib": mel_free * gain + bias,
        "tf_raw": res["pred_mel"][:1, :, :T],
        "tf_calib": res["pred_mel"][:1, :, :T] * gain + bias,
    }
    print(f"文本: {batch.texts[0]}")
    for name, mel in variants.items():
        w = pipe.vocoder.infer(mel.float()).squeeze().cpu().numpy()
        p = f"output/calib_probe_{name}.wav"
        sf.write(p, w, 22050)
        print(f"  {name:>12}: ASR={asr.transcribe(p).text!r}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="adr/vocoder/mel_calib.npz")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()
    if not args.probe_only:
        fit_calibration(args.ckpt, args.n, args.out)
    probe_asr(args.ckpt, args.out)
