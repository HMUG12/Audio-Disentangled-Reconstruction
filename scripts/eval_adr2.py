"""ADR2 checkpoint 快速诊断 (M9 迭代用)。

输出: MAS 时长分布 / teacher-forced corr / 采样 T 与统计 / 对齐健康度判定。

用法: python scripts/eval_adr2.py <ckpt> [--save-png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate(ckpt: str, n_samples: int = 4) -> dict:
    from adr.inference.pipeline import InferPipeline
    from adr.training import VoiceCloneDataset

    pipe = InferPipeline.from_checkpoint(ckpt)
    model = pipe.backbone.eval()
    dev = pipe.device

    ds = VoiceCloneDataset(npz_dir="data/opencpop_npz/test",
                           max_samples=max(16, n_samples))
    batch = ds.collate(ds.samples[:n_samples])
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
        out = model(bd)

    pred = out["pred_mel"].cpu().numpy()
    gt = batch.target_mel.numpy()
    mask = batch.target_mel_mask.numpy()
    corrs, l1s = [], []
    for i in range(len(pred)):
        T = int(mask[i].sum())
        if pred[i][:, :T].std() < 1e-8:
            continue
        corrs.append(np.corrcoef(pred[i][:, :T].flatten(),
                                 gt[i][:, :T].flatten())[0, 1])
        l1s.append(np.abs(pred[i][:, :T] - gt[i][:, :T]).mean())

    # 对齐健康度: 最大时长占比 (健康应 <30%, 退化则 >70%)
    dur = out["mas_durations"].float()
    max_ratio = (dur.max(dim=1).values / dur.sum(dim=1).clamp(min=1)).mean().item()

    # 采样
    mel = model.sample(bd["phoneme_ids"][:1], bd["ref_mel"][:1])
    sample_T = mel.size(-1)

    res = {
        "ckpt": ckpt,
        "tf_corr": round(float(np.mean(corrs)), 3) if corrs else None,
        "tf_l1": round(float(np.mean(l1s)), 4) if l1s else None,
        "mas_max_ratio": round(max_ratio, 3),
        "mas_dur_sample": out["mas_durations"][0].tolist(),
        "sample_T": sample_T,
        "sample_std": round(mel.std().item(), 2),
        "healthy_alignment": max_ratio < 0.4,
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    r = evaluate(args.ckpt, args.n)
    print(f"ckpt:           {r['ckpt']}")
    print(f"TF corr:        {r['tf_corr']}  (越高越好)")
    print(f"TF L1:          {r['tf_l1']}")
    print(f"MAS 最大时长占比: {r['mas_max_ratio']}  ({'<0.4 健康' if r['healthy_alignment'] else '>=0.4 退化!'})")
    print(f"MAS dur[0]:     {r['mas_dur_sample']}")
    print(f"sample T:       {r['sample_T']}  std: {r['sample_std']}")


if __name__ == "__main__":
    main()
