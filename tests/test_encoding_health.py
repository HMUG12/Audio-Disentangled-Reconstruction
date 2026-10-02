"""编码健康门禁 (M8.4)。

一期致命回归: 音素编码断裂 (空 ids / 全 UNK) 存活近一周未被发现,
模型"从未见过文本"却训练成功、loss 下降。此测试把该防线焊死:

- collate_samples: npz → phoneme_ids 必须 100% 非空
- UNK 率必须 < 10% (OpenCpop npz 的原始音素应自动回退 text→G2P 路径)
- 短音素序列 (<3) 不得使 DurationPredictor/sample 崩溃
- encode_phonemes 中文路径必须产出非空 ids
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from adr.data.phoneme_dict import encode_phonemes, load_default_phoneme_dict


def _npz_dir():
    from pathlib import Path
    p = Path("data/opencpop_npz/test")
    if not p.exists() or not list(p.glob("*.npz")):
        pytest.skip("OpenCpop npz 不存在")
    return p


class TestEncodingHealth:
    """OpenCpop npz → collate 编码健康 (一期 bug 的回归门禁)。"""

    def test_collate_no_empty_ids(self):
        """collate 后每条样本 phoneme_ids 非空 (一期: 200/200 全空)。"""
        from adr.training import VoiceCloneDataset
        ds = VoiceCloneDataset(npz_dir=str(_npz_dir()), max_samples=100)
        batch = ds.collate(ds.samples[:50])
        ids = batch.phoneme_ids.numpy()
        lens = batch.phoneme_mask.numpy().sum(axis=1)  # mask 有效位 = 实际长度
        assert (lens > 0).all(), "存在空音素序列"
        # 每条至少 3 (DurationPredictor kernel=3 兜底)
        assert (lens >= 3).all(), "存在 <3 的短音素序列"

    def test_collate_unk_rate(self):
        """UNK 率 < 10% (一期: collate 路径 100% UNK)。"""
        from adr.training import VoiceCloneDataset
        ds = VoiceCloneDataset(npz_dir=str(_npz_dir()), max_samples=100)
        batch = ds.collate(ds.samples[:50])
        ids = batch.phoneme_ids.numpy()
        mask = batch.phoneme_mask.numpy()
        unk_id = 3
        total = unk = 0
        for i in range(ids.shape[0]):
            L = int(mask[i].sum())
            total += L
            unk += int((ids[i, :L] == unk_id).sum())
        rate = unk / max(total, 1)
        assert rate < 0.10, f"UNK 率过高: {rate:.1%} ({unk}/{total})"

    def test_g2p_chinese_nonempty(self):
        """中文文本 G2P 编码必须非空且多样 (不全 UNK)。"""
        ids = encode_phonemes("远方的客人请你留下来")
        assert len(ids) >= 5
        assert len(set(ids)) > 2, "编码结果过于单一, 疑似全 UNK"

    def test_short_phoneme_no_crash(self):
        """短音素序列 (1-2 个) 不得使 sample/forward 崩溃 (conv kernel=3)。"""
        from adr.models.sovits import SoVITS, SoVITSConfig
        cfg = SoVITSConfig(hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
                           content_dim=32, timbre_dim=16)
        model = SoVITS(cfg).eval()
        ids = torch.tensor([[5]])           # 长度 1
        ref = torch.randn(1, 80, 100)
        with torch.no_grad():
            mel = model.sample(ids, ref, n_timesteps=2)
        assert mel.dim() == 3
