"""PyTorch Dataset 封装 (M1 Day 8)。

从 npz 训练样本 → DataLoader 可用 collate。
支持:
- 按 batch 整理变长样本
- padding + mask
- 数据增强 (噪声/速度, M2 启用)

音素编码:
- 默认用 DiffSinger 607 PhonemeDict (与推理 pipeline 一致)
- 可选: 自定义 phoneme_to_id (旧测试兼容)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from adr.data.pipeline import TrainSample


@dataclass
class CollatedBatch:
    """整理后的 batch,字段都是 tensor。"""
    sample_ids: List[str]
    phoneme_ids: torch.Tensor           # (B, T_p_max) int64
    phoneme_mask: torch.Tensor          # (B, T_p_max) bool
    ref_mel: torch.Tensor               # (B, n_mels, T_ref)
    target_mel: torch.Tensor            # (B, n_mels, T_mel_max)
    target_mel_mask: torch.Tensor       # (B, T_mel_max) bool
    target_durations: torch.Tensor      # (B, T_p_max) int64
    texts: List[str]
    f0: torch.Tensor                    # (B, T_mel_max)
    waveform: torch.Tensor              # (B, T_wav_max)


def _pad_1d(x: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    """1D 数组右对齐 padding。"""
    if len(x) >= max_len:
        return x[:max_len]
    pad = np.full(max_len - len(x), pad_value, dtype=x.dtype)
    return np.concatenate([x, pad])


def _pad_2d(x: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
    """2D 数组 (D, T) 在最后一维 padding。"""
    if x.shape[-1] >= target_len:
        return x[..., :target_len]
    pad_shape = list(x.shape)
    pad_shape[-1] = target_len - x.shape[-1]
    pad = np.full(pad_shape, pad_value, dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1)


def _strip_tone(phoneme: str) -> str:
    """去除拼音末尾声调数字 ('zhong1' -> 'zhong')。"""
    if phoneme and phoneme[-1].isdigit() and phoneme[-1] in "12345":
        return phoneme[:-1]
    return phoneme


def collate_samples(
    samples: Sequence[TrainSample],
    phoneme_to_id: Optional[dict] = None,
) -> CollatedBatch:
    """把一组变长 TrainSample 整理成 batch tensor。

    Args:
        samples: TrainSample 列表
        phoneme_to_id: 音素 → id 映射 (None 则用 DiffSinger PhonemeDict)
    """
    B = len(samples)
    if B == 0:
        raise ValueError("Empty sample list")

    # 1. 音素编码 — 优先用真实 PhonemeDict
    if phoneme_to_id is None:
        try:
            from adr.data.phoneme_dict import load_default_phoneme_dict
            pd = load_default_phoneme_dict()
            phoneme_to_id = {tok: pd.encode(tok) for tok in pd._id2token}
        except Exception:
            # fallback: 局部构建
            phoneme_to_id = {}
            for s in samples:
                for p in s.phonemes:
                    if p not in phoneme_to_id:
                        phoneme_to_id[p] = len(phoneme_to_id) + 1  # 0 = padding

    phoneme_ids_list = []
    unk_id = phoneme_to_id.get("<unk>", 3)
    for s in samples:
        ids = []
        for p in s.phonemes:
            p_clean = _strip_tone(p)
            ids.append(phoneme_to_id.get(p_clean, unk_id))
        # OpenCpop npz 的 phonemes 是原始音素 (y/v/in/...), 不在音节字典里
        # (≥50% UNK) → 回退用中文 text 走 G2P 音节路径
        if s.text and ids and sum(i == unk_id for i in ids) * 2 >= len(ids):
            try:
                from adr.data.phoneme_dict import encode_phonemes
                ids_from_text = encode_phonemes(s.text)
                if ids_from_text:
                    ids = ids_from_text
            except Exception:
                pass
        if not ids:
            ids = [0]
        # DurationPredictor conv kernel=3, 音素序列至少 3 帧
        sp_id = phoneme_to_id.get("SP", 0)
        while len(ids) < 3:
            ids.append(sp_id)
        phoneme_ids_list.append(np.array(ids, dtype=np.int64))

    # 2. 计算 target duration (phoneme 帧数) — 从 mel 长度 + phoneme 长度估算
    target_durations_list = []
    for s, p_ids in zip(samples, phoneme_ids_list):
        if s.mel is not None and len(p_ids) > 0:
            total_dur = s.mel.shape[-1]
            avg = total_dur / len(p_ids)
            durs = np.full(len(p_ids), max(1, int(round(avg))), dtype=np.int64)
        else:
            durs = np.ones(len(p_ids), dtype=np.int64)
        target_durations_list.append(durs)

    # 3. 取最大长度
    max_phoneme_len = max(len(p) for p in phoneme_ids_list)
    max_phoneme_len = max(max_phoneme_len, 8)
    max_mel_len = max((s.mel.shape[-1] if s.mel is not None else 0) for s in samples)
    max_mel_len = max(max_mel_len, 64)
    max_wav_len = max(len(s.waveform) for s in samples)
    max_wav_len = max(max_wav_len, 1024)
    ref_mel_T = max((s.mel.shape[-1] if s.mel is not None else 0) for s in samples) // 2 + 16

    # 4. 填充
    phoneme_ids = np.zeros((B, max_phoneme_len), dtype=np.int64)
    phoneme_mask = np.zeros((B, max_phoneme_len), dtype=np.bool_)
    target_durations = np.zeros((B, max_phoneme_len), dtype=np.int64)
    target_mel = np.zeros((B, 80, max_mel_len), dtype=np.float32)
    target_mel_mask = np.zeros((B, max_mel_len), dtype=np.bool_)
    ref_mel = np.zeros((B, 80, ref_mel_T), dtype=np.float32)
    f0 = np.zeros((B, max_mel_len), dtype=np.float32)
    waveform = np.zeros((B, max_wav_len), dtype=np.float32)

    sample_ids = []
    texts = []

    for i, (s, p_ids, durs) in enumerate(zip(samples, phoneme_ids_list, target_durations_list)):
        n_p = len(p_ids)
        phoneme_ids[i, :n_p] = p_ids
        phoneme_mask[i, :n_p] = True
        target_durations[i, :n_p] = durs

        if s.mel is not None:
            T = s.mel.shape[-1]
            target_mel[i, :, :T] = s.mel
            target_mel_mask[i, :T] = True
            # ref_mel 取前半段 (简化)
            rT = min(T, ref_mel_T)
            ref_mel[i, :, :rT] = s.mel[:, :rT]
        if s.f0 is not None and len(s.f0) > 0:
            T = min(len(s.f0), max_mel_len)
            f0[i, :T] = s.f0[:T]
        waveform[i, :len(s.waveform)] = s.waveform

        sample_ids.append(s.sample_id)
        texts.append(s.text)

    return CollatedBatch(
        sample_ids=sample_ids,
        phoneme_ids=torch.from_numpy(phoneme_ids),
        phoneme_mask=torch.from_numpy(phoneme_mask),
        ref_mel=torch.from_numpy(ref_mel),
        target_mel=torch.from_numpy(target_mel),
        target_mel_mask=torch.from_numpy(target_mel_mask),
        target_durations=torch.from_numpy(target_durations),
        texts=texts,
        f0=torch.from_numpy(f0),
        waveform=torch.from_numpy(waveform),
    )


class VoiceCloneDataset(Dataset):
    """训练样本 Dataset。

    支持两种模式:
    1. from_npz_dir: 从 output/processed/samples/*.npz 加载
    2. from_samples: 直接传入 TrainSample 列表 (用于测试)
    """

    def __init__(
        self,
        npz_dir: Optional[Union[str, Path]] = None,
        samples: Optional[List[TrainSample]] = None,
        max_samples: Optional[int] = None,
    ):
        if samples is not None:
            self.samples = samples
        elif npz_dir is not None:
            self.samples = self._load_npz_dir(npz_dir)
        else:
            raise ValueError("Either npz_dir or samples must be provided")

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        # 构音素词典
        self.phoneme_to_id = self._build_phoneme_dict(self.samples)
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}

    @staticmethod
    def _load_npz_dir(npz_dir: Union[str, Path]) -> List[TrainSample]:
        """加载目录下所有 .npz 训练样本。"""
        npz_dir = Path(npz_dir)
        if not npz_dir.exists():
            raise FileNotFoundError(f"npz dir not found: {npz_dir}")

        samples = []
        for npz_file in sorted(npz_dir.glob("*.npz")):
            try:
                s = TrainSample.load(npz_file)
                if s.text and s.phonemes:
                    samples.append(s)
            except Exception as e:
                print(f"  [!] Failed to load {npz_file.name}: {e}")

        return samples

    @staticmethod
    def _build_phoneme_dict(samples: List[TrainSample]) -> dict:
        """构音素词典 (0 = padding)。

        优先用 DiffSinger PhonemeDict (607 音素) — 与模型 vocab_size 对齐。
        """
        try:
            from adr.data.phoneme_dict import load_default_phoneme_dict
            pd = load_default_phoneme_dict()
            # 真实 PhonemeDict 编码: <pad>=0, <unk>=1, <sos>=2, <eos>=3, ...
            d = {"<pad>": 0}
            for tok in pd._id2token:
                d[tok] = pd.encode(tok)
            return d
        except Exception:
            # fallback: 局部构建
            d = {"<pad>": 0, "<unk>": 1}
            for s in samples:
                for p in s.phonemes:
                    if p not in d:
                        d[p] = len(d)
            return d

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrainSample:
        return self.samples[idx]

    def collate(self, batch: Sequence[TrainSample]) -> CollatedBatch:
        return collate_samples(batch, phoneme_to_id=self.phoneme_to_id)

    def split(self, val_ratio: float = 0.1, seed: int = 42) -> tuple["VoiceCloneDataset", "VoiceCloneDataset"]:
        """划分 train/val。"""
        rng = random.Random(seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_ds = VoiceCloneDataset.__new__(VoiceCloneDataset)
        train_ds.samples = [self.samples[i] for i in train_idx]
        train_ds.phoneme_to_id = self.phoneme_to_id
        train_ds.id_to_phoneme = self.id_to_phoneme

        val_ds = VoiceCloneDataset.__new__(VoiceCloneDataset)
        val_ds.samples = [self.samples[i] for i in val_idx]
        val_ds.phoneme_to_id = self.phoneme_to_id
        val_ds.id_to_phoneme = self.id_to_phoneme

        return train_ds, val_ds

    def __repr__(self) -> str:
        return f"VoiceCloneDataset(n={len(self.samples)}, vocab={len(self.phoneme_to_id)})"
