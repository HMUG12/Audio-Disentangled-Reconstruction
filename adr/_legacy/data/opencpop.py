"""OpenCpop 数据加载器（兼容 mock 模式）。

数据格式（真实 OpenCpop）：
  F:/ADR_data/opencpop/
    wavs/                 # 100 个 wav, 44.1kHz 24bit 单声道
      001.wav ~ 100.wav
    transcriptions.txt    # 词级 + 音素级标注 (Praat TextGrid)
  或 DiffSinger 预处理后：
    data/opencpop/
      train.txt           # 每行：wav 路径|词序列|音素序列|音符序列|音符时长
      val.txt
      test.txt
"""
import csv
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class OpenCpopItem:
    """单条 OpenCpop 样本。"""

    def __init__(self, utt_id: str, wav_path: str,
                 words: List[str], phonemes: List[str],
                 notes: List[str], note_durs: List[float]):
        self.utt_id = utt_id
        self.wav_path = wav_path
        self.words = words
        self.phonemes = phonemes
        self.notes = notes  # 如 "C#4/Db4"
        self.note_durs = note_durs  # 秒

    def __repr__(self):
        return f"OpenCpopItem({self.utt_id}, n_phon={len(self.phonemes)}, n_notes={len(self.notes)})"


def parse_textgrid_line(line: str) -> Tuple[str, List[str], List[str], List[float]]:
    """简单解析单行标注 (仅占位，真实用 textgrid 库)。
    格式：'001|小 酒 窝|...|C#4 D#4|0.3 0.4'
    """
    parts = line.strip().split("|")
    if len(parts) < 5:
        return "", [], [], []
    utt_id = parts[0]
    words = parts[1].split()
    phonemes = parts[2].split()
    notes = parts[3].split()
    note_durs = [float(x) for x in parts[4].split()]
    return utt_id, words, phonemes, notes, note_durs


def midi_from_note_str(note: str) -> int:
    """将 'C#4/Db4' 转为 MIDI 编号。
    简单实现：取第一个音名 + 解析八度 + 半音偏移。
    """
    if "/" in note:
        note = note.split("/")[0]
    if note in ("rest", "SP", "AP"):
        return 0
    # 音名 → 半音偏移
    pitch_map = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
                 "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    name = note[:-1] if note[-1].isdigit() else note[:-2]
    octave = int(note[len(name):])
    return 12 * (octave + 1) + pitch_map.get(name, 0)


def load_opencpop_manifest(data_root: Path) -> List[OpenCpopItem]:
    """从 train.txt 加载清单（DiffSinger 格式）。"""
    items = []
    manifest = data_root / "train.txt"
    if not manifest.exists():
        return []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue
            utt_id, wav_path, words, phonemes, notes, durs = parts[:6]
            try:
                note_durs = [float(x) for x in durs.split()]
            except ValueError:
                continue
            items.append(OpenCpopItem(utt_id, wav_path, words.split(), phonemes.split(),
                                      notes.split(), note_durs))
    return items


class OpenCpopDataset(Dataset):
    """OpenCpop 数据集 PyTorch 加载器（CPU 端）。

    返回 dict：
      - phoneme_ids: (T_p,)
      - pitch_ids: (T_n,)
      - note_duration: (T_n,)
      - velocity: (T_n,)
      - is_singing: bool  (固定 True)
    mel 与 ref_mel 留待 mel 提取步骤
    """

    def __init__(self, data_root: str | Path, phoneme_dict,
                 max_phonemes: int = 256, max_notes: int = 128,
                 synthetic: bool = False, n_synthetic: int = 100,
                 sr: int = 44100, hop_size: int = 512):
        """
        synthetic=True 时用 mock 数据（不依赖真实数据集）
        """
        self.data_root = Path(data_root)
        self.pd = phoneme_dict
        self.max_phonemes = max_phonemes
        self.max_notes = max_notes
        self.sr = sr
        self.hop_size = hop_size

        if synthetic:
            self.items = self._build_synthetic(n_synthetic)
        else:
            self.items = load_opencpop_manifest(self.data_root)
            if not self.items:
                print(f"[OpenCpopDataset] No manifest found at {self.data_root}/train.txt")
                print("  → Falling back to synthetic mode (n=10) for smoke testing")
                self.items = self._build_synthetic(10)

    def _build_synthetic(self, n: int) -> List[OpenCpopItem]:
        """生成 mock OpenCpop 样本（用于骨架测试）。"""
        # 真实 OpenCpop 平均每首约 200-400 音素
        phoneme_pool = list(self.pd._token2id.keys())[6:30]  # 取部分业务音素
        items = []
        for i in range(n):
            n_p = random.randint(50, 200)
            n_n = random.randint(40, 150)
            phons = [random.choice(phoneme_pool) for _ in range(n_p)]
            notes = [random.choice(["C4", "D4", "E4", "F4", "G4", "A4"]) for _ in range(n_n)]
            durs = [random.uniform(0.1, 0.5) for _ in range(n_n)]
            items.append(OpenCpopItem(
                utt_id=f"synthetic_{i:04d}",
                wav_path="",
                words=[],
                phonemes=phons,
                notes=notes,
                note_durs=durs,
            ))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _encode_phoneme(self, phon: str) -> int:
        # 用切分后的第一个音素（粗略）
        splits = self.pd.split_tokens(phon)
        return self.pd.encode(splits[0] if splits else self.pd.UNK)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        # 音素编码
        phoneme_ids = [self._encode_phoneme(p) for p in item.phonemes[:self.max_phonemes]]
        phoneme_ids = [self.pd.BOS_ID] + phoneme_ids + [self.pd.EOS_ID]
        phoneme_ids = phoneme_ids[:self.max_phonemes]
        # 旋律编码
        pitch_ids = [midi_from_note_str(n) for n in item.notes[:self.max_notes]]
        note_durs = item.note_durs[:self.max_notes]
        # 速度随机（mock）
        velocity = [random.randint(60, 100) % 32 for _ in pitch_ids]
        # 补零
        while len(phoneme_ids) < self.max_phonemes:
            phoneme_ids.append(self.pd.PAD_ID)
        while len(pitch_ids) < self.max_notes:
            pitch_ids.append(0)
            note_durs.append(0.0)
            velocity.append(0)
        return {
            "utt_id": item.utt_id,
            "phoneme_ids": torch.tensor(phoneme_ids, dtype=torch.long),
            "phoneme_mask": torch.tensor([1 if p != self.pd.PAD_ID else 0
                                          for p in phoneme_ids], dtype=torch.bool),
            "pitch_ids": torch.tensor(pitch_ids, dtype=torch.long),
            "note_duration": torch.tensor(note_durs, dtype=torch.float),
            "velocity": torch.tensor(velocity, dtype=torch.long),
            "is_singing": True,
        }


# 给 PhonemeDict 添加 ID 常量（避免调用 .encode 字典的 hack）
def _patch_phoneme_dict_ids(pd_cls):
    pd_cls.PAD_ID = 0
    pd_cls.BOS_ID = 1
    pd_cls.EOS_ID = 2
    pd_cls.UNK_ID = 3
    pd_cls.SP_ID = 4
    pd_cls.AP_ID = 5


if __name__ == "__main__":
    from adr.data.phoneme_dict import PhonemeDict
    _patch_phoneme_dict_ids(PhonemeDict)

    dict_path = Path(__file__).resolve().parent.parent.parent / "third_party" / "DiffSinger" / "dictionaries" / "opencpop-extension.txt"
    pd = PhonemeDict.from_opencpop(dict_path)
    ds = OpenCpopDataset(data_root="F:/ADR_data/opencpop", phoneme_dict=pd, synthetic=True, n_synthetic=20)
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"Sample 0:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
    print()
    print("Batch test:")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}")
