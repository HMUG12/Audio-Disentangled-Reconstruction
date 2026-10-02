"""音素表加载器 — 复用 DiffSinger 字典格式。

格式：每行 "音素 <tab> 切分"
例如：'zhong <tab> zh ong'
"""
from pathlib import Path
from typing import Dict, List


class PhonemeDict:
    """音素表容器，支持 ID ↔ token 双向查询。"""

    PAD = "<pad>"       # 0
    BOS = "<bos>"       # 1
    EOS = "<eos>"       # 2
    UNK = "<unk>"       # 3
    SP = "SP"           # 4 - 短停顿
    AP = "AP"           # 5 - 静音

    def __init__(self, entries: Dict[str, List[str]]):
        """entries: 音素 → 切分列表"""
        # 特殊 token 在前，固定 ID
        specials = [self.PAD, self.BOS, self.EOS, self.UNK, self.SP, self.AP]
        self._id2token: List[str] = list(specials)
        self._token2id: Dict[str, int] = {t: i for i, t in enumerate(specials)}
        # 业务音素
        for phon in entries:
            if phon in self._token2id:
                continue
            self._token2id[phon] = len(self._id2token)
            self._id2token.append(phon)
        # 切分表
        self._split: Dict[str, List[str]] = dict(entries)
        # 给特殊 token 自己的切分
        self._split[self.PAD] = [self.PAD]
        self._split[self.BOS] = [self.BOS]
        self._split[self.EOS] = [self.EOS]
        self._split[self.UNK] = [self.UNK]

    @classmethod
    def from_opencpop(cls, dict_path: str | Path) -> "PhonemeDict":
        """从 DiffSinger opencpop-extension.txt 加载。"""
        entries: Dict[str, List[str]] = {}
        with open(dict_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    parts = line.split()  # 兼容空格分隔
                    if len(parts) != 2:
                        continue
                phon, split_str = parts[0], parts[1]
                entries[phon] = split_str.split()
        return cls(entries)

    @property
    def vocab_size(self) -> int:
        return len(self._id2token)

    def encode(self, phoneme: str) -> int:
        return self._token2id.get(phoneme, self._token2id[self.UNK])

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self._id2token):
            return self._id2token[idx]
        return self.UNK

    def split_tokens(self, phoneme: str) -> List[str]:
        return self._split.get(phoneme, [self.UNK])

    def __len__(self) -> int:
        return self.vocab_size


if __name__ == "__main__":
    import sys
    dict_path = Path(__file__).resolve().parent.parent.parent / "third_party" / "DiffSinger" / "dictionaries" / "opencpop-extension.txt"
    pd = PhonemeDict.from_opencpop(dict_path)
    print(f"Loaded {pd.vocab_size} phonemes (incl. 6 specials)")
    print(f"  PAD={pd.encode(PhonemeDict.PAD)}, BOS={pd.encode(PhonemeDict.BOS)}, EOS={pd.encode(PhonemeDict.EOS)}, UNK={pd.encode(PhonemeDict.UNK)}")
    print(f"  SP={pd.encode(PhonemeDict.SP)}, AP={pd.encode(PhonemeDict.AP)}")
    print(f"  'zhong' -> id {pd.encode('zhong')} -> '{pd.decode(pd.encode('zhong'))}'")
    print(f"  'zhong' split = {pd.split_tokens('zhong')}")
    print(f"  sample ids 0..10: {[pd.decode(i) for i in range(11)]}")
