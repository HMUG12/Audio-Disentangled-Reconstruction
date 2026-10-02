"""音素表加载器 — 复用 DiffSinger 字典格式。

格式: 每行 "音素 <tab> 切分"
例如: 'zhong <tab> zh ong'

核心组件:
- PhonemeDict: 双向 ID ↔ token 查询
- load_default_phoneme_dict(): 一键加载 DiffSinger opencpop-extension.txt (607 音素)
- encode_phonemes(text) -> List[int]: 文本 → 音素 ids (用默认字典)

设计原则:
- 单例 lazy-load: 第一次调用加载,后续直接复用
- 与 content_encoder 严格对齐: PAD=0, BOS=1, EOS=2, UNK=3
- 支持 custom 字典路径(用户可注入第三方字典)

参考:
- DiffSinger: https://github.com/MoonInTheRiver/DiffSinger
- opencpop-extension.txt: 中文 607 音素
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

from adr.core import get_logger


# 特殊 token (固定 ID,与 content_encoder padding_idx=0 对齐)
PAD_TOKEN = "<pad>"      # 0
BOS_TOKEN = "<bos>"      # 1
EOS_TOKEN = "<eos>"      # 2
UNK_TOKEN = "<unk>"      # 3
SP_TOKEN = "SP"          # 4 - 短停顿
AP_TOKEN = "AP"          # 5 - 静音

# 默认字典路径 (DiffSinger opencpop-extension.txt)
DEFAULT_DICT_PATHS = [
    Path(__file__).resolve().parent.parent.parent / "third_party" / "DiffSinger" / "dictionaries" / "opencpop-extension.txt",
    Path(r"F:\ADR_data\dictionaries\opencpop-extension.txt"),
    Path.home() / "BigVGAN" / "dictionaries" / "opencpop-extension.txt",
]


class PhonemeDict:
    """音素表容器,支持 ID ↔ token 双向查询。

    典型用法:
        >>> pd = PhonemeDict.from_opencpop(dict_path)
        >>> pd.vocab_size
        607
        >>> pd.encode("zhong")
        142
        >>> pd.decode(142)
        'zhong'
        >>> pd.split_tokens("zhong")
        ['zh', 'ong']
    """

    def __init__(self, entries: Dict[str, List[str]], name: str = "custom"):
        """entries: 音素 → 切分列表"""
        self.name = name
        # 特殊 token 在前,固定 ID
        specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SP_TOKEN, AP_TOKEN]
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
        self._split[PAD_TOKEN] = [PAD_TOKEN]
        self._split[BOS_TOKEN] = [BOS_TOKEN]
        self._split[EOS_TOKEN] = [EOS_TOKEN]
        self._split[UNK_TOKEN] = [UNK_TOKEN]

    @classmethod
    def from_opencpop(cls, dict_path: Union[str, Path]) -> "PhonemeDict":
        """从 DiffSinger opencpop-extension.txt 加载。"""
        entries: Dict[str, List[str]] = {}
        with open(dict_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                phon, split_str = parts[0], parts[1]
                entries[phon] = split_str.split()
        return cls(entries, name=f"opencpop:{Path(dict_path).name}")

    @property
    def vocab_size(self) -> int:
        return len(self._id2token)

    def encode(self, phoneme: str) -> int:
        """音素 → ID (找不到返回 UNK id)。"""
        return self._token2id.get(phoneme, self._token2id[UNK_TOKEN])

    def decode(self, idx: int) -> str:
        """ID → 音素。"""
        if 0 <= idx < len(self._id2token):
            return self._id2token[idx]
        return UNK_TOKEN

    def split_tokens(self, phoneme: str) -> List[str]:
        """音素 → 切分 (如 'zhong' → ['zh', 'ong'])。"""
        return self._split.get(phoneme, [UNK_TOKEN])

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"PhonemeDict(name={self.name}, vocab_size={self.vocab_size})"


def find_default_dict_path() -> Optional[Path]:
    """搜索默认 DiffSinger 字典路径。"""
    for cand in DEFAULT_DICT_PATHS:
        if cand.exists():
            return cand
    return None


@lru_cache(maxsize=4)
def load_default_phoneme_dict() -> PhonemeDict:
    """加载默认音素字典 (单例, lazy)。

    第一次调用会搜索 DiffSinger 字典,后续直接复用。
    如果找不到,会创建 fallback 字典 (只有 6 个 special token,所有 OOV 映射到 UNK)。
    """
    log = get_logger("adr.data.phoneme_dict")

    path = find_default_dict_path()
    if path is None:
        log.warning(
            "Default phoneme dict not found. Searched: "
            f"{[str(p) for p in DEFAULT_DICT_PATHS]}. "
            "Falling back to special-tokens-only dict (vocab_size=6)."
        )
        return PhonemeDict({}, name="fallback-special-only")

    log.info(f"Loading phoneme dict: {path}")
    pd = PhonemeDict.from_opencpop(path)
    log.info(f"  [OK] {pd}: {pd.vocab_size} entries")
    return pd


def _strip_tone(phoneme: str) -> str:
    """去除拼音末尾的声调数字 ('zhong1' -> 'zhong')。

    pypinyin TONE3 style 输出带声调数字 (e.g. 'zhong1', 'guo2')，
    而 DiffSinger opencpop-extension.txt 字典里是不带声调的 ('zhong', 'guo')。
    这个函数做兼容。
    """
    if not phoneme:
        return phoneme
    # 去除末尾 1-5 的声调数字
    if phoneme[-1].isdigit() and phoneme[-1] in "12345":
        return phoneme[:-1]
    return phoneme


def encode_phonemes(
    text_or_phonemes: Union[str, List[str]],
    phoneme_dict: Optional[PhonemeDict] = None,
    strip_tone: bool = True,
) -> List[int]:
    """文本/音素列表 → 音素 id 列表。

    Args:
        text_or_phonemes: 字符串 (会调 G2P) 或音素列表 (直接 encode)
        phoneme_dict: 字典 (None 用默认)
        strip_tone: 是否去除拼音末尾声调数字 (默认 True, 兼容 DiffSinger dict)

    Returns:
        音素 id 列表

    Examples:
        >>> encode_phonemes(["zhong", "guo"])
        [582, 187]
        >>> encode_phonemes("中国")  # 内部调 G2P + strip tone
        [582, 187]
    """
    pd = phoneme_dict or load_default_phoneme_dict()

    if isinstance(text_or_phonemes, str):
        # 调 G2P 拿音素列表
        from adr.data.g2p import G2P, G2PConfig
        g2p = G2P(G2PConfig(backend="pypinyin", with_tone=True))
        phonemes = g2p(text_or_phonemes)
        if not phonemes:
            return []
    else:
        phonemes = text_or_phonemes

    if strip_tone:
        phonemes = [_strip_tone(p) for p in phonemes]

    return [pd.encode(p) for p in phonemes]


__all__ = [
    "PhonemeDict",
    "load_default_phoneme_dict",
    "encode_phonemes",
    "find_default_dict_path",
    "PAD_TOKEN", "BOS_TOKEN", "EOS_TOKEN", "UNK_TOKEN", "SP_TOKEN", "AP_TOKEN",
]
