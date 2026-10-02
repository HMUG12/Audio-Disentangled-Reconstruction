"""G2P (Grapheme-to-Phoneme) 文本转音素。

后端支持:
- pypinyin: 中文 (简化,无多音字消歧)
- g2pW:    中文 (高级,带多音字消歧,可选)
- char:     字符级 (fallback, 不转音素)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from adr.core import get_logger


@dataclass
class G2PConfig:
    """G2P 配置。"""
    backend: str = "pypinyin"      # pypinyin / g2pW / char
    language: str = "zh"           # zh / en
    with_tone: bool = True         # 中文是否带声调
    use_pos: bool = False          # 是否使用词性 (g2pW 高级)


class G2P:
    """G2P 转换器。"""

    def __init__(self, config: Optional[G2PConfig] = None):
        self.config = config or G2PConfig()
        self.log = get_logger("adr.data.g2p")
        self._pypinyin_initialized = False

    def _ensure_pypinyin(self):
        if self._pypinyin_initialized:
            return
        try:
            from pypinyin import lazy_pinyin, Style
            self._lazy_pinyin = lazy_pinyin
            self._Style = Style
            self._pypinyin_initialized = True
        except ImportError as e:
            raise ImportError(
                "pypinyin not installed. Run: pip install pypinyin"
            ) from e

    def __call__(self, text: str) -> List[str]:
        """转文本为音素列表。

        Returns:
            音素列表
        """
        if not text or not text.strip():
            return []

        if self.config.backend == "pypinyin":
            return self._pinyin_convert(text)
        elif self.config.backend == "char":
            return list(text.strip())
        elif self.config.backend == "g2pW":
            return self._g2pw_convert(text)
        else:
            raise ValueError(f"Unknown G2P backend: {self.config.backend}")

    def _pinyin_convert(self, text: str) -> List[str]:
        """中文转拼音 (pypinyin)。"""
        self._ensure_pypinyin()
        # 过滤非中文字符
        text = "".join(c for c in text if "\u4e00" <= c <= "\u9fff" or c in " ,.!?;:")

        if self.config.with_tone:
            style = self._Style.TONE3
        else:
            style = self._Style.NORMAL

        result = self._lazy_pinyin(text, style=style)
        # 过滤空字符串
        return [p for p in result if p.strip()]

    def _g2pw_convert(self, text: str) -> List[str]:
        """g2pW 转换 (需要额外安装 g2pW)。"""
        try:
            from g2pW import G2PW
        except ImportError:
            self.log.warning("g2pW not installed, falling back to pypinyin")
            self.config.backend = "pypinyin"
            return self._pinyin_convert(text)

        model = G2PW()
        result = model(text)
        # 提取 phonemes
        phonemes = []
        for word, pron in zip(text, result):
            if pron:
                phonemes.extend(pron)
        return phonemes

    @staticmethod
    def is_chinese(text: str) -> bool:
        """判断文本是否主要为中文。"""
        if not text:
            return False
        chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        return chinese_count / len(text) > 0.5

    @staticmethod
    def detect_language(text: str) -> str:
        """自动检测语言。"""
        return "zh" if G2P.is_chinese(text) else "en"
