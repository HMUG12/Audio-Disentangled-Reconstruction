"""B4: 参考音频旋律桥 — F0 → 音符量化 → DiffSinger word 级输入。

流程: 参考音频 → F0Extractor → 按音节数均分时间轴 → 每段取浊音中位 midi
→ OpenCpop 风格音符名 ("C#4/Db4") + 时长。MVP 采用均分策略
(不做 onset 对齐), 对多数旋律已可用; 后续可加节拍/起始点精化。
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

# midi → OpenCpop 风格音符名 (升号/降号)
_NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def midi_to_note_name(midi: float) -> str:
    """midi 音高 → 'C#4/Db4' 风格 (OpenCpop 标注格式)。"""
    m = int(round(midi))
    octave = m // 12 - 1
    name = _NOTE_SHARP[m % 12]
    flat = _NOTE_FLAT[m % 12]
    return f"{name}{octave}/{flat}{octave}" if name != flat else f"{name}{octave}"


def count_syllables(text: str) -> int:
    """中文歌词音节数 (每汉字 1 音节; 忽略标点/空白/字母串按词数计)。"""
    text = re.sub(r"\s+", "", text)
    n = 0
    for token in re.findall(r"[一-鿿]|[a-zA-Z]+", text):
        n += 1 if re.fullmatch(r"[一-鿿]", token) else 1
    return max(n, 1)


def f0_to_notes(
    f0: np.ndarray,
    n_syllables: int,
    hop_length: int = 256,
    sample_rate: int = 22050,
    min_voiced_ratio: float = 0.3,
) -> Tuple[List[str], List[float]]:
    """F0 轨迹 → (note_seq, note_dur_seq), 长度 = n_syllables。

    策略: 时间轴按音节数均分; 每段浊音帧取 midi 中位数;
    浊音比例过低 → 'rest'。
    """
    f0 = np.asarray(f0, dtype=np.float64)
    T = len(f0)
    total_sec = T * hop_length / sample_rate
    notes: List[str] = []
    durs: List[float] = []
    bounds = np.linspace(0, T, n_syllables + 1).astype(int)
    for i in range(n_syllables):
        seg = f0[bounds[i]:bounds[i + 1]]
        dur = len(seg) * hop_length / sample_rate
        voiced = seg[seg > 0]
        if len(seg) == 0 or len(voiced) / max(len(seg), 1) < min_voiced_ratio:
            notes.append("rest")
            durs.append(max(dur, 0.05))
            continue
        midi = 69 + 12 * np.log2(np.median(voiced) / 440.0)
        notes.append(midi_to_note_name(midi))
        durs.append(round(dur, 5))
    return notes, durs


def build_word_level_input(
    text: str,
    f0: np.ndarray,
    hop_length: int = 256,
    sample_rate: int = 22050,
) -> dict:
    """文本 + F0 → DiffSinger word 级输入字典 (notes/notes_duration '|' 分隔)。"""
    n = count_syllables(text)
    notes, durs = f0_to_notes(f0, n, hop_length, sample_rate)
    return {
        "text": text,
        "notes": " | ".join(notes),
        "notes_duration": " | ".join(f"{d:.5f}" for d in durs),
        "input_type": "word",
    }
