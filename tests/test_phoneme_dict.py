"""PhonemeDict 单元测试 (风险 1 修复验证)。

覆盖:
- 加载 DiffSinger opencpop-extension.txt
- 音素 encode/decode 双向往返
- 切分 split_tokens
- 特殊 token (PAD/BOS/EOS/UNK/SP/AP) 固定 ID
- tone 数字剥离
- encode_phonemes 一站式 API
- singleton lazy load
- fallback 字典 (当 dict 文件不存在)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_phoneme_dict_load():
    """测试加载默认字典。"""
    from adr.data.phoneme_dict import load_default_phoneme_dict

    pd = load_default_phoneme_dict()
    assert pd.vocab_size == 607, f"Expected 607, got {pd.vocab_size}"
    assert pd.vocab_size == len(pd)
    assert "opencpop" in pd.name.lower()


def test_phoneme_dict_special_tokens():
    """测试特殊 token 固定 ID。"""
    from adr.data.phoneme_dict import (
        load_default_phoneme_dict,
        PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SP_TOKEN, AP_TOKEN,
    )

    pd = load_default_phoneme_dict()
    assert pd.encode(PAD_TOKEN) == 0
    assert pd.encode(BOS_TOKEN) == 1
    assert pd.encode(EOS_TOKEN) == 2
    assert pd.encode(UNK_TOKEN) == 3
    assert pd.encode(SP_TOKEN) == 4
    assert pd.encode(AP_TOKEN) == 5

    # decode 往返
    assert pd.decode(0) == PAD_TOKEN
    assert pd.decode(3) == UNK_TOKEN


def test_phoneme_dict_encode_decode_roundtrip():
    """测试已知音素 encode/decode。"""
    from adr.data.phoneme_dict import load_default_phoneme_dict

    pd = load_default_phoneme_dict()
    # 'zhong' 已知 id 582
    assert pd.encode("zhong") == 582
    assert pd.decode(582) == "zhong"
    # 'guo' 已知 id 187
    assert pd.encode("guo") == 187
    assert pd.decode(187) == "guo"


def test_phoneme_dict_unknown():
    """测试未知音素 → UNK。"""
    from adr.data.phoneme_dict import load_default_phoneme_dict

    pd = load_default_phoneme_dict()
    assert pd.encode("__not_a_phoneme__") == pd.encode("<unk>") == 3
    # decode 越界
    assert pd.decode(999999) == "<unk>"


def test_phoneme_dict_split_tokens():
    """测试音素切分。"""
    from adr.data.phoneme_dict import load_default_phoneme_dict

    pd = load_default_phoneme_dict()
    assert pd.split_tokens("zhong") == ["zh", "ong"]
    assert pd.split_tokens("guo") == ["g", "uo"]
    # 未知音素 fallback
    assert pd.split_tokens("__xx__") == ["<unk>"]


def test_strip_tone():
    """测试声调数字剥离。"""
    from adr.data.phoneme_dict import _strip_tone

    assert _strip_tone("zhong1") == "zhong"
    assert _strip_tone("guo2") == "guo"
    assert _strip_tone("a3") == "a"
    assert _strip_tone("zhong") == "zhong"  # 无声调
    assert _strip_tone("") == ""
    assert _strip_tone("a1") == "a"
    # 不剥离 '5' 之后的字符(只看最后 1 位)
    assert _strip_tone("zhong12") == "zhong1"  # '2' 不是 '12345' 之一? 应该是 '2',strip
    # 实际: 末尾 '2' 属于 '12345' 之一,strip
    assert _strip_tone("zhong12") == "zhong1"


def test_encode_phonemes_chinese():
    """测试中文文本 encode。"""
    from adr.data.phoneme_dict import encode_phonemes, load_default_phoneme_dict

    pd = load_default_phoneme_dict()
    ids = encode_phonemes("中国", phoneme_dict=pd, strip_tone=True)
    # 中国 → zh ong + gu o → 'zhong' + 'guo' → 2 个 ids
    assert len(ids) == 2
    # 都是真实音素 id
    assert all(0 < i < pd.vocab_size for i in ids)
    # 不应该是 UNK
    assert pd.encode("<unk>") not in ids or len(ids) > 2  # 至少部分不是 UNK


def test_encode_phonemes_with_tone_strip():
    """测试带声调的音素也能查表。"""
    from adr.data.phoneme_dict import encode_phonemes, load_default_phoneme_dict

    pd = load_default_phoneme_dict()
    # 直接传带声调的音素列表
    ids = encode_phonemes(["zhong1", "guo2"], phoneme_dict=pd, strip_tone=True)
    assert ids == [pd.encode("zhong"), pd.encode("guo")]


def test_encode_phonemes_no_strip():
    """测试不剥离声调 (OOV 全部变 UNK)。"""
    from adr.data.phoneme_dict import encode_phonemes, load_default_phoneme_dict

    pd = load_default_phoneme_dict()
    # 不 strip, "zhong1" 不在字典里 → UNK
    ids = encode_phonemes(["zhong1"], phoneme_dict=pd, strip_tone=False)
    assert ids == [pd.encode("<unk>")]


def test_encode_phonemes_empty():
    """测试空输入。"""
    from adr.data.phoneme_dict import encode_phonemes

    assert encode_phonemes("") == []
    assert encode_phonemes([]) == []


def test_phoneme_dict_singleton():
    """测试懒加载是单例 (lru_cache 命中)。"""
    from adr.data.phoneme_dict import load_default_phoneme_dict

    pd1 = load_default_phoneme_dict()
    pd2 = load_default_phoneme_dict()
    # 同一对象
    assert pd1 is pd2


def test_phoneme_dict_vs_inference_pipeline():
    """测试 PhonemeDict 与 InferPipeline.text_to_phoneme_ids 一致。"""
    from adr.data.phoneme_dict import load_default_phoneme_dict, encode_phonemes
    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS, SoVITSConfig

    cfg = SoVITSConfig(
        hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
        vocab_size=607, content_dim=16, timbre_dim=16,
    )
    model = SoVITS(cfg)
    pipe = InferPipeline(model, config=InferConfig())

    text = "你好世界"
    pd = load_default_phoneme_dict()

    pipe_ids = pipe.text_to_phoneme_ids(text)
    direct_ids = encode_phonemes(text, phoneme_dict=pd, strip_tone=True)

    # 两者应该一致
    assert pipe_ids == direct_ids, f"Mismatch: {pipe_ids} vs {direct_ids}"


def test_phoneme_dict_in_training_collate():
    """测试 PhonemeDict 集成到训练 collate。"""
    import numpy as np
    from adr.data.pipeline import TrainSample
    from adr.training.dataset import collate_samples

    # 构造 TrainSample 列表
    samples = [
        TrainSample(
            sample_id="test_001",
            waveform=np_random_waveform(8000),
            sample_rate=22050,
            text="中国",
            phonemes=["zhong1", "guo2"],  # 带声调
            f0=np.zeros(50, dtype=np.float32),
            mel=np_random_mel(80, 50),
        ),
        TrainSample(
            sample_id="test_002",
            waveform=np_random_waveform(8000),
            sample_rate=22050,
            text="北京",
            phonemes=["bei3", "jing1"],
            f0=np.zeros(50, dtype=np.float32),
            mel=np_random_mel(80, 50),
        ),
    ]
    batch = collate_samples(samples)

    # phoneme_ids 应该是真实音素 id (范围 [0, 606])
    assert batch.phoneme_ids.min() >= 0
    assert batch.phoneme_ids.max() < 607


# 辅助函数
def np_random_waveform(n: int):
    import numpy as np
    return np.random.randn(n).astype(np.float32) * 0.1


def np_random_mel(n_mels: int, T: int):
    import numpy as np
    return np.random.randn(n_mels, T).astype(np.float32) * 0.1
