"""说话人相似度客观门禁 (ERes2Net, GPT-SoVITS 官方 sv 模型)。

用法:
    from adr.eval.speaker_sim import similarity
    print(similarity("ref.wav", "clone.wav"))   # 0~1, 越高越像

经验区间 (ERes2Net 中文):
    >0.70 明显同一人; 0.55-0.70 接近; <0.45 基本不同人
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GSV_DIR = REPO_ROOT / "third_party" / "gpt_sovits"
ERES2NET_DIR = GSV_DIR / "GPT_SoVITS" / "eres2net"
SV_CKPT = GSV_DIR / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"

# kaldi.py 在 eres2net 目录内, 模块级注入 (embed 里 import 前必须就位)
if str(ERES2NET_DIR) not in sys.path:
    sys.path.insert(0, str(ERES2NET_DIR))

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    import torch
    from ERes2NetV2 import ERes2NetV2
    state = torch.load(SV_CKPT, map_location="cpu", weights_only=False)
    m = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    m.load_state_dict(state)
    m.eval()
    _model = m
    return m


def embed(wav_path: str) -> np.ndarray:
    """音频 → L2 归一化声纹向量。"""
    import soundfile as sf
    import torch
    import kaldi as Kaldi

    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000)
        from torchaudio.functional import resample
        wav = resample(torch.from_numpy(wav), sr // g, 16000 // g).numpy()
    m = _load_model()
    with torch.no_grad():
        feat = Kaldi.fbank(torch.from_numpy(wav).unsqueeze(0),
                           num_mel_bins=80, sample_frequency=16000, dither=0)
        emb = m.forward3(feat.unsqueeze(0))[0].numpy()
    return emb / (np.linalg.norm(emb) + 1e-8)


def similarity(ref_wav: str, test_wav: str) -> float:
    """余弦相似度 [−1,1]。"""
    return float(np.dot(embed(ref_wav), embed(test_wav)))
