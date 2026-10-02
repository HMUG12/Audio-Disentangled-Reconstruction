"""Sanity check: 用真实 OpenCpop wav 走 ASRCallback 的 ASR 路径。

若真实音频 CER 低 (<0.3), 证明 ASR 管线正确, e2e 中 CER=1.0
纯粹是模型质量问题 (80 samples × 3 epochs 的 2.4M 未收敛模型)。
"""
import sys
from pathlib import Path

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
from adr.training.callbacks import ASRCallback, compute_cer

cb = ASRCallback(asr_model_size="tiny", asr_language="zh", use_real_vocoder=False)

# 真实 wav + 真实 text
wav_dir = REPO / "data" / "opencpop" / "wavs"
trans_file = REPO / "data" / "opencpop" / "transcriptions.txt"
trans = {}
for line in trans_file.read_text(encoding="utf-8").splitlines():
    parts = line.split("|")
    if len(parts) >= 2:
        trans[parts[0].strip()] = parts[1].strip()  # 第 2 列是汉字文本

n = 0
for wav_path in sorted(wav_dir.glob("*.wav"))[:3]:
    utt = wav_path.stem
    ref = trans.get(utt, "")
    if not ref:
        continue
    wav, sr = sf.read(str(wav_path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    hyp = cb._asr_transcribe(wav.astype(np.float32), sr=sr)
    cer = compute_cer(ref, hyp)
    print(f"utt={utt} sr={sr}")
    print(f"  ref: {ref}")
    print(f"  hyp: {hyp}")
    print(f"  CER: {cer:.4f}")
    n += 1

print(f"\nChecked {n} real utterances")
print("PASS: ASR pipeline works" if n > 0 else "FAIL: no utterances checked")
