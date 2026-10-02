"""M0 修复脚本：生成 1 秒静音 prompt 音频，供重跑推理时用 --prompt-audio 指定。

用法：
  python scripts/m0_make_prompt.py

输出：output/silent_prompt.wav（1 秒静音，44.1kHz 单声道）
"""
from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path("output/silent_prompt.wav")
SR = 44100
DURATION = 1.0
# 极低幅度的白噪（避免纯静音触发异常，同时不引入语义内容）
amp = 0.001
samples = (np.random.randn(int(SR * DURATION)).astype(np.float32) * amp)
sf.write(OUT, samples, SR)
print(f"Saved {OUT} ({len(samples)} samples @ {SR}Hz = {DURATION}s)")
