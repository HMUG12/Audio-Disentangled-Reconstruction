"""诊断 output/codes_0.npy 是否包含 EOS / 是否陷入重复循环。

不重跑推理——直接分析已有输出。结论用于修复 M0 异常。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "fish-speech"))
from fish_speech.tokenizer import IM_END_TOKEN  # noqa: E402

CODES_NPY = Path("output/codes_0.npy")
codes = np.load(CODES_NPY)
T = codes.shape[1]

# 通过 Fish Speech tokenizer 查 im_end ID
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "fish-speech"))
from fish_speech.tokenizer import FishTokenizer  # noqa: E402

t = FishTokenizer.from_pretrained("third_party/fish-speech/checkpoints/openaudio-s1-mini")
im_end_id = t.get_token_id(IM_END_TOKEN)
print("im_end token:", IM_END_TOKEN, "id:", im_end_id)
print("semantic range:", t.semantic_begin_id, "->", t.semantic_end_id)
print()

# 检查 codebook 0 的最后 token
codebook0 = codes[0]
last_token = codebook0[-1]
print("codebook 0 最后一个 token:", last_token, "== im_end?", last_token == im_end_id)

# 寻找 im_end 出现位置
im_end_positions = np.where(codebook0 == im_end_id)[0]
print("im_end 出现位置:", im_end_positions[:20] if len(im_end_positions) else "（未出现）")

# 检测循环：最长无重复连续段长度
def longest_unique_run(arr):
    seen = {}
    best = 0
    start = 0
    for i, v in enumerate(arr):
        if v in seen and seen[v] >= start:
            start = seen[v] + 1
        seen[v] = i
        best = max(best, i - start + 1)
    return best, best / len(arr)

run, run_ratio = longest_unique_run(codebook0)
print(f"codebook 0 最长无重复连续段: {run} tokens ({run_ratio*100:.1f}% of {T})")

# top-10 重复 token
unique, counts = np.unique(codebook0, return_counts=True)
top = sorted(zip(counts, unique), reverse=True)[:10]
print("codebook 0 重复 top-10 (count, token_id):")
for c, u in top:
    print(f"  {c:5d}  {u:4d}  ({c/T*100:.1f}%)")
print()

# 检查前 100 与后 100 是否有相似模式（循环）
first100 = codebook0[:100]
last100 = codebook0[-100:]
match = sum(1 for a, b in zip(first100, last100) if a == b)
print(f"前 100 token 与最后 100 token 匹配数: {match}/100（高则说明循环）")

# 估计实际有效音频长度（连续相同 token > 100 的视为死循环丢弃）
non_trivial = 0
window = 50
for i in range(0, T - window, window):
    chunk = codebook0[i : i + window]
    if len(np.unique(chunk)) > window * 0.3:  # >30% unique
        non_trivial += window
print(f"有效 token 估计: ~{non_trivial} / {T} ({non_trivial/T*100:.1f}%)")
print(f"有效音频估计时长: ~{non_trivial/12/60:.1f} 分钟 (12Hz codec)")
print()
print("结论：")
if last_token != im_end_id and im_end_positions.size == 0:
    print("  模型未生成 im_end，跑到 max_seq_len 极限。可能原因：")
    print("  (1) 缺少 prompt 音频，模型不知道何时停止")
    print("  (2) 仅 '你好' 2 个字 + 无参考音色，模型进入重复模式")
    print("  (3) --half 精度可能在边界 token 上有数值问题")
print("  修复方案：重跑时加 1-2 秒静音 wav 作为 prompt，去掉 --half")
