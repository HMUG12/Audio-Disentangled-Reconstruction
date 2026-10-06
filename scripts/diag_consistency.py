"""Track E 诊断 v3: token 级取证 — every=8 的差异是配置效应还是进程状态效应?

设计: 同进程交替切配置 E8→E1→E8→E1→E8 (每跑 seed=0 + cut0 单句非流式)。
配合 t2s_model.py 的 ADR_AR_DEBUG_TOKENS=1 打印 [AR-DBG] token 指纹
(every/prefix/refree/gen/idx/first16/sha1)。

判读:
  - E8 三跑 sha1 恒定 036d 且 token first16 与 E1 不同 → 配置效应 (token 真分叉),
    查内核/RNG 级差异
  - E8 多跑 sha1 漂移 → 进程状态效应 (跑序/内存状态)
  - token sha1 相同但音频 sha1 不同 → 分叉在 vits 层
输出: 每跑 [AR-DBG] 行 + 音频 sha1 + E8#1 波形差异统计
"""
import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# numba 缓存必改道 (默认 TEMP 会被安全软件拦截卡死导入), 须在 import 前设置
os.environ.setdefault("NUMBA_CACHE_DIR", "E:\\adr_numba_cache")

TEXT = "您好，请问有什么可以帮您的吗？"
# 交替序列: 3x E8 + 2x E1, 判定配置效应 vs 跑序漂移
SEQUENCE = [("E8", {"ADR_AR_SYNC_LEGACY": "0", "ADR_AR_SYNC_EVERY": "8"}),
            ("E1", {"ADR_AR_SYNC_LEGACY": "0", "ADR_AR_SYNC_EVERY": "1"}),
            ("E8", {"ADR_AR_SYNC_LEGACY": "0", "ADR_AR_SYNC_EVERY": "8"}),
            ("E1", {"ADR_AR_SYNC_LEGACY": "0", "ADR_AR_SYNC_EVERY": "1"}),
            ("E8", {"ADR_AR_SYNC_LEGACY": "0", "ADR_AR_SYNC_EVERY": "8"})]


def synth(eng, ref, vits):
    import numpy as np
    wav, sr = eng.synthesize(TEXT, ref, vits_weights=vits,
                             split_method="cut0", seed=0)
    pcm = (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
           * 32767).astype(np.int16).tobytes()
    return hashlib.sha1(pcm).hexdigest()[:12], len(wav), wav


def diff_stats(w1, w2):
    import numpy as np
    a, b = np.asarray(w1), np.asarray(w2)
    d = np.abs(a - b)
    frac = float((d > 1e-4).mean())
    return (f"max|Δ|={d.max():.4f} frac>{1e-4}={frac:.4%} "
            f"first_diff_idx={int(np.argmax(d > 1e-4)) if (d > 1e-4).any() else -1}")


def main():
    from adr.models.gsv_engine import GSVEngine
    from adr.models.voice_library import load_voice

    os.environ["ADR_AR_DEBUG_TOKENS"] = "1"
    prof = load_voice("我的声音V2")
    ref, vits = prof["ref_audio"], prof.get("vits_weights")
    eng = GSVEngine()
    eng.warmup()

    runs = []
    for i, (name, env) in enumerate(SEQUENCE):
        for k, v in env.items():
            os.environ[k] = v
        h, n, wav = synth(eng, ref, vits)
        runs.append((name, h, n, wav))
        print(f"{name}#{sum(1 for t, *_ in runs[:i + 1] if t == name)}: "
              f"audio={h} samples={n}", flush=True)

    e8 = [(h, w) for t, h, n, w in runs if t == "E8"]
    e1 = [(h, w) for t, h, n, w in runs if t == "E1"]
    print("\n集合: E8 =", sorted({h for h, _ in e8}))
    print("      E1 =", sorted({h for h, _ in e1}))

    base_h, base_w = e8[0]
    print("\nE8#1 波形 vs:")
    for i, (h, w) in enumerate(e8[1:] + e1):
        print(f"  {h} {diff_stats(base_w, w)}")


if __name__ == "__main__":
    main()
