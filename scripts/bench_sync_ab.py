"""Track E (批次33): AR 同步风暴消除 + empty_cache 节流 A/B 基准。

三腿对比 (同进程切腿 — t2s_model._every 与 _throttled_empty_cache 均
在调用时实时读 env, 无需重载模块):
  A_legacy_full: ADR_AR_SYNC_LEGACY=1 + ADR_TTS_KEEP_EMPTY_CACHE=1 (完全旧行为)
  B_legacy_ec:   仅 ADR_AR_SYNC_LEGACY=1 (empty_cache 节流生效)
  C_new:         默认 (_every=8 批量同步结算 + 节流)
读数: C-A = Track E 总提升 | C-B = T1 纯 AR 循环提速 | B-A = T3 节流效果
每格: 预热 1 次 + 测 2 次取 min (抗偶发抖动)。

一致性 (warn-only): seed=0 + cut0 单句非流式。
diag v3 token 指纹已证明 A/C 的 AR 输出 token 序列逐位一致 (sha1 相同);
音频差异仅源于 every=8 在 EOS 检出前多消耗 multinomial RNG → vits flow
噪声偏移 (合法性等价, 非偏差)。故不做 bit 级判定, 仅报告
样本数一致性 + 波形差异统计 (max|Δ| / frac / first_diff_idx)。

用法: python scripts/bench_sync_ab.py
产物: output/bench_sync_ab.json
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# numba 缓存必改道: 默认 TEMP 下 librosa 导入时 ensure_cache_path 被
# 安全软件拦截可卡死导入 (py-spy 取证确认)。须在 import numpy/librosa 前设置。
os.environ.setdefault("NUMBA_CACHE_DIR", "E:\\adr_numba_cache")

# 基准测真实合成: 段缓存必关 (否则预热腿填缓存, 测量腿全 0.00s 假数据)
os.environ["ADR_SEG_CACHE"] = "0"

OUT_JSON = REPO / "output" / "bench_sync_ab.json"

TEXTS = {
    "long_45": "人工智能技术正在以前所未有的速度发展，语音合成系统已经能够生成接近真人水平的自然流畅音频。",
    "two_20x2": "今天早上我喝了一杯咖啡。然后在公园里散了一会儿步，天气非常不错。",
    "short_12": "您好，请问有什么可以帮您的吗？",
}
CONSIST_TEXT = "您好，请问有什么可以帮您的吗？"
REPS = 2  # 每格测 REPS 次取 min

# 三腿 env 配置; 缺省键一律回 "0" (节流生效 / 新同步)
LEGS = {
    "A_legacy_full": {"ADR_AR_SYNC_LEGACY": "1", "ADR_TTS_KEEP_EMPTY_CACHE": "1"},
    "B_legacy_ec": {"ADR_AR_SYNC_LEGACY": "1", "ADR_TTS_KEEP_EMPTY_CACHE": "0"},
    "C_new": {"ADR_AR_SYNC_LEGACY": "0", "ADR_TTS_KEEP_EMPTY_CACHE": "0"},
}


def set_leg(name: str) -> None:
    for k in ("ADR_AR_SYNC_LEGACY", "ADR_TTS_KEEP_EMPTY_CACHE"):
        os.environ[k] = LEGS[name].get(k, "0")


def measure_stream(eng, text, ref, vits) -> dict:
    """一次流式合成计时 (cut3 默认切句): t_first/chunks/total/audio/rtf。"""
    t0 = time.perf_counter()
    t_first = None
    n = 0
    audio_s = 0.0
    sr = None
    for chunk, sr in eng.synthesize_stream(text, ref, vits_weights=vits):
        if t_first is None:
            t_first = time.perf_counter() - t0
        n += 1
        audio_s += len(chunk) / sr
    total = time.perf_counter() - t0
    return {"t_first_s": round(t_first, 2), "chunks": n,
            "total_s": round(total, 2), "audio_s": round(audio_s, 2),
            "rtf": round(total / audio_s, 2) if audio_s else None}


def consist_wav(eng, ref, vits):
    wav, sr = eng.synthesize(CONSIST_TEXT, ref, vits_weights=vits,
                             split_method="cut0", seed=0)
    return int(sr), int(len(wav)), wav


def diff_stats(w1, w2) -> dict:
    import numpy as np
    a, b = np.asarray(w1, dtype=np.float64), np.asarray(w2, dtype=np.float64)
    if a.shape != b.shape:
        return {"samples_equal": False, "shape": [len(a), len(b)]}
    d = np.abs(a - b)
    return {"samples_equal": True,
            "max_abs_diff": round(float(d.max()), 6),
            "frac_gt_1e-4": round(float((d > 1e-4).mean()), 4),
            "first_diff_idx": int(d.argmax()) if (d > 1e-4).any() else -1}


def main():
    from adr.models.gsv_engine import GSVEngine
    from adr.models.voice_library import load_voice

    prof = load_voice("我的声音V2")
    ref = prof["ref_audio"]
    vits = prof.get("vits_weights")
    print(f"[bench] 档案权重: {Path(vits).name if vits else '(官方)'}")

    eng = GSVEngine()
    t0 = time.perf_counter()
    eng.warmup()
    print(f"[bench] 引擎预热 {time.perf_counter() - t0:.1f}s\n")

    results = {}
    consistency = {}
    for leg in LEGS:
        set_leg(leg)
        print(f"=== {leg}  env={LEGS[leg]} ===", flush=True)
        rows = []
        for tname, text in TEXTS.items():
            measure_stream(eng, text, ref, vits)   # 预热 1 次, 丢弃
            reps = [measure_stream(eng, text, ref, vits) for _ in range(REPS)]
            m = min(reps, key=lambda r: r["total_s"])  # 取 min 抗噪
            rows.append({"text": tname, **m})
            print(f"  {tname:10s} 首包 {m['t_first_s']:5.2f}s | 块 {m['chunks']:3d} | "
                  f"总 {m['total_s']:5.2f}s | 音频 {m['audio_s']:4.2f}s | "
                  f"RTF {m['rtf']}", flush=True)
        results[leg] = rows
        if leg in ("A_legacy_full", "C_new"):
            consistency[leg] = consist_wav(eng, ref, vits)
        print(flush=True)

    # 一致性 (warn-only): 样本数应一致; 波形差异为 vits 噪声 RNG 偏移, 非偏差
    if len(consistency) == 2:
        sr_a, n_a, w_a = consistency["A_legacy_full"]
        sr_c, n_c, w_c = consistency["C_new"]
        st = diff_stats(w_a, w_c)
        st["sr_equal"] = sr_a == sr_c
        consistency = {"A": {"sr": sr_a, "samples": n_a},
                       "C": {"sr": sr_c, "samples": n_c}, "diff": st}
        print(f"[一致性] samples A={n_a} C={n_c} {'OK' if n_a == n_c else 'MISMATCH!'}"
              f" | max|Δ|={st.get('max_abs_diff')} frac={st.get('frac_gt_1e-4')} "
              f"first_diff={st.get('first_diff_idx')} (warn-only: vits 噪声 RNG 偏移)")

    # 腿间对比: 各文本 total_s 的 C-A / C-B / B-A
    def total(leg, tname):
        return next(r["total_s"] for r in results[leg] if r["text"] == tname)

    summary = []
    for tname in TEXTS:
        a, b, c = (total(l, tname) for l in LEGS)
        summary.append({
            "text": tname,
            "C_vs_A_speedup": round(a / c, 2) if c else None,
            "C_vs_B_speedup": round(b / c, 2) if c else None,
            "B_vs_A_speedup": round(a / b, 2) if b else None,
        })
        print(f"[对比] {tname:10s} C/A x{summary[-1]['C_vs_A_speedup']} | "
              f"C/B x{summary[-1]['C_vs_B_speedup']} | B/A x{summary[-1]['B_vs_A_speedup']}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"legs": results, "consistency": consistency, "summary": summary},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bench] 已写 {OUT_JSON}")


if __name__ == "__main__":
    main()
