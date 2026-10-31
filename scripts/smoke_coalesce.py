"""批次 67 (m10 N=2 第三子批): 跨请求合批真权重 A/B 冒烟。

两腿同进程切腿 (COALESCE/BATCH_RNG 旋钮调用时实时读 env):
  A_solo:    ADR_T2S_COALESCE=0 — stock 串行路径 (每请求独立 synthesize)
  B_coalesce: COALESCE=1 + BATCH_RNG=1
             两请求线程并发提交 (同签名: 同 ref/prompt/语速/权重, 仅文本
             与种子不同) → coalescer worker 一轮抽干 → 同组混批
  注意: 调度并发不是实时旋钮 — 引擎构造期快照 (见下方 env 设置), 必须
  在 GSVEngine() 创建前定为 2, 否则 B 腿线程 B 排队到 A 完成才 submit,
  合批窗口永远落空 (批次 67 冒烟两轮问题根因)。

音质门 (G3-cross 真权重端到端实证): B 腿每请求输出 == A 腿同文本同种子
solo 输出, float32 位等 (np.array_equal)。t2s 行种子桥 (行隔离) +
vits 逐请求 cat + 全局流重播的位等设计在此落地验证 — 任一请求出现
"邻行渗漏 / RNG 污染 / max_len 移位漂移" 即字节不等, 冒烟 FAIL。
另含真路径门: B 腿必须走桥真路径 (行 generator 批解码 real>=1) 且不落
_group_solo 回退 — 档案无 prompt_text / vocoder 时合批空转, 门失效。

形状预热 (批次 67 诊断): 同 token 长度 shape 的首次运行存在与后续运行
位不等的 GPU 一次性伪影 — 同输入同 RNG 状态, 首跑 vs 后续差 ~244 int16
单位, 第二跑起全部相互位等, 与 COALESCE/BATCH_RNG env 无关 (诊断 3:
A1 vs A2..C1 244 单位, A2 起十组两两位等; logits/token/vits 入参逐层
指纹全同, 分歧在 vits 层首跑)。冒烟前每个 (text, seed) 形状先真跑丢弃、
并预走一遍 B 腿真合批批形状, 保证 A/B 双方比较时均已越过首跑 — 位等门
测的是设计而非伪影。

计时: A 腿串行两请求墙钟 vs B 腿并发两请求墙钟 (合批摊薄 AR 批边际
成本的直观读数; 非严格基准, 供回归观察; 双腿均已被形状预热撑平)。

用法: python scripts/smoke_coalesce.py [--voice yui的声音]
产物: output/smoke_coalesce/*.wav + report.json; 退出码 0=PASS 1=FAIL
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# numba 缓存必改道 (同 bench_sync_ab): librosa 导入期 ensure_cache_path
# 被安全软件拦截可卡死。须在 import numpy/librosa 前设置。
os.environ.setdefault("NUMBA_CACHE_DIR", "E:\\adr_numba_cache")
# 基准测真实合成: 段缓存必关 (否则 A 腿填缓存 B 腿走捷径, 计时失真)
os.environ["ADR_SEG_CACHE"] = "0"
# 调度并发是引擎构造期快照 (GSVEngine.__init__ → SynthesisScheduler(cap),
# slot() 满额条件变量排队): 引擎在任何 set_leg 之前创建, 默认 cap=1 会让
# B 腿线程 B 在 _admit 排队到线程 A 合成完毕才 submit → 合批窗口永远
# 落空 (两轮各单请求, 并发退化串行 — 批次 67 冒烟两轮问题根因)。
# 进程级固定 2: A 腿顺序提交, cap=2 无行为影响; B 腿 coalesce_on
# (实时读) 要求 >=2。
os.environ["ADR_SCHED_MAX_CONCURRENCY"] = "2"

OUT_DIR = REPO / "output" / "smoke_coalesce"

# A: 长文本 (默认 cut1 切多段 — 混批行数多, 分桶压力真实)
# B: 短文本 (cut0 单段 — 与 A 请求段长悬殊, max_len 移位位等的实证面)
TEXT_A = ("人工智能技术正在以前所未有的速度发展，语音合成系统已经能够生成"
          "接近真人水平的自然流畅音频。从早期的拼接合成到如今的端到端神经"
          "网络，质量提升有目共睹。")
TEXT_B = "您好，请问有什么可以帮您的吗？"
SEED_A, SEED_B = 0, 123

LEGS = {
    "A_solo":     {"ADR_T2S_COALESCE": "0", "ADR_T2S_BATCH_RNG": "0"},
    "B_coalesce": {"ADR_T2S_COALESCE": "1", "ADR_T2S_BATCH_RNG": "1"},
}


def set_leg(name: str) -> None:
    for k in ("ADR_T2S_COALESCE", "ADR_T2S_BATCH_RNG"):
        os.environ[k] = LEGS[name][k]


# 真路径观察 (批次 67): 桥真解码计数 + _group_solo 回退计数。导入即装,
# 覆盖任何 env 下的合批调用 (含预热); 音质门前清零, 只计 B 腿本体。
import adr.models.adr_t2s_batchrng as _br

REAL_HITS = [0]
_orig_brd = _br._batch_rng_decode


def _brd_spy(*a, **k):
    REAL_HITS[0] += 1
    return _orig_brd(*a, **k)


_br._batch_rng_decode = _brd_spy

from adr.models.gsv_coalescer import _Coalescer

SOLO_FALLBACK_HITS = [0]
_orig_solo = _Coalescer._group_solo


def _solo_spy(self, tts, gjobs):
    SOLO_FALLBACK_HITS[0] += 1
    return _orig_solo(self, tts, gjobs)


_Coalescer._group_solo = _solo_spy


def diff_stats(w1, w2) -> dict:
    import numpy as np

    a, b = np.asarray(w1), np.asarray(w2)
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"bit_equal": False, "shape": [list(a.shape), list(b.shape)],
                "dtype": [str(a.dtype), str(b.dtype)]}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"bit_equal": bool(np.array_equal(a, b)),
            "samples": int(a.shape[0]),
            "max_abs_diff": float(d.max())}


def save_wav(path: Path, wav, sr) -> None:
    import soundfile as sf

    sf.write(str(path), wav, sr, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="yui的声音")
    args = ap.parse_args()

    from adr.models.gsv_engine import GSVEngine
    from adr.models.voice_library import load_voice

    prof = load_voice(args.voice)
    ref = prof["ref_audio"]
    prompt_text = prof.get("prompt_text") or ""
    vits = prof.get("vits_weights")
    t2s = prof.get("t2s_weights")
    if not prompt_text:
        print("[smoke] 警告: 档案无 prompt_text — 真路径门将 FAIL "
              "(合批整组落 _group_solo 回退)")
    print(f"[smoke] 档案: {args.voice} | ref={Path(ref).name} | "
          f"prompt={prompt_text[:20]!r} | "
          f"vits={Path(vits).name if vits else '(官方)'} | "
          f"t2s={Path(t2s).name if t2s else '(官方)'}")

    eng = GSVEngine()
    t0 = time.perf_counter()
    eng.warmup()
    print(f"[smoke] 引擎预热 {time.perf_counter() - t0:.1f}s\n")

    def synth(text, seed, split):
        return eng.synthesize(text, ref, prompt_text=prompt_text,
                              vits_weights=vits, t2s_weights=t2s,
                              seed=seed, split_method=split)

    def concurrent_pair():
        """两线程并发提交 (TEXT_A, TEXT_B), 返回 (输出, 异常, 墙钟)。"""
        out, err = {}, {}

        def _run(key, text, seed, split):
            try:
                out[key] = synth(text, seed, split)
            except BaseException as e:  # 线程边界: 先收异常再统一报告
                err[key] = e

        threads = [threading.Thread(target=_run, args=(k, t, s, sp),
                                    name=f"smoke-{k}")
                   for k, t, s, sp in (("A", TEXT_A, SEED_A, "cut1"),
                                       ("B", TEXT_B, SEED_B, "cut0"))]
        t0 = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        return out, err, time.perf_counter() - t0

    # ---- 形状预热 (批次 67 诊断): 越过 "同 shape 首跑位不等" 伪影 --------
    # 每个 (text, seed) 形状 solo 真跑一次丢弃 + B 腿真合批批形状预走一遍;
    # 顺带加载用户权重, 使 A/B 腿计时不再含首载成本。
    print("[smoke] 形状预热 (输出丢弃)...", flush=True)
    set_leg("A_solo")
    synth(TEXT_A, SEED_A, "cut1")
    synth(TEXT_B, SEED_B, "cut0")
    set_leg("B_coalesce")
    wu_out, wu_err, _ = concurrent_pair()
    if wu_err:
        for k, e in wu_err.items():
            print(f"  [预热FAIL] {k}: {type(e).__name__}: {e}", flush=True)
        raise SystemExit(1)
    print("[smoke] 形状预热完成\n", flush=True)

    # ---- A 腿: solo 串行基线 (每请求单独计时, 保留波形供位等) ---------
    set_leg("A_solo")
    print(f"=== A_solo  env={LEGS['A_solo']} ===", flush=True)
    results, waves = {}, {}
    for key, text, seed, split in (("A", TEXT_A, SEED_A, "cut1"),
                                   ("B", TEXT_B, SEED_B, "cut0")):
        t0 = time.perf_counter()
        wav, sr = synth(text, seed, split)
        waves[key] = (wav, sr)
        results[key] = {"text": text, "seed": seed, "sr": int(sr),
                        "samples": int(len(wav)),
                        "total_s": round(time.perf_counter() - t0, 2)}
        print(f"  {key}: {results[key]['total_s']:6.2f}s | "
              f"{len(wav)/sr:5.2f}s 音频 | {len(wav)} samples", flush=True)
    a_serial_s = results["A"]["total_s"] + results["B"]["total_s"]
    print(f"  串行总计 {a_serial_s:.2f}s\n", flush=True)

    # ---- B 腿: 合批并发 (两线程同发, worker 一轮抽干) ------------------
    set_leg("B_coalesce")
    print(f"=== B_coalesce  env={LEGS['B_coalesce']} ===", flush=True)
    REAL_HITS[0] = 0            # 真路径门只计 B 腿本体 (预热已验可达性)
    SOLO_FALLBACK_HITS[0] = 0
    b_out, b_err, b_parallel_s = concurrent_pair()
    if b_err:
        for k, e in b_err.items():
            print(f"  [FAIL] {k}: {type(e).__name__}: {e}", flush=True)
        raise SystemExit(1)
    for key in ("A", "B"):
        wav, sr = b_out[key]
        print(f"  {key}: 墙钟内完成 | {len(wav)/sr:5.2f}s 音频 | "
              f"{len(wav)} samples", flush=True)
    print(f"  并发总计 {b_parallel_s:.2f}s\n", flush=True)

    # ---- 音质门: 每请求 B(合批) vs A(solo) float32 位等 ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "voice": args.voice,
        "legs": {name: LEGS[name] for name in LEGS},
        "solo": results,
        "timing": {"a_serial_s": a_serial_s,
                   "b_parallel_s": round(b_parallel_s, 2),
                   "speedup": round(a_serial_s / b_parallel_s, 2)
                   if b_parallel_s else None},
        "equivalence": {},
    }
    verdict = True
    for key in ("A", "B"):
        wav_a, sr_a = waves[key]
        wav_b, sr_b = b_out[key]
        st = diff_stats(wav_a, wav_b)
        st["sr_equal"] = bool(sr_a == sr_b)
        report["equivalence"][key] = st
        ok = st["bit_equal"] and st["sr_equal"]
        verdict &= ok
        print(f"  [音质门] {key}: {'PASS' if ok else 'FAIL'} | "
              f"bit_equal={st['bit_equal']} sr_equal={st['sr_equal']} "
              f"max|Δ|={st.get('max_abs_diff')}", flush=True)
        save_wav(OUT_DIR / f"solo_{key}.wav", wav_a, sr_a)
        save_wav(OUT_DIR / f"coalesce_{key}.wav", wav_b, sr_b)

    # 真路径门: B 腿必须真走桥 (行 generator 批解码), 不许整组落回退
    real_ok = REAL_HITS[0] >= 1 and SOLO_FALLBACK_HITS[0] == 0
    report["real_path"] = {"batch_rng_decode": REAL_HITS[0],
                           "group_solo_fallback": SOLO_FALLBACK_HITS[0]}
    verdict &= real_ok
    print(f"  [真路径门] {'PASS' if real_ok else 'FAIL'} | "
          f"real={REAL_HITS[0]} solo_fallback={SOLO_FALLBACK_HITS[0]}",
          flush=True)

    report["verdict"] = "PASS" if verdict else "FAIL"
    print(f"\n[smoke] 加速比: 串行 {a_serial_s:.2f}s vs 并发 "
          f"{b_parallel_s:.2f}s → x{report['timing']['speedup']}")
    print(f"[smoke] 判定: {report['verdict']}")
    json.dump(report, open(OUT_DIR / "report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[smoke] 已写 {OUT_DIR / 'report.json'}")
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
