"""W2: 显存预算模拟 — 用 memory fraction 在本机模拟低显存卡 (2GB/4GB)。

原理: torch.cuda.set_per_process_memory_fraction(budget/total) 强制本进程
CUDA 分配上限, 超限即 OOM — 等效于在更小的卡上跑。WDDM 共享显存可能
兜底但不影响 torch 侧判断。

用法: python scripts/bench_vram_budget.py --mb 2000
产物: 控制台结果 (追加写 output/lowres_baseline.json)
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "output" / "lowres_baseline.json"

PROBES = [
    "您好，请问有什么可以帮您的吗？",
    "今天天气不错，我们去公园散步吧。",
    "声音克隆是一项非常有意思的技术，它只需要几秒钟的参考音频，就能够生成非常接近真人的语音效果。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=2000, help="显存预算 (MB)")
    args = ap.parse_args()

    import torch
    total = torch.cuda.get_device_properties(0).total_memory / 1e6
    frac = args.mb / total
    assert frac < 1.0, f"预算 {args.mb}MB 不小于总显存 {total:.0f}MB"
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f"[vram] 模拟 {args.mb}MB 预算 (本机 {total:.0f}MB, fraction={frac:.3f})")

    from adr.models.gsv_engine import GSVEngine
    from adr.models.voice_library import load_voice
    prof = load_voice("我的声音V2")
    eng = GSVEngine()
    eng.config.device = "cuda"
    eng.config.half = True
    eng.warmup()
    torch.cuda.reset_peak_memory_stats()

    ok = True
    rows = []
    for text in PROBES:
        try:
            t0 = time.perf_counter()
            wav, sr = eng.synthesize(text, prof["ref_audio"],
                                     vits_weights=prof.get("vits_weights"))
            dt = time.perf_counter() - t0
            audio_s = len(wav) / sr
            rows.append({"text": text[:10] + "...", "gen_s": round(dt, 2),
                         "audio_s": round(audio_s, 2),
                         "rtf": round(dt / audio_s, 2)})
            print(f"  {text[:10]}...: {dt:.1f}s / {audio_s:.1f}s "
                  f"= RTF {dt / audio_s:.2f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            ok = False
            print(f"  {text[:10]}...: OOM (预算 {args.mb}MB 不够)", flush=True)
            break

    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"[vram] 峰值分配 {peak:.0f}MB / 预算 {args.mb}MB → "
          f"{'通过' if ok else '超预算 OOM'}")

    data = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else []
    data.append({"mode": f"vram_budget_{args.mb}mb", "ok": ok,
                 "peak_mb": round(peak), "probes": rows,
                 "date": time.strftime("%Y-%m-%d %H:%M")})
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
