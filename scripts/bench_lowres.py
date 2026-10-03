"""W2: 低资源基线测量 — 显存分档 / CPU RTF / int8 动态量化对比。

用法:
    python scripts/bench_lowres.py gpu        # 显存峰值 + GPU RTF
    python scripts/bench_lowres.py cpu        # CPU fp32 RTF 基线
    python scripts/bench_lowres.py cpu-int8   # t2s int8 动态量化 RTF 对比
产物: output/lowres_baseline.json (追加式)
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "output" / "lowres_baseline.json"

# 短文本为主 (CPU 上长文本代价太高)
PROBES = [
    "您好，请问有什么可以帮您的吗？",
    "今天天气不错，我们去公园散步吧。",
]


def bench_engine(mode: str) -> list[dict]:
    from adr.models.gsv_engine import GSVEngine
    from adr.models.voice_library import load_voice

    prof = load_voice("我的声音V2")
    ref = prof["ref_audio"]
    vits = prof.get("vits_weights")

    if mode == "gpu":
        cfg_device, half = "cuda", True
    else:
        cfg_device, half = "cpu", False

    eng = GSVEngine()
    eng.config.device = cfg_device
    eng.config.half = half
    t0 = time.perf_counter()
    eng.warmup()
    load_s = time.perf_counter() - t0
    print(f"[{mode}] 引擎加载 {load_s:.1f}s")

    if mode == "gpu":
        import torch
        torch.cuda.reset_peak_memory_stats()

    if mode in ("cpu-int8", "cpu-int8-bert"):
        # t2s (GPT 语义) 是自回归采样主力, Linear 层 int8 动态量化
        import torch
        tts_obj = eng._tts
        targets = ["t2s_model"] if mode == "cpu-int8" else \
            ["t2s_model", "bert_model"]  # vits 有 weight_norm, 不兼容 dynamic quant
        for attr in targets:
            m = getattr(tts_obj, attr, None)
            if m is None or not hasattr(m, "modules"):
                print(f"[{mode}] {attr}: 不存在, 跳过")
                continue
            n_lin = sum(1 for x in m.modules() if x.__class__.__name__ == "Linear")
            if n_lin == 0:
                print(f"[{mode}] {attr}: 无 Linear 层, 跳过")
                continue
            t0 = time.perf_counter()
            q = torch.ao.quantization.quantize_dynamic(
                m, {torch.nn.Linear}, dtype=torch.qint8)
            setattr(tts_obj, attr, q)
            print(f"[{mode}] {attr}: 量化 {n_lin} 个 Linear "
                  f"({time.perf_counter() - t0:.1f}s)")

    rows = []
    for text in PROBES:
        t0 = time.perf_counter()
        wav, sr = eng.synthesize(text, ref, vits_weights=vits)
        dt = time.perf_counter() - t0
        audio_s = len(wav) / sr
        row = {"mode": mode, "text": text[:12] + "...",
               "gen_s": round(dt, 2), "audio_s": round(audio_s, 2),
               "rtf": round(dt / audio_s, 1)}
        rows.append(row)
        print(f"  [{mode}] {row['text']}: {dt:.1f}s / {audio_s:.1f}s 音频 "
              f"= RTF {row['rtf']}x", flush=True)

    if mode == "gpu":
        import torch
        peak = torch.cuda.max_memory_allocated() / 1e6
        rows.append({"mode": "gpu", "peak_vram_mb": round(peak)})
        print(f"[gpu] 峰值显存 {peak:.0f} MB")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gpu", "cpu", "cpu-int8", "cpu-int8-bert"])
    args = ap.parse_args()

    rows = bench_engine(args.mode)
    data = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else []
    stamp = time.strftime("%Y-%m-%d %H:%M")
    for r in rows:
        r["date"] = stamp
    data.extend(rows)
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[bench] 已追加写 {OUT_JSON}")


if __name__ == "__main__":
    main()
