"""批次49 (m10 C 节): 推理 CPU/GPU 档端到端计时 + torch.profiler 组件归因。

背景: C 节原预期 "BigVGAN→ORT 预期最大收益", 实锤 ADR 底模 v2 系
(gsv_engine.py GSVEngineConfig.version="v2") 的声码在 VITS decoder
内部 — 不走 BigVGAN 外挂 (仅 v3/v4 挂 BigVGAN, api.py/TTS.py init_vocoder
"v3" 分支)。CPU 档剩余大头 = t2s AR + VITS decode, 本脚本实测归因
定格局, 驱动后续加速投入 (线程/量化/ORT 化取舍)。

计时口径 (沿 bench_sync_ab.py, 跨批次可比):
1. 端到端: 每句预热 1 次丢弃 + REPS 次取 min — 首包 / 总耗时 / 音频时长
   / RTF (实时率, <1 = 快于实时)
2. 归因: torch.profiler 跑 short_12 一句 (cut0 seed0 非流式确定性),
   key_averages top 表落 output/bench_cpu_e2e_<dev>_profile.txt —
   aten::einsum/mm/bmm 分布归因 t2s AR vs VITS decode

用法:
  python scripts/bench_cpu_e2e.py --device cpu  [--threads N] [--profile]
  python scripts/bench_cpu_e2e.py --device cuda [--profile]
产物: output/bench_cpu_e2e_<device>.json (+ _profile.txt)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# numba 缓存必改道 (bench_sync_ab 同口径): 默认 TEMP 下 librosa 导入时
# ensure_cache_path 被安全软件拦截可卡死导入。须在 import numpy 前设置。
os.environ.setdefault("NUMBA_CACHE_DIR", "E:\\adr_numba_cache")
# 基准测真实合成: 段缓存必关 (预热腿填缓存, 测量腿全 0.00s 假数据)
os.environ["ADR_SEG_CACHE"] = "0"

OUT = REPO / "output"
TEXTS = {
    "long_45": "人工智能技术正在以前所未有的速度发展，语音合成系统已经能够生成接近真人水平的自然流畅音频。",
    "two_20x2": "今天早上我喝了一杯咖啡。然后在公园里散了一会儿步，天气非常不错。",
    "short_12": "您好，请问有什么可以帮您的吗？",
}


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


def run_profile(eng, text, ref, vits, device: str, out: dict) -> None:
    """torch.profiler 算子归因: 一句 cut0 seed0 非流式 (确定性路径)。

    首跑不进 profiler (懒初始化/JIT 噪声), 第二跑才采样。
    """
    import torch
    from torch.profiler import profile, ProfilerActivity

    eng.synthesize(text, ref, vits_weights=vits, split_method="cut0", seed=0)
    acts = [ProfilerActivity.CPU]
    if device != "cpu" and torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as p:
        eng.synthesize(text, ref, vits_weights=vits, split_method="cut0", seed=0)
    sort_key = "self_cuda_time_total" if (device != "cpu" and torch.cuda.is_available()) \
        else "self_cpu_time_total"
    tbl = p.key_averages().table(sort_by=sort_key, row_limit=30)
    OUT.mkdir(exist_ok=True)
    ptxt = OUT / f"bench_cpu_e2e_{device}_profile.txt"
    ptxt.write_text(tbl, encoding="utf-8")
    evts = p.key_averages()
    out["profile"] = {
        "total_cpu_ms": round(sum(e.self_cpu_time_total for e in evts) / 1000, 1),
        "total_cuda_ms": round(sum(getattr(e, "self_cuda_time_total", 0) for e in evts) / 1000, 1),
        "table": str(ptxt),
    }
    print(tbl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="auto=引擎默认; cpu/cuda 强制 (ADR_TTS_DEVICE)")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch 线程数 (0=PyTorch 默认; ADR_TORCH_THREADS)")
    ap.add_argument("--reps", type=int, default=2, help="每句计时次数 (取 min)")
    ap.add_argument("--profile", action="store_true", help="torch.profiler 算子归因")
    args = ap.parse_args()

    if args.device != "auto":
        os.environ["ADR_TTS_DEVICE"] = args.device
    if args.threads > 0:
        os.environ["ADR_TORCH_THREADS"] = str(args.threads)

    import torch
    from adr.models.gsv_engine import GSVEngine
    from adr.models.voice_library import load_voice

    prof = load_voice("我的声音V2")
    ref = prof["ref_audio"]
    vits = prof.get("vits_weights")
    print(f"[bench] 档案权重: {Path(vits).name if vits else '(官方)'}")

    eng = GSVEngine()
    t0 = time.perf_counter()
    eng.warmup()
    print(f"[bench] device={args.device} threads={args.threads or '默认'} "
          f"(实际 torch.threads={torch.get_num_threads()}) "
          f"预热 {time.perf_counter() - t0:.1f}s\n", flush=True)

    out = {
        "device": args.device,
        "torch_threads": torch.get_num_threads(),
        "arch": "v2 (vits 端到端声码, 无 BigVGAN 外挂)",
        "rows": [],
    }
    for tname, text in TEXTS.items():
        measure_stream(eng, text, ref, vits)   # 预热 1 次, 丢弃
        reps = [measure_stream(eng, text, ref, vits) for _ in range(args.reps)]
        m = min(reps, key=lambda r: r["total_s"])
        out["rows"].append({"text": tname, **m})
        print(f"  {tname:10s} 首包 {m['t_first_s']:5.2f}s | 块 {m['chunks']:3d} | "
              f"总 {m['total_s']:5.2f}s | 音频 {m['audio_s']:4.2f}s | "
              f"RTF {m['rtf']}", flush=True)

    if args.profile:
        print("\n[profile] short_12 cut0 seed0 算子归因 (CPU"
              f"{' + CUDA' if args.device != 'cpu' else ''}):")
        run_profile(eng, TEXTS["short_12"], ref, vits, args.device, out)

    OUT.mkdir(exist_ok=True)
    jf = OUT / f"bench_cpu_e2e_{args.device}.json"
    jf.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {jf}")


if __name__ == "__main__":
    main()
