"""批次4-2: torch.compile 验证 — DiffSinger 提速, Windows 可行性实测。

背景 (批次3 profile 定案): 33s 固定开销 = ~20 万微算子的 kernel launch
开销 (Self CPU 43s vs Self CUDA 10s), 与扩散步数无关。
torch.compile 融合微算子 → 内核数骤减; mode="reduce-overhead" 额外用
CUDA Graph 把整个 launch 序列折叠成一次回放。

依赖: triton-windows (Windows CUDA 编译后端, torch 2.10 实测 3.8.0 可用)。

用法: python scripts/bench_diffsinger_compile.py [--steps N] [--mode MODE]
输出: output/bench_diffsinger_compile.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _time_synth(eng, kwargs, speedup, n=2):
    """稳态计时: 先跑一发丢弃 (JIT/缓存), 再平均 n 发。"""
    import torch
    eng.synthesize(speedup=speedup, **kwargs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        eng.synthesize(speedup=speedup, **kwargs)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts), sum(ts) / len(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="reduce-overhead",
                    choices=["reduce-overhead", "default", "cudagraphs"],
                    help="reduce-overhead=inductor+CUDA Graph (最优), default=仅融合, "
                         "cudagraphs=不依赖 triton 的图回放后端")
    ap.add_argument("--speedup", type=int, default=80)
    ap.add_argument("--compile-vocoder", action="store_true",
                    help="同时编译 vocoder (默认只编译主模型)")
    args = ap.parse_args()

    # 环境前置检查
    try:
        import triton  # noqa: F401
        print(f"[env] triton {triton.__version__}")
    except ImportError:
        if args.mode != "cudagraphs":
            print("[FAIL] triton 未安装 (pip install triton-windows); "
                  "Windows CUDA 后端必需, 或改用 --mode cudagraphs")
            return 1

    import torch
    from adr.models.diffsinger_engine import get_engine

    eng = get_engine()
    trans = REPO / "data" / "opencpop" / "transcriptions.txt"
    utt_id = None
    for line in open(trans, encoding="utf-8"):
        f = line.strip().split("|")
        if len(f) >= 2:
            utt_id = f[0].strip()
            break
    kwargs = eng.load_opencpop_annotation(utt_id)
    kwargs.pop("text", None)

    # 基线 (eager, 引擎内部预热后稳态)
    base_min, base_avg = _time_synth(eng, kwargs, args.speedup)
    print(f"[baseline] eager: min={base_min:.2f}s avg={base_avg:.2f}s "
          f"(speedup={args.speedup})", flush=True)

    # 编译: 主模型 (fs2+diffusion)。vocoder 单独开关 (其内含 NSF 反共振
    # 结构, 图捕获收益未知, 先隔离验证)
    infer = eng._infer
    n_params = sum(p.numel() for p in infer.model.parameters())
    print(f"[compile] torch {torch.__version__}, model params={n_params/1e6:.1f}M, "
          f"mode={args.mode}", flush=True)
    t0 = time.perf_counter()
    infer.model = torch.compile(infer.model, mode=args.mode)
    if args.compile_vocoder:
        infer.vocoder = torch.compile(infer.vocoder, mode=args.mode)
    try:
        c1, _ = _time_synth(eng, kwargs, args.speedup, n=1)  # 首发含编译耗时
    except Exception as e:
        print(f"[FAIL] {args.mode} 编译/执行失败: {type(e).__name__}: {e}",
              flush=True)
        result = {"mode": args.mode, "feasible": False,
                  "error": f"{type(e).__name__}: {e}",
                  "baseline_min_s": round(base_min, 2)}
        out = REPO / "output" / "bench_diffsinger_compile.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return 1
    compile_s = time.perf_counter() - t0
    cmin, cavg = _time_synth(eng, kwargs, args.speedup)
    print(f"[compiled] 首发含编译={compile_s:.1f}s, 稳态 min={cmin:.2f}s "
          f"avg={cavg:.2f}s", flush=True)
    print(f"[结果] {base_min:.2f}s -> {cmin:.2f}s "
          f"({base_min / cmin:.2f}x)", flush=True)

    result = {
        "mode": args.mode, "feasible": True,
        "torch": torch.__version__,
        "compile_first_call_s": round(compile_s, 1),
        "baseline_min_s": round(base_min, 2),
        "compiled_min_s": round(cmin, 2),
        "speedup_x": round(base_min / cmin, 2),
        "speedup_setting": args.speedup,
        "compile_vocoder": args.compile_vocoder,
        "utt_id": utt_id,
    }
    out = REPO / "output" / "bench_diffsinger_compile.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
