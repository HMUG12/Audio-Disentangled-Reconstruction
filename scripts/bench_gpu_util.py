"""GPU 利用率曲线采样器 (M10 批次47 性能埋点, 为低配加速批次备料)。

训练/推理期间旁路运行, 按固定间隔采样 nvidia-smi 写 CSV:
    timestamp, gpu_util_pct, mem_used_mb, mem_total_mb, power_w, temp_c

用法 (另开一个终端, 与训练同时跑):
    python scripts/bench_gpu_util.py -o gpu_util.csv                # 采到 Ctrl+C
    python scripts/bench_gpu_util.py --duration 600 --interval 1.0  # 限时采样
    python scripts/bench_gpu_util.py --gpu 1                        # 指定卡

结束时打印摘要: 利用率均值/中位/p95 + "饿卡窗口"清单
(连续 util < --starve-threshold 的区段 — GPU 吃不饱的时间段,
 定位 dataloader / 前后端处理瓶颈的直接证据)。
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_QUERY = ("timestamp,utilization.gpu,memory.used,memory.total,"
          "power.draw,temperature.gpu")


def sample(gpu: str | None) -> tuple[int, float, float, float, float] | None:
    """采一次样 → (util%, mem_used, mem_total, power_w, temp_c); 失败 None。"""
    cmd = ["nvidia-smi", f"--id={gpu}" if gpu else "--id=0",
           f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        # nvidia-smi timestamp 列含日期时间, 采样时刻自己记
        return (int(parts[1]), float(parts[2]), float(parts[3]),
                float(parts[4].replace("N/A", "0")),
                float(parts[5].replace("N/A", "0")))
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return None


def summarize(rows: list[dict], starve_th: int, starve_min: int) -> None:
    if not rows:
        print("无样本, 跳过摘要")
        return
    utils = sorted(r["gpu_util_pct"] for r in rows)
    n = len(utils)
    print("\n===== GPU 利用率摘要 =====")
    print(f"样本: {n}  时长: "
          f"{(rows[-1]['ts'] - rows[0]['ts']).total_seconds():.0f}s")
    print(f"util 均值 {sum(utils)/n:.1f}%  中位 {utils[n//2]:.0f}%  "
          f"p95 {utils[int(n*0.95)]:.0f}%  峰值 {utils[-1]:.0f}%")
    # 饿卡窗口: 连续 starve_min 个样本低于阈值
    runs: list[tuple[datetime, datetime, int]] = []
    run_start = None
    run_len = 0
    for r in rows + [dict(rows[-1], gpu_util_pct=100)]:   # 哨兵收尾
        if r["gpu_util_pct"] < starve_th:
            if run_start is None:
                run_start = r["ts"]
            run_len += 1
        else:
            if run_len >= starve_min:
                runs.append((run_start, r["ts"], run_len))
            run_start, run_len = None, 0
    if runs:
        print(f"饿卡窗口 (util<{starve_th}% 连续≥{starve_min}样本): "
              f"{len(runs)} 段")
        for s, e, ln in runs[:10]:
            print(f"  {s:%H:%M:%S} → {e:%H:%M:%S}  ({ln} 样本)")
        print("→ 这些时段 GPU 吃不饱: 排查 dataloader/前后端处理/逐句间隙")
    else:
        print(f"无饿卡窗口 (util 恒 ≥{starve_th}%) — 训练侧 GPU 喂养正常")


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU 利用率曲线采样器")
    ap.add_argument("-o", "--out", default="gpu_util.csv", help="CSV 输出路径")
    ap.add_argument("--interval", type=float, default=1.0, help="采样间隔秒")
    ap.add_argument("--duration", type=float, default=0,
                    help="限时秒 (0=一直采到 Ctrl+C)")
    ap.add_argument("--gpu", default=None, help="GPU 序号 (多卡)")
    ap.add_argument("--starve-threshold", type=int, default=20,
                    help="饿卡判定阈值 util%% (默认 20)")
    ap.add_argument("--starve-min", type=int, default=5,
                    help="饿卡最短连续样本数 (默认 5)")
    a = ap.parse_args()

    rows: list[dict] = []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"采样中 → {out} (interval={a.interval}s, Ctrl+C 结束并出摘要)")
    try:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "gpu_util_pct", "mem_used_mb",
                        "mem_total_mb", "power_w", "temp_c"])
            while True:
                ts = datetime.now()
                s = sample(a.gpu)
                if s is not None:
                    util, mu, mt, pw, tp = s
                    w.writerow([ts.isoformat(timespec="seconds"),
                                util, mu, mt, pw, tp])
                    f.flush()
                    rows.append({"ts": ts, "gpu_util_pct": util,
                                 "mem_used_mb": mu})
                if a.duration and time.time() - t0 >= a.duration:
                    break
                time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    summarize(rows, a.starve_threshold, a.starve_min)


if __name__ == "__main__":
    sys.exit(main())
