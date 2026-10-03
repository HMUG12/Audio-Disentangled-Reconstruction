"""W3 诊断: s2 训练显存分解 — reserved(分配器缓存) vs 真实占用。

背景: 模型+梯度+AdamW 状态合计 ~400MB, 但实测峰值 7979MB 且与 batch 无关
(C7)。怀疑大头是 CUDA caching allocator 的 reserved 内存。本脚本在独立
exp (user_diag, 复制 user_holdout 产物, 不碰档案权重) 上跑 1 epoch,
外部轮询 nvidia-smi 记录显存曲线, 对比:
  A 默认分配器   B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

用法: python scripts/diag_s2_vram.py [--alloc-conf expandable_segments:True]
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GSV = REPO / "third_party" / "gpt_sovits"
PY = sys.executable


def poll_vram(stop_evt, out):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            out.append((time.time(), int(r.stdout.strip().splitlines()[0])))
        except Exception:
            pass
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc-conf", default=None,
                    help="如 expandable_segments:True; 不传=默认分配器")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--bs", type=int, default=None,
                    help="覆盖 batch_size (激活缩放二分定位)")
    ap.add_argument("--clip-sec", type=float, default=None,
                    help="ADR_MAX_CLIP_SEC (长 clip 截断秒数)")
    args = ap.parse_args()

    # 独立 exp: 复制特征产物, 不碰 user_holdout 权重
    src_logs = GSV / "logs" / "user_holdout"
    dst_logs = GSV / "logs" / "user_diag"
    if dst_logs.exists():
        shutil.rmtree(dst_logs)
    shutil.copytree(src_logs, dst_logs,
                    ignore=shutil.ignore_patterns("logs_s2_v2", "logs_s1_v2", "STOP"))
    (GSV / "SoVITS_weights_v2").mkdir(exist_ok=True)
    asr_out = GSV / "output" / "asr_opt"
    asr_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asr_out / "user_holdout.list", asr_out / "user_diag.list")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{GSV};{GSV / 'GPT_SoVITS'}"
    if args.alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.alloc_conf
    if args.clip_sec:
        env["ADR_MAX_CLIP_SEC"] = str(args.clip_sec)

    stop_evt = threading.Event()
    samples = []
    t = threading.Thread(target=poll_vram, args=(stop_evt, samples), daemon=True)
    t.start()

    t0 = time.time()
    cmd = [PY, "-u", str(REPO / "scripts" / "gsv_finetune.py"),
           str(REPO / "8月15日.mp3"), "--exp", "user_diag",
           "--skip-to", "s2", "--skip-s1", "--s2-epochs", "1"]
    if args.bs:
        cmd += ["--batch-size", str(args.bs)]
    r = subprocess.run(cmd, env=env, cwd=str(GSV))
    dur = time.time() - t0
    stop_evt.set()
    time.sleep(3)

    peak = max((m for _, m in samples), default=0)
    print(f"\n[diag] tag={args.tag} exit={r.returncode} 用时 {dur:.0f}s "
          f"采样 {len(samples)} 点 → 显存峰值 {peak}MB")
    if samples:
        for ts, m in samples[:: max(1, len(samples) // 12)]:
            print(f"  +{ts - samples[0][0]:5.0f}s  {m}MB")


if __name__ == "__main__":
    main()
