"""C4: 训练内相似度门禁 — 监控 s2 权重产出, 逐轮克隆打分, 达标早停。

与 gsv_finetune.py 并行运行:
    python scripts/train_gate.py --exp user_voice --target 0.80

机制:
  1. 轮询 SoVITS_weights_v2/<exp>_e*.pth, 新权重出现即克隆+ERes2Net 打分
  2. 分数曲线写 output/train_gate_<exp>.jsonl
  3. sim ≥ target → 在 logs/<exp>/ 下放 STOP 信号文件 → s2_train 每轮末检查后退出
"""
import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GSV = REPO / "third_party" / "gpt_sovits"

_PROBE_TEXT = "人工智能技术正在以前所未有的速度发展,语音合成系统已经能够生成接近真人水平的自然流畅音频"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="user_voice")
    ap.add_argument("--ref", default=str(REPO / "data" / "voices" / "我的声音V2" / "ref.wav"))
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--poll", type=int, default=15)
    args = ap.parse_args()

    weights_dir = GSV / "SoVITS_weights_v2"
    stop_file = GSV / "logs" / args.exp / "STOP"
    curve_path = REPO / "output" / f"train_gate_{args.exp}.jsonl"

    from adr.eval.speaker_sim import similarity
    from adr.models.gsv_engine import get_gsv_engine
    eng = get_gsv_engine()

    # 基线: 零样本先打一发
    import soundfile as sf
    import numpy as np
    wav, sr = eng.synthesize(_PROBE_TEXT, args.ref)
    base = similarity(args.ref, str(_save_tmp(wav, sr)))
    print(f"[gate] 零样本基线 sim={base:.3f}, 目标 {args.target}", flush=True)
    with open(curve_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": 0, "sim": round(base, 4), "kind": "zero_shot"},
                           ensure_ascii=False) + "\n")

    seen = set()
    while True:
        for w in sorted(weights_dir.glob(f"{args.exp}_e*.pth")):
            if w.name in seen:
                continue
            seen.add(w.name)
            time.sleep(5)  # 等写盘完
            try:
                wav, sr = eng.synthesize(_PROBE_TEXT, args.ref, vits_weights=str(w))
                sim = similarity(args.ref, str(_save_tmp(wav, sr)))
                ep = w.name.split("_e")[1].split("_")[0]
                print(f"[gate] {w.name}: sim={sim:.3f}", flush=True)
                with open(curve_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": int(ep), "sim": round(sim, 4),
                                        "weight": w.name}, ensure_ascii=False) + "\n")
                if sim >= args.target:
                    stop_file.parent.mkdir(parents=True, exist_ok=True)
                    stop_file.write_text(f"sim={sim:.3f} @ {w.name}", encoding="utf-8")
                    print(f"[gate] 达标 {sim:.3f} ≥ {args.target}, 已发早停信号", flush=True)
                    return
            except Exception as e:
                print(f"[gate] {w.name} 打分失败: {e}", flush=True)
        time.sleep(args.poll)


def _save_tmp(wav, sr):
    import soundfile as sf
    p = REPO / "output" / "_gate_probe.wav"
    sf.write(str(p), wav, sr)
    return p


if __name__ == "__main__":
    main()
