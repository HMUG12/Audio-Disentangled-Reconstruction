"""C4: 训练内相似度门禁 — 监控 s2 权重产出, 逐轮克隆打分, 达标早停。

与 gsv_finetune.py 并行运行:
    python scripts/train_gate.py --exp user_voice --target 0.80

机制:
  1. 轮询 SoVITS_weights_v2/<exp>_e*.pth, 新权重出现即克隆+ERes2Net 打分
  2. 每个权重用 5 句探测集 (adr/eval/probes.py, 与 eval_heldout.py 同源)
     逐句打分取均值 — 单句探测方差大 (短句天然弱, 波动 ±0.03+), 均值更稳
  3. 分数曲线写 output/train_gate_<exp>.jsonl (含逐句分)
  4. mean sim ≥ target → 在 logs/<exp>/ 下放 STOP 信号文件 → s2_train 每轮末检查后退出
"""
import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GSV = REPO / "third_party" / "gpt_sovits"

# 探测集 / 预训练钉死常量 / 克隆打分 / ref 裁剪 — 批次 51 收敛到 adr.eval.probes
# (此前与 eval_heldout.py 各自复制一份 PROBES, 有双源漂移风险)
from adr.eval.probes import PRE_S2G, PRE_T2S, clip_ref, score_probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="user_voice")
    ap.add_argument("--ref", default=str(REPO / "data" / "voices" / "我的声音V2" / "ref.wav"))
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--poll", type=int, default=15)
    ap.add_argument("--once", action="store_true",
                    help="对当前已有权重各打一轮分后退出 (重打分/验证用, 不轮询)")
    ap.add_argument("--cpu", action="store_true",
                    help="打分引擎走 CPU (训练占 GPU 时避免争抢显存; 离线门禁慢点无妨)")
    ap.add_argument("--texts", default=None,
                    help="自定义门禁句集文件 (每行一句) — 人设定制: 用目标场景的句子测, "
                         "如 MOSS 人设用系统播报腔句子; 基线与训练后自动同句集")
    args = ap.parse_args()

    sents = None
    if args.texts:
        sents = [ln.strip() for ln in Path(args.texts).read_text(
            encoding="utf-8-sig").splitlines() if ln.strip()]
        assert len(sents) >= 3, f"自定义句集至少 3 句, 当前 {len(sents)}"
        print(f"[gate] 自定义门禁句集 {len(sents)} 句 (来自 {args.texts})")

    weights_dir = GSV / "SoVITS_weights_v2"
    stop_file = GSV / "logs" / args.exp / "STOP"
    curve_path = REPO / "output" / f"train_gate_{args.exp}.jsonl"

    from adr.models.gsv_engine import get_gsv_engine, GSVEngine, GSVEngineConfig
    if args.cpu:
        eng = GSVEngine(GSVEngineConfig(device="cpu", half=False))
    else:
        eng = get_gsv_engine()

    # 基线: 真零样本 (官方预训练 s1+s2), 不受 tts_infer.yaml custom 段漂移影响
    args.ref = clip_ref(args.ref, args.exp)
    base, base_sims = score_probes(eng, args.ref, sents=sents)
    print(f"[gate] 零样本基线 mean={base:.3f} 逐句={['%.3f' % s for s in base_sims]}, "
          f"目标 {args.target}", flush=True)
    with open(curve_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": 0, "sim": round(base, 4), "sims": [round(s, 4) for s in base_sims],
                            "kind": "zero_shot"}, ensure_ascii=False) + "\n")

    seen = set()
    while True:
        for w in sorted(weights_dir.glob(f"{args.exp}_e*.pth")):
            if w.name in seen:
                continue
            seen.add(w.name)
            time.sleep(5)  # 等写盘完
            try:
                mean, sims = score_probes(eng, args.ref, vits=str(w), sents=sents)
                ep = w.name.split("_e")[1].split("_")[0]
                print(f"[gate] {w.name}: mean={mean:.3f} 逐句={['%.3f' % s for s in sims]}", flush=True)
                with open(curve_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": int(ep), "sim": round(mean, 4),
                                        "sims": [round(s, 4) for s in sims],
                                        "weight": w.name}, ensure_ascii=False) + "\n")
                if mean >= args.target and not args.once:
                    stop_file.parent.mkdir(parents=True, exist_ok=True)
                    stop_file.write_text(f"mean_sim={mean:.3f} @ {w.name}", encoding="utf-8")
                    print(f"[gate] 达标 {mean:.3f} ≥ {args.target}, 已发早停信号", flush=True)
                    return
            except Exception as e:
                print(f"[gate] {w.name} 打分失败: {e}", flush=True)
        if args.once:
            print("[gate] --once 模式, 一轮打分完成, 退出", flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
