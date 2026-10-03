"""C4: 训练内相似度门禁 — 监控 s2 权重产出, 逐轮克隆打分, 达标早停。

与 gsv_finetune.py 并行运行:
    python scripts/train_gate.py --exp user_voice --target 0.80

机制:
  1. 轮询 SoVITS_weights_v2/<exp>_e*.pth, 新权重出现即克隆+ERes2Net 打分
  2. 每个权重用 5 句探测集 (与 eval_heldout.py 同源) 逐句打分取均值 —
     单句探测方差大 (短句天然弱, 波动 ±0.03+), 均值更稳
  3. 分数曲线写 output/train_gate_<exp>.jsonl (含逐句分)
  4. mean sim ≥ target → 在 logs/<exp>/ 下放 STOP 信号文件 → s2_train 每轮末检查后退出
"""
import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GSV = REPO / "third_party" / "gpt_sovits"

# 与 scripts/eval_heldout.py PROBES 保持同源 (长/中/短/混合/叙事 全覆盖)
PROBES = [
    "人工智能技术正在以前所未有的速度发展，语音合成系统已经能够生成接近真人水平的自然流畅音频。",
    "今天早上我喝了一杯咖啡，然后在公园里散了一会儿步，天气非常不错。",
    "您好，请问有什么可以帮您的吗？",
    "这是一段用于测试声音克隆效果的文本，包含数字一二三四五和英文 Hello World。",
    "夜色渐深，城市的灯火次第亮起，街道上行人渐渐稀少。",
]

# 显式钉死官方预训练 s1/s2: tts_infer.yaml 的 custom 段可能被外部改写指向
# 微调权重 (实测发生过), 隐式默认会让"零样本基线"和打分条件漂移
PRE_T2S = str(GSV / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
              / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
PRE_S2G = str(GSV / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
              / "s2G2333k.pth")


def _score_probes(eng, ref: str, vits: str = PRE_S2G, t2s: str = PRE_T2S,
                  sents=None):
    """5 句探测逐句克隆打分, 返回 (均值, 逐句分列表)。

    s1 固定用官方预训练 (与 --skip-s1 训练条件一致), s2 传被测权重 —
    不传任何权重时即真零样本基线。
    sents: 自定义句集 (批次 8, 人设定制门禁) — 基线与训练后必须同句集,
    否则分数不可比。
    """
    from adr.eval.speaker_sim import similarity
    sims = []
    for text in (sents or PROBES):
        wav, sr = eng.synthesize(text, ref, t2s_weights=t2s, vits_weights=vits)
        sims.append(similarity(ref, str(_save_tmp(wav, sr))))
    return sum(sims) / len(sims), sims


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
    base, base_sims = _score_probes(eng, args.ref, sents=sents)
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
                mean, sims = _score_probes(eng, args.ref, vits=str(w), sents=sents)
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


def _save_tmp(wav, sr):
    import soundfile as sf
    p = REPO / "output" / "_gate_probe.wav"
    sf.write(str(p), wav, sr)
    return p


if __name__ == "__main__":
    main()
