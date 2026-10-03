"""批次5-2: 短句克隆质量实测 — 定框架级策略 (不重训模型)。

背景: 门禁/评估反复显示短句 (≤14 字) 相似度比长句低 5-10 个点
(0.688~0.747 vs 0.79+), 短句是用户高频场景 (助手应答/播报)。
可疑因素: ① GPT AR 采样方差 (短文本上下文少, 一锤定音) ② GSV
对首逗号前 <4 字的文本自动加前导"。" (TextPreprocessor) ③ top_k=15
宽松采样在短句上更易跑偏。

实验矩阵 (有界): 4 短句 × 5 seeds × {top_k 15 / 5} ≈ 40 次合成,
以 ERes2Net 相似度为指标, 输出各条件均值/方差 → 定引擎短句策略。

用法: python scripts/bench_short_sent.py --ref <参考音频>
输出: output/bench_short_sent.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SENTS = [
    "您好，请问有什么可以帮您的吗？",   # 门禁探测句 3 (最弱, 前导"。"触发)
    "好的，没问题。",                   # 6 字, 前导"。"触发
    "今天天气怎么样？",                 # 8 字, 无逗号 → 不触发前导
    "感谢您的收听，我们下期再见。",     # 14 字对照
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(REPO / "data" / "voices" / "我的声音V2" / "ref.wav"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--topk", type=int, nargs="+", default=[15, 5])
    args = ap.parse_args()

    import soundfile as sf
    from adr.eval.speaker_sim import similarity
    from adr.models.gsv_engine import get_gsv_engine
    from adr.models.voice_library import load_voice

    eng = get_gsv_engine()
    # 用部署档案权重热换预热 (对齐真实使用状态), 并消除首次罚金
    prof = load_voice("我的声音V2")
    eng.warmup(vits_weights=prof.get("vits_weights"),
               t2s_weights=prof.get("t2s_weights"))
    tmp = REPO / "output" / "_short_sent_probe.wav"
    results = {}

    for topk in args.topk:
        for si, text in enumerate(SENTS):
            sims = []
            for seed in args.seeds:
                wav, sr = eng.synthesize(text, args.ref, seed=seed, top_k=topk)
                sf.write(str(tmp), wav, sr)
                sims.append(similarity(args.ref, str(tmp)))
            key = f"topk{topk}_s{si}"
            results[key] = {
                "text": text, "top_k": topk,
                "mean": round(statistics.mean(sims), 4),
                "std": round(statistics.pstdev(sims), 4),
                "sims": [round(s, 4) for s in sims],
            }
            print(f"[topk={topk}] {text!r}: mean={results[key]['mean']:.3f} "
                  f"std={results[key]['std']:.3f} 逐seed={results[key]['sims']}",
                  flush=True)

    # 前导"。"对照: 同句手工前置"。"对比 (验证 GSV 自动前缀是否伤分)
    text = SENTS[1]
    sims = []
    for seed in args.seeds[:3]:
        wav, sr = eng.synthesize("。" + text, args.ref, seed=seed)
        sf.write(str(tmp), wav, sr)
        sims.append(similarity(args.ref, str(tmp)))
    results["leading_period_s1"] = {
        "text": "。" + text, "mean": round(statistics.mean(sims), 4),
        "std": round(statistics.pstdev(sims), 4),
        "sims": [round(s, 4) for s in sims],
    }
    print(f"[对照] 前导。: mean={results['leading_period_s1']['mean']:.3f}", flush=True)

    out = REPO / "output" / "bench_short_sent.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
