"""W1: 首包方差验证 — head_seed 固定首段采样 vs 随机。

背景: 流式首包实测 2.7~4.9s 波动, 来源=语义 token 自回归采样长度随 seed 变化。
引擎新增 head_seed 参数 (只固定首段)。

用法: python scripts/bench_stream_variance.py
产物: output/bench_stream_variance.json
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "output" / "bench_stream_variance.json"

TEXT = ("声音克隆是一项非常有意思的技术，它只需要几秒钟的参考音频，"
        "就能够生成非常接近真人的语音效果，让我们一起来看看吧。")
REPEATS = 3


def main():
    from adr.models.gsv_engine import GSVEngine
    from adr.models.voice_library import load_voice

    prof = load_voice("我的声音V2")
    eng = GSVEngine()
    # 用档案权重预热 — 否则首次流式调用要付 ~12s init_vits_weights 罚金
    eng.warmup(vits_weights=prof.get("vits_weights"),
               t2s_weights=prof.get("t2s_weights"))
    print("[bench] 引擎已预热 (含档案权重热换)")

    results = {"text": TEXT[:20] + "...", "repeats": REPEATS, "runs": []}
    for label, seed in [("seeded", 42), ("random", -1)]:
        firsts = []
        for r in range(REPEATS):
            t0 = time.perf_counter()
            t_first = None
            for chunk, sr in eng.synthesize_stream(
                    TEXT, prof["ref_audio"], vits_weights=prof.get("vits_weights"),
                    split_method="cut3", head_seed=seed):
                if t_first is None:
                    t_first = time.perf_counter() - t0
            firsts.append(round(t_first, 2))
            print(f"  [{label}] 第{r+1}次: 首包 {t_first:.2f}s", flush=True)
        results["runs"].append({"label": label, "firsts": firsts,
                                "spread": round(max(firsts) - min(firsts), 2)})

    seeded = results["runs"][0]["firsts"]
    print(f"[bench] seeded 极差 {max(seeded)-min(seeded):.2f}s "
          f"(应为 0 或接近 0) | random 极差 "
          f"{results['runs'][1]['spread']:.2f}s")
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
