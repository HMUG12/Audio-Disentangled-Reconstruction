"""held-out 评估: 量化训练-评估同源污染对克隆相似度分数的影响。

背景 (2026-10-02 审计发现):
  原配方 0.807 的评分参考 ref.wav (录音前 7s) 与训练数据同源 (同一段
  8月15日.mp3), 分数可能包含"复述训练素材"的成分。

方法:
  剔除与 ref 重叠的切片 (前 16.5s), 用剩余 ~58s 重训 user_holdout,
  同一 ref、同一组探测句下对比三组条件:
    A zero_shot    零样本 (官方预训练)
    B contaminated 污染权重 (user_voice, 训练集含 ref 段)
    C clean        干净权重 (user_holdout, 训练集与 ref 零重叠)
  另测 sim(ref, 真实录音切片) 作为同说话人真音频锚点。

用法: python scripts/eval_heldout.py
产物: output/heldout_eval.json + output/heldout_wavs/ (可试听)
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REF = REPO / "data" / "voices" / "我的声音V2" / "ref.wav"
CONTAMINATED_VITS = REPO / "third_party" / "gpt_sovits" / "SoVITS_weights_v2" / \
    "user_voice_e2_s36.pth"
HOLDOUT_DIR = REPO / "third_party" / "gpt_sovits" / "SoVITS_weights_v2"
SLICES_DIR = REPO / "third_party" / "gpt_sovits" / "output" / "slicer_opt" / "user_voice"
OUT_JSON = REPO / "output" / "heldout_eval.json"
WAV_DIR = REPO / "output" / "heldout_wavs"

PROBES = [
    "人工智能技术正在以前所未有的速度发展，语音合成系统已经能够生成接近真人水平的自然流畅音频。",
    "今天早上我喝了一杯咖啡，然后在公园里散了一会儿步，天气非常不错。",
    "您好，请问有什么可以帮您的吗？",
    "这是一段用于测试声音克隆效果的文本，包含数字一二三四五和英文 Hello World。",
    "夜色渐深，城市的灯火次第亮起，街道上行人渐渐稀少。",
]


def find_clean_weights() -> Path:
    ws = sorted(HOLDOUT_DIR.glob("user_holdout_e*.pth"), key=lambda p: p.stat().st_mtime)
    assert ws, "未找到 user_holdout 权重 (先跑 gsv_finetune.py --exp user_holdout)"
    return ws[-1]


def main():
    import numpy as np
    import soundfile as sf
    from adr.eval.speaker_sim import similarity
    from adr.models.gsv_engine import GSVEngine

    clean_w = find_clean_weights()
    print(f"[eval] 干净权重: {clean_w.name}")
    print(f"[eval] 污染权重: {CONTAMINATED_VITS.name}")

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[eval] GPU: {gpu}")

    eng = GSVEngine()
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    ref = str(REF)

    # 真实语音锚点: 真人录音切片 vs ref (slice2 已从 holdout 训练集剔除, 但与 ref 部分重叠)
    slice2 = next(SLICES_DIR.glob("*0000222400_0001656640.wav"), None)
    real_anchor = similarity(ref, str(slice2)) if slice2 else None
    print(f"[eval] 真实语音锚点 sim(ref, 真切片#2, 与ref部分重叠) = {real_anchor:.4f}")

    conditions = {
        "zero_shot": {"vits_weights": None},
        "contaminated": {"vits_weights": str(CONTAMINATED_VITS)},
        "clean": {"vits_weights": str(clean_w)},
    }
    results = {"gpu": gpu, "ref": str(REF), "real_anchor": round(real_anchor, 4),
               "clean_weights": clean_w.name, "probes": PROBES, "conditions": {}}

    for cond, kw in conditions.items():
        sims = []
        for i, text in enumerate(PROBES):
            t0 = time.time()
            wav, sr = eng.synthesize(text, ref, **kw)
            out_p = WAV_DIR / f"{cond}_p{i}.wav"
            sf.write(str(out_p), wav, sr)
            sim = similarity(ref, str(out_p))
            sims.append(sim)
            print(f"  [{cond}] 探测句{i+1}: sim={sim:.4f} ({time.time()-t0:.1f}s)")
        results["conditions"][cond] = {
            "sims": [round(s, 4) for s in sims],
            "mean": round(float(np.mean(sims)), 4),
            "std": round(float(np.std(sims)), 4),
        }
        print(f"[eval] {cond}: mean={np.mean(sims):.4f} ± {np.std(sims):.4f}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[eval] 结果已写 {OUT_JSON}")


if __name__ == "__main__":
    main()
