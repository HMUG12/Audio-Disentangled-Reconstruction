"""部署态 t2s 配对补测: s1 微调版 vs 官方预训练, 配 holdout_e8 s2。

背景: 引擎默认 tts_infer.yaml 指向 user_voice-e8.ckpt (s1 微调, 训练数据
含 ref 段)。held-out 评估三条件共享了它, 组间对比有效, 但部署态绝对值
需补测。本脚本对比两种 t2s 配 holdout_e8 vits 的 5 句探测。

用法: python scripts/eval_t2s_pairing.py
产物: output/eval_t2s_pairing.json
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REF = REPO / "data" / "voices" / "我的声音V2" / "ref.wav"
HOLDOUT_VITS = REPO / "third_party" / "gpt_sovits" / "SoVITS_weights_v2" / \
    "user_holdout_e8_s176.pth"
S1_FT = REPO / "third_party" / "gpt_sovits" / "GPT_weights_v2" / "user_voice-e8.ckpt"
S1_OFFICIAL = REPO / "third_party" / "gpt_sovits" / "GPT_SoVITS" / \
    "pretrained_models" / "gsv-v2final-pretrained" / \
    "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
OUT_JSON = REPO / "output" / "eval_t2s_pairing.json"
WAV_DIR = REPO / "output" / "heldout_wavs"

sys.path.insert(0, str(Path(__file__).parent))
from eval_heldout import PROBES  # noqa: E402


def main():
    import numpy as np
    import soundfile as sf
    from adr.eval.speaker_sim import similarity
    from adr.models.gsv_engine import GSVEngine

    assert S1_OFFICIAL.exists(), f"官方 s1 不存在: {S1_OFFICIAL}"
    eng = GSVEngine()
    eng.warmup()
    ref = str(REF)
    results = {}

    for label, t2s in [("s1_finetuned", str(S1_FT)), ("s1_official", str(S1_OFFICIAL))]:
        sims = []
        for i, text in enumerate(PROBES):
            wav, sr = eng.synthesize(text, ref, t2s_weights=t2s,
                                     vits_weights=str(HOLDOUT_VITS))
            out_p = WAV_DIR / f"t2s_{label}_p{i}.wav"
            sf.write(str(out_p), wav, sr)
            sim = similarity(ref, str(out_p))
            sims.append(sim)
            print(f"  [{label}] 探测句{i+1}: sim={sim:.4f}", flush=True)
        results[label] = {"sims": [round(s, 4) for s in sims],
                          "mean": round(float(np.mean(sims)), 4),
                          "std": round(float(np.std(sims)), 4)}
        print(f"[eval] {label}: mean={np.mean(sims):.4f} ± {np.std(sims):.4f}")

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[eval] 已写 {OUT_JSON}")


if __name__ == "__main__":
    main()
