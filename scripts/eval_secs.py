"""SECS 档案评估 — 音色保持度的客观口径 (Speaker Embedding Cosine Similarity)。

对单个档案做 zero-shot 基线 vs 微调绑定权重 的同句集对比, 可选真实语音锚点,
输出 JSON 报告 + 可试听 wav:
    python scripts/eval_secs.py --voice 我的声音V2 [--cpu] [--anchor ref.wav]

评估口径 (批次 51 公信力修正 — 早期文档规划的 PESQ/ESTOI 不适用于零样本
TTS 评估: 需要平行语料且度量的是信号保真而非音色保持), 实际三件套:
  - SECS  本脚本: ERes2NetV2 说话人余弦相似度, 对 ref 的音色保持
          (>0.70 明显同一人 / 0.55-0.70 接近 / <0.45 基本不同人)
  - MOS   scripts/mos_ab_test.py: A/B 盲测人评 (零样本 vs 微调)
  - CER   faster-whisper 回译: scripts/render_sample_pack.py
          (adr.training.callbacks.compute_cer 同源), 可懂度
产物: output/eval_secs_<voice>.json + output/eval_secs_wavs/<voice>/ (可试听)
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    from adr.models.voice_library import load_voice
    from adr.eval.probes import PRE_S2G, PRE_T2S, PROBES, clip_ref

    ap = argparse.ArgumentParser(description="SECS 档案评估 (音色保持度)")
    ap.add_argument("--voice", required=True, help="音色档案名 (data/voices/<name>)")
    ap.add_argument("--ref", default=None, help="覆盖参考音频 (默认档案 ref.wav)")
    ap.add_argument("--weights", default=None,
                    help="显式指定被测 s2 权重 (默认档案绑定的 vits_weights)")
    ap.add_argument("--anchor", default=None,
                    help="真实语音锚点 wav — sim(ref, 真人音频) 应高于合成条件, "
                         "验证打分链路灵敏度")
    ap.add_argument("--texts", default=None,
                    help="自定义句集文件 (每行一句, ≥3 句) — 基线与微调自动同句集")
    ap.add_argument("--cpu", action="store_true",
                    help="合成/打分引擎走 CPU (无 GPU 环境复现用, 慢点无妨)")
    args = ap.parse_args()

    meta = load_voice(args.voice)  # 含 is_safe_name 校验
    ref = args.ref or meta["ref_audio"]
    vits = args.weights or meta.get("vits_weights")
    if not vits:
        sys.exit(f"档案「{args.voice}」无绑定微调权重 (meta.vits_weights 为空), "
                 f"只能测零样本 — 用 scripts/train_gate.py --once 或传 --weights")

    sents = None
    if args.texts:
        p_texts = Path(args.texts)
        if not p_texts.exists():
            sys.exit(f"--texts 文件不存在: {args.texts}")
        sents = [ln.strip() for ln in p_texts.read_text(
            encoding="utf-8-sig").splitlines() if ln.strip()]
        if len(sents) < 3:  # 显式校验 (assert 在 python -O 下会被剥离)
            sys.exit(f"自定义句集至少 3 句, 当前 {len(sents)} ({args.texts})")
        print(f"[secs] 自定义句集 {len(sents)} 句 (来自 {args.texts})")

    from adr.models.gsv_engine import GSVEngine, GSVEngineConfig
    if args.cpu:
        eng = GSVEngine(GSVEngineConfig(device="cpu", half=False))
    else:
        eng = GSVEngine()

    wav_dir = REPO / "output" / "eval_secs_wavs" / args.voice
    wav_dir.mkdir(parents=True, exist_ok=True)
    for old in wav_dir.glob("*_p*.wav"):  # 清上次产物, 句数变少时避免新旧混淆
        old.unlink()
    ref = clip_ref(ref, args.voice)

    import numpy as np
    import soundfile as sf
    from adr.eval.speaker_sim import similarity

    def _run_cond(kind: str, vits_path: str) -> dict:
        """单条件: 逐句合成 → 落盘试听 wav → SECS 打分。"""
        sims = []
        for i, text in enumerate(sents_probe):
            t0 = time.time()
            wav, sr = eng.synthesize(text, ref, t2s_weights=PRE_T2S,
                                     vits_weights=vits_path)
            out_p = wav_dir / f"{kind}_p{i}.wav"
            sf.write(str(out_p), wav, sr)
            sim = similarity(ref, str(out_p))
            sims.append(sim)
            print(f"  [{kind}] 句{i + 1}/{len(sents_probe)}: SECS={sim:.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        return {"sims": [round(s, 4) for s in sims],
                "mean": round(float(np.mean(sims)), 4),
                "std": round(float(np.std(sims)), 4)}

    sents_probe = sents or PROBES  # 基线与微调必须同句集
    print(f"[secs] 档案: {args.voice}  ref: {Path(ref).name}  "
          f"被测权重: {Path(vits).name}", flush=True)

    # 真实语音锚点: 真人音频 vs ref — 合成条件分数应低于它 (打分链路灵敏度)
    anchor_sim = None
    if args.anchor:
        anchor_sim = similarity(ref, args.anchor)
        print(f"[secs] 锚点 sim(ref, {Path(args.anchor).name}) = {anchor_sim:.4f}",
              flush=True)

    conditions = {
        "zero_shot": {"vits_path": PRE_S2G},
        "finetuned": {"vits_path": str(vits)},
    }
    results = {
        "voice": args.voice,
        "ref": str(ref),
        "weights": str(vits),
        "probe_source": args.texts or "adr.eval.probes.PROBES",
        "sents": sents_probe,
        "anchor": ({"path": args.anchor, "secs_vs_ref": round(anchor_sim, 4)}
                   if anchor_sim is not None else None),
        "conditions": {},
    }
    for kind, kw in conditions.items():
        print(f"[secs] 条件: {kind}", flush=True)
        results["conditions"][kind] = _run_cond(kind, kw["vits_path"])
        m = results["conditions"][kind]["mean"]
        print(f"[secs] {kind}: mean={m:.4f}", flush=True)

    zs = results["conditions"]["zero_shot"]["mean"]
    ft = results["conditions"]["finetuned"]["mean"]
    results["delta"] = round(ft - zs, 4)
    results["device"] = "cpu" if args.cpu else "cuda"

    out_json = REPO / "output" / f"eval_secs_{args.voice}.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[secs] 结论: zero_shot={zs:.4f}  finetuned={ft:.4f}  Δ={results['delta']:+.4f}",
          flush=True)
    print(f"[secs] 结果已写 {out_json}  (wav 可试听: {wav_dir})", flush=True)


if __name__ == "__main__":
    main()
