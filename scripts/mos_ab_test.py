"""主观听感验收 (MOS A/B 盲测) — 给 0.80 客观分以听感背书。

流程:
  1) 生成: python scripts/mos_ab_test.py --voice <档案名> [--ref wav] [--texts file]
     每句 × 两系统 (zero-shot 官方预训练 / fine-tuned 档案权重) 合成,
     A/B 随机混淆命名, 生成 score_sheet.csv 评分模板 (附客观 ERes2Net 分到 answer_key)
  2) 听测: 对照 ref 逐样本打分 (naturalness/similarity 各 1-5), 填入 csv
  3) 汇总: python scripts/mos_ab_test.py --score output/mos_ab/<ts>
     按系统聚合 MOS + 揭晓 A/B 映射 + 客观/主观对照

评分标准 (打印在模板头部):
  naturalness 自然度: 1=机器感重 / 3=可接受 / 5=接近真人录音
  similarity  相似度: 1=明显不同人 / 3=像但听得出差距 / 5=几乎一样

输出: output/mos_ab/<ts>/  (samples + score_sheet.csv + answer_key.json)
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# 默认句集: 门禁 4 句 (长/中) + 短句 2 句 (高频场景弱项)
DEFAULT_SENTS = [
    "地面部队已恢复对该区域的控制，根据推演，后续局势将逐步稳定。",
    "感谢您的收听，我们下期再见。",
    "您好，请问有什么可以帮您的吗？",
    "今天天气怎么样？",
    "好的，没问题。",
    "收到，马上出发。",
]

SYSTEMS = ["zeroshot", "finetuned"]


def cmd_generate(args):
    import soundfile as sf
    from adr.eval.speaker_sim import similarity
    from adr.models.gsv_engine import get_gsv_engine
    from adr.models.voice_library import load_voice

    prof = load_voice(args.voice)
    ref = args.ref or prof["ref_audio"]
    prompt_text = prof.get("prompt_text", "")
    ft_t2s = prof.get("t2s_weights")
    ft_vits = prof.get("vits_weights")
    assert ft_vits, f"档案「{args.voice}」无微调权重 (vits_weights), 无法对比微调效果"

    # 官方预训练 (zero-shot 侧; 与门禁评估口径一致)
    pre = REPO / "third_party" / "gpt_sovits" / "GPT_SoVITS" / "pretrained_models"
    zs_t2s = str(pre / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
    zs_vits = str(pre / "gsv-v2final-pretrained" / "s2G2333k.pth")
    # 每系统每句都显式传权重 — 引擎 None=保持当前状态, 不显式会让 AB 混同
    weight_kw = {
        "zeroshot": {"t2s_weights": zs_t2s, "vits_weights": zs_vits},
        "finetuned": {"t2s_weights": ft_t2s or zs_t2s, "vits_weights": ft_vits},
    }

    sents = ([ln.strip() for ln in Path(args.texts).read_text(encoding="utf-8").splitlines()
              if ln.strip()] if args.texts else DEFAULT_SENTS)

    out_dir = REPO / "output" / "mos_ab" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    eng = get_gsv_engine()
    eng.warmup(vits_weights=ft_vits, t2s_weights=ft_t2s or zs_t2s)  # 消首次罚金

    answer = {"ref": ref, "voice": args.voice, "seed": args.seed, "samples": {}}
    rows = []
    for i, text in enumerate(sents, 1):
        sys_wavs = {}
        for sysname in SYSTEMS:
            wav, sr = eng.synthesize(text, ref, prompt_text=prompt_text,
                                     seed=args.seed, **weight_kw[sysname])
            tmp = out_dir / f"_tmp_{sysname}.wav"
            sf.write(str(tmp), wav, sr)
            sim = similarity(ref, str(tmp))
            sys_wavs[sysname] = (tmp, sim)
        # A/B 随机混淆 (seed 固定可复现)
        order = SYSTEMS[:]
        random.Random(args.seed + i).shuffle(order)
        for slot, sysname in zip(("A", "B"), order):
            final = out_dir / f"sample_{i:02d}_{slot}.wav"
            sys_wavs[sysname][0].rename(final)
            answer["samples"][f"sample_{i:02d}_{slot}"] = {
                "system": sysname, "text": text,
                "obj_sim": round(sys_wavs[sysname][1], 4),
            }
            rows.append([f"sample_{i:02d}_{slot}", final.name, "", "", ""])
        print(f"[gen] sample_{i:02d} (zeroshot obj={sys_wavs['zeroshot'][1]:.3f} "
              f"finetuned obj={sys_wavs['finetuned'][1]:.3f}) {text[:18]!r}", flush=True)

    answer["sents"] = sents
    (out_dir / "answer_key.json").write_text(
        json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "听感盲测评分说明\n"
        "================\n"
        "对照 ref.wav 逐样本听, 在 score_sheet.csv 的 naturalness / similarity 列填 1-5 分:\n"
        "  naturalness 自然度: 1=机器感重 / 3=可接受 / 5=接近真人录音\n"
        "  similarity  相似度: 1=明显不同人 / 3=像但听得出差距 / 5=几乎一样\n"
        "填完把 score_sheet.csv 重命名为 score_sheet_filled.csv, 然后跑汇总:\n"
        f"  python scripts/mos_ab_test.py --score {out_dir.name}\n"
        "(样本顺序已随机混淆, answer_key.json 最后再看的)", encoding="utf-8")
    sheet = out_dir / "score_sheet.csv"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "file", "naturalness", "similarity", "comment"])
        w.writerows(rows)
    print(f"\n[done] {len(sents)*2} 个盲测样本 → {out_dir}")
    print(f"[next] 听测后填 {sheet.name}, 再跑: "
          f"python scripts/mos_ab_test.py --score {out_dir.name}")


def cmd_score(args):
    import statistics

    d = Path(args.score)
    sheet_p = d / (args.sheet or "score_sheet_filled.csv")
    key_p = d / "answer_key.json"
    key = json.loads(key_p.read_text(encoding="utf-8"))

    scores = {}
    with open(sheet_p, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if (len(row) >= 4 and row[0].startswith("sample_")
                    and row[0] != "sample_id" and row[2].strip() and row[3].strip()):
                scores[row[0]] = (float(row[2]), float(row[3]))
    assert scores, f"评分表无有效数据行 (检查 {sheet_p} 是否已填分)"

    agg = {s: {"n": [], "s": [], "obj": []} for s in SYSTEMS}
    for sid, (n, s) in scores.items():
        info = key["samples"][sid]
        agg[info["system"]]["n"].append(n)
        agg[info["system"]]["s"].append(s)
        agg[info["system"]]["obj"].append(info["obj_sim"])

    print(f"\n===== 听感验收汇总 ({key['voice']}, seed={key['seed']}, "
          f"{len(scores)} 样本已评) =====")
    print(f"{'系统':<12}{'MOS-自然度':<12}{'MOS-相似':<10}{'客观ERes2Net':<14}")
    for s in SYSTEMS:
        a = agg[s]
        if not a["n"]:
            print(f"{s:<12}(无评分)")
            continue
        print(f"{s:<12}{statistics.mean(a['n']):<12.2f}"
              f"{statistics.mean(a['s']):<10.2f}"
              f"{statistics.mean(a['obj']):<14.3f}")

    print("\n===== A/B 揭晓 =====")
    for sid in sorted(key["samples"]):
        info = key["samples"][sid]
        sc = scores.get(sid, ("-", "-"))
        print(f"  {sid}: {info['system']:<10} "
              f"自然={sc[0]} 相似={sc[1]} 客obj={info['obj_sim']}")

    if agg["zeroshot"]["s"] and agg["finetuned"]["s"]:
        ds = statistics.mean(agg["finetuned"]["s"]) - statistics.mean(agg["zeroshot"]["s"])
        dn = statistics.mean(agg["finetuned"]["n"]) - statistics.mean(agg["zeroshot"]["n"])
        verdict = ("微调显著更优" if ds > 0.3 else
                   "微调略优" if ds > 0 else "微调无主观优势 (需复查)")
        print(f"\n[结论] 微调 vs zero-shot: 相似 {ds:+.2f}, 自然度 {dn:+.2f} → {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default="我的声音V2", help="音色档案名 (fine-tuned 侧)")
    ap.add_argument("--ref", default=None, help="参考音频 (默认用档案 ref.wav)")
    ap.add_argument("--texts", default=None, help="句子文件 (每行一句; 默认内置 6 句)")
    ap.add_argument("--seed", type=int, default=11, help="合成种子 + A/B 混淆基种子")
    ap.add_argument("--score", default=None, help="汇总模式: output/mos_ab/<ts> 目录")
    ap.add_argument("--sheet", default=None, help="评分表文件名 (默认 score_sheet_filled.csv)")
    args = ap.parse_args()

    if args.score:
        cmd_score(args)
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()
