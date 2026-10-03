"""批次5-1: 端到端管线分阶段计时审计 (10 切片 / ~3 分钟音频实测)。

目标: 对齐 "7 秒参考 + ≤30 分钟 → ≥80% 相似度" 承诺, 找框架可压缩点。
不训练模型 — s2 训练逐轮耗时用 train.log 考古 (见下), 本脚本只实测
数据准备各阶段 + s2 启动 A/B (num_workers 5 vs 0)。

考古基线 (logs/train.log, 同一 10 切片数据集):
- holdout bs4:  27s/epoch 稳态, 8ep 全程 7.0min (含启动 3.2min)
- h4gb bs1+cap10s: 105s/epoch, 3ep 全程 11.7min (含启动 ~6.2min*)
  * 首轮启动被并行门禁打分拖慢, 纯启动估计 3-4min
- 启动开销 = 5 worker × Windows spawn 重导入 torch + 首批 — 小数据集最大浪费

用法: python scripts/bench_pipeline_stages.py [--skip-s2ab]
输出: output/bench_pipeline_stages.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GSV = REPO / "third_party" / "gpt_sovits"
PY = sys.executable
EXP = "timing_audit"
PRE = GSV / "GPT_SoVITS" / "pretrained_models"
PRE_GSV = PRE / "gsv-v2final-pretrained"

SLICED = GSV / "output" / "slicer_opt" / EXP
ASR_OUT = GSV / "output" / "asr_opt"
LIST_FILE = ASR_OUT / f"{EXP}.list"
LOGS = GSV / "logs" / EXP
SRC_SLICES = GSV / "output" / "slicer_opt" / "user_holdout"  # 现成 10 切片


def _run(cmd, desc, env=None, cwd=GSV):
    e = {**os.environ, **(env or {}), "PYTHONIOENCODING": "utf-8"}
    # 官方脚本依赖从仓库根 import (tools./GPT_SoVITS.), 同 gsv_finetune.run()
    e["PYTHONPATH"] = f"{GSV}{os.pathsep}{GSV / 'GPT_SoVITS'}" + \
        (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(cwd), env=e,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"[FAIL] {desc} (rc={r.returncode})", flush=True)
        sys.exit(1)
    print(f"[stage] {desc}: {dt:.1f}s", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-s2ab", action="store_true",
                    help="跳过 s2 启动 A/B (只要数据准备阶段耗时)")
    ap.add_argument("--s2ab-only", action="store_true",
                    help="只跑 s2 启动 A/B (复用已有 timing_audit 产物)")
    ap.add_argument("--workers-list", nargs="+", default=["5", "0"],
                    help="A/B 的 worker 数列表 (默认 5 和 0 各一轮)")
    args = ap.parse_args()

    if not args.s2ab_only:
        for d in (SLICED, LOGS):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        LIST_FILE.unlink(missing_ok=True)
    else:
        LOGS.mkdir(parents=True, exist_ok=True)

    results = {"dataset": {"src_slices": SRC_SLICES.name}}

    # 音频总量 (秒)
    import soundfile as sf
    srcs = sorted(SRC_SLICES.glob("*.wav"))
    total_s = sum(len(sf.read(str(p), dtype="float32")[0]) /
                  sf.read(str(p), dtype="float32")[1] for p in srcs)
    results["dataset"]["n_slices"] = len(srcs)
    results["dataset"]["total_audio_min"] = round(total_s / 60, 2)
    print(f"[data] {len(srcs)} 切片, 共 {total_s/60:.1f} 分钟音频", flush=True)

    # 1. slice: 先拼回一条 ~3min 长音频 (模拟用户原始录音), 再切
    common_env = {
        "inp_text": str(LIST_FILE), "inp_wav_dir": str(SLICED),
        "exp_name": EXP, "i_part": "0", "all_parts": "1",
        "opt_dir": str(LOGS), "is_half": "True", "version": "v2",
    }
    if not args.s2ab_only:
        import numpy as np
        wavs = []
        sr0 = None
        for p in srcs:
            w, sr = sf.read(str(p), dtype="float32")
            wavs.append(w)
            sr0 = sr
        raw_dir = GSV / "TEMP" / "timing_audit_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_wav = raw_dir / "raw_concat.wav"
        sf.write(str(raw_wav), np.concatenate(wavs), sr0)
        results["slice_s"] = round(_run(
            [PY, "-s", "tools/slice_audio.py", str(raw_wav), str(SLICED),
             "-34", "4000", "300", "10", "500", "0.9", "0.25", "0", "1"],
            "1/6 静音切片"), 1)

        # 2. asr
        results["asr_s"] = round(_run(
            [PY, "-s", "tools/asr/fasterwhisper_asr.py", "-i", str(SLICED),
             "-o", str(ASR_OUT), "-s", "medium", "-l", "zh", "-p", "float16"],
            "2/6 ASR 转录 (fasterwhisper medium)"), 1)
        # fasterwhisper 以输入目录名命名 list → 重命名对齐 exp
        auto_list = ASR_OUT / f"{SLICED.name}.list"
        if auto_list.exists() and auto_list != LIST_FILE:
            auto_list.replace(LIST_FILE)

        # 3. text+bert
        results["text_s"] = round(_run(
            [PY, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"],
            "3/6 文本+BERT 特征",
            env={**common_env, "bert_pretrained_dir": str(PRE / "chinese-roberta-wwm-ext-large")}), 1)
        # 4. hubert
        results["hubert_s"] = round(_run(
            [PY, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
            "4/6 HuBERT 特征",
            env={**common_env, "cnhubert_base_dir": str(PRE / "chinese-hubert-base")}), 1)
        # 5. semantic
        results["semantic_s"] = round(_run(
            [PY, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
            "5/6 语义 token",
            env={**common_env, "pretrained_s2G": str(PRE_GSV / "s2G2333k.pth"),
                 "s2config_path": str(GSV / "GPT_SoVITS" / "configs" / "s2.json")}), 1)
        for src, dst in [(LOGS / "2-name2text-0.txt", LOGS / "2-name2text.txt"),
                         (LOGS / "6-name2semantic-0.tsv", LOGS / "6-name2semantic.tsv")]:
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    # 6. s2 启动 A/B: 1 epoch, workers 5 vs 0 (补丁 #19 ADR_S2_NUM_WORKERS)
    if not args.skip_s2ab:
        cfg = json.loads((GSV / "GPT_SoVITS" / "configs" / "s2.json").read_text())
        cfg["train"].update({
            "batch_size": 4, "epochs": 1, "fp16_run": True, "grad_ckpt": True,
            "pretrained_s2G": str(PRE_GSV / "s2G2333k.pth"),
            "pretrained_s2D": str(PRE_GSV / "s2D488k.pth"),
            "if_save_latest": True, "if_save_every_weights": True,
            "save_every_epoch": 1, "gpu_numbers": "0", "lora_rank": 32,
        })
        cfg["data"].update({"exp_dir": str(LOGS)})  # 训练列表由 exp_dir 产物推导 (data_utils)
        cfg["s2_ckpt_dir"] = str(LOGS)
        cfg["save_weight_dir"] = "SoVITS_weights_v2"
        cfg["name"] = EXP
        cfg["version"] = "v2"
        cfg["model"]["version"] = "v2"  # TextAudioSpeakerLoader 读 hps.model.version
        tmp_cfg = GSV / "TEMP" / "tmp_s2_timing.json"
        tmp_cfg.parent.mkdir(exist_ok=True)
        tmp_cfg.write_text(json.dumps(cfg), encoding="utf-8")
        for workers in args.workers_list:
            # 清 ckpt 保证两轮等价; logs_s2_v2 必须重建 (保存目标目录)
            ck = LOGS / "logs_s2_v2"
            if ck.exists():
                shutil.rmtree(ck)
            ck.mkdir(parents=True)
            for w in (GSV / "SoVITS_weights_v2").glob(f"{EXP}_e*.pth"):
                w.unlink()
            results[f"s2_1ep_workers{workers}_s"] = round(_run(
                [PY, "-s", "GPT_SoVITS/s2_train.py", "--config", str(tmp_cfg)],
                f"6/6 s2 1 epoch (workers={workers})",
                env={**common_env, "ADR_S2_NUM_WORKERS": workers}), 1)
        for w in (GSV / "SoVITS_weights_v2").glob(f"{EXP}_e*.pth"):
            w.unlink()  # 清实测产物

    # 汇总: 折算到每分钟音频 + 30min 预算对照 (含考古的训练/门禁数据)
    per_min = {k: round(v / (total_s / 60), 1)
               for k, v in results.items() if k.endswith("_s")}
    results["per_min_audio"] = per_min
    results["archaeology"] = {
        "s2_epoch_bs4_s": 27, "s2_epoch_bs1_cap10_s": 105,
        "startup_workers5_s_min": "192-369 (两轮实测 3.2-6.2min)",
        "gate_per_ckpt_s": "~90 (5 句 CPU 打分 ~60s + 引擎开销)",
    }
    out = REPO / "output" / "bench_pipeline_stages.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
