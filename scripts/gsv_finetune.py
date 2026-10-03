"""C2: 用户声音 → GPT-SoVITS 微调一条龙 (切片→ASR→特征→s2→s1)。

用法: python scripts/gsv_finetune.py <音频> [--exp user_voice] [--skip-to STEP]
步骤: slice / asr / text / hubert / semantic / s2 / s1
产物: third_party/gpt_sovits/SoVITS_weights_v2/<exp>_e*.pth
       third_party/gpt_sovits/GPT_weights_v2/<exp>-e*.ckpt
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GSV = REPO / "third_party" / "gpt_sovits"
PY = sys.executable

PRE_GSV = GSV / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
PRE = GSV / "GPT_SoVITS" / "pretrained_models"


def run(cmd: str, env: dict = None, desc: str = ""):
    print(f"\n{'=' * 60}\n>>> {desc}\n{cmd}\n{'=' * 60}", flush=True)
    e = os.environ.copy()
    # 官方脚本依赖从仓库根 import (tools./GPT_SoVITS.), 需显式 PYTHONPATH
    e["PYTHONPATH"] = f"{GSV};{GSV / 'GPT_SoVITS'}" + \
        (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    if env:
        e.update({k: str(v) for k, v in env.items()})
    r = subprocess.run(cmd, shell=True, cwd=str(GSV), env=e)
    if r.returncode != 0:
        raise RuntimeError(f"步骤失败 ({desc}): exit={r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="原始音频 (mp3/wav)")
    ap.add_argument("--exp", default="user_voice")
    ap.add_argument("--s2-epochs", type=int, default=8)
    ap.add_argument("--s1-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4, help="8GB 显存安全值")
    ap.add_argument("--max-clip-sec", type=float, default=None,
                    help="W3: 长 clip 截断秒数 (4GB 显存训练用 10, 配 --batch-size 1~2; 不传=不截)"
                         " 需 s2 训练补丁 (apply_compat_patches.py #16/#17) 配合生效")
    ap.add_argument("--skip-to", default="slice",
                    choices=["slice", "asr", "text", "hubert", "semantic", "s2", "s1"])
    ap.add_argument("--skip-s1", action="store_true",
                    help="C2 快配方: 只微调 s2 (音色), ~7 分钟到 sim 0.80+;"
                         " s1 全量微调 ~50min 且对相似度无益")
    ap.add_argument("--resume-from", default=None, metavar="voice:名字",
                    help="C3 增量续练: 从音色档案的微调权重出发训练 (新数据叠练不重来)")
    ap.add_argument("--bind-voice", default=None,
                    help="C3 训完自动把最新权重绑定到该音色档案")
    args = ap.parse_args()

    exp = args.exp
    if args.max_clip_sec:
        os.environ["ADR_MAX_CLIP_SEC"] = str(args.max_clip_sec)

    # C3: 从音色档案续练 — 读档案的 s2 微调权重作为初始化
    resume_s2g = None
    if args.resume_from:
        assert args.resume_from.startswith("voice:"), "--resume-from 仅支持 voice:名字"
        from adr.models.voice_library import load_voice
        v = load_voice(args.resume_from[6:])
        resume_s2g = v.get("vits_weights")
        assert resume_s2g and Path(resume_s2g).exists(), \
            f"档案「{v['name']}」无微调权重, 无法续练 (先跑一次普通微调)"
        # 微调权重缺 enc_q (后验编码器, 推理不需要被 savee 丢弃, 但训练必需)
        # → 以官方预训练补全缺失键, 合并为完整初始化 ckpt
        import torch
        sys.path.insert(0, str(GSV / "GPT_SoVITS"))  # utils pickle 依赖
        try:
            official = torch.load(PRE_GSV / "s2G2333k.pth", map_location="cpu",
                                  weights_only=False)["weight"]
            finetuned = torch.load(resume_s2g, map_location="cpu",
                                   weights_only=False)["weight"]
            merged = {**official, **finetuned}
            merged_p = GSV / "TEMP" / f"resume_init_{exp}.pth"
            merged_p.parent.mkdir(exist_ok=True)
            torch.save({"weight": merged}, merged_p)
            print(f"[resume] 合并初始化 ckpt (补 enc_q 等 {len(official)-len(set(finetuned)&set(official))} 键): {merged_p}")
            resume_s2g = str(merged_p)
        finally:
            sys.path.remove(str(GSV / "GPT_SoVITS"))
        print(f"[resume] 从档案「{v['name']}」权重续练: {v.get('vits_weights')}")
    sliced = GSV / "output" / "slicer_opt" / exp
    asr_out = GSV / "output" / "asr_opt"
    list_file = asr_out / f"{exp}.list"  # fasterwhisper_asr 以输入目录名 (即 exp) 命名
    logs = GSV / "logs" / exp
    audio = str(Path(args.audio).resolve())
    # 权重输出根目录 (官方 webui 启动时创建, 这里补建)
    for d in ["SoVITS_weights_v2", "GPT_weights_v2"]:
        (GSV / d).mkdir(exist_ok=True)
    steps = ["slice", "asr", "text", "hubert", "semantic", "s2", "s1"]
    start = steps.index(args.skip_to)

    if start <= 0:
        run(f'"{PY}" -s tools/slice_audio.py "{audio}" "{sliced}" -34 4000 300 10 500 0.9 0.25 0 1',
            desc="1/7 静音切片")
    if start <= 1:
        # ASR: 中文自动转 FunASR (modelscope 下载 paraformer, 一次性的)
        run(f'"{PY}" -s tools/asr/fasterwhisper_asr.py -i "{sliced}" -o "{asr_out}" -s medium -l zh -p float16',
            desc="2/7 ASR 转录 (中文走 FunASR)")
        list_file = asr_out / f"{sliced.name}.list"

    common_env = {
        "inp_text": list_file, "inp_wav_dir": sliced, "exp_name": exp,
        "i_part": 0, "all_parts": 1, "opt_dir": logs,
        "is_half": "True", "version": "v2",
    }
    if start <= 2:
        run(f'"{PY}" -s GPT_SoVITS/prepare_datasets/1-get-text.py',
            env={**common_env,
                 "bert_pretrained_dir": PRE / "chinese-roberta-wwm-ext-large"},
            desc="3/7 文本+BERT 特征")
    if start <= 3:
        run(f'"{PY}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py',
            env={**common_env, "cnhubert_base_dir": PRE / "chinese-hubert-base"},
            desc="4/7 HuBERT 特征")
    if start <= 4:
        run(f'"{PY}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py',
            env={**common_env, "pretrained_s2G": PRE_GSV / "s2G2333k.pth",
                 "s2config_path": GSV / "GPT_SoVITS" / "configs" / "s2.json"},
            desc="5/7 语义 token")

    # 合并单分片产物 (all_parts=1 时官方 webui 也会做这步)
    import shutil
    for src, dst in [
        (logs / "2-name2text-0.txt", logs / "2-name2text.txt"),
        (logs / "6-name2semantic-0.tsv", logs / "6-name2semantic.tsv"),
    ]:
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"[merge] {src.name} -> {dst.name}")

    if start <= 5:
        # s2 (SoVITS/VITS 声学) 微调
        os.makedirs(logs / "logs_s2_v2", exist_ok=True)  # ckpt 保存需要
        # 清残留 STOP: 上次门禁早停的信号文件若不删, 本次第 1 轮就会假早停
        stale_stop = logs / "STOP"
        if stale_stop.exists():
            stale_stop.unlink()
            print(f"[finetune] 已清除残留早停信号: {stale_stop}")
        cfg = json.loads(open(GSV / "GPT_SoVITS" / "configs" / "s2.json").read())
        cfg["train"]["batch_size"] = args.batch_size
        cfg["train"]["epochs"] = args.s2_epochs
        cfg["train"]["fp16_run"] = True
        cfg["train"]["pretrained_s2G"] = resume_s2g or str(PRE_GSV / "s2G2333k.pth")
        cfg["train"]["pretrained_s2D"] = str(PRE_GSV / "s2D2333k.pth")
        cfg["train"]["if_save_latest"] = True
        cfg["train"]["if_save_every_weights"] = True
        cfg["train"]["save_every_epoch"] = 1
        cfg["train"]["grad_ckpt"] = True     # 梯度检查点省显存
        cfg["train"]["gpu_numbers"] = "0"    # s2_train 需要 (webui 默认填)
        cfg["train"]["text_low_lr_rate"] = 0.4
        cfg["train"]["lora_rank"] = 32
        cfg["model"]["version"] = "v2"
        cfg["data"]["exp_dir"] = str(logs)
        cfg["s2_ckpt_dir"] = str(logs)
        cfg["save_weight_dir"] = "SoVITS_weights_v2"
        cfg["name"] = exp
        cfg["version"] = "v2"
        tmp_cfg = GSV / "TEMP" / "tmp_s2.json"
        tmp_cfg.parent.mkdir(exist_ok=True)
        tmp_cfg.write_text(json.dumps(cfg), encoding="utf-8")
        run(f'"{PY}" -s GPT_SoVITS/s2_train.py --config "{tmp_cfg}"',
            desc=f"6/7 SoVITS 微调 ({args.s2_epochs} epochs)")

        # C3: 训完自动把最新 s2 权重绑定到音色档案 (不存在则自动建档, 闭环断点修复)
        if args.bind_voice:
            ws = list((GSV / "SoVITS_weights_v2").glob(f"{exp}_e*.pth"))
            ws.sort(key=lambda p: p.stat().st_mtime)  # 按时间取最新 (跨次训练轮次号会回卷)
            if ws:
                from adr.models.voice_library import VOICES_DIR, save_voice
                meta_p = VOICES_DIR / args.bind_voice / "meta.json"
                if meta_p.exists():
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                    meta["vits_weights"] = str(ws[-1].resolve())
                    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
                    print(f"[bind] 档案「{args.bind_voice}」已绑定最新权重: {ws[-1].name}")
                else:
                    # 首次训练用户无档案: 用本次切片的第一个干净切片当 ref 自动建档
                    # (切片已静音修剪, 免 GPU 探测; t2s 留空 = 预训练 s1, 与 skip-s1 训练一致)
                    slices = sorted((GSV / "output" / "slicer_opt" / exp).glob("*.wav"))
                    ref = str(slices[0].resolve()) if slices else audio
                    save_voice(args.bind_voice, ref, "",
                               vits_weights=str(ws[-1].resolve()))
                    print(f"[bind] 已新建档案「{args.bind_voice}」(ref={Path(ref).name}, "
                          f"权重 {ws[-1].name}); Clone 页点「刷新音色列表」即可用")

    if start <= 6 and not args.skip_s1:
        # s1 (GPT 语义) 微调
        import yaml
        os.makedirs(logs / "logs_s1_v2", exist_ok=True)  # ckpt 保存需要
        cfg = yaml.safe_load(open(GSV / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml",
                                  encoding="utf-8"))
        cfg["train"]["batch_size"] = args.batch_size
        cfg["train"]["epochs"] = args.s1_epochs
        cfg["train"]["precision"] = "16-mixed"
        cfg["pretrained_s1"] = str(PRE_GSV / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
        cfg["train"]["save_every_n_epoch"] = 1
        cfg["train"]["if_save_every_weights"] = True
        cfg["train"]["if_save_latest"] = True
        cfg["train"]["if_dpo"] = False
        cfg["train"]["half_weights_save_dir"] = "GPT_weights_v2"
        cfg["train"]["exp_name"] = exp
        cfg["train_semantic_path"] = str(logs / "6-name2semantic.tsv")
        cfg["train_phoneme_path"] = str(logs / "2-name2text.txt")
        cfg["output_dir"] = str(logs / "logs_s1_v2")
        tmp_cfg = GSV / "TEMP" / "tmp_s1.yaml"
        tmp_cfg.parent.mkdir(exist_ok=True)
        tmp_cfg.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
        env = {"_CUDA_VISIBLE_DEVICES": "0", "hz": "25hz"}
        run(f'"{PY}" -s GPT_SoVITS/s1_train.py --config_file "{tmp_cfg}"',
            env=env, desc=f"7/7 GPT 微调 ({args.s1_epochs} epochs)")

    print(f"\n[DONE] 权重: {GSV}/SoVITS_weights_v2 与 GPT_weights_v2 下找 {exp}")


if __name__ == "__main__":
    main()
