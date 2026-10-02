"""D1: RVC 歌声/说话音色转换模型训练一条龙 (新版 RVC WebUI 重构版)。

流程: preprocess (切片/响度) → f0(rmvpe) → hubert 特征 → train (v2 48k) → index
产物: third_party/rvc/assets/weights/<exp>.pth (+ logs/<exp>/added_*.index)

用法: python scripts/rvc_train.py <音频或目录> [--exp user_rvc] [--epochs 60] [--skip-to STEP]
步骤: prep / f0 / hubert / train / index
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RVC = REPO / "third_party" / "rvc"
PY = sys.executable


def run(cmd: str, desc: str = ""):
    print(f"\n{'=' * 60}\n>>> {desc}\n{cmd}\n{'=' * 60}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RVC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    r = subprocess.run(cmd, shell=True, cwd=str(RVC), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"步骤失败 ({desc}): exit={r.returncode}")


def _gen_filelist(exp_dir: Path):
    """复刻官方 webui 的 filelist.txt 生成 (f0+v2+48k, 单说话人 id=0)。

    行格式: gt.wav|feat.npy|f0.npy|f0nsf.npy|0, 另加 2 条 mute 静音行。
    """
    import random

    gt = exp_dir / "0_gt_wavs"
    fea = exp_dir / "3_feature768"
    f0 = exp_dir / "2a_f0"
    f0nsf = exp_dir / "2b-f0nsf"
    def stem(p: Path) -> str:  # '0_1.wav.npy'/'0_1.npy'/'0_1.wav' → '0_1'
        return p.name.replace(".wav.npy", "").replace(".npy", "").replace(".wav", "")

    names = {stem(p) for p in gt.glob("*.wav")}
    names &= {stem(p) for p in fea.glob("*.npy")}
    names &= {stem(p) for p in f0.glob("*.npy")}
    names &= {stem(p) for p in f0nsf.glob("*.npy")}
    if not names:
        raise RuntimeError("无有效训练样本 (特征目录交集为空)")

    def esc(p: Path) -> str:
        return str(p).replace("\\", "\\\\")

    opt = []
    for n in sorted(names):
        opt.append(f"{esc(gt)}\\\\{n}.wav|{esc(fea)}\\\\{n}.npy|"
                   f"{esc(f0)}\\\\{n}.wav.npy|{esc(f0nsf)}\\\\{n}.wav.npy|0")
    now = str(RVC).replace("\\", "\\\\")
    for _ in range(2):
        opt.append(f"{now}\\\\logs\\\\mute\\\\0_gt_wavs\\\\mute48k.wav|"
                   f"{now}\\\\logs\\\\mute\\\\3_feature768\\\\mute.npy|"
                   f"{now}\\\\logs\\\\mute\\\\2a_f0\\\\mute.wav.npy|"
                   f"{now}\\\\logs\\\\mute\\\\2b-f0nsf\\\\mute.wav.npy|0")
    random.shuffle(opt)
    (exp_dir / "filelist.txt").write_text("\n".join(opt), encoding="utf-8")
    print(f"[filelist] {len(opt)} 行 (含 2 条 mute)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="原始音频文件或目录")
    ap.add_argument("--exp", default="user_rvc")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4, help="8GB 安全值")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--skip-to", default="prep",
                    choices=["prep", "f0", "hubert", "train", "index"])
    args = ap.parse_args()

    exp = args.exp
    exp_dir = RVC / "logs" / exp
    exp_dir.mkdir(parents=True, exist_ok=True)  # preprocess.log 在建目录前就要写
    audio = Path(args.audio).resolve()
    # preprocess 需要目录输入
    inp_root = audio if audio.is_dir() else audio.parent
    steps = ["prep", "f0", "hubert", "train", "index"]
    start = steps.index(args.skip_to)

    if start <= 0:
        run(f'"{PY}" -m train.preprocess "{inp_root}" 48000 4 "{exp_dir}" False 3.7',
            desc="1/5 预处理 (切片+响度+滤波)")
    if start <= 1:
        run(f'"{PY}" -m train.dataset.extract_f0 cuda 1 0 0 "{exp_dir}" True',
            desc="2/5 F0 提取 (rmvpe)")
    if start <= 2:
        run(f'"{PY}" -m train.dataset.extract_hubert_feature cuda 1 0 0 "{exp_dir}" v2 True',
            desc="3/5 HuBERT 特征")
    if start <= 3:
        # 官方 webui 在训练前生成 filelist.txt + config.json, 这里复刻
        _gen_filelist(exp_dir)
        import shutil
        shutil.copy2(RVC / "configs" / "v2" / "48k.json", exp_dir / "config.json")
        run(f'"{PY}" -m train.train -se {args.save_every} -te {args.epochs} '
            f'-pg assets/pretrained_v2/f0G48k.pth -pd assets/pretrained_v2/f0D48k.pth '
            f'-g 0 -bs {args.batch_size} -e "{exp}" -sr 48k -sw 1 -v v2 -f0 1 -l 0 -c 0',
            desc=f"4/5 训练 ({args.epochs} epochs)")
    if start <= 4:
        run(f'"{PY}" -m train.train_index {exp} v2 "" 4',
            desc="5/5 检索索引 (faiss)")

    weights = list((RVC / "assets" / "weights").glob(f"{exp}*.pth")) \
        if (RVC / "assets" / "weights").exists() else []
    print(f"\n[DONE] 权重: {[str(w) for w in weights] or '见 logs/' + exp}")


if __name__ == "__main__":
    main()
