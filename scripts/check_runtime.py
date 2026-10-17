"""打包前 runtime 体检 (批次49b, 分发瘦身拍板落地)。

背景 (2026-10-10 实测): v1.0.0 安装包 runtime 3.89GB 中 2.06GB 是
third_party/gpt_sovits 根级的实验临时权重副本 (时间戳命名 .pth, 3×585MB
为同一文件复制三份, 无任何代码/meta 引用) 被全量拷贝带入。清理后
runtime 1.78GB。本脚本固化排除规则, 防止垃圾再混入。

检查项:
1. third_party/gpt_sovits 根级裸 .pth → FAIL (权重只允许在固定子目录)
2. 个人数据面 (data/output/logs/GPT_weights_v2/SoVITS_weights_v2/) → FAIL
   (便携包通用分发, 不带个人档案/训练产物/日志)
3. 底模 pretrained_models/ → FAIL (按设计首启 bootstrap 下载, 打包含底模
   会让安装包 +5GB)
4. CPU 包不得含 nvidia_* CUDA 库 (torch CUDA 版 ~2.5GB 经 pip 包拆分进入;
   GPU 包构建时传 --allow-cuda 放行) — 双包拍板见 docs/m10-plan.md G 节
5. 体积分布 + 总量告警 (>2.2GB WARN: 高于清理后基线 1.78GB+余量)
6. configs/*.yaml 必须随包 (v1.0.0 缺口: build_runtime.ps1 未拷 configs/,
   安装机 preset 全部静默回落内置默认值, 日志报 default.yaml 缺失)
7. 多语言 G2P 包必须随包 (批次53: pyopenjtalk/jamo/ToJyutping/ko_pron —
   日/韩/粤合成的 import 链, 缺失则 text_lang=ja/ko/yue 请求 400)

用法:
  python scripts/check_runtime.py                 # 默认 build/runtime, CPU 包
  python scripts/check_runtime.py --allow-cuda    # GPU 包构建体检
退出码: 0 通过 / 1 有 FAIL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME = REPO / "build" / "runtime"

# 个人数据面: 通用分发不得携带 (档案/训练产物/日志/实验输出)
PRIVATE_DIRS = ("data", "output", "logs", "GPT_weights_v2", "SoVITS_weights_v2")
# GSV 运行时必需的代码/文本资源之外, 根级不允许出现的权重形态
GSV_ROOT = Path("third_party") / "gpt_sovits"
WARN_TOTAL_GB_CPU = 2.2   # 清理后基线 1.78GB + 余量
WARN_TOTAL_GB_GPU = 4.6   # cu130 torch + nvidia_* 库实测 3.98GB + 余量


def dir_size_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024**3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--allow-cuda", action="store_true",
                    help="GPU 包构建: 放行 nvidia_* CUDA 库")
    args = ap.parse_args()
    rt = args.runtime.resolve()

    fails: list[str] = []
    warns: list[str] = []

    if not rt.is_dir():
        print(f"[FAIL] runtime 目录不存在: {rt}")
        return 1

    # 1) GSV 根级裸 .pth (垃圾权重混入的第一现场)
    gsv_root = rt / GSV_ROOT
    if gsv_root.is_dir():
        stray = sorted(f.name for f in gsv_root.glob("*.pth") if f.is_file())
        if stray:
            fails.append(
                f"third_party/gpt_sovits 根级裸 .pth ×{len(stray)} (实验临时权重混入, "
                f"v1.0.0 曾因此安装包多 2.06GB): {stray[:4]}"
                + (" ..." if len(stray) > 4 else ""))

    # 2) 个人数据面
    for d in PRIVATE_DIRS:
        if (rt / d).exists():
            fails.append(f"个人数据面混入: {d}/ (通用分发不带档案/训练产物/日志)")
    if gsv_root.is_dir():
        for d in PRIVATE_DIRS:
            if (gsv_root / d).exists():
                fails.append(f"个人数据面混入: third_party/gpt_sovits/{d}/")

    # 3) 底模应走首启 bootstrap 下载
    pre = gsv_root / "GPT_SoVITS" / "pretrained_models"
    if pre.exists():
        fails.append(
            f"pretrained_models/ 不应打包 ({dir_size_gb(pre):.2f}GB) — "
            "按设计首启 bootstrap 下载 (gsv_engine 批次44)")

    # 4) CPU 包不得含 CUDA 库 (双包拍板: GPU 包显式 --allow-cuda)
    sp = rt / "python" / "Lib" / "site-packages"
    if sp.is_dir() and not args.allow_cuda:
        nvidia = sorted(p.name for p in sp.glob("nvidia*") if p.is_dir())
        if nvidia:
            fails.append(f"CPU 包含 nvidia_* CUDA 库 ({', '.join(nvidia[:4])}) — "
                         "GPU 包请传 --allow-cuda")
        # Windows cu13x 轮子把 CUDA DLL 打进 torch/lib (无独立 nvidia_* 包,
        # 仅查 nvidia* 目录会漏检 — 批次 52 实测补丁)
        torch_lib = sp / "torch" / "lib"
        if torch_lib.is_dir():
            cuda_dlls = sorted(f.name for f in torch_lib.glob("cudart64_*.dll"))
            if cuda_dlls:
                fails.append(f"CPU 包含 CUDA torch (torch/lib/{cuda_dlls[0]} 等) — "
                             "GPU 包请传 --allow-cuda")

    # 5) 体积分布 + 总量
    total = dir_size_gb(rt)
    warn_gb = WARN_TOTAL_GB_GPU if args.allow_cuda else WARN_TOTAL_GB_CPU
    rows = sorted(
        ((d.name, dir_size_gb(d)) for d in rt.iterdir() if d.is_dir()),
        key=lambda x: -x[1])
    print(f"[体检] {rt}  总体积 {total:.2f}GB  "
          f"[{'GPU' if args.allow_cuda else 'CPU'} 包]")
    for name, gb in rows:
        print(f"  {name:14s} {gb:6.2f}GB")
    if total > warn_gb:
        warns.append(f"总体积 {total:.2f}GB 超告警线 {warn_gb}GB "
                     f"(基线: CPU 1.78GB / GPU 3.98GB)")

    # 6) configs preset 必须随包 (adr.core.config 按 REPO/configs/*.yaml 加载,
    #    缺失静默回落内置默认值 — v1.0.0 安装机日志实锤的打包缺口)
    conf = rt / "configs"
    if not conf.is_dir() or not any(conf.glob("*.yaml")):
        fails.append("configs/*.yaml 缺失 (训练 preset 全部回落内置默认值)")
    elif not (conf / "default.yaml").is_file():
        fails.append("configs/default.yaml 缺失 (基础 preset)")

    # 7) 多语言 G2P 包随包检查 (批次53: 日/韩/粤合成的 import 链)
    py = rt / "python" / "python.exe"
    if py.is_file():
        import subprocess
        probe = subprocess.run(
            [str(py), "-c",
             "import pyopenjtalk, jamo, ToJyutping, ko_pron"],
            capture_output=True, text=True, timeout=120)
        if probe.returncode != 0:
            fails.append(f"多语言 G2P 包缺失 (ja/ko/yue 合成将 400): "
                         f"{probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else probe.returncode}")

    for w in warns:
        print(f"[WARN] {w}")
    for f in fails:
        print(f"[FAIL] {f}")
    if fails:
        print(f"\n体检不通过: {len(fails)} 项 FAIL")
        return 1
    print("\n体检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
