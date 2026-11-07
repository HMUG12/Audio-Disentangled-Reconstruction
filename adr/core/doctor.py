"""ADR Doctor: 一键环境自检与常见问题修复 (批次18)。

动机: "下载失败/合成报错/训练无曲线"类问题缺乏统一排查入口, 把历史踩坑
(ref 超长、STOP 残留、预训练缺失、端口占用) 沉淀为检查项。

CLI:  adr doctor [--fix]
API:  GET /api/adr/v1/system/doctor?fix=1   (console.py 挂载)
分层: 仅依赖标准库 + numpy/soundfile + adr.core — 不 import models
(架构单向依赖 core←models, 见批次8审计)。
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import socket
import sys
import time
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
GSV = REPO / "third_party" / "gpt_sovits" / "GPT_SoVITS"
VOICES = REPO / "data" / "voices"
LOGS = REPO / "third_party" / "gpt_sovits" / "logs"  # GSV 训练/早停产物根

# GSV 预训练关键文件 (相对 pretrained_models/, 缺任一 → 零样本/训练不可用)
_GSV_REQUIRED = [
    "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "gsv-v2final-pretrained/s2G2333k.pth",
    "chinese-hubert-base/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/tokenizer.json",
    "sv/pretrained_eres2netv2w24s4ep4.ckpt",
]


def _clip_profile_ref(ref: Path, dur: float) -> str:
    """档案 ref 超长裁剪: 去首尾静音取 8s (与 voice_library 兜底同思路)。

    批次66: 先备份原 ref, 再写临时文件 os.replace 原子替换 (保留原 subtype);
    meta.json 记 ref_clipped 标记 prompt_text 可能与裁后音频不符。返回备份文件名。
    """
    import numpy as np
    import soundfile as sf
    subtype = sf.info(str(ref)).subtype
    wav, sr = sf.read(str(ref), dtype="float32")
    orig = wav
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
        orig = wav
    frame = max(1, int(0.025 * sr))
    n = len(wav) // frame
    if n >= 4:
        db = 20 * np.log10(np.sqrt(
            (wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-8)
        loud = np.where(db > db.max() - 30.0)[0]
        if len(loud):
            wav = wav[loud[0] * frame: (loud[-1] + 1) * frame]
    wav = wav[: int(8.0 * sr)]
    if len(wav) < int(3.0 * sr):
        wav = orig[: int(8.0 * sr)]  # 几乎全静音兜底
    backup = ref.with_name(f"ref.orig.{time.strftime('%Y%m%d-%H%M%S')}.wav")
    tmp = ref.with_name(f"ref.wav.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        sf.write(str(tmp), wav, sr, subtype=subtype, format="WAV")
        shutil.copy2(ref, backup)
        os.replace(tmp, ref)
    finally:
        tmp.unlink(missing_ok=True)
    meta_path = ref.with_name("meta.json")
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    m["ref_clipped"] = {"from_s": round(dur, 2), "backup": backup.name,
                        "prompt_text_stale": True}
    mtmp = meta_path.with_name(f"meta.json.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        mtmp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(mtmp, meta_path)
    finally:
        mtmp.unlink(missing_ok=True)
    return backup.name


def _version() -> str:
    """尽力取版本号, 不 import 顶层 adr (会连带加载 torch/models)。"""
    try:
        from importlib.metadata import version
        return version("adr")
    except Exception:
        try:
            import tomllib
            with open(REPO / "pyproject.toml", "rb") as f:
                return str(tomllib.load(f)["project"]["version"])
        except Exception:
            return "unknown"


def _has_module(name: str) -> bool:
    """只查 import 链是否存在, 不真正导入 (避免 doctor 触发重依赖加载)。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _nltk_roots() -> list[Path]:
    roots = []
    env = os.environ.get("NLTK_DATA", "").strip()
    if env:
        roots.append(Path(env))
    roots += [REPO / "python" / "nltk_data", REPO / "nltk_data",
              Path.home() / "nltk_data", Path(sys.prefix) / "nltk_data"]
    return roots


def run_checks(fix: bool = False) -> dict:
    """执行全部检测, 返回结构化报告 (CLI 与 API 共用)。

    每个检查项统一带 ``hint``: 面向用户的下一步动作 (可点/可执行). UI 在
    warn/fail 时展开显示, 避免"只报错不给出路" (批次92 用户需求).
    """
    checks: list[dict] = []

    def add(cid: str, name: str, status: str, detail: str,
            fixable: bool = False, hint: str | None = None):
        checks.append({"id": cid, "name": name, "status": status,
                       "detail": detail, "fixable": fixable, "hint": hint})

    # 1. 设备 / torch -----------------------------------------------
    try:
        import torch
        from adr.core.device import detect_device
        dev = detect_device()
        add("device", "设备 / GPU",
            "ok" if dev.device == "cuda" else "warn",
            f"torch {torch.__version__}, device={dev.device}, "
            f"GPU={dev.gpu_name or '-'}",
            hint=None if dev.device == "cuda" else
            "当前走 CPU 推理 (较慢)。有 NVIDIA 显卡可改用 GPU 安装包; "
            "无独显则忽略此项")
        if dev.device == "cuda":
            try:
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / 2 ** 30, 1)
                cap = f"{props.major}.{props.minor}"
                add("gpu_adapt", "GPU 适配",
                    "ok" if vram_gb >= 6 else "warn",
                    f"{props.name}, 显存 {vram_gb}GB, compute capability {cap}",
                    hint=None if vram_gb >= 6 else
                    "显存 <6GB: 请使用 4GB/6GB 训练配方, 推理不受影响")
            except Exception as e:
                add("gpu_adapt", "GPU 适配", "warn", f"无法读取显卡属性: {e}")
        else:
            # CPU 版 torch + 机器有 NVIDIA 驱动 → 装了 CPU 包, 属常见"以为用了显卡"
            has_nv = bool(shutil.which("nvidia-smi"))
            add("gpu_adapt", "GPU 适配",
                "warn" if has_nv else "info",
                "检测到 NVIDIA 驱动, 但当前 torch 不支持 CUDA (安装的是 CPU 包)"
                if has_nv else "未检测到可用的 NVIDIA GPU (CPU 推理属正常)",
                hint="如需 GPU 加速, 请下载 GPU 安装包 (文件名带 -gpu)" if has_nv else None)
    except Exception as e:
        add("device", "设备 / GPU", "fail", f"检测失败: {e}",
            hint="重装 torch: pip install torch torchaudio")

    # 2. 关键依赖 ----------------------------------------------------
    # 批次41b: onnxruntime 拆出 — 它只服务 RVC 声线转换, 未装不应让
    # "关键依赖"整项 fail, 降级为独立 warn 提示
    missing = []
    for mod in ("numpy", "soundfile", "librosa", "fastapi", "uvicorn"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    add("deps", "关键依赖", "ok" if not missing else "fail",
        "全部可导入" if not missing else f"缺失: {', '.join(missing)}",
        hint=None if not missing else "重装依赖: pip install -r requirements.txt")

    # 2b. 多语言 G2P (批次92: 日/韩/粤合成 import 链, 缺失只影响对应语种)
    g2p_missing = [m for m in ("pyopenjtalk", "jamo", "ToJyutping", "ko_pron")
                   if not _has_module(m)]
    add("multilang", "多语言 G2P (日/韩/粤)", "ok" if not g2p_missing else "warn",
        "四套 G2P 齐全" if not g2p_missing else
        f"缺失: {', '.join(g2p_missing)} — 对应语种合成会 400 (中/英不受影响)",
        hint=None if not g2p_missing else
        "使用完整安装包; 源码环境: pip install pyopenjtalk jamo ToJyutping ko_pron")

    # 2c. 英文 G2P 资源 (批次92: nltk cmudict/词性标注, 无外网机器缺失即英文挂)
    if not _has_module("nltk"):
        add("nltk_data", "英文 G2P 资源 (nltk)", "warn", "nltk 未安装",
            hint="pip install nltk 并运行 python -m nltk.downloader cmudict "
                 "averaged_perceptron_tagger_eng")
    else:
        need = (("corpora", "cmudict"),
                ("taggers", "averaged_perceptron_tagger_eng"))
        miss_res = [f"{a}/{b}" for a, b in need
                    if not any((r / a / b).exists() or (r / a / f"{b}.zip").exists()
                               for r in _nltk_roots())]
        add("nltk_data", "英文 G2P 资源 (nltk)",
            "ok" if not miss_res else "warn",
            "cmudict + 词性标注齐全" if not miss_res else
            f"缺失: {', '.join(miss_res)} — 英文合成可能触发在线下载失败",
            hint=None if not miss_res else
            "python -m nltk.downloader cmudict averaged_perceptron_tagger_eng")

    # 2d. VC++ 运行库 (批次92: 精简系统缺 CRT 时装完 python.exe 起不来)
    if os.name != "nt":
        add("vc_runtime", "VC++ 运行库", "info", "非 Windows, 跳过")
    else:
        sys32 = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32"
        miss_crt = [d for d in ("vcruntime140.dll", "vcruntime140_1.dll")
                    if not (sys32 / d).exists()]
        add("vc_runtime", "VC++ 运行库", "ok" if not miss_crt else "fail",
            "vcruntime140 / vcruntime140_1 均存在" if not miss_crt else
            f"缺失: {', '.join(miss_crt)} — 训练/推理子进程可能无法启动",
            hint=None if not miss_crt else
            "运行安装包 redist/vc_redist.x64.exe (或安装 VC++ 2015-2022 运行库)")

    try:
        import onnxruntime
        add("onnxruntime", "onnxruntime (可选)", "ok",
            f"onnxruntime {onnxruntime.__version__} 可用")
    except Exception as e:
        add("onnxruntime", "onnxruntime (可选)", "warn",
            f"onnxruntime 未安装 ({type(e).__name__}) — "
            "仅影响 RVC 声线转换, 其余功能不受影响",
            hint="如不用 RVC 声线转换可忽略; 需要则 pip install onnxruntime")

    # 3. ffmpeg / ffprobe (切片/ASR 依赖) -----------------------------
    def _find_ff(name: str):
        p = shutil.which(name)
        if p:
            return p
        cand = GSV.parent / f"{name}.exe"
        return str(cand) if cand.exists() else None

    ff, fp = _find_ff("ffmpeg"), _find_ff("ffprobe")
    add("ffmpeg", "ffmpeg / ffprobe", "ok" if ff and fp else "warn",
        "两者均可用" if ff and fp else
        f"ffmpeg={'可用' if ff else '缺失'} ffprobe={'可用' if fp else '缺失'}",
        hint=None if ff and fp else
        "安装包应自带 bin/ffmpeg.exe + bin/ffprobe.exe; 源码环境请安装 ffmpeg 并加入 PATH")

    # 4. 训练配置 preset ---------------------------------------------
    try:
        from adr.core.config import load_config
        bad = []
        for preset in ("default", "vram_4gb", "vram_6gb", "vram_8gb"):
            try:
                load_config(preset=preset)
            except Exception as e:
                bad.append(f"{preset}: {e}")
        add("presets", "训练配置 preset", "ok" if not bad else "fail",
            "4 档全部加载正常" if not bad else "; ".join(bad),
            hint=None if not bad else
            "确认 configs/*.yaml 随包/存在 (安装包缺 configs 会静默回落默认值)")
    except Exception as e:
        add("presets", "训练配置 preset", "fail", str(e),
            hint="确认 configs/default.yaml 存在")

    # 5. GSV 预训练权重 ----------------------------------------------
    pm = GSV / "pretrained_models"
    miss = [rel for rel in _GSV_REQUIRED if not (pm / rel).exists()]
    add("gsv_pretrained", "GSV 预训练权重",
        "ok" if not miss else "fail",
        f"{len(_GSV_REQUIRED) - len(miss)}/{len(_GSV_REQUIRED)} 关键文件齐全"
        if not miss else f"缺失: {'; '.join(miss)}",
        hint=None if not miss else
        "首次启动会自动下载 (国内可设 HF_ENDPOINT=https://hf-mirror.com); "
        "下载失败请检查网络/磁盘后重启服务")

    # 6. 音色档案完整性 (ref 存在 / 时长 3~10s / meta 权重路径) --------
    vissues: list[str] = []
    vfixed: list[str] = []
    n_profiles = 0
    if VOICES.exists():
        import soundfile as sf
        for d in sorted(VOICES.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            ref, meta = d / "ref.wav", d / "meta.json"
            if not ref.exists() or not meta.exists():
                vissues.append(f"「{d.name}」缺 ref.wav 或 meta.json")
                continue
            try:
                info = sf.info(str(ref))
                dur = info.frames / info.samplerate
            except Exception:
                vissues.append(f"「{d.name}」ref.wav 无法读取 (损坏)")
                continue
            n_profiles += 1
            if dur > 10.0:  # GSV 合成硬限制 (TTS.py:815)
                if fix:
                    # 批次66: 单档裁剪失败 (文件被占用/只读) 只记问题, 不中断后续检查
                    try:
                        bak = _clip_profile_ref(ref, dur)
                        vfixed.append(f"「{d.name}」ref {dur:.1f}s → 已裁 8s "
                                      f"(原文件备份为 {bak}, 请核对 prompt_text)")
                    except Exception as e:
                        vissues.append(f"「{d.name}」ref 裁剪失败: {e}")
                else:
                    vissues.append(
                        f"「{d.name}」ref {dur:.1f}s 超 10s — 合成将报 "
                        f"tts failed (可 --fix 自动裁剪)")
            elif dur < 3.0:
                vissues.append(f"「{d.name}」ref 仅 {dur:.1f}s (<3s 合成会拒绝)")
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                for k in ("t2s_weights", "vits_weights"):
                    w = m.get(k)
                    if w and not Path(w).exists():
                        vissues.append(
                            f"「{d.name}」meta.{k} 指向的权重不存在: {Path(w).name}")
            except Exception:
                vissues.append(f"「{d.name}」meta.json 解析失败")
    voices_hint = "可点「自动修复问题项」(adr doctor --fix) 裁剪超长 ref/清理残留; " \
                  "权重缺失请重新绑定或导入音色包"
    if vfixed and not vissues:
        add("voices", "音色档案", "fixed",
            f"{n_profiles} 个档案; 已修复: {'; '.join(vfixed)}", fixable=True,
            hint="已自动修复; 若提示 prompt_text_stale, 请核对参考音频对应文本")
    elif vfixed:
        add("voices", "音色档案", "warn",
            f"已修复: {'; '.join(vfixed)}; 仍存在问题: {'; '.join(vissues)}",
            fixable=True, hint=voices_hint)
    else:
        add("voices", "音色档案", "ok" if not vissues else "warn",
            f"{n_profiles} 个档案全部正常" if not vissues
            else "; ".join(vissues), fixable=bool(vissues),
            hint=None if not vissues else voices_hint)

    # 7. 早停 STOP 残留 (会让下次训练启动即停) -------------------------
    stops = list(LOGS.glob("*/STOP")) if LOGS.exists() else []
    if fix and stops:
        for s in stops:
            s.unlink(missing_ok=True)
    add("stop_files", "早停 STOP 残留",
        "fixed" if fix and stops else ("warn" if stops else "ok"),
        "无残留" if not stops else
        f"{len(stops)} 个 ({', '.join(s.parent.name for s in stops)}), "
        f"{'已清理' if fix else '下次训练会被秒停 — 可 --fix 清理'}",
        fixable=bool(stops),
        hint=None if not stops else "点「自动修复问题项」清理 (adr doctor --fix)")

    # 8. 磁盘空间 (批次41b: 历史硬编码 "F:\" 改为 ADR 数据目录所在盘) --
    from adr.core.config import adr_data_dir
    data_dir = adr_data_dir()
    data_drive = str(data_dir.anchor)
    low = []
    for drive in sorted({str(REPO.anchor), data_drive}):
        try:
            free = shutil.disk_usage(drive).free / 2 ** 30
            if free < 5:
                low.append(f"{drive} 剩余 {free:.1f}GB (<5GB, 训练/导出可能失败)")
        except Exception:
            pass
    add("disk", "磁盘空间", "ok" if not low else "warn",
        "充足" if not low else "; ".join(low),
        hint=None if not low else "清理磁盘或迁移数据目录 (设置中心 → 声音档案存放位置)")

    # 8b. 数据目录可写 (批次92: 只读/权限问题会让建档/导出/训练随机失败)
    probe = data_dir / f".doctor_probe_{os.getpid()}"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        add("data_write", "数据目录可写", "ok", str(data_dir))
    except Exception as e:
        add("data_write", "数据目录可写", "fail",
            f"{data_dir} 写入失败: {e}",
            hint="在设置中心改到有写权限的目录 (或检查杀毒/只读盘), 然后重启服务")

    # 9. 默认端口 + 服务探活 (info 级, 壳模式端口动态不影响) -----------
    def _port_free(port: int) -> bool:
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    svc_stage = None
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:9881/api/adr/v1/system/stats",
                timeout=2) as r:
            svc_stage = json.loads(r.read()).get("engine_stage")
    except Exception:
        pass
    add("service", "ADR 服务 (默认端口 9881)", "info",
        f"端口{'空闲' if _port_free(9881) else '被占用 (BOUND/LISTEN)'}; "
        + (f"服务在线, engine_stage={svc_stage}" if svc_stage
           else "本端口未检测到服务 (壳模式为动态端口, 属正常)"))

    cnt = Counter(c["status"] for c in checks)
    return {"checks": checks, "summary": dict(cnt),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            # 批次92: 便于用户复制粘贴报障 (不含路径/用户名以外的隐私)
            "env": {
                "version": _version(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "data_dir": str(data_dir),
            }}
