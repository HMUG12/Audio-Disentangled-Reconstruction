"""ADR Doctor: 一键环境自检与常见问题修复 (批次18)。

动机: "下载失败/合成报错/训练无曲线"类问题缺乏统一排查入口, 把历史踩坑
(ref 超长、STOP 残留、预训练缺失、端口占用) 沉淀为检查项。

CLI:  adr doctor [--fix]
API:  GET /api/adr/v1/system/doctor?fix=1   (console.py 挂载)
分层: 仅依赖标准库 + numpy/soundfile + adr.core — 不 import models
(架构单向依赖 core←models, 见批次8审计)。
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
import urllib.request
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


def _clip_profile_ref(ref: Path) -> None:
    """档案 ref 超长就地裁剪: 去首尾静音取 8s (与 voice_library 兜底同思路)。"""
    import numpy as np
    import soundfile as sf
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
    sf.write(str(ref), wav, sr, subtype="PCM_16")


def run_checks(fix: bool = False) -> dict:
    """执行全部检测, 返回结构化报告 (CLI 与 API 共用)。"""
    checks: list[dict] = []

    def add(cid: str, name: str, status: str, detail: str, fixable: bool = False):
        checks.append({"id": cid, "name": name, "status": status,
                       "detail": detail, "fixable": fixable})

    # 1. 设备 / torch -----------------------------------------------
    try:
        import torch
        from adr.core.device import detect_device
        dev = detect_device()
        add("device", "设备 / GPU",
            "ok" if dev.device == "cuda" else "warn",
            f"torch {torch.__version__}, device={dev.device}, "
            f"GPU={dev.gpu_name or '-'}")
    except Exception as e:
        add("device", "设备 / GPU", "fail", f"检测失败: {e}")

    # 2. 关键依赖 ----------------------------------------------------
    missing = []
    for mod in ("numpy", "soundfile", "librosa", "fastapi", "uvicorn",
                "onnxruntime"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    add("deps", "关键依赖", "ok" if not missing else "fail",
        "全部可导入" if not missing else f"缺失: {', '.join(missing)}")

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
        f"ffmpeg={'可用' if ff else '缺失'} ffprobe={'可用' if fp else '缺失'}")

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
            "4 档全部加载正常" if not bad else "; ".join(bad))
    except Exception as e:
        add("presets", "训练配置 preset", "fail", str(e))

    # 5. GSV 预训练权重 ----------------------------------------------
    pm = GSV / "pretrained_models"
    miss = [rel for rel in _GSV_REQUIRED if not (pm / rel).exists()]
    add("gsv_pretrained", "GSV 预训练权重",
        "ok" if not miss else "fail",
        f"{len(_GSV_REQUIRED) - len(miss)}/{len(_GSV_REQUIRED)} 关键文件齐全"
        if not miss else f"缺失: {'; '.join(miss)}")

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
                    _clip_profile_ref(ref)
                    vfixed.append(f"「{d.name}」ref {dur:.1f}s → 已裁 8s")
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
    if vfixed and not vissues:
        add("voices", "音色档案", "fixed",
            f"{n_profiles} 个档案; 已修复: {'; '.join(vfixed)}", fixable=True)
    elif vfixed:
        add("voices", "音色档案", "warn",
            f"已修复: {'; '.join(vfixed)}; 仍存在问题: {'; '.join(vissues)}",
            fixable=True)
    else:
        add("voices", "音色档案", "ok" if not vissues else "warn",
            f"{n_profiles} 个档案全部正常" if not vissues
            else "; ".join(vissues), fixable=bool(vissues))

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
        fixable=bool(stops))

    # 8. 磁盘空间 -----------------------------------------------------
    low = []
    for drive in sorted({str(REPO.anchor), "F:\\"}):
        try:
            free = shutil.disk_usage(drive).free / 2 ** 30
            if free < 5:
                low.append(f"{drive} 剩余 {free:.1f}GB (<5GB, 训练/导出可能失败)")
        except Exception:
            pass
    add("disk", "磁盘空间", "ok" if not low else "warn",
        "充足" if not low else "; ".join(low))

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
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
