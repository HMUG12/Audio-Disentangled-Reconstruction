"""批次14: 控制台后端 — 系统状态 / 声音克隆训练 / 模型管理 / 静态控制台页。

服务对象 (由 register_pages 挂载到服务根):
- /        index.html  浏览器直达索引
- /pro     pro.html    专业控制台 (深色数据可视化, 全参数)
- /easy    easy.html   新手控制台 (向导式, 零参数)
- /call    call.html   调用控制台 (局域网 API 调试, 批次26 补记)

训练链路与 webui._run_gsv_finetune 完全同构:
    python -u scripts/gsv_finetune.py <audio> --exp <name> --s2-epochs <N>
         --batch-size 4|1 --skip-s1 [--max-clip-sec 10] [--bind-voice <v>]
  + 可选质量门禁 scripts/train_gate.py (CPU 打分, sim>=0.80 写 STOP 早停)
  曲线: output/train_gate_<exp>.jsonl (kind=zero_shot 基线 + epoch 打分)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).parent / "static"
OUTPUT_DIR = REPO / "output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
TRAIN_TIMEOUT = 7200  # 与 webui 一致: 2h 兜底 (超时强杀由 reader 线程执行)

_BOOT_T0 = time.time()

router = APIRouter(prefix="/api/adr/v1", tags=["adr-console"])


# ---------------------------------------------------------------------------
# 系统状态 (零新依赖: torch.cuda + ctypes + shutil)
# ---------------------------------------------------------------------------

def _gpu_stats() -> dict | None:
    """显存占用 (torch 已在进程内才查, 否则返回 None)。"""
    try:
        import sys
        if "torch" not in sys.modules:
            return None  # 预热导入 torch 期间不并发 import (抢锁阻塞线程池)
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return {
            "name": torch.cuda.get_device_name(0),
            "used_gb": round((total - free) / 2**30, 2),
            "total_gb": round(total / 2**30, 2),
        }
    except Exception:
        return None


def _ram_stats() -> dict | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = _Mem()
            m.dwLength = ctypes.sizeof(_Mem)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return None
            used = m.ullTotalPhys - m.ullAvailPhys
            return {
                "used_gb": round(used / 2**30, 2),
                "total_gb": round(m.ullTotalPhys / 2**30, 2),
                "percent": m.dwMemoryLoad,
            }
        info: dict[str, float] = {}
        txt = Path("/proc/meminfo").read_text(encoding="utf-8")
        for ln in txt.splitlines():
            k, _, v = ln.partition(":")
            info[k.strip()] = float(v.strip().split()[0]) * 1024  # kB → B
        total, avail = info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0)
        used = total - avail
        return {
            "used_gb": round(used / 2**30, 2),
            "total_gb": round(total / 2**30, 2),
            "percent": round(used / total * 100, 1) if total else 0.0,
        }
    except Exception:
        return None


def _disk_stats() -> dict | None:
    try:
        u = shutil.disk_usage(REPO)
        return {
            "used_gb": round(u.used / 2**30, 1),
            "total_gb": round(u.total / 2**30, 1),
            "free_gb": round(u.free / 2**30, 1),
        }
    except Exception:
        return None


def _profile_count() -> int:
    try:
        from adr.models import voice_library
        return len(voice_library.list_voices())
    except Exception:
        return 0


def _model_stats() -> dict:
    try:
        from adr.models.hub import PRETRAINED_REGISTRY, get_hub
        hub = get_hub()
        return {
            "cached": sum(1 for n in PRETRAINED_REGISTRY if hub.is_downloaded(n)),
            "total": len(PRETRAINED_REGISTRY),
        }
    except Exception:
        return {"cached": 0, "total": 0}


@router.get("/system/stats", summary="控制台总览: 硬件 + 引擎 + 业务状态")
def system_stats():
    # 普通 def (非 async): FastAPI 自动丢线程池执行 — GPU 查询/文件 IO
    # 慢时只占工作线程, 不再卡死事件循环 (曾导致全接口僵死, 实测踩坑)
    from adr.models import gsv_engine
    return {
        "gpu": _gpu_stats(),
        "ram": _ram_stats(),
        "disk": _disk_stats(),
        # 引擎三态: loading=后台预热中 / ready=_tts 就绪可合成
        "engine_loading": gsv_engine.is_loading(),
        "engine_ready": gsv_engine.is_ready(),
        # 预热细分阶段: queued/importing/loading/kernel/ready/failed/""
        "engine_stage": gsv_engine.stage(),
        # 批次23: 并发排队可视化 (排队等待数 / 合成中数)
        "queue_depth": gsv_engine.queue_depth(),
        "synth_busy": gsv_engine.synth_busy(),
        "profiles": _profile_count(),
        "models": _model_stats(),
        # 注意: train_status 是 async, 直接调用返回 coroutine 会被
        # jsonable_encoder 拒绝 → 整个 stats 500 (实测踩坑), 必须用同步快照
        "train": _train_snapshot(),
        "uptime_sec": round(time.time() - _BOOT_T0, 1),
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# 声音克隆训练 (gsv_finetune 子进程 + 可选门禁, 同 webui._run_gsv_finetune)
# ---------------------------------------------------------------------------

_train_lock = threading.Lock()
_train: dict = {
    "state": "idle",     # idle | running | done | failed | stopped
    "proc": None,
    "gate": None,
    "exp": "",
    "recipe": "",
    "epochs": 0,
    "started_at": 0.0,
    "exit_code": None,
    "early_stop": False,
    "logs": deque(maxlen=400),
}


def _gate_entries(exp: str) -> list[dict]:
    """读门禁曲线 jsonl (小文件, 每次全量解析)。"""
    path = OUTPUT_DIR / f"train_gate_{exp}.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def _train_snapshot() -> dict:
    with _train_lock:
        snap = {
            "state": _train["state"],
            "exp": _train["exp"],
            "recipe": _train["recipe"],
            "epochs": _train["epochs"],
            "elapsed_sec": (round(time.time() - _train["started_at"], 1)
                            if _train["state"] == "running" and _train["started_at"]
                            else None),
            "exit_code": _train["exit_code"],
            "early_stop": _train["early_stop"],
            "log_tail": list(_train["logs"])[-80:],
        }
    entries = _gate_entries(snap["exp"]) if snap["exp"] else []
    base = next((e for e in entries if e.get("kind") == "zero_shot"), None)
    scored = [e for e in entries if e.get("kind") != "zero_shot"]
    snap["zero_shot"] = base.get("sim") if base else None
    snap["best_sim"] = max((e["sim"] for e in scored if "sim" in e), default=None)
    snap["gate_curve"] = [{"epoch": e.get("epoch"), "sim": e.get("sim")}
                          for e in scored]
    return snap


def _train_reader(proc: subprocess.Popen) -> None:
    """后台读训练 stdout → 日志环形缓冲; EOF 后收割退出码并收尾门禁。"""
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        with _train_lock:
            _train["logs"].append(line)
            if "收到训练门禁早停信号" in line:
                _train["early_stop"] = True
    rc = proc.wait()
    with _train_lock:
        _train["exit_code"] = rc
        if _train["state"] == "running":
            _train["state"] = "done" if rc == 0 else "failed"
    gate = _train.get("gate")
    if gate and gate.poll() is None:
        gate.terminate()
        try:
            gate.wait(timeout=10)
        except Exception:
            gate.kill()
    _train["gate"] = None


class TrainStartReq(BaseModel):
    audio_path: str
    exp_name: str = "my_voice"
    epochs: int = 8
    recipe: str = "8gb"      # 8gb: bs4 | 4gb: bs1 + clip≤10s
    gate_on: bool = True
    bind_voice: str = ""


@router.post("/train/start", summary="启动声音克隆微调 (gsv_finetune 子进程)")
async def train_start(body: TrainStartReq):
    audio = Path(body.audio_path)
    if not audio.exists():
        raise HTTPException(400, f"音频不存在: {body.audio_path}")
    script = REPO / "scripts" / "gsv_finetune.py"
    if not script.exists():
        raise HTTPException(500, f"缺少训练脚本: {script}")

    with _train_lock:
        if _train["state"] == "running":
            raise HTTPException(409, "已有训练在进行, 请先停止或等待完成")
        exp = (body.exp_name or "my_voice").strip()
        exp = "".join(c for c in exp if c not in '\\/:*?"<>|') or "my_voice"
        epochs = max(1, min(int(body.epochs), 50))
        recipe = body.recipe if body.recipe in ("8gb", "4gb") else "8gb"

        cmd = [sys.executable, "-u", str(script), str(audio),
               "--exp", exp, "--s2-epochs", str(epochs),
               "--batch-size", "1" if recipe == "4gb" else "4",
               "--skip-s1"]  # s1 全量微调降相似度 (webui 实测结论)
        if recipe == "4gb":
            cmd += ["--max-clip-sec", "10"]
        if body.bind_voice:
            cmd += ["--bind-voice", body.bind_voice]

        # 质量门禁: CPU 打分, 与训练并行, sim≥0.80 写 STOP 早停
        gate_proc = None
        if body.gate_on:
            gate_log = OUTPUT_DIR / f"train_gate_{exp}.log"
            gate_log.parent.mkdir(exist_ok=True)
            gate_cmd = [sys.executable, "-u",
                        str(REPO / "scripts" / "train_gate.py"),
                        "--exp", exp, "--ref", str(audio),
                        "--target", "0.80", "--cpu"]
            try:
                fh = open(gate_log, "a", encoding="utf-8", errors="replace")
                gate_proc = subprocess.Popen(
                    gate_cmd, stdout=fh, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", cwd=str(REPO),
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
            except Exception:
                gate_proc = None  # 门禁失败不影响训练

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(REPO), bufsize=1,
                env={**os.environ, "PYTHONIOENCODING": "utf-8",
                     "PYTHONUTF8": "1"})  # 训练链全 utf-8, 否则 Windows GBK 输出乱码
        except Exception as e:
            if gate_proc:
                gate_proc.terminate()
            raise HTTPException(500, f"训练进程启动失败: {e}")

        _train.update(state="running", proc=proc, gate=gate_proc, exp=exp,
                      recipe=recipe, epochs=epochs, started_at=time.time(),
                      exit_code=None, early_stop=False)
        _train["logs"].clear()
        _train["logs"].append(f"[CMD] {' '.join(cmd)}")

    threading.Thread(target=_train_reader, args=(proc,), daemon=True).start()
    return {"state": "running", "exp": exp, "epochs": epochs, "recipe": recipe}


@router.get("/train/status", summary="训练状态 + 门禁曲线 + 日志尾部")
async def train_status():
    return _train_snapshot()


@router.post("/train/stop", summary="停止训练 (含门禁)")
async def train_stop():
    with _train_lock:
        proc = _train["proc"]
        if not proc or proc.poll() is not None:
            raise HTTPException(409, "当前没有进行中的训练")
        _train["state"] = "stopped"
    try:
        proc.terminate()
    except Exception:
        pass
    return {"state": "stopped"}


_UPLOAD_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".zip"}
_UPLOAD_MAX_SIZE = 500 * 1024 * 1024      # 500MB (训练素材异常上传防呆)
_UPLOAD_MAX_DURATION = 7200.0             # 2h; 训练素材建议 5~30 分钟


def _probe_audio(path: Path) -> dict:
    """音频真伪/时长探测 (批次18 兜底): soundfile 优先, 失败回落 ffprobe
    (m4a/aac 等 libsndfile 不认的格式)。两者都失败 → 抛 ValueError 拒收。"""
    import soundfile as sf
    try:
        info = sf.info(str(path))
        return {"duration_s": round(info.frames / info.samplerate, 1),
                "samplerate": info.samplerate, "via": "soundfile"}
    except Exception:
        pass
    exe = shutil.which("ffprobe")
    if not exe:  # GSV 仓库根通常自带
        for cand in (OUTPUT_DIR.parent.parent / "third_party" / "gpt_sovits" / "ffmpeg.exe",
                     OUTPUT_DIR.parent.parent / "third_party" / "gpt_sovits" / "ffprobe.exe"):
            if cand.exists():
                exe = str(cand)
                break
    if not exe:
        # 无 ffprobe 无法判定真伪 — 放行 (训练管线 ffmpeg 会再报错), 不阻塞上传
        return {"duration_s": None, "samplerate": None, "via": "none"}
    r = subprocess.run(
        [exe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        return {"duration_s": round(float(r.stdout.strip()), 1),
                "samplerate": None, "via": "ffprobe"}
    except ValueError:
        raise ValueError("音频文件无法解析 (损坏或编码不支持)")


@router.get("/system/doctor", summary="一键环境自检 (fix=true 自动修复可修项)")
async def system_doctor(fix: bool = False):
    """检测依赖/GPU/权重/档案/端口/磁盘; fix=true 时裁剪超长 ref、清理 STOP 残留。"""
    from adr.core.doctor import run_checks

    return run_checks(fix=fix)


@router.post("/train/upload", summary="上传克隆录音 (wav/mp3/flac/m4a/ogg/zip)")
async def train_upload(file: UploadFile = File(...)):
    orig = Path(file.filename or "audio.wav")
    ext = orig.suffix.lower()
    if ext not in _UPLOAD_EXT:
        raise HTTPException(
            415, f"不支持的格式 {ext}, 可用: {'/'.join(sorted(_UPLOAD_EXT))}")
    stem = re.sub(r"[^\w\-]+", "_", orig.stem)[:40] or "audio"
    fname = f"{datetime.now():%Y%m%d_%H%M%S}_{stem}{ext}"
    dest = UPLOAD_DIR / fname
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > _UPLOAD_MAX_SIZE:
                    raise HTTPException(
                        413, f"文件超过 {_UPLOAD_MAX_SIZE // (1024 * 1024)}MB 上限"
                             " (训练素材建议 5~30 分钟)")
                fh.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "上传内容为空")
    duration_s = samplerate = None
    if ext != ".zip":
        try:
            probe = _probe_audio(dest)
        except ValueError as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, str(e))
        duration_s, samplerate = probe["duration_s"], probe["samplerate"]
        if duration_s and duration_s > _UPLOAD_MAX_DURATION:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                413, f"音频时长 {duration_s / 60:.0f} 分钟超过 "
                     f"{_UPLOAD_MAX_DURATION // 3600} 小时上限; "
                     "训练素材建议 5~30 分钟, 请裁剪后上传")
    return {"path": str(dest), "size": size, "filename": fname,
            "duration_s": duration_s, "samplerate": samplerate}


# ---------------------------------------------------------------------------
# 音色档案建档 (save_voice_auto 后台线程: 长音频扫段需引擎, 约 1-3 分钟)
# ---------------------------------------------------------------------------

_profile_lock = threading.Lock()
_profile_job: dict = {"state": "idle", "name": "", "message": "", "info": None,
                      "started_at": 0.0}


class ProfileCreateReq(BaseModel):
    name: str
    audio_path: str
    prompt_text: str = ""
    style: str = ""


def _profile_worker(name: str, audio_path: str, prompt_text: str, style: str) -> None:
    from adr.models import voice_library

    def _prog(msg: str) -> None:
        with _profile_lock:
            _profile_job["message"] = msg

    try:
        vdir, info = voice_library.save_voice_auto(
            name, audio_path, prompt_text, progress=_prog, style=style)
        with _profile_lock:
            _profile_job.update(state="done", info={**info, "path": str(vdir)},
                                message=f"建档完成: {name}")
    except Exception as e:
        with _profile_lock:
            _profile_job.update(state="failed", message=str(e))


@router.post("/profiles/create", summary="从录音新建音色档案 (后台扫段选优)")
async def profile_create(body: ProfileCreateReq):
    from adr.models import voice_library
    audio = Path(body.audio_path)
    if not audio.exists():
        raise HTTPException(400, f"音频不存在: {body.audio_path}")
    name = (body.name or "我的声音").strip()
    name = "".join(c for c in name if c not in '\\/:*?"<>|') or "我的声音"
    if name in voice_library.list_voices():
        raise HTTPException(409, f"已存在同名声音「{name}」, 请换一个名字")
    with _profile_lock:
        if _profile_job["state"] == "working":
            raise HTTPException(409, "已有建档任务在进行, 请稍候")
        _profile_job.update(state="working", name=name, message="准备中…",
                            info=None, started_at=time.time())
    threading.Thread(target=_profile_worker,
                     args=(name, str(audio), body.prompt_text.strip(),
                           body.style.strip()), daemon=True).start()
    return {"state": "working", "name": name}


@router.get("/profiles/create/status", summary="建档进度")
async def profile_create_status():
    with _profile_lock:
        return {
            "state": _profile_job["state"],
            "name": _profile_job["name"],
            "message": _profile_job["message"],
            "info": _profile_job["info"],
            "elapsed_sec": (round(time.time() - _profile_job["started_at"], 1)
                            if _profile_job["state"] == "working" else None),
        }


# ---------------------------------------------------------------------------
# 预训练模型管理 (adr.models.hub + adr.cli model download 子进程)
# ---------------------------------------------------------------------------

@router.get("/models", summary="预训练模型清单 + 本地缓存状态")
async def models_list():
    from adr.models.hub import PRETRAINED_REGISTRY, get_hub
    hub = get_hub()
    items = [{
        "name": i.name,
        "category": i.category,
        "description": i.description,
        "version": i.version,
        "size_mb": round(i.size_mb),
        "downloaded": hub.is_downloaded(i.name),
    } for i in PRETRAINED_REGISTRY.values()]
    return {"models": items, "cache_dir": str(hub.cache_dir)}


_dl_lock = threading.Lock()
_dl: dict = {"state": "idle", "proc": None, "name": "", "percent": None,
             "exit_code": None, "started_at": 0.0, "logs": deque(maxlen=200)}
_TQDM_RE = re.compile(r"(\d{1,3})%")


def _dl_reader(proc: subprocess.Popen) -> None:
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        with _dl_lock:
            _dl["logs"].append(line)
            m = _TQDM_RE.search(line)
            if m:
                _dl["percent"] = int(m.group(1))
    rc = proc.wait()
    with _dl_lock:
        _dl["exit_code"] = rc
        if _dl["state"] == "downloading":
            _dl["state"] = "done" if rc == 0 else "failed"
            if rc == 0:
                _dl["percent"] = 100


class ModelDownloadReq(BaseModel):
    name: str
    force: bool = False


@router.post("/models/download", summary="后台下载预训练模型")
async def models_download(body: ModelDownloadReq):
    from adr.models.hub import PRETRAINED_REGISTRY, get_hub
    if body.name not in PRETRAINED_REGISTRY:
        raise HTTPException(404, f"未知模型: {body.name}")
    if get_hub().is_downloaded(body.name) and not body.force:
        return {"state": "already", "name": body.name}
    with _dl_lock:
        if _dl["state"] == "downloading":
            raise HTTPException(409, "已有下载在进行")
        cmd = [sys.executable, "-u", "-m", "adr.cli", "model", "download",
               body.name]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(REPO), bufsize=1)
        except Exception as e:
            raise HTTPException(500, f"下载进程启动失败: {e}")
        _dl.update(state="downloading", proc=proc, name=body.name,
                   percent=None, exit_code=None, started_at=time.time())
        _dl["logs"].clear()
    threading.Thread(target=_dl_reader, args=(proc,), daemon=True).start()
    return {"state": "downloading", "name": body.name}


@router.get("/models/download/status", summary="模型下载进度")
async def models_download_status():
    with _dl_lock:
        return {
            "state": _dl["state"],
            "name": _dl["name"],
            "percent": _dl["percent"],
            "exit_code": _dl["exit_code"],
            "elapsed_sec": (round(time.time() - _dl["started_at"], 1)
                            if _dl["state"] == "downloading" else None),
            "log_tail": list(_dl["logs"])[-30:],
        }


# ---------------------------------------------------------------------------
# 静态控制台页 (同源挂载: 无 CORS, 浏览器可直达)
# ---------------------------------------------------------------------------

def register_pages(app) -> None:
    """把静态控制台页挂到服务根: / → index, /pro, /easy, /call。"""

    @app.get("/", include_in_schema=False)
    async def _index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/pro", include_in_schema=False)
    async def _pro():
        return FileResponse(STATIC_DIR / "pro.html", media_type="text/html")

    @app.get("/easy", include_in_schema=False)
    async def _easy():
        return FileResponse(STATIC_DIR / "easy.html", media_type="text/html")

    @app.get("/call", include_in_schema=False)
    async def _call():
        return FileResponse(STATIC_DIR / "call.html", media_type="text/html")
