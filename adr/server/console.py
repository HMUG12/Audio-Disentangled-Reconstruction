"""批次14: 控制台后端 — 系统状态 / 声音克隆训练 / 模型管理 / 静态控制台页。

服务对象 (由 register_pages 挂载到服务根):
- /        index.html  浏览器直达索引
- /pro     pro.html    专业控制台 (深色数据可视化, 全参数)
- /easy    easy.html   新手控制台 (向导式, 零参数)
- /call    call.html   调用控制台 (局域网 API 调试, 批次26 补记)


训练链路与 gsv_finetune.py 完全同构:
    python -u scripts/gsv_finetune.py <audio> --exp <name> --s2-epochs <N>
         --batch-size 4|1 --skip-s1 [--max-clip-sec 10] [--bind-voice <v>]
  + 可选质量门禁 scripts/train_gate.py (CPU 打分, sim>=0.80 写 STOP 早停)
  曲线: output/train_gate_<exp>.jsonl (kind=zero_shot 基线 + epoch 打分)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from adr.core.naming import sanitize_exp_name
from adr.server.exceptions import ProfileNameInvalid
from adr.server.origin_guard import require_browser_safe
from adr.server.pathsafe import is_safe_name, resolve_within

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).parent / "static"
OUTPUT_DIR = REPO / "output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
# F11: 训练进程登记 — sidecar 崩溃后由下一进程据此回收孤儿训练
TRAIN_PID_FILE = OUTPUT_DIR / "train.pid"
# 2h 兜底; 批次66: 真正生效 (reader 线程起看门狗定时强杀), 可用环境变量放宽
def _parse_timeout(raw: str | None, default: float = 7200.0) -> float:
    """批次66: 非法/非正/非有限值回落默认并告警, 不让坏环境变量拖垮导入或秒杀训练。"""
    try:
        v = float(raw) if raw is not None else default
    except ValueError:
        v = math.nan
    if not (math.isfinite(v) and v > 0):
        logger.warning("ADR_TRAIN_TIMEOUT=%r 非法, 回落 %.0fs", raw, default)
        return default
    return v


TRAIN_TIMEOUT = _parse_timeout(os.environ.get("ADR_TRAIN_TIMEOUT"))


def _resolve_upload_audio(raw: str) -> Path:
    """批次78 (CODE_REVIEW #11): 训练/建档音频必须是上传目录内的普通文件。

    训练/建档入口原来只做 ``exists()``: 任意本地文件 (设备文件/系统文件/别处素材)
    都能被当作训练素材或参考音频, 且错误文案可当磁盘路径存在性探针。现在统一走
    ``resolve_within`` 白名单 + ``is_file()``, 越界/不存在/非普通文件同一文案
    (不区分原因, 不构成探针)。控制台页面本身经 /train/upload 落盘, 不受影响;
    v2 的 ``ref_audio_path`` 属另一套契约 (见 docs/tts-api-spec.md §7.3)。
    """
    p = resolve_within(UPLOAD_DIR, raw)
    if p is None or not p.is_file():
        raise HTTPException(
            400, "音频不存在或不可用: 请先通过 /api/adr/v1/train/upload 上传"
                 " (仅接受上传目录内的普通音频文件)")
    return p


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
        # 预热细分阶段: queued/importing/downloading/loading/kernel/ready/failed/""
        "engine_stage": gsv_engine.stage(),
        # 批次44: downloading 阶段的实时下载进度文案 (其余阶段空串)
        "engine_stage_text": gsv_engine.stage_text(),
        # 批次73: failed 阶段的原因摘要 (traceback 末几行, 其余阶段空串)
        "engine_error": gsv_engine.stage_error(),
        # 批次23: 并发排队可视化 (排队等待数 / 合成中数)
        "queue_depth": gsv_engine.queue_depth(),
        "synth_busy": gsv_engine.synth_busy(),
        # 批次63: 调度报告 (预算/累计准入/等待时延), 引擎未初始化时 None
        "sched": gsv_engine.sched_stats(),
        "profiles": _profile_count(),
        "models": _model_stats(),
        # 注意: train_status 是 async, 直接调用返回 coroutine 会被
        # jsonable_encoder 拒绝 → 整个 stats 500 (实测踩坑), 必须用同步快照
        "train": _train_snapshot(),
        "uptime_sec": round(time.time() - _BOOT_T0, 1),
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# 声音克隆训练 (gsv_finetune 子进程 + 可选门禁)
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
    "stage": "",         # F5: gsv_finetune 的 [[STAGE]] 当前步骤 (前端直显)
    "bind_voice": "",    # 批次50: 绑定目标 (收尾相似度结算用)
    "degraded": None,    # 批次50: 收尾退化门结果 {new_sim, old_sim} | None
    "run_id": 0,         # 批次66: 每轮自增代号, 旧 reader 线程据此不碰新一轮状态
    "reader": None,      # 批次66: 本轮 reader 线程; 存活=仍有后代进程占着管道
    "logs": deque(maxlen=400),
}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            if not h:
                return False
            try:
                code = ctypes.c_ulong()
                if not k.GetExitCodeProcess(h, ctypes.byref(code)):
                    return False
                return code.value == 259  # STILL_ACTIVE
            finally:
                k.CloseHandle(h)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_python_pid(pid: int) -> bool:
    """F11: Windows 上按镜像名过滤 PID 复用 — 只有 python*.exe 才可能是训练进程。"""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if not k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return False
            return Path(buf.value).name.lower().startswith("python")
        finally:
            k.CloseHandle(h)
    except Exception:
        return False


def _record_train_pid(proc: subprocess.Popen, exp: str) -> None:
    try:
        TRAIN_PID_FILE.write_text(
            json.dumps({"pid": proc.pid, "exp": exp, "ts": time.time()}),
            encoding="utf-8")
    except OSError:
        logger.warning("写入训练 pid 文件失败", exc_info=True)


def _clear_train_pid(pid: int | None = None) -> None:
    """清登记; 指定 pid 时只在登记仍是该进程时删除, 避免旧 reader 删掉新一轮。"""
    if pid is not None:
        try:
            cur = json.loads(TRAIN_PID_FILE.read_text(encoding="utf-8"))
            if int(cur.get("pid") or 0) != int(pid):
                return
        except (OSError, ValueError, TypeError):
            pass
    TRAIN_PID_FILE.unlink(missing_ok=True)


def reap_orphan_train() -> None:
    """F11: 启动时回收上次 sidecar 崩溃遗留的训练进程 (独立进程组不会随父退出)。

    只处理"进程仍存活且镜像仍是 python"的登记, 降低 PID 复用误杀风险。
    """
    try:
        meta = json.loads(TRAIN_PID_FILE.read_text(encoding="utf-8"))
        pid = int(meta.get("pid") or 0)
    except (OSError, ValueError, TypeError):
        return
    if _pid_alive(pid) and _is_python_pid(pid):
        logger.warning("检测到上次遗留的训练进程 pid=%s, 正在回收", pid)
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                               capture_output=True, timeout=15)
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            logger.warning("回收遗留训练进程失败 pid=%s", pid, exc_info=True)
    _clear_train_pid()


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
            "stage": _train.get("stage", ""),
            "early_stop": _train["early_stop"],
            "bind_voice": _train.get("bind_voice", ""),
            "degraded": _train.get("degraded"),
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


def _kill_tree(p: subprocess.Popen, graceful: bool = False) -> None:
    """批次66: 连同孙进程整树终止 — gsv_finetune 再起 s2_train 子进程 (批次117
    起 shell=False 列表参数), 只杀父进程时孙进程继续占显卡并握着 stdout 管道,
    reader 永远等不到 EOF。
    Windows: taskkill /T /F (须在父进程存活时调用, 否则查不到子树);
    POSIX: 子进程以 start_new_session 起, 按进程组发信号。"""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)],
                           capture_output=True, timeout=15,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.SubprocessError):
            logger.warning("taskkill 失败, 回落只杀父进程 pid=%s", p.pid, exc_info=True)
    else:
        try:
            os.killpg(p.pid, signal.SIGTERM if graceful else signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # 整组已退出 / 非组长 (未用 start_new_session 起的进程)
    if p.poll() is None:  # 兜底: 至少保证父进程被处理
        p.terminate() if graceful else p.kill()


def _stop_proc(p: subprocess.Popen | None, timeout: float = 10) -> None:
    """批次66: 整树终止子进程并收割 (先优雅 → 超时强杀), 失败只记日志。"""
    if p is None or p.poll() is not None:
        return
    try:
        _kill_tree(p, graceful=True)
        p.wait(timeout=timeout)
    except Exception:
        try:
            _kill_tree(p)
            p.wait(timeout=5)
        except Exception:
            logger.warning("子进程强杀失败 pid=%s", p.pid, exc_info=True)


# S5: 子进程 (训练/门禁/下载/重建) 不需要服务端凭据; 继承整个 os.environ 会把
# ADR_TTS_API_KEY 等带进任何能被日志/崩溃转储看到的子进程。统一白名单外剔除。
_SENSITIVE_ENV = ("ADR_TTS_API_KEY", "ADR_API_KEY", "ADR_ALLOW_QUERY_API_KEY",
                  "ADR_ADMIN_PASSWORD", "ADR_ACCOUNT_PASSWORD")


def _child_env(extra: dict | None = None) -> dict:
    e = {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV}
    e.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
              "PYTHONUNBUFFERED": "1"})
    if extra:
        e.update({k: str(v) for k, v in extra.items()})
    return e


def _release_engine_for_train() -> None:
    """训练前释放常驻推理引擎, 避免 6/7 训练 OOM (F6)。

    只在引擎已就绪且没有合成在跑时释放; 释放失败不影响训练 (只记日志),
    训练结束后下一次合成会自动懒加载。
    """
    try:
        if gsv_engine.is_ready() and gsv_engine.synth_busy() == 0:
            gsv_engine.reset_engine("train")
            logger.info("训练前已释放推理引擎显存")
    except Exception:
        logger.warning("训练前释放引擎失败 (继续训练)", exc_info=True)


_ASR_TEXT_RE = re.compile(r"\|[A-Za-z]{2,3}\|")


def _redact_log_line(line: str) -> str:
    """S4/S10: 日志环脱敏 — 本机路径/用户名/ASR 原文不出前端。

    上游 1-get-text / 3-get-semantic 会整行打印「切片路径|exp|LANG|识别原文」,
    识别原文属个人敏感信息, 整行丢弃; 其余行只抹路径前缀。
    """
    try:
        up = str(UPLOAD_DIR.resolve())
        if up and up in line:
            line = line.replace(up, "<UPLOAD>")
        home = str(Path.home())
        if home and home in line:
            line = line.replace(home, "~")
        if _ASR_TEXT_RE.search(line):
            return "[已隐藏 ASR 原文行]"
    except Exception:  # 脱敏失败也不能让 reader 线程退出
        pass
    return line


def _train_reader(proc: subprocess.Popen, gate: subprocess.Popen | None = None,
                  run_id: int | None = None) -> None:
    """后台读训练 stdout → 日志环形缓冲; EOF 后收割退出码并收尾门禁。

    批次66: 只收尾本轮自己的 proc/gate; 所有 _train 读写持锁且校验 run_id,
    旧 reader 迟到时不会覆盖/杀掉新一轮。TRAIN_TIMEOUT 看门狗超时强杀。"""
    if run_id is None:
        with _train_lock:
            run_id = _train["run_id"]

    def _mine() -> bool:  # 调用方须持 _train_lock
        return _train["run_id"] == run_id

    def _on_timeout() -> None:
        if proc.poll() is None:
            try:
                _kill_tree(proc)  # 整树杀, 孙进程不放管道 reader 才能 EOF
                msg = f"[timeout] 训练超过 {TRAIN_TIMEOUT:.0f}s 未结束, 已强杀"
            except Exception as e:
                logger.warning("训练超时强杀失败", exc_info=True)
                msg = f"[timeout] 训练超过 {TRAIN_TIMEOUT:.0f}s 未结束, 强杀失败: {e}"
            with _train_lock:
                if _mine():
                    _train["logs"].append(msg)

    watchdog = threading.Timer(TRAIN_TIMEOUT, _on_timeout)
    watchdog.daemon = True
    watchdog.start()
    rc = None
    read_err = ""
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            line = _redact_log_line(line)  # S4: 前端日志环不留路径/ASR 原文
            with _train_lock:
                if not _mine():
                    continue
                _train["logs"].append(line)
                # F5: gsv_finetune 每步打印 [[STAGE]] 7/7 xxx, 前端直接展示,
                # 不再只能看滚动日志猜进度
                if line.startswith("[[STAGE]]"):
                    _train["stage"] = line[len("[[STAGE]]"):].strip()
                if "收到训练门禁早停信号" in line:
                    _train["early_stop"] = True
        rc = proc.wait()
    except Exception as e:
        # F9: 管道/解码异常也必须收尾, 否则 state 永远 running、按钮锁死
        read_err = f"{type(e).__name__}: {e}"
        logger.warning("训练日志读取异常", exc_info=True)
        try:
            rc = proc.wait(timeout=5)
        except Exception:
            rc = None
    finally:
        watchdog.cancel()
        watchdog.join()  # 批次66: 已触发时等它写完 [timeout] 日志再收尾
        _clear_train_pid(proc.pid)  # F11: 正常收尾后清登记
    # F5: 门禁子进程若提前死亡 (OOM/脚本错), 训练会失去早停保护 — 明示出来
    gate_rc = gate.poll() if gate is not None else None
    if gate is not None and gate_rc not in (None, 0):
        with _train_lock:
            if _mine():
                _train["logs"].append(
                    f"[gate] 门禁进程已以 exit={gate_rc} 退出, 本轮无相似度早停")
    _stop_proc(gate)  # 只收本轮门禁, 不读 _train["gate"] (可能已是新一轮的)
    with _train_lock:
        if not _mine():
            return
        _train["exit_code"] = rc
        if _train["state"] == "running":
            _train["state"] = "done" if rc == 0 else "failed"
        if read_err:
            _train["logs"].append(f"[abort] 日志读取异常: {read_err}")
        _train["gate"] = None
        exp, bind_voice = _train["exp"], _train.get("bind_voice")
    # 批次50: 退化门收尾 — 训练成功且有绑定档案时, 用门禁最优 sim 结算
    # (与 history[-1].sim 比对, 退化 Δ<-0.05 置标记; voice_library.finalize_bind_sim)
    if rc == 0 and bind_voice:
        degraded, err = None, None
        try:
            from adr.models.voice_library import finalize_bind_sim
            entries = _gate_entries(exp)
            sims = [e["sim"] for e in entries
                    if e.get("kind") != "zero_shot" and "sim" in e]
            if sims:
                degraded = finalize_bind_sim(bind_voice, max(sims))
        except Exception as e:
            logger.exception("训练收尾相似度结算失败")
            err = e
        with _train_lock:
            if _mine():
                _train["degraded"] = degraded
                if err is not None:
                    _train["logs"].append(f"[finalize] 相似度结算失败: {err}")


class TrainStartReq(BaseModel):
    audio_path: str
    exp_name: str = "my_voice"
    epochs: int = 8
    recipe: str = "8gb"      # 8gb: bs4 | 4gb: bs1 + clip≤10s
    gate_on: bool = True
    bind_voice: str = ""
    resume_from: str = ""    # 批次50: 从已有档案续训 (voice:名字 的名字)


@router.post("/train/start", summary="启动声音克隆微调 (gsv_finetune 子进程)")
async def train_start(body: TrainStartReq):
    audio = _resolve_upload_audio(body.audio_path)
    script = REPO / "scripts" / "gsv_finetune.py"
    if not script.exists():
        raise HTTPException(500, f"缺少训练脚本: {script}")

    with _train_lock:
        # 批次66: 已 stop 但旧进程尚未真正退出时也拒绝, 防两轮训练抢同一张卡
        # 批次66: 上一轮 reader 未结束 (孙进程仍握管道/占卡) 也拒绝
        old, rd = _train["proc"], _train.get("reader")
        if (_train["state"] == "running" or (old is not None and old.poll() is None)
                or (rd is not None and rd.is_alive())):
            raise HTTPException(409, "已有训练在进行, 请先停止或等待完成")
        exp = sanitize_exp_name(body.exp_name)
        epochs = max(1, min(int(body.epochs), 50))
        recipe = body.recipe if body.recipe in ("8gb", "4gb") else "8gb"

        cmd = [sys.executable, "-u", str(script), str(audio),
               "--exp", exp, "--s2-epochs", str(epochs),
               "--batch-size", "1" if recipe == "4gb" else "4",
               "--skip-s1"]  # s1 全量微调降相似度 (实测结论)
        if recipe == "4gb":
            cmd += ["--max-clip-sec", "10"]
        # 批次50: 续训默认绑回同一档案 (训完自动版本化绑定, 闭环)
        bind_voice = body.bind_voice or body.resume_from
        if bind_voice:
            cmd += ["--bind-voice", bind_voice]
        if body.resume_from:
            # 批次50: 续训 — 档案须存在且有微调权重 (CLI --resume-from voice:名)
            from adr.models.voice_library import load_voice
            # 批次66: 按异常类型区分「不存在 / 损坏 / 名称非法 / 读取失败」, 保留异常链
            try:
                rv = load_voice(body.resume_from)
            except FileNotFoundError as e:
                raise HTTPException(400, f"续训档案不存在: {body.resume_from}") from e
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise HTTPException(
                    400, f"续训档案 meta.json 损坏: {body.resume_from} ({e})") from e
            except ValueError as e:
                raise HTTPException(400, f"续训档案名非法: {body.resume_from}") from e
            except Exception as e:
                logger.exception("续训档案读取失败: %s", body.resume_from)
                raise HTTPException(
                    400, f"续训档案读取失败: {body.resume_from} ({e})") from e
            if not rv.get("vits_weights"):
                raise HTTPException(
                    400, f"档案「{body.resume_from}」无微调权重, 无法续训 (先完成一次普通微调)")
            cmd += ["--resume-from", f"voice:{body.resume_from}"]

        # F6: 训练与合成都吃显存, 训练前主动释放常驻引擎 (空闲才释放;
        # 有合成在跑就保持现状, 由调度器排队, 不粗暴打断用户请求)
        await asyncio.to_thread(_release_engine_for_train)

        # 质量门禁: CPU 打分, 与训练并行, sim≥0.80 写 STOP 早停
        gate_proc = None
        gate_err = ""
        if body.gate_on:
            gate_log = OUTPUT_DIR / f"train_gate_{exp}.log"
            gate_cmd = [sys.executable, "-u",
                        str(REPO / "scripts" / "train_gate.py"),
                        "--exp", exp, "--ref", str(audio),
                        "--target", "0.80", "--cpu"]
            try:
                gate_log.parent.mkdir(exist_ok=True)
                # 批次66: 子进程已继承句柄, 父进程 with 退出即关, 不再泄漏/锁日志
                with open(gate_log, "a", encoding="utf-8", errors="replace") as fh:
                    gate_proc = subprocess.Popen(
                        gate_cmd, stdout=fh, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace", cwd=str(REPO),
                        start_new_session=True,  # 批次66: POSIX 整组可杀
                        env=_child_env())
            except Exception as e:
                gate_proc = None  # 门禁失败不影响训练, 但批次66 起记日志可见
                gate_err = f"[gate] 质量门禁启动失败, 本轮无门禁: {e}"
                logger.warning("训练质量门禁启动失败", exc_info=True)

        spawn_err = None
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(REPO), bufsize=1, start_new_session=True,  # 批次66: POSIX 整组可杀
                env=_child_env())
            # 训练链全 utf-8, 否则 Windows GBK 输出乱码; F5: -u/UNBUFFERED
            # 保证上游 print 立即进管道, 不再长时间空白
        except Exception as e:
            spawn_err = e
        else:
            run_id = _train["run_id"] + 1
            reader = threading.Thread(target=_train_reader,
                                      args=(proc, gate_proc, run_id), daemon=True)
            _train.update(state="running", proc=proc, gate=gate_proc, exp=exp,
                          recipe=recipe, epochs=epochs, started_at=time.time(),
                          exit_code=None, early_stop=False, stage="准备中",
                          bind_voice=bind_voice, degraded=None, run_id=run_id,
                          reader=reader)
            _train["logs"].clear()
            # S4/S6: 不把上传音频绝对路径/音色名写进日志环, 只留可判定的启动标记
            _train["logs"].append("[CMD] gsv_finetune.py 已启动 (音频/音色参数已脱敏)")
            if gate_err:
                _train["logs"].append(gate_err)
            _record_train_pid(proc, exp)  # F11: 供下次启动回收孤儿

    if spawn_err is not None:
        # 批次66: 回滚门禁要收割不留孤儿; 放锁外线程池, 不卡事件循环/status
        await asyncio.to_thread(_stop_proc, gate_proc)
        raise HTTPException(500, f"训练进程启动失败: {spawn_err}") from spawn_err
    reader.start()
    return {"state": "running", "exp": exp, "epochs": epochs, "recipe": recipe}


@router.get("/train/status", summary="训练状态 + 门禁曲线 + 日志尾部")
def train_status():
    # 普通 def (非 async): 要读门禁 jsonl (文件 IO), FastAPI 自动丢线程池,
    # 不占用事件循环 — 前端 2s 轮询 + 合成并发时保持响应 (P7)
    return _train_snapshot()


@router.post("/train/stop", summary="停止训练 (含门禁)")
async def train_stop():
    with _train_lock:
        proc = _train["proc"]
        if not proc or proc.poll() is not None:
            raise HTTPException(409, "当前没有进行中的训练")
        run_id = _train["run_id"]
        prev = _train["state"]
        _train["state"] = "stopped"  # 先置位, reader 收尾时不会改写成 failed
    # 批次66: 确认进程真退出才报 stopped; 等待丢线程池, 不卡事件循环
    await asyncio.to_thread(_stop_proc, proc)
    if proc.poll() is None:
        with _train_lock:
            if _train["run_id"] == run_id and _train["state"] == "stopped":
                _train["state"] = prev
        raise HTTPException(500, "停止训练失败: 进程仍在运行")
    return {"state": "stopped"}


# 批次99 (R4-DATA-01): 不接受 .zip — 训练素材下游按单文件探测/切片, 旧实现
# 收下 zip 但无人解压, 建档时才失败; 前端文案同步去掉 zip。
_UPLOAD_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_UPLOAD_MAX_SIZE = 500 * 1024 * 1024      # 500MB (训练素材异常上传防呆)
_UPLOAD_MAX_DURATION = 7200.0             # 2h; 训练素材建议 5~30 分钟


def _probe_audio(path: Path) -> dict:
    """音频真伪/时长探测 (批次18 兜底): soundfile 优先, 失败回落 ffprobe
    (m4a/aac 等 libsndfile 不认的格式)。两者都失败 → 抛 ValueError 拒收。"""
    try:
        import soundfile as sf  # 批次66: 移入 try, 缺模块时走 ffprobe 回落而非 500
        info = sf.info(str(path))
        return {"duration_s": round(info.frames / info.samplerate, 1),
                "samplerate": info.samplerate, "via": "soundfile"}
    except Exception:
        pass
    exe = shutil.which("ffprobe")
    if not exe:  # GSV 仓库根通常自带
        for cand in (REPO / "third_party" / "gpt_sovits" / "ffmpeg.exe",
                     REPO / "third_party" / "gpt_sovits" / "ffprobe.exe"):
            if cand.exists():
                exe = str(cand)
                break
    if not exe:
        # 无 ffprobe 无法判定真伪 — 放行 (训练管线 ffmpeg 会再报错), 不阻塞上传
        return {"duration_s": None, "samplerate": None, "via": "none"}
    try:
        r = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (subprocess.TimeoutExpired, OSError) as e:
        # 批次66: 探测超时/不可执行 → 归为 ValueError, 调用方 400 并删落盘文件
        raise ValueError(f"音频文件探测失败 (超时或探测工具不可用): {e}") from e
    try:
        return {"duration_s": round(float(r.stdout.strip()), 1),
                "samplerate": None, "via": "ffprobe"}
    except ValueError:
        raise ValueError("音频文件无法解析 (损坏或编码不支持)")


def _doctor_engine_check() -> dict:
    """批次92: 把引擎启动失败原因/降级状态并入 doctor 报告。

    引擎状态在 models 层, core.doctor 刻意不 import models (单向依赖), 因此
    这项检查由 server 层追加 — UI 一处即可看到"为什么起不来 + 怎么办"。
    """
    from adr.models import gsv_engine as g

    stage = g.stage()
    err = (g.stage_error() or "").strip()
    stalled = g.is_stalled()
    degraded = g.degraded_seconds()
    watchdog = g.watchdog_grace_sec()
    tail = " | ".join(err.splitlines()[-3:]) if err else ""
    if stalled or degraded > 0:
        return {"id": "engine", "name": "引擎运行状态", "status": "warn",
                "detail": f"卡死降级 {degraded:.0f}s (stage={stage})"
                          + (f": {tail}" if tail else ""),
                "fixable": False,
                "hint": f"超过 {watchdog:.0f}s 未恢复将自动重启进程; "
                        "仍不恢复请重启服务并运行完整诊断"}
    if stage == "failed":
        return {"id": "engine", "name": "引擎运行状态", "status": "fail",
                "detail": "启动失败" + (f": {tail}" if tail else " (无错误详情)"),
                "fixable": False,
                "hint": "按上方原因处理后重启服务; 常见为权重下载失败、"
                        "显存不足或模型文件损坏"}
    if g.is_ready():
        return {"id": "engine", "name": "引擎运行状态", "status": "ok",
                "detail": f"已就绪 (看门狗 {watchdog:.0f}s)", "fixable": False,
                "hint": None}
    return {"id": "engine", "name": "引擎运行状态", "status": "info",
            "detail": f"未加载 (stage={stage}); 首次合成会触发加载",
            "fixable": False, "hint": None}


@router.get("/system/doctor", summary="一键环境自检 (fix=true 自动修复可修项)")
async def system_doctor(fix: bool = False,
                        _guard: None = Depends(require_browser_safe)):
    """检测依赖/GPU/权重/档案/端口/磁盘; fix=true 时裁剪超长 ref、清理 STOP 残留。"""
    from adr.core.doctor import run_checks

    # F8: run_checks 内含 GPU 探测/ffmpeg 子进程/磁盘统计, 同步执行会冻住
    # 事件循环 (实测 ~2s), 期间 /health 与流式首包全部停摆 — 丢线程池。
    rep = await asyncio.to_thread(run_checks, fix=fix)
    rep["checks"].append(_doctor_engine_check())
    summary: dict[str, int] = {}
    for c in rep["checks"]:
        summary[c["status"]] = summary.get(c["status"], 0) + 1
    rep["summary"] = summary
    return rep


@router.post("/system/doctor", summary="一键环境自检 (POST; fix=true 自动修复可修项)")
async def system_doctor_post(fix: bool = False,
                             _guard: None = Depends(require_browser_safe)):
    """批次77: 副作用接口的 POST 别名 (浏览器跨站 GET 已由依赖拒绝)。"""
    return await system_doctor(fix)


@router.post("/train/upload", summary="上传克隆录音 (wav/mp3/flac/m4a/ogg)")
async def train_upload(file: UploadFile = File(...),
                       source_declaration: str = Form("")):
    """批次62 上传侧声音安全: ``source_declaration`` 为素材来源声明 (谁的声音/
    授权关系), 只入审计不落业务库; 建档时请在 /profiles/create 再次携带并
    传 ``consent_granted=true``。"""
    from adr.core.audit import log_event, text_fp
    orig = Path(file.filename or "audio.wav")
    ext = orig.suffix.lower()
    if ext not in _UPLOAD_EXT:
        raise HTTPException(
            415, f"不支持的格式 {ext}, 可用: {'/'.join(sorted(_UPLOAD_EXT))}")
    # S4: 落盘名只用 uuid, 不保留原文件名 (可能含人名); 原名只在响应里回显
    fname = f"{uuid.uuid4().hex}{ext}"
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
    except BaseException:
        # 批次65 上传回滚: 客户端断连/磁盘满/取消等任何中断都删半截文件
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "上传内容为空")
    duration_s = samplerate = None
    try:
        # 批次87 (PERF-02): 探测含 soundfile/ffprobe 子进程, 移出事件循环
        probe = await asyncio.to_thread(_probe_audio, dest)
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
    # 批次62 上传侧审计: 不记文件名/路径 (可能含人名), 声明只记指纹
    decl = (source_declaration or "").strip()
    log_event("material_upload", ext=ext, size=size,
              decl_fp=text_fp(decl) if decl else None)
    return {"path": str(dest), "size": size, "filename": fname,
            "original_name": orig.name,
            "duration_s": duration_s, "samplerate": samplerate,
            "source_declaration": (source_declaration or "").strip()[:200]}


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
    language: str = "zh"           # 批次53: 参考音频语种 → 语言默认联动面
    # 批次62 上传侧声音安全: 声纹属个人敏感信息, 建档必须声明来源并勾选权利人同意
    source_declaration: str = ""
    consent_granted: bool = False


def _profile_worker(name: str, audio_path: str, prompt_text: str, style: str,
                    language: str = "zh", source: str = "",
                    consent: dict | None = None) -> None:
    from adr.models import voice_library

    def _prog(msg: str) -> None:
        with _profile_lock:
            _profile_job["message"] = msg

    try:
        vdir, info = voice_library.save_voice_auto(
            name, audio_path, prompt_text, progress=_prog, style=style,
            language=language, source=source, consent=consent)
        with _profile_lock:
            _profile_job.update(state="done", info={**info, "path": str(vdir)},
                                message=f"建档完成: {name}")
        # v1.3.x: 首个档案建好即补跑本机采样, 新用户不必等下次启动
        from adr.models.gsv_engine import try_schedule_profile
        try_schedule_profile()
    except Exception as e:
        # S6: 档案名可能是人名 — 日志只留指纹, 与审计口径一致
        from adr.core.audit import text_fp
        logger.exception("控制台建档失败 (profile_fp=%s)", text_fp(name))
        with _profile_lock:
            _profile_job.update(state="failed", message=str(e))


@router.post("/profiles/create", summary="从录音新建音色档案 (后台扫段选优)")
async def profile_create(body: ProfileCreateReq):
    from adr.models import voice_library
    # 批次41a: 档案名直接拼档案库路径, 清洗后仍可能残留 ".." (穿越成分),
    # 先做单段纯名校验 (400 + code=profile_invalid), 再查音频存在性
    name = (body.name or "我的声音").strip()
    name = "".join(c for c in name if c not in '\\/:*?"<>|') or "我的声音"
    if not is_safe_name(name):
        raise ProfileNameInvalid(name)
    audio = _resolve_upload_audio(body.audio_path)
    # 批次62 上传侧声音安全: 声纹建档需声音权利人单独同意 (不信任客户端
    # 传入的同意结构, 服务端按声明重组后落 meta/审计)
    if not body.consent_granted:
        raise HTTPException(
            422, "未获得声音权利人同意, 无法建档 (声纹属个人敏感信息); "
                 "请确认你已获授权后传 consent_granted=true")
    decl = (body.source_declaration or "").strip()
    consent = {"granted": True,
               "at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "statement": decl[:200]}
    from adr.core.audit import log_event, text_fp
    # 审计脱敏: 档案名可能含人名, 只记指纹 (与素材上传"不记文件名"同一边界)
    log_event("voice_enroll",
              name_fp=text_fp(name) if name else None,
              decl_fp=text_fp(decl) if decl else None)
    if name in voice_library.list_voices():
        raise HTTPException(409, f"已存在同名声音「{name}」, 请换一个名字")
    with _profile_lock:
        if _profile_job["state"] == "working":
            raise HTTPException(409, "已有建档任务在进行, 请稍候")
        _profile_job.update(state="working", name=name, message="准备中…",
                            info=None, started_at=time.time())
    threading.Thread(target=_profile_worker,
                     args=(name, str(audio), body.prompt_text.strip(),
                           body.style.strip(),
                           (body.language or "zh").lower().strip(),
                           decl, consent),
                     daemon=True).start()
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
        "filename": i.filename,
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
    rc = None
    try:
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
    except Exception:
        # F9: 管道异常时也必须落地终态, 否则下载面板永远 downloading
        logger.warning("模型下载日志读取异常", exc_info=True)
        try:
            rc = proc.wait(timeout=5)
        except Exception:
            rc = None
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
                cwd=str(REPO), bufsize=1,
                start_new_session=True)  # 批次99 (P3-04): 与训练/门禁一致, POSIX 整组可杀
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


_MODEL_CHUNK = 1024 * 1024


def _model_target(name: str) -> Path:
    from adr.models.hub import PRETRAINED_REGISTRY, get_hub
    if name not in PRETRAINED_REGISTRY:
        raise HTTPException(404, f"未知模型: {name}")
    return get_hub().cache_dir / PRETRAINED_REGISTRY[name].filename


@router.post("/models/{name}/verify", summary="校验本地模型文件 SHA256")
async def models_verify(name: str):
    from adr.models.hub import PRETRAINED_REGISTRY
    info = PRETRAINED_REGISTRY.get(name)
    target = _model_target(name)
    if not target.is_file():
        raise HTTPException(404, "本地未找到该模型文件")

    def _digest() -> tuple[str, int]:
        sha = hashlib.sha256()
        size = 0
        with open(target, "rb") as f:
            for chunk in iter(lambda: f.read(_MODEL_CHUNK), b""):
                sha.update(chunk)
                size += len(chunk)
        return sha.hexdigest(), size

    # 大文件哈希放到线程里, 不阻塞事件循环 (合成/进度轮询保持响应)
    actual, size = await asyncio.to_thread(_digest)
    if not info.sha256:  # 注册表无官方校验值: 只做存在性+体积检查
        return {"ok": size > 1024, "checked": "size", "sha256": actual,
                "expected": None, "size_bytes": size}
    return {"ok": actual == info.sha256, "checked": "sha256", "sha256": actual,
            "expected": info.sha256, "size_bytes": size}


@router.delete("/models/{name}", summary="删除本地模型缓存")
async def models_delete(name: str):
    target = _model_target(name)
    with _dl_lock:
        if _dl["state"] == "downloading" and _dl["name"] == name:
            raise HTTPException(409, "该模型正在下载, 请稍后再删")
    removed = 0
    for p in (target, target.with_suffix(target.suffix + ".part")):
        if p.is_file():
            p.unlink()
            removed += 1
    return {"state": "deleted" if removed else "absent", "name": name}


@router.post("/models/{name}/import", summary="导入本地权重 (离线恢复, SHA256 校验)")
async def models_import(name: str, file: UploadFile = File(...)):
    from adr.models.hub import PRETRAINED_REGISTRY
    info = PRETRAINED_REGISTRY.get(name)
    target = _model_target(name)
    if Path(file.filename or "").name != info.filename:
        raise HTTPException(400, f"文件名不匹配: 该模型应为 {info.filename}")
    with _dl_lock:
        if _dl["state"] == "downloading":
            raise HTTPException(409, "已有下载在进行")

    # 原子写入: 先落 .part, 校验通过才 os.replace, 失败不破坏已有文件
    part = target.with_suffix(target.suffix + ".part")
    sha = hashlib.sha256()
    total = 0
    try:
        with open(part, "wb") as out:
            while True:
                chunk = await file.read(_MODEL_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                sha.update(chunk)
                out.write(chunk)
        actual = sha.hexdigest()
        if total <= 1024:
            raise HTTPException(400, "文件过小, 不是有效权重")
        if info.sha256 and actual != info.sha256:
            raise HTTPException(
                400,
                f"SHA256 校验失败: 期望 {info.sha256[:12]}…, 实际 {actual[:12]}…")
        os.replace(part, target)
    except HTTPException:
        part.unlink(missing_ok=True)
        raise
    except Exception as e:
        part.unlink(missing_ok=True)
        raise HTTPException(500, f"导入失败: {e}")
    return {"state": "imported", "name": name, "size_bytes": total,
            "verified": bool(info.sha256), "sha256": actual}


# ---------------------------------------------------------------------------
# 退出回收 (批次90 / R3-LIFE-01)
# ---------------------------------------------------------------------------

def stop_background_procs() -> None:
    """服务退出时回收训练/门禁/下载子进程 (可重复调用)。

    这些子进程以 ``start_new_session=True`` 起 (POSIX setsid / Windows 独立
    进程组), 父进程退出不会传导终止信号 — 不主动回收就会留下孤儿训练继续
    写输出目录并占住 GPU 显存, 下次训练直接 OOM。由 FastAPI lifespan 的
    shutdown 阶段与 atexit 兜底调用。
    """
    with _train_lock:
        procs = [_train["proc"], _train.get("gate")]
    with _dl_lock:
        procs.append(_dl["proc"])
    for p in procs:
        try:
            _stop_proc(p)
        except Exception:
            logger.warning("退出回收子进程失败 pid=%s",
                           getattr(p, "pid", None), exc_info=True)


# ---------------------------------------------------------------------------
# 静态控制台页 (同源挂载: 无 CORS, 浏览器可直达)
# ---------------------------------------------------------------------------

def register_pages(app) -> None:
    """把静态控制台页挂到服务根: / → index, /pro, /easy, /call, /settings。"""

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

    @app.get("/settings", include_in_schema=False)
    async def _settings():
        return FileResponse(STATIC_DIR / "settings.html", media_type="text/html")

    # 批次54: 控制台中英文切换运行时 (页面统一引用 /i18n.js, 同源不撞 CSP)
    @app.get("/i18n.js", include_in_schema=False)
    async def _i18n_js():
        return FileResponse(STATIC_DIR / "i18n.js", media_type="application/javascript")

    # 批次54: 各页字典 /i18n-easy.js /i18n-pro.js /i18n-call.js /i18n-settings.js (白名单, 缺文件 404)
    @app.get("/i18n-{page}.js", include_in_schema=False)
    async def _i18n_dict_js(page: str):
        if page not in ("easy", "pro", "call", "index", "settings"):
            raise HTTPException(status_code=404)
        f = STATIC_DIR / f"i18n-{page}.js"
        if not f.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(f, media_type="application/javascript")

    # UI 主题层运行时 (default/classic 双主题, 缺省 default; 页面 <head> 同步引用防闪烁)
    @app.get("/theme.js", include_in_schema=False)
    async def _theme_js():
        return FileResponse(STATIC_DIR / "theme.js", media_type="application/javascript")

    # 统一外壳层: 顶栏 / 全局导航 / 文档抽屉 (各页 </head> 前引用, 覆盖页面排版)
    @app.get("/app-shell.css", include_in_schema=False)
    async def _app_shell_css():
        return FileResponse(STATIC_DIR / "app-shell.css", media_type="text/css")

    @app.get("/app-shell.js", include_in_schema=False)
    async def _app_shell_js():
        return FileResponse(STATIC_DIR / "app-shell.js", media_type="application/javascript")
