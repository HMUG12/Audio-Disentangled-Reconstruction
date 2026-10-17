"""GSV 预训练资源引导下载 (批次44 便携 runtime)。

便携安装包不含大权重 (pretrained_models ~1.4GB / G2PWModel ~560MB),
首次预热时按需下载解压。三源取自 GSV 官方 install.ps1/install.sh:
XXXXRT/GPT-SoVITS-Pretrained (hf-mirror / HF / ModelScope)。

进度状态经 status() 暴露, 由 gsv_engine.stage_text() 组装成人读文案,
server stats 透传桌面壳 prewarm 页显示 (用户要求的下载提示)。
"""
from __future__ import annotations

import os
import shutil
import threading
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GSV_DIR = REPO_ROOT / "third_party" / "gpt_sovits"

_REPO = "XXXXRT/GPT-SoVITS-Pretrained"

# 源优先级: 环境变量 ADR_MODEL_SOURCE 显式指定优先, 默认国内镜像先行
# (hf-mirror 为 GSV 官方推荐的国内代理, 海外亦可达; ModelScope 兜底)
_SOURCE_ORDER = ["hfmirror", "hf", "modelscope"]


def _urls(fname: str) -> dict[str, str]:
    return {
        "hfmirror": f"https://hf-mirror.com/{_REPO}/resolve/main/{fname}",
        "hf": f"https://huggingface.co/{_REPO}/resolve/main/{fname}",
        "modelscope": f"https://www.modelscope.cn/models/{_REPO}/resolve/master/{fname}",
    }


# 与 GSV 官方 install.ps1 同款判据/同款解压目标:
# - pretrained_models.zip 解到 GPT_SoVITS/ (zip 根含 pretrained_models/)
#   就绪判据 pretrained_models/sv 目录存在
# - G2PWModel.zip 解到 GPT_SoVITS/text/ (zip 根含 G2PWModel/; 老包解出
#   G2PWModel_1.1 → 重命名)
_MANIFEST: list[dict] = [
    {
        "key": "pretrained",
        "label": "预训练底模",
        "zip_name": "pretrained_models.zip",
        "urls": _urls("pretrained_models.zip"),
        "extract_to": GSV_DIR / "GPT_SoVITS",
        "marker": GSV_DIR / "GPT_SoVITS" / "pretrained_models" / "sv",
    },
    {
        "key": "g2pw",
        "label": "中文拼音前端 (G2PW)",
        "zip_name": "G2PWModel.zip",
        "urls": _urls("G2PWModel.zip"),
        "extract_to": GSV_DIR / "GPT_SoVITS" / "text",
        "marker": GSV_DIR / "GPT_SoVITS" / "text" / "G2PWModel",
    },
]

_STATE_LOCK = threading.Lock()
_STATE: dict = {
    "stage": "idle",       # idle / downloading / extracting / done / failed
    "key": "",
    "file": "",
    "downloaded": 0,
    "total": 0,
    "error": "",
}


def _set_state(**kw) -> None:
    with _STATE_LOCK:
        _STATE.update(kw)


def status() -> dict:
    """当前 bootstrap 进度快照 (线程安全)。"""
    with _STATE_LOCK:
        return dict(_STATE)


def pending() -> list[str]:
    """缺失资源的 key 列表 (不触发下载)。"""
    return [e["key"] for e in _MANIFEST if not e["marker"].exists()]


def _ordered_sources(urls: dict[str, str]) -> list[tuple[str, str]]:
    order = list(_SOURCE_ORDER)
    env = os.environ.get("ADR_MODEL_SOURCE", "").strip().lower()
    if env in urls:
        order = [env] + [s for s in order if s != env]
    return [(s, urls[s]) for s in order if s in urls]


# 无代理 opener: 代理环境变量指向未运行的本地代理 (死代理) 时, urllib 默认
# 走代理 → 连接被拒 → 三源全挂。每源下载失败后用无代理 opener 回退重试一次。
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _download(url: str, dest: Path, no_proxy: bool = False) -> None:
    """单源下载 (带进度回调), 失败抛异常由调用方换源重试。"""
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "adr-bootstrap/1.0"})
    opener = _NO_PROXY_OPENER if no_proxy else urllib.request.build_opener()
    with opener.open(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        _set_state(stage="downloading", file=dest.name, total=total, downloaded=0)
        with open(part, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                _set_state(downloaded=done, total=total or done)
    if total > 0 and done != total:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"truncated download: expected {total}, got {done}")
    part.replace(dest)


def _extract(entry: dict) -> None:
    zip_path = entry["extract_to"] / entry["zip_name"]
    _set_state(stage="extracting", file=zip_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(entry["extract_to"])
    # 老版 G2PW 包解出 G2PWModel_1.1 → 重命名为标准目录名
    if not entry["marker"].exists():
        alt = entry["extract_to"] / "G2PWModel_1.1"
        if alt.is_dir():
            alt.rename(entry["marker"])
    if not entry["marker"].exists():
        raise RuntimeError(
            f"extract ok but marker missing: {entry['marker']}"
        )
    zip_path.unlink(missing_ok=True)


def _acquire(entry: dict) -> None:
    zip_path = entry["extract_to"] / entry["zip_name"]
    # 残留 zip (上次中断于解压前): 校验后直接复用, 损坏则重新下载
    if zip_path.exists():
        try:
            _extract(entry)
            return
        except (zipfile.BadZipFile, RuntimeError):
            zip_path.unlink(missing_ok=True)

    last_err: Exception | None = None
    for source, url in _ordered_sources(entry["urls"]):
        _set_state(key=entry["key"])
        # 每源两段式: 默认 (继承系统/环境代理) → 失败且本机配了代理时绕代理重试
        attempts = [(False, url)]
        if urllib.request.getproxies():
            attempts.append((True, url))
        for no_proxy, u in attempts:
            try:
                _download(u, zip_path, no_proxy=no_proxy)
                break
            except Exception as e:  # 换源/绕代理重试
                last_err = e
                zip_path.with_suffix(zip_path.suffix + ".part").unlink(missing_ok=True)
                zip_path.unlink(missing_ok=True)
                if no_proxy:
                    continue
        else:
            continue
        try:
            _extract(entry)
            return
        except (zipfile.BadZipFile, RuntimeError) as e:
            last_err = e
            zip_path.unlink(missing_ok=True)
            continue
    raise RuntimeError(
        f"all sources failed for {entry['zip_name']}: {last_err} — "
        "请检查网络与代理 (需可达 hf-mirror.com / huggingface.co / modelscope.cn), "
        "修复后重启引擎重试")


def ensure() -> None:
    """确保全部预训练资源就位; 缺失则下载解压 (阻塞, 预热线程调用)。

    失败抛 RuntimeError, 状态置 failed — gsv_engine.prewarm 捕获后走
    既有 failed 阶段 (壳可重试, 重试会重新进入本函数续传)。
    """
    if not pending():
        _set_state(stage="done", error="", file="")
        return
    if not GSV_DIR.exists():
        raise RuntimeError(f"GSV source tree not found: {GSV_DIR}")
    try:
        for entry in _MANIFEST:
            if entry["marker"].exists():
                continue
            _acquire(entry)
        _set_state(stage="done", error="")
    except Exception as e:
        _set_state(stage="failed", error=str(e))
        raise


def cleanup_cache() -> None:
    """清理下载残留 .part (维护用)。"""
    for entry in _MANIFEST:
        part = entry["extract_to"] / (entry["zip_name"] + ".part")
        if part.exists():
            shutil.rmtree(part, ignore_errors=True)
