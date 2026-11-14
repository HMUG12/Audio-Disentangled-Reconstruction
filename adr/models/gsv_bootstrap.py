"""GSV 预训练资源引导下载 (批次44 便携 runtime)。

便携安装包不含大权重 (pretrained_models ~1.4GB / G2PWModel ~560MB),
首次预热时按需下载解压。三源取自 GSV 官方 install.ps1/install.sh:
XXXXRT/GPT-SoVITS-Pretrained (hf-mirror / HF / ModelScope)。

进度状态经 status() 暴露, 由 gsv_engine.stage_text() 组装成人读文案,
server stats 透传桌面壳 prewarm 页显示 (用户要求的下载提示)。
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:      # 批次83 (#15): 跨进程文件锁 (Windows 便携包主战场)
    import msvcrt
except ImportError:      # pragma: no cover - 非 Windows 平台
    msvcrt = None
try:
    import fcntl
except ImportError:      # pragma: no cover - Windows 无 fcntl
    fcntl = None

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
# 批次83 (#15): sha256 取 HF 仓库 tree API 的 lfs.oid (三源同一份字节)。
# 只校验 Content-Length 挡不住 CDN 错误页/代理替换, 也挡不住大小一致的损坏。
_MANIFEST: list[dict] = [
    {
        "key": "pretrained",
        "label": "预训练底模",
        "zip_name": "pretrained_models.zip",
        "urls": _urls("pretrained_models.zip"),
        "extract_to": GSV_DIR / "GPT_SoVITS",
        "marker": GSV_DIR / "GPT_SoVITS" / "pretrained_models" / "sv",
        "sha256": "66274394318cbf134b78d0d5aeeccb73e96f5d43cf6876ac43560a972cb1f3fc",
    },
    {
        "key": "g2pw",
        "label": "中文拼音前端 (G2PW)",
        "zip_name": "G2PWModel.zip",
        "urls": _urls("G2PWModel.zip"),
        "extract_to": GSV_DIR / "GPT_SoVITS" / "text",
        "marker": GSV_DIR / "GPT_SoVITS" / "text" / "G2PWModel",
        "sha256": "46292be0374a49308069233cd5c147ae4c41806558e4781a2467a31a4d8099da",
    },
]

# 批次83 (#15): 双开壳 / 预热线程并发 ensure() 时的互斥等待上限。
# 另开进程正在下 1.4GB 权重时等它下完, 拿到锁后 marker 已存在则直接返回。
_LOCK_WAIT_SEC = 900.0

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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _matches_entry(path: Path, entry: dict) -> bool:
    """残留 zip 是否可信: 清单登记了 sha256 就必须一致 (批次83 #15)。"""
    want = entry.get("sha256") or ""
    if not want:
        return True
    try:
        return _sha256_file(path) == want
    except OSError:
        return False


def _clear_parts(entry: dict) -> None:
    """清理该条目的下载残留 (批次83 #15: 临时名带 pid/uuid, 需按前缀扫)。"""
    d = entry["extract_to"]
    for p in d.glob(entry["zip_name"] + ".*.part"):
        p.unlink(missing_ok=True)
    (d / (entry["zip_name"] + ".part")).unlink(missing_ok=True)   # 固定续传名


def _clear_legacy_parts(entry: dict) -> None:
    """只清旧版 pid/uuid 临时文件, 保留固定 `.part` 供断点续传 (批次95)。"""
    d = entry["extract_to"]
    for p in d.glob(entry["zip_name"] + ".*.part"):
        p.unlink(missing_ok=True)


def _rate_limit_mbps() -> float:
    """批次95: ADR_DOWNLOAD_RATE_MBPS — 下载限速 (MB/s); 0/非法 = 不限速。

    有效区间 0.1~1024, 与设置中心 / Rust 注入面一致 (低于 0.1 视为未配置,
    否则 1.4GB 权重会下到天荒地老)。
    """
    raw = os.environ.get("ADR_DOWNLOAD_RATE_MBPS", "").strip()
    try:
        v = float(raw)
    except ValueError:
        return 0.0
    return v if 0.1 <= v <= 1024 else 0.0


def _throttle_delay(done: int, elapsed: float, mbps: float) -> float:
    """限速补觉时长: 已写 ``done`` 字节 / 耗时 ``elapsed`` 秒, 还差几秒才合规。

    纯函数便于单测; ``mbps<=0`` 或参数异常 → 0 (不限速)。
    """
    if mbps <= 0 or done <= 0:
        return 0.0
    target = done / (mbps * 1024 * 1024)
    return max(0.0, target - elapsed)


def _resp_status(resp) -> int:
    """HTTP 状态码 (urllib 响应对象形态不一, 取不到按 200)。"""
    try:
        code = resp.getcode()
        if code:
            return int(code)
    except Exception:
        pass
    return int(getattr(resp, "status", 200) or 200)


def _resp_total(resp, resume_from: int) -> int:
    """完整文件大小: 206 用 Content-Range 的总量, 否则 offset + Content-Length。"""
    cr = resp.headers.get("Content-Range", "") if resp.headers else ""
    if "/" in cr:
        tail = cr.rsplit("/", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
    try:
        n = int(resp.headers.get("Content-Length", 0) or 0)
    except (TypeError, ValueError):
        n = 0
    return resume_from + n if n > 0 else 0


@contextlib.contextmanager
def _bootstrap_lock():
    """批次83 (#15): 进程级互斥 — 双开壳并发 ensure() 会互相覆盖同一 zip/解压目录。

    用锁文件而非进程内 threading.Lock: 两个便携包实例是不同进程。
    msvcrt.locking / fcntl.flock 都是非阻塞轮询, 超时后报可读错误。
    """
    GSV_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(GSV_DIR / ".bootstrap.lock", "a+b")
    try:
        deadline = time.monotonic() + _LOCK_WAIT_SEC
        while True:
            try:
                fh.seek(0)
                if msvcrt is not None:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "另一个 ADR 进程正在下载预训练资源, 等待超时; "
                        "请稍后重试或关闭其他 ADR 窗口") from None
                time.sleep(0.2)
        try:
            yield
        finally:
            try:
                fh.seek(0)
                if msvcrt is not None:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        fh.close()


def _download(url: str, dest: Path, sha256: str = "", no_proxy: bool = False) -> None:
    """单源下载 (断点续传 + 限速 + sha256 校验), 失败抛异常由调用方换源重试。

    批次83 (#15): 结束校验 sha256, 挡 CDN 错误页/代理替换/等长损坏。
    批次95: 固定 ``<dest>.part`` 作续传位, 中断后下次请求带 ``Range`` 续传
    (源不支持 Range 返回 200 则自动从头下); 内容型错误 (sha 不符 / 416)
    丢弃残留, 网络类错误保留残片供下次续传。``ADR_DOWNLOAD_RATE_MBPS``
    可限速, 不影响校验与进度显示。
    """
    part = dest.with_name(dest.name + ".part")
    opener = _NO_PROXY_OPENER if no_proxy else urllib.request.build_opener()
    resume_from = part.stat().st_size if part.exists() else 0
    while True:
        headers = {"User-Agent": "adr-bootstrap/1.0"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            resp = opener.open(req, timeout=60)
            break
        except urllib.error.HTTPError as e:
            # 416: 本地残片比远端还长 (换版本/坏残留) → 丢弃重来
            if e.code == 416 and resume_from:
                part.unlink(missing_ok=True)
                resume_from = 0
                continue
            raise
    with resp:
        status = _resp_status(resp)
        if resume_from and status != 206:
            resume_from = 0          # 源忽略 Range (200) → 从头下
        total = _resp_total(resp, resume_from)
        digest = hashlib.sha256()
        if resume_from:
            with open(part, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
        done = resume_from
        limit = _rate_limit_mbps()
        started = time.monotonic()
        _set_state(stage="downloading", file=dest.name, total=total or done,
                   downloaded=done)
        try:
            with open(part, "ab" if resume_from else "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    _set_state(downloaded=done, total=total or done)
                    if limit:
                        delay = _throttle_delay(done - resume_from,
                                                time.monotonic() - started, limit)
                        if delay > 0:
                            time.sleep(delay)
        except BaseException:
            # 网络中断: 保留 .part, 下次 (换源/重启) 从断点续传
            raise
        if total > 0 and done != total:
            raise RuntimeError(
                f"truncated download: expected {total}, got {done}")
        got = digest.hexdigest()
        if sha256 and got != sha256:
            # 内容型错误: 残片不可信, 丢弃避免下次继续续传坏字节
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch: expected {sha256}, got {got} (源返回内容被替换或损坏)")
        os.replace(str(part), str(dest))


def _remove_path(path: Path) -> None:
    """删除文件或目录 (目录递归, 不存在则忽略)。"""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _extract(entry: dict) -> None:
    """解压并原子落位 (批次126): 先解到同卷临时目录, 成功后逐项 rename 进目标。

    旧实现直接 ``extractall`` 进活目录: pretrained 的就绪 marker
    (``pretrained_models/sv``) 会在解压途中出现, 进程被打断后下次启动
    误判为完整模型树, 运行时报出难以定位的错误。现改为::

        <extract_to>/.extract-<zip_name>-<pid>/   ← 解压 + 老包改名都在这里
        成功后逐顶层项 os.replace 进 <extract_to>

    同一卷内 rename 是原子的, 所以 marker 只可能随整份载荷一起出现;
    解压中途失败/被杀时 finally 清掉临时目录, 目标的旧内容不被触碰。

    不引入 ``.complete`` 哨兵: 既有安装没有该文件, 加了会让老用户重下
    1.4GB; 原子 rename 本身就是完成信号。
    """
    extract_to = Path(entry["extract_to"])
    zip_path = extract_to / entry["zip_name"]
    _set_state(stage="extracting", file=zip_path.name)
    extract_to.mkdir(parents=True, exist_ok=True)
    tmp_dir = extract_to / f".extract-{entry['zip_name']}-{os.getpid()}"
    _remove_path(tmp_dir)          # 同 pid 残留 (pid 复用) 先清掉
    try:
        tmp_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)
        # 老版 G2PW 包解出 G2PWModel_1.1 → 临时目录内先改名为标准目录名
        try:
            rel = Path(entry["marker"]).relative_to(extract_to)
        except ValueError:
            rel = None
        if rel is not None and not (tmp_dir / rel).exists():
            alt = tmp_dir / rel.parent / (rel.name + "_1.1")
            if alt.is_dir():
                alt.rename(tmp_dir / rel)
        # 逐顶层项落位: 同名旧目标先挪到 <name>.old-<pid> 备份, 移完再删备份
        for item in sorted(tmp_dir.iterdir()):
            dest = extract_to / item.name
            backup = extract_to / f"{item.name}.old-{os.getpid()}"
            if dest.exists() or dest.is_symlink():
                _remove_path(backup)
                os.replace(str(dest), str(backup))
            os.replace(str(item), str(dest))
            if backup.exists():
                _remove_path(backup)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if not entry["marker"].exists():
        raise RuntimeError(
            f"extract ok but marker missing: {entry['marker']}"
        )
    zip_path.unlink(missing_ok=True)


def _acquire(entry: dict) -> None:
    zip_path = entry["extract_to"] / entry["zip_name"]
    # 残留 zip (上次中断于解压前): sha256 与包结构都通过才复用
    # 批次83 (#15): 旧实现只看 zip 能否打开, 内容被替换/截断但结构合法的包
    # 会直接解压进运行时。
    if zip_path.exists():
        if _matches_entry(zip_path, entry):
            try:
                _extract(entry)
                return
            except (zipfile.BadZipFile, RuntimeError):
                zip_path.unlink(missing_ok=True)
        else:
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
                _download(u, zip_path, sha256=entry.get("sha256") or "",
                          no_proxy=no_proxy)
                break
            except Exception as e:  # 换源/绕代理重试
                last_err = e
                # 批次95: 保留固定 .part 供断点续传; 只清旧版 pid/uuid 残片
                _clear_legacy_parts(entry)
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

    批次83 (#15): 全程持进程级文件锁, 双开壳不会互相覆盖 .part/解压目录;
    拿到锁后重新判 marker (另一进程可能已下完)。
    """
    if not pending():
        _set_state(stage="done", error="", file="")
        return
    if not GSV_DIR.exists():
        raise RuntimeError(f"GSV source tree not found: {GSV_DIR}")
    try:
        with _bootstrap_lock():
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
        # 批次66: .part 是文件, rmtree(ignore_errors) 会静默删不掉
        # 批次83 (#15): 临时名已带 pid/uuid, 兜底扫前缀 + 兼容旧固定名
        _clear_parts(entry)
