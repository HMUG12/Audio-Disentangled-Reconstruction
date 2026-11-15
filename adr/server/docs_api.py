"""内置文档中心 API — 本地全量 + 官方仓库增量。

范围 (与产品口径一致): 仅用户文档 / API 规范 / 发布说明。
内部计划 (m*-plan、审计、路线图) 一律不进入白名单, 避免经公开服务外泄。

路由 (挂 /api/adr/v1/docs):
  GET  /              文档索引 (本地文件清单 + 标题/摘要)
  GET  /check         与官方仓库逐篇比对 sha256, 返回有更新的 slug
  GET  /{slug}        读单篇 Markdown (?source=local 默认 | online 拉官方最新)

零新依赖: urllib + hashlib; 网络失败一律降级为本地结果, 不阻塞控制台。
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ONLINE_BASE = (
    "https://raw.githubusercontent.com/HMUG12/Audio-Disentangled-Reconstruction/main/"
)

# slug -> (标题, 分组, 摘要, 仓库相对路径)
# 只列用户可读文档; m*/audit/implementation-plan 等内部材料不在白名单。
DOC_CATALOG: dict[str, tuple[str, str, str, str]] = {
    "quickstart": ("快速开始", "入门", "安装、启动与第一次语音合成", "README.md"),
    "demos": ("功能演示", "入门", "每个功能的操作方式与预期结果", "docs/demos.md"),
    "gpu": ("GPU 兼容性", "指南", "显卡档位、显存需求与降级策略", "docs/gpu-compat.md"),
    "install": ("安装与验收", "指南", "Windows 安装包验收清单", "docs/architecture/windows-install-acceptance.md"),
    "update": ("更新清单说明", "指南", "增量更新包的结构与校验方式", "docs/update-manifest.md"),
    "api": ("TTS API 规范", "开发", "HTTP / WebSocket 接口与参数", "docs/tts-api-spec.md"),
    "arch": ("架构总览", "开发", "模块分层、运行时与数据流", "docs/architecture-design.md"),
    "release": ("1.2.0 发布说明", "发布", "上一版变更与已知问题", "docs/release-notes-v1.2.0.md"),
}

router = APIRouter(prefix="/api/adr/v1/docs", tags=["adr-docs"])

_MAX_BYTES = 800_000


def _local_path(rel: str) -> Path:
    p = (REPO / rel).resolve()
    try:
        p.relative_to(REPO)
    except ValueError:
        raise HTTPException(400, "非法文档路径")
    return p


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines()[:20]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def _read_local(rel: str) -> tuple[str, str, str]:
    """返回 (标题, 正文, 修改日期)。文件缺失/过大抛 HTTPException。"""
    p = _local_path(rel)
    if not p.is_file():
        raise HTTPException(404, "该文档未随当前安装包提供")
    if p.stat().st_size > _MAX_BYTES:
        raise HTTPException(413, "文档超过 800KB, 已拒绝加载")
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise HTTPException(500, f"读取失败: {e.__class__.__name__}")
    stamp = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")
    return _title_of(text, rel), text, stamp


def _online_base() -> str:
    base = os.environ.get("ADR_DOCS_ONLINE_BASE", DEFAULT_ONLINE_BASE).strip()
    return base if base.endswith("/") else base + "/"


def _read_online(rel: str) -> tuple[str, str, str]:
    """拉官方仓库原文; 网络不可用/超时/404 统一 502 (由前端提示)。"""
    url = _online_base() + rel.lstrip("/")
    req = Request(url, headers={"User-Agent": "ADR-Studio-Docs/1.3"})
    try:
        with urlopen(req, timeout=6) as resp:  # noqa: S310 — 固定 https 官方仓库
            raw = resp.read(_MAX_BYTES + 1)
            if len(raw) > _MAX_BYTES:
                raise HTTPException(413, "在线文档超过 800KB")
            text = raw.decode("utf-8", errors="replace")
            stamp = datetime.now().strftime("%Y-%m-%d")
    except HTTPError as e:
        # 404 表示官方仓库尚无该篇 (未发布), 交由调用方跳过; 其余按上游故障处理
        if e.code == 404:
            raise HTTPException(404, "官方仓库暂无该文档")
        raise HTTPException(502, f"官方仓库返回 HTTP {e.code}")
    except (URLError, TimeoutError, OSError):
        raise HTTPException(502, "无法连接官方仓库 (离线或网络受限)")
    return _title_of(text, rel), text, stamp


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _summary(rel: str, text: str, fallback: str) -> str:
    """取正文第一段可读文字作摘要 (跳过徽章/图片/表格/代码块, 剥离行内语法)。"""
    in_fence = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not s:
            continue
        # 标题 / 表格 / 引用 / HTML 块 / 徽章行
        if s.startswith(("#", ">", "|", "<", "![", "[![")):
            continue
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
        s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
        s = re.sub(r"`([^`]+)`", r"\1", s)
        s = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", s)
        s = re.sub(r"^([-*+]|\d+\.)\s+", "", s).strip()
        if not s or s.startswith(("[", "!", "|")):
            continue
        return s[:56] + ("…" if len(s) > 56 else "")
    return fallback


@router.get("", summary="文档索引")
@router.get("/", include_in_schema=False)
def docs_index():
    out = []
    for slug, (title, group, desc, rel) in DOC_CATALOG.items():
        try:
            t, text, stamp = _read_local(rel)
        except HTTPException:
            continue
        out.append({
            "slug": slug,
            "title": t or title,
            "group": group,
            "summary": _summary(rel, text, desc),
            "updated": stamp,
            "bytes": len(text.encode("utf-8")),
        })
    return {"docs": out, "online": bool(os.environ.get("ADR_DOCS_ONLINE_BASE", DEFAULT_ONLINE_BASE).strip())}


@router.get("/check", summary="比对官方仓库是否有更新")
def docs_check():
    updates: list[str] = []
    checked = 0
    local = 0
    for slug, (_t, _g, _d, rel) in DOC_CATALOG.items():
        try:
            _title, local_text, _stamp = _read_local(rel)
        except HTTPException:
            continue
        local += 1
        try:
            _ot, online_text, _ostamp = _read_online(rel)
        except HTTPException as e:
            if e.status_code == 404:
                continue  # 官方仓库暂无该篇 (未发布/路径不同), 跳过而不影响整体比对
            return {"online": False, "updates": [], "error": e.detail, "checked": checked, "local": local}
        checked += 1
        if _sha(online_text) != _sha(local_text):
            updates.append(slug)
    # local = 本机随包文档数: 为 0 时 checked 必然 0, 前端据此提示"未随包提供文档",
    # 不能把它当成"已是最新" (安装包漏拷 docs/ 的静默故障, 2026-11-14 深审 A1)。
    return {"online": True, "updates": updates, "checked": checked, "local": local}


@router.get("/{slug}", summary="读取单篇文档")
def docs_read(slug: str, source: str = "local"):
    hit = DOC_CATALOG.get(slug)
    if hit is None:
        raise HTTPException(404, "未知文档")
    title, group, desc, rel = hit
    if source == "online":
        t, text, stamp = _read_online(rel)
        origin = "online"
    else:
        t, text, stamp = _read_local(rel)
        origin = "local"
    return {
        "slug": slug,
        "title": t or title,
        "group": group,
        "summary": _summary(rel, text, desc),
        "updated": stamp,
        "markdown": text,
        "source": origin,
        "sha256": _sha(text),
    }
