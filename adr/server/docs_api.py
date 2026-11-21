"""内置文档中心 API — 本地全量 + 官方仓库增量 + 中英双语 + 模糊搜索。

范围 (与产品口径一致): 仅用户文档 / API 规范 / 发布说明。
内部计划 (m*-plan、审计、路线图) 一律不进入白名单, 避免经公开服务外泄。

路由 (挂 /api/adr/v1/docs):
  GET  /              文档索引 (?lang=zh|en)
  GET  /search        模糊搜索 (?q=&lang=&limit=), 拼错/缺英文时给推荐
  GET  /check         与官方仓库逐篇比对 sha256, 返回有更新的 slug
  GET  /{slug}        读单篇 Markdown (?lang=&source=local|online)

语言口径: 请求语言优先, 该篇没有对应语言文件时回落中文并在响应里
标 ``lang/fallback`` — 前端据此提示「英文版待补」, 而不是报错或空白。

零新依赖: urllib + hashlib + difflib; 网络失败一律降级为本地结果。
"""
from __future__ import annotations

import difflib
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException, Query

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ONLINE_BASE = (
    "https://raw.githubusercontent.com/HMUG12/Audio-Disentangled-Reconstruction/main/"
)


@dataclass(frozen=True)
class DocEntry:
    """白名单条目: 双语分组/摘要兜底 + 各语言仓库路径。"""

    group: dict[str, str]
    desc: dict[str, str]
    paths: dict[str, str]


# slug -> 条目。只列用户可读文档; m*/audit/implementation-plan 等内部材料不在白名单。
DOC_CATALOG: dict[str, DocEntry] = {
    "quickstart": DocEntry(
        group={"zh": "入门", "en": "Getting started"},
        desc={"zh": "安装、启动与第一次语音合成", "en": "Install, launch and synthesize your first clip"},
        paths={"zh": "README.md", "en": "docs/en/quickstart.md"},
    ),
    "guide": DocEntry(
        group={"zh": "入门", "en": "Getting started"},
        desc={"zh": "控制台逐屏说明、合成与训练流程、导出与调用", "en": "Screen-by-screen tour, synthesis and training flows"},
        paths={"zh": "docs/guide.md", "en": "docs/en/guide.md"},
    ),
    "demos": DocEntry(
        group={"zh": "入门", "en": "Getting started"},
        desc={"zh": "每个功能的操作方式与预期结果", "en": "How each feature behaves and what to expect"},
        paths={"zh": "docs/demos.md"},
    ),
    "troubleshooting": DocEntry(
        group={"zh": "指南", "en": "Guides"},
        desc={"zh": "启动失败、显存不足、合成异常、训练中断的排查路径", "en": "Startup, VRAM, synthesis and training failure playbooks"},
        paths={"zh": "docs/troubleshooting.md", "en": "docs/en/troubleshooting.md"},
    ),
    "gpu": DocEntry(
        group={"zh": "指南", "en": "Guides"},
        desc={"zh": "显卡档位、显存需求与降级策略", "en": "GPU tiers, VRAM needs and fallbacks"},
        paths={"zh": "docs/gpu-compat.md"},
    ),
    "install": DocEntry(
        group={"zh": "指南", "en": "Guides"},
        desc={"zh": "Windows 安装包验收清单", "en": "Windows installer acceptance checklist"},
        paths={"zh": "docs/architecture/windows-install-acceptance.md"},
    ),
    "update": DocEntry(
        group={"zh": "指南", "en": "Guides"},
        desc={"zh": "增量更新包的结构与校验方式", "en": "Incremental update package layout and verification"},
        paths={"zh": "docs/update-manifest.md"},
    ),
    "api": DocEntry(
        group={"zh": "开发", "en": "Developers"},
        desc={"zh": "HTTP / WebSocket 接口与参数", "en": "HTTP and WebSocket endpoints and parameters"},
        paths={"zh": "docs/tts-api-spec.md"},
    ),
    "arch": DocEntry(
        group={"zh": "开发", "en": "Developers"},
        desc={"zh": "模块分层、运行时与数据流", "en": "Module layers, runtime and data flow"},
        paths={"zh": "docs/architecture-design.md"},
    ),
    "release": DocEntry(
        group={"zh": "发布", "en": "Releases"},
        desc={"zh": "上一版变更与已知问题", "en": "Previous release changes and known issues"},
        paths={"zh": "docs/release-notes-v1.2.0.md"},
    ),
}

# 搜索无结果时前端推荐的兜底篇目 (问题处理 + 使用指南)
SUGGEST_SLUGS = ("troubleshooting", "guide")

router = APIRouter(prefix="/api/adr/v1/docs", tags=["adr-docs"])

_MAX_BYTES = 800_000


def _norm_lang(lang: str | None) -> str:
    return "en" if (lang or "").strip().lower().startswith("en") else "zh"


def _local_path(rel: str) -> Path:
    p = (REPO / rel).resolve()
    try:
        p.relative_to(REPO)
    except ValueError:
        raise HTTPException(400, "非法文档路径")
    return p


def _pick(entry: DocEntry, lang: str) -> tuple[str, str]:
    """请求语言有文件就用它, 否则回落中文。返回 (实际语言, 仓库路径)。"""
    rel = entry.paths.get(lang)
    if rel and _local_path(rel).is_file():
        return lang, rel
    return "zh", entry.paths["zh"]


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
    stamp = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    return _title_of(text, rel), text, stamp


def _online_base() -> str:
    base = os.environ.get("ADR_DOCS_ONLINE_BASE", DEFAULT_ONLINE_BASE).strip()
    return base if base.endswith("/") else base + "/"


def _read_online(rel: str) -> tuple[str, str, str]:
    """拉官方仓库原文; 网络不可用/超时/404 统一 502 (由前端提示)。"""
    url = _online_base() + rel.lstrip("/")
    req = Request(url, headers={"User-Agent": "ADR-Studio-Docs/1.3"})
    try:
        with urlopen(req, timeout=6) as resp:
            raw = resp.read(_MAX_BYTES + 1)
            if len(raw) > _MAX_BYTES:
                raise HTTPException(413, "在线文档超过 800KB")
            text = raw.decode("utf-8", errors="replace")
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
def docs_index(lang: str = ""):
    want = _norm_lang(lang)
    out = []
    for slug, entry in DOC_CATALOG.items():
        actual, rel = _pick(entry, want)
        try:
            t, text, stamp = _read_local(rel)
        except HTTPException:
            continue
        out.append({
            "slug": slug,
            "title": t or entry.desc[want],
            "group": entry.group[want],
            "summary": _summary(rel, text, entry.desc[want]),
            "updated": stamp,
            "bytes": len(text.encode("utf-8")),
            "lang": actual,
            "fallback": actual != want,
        })
    return {
        "docs": out,
        "lang": want,
        "online": bool(os.environ.get("ADR_DOCS_ONLINE_BASE", DEFAULT_ONLINE_BASE).strip()),
    }


def _snippet(text: str, needles: list[str]) -> str:
    """在正文里找第一个命中位置, 截 80 字上下文 (剥离 Markdown 记号)。"""
    low = text.lower()
    pos = -1
    for n in needles:
        if not n:
            continue
        pos = low.find(n)
        if pos >= 0:
            break
    if pos < 0:
        return ""
    start = max(0, pos - 32)
    raw = text[start:start + 110]
    raw = re.sub(r"[`*_#>]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return ("…" if start else "") + raw + "…"


@router.get("/search", summary="文档搜索 (模糊匹配 + 缺英文降级)")
def docs_search(
    q: str = Query("", description="关键词, 支持中文/英文/拼写近似"),
    lang: str = "",
    limit: int = 20,
):
    want = _norm_lang(lang)
    query = (q or "").strip()
    if not query:
        return {"q": "", "lang": want, "results": [], "suggestions": list(SUGGEST_SLUGS)}

    low_q = query.lower()
    terms = [t for t in re.split(r"[\s,，、;；/|]+", low_q) if t]
    needles = sorted(set(terms + [low_q]), key=len, reverse=True)
    cjk = [c for c in low_q if "\u4e00" <= c <= "\u9fff"]
    results = []
    for slug, entry in DOC_CATALOG.items():
        actual, rel = _pick(entry, want)
        try:
            title, text, stamp = _read_local(rel)
        except HTTPException:
            continue
        t_low = title.lower()
        d_low = entry.desc[want].lower()
        hay = text.lower()
        score = 0.0
        for term in needles:
            if term in t_low:
                score += 70 if term == low_q else 40
            if term in slug:
                score += 25
            if term in d_low:
                score += 18
            hits = hay.count(term)
            if hits:
                score += min(16, 3 + hits * 2)
        # 拼写近似 (英文输错/中文夹英文): 标题或 slug 与查询整体相似
        if difflib.get_close_matches(low_q, [slug, t_low], n=1, cutoff=0.55):
            score += 30
        # 中文短句无空格: 按字符覆盖率兜底, 避免整句只能精确命中
        if cjk:
            cov = sum(1 for c in set(cjk) if c in t_low or c in hay) / len(set(cjk))
            score += cov * 36
        if score < 12:
            continue
        results.append({
            "slug": slug,
            "title": title,
            "group": entry.group[want],
            "summary": _summary(rel, text, entry.desc[want]),
            "snippet": _snippet(text, needles),
            "lang": actual,
            "fallback": actual != want,
            "updated": stamp,
            "score": round(score, 1),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    limit = max(1, min(int(limit or 20), 50))
    return {
        "q": query,
        "lang": want,
        "results": results[:limit],
        "suggestions": list(SUGGEST_SLUGS),
    }


@router.get("/check", summary="比对官方仓库是否有更新")
def docs_check():
    updates: list[str] = []
    checked = 0
    local = 0
    targets: list[tuple[str, str, str]] = []
    for slug, entry in DOC_CATALOG.items():
        rel = entry.paths["zh"]
        try:
            _title, local_text, _stamp = _read_local(rel)
        except HTTPException:
            continue
        local += 1
        targets.append((slug, rel, _sha(local_text)))
    if targets:
        # 并发拉取 (上限 4): 串行时单篇 6s 超时会把最坏耗时堆到 60s,
        # 长期占用一个线程池工作线程, 挤占同源控制台请求。
        def _fetch(item: tuple[str, str, str]):
            slug, rel, local_sha = item
            try:
                _ot, online_text, _ostamp = _read_online(rel)
            except HTTPException as e:
                return slug, None, e
            return slug, _sha(online_text), None

        with ThreadPoolExecutor(max_workers=4) as pool:
            for slug, online_sha, err in pool.map(_fetch, targets):
                if err is not None:
                    if err.status_code == 404:
                        continue  # 官方仓库暂无该篇 (未发布/路径不同), 跳过而不影响整体比对
                    return {"online": False, "updates": [], "error": err.detail,
                            "checked": checked, "local": local}
                checked += 1
                if online_sha != local_sha:
                    updates.append(slug)
    # local = 本机随包文档数: 为 0 时 checked 必然 0, 前端据此提示"未随包提供文档",
    # 不能把它当成"已是最新" (安装包漏拷 docs/ 的静默故障, 2026-11-14 深审 A1)。
    return {"online": True, "updates": updates, "checked": checked, "local": local}


@router.get("/{slug}", summary="读取单篇文档")
def docs_read(slug: str, source: str = "local", lang: str = ""):
    entry = DOC_CATALOG.get(slug)
    if entry is None:
        raise HTTPException(404, "未知文档")
    want = _norm_lang(lang)
    actual, rel = _pick(entry, want)
    if source == "online":
        t, text, stamp = _read_online(rel)
        origin = "online"
    else:
        t, text, stamp = _read_local(rel)
        origin = "local"
    return {
        "slug": slug,
        "title": t or entry.desc[actual],
        "group": entry.group[want],
        "summary": _summary(rel, text, entry.desc[actual]),
        "updated": stamp,
        "markdown": text,
        "source": origin,
        "lang": actual,
        "fallback": actual != want,
        "sha256": _sha(text),
    }
