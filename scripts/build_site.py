#!/usr/bin/env python3
"""生成 ADR Studio 官网文档站 (纯静态, 可直接部署到 Cloudflare Pages)。

口径: 文档白名单与 ``adr/server/docs_api.py`` 的 ``DOC_CATALOG`` 完全一致
(slug / 分组 / 摘要)。中文为主语言; 英文页仅在 ``docs/en/`` 确有译文时生成,
缺译篇目在英文站标注 "English version pending" 并链到中文页 — 不伪造翻译。

零新依赖: 仅标准库 (Markdown 渲染器也在本文件内, 先转义再渲染)。

用法::

    D:\\pyhon\\python.exe scripts\\build_site.py           # 生成 site/ 并自检
    D:\\pyhon\\python.exe scripts\\build_site.py --check   # 只自检已生成产物
"""
from __future__ import annotations

import argparse
import html
import posixpath
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
DOCS_DIR = SITE / "docs"
EN_DOCS_DIR = SITE / "en" / "docs"
ASSETS = SITE / "assets"

UPDATES_BASE = "https://adr-studio.pages.dev/"
GITHUB_REPO = "https://github.com/HMUG12/Audio-Disentangled-Reconstruction"
GITHUB_BLOB = GITHUB_REPO + "/blob/main/"
RELEASES = GITHUB_REPO + "/releases"
# 腾讯文档「ADR Studio 问题与意见收集表」— 站内唯一反馈入口
FEEDBACK_FORM = "https://docs.qq.com/form/page/DSkhyWHJnS0ZUa1ZB"


# --------------------------------------------------------------------------- #
# 文档白名单 — 与 adr/server/docs_api.py::DOC_CATALOG 同口径
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Doc:
    slug: str
    group_zh: str
    group_en: str
    title_zh: str
    title_en: str
    desc_zh: str
    desc_en: str
    path_zh: str
    path_en: str = ""


DOCS: tuple[Doc, ...] = (
    Doc(
        "quickstart", "入门", "Getting started",
        "快速开始", "Quick Start",
        "安装、启动与第一次语音合成",
        "Install, launch and synthesize your first clip",
        "README.md", "docs/en/quickstart.md",
    ),
    Doc(
        "guide", "入门", "Getting started",
        "使用指南", "User Guide",
        "控制台逐屏说明、合成与训练流程、导出与调用",
        "Screen-by-screen tour, synthesis and training flows",
        "docs/guide.md", "docs/en/guide.md",
    ),
    Doc(
        "demos", "入门", "Getting started",
        "功能演示", "Demos",
        "每个功能的操作方式与预期结果",
        "How each feature behaves and what to expect",
        "docs/demos.md",
    ),
    Doc(
        "troubleshooting", "指南", "Guides",
        "故障排查", "Troubleshooting",
        "启动失败、显存不足、合成异常、训练中断的排查路径",
        "Startup, VRAM, synthesis and training failure playbooks",
        "docs/troubleshooting.md", "docs/en/troubleshooting.md",
    ),
    Doc(
        "gpu", "指南", "Guides",
        "GPU 兼容性", "GPU Compatibility",
        "显卡档位、显存需求与降级策略",
        "GPU tiers, VRAM needs and fallbacks",
        "docs/gpu-compat.md",
    ),
    Doc(
        "install", "指南", "Guides",
        "安装验收", "Install & Acceptance",
        "Windows 安装包验收清单",
        "Windows installer acceptance checklist",
        "docs/architecture/windows-install-acceptance.md",
    ),
    Doc(
        "update", "指南", "Guides",
        "更新清单", "Update Manifest",
        "增量更新包的结构与校验方式",
        "Incremental update package layout and verification",
        "docs/update-manifest.md",
    ),
    Doc(
        "api", "开发", "Developers",
        "TTS API 规范", "TTS API Spec",
        "HTTP / WebSocket 接口与参数",
        "HTTP and WebSocket endpoints and parameters",
        "docs/tts-api-spec.md",
    ),
    Doc(
        "arch", "开发", "Developers",
        "架构设计", "Architecture",
        "模块分层、运行时与数据流",
        "Module layers, runtime and data flow",
        "docs/architecture-design.md",
    ),
    Doc(
        "release", "发布", "Releases",
        "1.2.0 发布说明", "Release Notes 1.2.0",
        "上一版变更与已知问题",
        "Previous release changes and known issues",
        "docs/release-notes-v1.2.0.md",
    ),
)

STRINGS = {
    "zh": {
        "site": "ADR Studio 文档",
        "brand_sub": "文档",
        "nav_features": "特性",
        "nav_docs": "文档",
        "nav_download": "下载",
        "nav_feedback": "问题反馈",
        "switch": "English",
        "back": "← 返回文档列表",
        "prev": "上一篇",
        "next": "下一篇",
        "toc": "文档目录",
        "on_page": "本页目录",
        "footer": "文档由仓库 docs/ 生成 · 更新服务",
        "feedback": "问题与意见收集表",
        "pending": "English version pending",
    },
    "en": {
        "site": "ADR Studio Docs",
        "brand_sub": "Docs",
        "nav_features": "Features",
        "nav_docs": "Docs",
        "nav_download": "Download",
        "nav_feedback": "Feedback",
        "switch": "中文",
        "back": "← Back to docs",
        "prev": "Previous",
        "next": "Next",
        "toc": "Documentation",
        "on_page": "On this page",
        "footer": "Generated from docs/ in the repository · Update service",
        "feedback": "Feedback form",
        "pending": "English version pending",
    },
}


# --------------------------------------------------------------------------- #
# Markdown 渲染
# --------------------------------------------------------------------------- #
FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
HEADING_RE = re.compile(r"^( {0,3})(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
HR_RE = re.compile(r"^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$")
UL_RE = re.compile(r"^(\s*)([-*+])[ \t]+(.*)$")
OL_RE = re.compile(r"^(\s*)(\d{1,9})[.)][ \t]+(.*)$")
CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1", re.S)
IMG_RE = re.compile(r"!\[([^\]]*)\]\(\s*([^\s)]*)(?:\s+&quot;([^&]*)&quot;)?\s*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(\s*([^\s)]*)(?:\s+&quot;([^&]*)&quot;)?\s*\)")
AUTOLINK_RE = re.compile(r"(?<![\w\"'=/])(https?://[^\s<>\x00\"]+)")
SLOT_RE = re.compile("\x00([IX])(\\d+)\x00")
STRONG3_RE = re.compile(r"\*\*\*(?!\s)(.+?)(?<!\s)\*\*\*")
STRONG_RE = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*")
UNDER2_RE = re.compile(r"(?<![\w])__(?!\s)(.+?)(?<!\s)__(?![\w])")
EM_RE = re.compile(r"(?<![\*])\*(?![\s*])(.+?)(?<![\s*])\*(?!\*)")
UNDER1_RE = re.compile(r"(?<![\w])_(?![\s_])(.+?)(?<![\s_])_(?![\w])")
STRIKE_RE = re.compile(r"~~(?!\s)(.+?)(?<!\s)~~")
CJK_RE = re.compile(r"[\u2e80-\u9fff\u3000-\u303f\uff00-\uffef]")
TAG_LINE_RE = re.compile(r"^ {0,3}</?([a-zA-Z][a-zA-Z0-9]*)((?:\s[^<>]*)?)/?>[ \t]*$")
COMMENT_RE = re.compile(r"^ {0,3}<!--.*-->[ \t]*$")
SAFE_TAGS = {
    "details", "summary", "br", "hr", "kbd", "sub", "sup", "b", "i", "em",
    "strong", "code", "span", "div", "p", "ul", "ol", "li", "table", "thead",
    "tbody", "tr", "td", "th", "a", "img", "small", "pre",
}


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def warn(msg: str) -> None:
    print(f"警告: {msg}", file=sys.stderr)


def fail(msg: str) -> None:
    print(f"错误: {msg}", file=sys.stderr)
    raise SystemExit(2)


def page_href(from_page: str, to_page: str) -> str:
    """站点内相对链接 (均为 posix 路径)."""
    start = posixpath.dirname(from_page) or "."
    return posixpath.relpath(to_page, start)


def slugify(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"[`*_~\[\]()#+>]", "", s).strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    s = s.replace(" ", "-")
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "section"


def _is_cjk(ch: str) -> bool:
    return bool(ch) and bool(CJK_RE.match(ch))


def join_soft(lines: list[str]) -> str:
    """软换行合并: 中文之间不插空格, 西文之间插一个空格。"""
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
        elif _is_cjk(out[-1]) or _is_cjk(line[0]):
            out += line
        else:
            out += " " + line
    return out


@dataclass
class Ctx:
    doc: Doc
    lang: str
    source_rel: str
    page: str
    ids: set[str]
    zh_index: dict[str, str]
    en_index: dict[str, str]
    images: list[tuple[str, str]] = field(default_factory=list)


def _doc_page(lang: str, slug: str) -> str:
    return f"docs/{slug}.html" if lang == "zh" else f"en/docs/{slug}.html"


def _file_url_target(raw: str) -> str | None:
    """把 file:///e:/.../ADR/<rel> 还原成仓库相对路径 (用于 GitHub 链接)。"""
    low = raw.lower()
    if not low.startswith("file:"):
        return None
    path = raw[5:].partition("#")[0].lstrip("/")
    idx = path.lower().find("/adr/")
    if idx < 0:
        return None
    rel = path[idx + len("/adr/"):]
    return rel if (REPO / rel).exists() else None


def resolve_href(raw: str, ctx: Ctx) -> tuple[str, str, str] | None:
    """返回 (url, css_class, hreflang); None 表示目标缺失, 降级为纯文字。"""
    raw = raw.strip()
    if not raw:
        return None
    low = raw.lower()
    if raw.startswith("#"):
        anchor = raw[1:].strip()
        return ("#" + anchor, "", "") if anchor in ctx.ids else None
    if low.startswith(("http://", "https://")):
        return (raw, "external", "")
    if low.startswith("mailto:"):
        return (raw, "", "")
    if low.startswith("file:"):
        rel = _file_url_target(raw)
        return None if rel is None else (GITHUB_BLOB + rel, "external", "")
    if "://" in low:
        return None

    path = raw.partition("#")[0].split("?")[0]
    if not path:
        return None
    target = posixpath.normpath(
        posixpath.join(posixpath.dirname(ctx.source_rel), path)
    )
    if target.startswith(".."):
        return None
    if ctx.lang == "en" and target in ctx.en_index:
        slug = ctx.en_index[target]
        return (page_href(ctx.page, f"en/docs/{slug}.html"), "", "en")
    if target in ctx.zh_index:
        slug = ctx.zh_index[target]
        href = page_href(ctx.page, f"docs/{slug}.html")
        if ctx.lang == "en":
            return (href, "lang-zh", "zh-CN")
        return (href, "", "")
    if target.endswith("/README.md") and "README.md" in ctx.zh_index:
        # docs/guide.md 等按 GitHub 习惯写 README.md, 实际指仓库根 README
        if ctx.lang == "en" and "docs/en/quickstart.md" in ctx.en_index:
            return (page_href(ctx.page, "en/docs/quickstart.html"), "", "en")
        return (page_href(ctx.page, "docs/quickstart.html"), "", "")
    path = REPO / target
    if path.is_file():
        return (GITHUB_BLOB + target, "external", "")
    if path.is_dir():
        return (GITHUB_REPO + "/tree/main/" + target, "external", "")
    return None


def _image_markup(index: int, ctx: Ctx, inline: bool) -> str:
    alt, src = ctx.images[index]
    label = alt or "图片"
    if inline:
        return f'<span class="img-ref" title="{esc(src)}">{label}</span>'
    resolved = resolve_href(src, ctx)
    if resolved is None:
        return f'<span class="img-ref" title="{esc(src)}">{label}</span>'
    url = resolved[0]
    if url.startswith("http"):
        return (
            f'<a class="img-ref" href="{esc(url)}" target="_blank" '
            f'rel="noopener noreferrer">{label}</a>'
        )
    return f'<span class="img-ref" title="{esc(src)}">{label}</span>'


def _link_markup(label: str, raw_href: str, ctx: Ctx) -> str:
    resolved = resolve_href(raw_href, ctx)
    if resolved is None:
        return label
    url, css, hreflang = resolved
    attrs = f' href="{esc(url)}"'
    if url.startswith("http"):
        attrs += ' target="_blank" rel="noopener noreferrer"'
    if css:
        attrs += f' class="{css}"'
    if hreflang:
        attrs += f' hreflang="{hreflang}"'
    return f"<a{attrs}>{label}</a>"


class Slots:
    """行内片段占位符 — 先摘出已生成的 HTML, 避免被后续格式规则二次处理。"""

    def __init__(self) -> None:
        self.items: list[str] = []

    def put(self, fragment: str) -> str:
        self.items.append(fragment)
        return f"\x00X{len(self.items) - 1}\x00"

    def restore(self, text: str, ctx: Ctx, inline_images: bool = False) -> str:
        text = re.sub(
            "\x00I(\\d+)\x00",
            lambda m: _image_markup(int(m.group(1)), ctx, inline_images),
            text,
        )
        for _ in range(4):
            new = re.sub(
                "\x00X(\\d+)\x00", lambda m: self.items[int(m.group(1))], text
            )
            if new == text:
                break
            text = new
        return text


def render_inline(text: str, ctx: Ctx) -> str:
    """行内渲染: 先抽出代码/链接并转义, 再套格式, 防注入。"""
    slots = Slots()

    def image_replace(m: re.Match[str]) -> str:
        ctx.images.append((esc(m.group(1).strip()), m.group(2).strip()))
        return f"\x00I{len(ctx.images) - 1}\x00"

    def autolink_replace(m: re.Match[str]) -> str:
        url = m.group(0)
        trailing = ""
        while url and url[-1] in ").,;:!?，。；：！？、":
            trailing = url[-1] + trailing
            url = url[:-1]
        return slots.put(_link_markup(url, url, ctx)) + trailing

    # 1) 行内代码先摘出, 避免其中的 * _ [ ] 被当成格式
    text = CODE_SPAN_RE.sub(
        lambda m: slots.put(f"<code>{esc(m.group(2).strip())}</code>"), text
    )
    # 2) 整体转义 (此后所有内容都是纯文本, 只有我们主动拼的标签是 HTML)
    text = esc(text)
    # 3) 图片 / 链接 / 裸 URL
    text = IMG_RE.sub(image_replace, text)
    text = LINK_RE.sub(
        lambda m: slots.put(
            _link_markup(
                slots.restore(m.group(1), ctx, inline_images=True),
                m.group(2),
                ctx,
            )
        ),
        text,
    )
    text = AUTOLINK_RE.sub(autolink_replace, text)
    # 4) 强调 / 删除线 (占位符不含标记字符, 不会被误伤)
    text = STRONG3_RE.sub(r"<strong><em>\1</em></strong>", text)
    text = STRONG_RE.sub(r"<strong>\1</strong>", text)
    text = UNDER2_RE.sub(r"<strong>\1</strong>", text)
    text = EM_RE.sub(r"<em>\1</em>", text)
    text = UNDER1_RE.sub(r"<em>\1</em>", text)
    text = STRIKE_RE.sub(r"<del>\1</del>", text)
    # 5) 还原
    return slots.restore(text, ctx)


class Renderer:
    """块级解析: 标题/段落/围栏代码/表格/列表/引用/分隔线/安全 HTML。"""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.headings: list[tuple[int, str, str]] = []
        self._id_counts: dict[str, int] = {}
        self._skipped_h1 = False

    # -- 工具 -------------------------------------------------------------
    def _heading_id(self, text: str) -> str:
        base = slugify(text)
        count = self._id_counts.get(base, 0)
        self._id_counts[base] = count + 1
        return base if count == 0 else f"{base}-{count + 1}"

    def is_block_start(self, lines: list[str], i: int) -> bool:
        line = lines[i]
        if not line.strip():
            return True
        if FENCE_RE.match(line) or HEADING_RE.match(line) or HR_RE.match(line):
            return True
        if line.lstrip().startswith(">"):
            return True
        if UL_RE.match(line) or OL_RE.match(line):
            return True
        if COMMENT_RE.match(line) or TAG_LINE_RE.match(line):
            return True
        return self.is_table_start(lines, i)

    @staticmethod
    def is_table_start(lines: list[str], i: int) -> bool:
        if i + 1 >= len(lines) or "|" not in lines[i] or "|" not in lines[i + 1]:
            return False
        head = [c for c in split_row(lines[i])]
        sep = split_row(lines[i + 1])
        if len(head) < 1 or len(sep) != len(head):
            return False
        return all(re.fullmatch(r"\s*:?-{2,}:?\s*", c) for c in sep)

    # -- 入口 -------------------------------------------------------------
    def render(self, lines: list[str]) -> str:
        return "\n".join(b for b in self.blocks(lines) if b)

    def blocks(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            fence = FENCE_RE.match(line)
            if fence:
                block, i = self.code_block(lines, i, fence)
                out.append(block)
                continue
            heading = HEADING_RE.match(line)
            if heading:
                block = self.heading_block(heading)
                if block:
                    out.append(block)
                i += 1
                continue
            if HR_RE.match(line):
                out.append("<hr>")
                i += 1
                continue
            if self.is_table_start(lines, i):
                block, i = self.table_block(lines, i)
                out.append(block)
                continue
            if line.lstrip().startswith(">"):
                block, i = self.quote_block(lines, i)
                out.append(block)
                continue
            if UL_RE.match(line) or OL_RE.match(line):
                block, i = self.list_block(lines, i)
                out.append(block)
                continue
            if COMMENT_RE.match(line):
                i += 1
                continue
            tag = TAG_LINE_RE.match(line)
            if tag:
                allowed, cleaned = self.html_line(line, tag)
                if allowed:
                    out.append(cleaned)
                else:
                    out.append(f"<p>{esc(line.strip())}</p>")
                i += 1
                continue
            block, i = self.paragraph_block(lines, i)
            out.append(block)
        return out

    # -- 各块 -------------------------------------------------------------
    def code_block(self, lines: list[str], i: int, fence: re.Match[str]):
        indent = len(fence.group(1))
        marker = fence.group(2)
        info = fence.group(3).strip().split()[0] if fence.group(3).strip() else ""
        info = re.sub(r"[^\w+#.-]", "", info)[:20]
        close = re.compile(r"^\s*" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$")
        body: list[str] = []
        i += 1
        while i < len(lines):
            if close.match(lines[i]):
                i += 1
                break
            raw = lines[i]
            body.append(raw[indent:] if raw.startswith(" " * indent) else raw.lstrip())
            i += 1
        code = esc("\n".join(body))
        lang_attr = f' data-lang="{esc(info)}"' if info else ""
        cls = ' class="mermaid"' if info == "mermaid" else ""
        return f"<pre{lang_attr}{cls}><code>{code}</code></pre>", i

    def heading_block(self, m: re.Match[str]) -> str:
        level = len(m.group(2))
        text = m.group(3).strip()
        if not text:
            return ""
        if level == 1 and not self._skipped_h1:
            # 文档自身 H1 已由页面标题呈现, 跳过避免重复
            self._skipped_h1 = True
            return ""
        label = render_inline(text, self.ctx)
        hid = self._heading_id(text)
        if level in (2, 3):
            self.headings.append((level, hid, label))
        return f'<h{level} id="{esc(hid)}">{label}</h{level}>'

    def table_block(self, lines: list[str], i: int):
        head = split_row(lines[i])
        aligns = []
        for cell in split_row(lines[i + 1]):
            cell = cell.strip()
            if cell.startswith(":") and cell.endswith(":"):
                aligns.append("t-center")
            elif cell.endswith(":"):
                aligns.append("t-right")
            else:
                aligns.append("t-left")
        rows: list[list[str]] = []
        i += 2
        while i < len(lines) and lines[i].strip() and "|" in lines[i]:
            rows.append(split_row(lines[i]))
            i += 1
        head_html = "".join(
            f'<th class="{aligns[k] if k < len(aligns) else "t-left"}">'
            f"{render_inline(c, self.ctx)}</th>"
            for k, c in enumerate(head)
        )
        body_html = []
        for row in rows:
            cells = []
            for k in range(len(head)):
                cell = row[k] if k < len(row) else ""
                cells.append(
                    f'<td class="{aligns[k] if k < len(aligns) else "t-left"}">'
                    f"{render_inline(cell, self.ctx)}</td>"
                )
            body_html.append("<tr>" + "".join(cells) + "</tr>")
        return (
            '<div class="table-wrap"><table><thead><tr>'
            + head_html
            + "</tr></thead><tbody>"
            + "".join(body_html)
            + "</tbody></table></div>",
            i,
        )

    def quote_block(self, lines: list[str], i: int):
        inner: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.lstrip().startswith(">"):
                inner.append(re.sub(r"^\s*>\s?", "", line))
                i += 1
                continue
            if line.strip() and inner and inner[-1].strip() and not self.is_block_start(lines, i):
                inner.append(line)
                i += 1
                continue
            break
        return f"<blockquote>\n{self.render(inner)}\n</blockquote>", i

    def list_block(self, lines: list[str], i: int):
        first = UL_RE.match(lines[i]) or OL_RE.match(lines[i])
        if first is None:
            return "", i + 1
        ordered = OL_RE.match(lines[i]) is not None
        pattern = OL_RE if ordered else UL_RE
        base = len(first.group(1))
        start_number = int(first.group(2)) if ordered else 0
        items: list[str] = []

        while i < len(lines):
            match = pattern.match(lines[i])
            if not match or len(match.group(1)) != base:
                break
            content_indent = match.start(3)
            content = [match.group(3)]
            i += 1
            while i < len(lines):
                line = lines[i]
                if not line.strip():
                    nxt = next((l for l in lines[i + 1:] if l.strip()), "")
                    nxt_match = pattern.match(nxt) if nxt else None
                    nxt_indent = len(nxt) - len(nxt.lstrip())
                    if nxt and (
                        nxt_indent > base
                        or (nxt_match and len(nxt_match.group(1)) == base)
                    ):
                        content.append("")
                        i += 1
                        continue
                    break
                indent = len(line) - len(line.lstrip())
                if indent > base:
                    content.append(line[min(content_indent, indent):])
                    i += 1
                    continue
                sibling = pattern.match(line)
                if sibling and len(sibling.group(1)) == base:
                    break
                if (
                    HR_RE.match(line)
                    or HEADING_RE.match(line)
                    or FENCE_RE.match(line)
                    or line.lstrip().startswith(">")
                ):
                    break
                content.append(line.strip())
                i += 1
            inner = self.render(content)
            if inner.startswith("<p>") and inner.count("<p>") == 1 and inner.endswith("</p>"):
                inner = inner[3:-4]
            items.append(f"<li>{inner}</li>")
            # 列表项之间的空行
            while i < len(lines) and not lines[i].strip():
                nxt = next((l for l in lines[i + 1:] if l.strip()), "")
                sibling = pattern.match(nxt) if nxt else None
                if sibling and len(sibling.group(1)) == base:
                    i += 1
                    continue
                break
            if i < len(lines) and not lines[i].strip():
                break
        if ordered:
            start_attr = f' start="{start_number}"' if start_number != 1 else ""
            return f"<ol{start_attr}>" + "".join(items) + "</ol>", i
        return "<ul>" + "".join(items) + "</ul>", i

    def paragraph_block(self, lines: list[str], i: int):
        buf = [lines[i]]
        i += 1
        while i < len(lines) and lines[i].strip() and not self.is_block_start(lines, i):
            buf.append(lines[i])
            i += 1
        return f"<p>{render_inline(join_soft(buf), self.ctx)}</p>", i

    @staticmethod
    def html_line(line: str, m: re.Match[str]) -> tuple[bool, str]:
        tag = m.group(1).lower()
        attrs = m.group(2) or ""
        if tag not in SAFE_TAGS:
            return False, ""
        if re.search(r"\bon[a-z]+\s*=", attrs, re.I) or "javascript:" in attrs.lower():
            return False, ""
        return True, line.strip()


def split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells: list[str] = []
    buf: list[str] = []
    in_code = False
    k = 0
    while k < len(s):
        ch = s[k]
        if ch == "\\" and k + 1 < len(s) and s[k + 1] in "|\\":
            buf.append(s[k + 1])
            k += 2
            continue
        if ch == "`":
            in_code = not in_code
            buf.append(ch)
            k += 1
            continue
        if ch == "|" and not in_code:
            cells.append("".join(buf).strip())
            buf = []
            k += 1
            continue
        buf.append(ch)
        k += 1
    cells.append("".join(buf).strip())
    return cells


def collect_heading_ids(lines: list[str]) -> set[str]:
    ids: set[str] = set()
    fence: str | None = None
    for line in lines:
        m = FENCE_RE.match(line)
        if fence is None and m:
            fence = m.group(2)[0] * len(m.group(2))
            continue
        if fence is not None:
            if line.strip().startswith(fence[0] * len(fence)) and not line.strip().strip(fence[0]):
                fence = None
            continue
        h = HEADING_RE.match(line)
        if h:
            ids.add(slugify(h.group(3)))
    return ids


# --------------------------------------------------------------------------- #
# 站点样式 (无 CDN / 无外链字体 / 无外链图片)
# --------------------------------------------------------------------------- #
SITE_CSS = """*,*::before,*::after{box-sizing:border-box}
:root{color-scheme:light dark;--bg:#fbfaf7;--panel:#fff;--fg:#1a1e25;--muted:#5e6673;--line:#e5e6e3;--accent:#2857d6;--soft:#eef2ff;--code:#f4f5f7;--radius:14px;--max:1120px}
@media (prefers-color-scheme:dark){:root{--bg:#101216;--panel:#171a20;--fg:#e9ebf0;--muted:#9aa4b2;--line:#272c35;--accent:#93b1ff;--soft:#1a2233;--code:#1b202a}}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.75 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:var(--max);margin:0 auto;padding:0 24px}
.site-header{position:sticky;top:0;z-index:20;background:var(--bg);border-bottom:1px solid var(--line)}
.header-inner{display:flex;align-items:center;justify-content:space-between;gap:16px;height:60px}
.brand{display:flex;align-items:center;gap:10px;color:var(--fg);font-weight:650;letter-spacing:.01em}
.brand:hover{text-decoration:none}
.brand-mark{display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:9px;background:var(--fg);color:var(--bg);font-size:12px;font-weight:700;letter-spacing:.06em}
.brand-text{font-size:15px}
.site-nav{display:flex;align-items:center;gap:18px;font-size:14px}
.site-nav a{color:var(--muted)}
.site-nav a:hover{color:var(--fg)}
.hero{padding:72px 0 48px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,var(--soft),transparent 70%)}
.eyebrow{margin:0 0 14px;color:var(--accent);font-size:13px;font-weight:650;letter-spacing:.08em;text-transform:uppercase}
.hero h1{margin:0;max-width:16em;font-size:clamp(30px,5vw,50px);line-height:1.16;letter-spacing:-.02em}
.lede{max-width:46em;margin:18px 0 0;color:var(--muted);font-size:17px}
.cta{display:flex;flex-wrap:wrap;gap:12px;margin-top:28px}
.btn{display:inline-flex;align-items:center;padding:10px 18px;border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--fg);font-size:14px;font-weight:600}
.btn:hover{text-decoration:none;border-color:var(--muted)}
.btn-primary{background:var(--fg);border-color:var(--fg);color:var(--bg)}
.btn-primary:hover{opacity:.88}
.btn-ghost{background:transparent}
.hero-note{margin:18px 0 0;color:var(--muted);font-size:13px}
.section{padding:56px 0;border-bottom:1px solid var(--line)}
.section h2{margin:0 0 8px;font-size:24px;letter-spacing:-.01em}
.section-note{margin:0 0 26px;color:var(--muted);font-size:14px}
.grid{display:grid;gap:16px}
.features{grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.feature{padding:20px;border:1px solid var(--line);border-radius:var(--radius);background:var(--panel)}
.feature h3{margin:0 0 8px;font-size:15px}
.feature p{margin:0;color:var(--muted);font-size:14px}
.entries{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}
.entry{display:block;padding:20px;border:1px solid var(--line);border-radius:var(--radius);background:var(--panel);color:var(--fg)}
.entry:hover{text-decoration:none;border-color:var(--accent)}
.entry .path{display:inline-block;margin-bottom:8px;padding:2px 8px;border-radius:999px;background:var(--soft);color:var(--accent);font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}
.entry h3{margin:0 0 6px;font-size:16px}
.entry p{margin:0;color:var(--muted);font-size:14px}
.doc-group{margin-top:26px}
.group-title{margin:0 0 12px;color:var(--muted);font-size:13px;font-weight:650;letter-spacing:.06em;text-transform:uppercase}
.cards{grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.doc-card{display:flex;flex-direction:column;gap:6px;padding:18px;border:1px solid var(--line);border-radius:var(--radius);background:var(--panel);color:var(--fg)}
.doc-card:hover{text-decoration:none;border-color:var(--accent);transform:translateY(-1px)}
.doc-card h4{margin:0;font-size:16px}
.doc-card p{margin:0;color:var(--muted);font-size:14px}
.doc-kicker{color:var(--muted);font-size:12px;letter-spacing:.04em;text-transform:uppercase}
.badge-pending{display:inline-block;margin-top:6px;padding:2px 8px;border:1px solid var(--line);border-radius:999px;color:var(--muted);font-size:12px}
.lang-hint{margin-left:6px;color:var(--muted);font-size:12px}
.page{display:grid;grid-template-columns:250px minmax(0,1fr);gap:48px;max-width:var(--max);margin:0 auto;padding:32px 24px 72px}
.sidebar{position:sticky;top:84px;align-self:start;max-height:calc(100vh - 108px);overflow:auto;font-size:14px}
.toc>summary{display:none}
.toc-group{margin:18px 0 8px;color:var(--muted);font-size:12px;font-weight:650;letter-spacing:.06em;text-transform:uppercase}
.toc-group:first-of-type{margin-top:0}
.toc-list{list-style:none;margin:0;padding:0}
.toc-list li{margin:2px 0}
.toc-list a{display:block;padding:4px 10px;border-radius:8px;color:var(--muted);line-height:1.5}
.toc-list a:hover{background:var(--soft);color:var(--fg);text-decoration:none}
.toc-list a[aria-current]{background:var(--soft);color:var(--accent);font-weight:600}
.toc-inpage a{font-size:13px}
.toc-inpage .lvl3 a{padding-left:22px}
.pending-tag{margin-left:6px;padding:0 6px;border:1px solid var(--line);border-radius:999px;font-size:11px}
.doc{min-width:0}
.back{display:inline-block;margin-bottom:18px;color:var(--muted);font-size:14px}
.notice{margin:0 0 20px;padding:12px 14px;border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;background:var(--panel);color:var(--muted);font-size:14px}
.breadcrumb{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px}
.doc-head{margin-bottom:28px;padding-bottom:22px;border-bottom:1px solid var(--line)}
.doc-head h1{margin:8px 0 10px;font-size:clamp(26px,3.4vw,38px);line-height:1.22;letter-spacing:-.015em}
.doc-head .lede{margin:0;font-size:16px}
.meta{margin:12px 0 0;color:var(--muted);font-size:13px}
.prose{font-size:16px}
.prose h2{margin:44px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line);font-size:22px;letter-spacing:-.01em}
.prose h3{margin:30px 0 10px;font-size:18px}
.prose h4{margin:24px 0 8px;font-size:16px}
.prose p{margin:12px 0}
.prose ul,.prose ol{margin:12px 0;padding-left:26px}
.prose li{margin:5px 0}
.prose li>ul,.prose li>ol{margin:6px 0}
.prose code{padding:2px 6px;border-radius:6px;background:var(--code);font:13.5px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;word-break:break-word}
.prose pre{margin:16px 0;padding:16px;border:1px solid var(--line);border-radius:12px;background:var(--code);overflow-x:auto}
.prose pre code{padding:0;background:none;font-size:13px;line-height:1.65}
.prose pre[data-lang]::before{content:attr(data-lang);display:block;margin-bottom:8px;color:var(--muted);font:11px/1 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase}
.prose blockquote{margin:16px 0;padding:10px 16px;border-left:3px solid var(--accent);border-radius:0 10px 10px 0;background:var(--panel);color:var(--muted)}
.prose blockquote p{margin:6px 0}
.prose hr{margin:32px 0;border:0;border-top:1px solid var(--line)}
.prose a{text-decoration:underline;text-underline-offset:2px}
.prose .external::after{content:" ↗";color:var(--muted);font-size:.85em;text-decoration:none}
.prose table{border-collapse:collapse;width:100%;font-size:14px}
.table-wrap{margin:16px 0;overflow-x:auto;border:1px solid var(--line);border-radius:12px}
.table-wrap table{min-width:520px}
.prose th,.prose td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
.prose th{background:var(--code);font-weight:600;text-align:left}
.prose tr:last-child td{border-bottom:0}
.t-left{text-align:left}.t-center{text-align:center}.t-right{text-align:right}
.img-ref{color:var(--muted);font-size:13px}
.pager{display:flex;gap:14px;margin:48px 0 0;padding-top:22px;border-top:1px solid var(--line)}
.pager-link{flex:1;display:flex;flex-direction:column;gap:4px;padding:14px 16px;border:1px solid var(--line);border-radius:var(--radius);background:var(--panel);color:var(--fg)}
.pager-link:hover{text-decoration:none;border-color:var(--accent)}
.pager-link span{color:var(--muted);font-size:12px}
.pager-link strong{font-size:15px;font-weight:600}
.pager-link.next{text-align:right;align-items:flex-end}
.doc-foot{margin-top:32px}
.site-footer{padding:34px 0;color:var(--muted);font-size:13px}
.site-footer p{margin:4px 0}
@media (max-width:900px){
  .page{display:block;padding:20px 20px 56px}
  .sidebar{position:static;max-height:none;margin-bottom:24px;padding:12px 14px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}
  .toc>summary{display:block;cursor:pointer;font-weight:600;color:var(--fg)}
  .toc[open] .toc-body{margin-top:10px}
  .site-nav{gap:12px;font-size:13px}
  .hero{padding:48px 0 36px}
  .section{padding:40px 0}
  .prose h2{margin-top:34px}
  .pager{flex-direction:column}
}
"""


# --------------------------------------------------------------------------- #
# 页面骨架
# --------------------------------------------------------------------------- #
def has_en(doc: Doc) -> bool:
    return bool(doc.path_en) and (REPO / doc.path_en).is_file()


def available_docs() -> list[Doc]:
    return [d for d in DOCS if (REPO / d.path_zh).is_file()]


def home_page(lang: str) -> str:
    return "index.html" if lang == "zh" else "en/index.html"


def head_html(lang: str, title: str, description: str, page: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="{lang}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n"
        f'<meta name="description" content="{esc(description)}">\n'
        '<meta name="color-scheme" content="light dark">\n'
        '<meta property="og:type" content="website">\n'
        f'<meta property="og:title" content="{esc(title)}">\n'
        f'<meta property="og:description" content="{esc(description)}">\n'
        f'<link rel="stylesheet" href="{esc(page_href(page, "assets/site.css"))}">\n'
        "</head>\n"
    )


def header_html(lang: str, page: str, slug: str = "") -> str:
    s = STRINGS[lang]
    home = home_page(lang)
    if lang == "zh":
        switch_target = (
            f"en/docs/{slug}.html"
            if slug and any(d.slug == slug and has_en(d) for d in DOCS)
            else "en/index.html"
        )
        switch_label = s["switch"]
    else:
        switch_target = f"docs/{slug}.html" if slug else "index.html"
        switch_label = s["switch"]
    return (
        '<header class="site-header"><div class="wrap header-inner">'
        f'<a class="brand" href="{esc(page_href(page, home))}">'
        '<span class="brand-mark">ADR</span>'
        f'<span class="brand-text">ADR Studio {s["brand_sub"]}</span></a>'
        '<nav class="site-nav">'
        f'<a href="{esc(page_href(page, home))}#features">{s["nav_features"]}</a>'
        f'<a href="{esc(page_href(page, home))}#docs">{s["nav_docs"]}</a>'
        f'<a href="{esc(page_href(page, switch_target))}" hreflang="{ "en" if lang == "zh" else "zh-CN" }">{switch_label}</a>'
        f'<a href="{FEEDBACK_FORM}" target="_blank" rel="noopener noreferrer">{s["nav_feedback"]}</a>'
        f'<a href="{RELEASES}" target="_blank" rel="noopener noreferrer">{s["nav_download"]}</a>'
        "</nav></div></header>\n"
    )


def footer_html(lang: str) -> str:
    s = STRINGS[lang]
    return (
        '<footer class="site-footer"><div class="wrap">'
        f'<p>ADR Studio · <a href="{GITHUB_REPO}" target="_blank" '
        'rel="noopener noreferrer">GitHub</a> · AGPL-3.0</p>'
        f'<p>{s["footer"]} <a href="{UPDATES_BASE}" target="_blank" '
        'rel="noopener noreferrer">adr-studio.pages.dev</a></p>'
        f'<p><a href="{FEEDBACK_FORM}" target="_blank" '
        f'rel="noopener noreferrer">{s["feedback"]}</a></p>'
        "</div></footer>\n"
    )


def doc_href(doc: Doc, lang: str, page: str) -> str:
    if lang == "en" and not has_en(doc):
        return page_href(page, f"docs/{doc.slug}.html")
    return page_href(page, _doc_page(lang, doc.slug))


def doc_label(doc: Doc, lang: str) -> str:
    return doc.title_zh if lang == "zh" else doc.title_en


def doc_desc(doc: Doc, lang: str) -> str:
    return doc.desc_zh if lang == "zh" else doc.desc_en


def groups(lang: str) -> list[str]:
    out: list[str] = []
    for doc in DOCS:
        name = doc.group_zh if lang == "zh" else doc.group_en
        if name not in out:
            out.append(name)
    return out


def group_of(doc: Doc, lang: str) -> str:
    return doc.group_zh if lang == "zh" else doc.group_en


def doc_cards(lang: str, page: str, docs: list[Doc]) -> str:
    out: list[str] = []
    for group in groups(lang):
        members = [d for d in docs if group_of(d, lang) == group]
        if not members:
            continue
        cards = []
        for doc in members:
            pending = lang == "en" and not has_en(doc)
            badge = (
                f'<span class="badge-pending">{STRINGS["en"]["pending"]}'
                '<span class="lang-hint">中文</span></span>'
                if pending
                else ""
            )
            attrs = ' class="doc-card is-pending"' if pending else ' class="doc-card"'
            hreflang = ' hreflang="zh-CN"' if pending else ""
            cards.append(
                f'<a{attrs} href="{esc(doc_href(doc, lang, page))}"{hreflang}>'
                f'<span class="doc-kicker">{esc(group)}</span>'
                f"<h4>{esc(doc_label(doc, lang))}</h4>"
                f"<p>{esc(doc_desc(doc, lang))}</p>{badge}</a>"
            )
        out.append(
            '<div class="doc-group">'
            f'<h3 class="group-title">{esc(group)}</h3>'
            f'<div class="grid cards">{"".join(cards)}</div></div>'
        )
    return "".join(out)


def sidebar_html(
    lang: str,
    page: str,
    docs: list[Doc],
    current_slug: str = "",
    headings: list[tuple[int, str, str]] | None = None,
) -> str:
    s = STRINGS[lang]
    parts = [
        '<aside class="sidebar"><details class="toc" open>',
        f'<summary>{s["toc"]}</summary><div class="toc-body">',
    ]
    for group in groups(lang):
        members = [d for d in docs if group_of(d, lang) == group]
        if not members:
            continue
        parts.append(f'<p class="toc-group">{esc(group)}</p><ul class="toc-list">')
        for doc in members:
            current = ' aria-current="page"' if doc.slug == current_slug else ""
            pending = lang == "en" and not has_en(doc)
            tag = '<span class="pending-tag">中文</span>' if pending else ""
            hreflang = ' hreflang="zh-CN"' if pending else ""
            parts.append(
                f'<li><a href="{esc(doc_href(doc, lang, page))}"{current}{hreflang}>'
                f"{esc(doc_label(doc, lang))}{tag}</a></li>"
            )
        parts.append("</ul>")
    if headings:
        parts.append(
            f'<p class="toc-group">{esc(STRINGS[lang]["on_page"])}</p>'
            '<ul class="toc-list toc-inpage">'
        )
        for level, hid, label in headings:
            parts.append(
                f'<li class="lvl{level}"><a href="#{esc(hid)}">{label}</a></li>'
            )
        parts.append("</ul>")
    parts.append("</div></details></aside>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# 落地页
# --------------------------------------------------------------------------- #
def landing_zh(page: str, docs: list[Doc]) -> str:
    features = [
        ("低资源训练", "LoRA / QLoRA 只训约 1% 参数，显存占用降低约 70%；4-8GB 显存即可训练，CPU 也能推理。"),
        ("流式 TTS 服务", "实现 GPT-SoVITS api_v3 双工流式协议，边合成边返回 WAV 音频块。"),
        ("OpenAI 兼容接口", "POST /v1/audio/speech 与 GET /v1/models，现有 OpenAI TTS 客户端可直接接入。"),
        ("N.E.K.O 零配置", "provider 选 GPT-SoVITS，地址保持默认 127.0.0.1:9881，音色下拉自动列出声音档案。"),
        ("全离线桌面端", "内置便携 Python 运行时与全部依赖，安装即用；底模下载完成后不再需要网络。"),
        ("多语言合成", "中 / 英 / 日 / 粤及中英混说，声音档案可标记语种，合成默认语言自动联动。"),
    ]
    entries = [
        ("/easy", "新手控制台", "选声音 → 打字 → 听效果，三步出声，零参数。"),
        ("/pro", "专业控制台", "训练、调参、看曲线、管模型，声音克隆与语音合成都在这里。"),
        ("/call", "调用控制台", "双端点调试、参数说明、生成 cURL / Python / JS 代码。"),
    ]
    body = (
        '<section class="hero"><div class="wrap">'
        '<p class="eyebrow">ADR Studio · 本地语音克隆与合成</p>'
        "<h1>把声音克隆留在自己的电脑里</h1>"
        '<p class="lede">ADR（Audio Disentangled Reconstruction）是一个运行在本机的语音克隆 / TTS '
        "工作台：几分钟参考音频就能完成克隆训练，并提供 GPT-SoVITS 兼容的流式合成服务。"
        "推理与训练全部本地完成，音频与文本不会上传到任何服务器。</p>"
        '<div class="cta">'
        f'<a class="btn btn-primary" href="{esc(page_href(page, "docs/quickstart.html"))}">快速开始</a>'
        f'<a class="btn" href="{RELEASES}" target="_blank" rel="noopener noreferrer">下载 Windows 安装包</a>'
        f'<a class="btn btn-ghost" href="{esc(page_href(page, "docs/troubleshooting.html"))}">遇到问题</a>'
        f'<a class="btn btn-ghost" href="{FEEDBACK_FORM}" target="_blank" rel="noopener noreferrer">问题与意见</a>'
        "</div>"
        '<p class="hero-note">CPU / GPU 双安装包 · 首次启动自动下载底模，之后完全离线</p>'
        "</div></section>"
        '<section class="section" id="features"><div class="wrap">'
        "<h2>特性</h2>"
        '<p class="section-note">训练、合成、对外服务同源，桌面端、服务端与容器版共用一套引擎。</p>'
        '<div class="grid features">'
        + "".join(f"<div class=\"feature\"><h3>{t}</h3><p>{d}</p></div>" for t, d in features)
        + "</div></div></section>"
        '<section class="section" id="entries"><div class="wrap">'
        "<h2>三个控制台入口</h2>"
        '<p class="section-note">能力同源、数据互通，只是操作颗粒度不同。</p>'
        '<div class="grid entries">'
        + "".join(
            f'<a class="entry" href="{esc(page_href(page, "docs/guide.html"))}">'
            f'<span class="path">{p}</span><h3>{t}</h3><p>{d}</p></a>'
            for p, t, d in entries
        )
        + "</div></div></section>"
        '<section class="section" id="docs"><div class="wrap">'
        "<h2>文档</h2>"
        '<p class="section-note">中文为主语言；英文站只收录已有译文的篇目，缺译篇目会标注 English version pending。</p>'
        + doc_cards("zh", page, docs)
        + "</div></section>"
        '<section class="section" id="update"><div class="wrap">'
        "<h2>更新服务</h2>"
        f'<p>桌面端的更新检查指向 <a href="{UPDATES_BASE}" target="_blank" '
        f'rel="noopener noreferrer">{UPDATES_BASE}</a> 下的 <code>update.json</code>；'
        "本站同样部署在该基址，文档与更新清单同域分发。</p>"
        '<p class="hero-note">文档来自仓库 docs/，修改文档后需重新构建本站。</p>'
        "</div></section>"
    )
    return (
        head_html(
            "zh-CN",
            "ADR Studio — 本地语音克隆与合成工作台",
            "ADR Studio 是运行在本机的语音克隆 / TTS 工作台：低资源 LoRA 训练、"
            "GPT-SoVITS 兼容流式合成、OpenAI 兼容接口，全部离线完成。",
            page,
        )
        + "<body>\n"
        + header_html("zh", page)
        + "<main>\n"
        + body
        + "\n</main>\n"
        + footer_html("zh")
        + "</body>\n</html>\n"
    )


def landing_en(page: str, docs: list[Doc]) -> str:
    features = [
        ("Low-resource training", "LoRA / QLoRA tunes about 1% of parameters and cuts VRAM use by roughly 70%. Training fits 4-8 GB GPUs; CPU inference works too."),
        ("Streaming TTS", "Implements the GPT-SoVITS api_v3 duplex streaming protocol and returns WAV chunks as they are generated."),
        ("OpenAI-compatible API", "POST /v1/audio/speech and GET /v1/models, so existing OpenAI TTS clients can connect unchanged."),
        ("Zero-config N.E.K.O", "Pick GPT-SoVITS as the provider, keep the default 127.0.0.1:9881, and voice profiles appear automatically."),
        ("Fully offline desktop app", "Ships a portable Python runtime and every dependency. After the base model is downloaded, no network is needed."),
        ("Multilingual synthesis", "Chinese, English, Japanese, Cantonese and mixed Chinese-English, with per-profile language hints."),
    ]
    entries = [
        ("/easy", "Easy console", "Pick a voice, type, listen. Three steps, no parameters."),
        ("/pro", "Pro console", "Train, tune, watch curves and manage models."),
        ("/call", "Call console", "Debug both endpoints, read parameter docs, generate cURL / Python / JS."),
    ]
    body = (
        '<section class="hero"><div class="wrap">'
        '<p class="eyebrow">ADR Studio · local voice cloning and TTS</p>'
        "<h1>Voice cloning that stays on your machine</h1>"
        '<p class="lede">ADR (Audio Disentangled Reconstruction) is a local voice-cloning / TTS '
        "workbench. Fine-tune a voice from a few minutes of reference audio, then serve it with a "
        "GPT-SoVITS-compatible streaming API. Training and inference run locally; audio and text "
        "never leave the computer.</p>"
        '<div class="cta">'
        f'<a class="btn btn-primary" href="{esc(page_href(page, "en/docs/quickstart.html"))}">Quick start</a>'
        f'<a class="btn" href="{RELEASES}" target="_blank" rel="noopener noreferrer">Download for Windows</a>'
        f'<a class="btn btn-ghost" href="{esc(page_href(page, "en/docs/troubleshooting.html"))}">Troubleshooting</a>'
        f'<a class="btn btn-ghost" href="{FEEDBACK_FORM}" target="_blank" rel="noopener noreferrer">Feedback</a>'
        "</div>"
        '<p class="hero-note">CPU and GPU installers · base model downloads on first launch, offline afterwards</p>'
        "</div></section>"
        '<section class="section" id="features"><div class="wrap">'
        "<h2>Features</h2>"
        '<p class="section-note">One engine behind the desktop app, the headless server and the container image.</p>'
        '<div class="grid features">'
        + "".join(f'<div class="feature"><h3>{t}</h3><p>{d}</p></div>' for t, d in features)
        + "</div></div></section>"
        '<section class="section" id="entries"><div class="wrap">'
        "<h2>Three consoles</h2>"
        '<p class="section-note">Same engine and data; different level of control.</p>'
        '<div class="grid entries">'
        + "".join(
            f'<a class="entry" href="{esc(page_href(page, "en/docs/guide.html"))}">'
            f'<span class="path">{p}</span><h3>{t}</h3><p>{d}</p></a>'
            for p, t, d in entries
        )
        + "</div></div></section>"
        '<section class="section" id="docs"><div class="wrap">'
        "<h2>Documentation</h2>"
        '<p class="section-note">English pages exist only where a translation exists. '
        'Untranslated docs are marked English version pending and link to the Chinese original.</p>'
        + doc_cards("en", page, docs)
        + "</div></section>"
        '<section class="section" id="update"><div class="wrap">'
        "<h2>Update service</h2>"
        f'<p>The desktop app checks <code>update.json</code> under '
        f'<a href="{UPDATES_BASE}" target="_blank" rel="noopener noreferrer">{UPDATES_BASE}</a>. '
        "This site is deployed at the same base, so docs and update manifests share one origin.</p>"
        '<p class="hero-note">Docs are generated from docs/ in the repository; rebuild the site after editing them.</p>'
        "</div></section>"
    )
    return (
        head_html(
            "en",
            "ADR Studio — local voice cloning and TTS workbench",
            "ADR Studio is a local voice-cloning and text-to-speech workbench: "
            "low-resource LoRA training, GPT-SoVITS streaming synthesis and an OpenAI-compatible API, fully offline.",
            page,
        )
        + "<body>\n"
        + header_html("en", page)
        + "<main>\n"
        + body
        + "\n</main>\n"
        + footer_html("en")
        + "</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# 文档页
# --------------------------------------------------------------------------- #
def render_document(doc: Doc, lang: str, docs: list[Doc], indices: dict) -> str:
    source = doc.path_zh if lang == "zh" else doc.path_en
    try:
        raw = (REPO / source).read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"读取失败 {source}: {exc}")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    page = _doc_page(lang, doc.slug)
    ctx = Ctx(
        doc=doc,
        lang=lang,
        source_rel=source,
        page=page,
        ids=collect_heading_ids(lines),
        zh_index=indices["zh"],
        en_index=indices["en"],
    )
    renderer = Renderer(ctx)
    content = renderer.render(lines)
    if not content.strip():
        fail(f"渲染结果为空, 拒绝生成空页: {source}")
    s = STRINGS[lang]
    title = doc_label(doc, lang)
    desc = doc_desc(doc, lang)
    home = home_page(lang)
    idx = docs.index(doc)
    prev_doc = docs[idx - 1] if idx > 0 else None
    next_doc = docs[idx + 1] if idx + 1 < len(docs) else None

    pager = ['<nav class="pager">']
    if prev_doc:
        pager.append(
            f'<a class="pager-link prev" href="{esc(doc_href(prev_doc, lang, page))}">'
            f'<span>{s["prev"]}</span><strong>{esc(doc_label(prev_doc, lang))}</strong></a>'
        )
    if next_doc:
        pager.append(
            f'<a class="pager-link next" href="{esc(doc_href(next_doc, lang, page))}">'
            f'<span>{s["next"]}</span><strong>{esc(doc_label(next_doc, lang))}</strong></a>'
        )
    pager.append("</nav>")

    notice = ""
    if lang == "en" and any(not has_en(d) for d in docs):
        notice = (
            '<p class="notice">Some documents are only available in Chinese. '
            f'<a href="{esc(page_href(page, "index.html"))}">Open the Chinese docs</a> '
            '— the English translation is pending.</p>'
        )

    meta_label = "源文件" if lang == "zh" else "Source"
    body = (
        '<div class="page">\n'
        + sidebar_html(lang, page, docs, doc.slug, renderer.headings)
        + '\n<main class="doc">'
        f'<a class="back" href="{esc(page_href(page, home))}#docs">{s["back"]}</a>'
        + notice
        + '<header class="doc-head">'
        f'<p class="doc-kicker">{esc(group_of(doc, lang))}</p>'
        f"<h1>{esc(title)}</h1>"
        f'<p class="lede">{esc(desc)}</p>'
        f'<p class="meta">{meta_label} <code>{esc(source)}</code></p>'
        "</header>"
        f'<article class="prose">\n{content}\n</article>'
        + "".join(pager)
        + f'<p class="doc-foot"><a class="back" href="{esc(page_href(page, home))}#docs">{s["back"]}</a></p>'
        "</main>\n</div>\n"
    )
    return (
        head_html(
            "zh-CN" if lang == "zh" else "en",
            f"{title} · {s['site']}",
            desc,
            page,
        )
        + "<body>\n"
        + header_html(lang, page, doc.slug)
        + body
        + footer_html(lang)
        + "</body>\n</html>\n"
    )


def write_page(rel: str, content: str) -> None:
    path = SITE / rel
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    except OSError as exc:
        fail(f"写入失败 {rel}: {exc}")


def build() -> None:
    docs = available_docs()
    for doc in DOCS:
        if doc not in docs:
            warn(f"源文件缺失, 跳过该篇: {doc.path_zh} (slug={doc.slug})")
    if not docs:
        fail("白名单内没有任何可用文档, 终止")

    indices = {
        "zh": {d.path_zh: d.slug for d in docs},
        "en": {d.path_en: d.slug for d in docs if has_en(d)},
    }

    write_page("assets/site.css", SITE_CSS)
    write_page("index.html", landing_zh("index.html", docs))
    write_page("en/index.html", landing_en("en/index.html", docs))

    zh_count = 0
    en_count = 0
    for doc in docs:
        write_page(f"docs/{doc.slug}.html", render_document(doc, "zh", docs, indices))
        zh_count += 1
        if has_en(doc):
            write_page(
                f"en/docs/{doc.slug}.html", render_document(doc, "en", docs, indices)
            )
            en_count += 1

    pending = [d.slug for d in docs if not has_en(d)]
    print(f"已生成 site/: 中文文档 {zh_count} 篇, 英文文档 {en_count} 篇, 落地页 2 个")
    if pending:
        print(f"英文缺译 (链接到中文页): {', '.join(pending)}")
    if not check_site(docs):
        raise SystemExit(1)
    print("构建完成。")


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def check_site(docs: list[Doc] | None = None) -> bool:
    docs = docs if docs is not None else available_docs()
    problems: list[str] = []
    pages = ["index.html", "en/index.html"]
    pages += [f"docs/{d.slug}.html" for d in docs]
    pages += [f"en/docs/{d.slug}.html" for d in docs if has_en(d)]

    if not (SITE / "assets" / "site.css").is_file():
        problems.append("缺少 assets/site.css")

    for rel in pages:
        path = SITE / rel
        if not path.is_file():
            problems.append(f"缺少页面: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        match = re.search(r"<title>(.*?)</title>", text, re.S)
        if not match or not match.group(1).strip():
            problems.append(f"{rel}: <title> 为空")
        if 'name="description"' not in text:
            problems.append(f"{rel}: 缺少 meta description")
        if "<html lang=" not in text:
            problems.append(f"{rel}: 缺少 lang 属性")
        if "\x00" in text:
            problems.append(f"{rel}: 残留渲染占位符 (NUL)")
        if "{{" in text:
            problems.append(f"{rel}: 存在未替换的 {{ 占位符")
        if "TODO" in text:
            problems.append(f"{rel}: 残留 TODO")
        if rel.startswith("docs/") and "返回文档列表" not in text:
            problems.append(f"{rel}: 缺少「返回文档列表」")
        if rel.startswith("docs/") and '<article class="prose">' not in text:
            problems.append(f"{rel}: 缺少正文容器")
        for href in re.findall(r'href="([^"]+)"', text):
            target = html.unescape(href)
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.partition("#")[0].split("?")[0]
            if not target:
                continue
            dest = posixpath.normpath(posixpath.join(posixpath.dirname(rel), target))
            if dest.startswith("..") or not (SITE / dest).exists():
                problems.append(f"{rel}: 站内死链 {target}")

    if any(not has_en(d) for d in docs):
        en_index = SITE / "en" / "index.html"
        if en_index.is_file() and "English version pending" not in en_index.read_text(encoding="utf-8"):
            problems.append("en/index.html: 缺译篇目未标注 English version pending")

    if problems:
        for item in problems:
            print(f"自检失败: {item}", file=sys.stderr)
        return False
    print(f"自检通过: {len(pages)} 个页面 + assets/site.css")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 ADR Studio 官网静态站点 (site/)")
    parser.add_argument("--check", action="store_true", help="只自检已生成的 site/, 不重新构建")
    args = parser.parse_args(argv)
    if args.check:
        return 0 if check_site() else 1
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
