"""按「1.3.X-beta.<修次数>」统一版本面 (命名规则 2026-11-15 拍板)。

修次数 = 自 base tag (默认 v1.3.1) 起的提交数, 由 git 自动计算, 无需手工维护:
    1.3.1-beta.92   # 92 = git rev-list --count v1.3.1..HEAD
正式发布时 (用户明确指定) 用 --official 去掉后缀, 回到 1.3.1。

本脚本只改版本面, 不改任何行为; 幂等: 重复执行 (无新提交) 不产生 diff。
改完必须紧跟 scripts/check_versions.py 校验 16 个面一致。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BETA_RE = re.compile(r"^(?P<base>\d+\.\d+\.\d+)(?:-beta\.\d+)?$")


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _write(rel: str, text: str) -> bool:
    p = ROOT / rel
    old = p.read_text(encoding="utf-8")
    if old == text:
        return False
    # 保留原文件换行风格, 避免整文件 CRLF/LF 翻转产生噪音 diff
    raw = p.read_bytes()
    p.write_text(text, encoding="utf-8", newline="\r\n" if b"\r\n" in raw else "\n")
    return True


def _commits_since(tag: str) -> int:
    out = subprocess.run(
        ["git", "rev-list", "--count", f"{tag}..HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def _replace(text: str, old: str, new: str, where: str) -> str:
    if old == new:
        return text
    if old not in text:
        print(f"  [skip] {where}: 未找到 {old}", file=sys.stderr)
        return text
    return text.replace(old, new)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", action="store_true",
                    help="去掉 -beta.N 后缀, 回到正式版号")
    ap.add_argument("--base-tag", default="",
                    help="基准 tag (默认 v<当前版本去掉 beta 后缀>)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cur = re.search(r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"), re.MULTILINE)
    if not cur:
        print("stamp: pyproject.toml 未找到 version", file=sys.stderr)
        return 2
    old_ver = cur.group(1)
    m = BETA_RE.match(old_ver)
    if not m:
        print(f"stamp: 当前版本 {old_ver} 不符合 X.Y.Z[-beta.N] 规则", file=sys.stderr)
        return 2
    base = m.group("base")
    tag = args.base_tag or f"v{base}"

    n = 0 if args.official else _commits_since(tag)
    new_ver = base if args.official else f"{base}-beta.{n}"
    numeric = f"{base}.{n}"
    print(f"stamp: {old_ver} -> {new_ver} (修次数 {n}, 基准 {tag})")

    changed: list[str] = []

    def apply(rel: str, fn) -> None:
        text = _read(rel)
        new_text = fn(text)
        if args.dry_run:
            if new_text != text:
                changed.append(rel)
            return
        if _write(rel, new_text):
            changed.append(rel)

    # 1. pyproject.toml (事实源)
    apply("pyproject.toml", lambda t: re.sub(
        r'^version\s*=\s*"[^"]+"', f'version = "{new_ver}"', t, count=1, flags=re.MULTILINE))

    # 2. adr/__init__.py
    apply("adr/__init__.py", lambda t: re.sub(
        r'^__version__ = "[^"]+"', f'__version__ = "{new_ver}"', t, count=1, flags=re.MULTILINE))

    # 3. tauri.conf.json
    apply("desktop/src-tauri/tauri.conf.json",
          lambda t: t.replace(f'"version": "{old_ver}"', f'"version": "{new_ver}"', 1))

    # 4. Cargo.toml [package] 段
    def cargo(t: str) -> str:
        head, sep, tail = t.partition("[package]")
        if not sep:
            return t
        body, sep2, rest = tail.partition("\n[")
        body = re.sub(r'version\s*=\s*"[^"]+"', f'version = "{new_ver}"', body, count=1)
        return head + sep + body + sep2 + rest

    apply("desktop/src-tauri/Cargo.toml", cargo)

    # 5/6. package.json + package-lock.json
    def pkg(t: str) -> str:
        return t.replace(f'"version": "{old_ver}"', f'"version": "{new_ver}"')

    apply("desktop/package.json", pkg)
    apply("desktop/package-lock.json", pkg)

    # 7. installer.iss: 版本号 + Windows 数值版 (beta 串不能进文件属性)
    def iss(t: str) -> str:
        t = re.sub(r'#define MyAppVersion "[^"]+"', f'#define MyAppVersion "{new_ver}"', t, count=1)
        if "#define MyAppVersionNumeric" in t:
            t = re.sub(r'#define MyAppVersionNumeric "[^"]+"',
                       f'#define MyAppVersionNumeric "{numeric}"', t, count=1)
        else:
            t = t.replace(f'#define MyAppVersion "{new_ver}"',
                          f'#define MyAppVersion "{new_ver}"\n'
                          f'#define MyAppVersionNumeric "{numeric}"', 1)
        if "VersionInfoVersion=" not in t:
            t = t.replace("AppVersion={#MyAppVersion}",
                          "AppVersion={#MyAppVersion}\nVersionInfoVersion={#MyAppVersionNumeric}", 1)
        return t

    apply("scripts/installer/installer.iss", iss)

    # 8. launcher.html about 兜底文案
    apply("desktop/src/launcher.html", lambda t: t.replace(
        f'"version {old_ver}"', f'"version {new_ver}"'))

    # 9. README.md: 安装包名 + release 链接 + 镜像 tag
    def readme(t: str) -> str:
        for needle in (f"ADR-Studio-{old_ver}", f"v{old_ver}", f"adr-server:{old_ver}"):
            t = t.replace(needle, needle.replace(old_ver, new_ver))
        return t

    apply("README.md", readme)

    # 10-12. Docker 面
    apply("docker/Dockerfile", lambda t: t.replace(
        f"ARG ADR_VERSION={old_ver}", f"ARG ADR_VERSION={new_ver}"))
    apply("docker/docker-compose.yml", lambda t: t.replace(
        f"adr-server:{old_ver}", f"adr-server:{new_ver}"))
    apply("docker/docker-compose.gpu.yml", lambda t: t.replace(
        f"adr-server:{old_ver}", f"adr-server:{new_ver}"))
    apply("docker/README.md", lambda t: t.replace(
        f"adr-server:{old_ver}", f"adr-server:{new_ver}"))

    if args.dry_run:
        print("stamp(dry-run): 需修改 " + (", ".join(changed) if changed else "无"))
    else:
        print("stamp: 已更新 " + (", ".join(changed) if changed else "无 (已是最新)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
