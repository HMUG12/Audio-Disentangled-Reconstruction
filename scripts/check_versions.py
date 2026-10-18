#!/usr/bin/env python3
"""版本面一致性断言：所有版本字段必须与 pyproject.toml 一致。

覆盖面（批次 60 引入，源于批次 55 漏 bump 教训）：
  1. pyproject.toml                     （单一事实源）
  2. adr/__init__.py                    __version__
  3. desktop/src-tauri/tauri.conf.json  version
  4. desktop/src-tauri/Cargo.toml       [package] version
  5. desktop/package.json               version
  6. desktop/package-lock.json          顶层 version + packages[""].version
  7. scripts/installer/installer.iss    #define MyAppVersion
  8. desktop/src/launcher.html          about 弹窗 fallback 文案
  9. README.md                          安装包文件名 ADR-Studio-<ver>-x64

用法：python scripts/check_versions.py
全部一致返回 0，存在差异列出明细返回 1（CI smoke 与双包构建前置均可复用）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def main() -> int:
    m = re.search(r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"), re.M)
    if not m:
        print("check_versions: pyproject.toml 未找到 [project] version", file=sys.stderr)
        return 2
    ver = m.group(1)

    problems: list[str] = []

    if f'__version__ = "{ver}"' not in _read("adr/__init__.py"):
        problems.append(f"adr/__init__.py: __version__ != {ver}")

    tauri = json.loads(_read("desktop/src-tauri/tauri.conf.json"))
    if tauri.get("version") != ver:
        problems.append(f"tauri.conf.json: {tauri.get('version')} != {ver}")

    pkg = _read("desktop/src-tauri/Cargo.toml").split("[package]", 1)[1].split("[", 1)[0]
    m2 = re.search(r'version\s*=\s*"([^"]+)"', pkg)
    if not m2 or m2.group(1) != ver:
        problems.append(f"Cargo.toml [package] version != {ver}")

    pkgjson = json.loads(_read("desktop/package.json"))
    if pkgjson.get("version") != ver:
        problems.append(f"package.json: {pkgjson.get('version')} != {ver}")

    lock = json.loads(_read("desktop/package-lock.json"))
    lock_faces = (
        ("顶层", lock.get("version")),
        ('packages[""]', lock.get("packages", {}).get("", {}).get("version")),
    )
    for where, v in lock_faces:
        if v != ver:
            problems.append(f"package-lock.json {where}: {v} != {ver}")

    if f'#define MyAppVersion "{ver}"' not in _read("scripts/installer/installer.iss"):
        problems.append(f"installer.iss: MyAppVersion != {ver}")

    if f'"version {ver}"' not in _read("desktop/src/launcher.html"):
        problems.append(f"launcher.html: about fallback 文案 != version {ver}")

    if f"ADR-Studio-{ver}-x64" not in _read("README.md"):
        problems.append(f"README.md: 安装包文件名未含 ADR-Studio-{ver}-x64（下载链接/静默安装示例需同步）")

    if problems:
        for p in problems:
            print(f"check_versions: MISMATCH {p}", file=sys.stderr)
        print(f"check_versions: {len(problems)} 处不一致（事实源 pyproject = {ver}）", file=sys.stderr)
        return 1
    print(f"check_versions: all faces = {ver} (9 checks OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
