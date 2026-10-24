#!/usr/bin/env python3
"""合规门禁断言（批次 61，源自 2026-10-24 会议纪要"立即执行"四项）。

门禁清单（docs/m10-plan.md 规划重排 ③，方案 C 定稿）：
  1. LICENSE 为 AGPL-3.0 全文；pyproject / Cargo.toml / package.json 许可证精确为 AGPL-3.0-only
  2. README 含源码获取途径（AGPL-3.0 §6 源码随包提供）
  3. README 含风险披露与禁止性条款
  4. THIRD_PARTY_LICENSES.md 存在且含核心组件条目
  5. 首启条款流存在：prewarm.html 条款视图 + lib.rs 三命令 + .terms_agreed_v1 标记
  6. CI smoke 已接入本脚本（防门禁被静默移除）

用法：python scripts/check_compliance.py
全部通过返回 0，存在缺失列出明细返回 1（CI smoke 与发布前置均可复用）。
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
    problems: list[str] = []

    # 1. LICENSE 全文 + 许可证字段精确化 (纪要: AGPL-3.0 → AGPL-3.0-only)
    lic = ROOT / "LICENSE"
    if not lic.exists():
        problems.append("LICENSE: 文件不存在")
    else:
        lic_text = lic.read_text(encoding="utf-8", errors="replace")
        if "GNU AFFERO GENERAL PUBLIC LICENSE" not in lic_text.upper():
            problems.append("LICENSE: 内容不是 AGPL 全文")
        if "Version 3" not in lic_text:
            problems.append('LICENSE: 未找到 "Version 3" (AGPL-3.0 全文标记)')

    m = re.search(r'^license\s*=\s*\{text\s*=\s*"([^"]+)"\}', _read("pyproject.toml"), re.M)
    if not m or m.group(1) != "AGPL-3.0-only":
        problems.append(
            f'pyproject.toml: license 应为 {{text = "AGPL-3.0-only"}}，实际 {m.group(1) if m else "未找到"}'
        )

    cargo = _read("desktop/src-tauri/Cargo.toml")
    if 'license = "AGPL-3.0-only"' not in cargo:
        problems.append('Cargo.toml: [package] license != "AGPL-3.0-only"')

    pkgjson = json.loads(_read("desktop/package.json"))
    if pkgjson.get("license") != "AGPL-3.0-only":
        problems.append(f'package.json: license != "AGPL-3.0-only" (实际 {pkgjson.get("license")})')

    # 2+3. README: 风险披露 + 禁止性条款 + 源码获取途径 (AGPL-3.0 §6)
    readme = _read("README.md")
    if "## 风险披露与使用边界" not in readme:
        problems.append("README.md: 缺少「风险披露与使用边界」章节")
    if "AGPL-3.0-only" not in readme:
        problems.append("README.md: License 表述未精确为 AGPL-3.0-only")
    if "源码获取" not in readme or "Releases" not in readme:
        problems.append("README.md: 缺少源码获取途径说明 (AGPL-3.0 §6 源码随包)")

    # 4. 第三方许可清单
    tp = ROOT / "THIRD_PARTY_LICENSES.md"
    if not tp.exists():
        problems.append("THIRD_PARTY_LICENSES.md: 文件不存在")
    else:
        tp_text = tp.read_text(encoding="utf-8")
        for comp in ("GPT-SoVITS", "BigVGAN", "Tauri", "python-build-standalone"):
            if comp not in tp_text:
                problems.append(f"THIRD_PARTY_LICENSES.md: 缺少核心组件条目 {comp}")

    # 5. 首启条款流 (方案 C): prewarm 前置条款页 + 三命令 + 标记文件
    prewarm = _read("desktop/src/prewarm.html")
    for invoke in ('invoke("terms_agreed")', 'invoke("agree_terms")', 'invoke("reject_terms")'):
        if invoke not in prewarm:
            problems.append(f"prewarm.html: 缺少条款流调用 {invoke}")

    librs = _read("desktop/src-tauri/src/lib.rs")
    for cmd in ("fn terms_agreed", "fn agree_terms", "fn reject_terms", "fn terms_agreed_flag"):
        if cmd not in librs:
            problems.append(f"lib.rs: 缺少条款命令/辅助函数 {cmd}")
    if ".terms_agreed_v1" not in librs:
        problems.append("lib.rs: 缺少 .terms_agreed_v1 标记路径")

    # 6. CI smoke 已接入本脚本 (防门禁被静默移除)
    if "check_compliance.py" not in _read(".github/workflows/smoke.yml"):
        problems.append("smoke.yml: 未接入 check_compliance.py 门禁步骤")

    if problems:
        for p in problems:
            print(f"check_compliance: MISSING {p}", file=sys.stderr)
        print(f"check_compliance: {len(problems)} 项未达标", file=sys.stderr)
        return 1
    print("check_compliance: all gates OK (6 组检查通过)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
