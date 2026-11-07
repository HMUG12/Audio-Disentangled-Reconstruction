#!/usr/bin/env python3
"""依赖版本漂移门禁（批次76 #23）。

背景：`requirements.txt` / `pyproject.toml` 原来只有 `>=` 下限，而发货的
Windows 便携 runtime 与 Linux 容器都是**钉版**（`scripts/build_runtime.ps1`、
`docker/requirements-linux.txt`）。声明与发货各说各话，就会出现「本地能跑、
装机报错」——批次 75 的 modelscope 缺件、torch 大版本差异都属于这一类。

三项断言（全部纯源码解析，不需要装任何依赖，CI 发布门禁可直接跑）：
  1. 每条核心依赖必须带**上限**（`<`）或精确钉版（`==`），不允许裸下限；
  2. `requirements.txt` 与 `pyproject.toml` 的 `[project].dependencies` 逐条同源；
  3. 声明的区间必须**包含**实际发货的钉版（便携 runtime + Linux 容器），
     否则「声明允许、但没测过」的版本会被解析进来。

用法：python scripts/check_requirements.py
通过返回 0；任一项不满足列出明细返回 1。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"
RUNTIME_PS1 = ROOT / "scripts" / "build_runtime.ps1"
CONTAINER_REQS = ROOT / "docker" / "requirements-linux.txt"

# "name[extras]>=1.0,<2" -> ("name[extras]", ">=1.0,<2")
_REQ = re.compile(r"^([A-Za-z0-9_.\-]+(?:\s*\[[^\]]+\])?)\s*(.*)$")
_SPEC_PART = re.compile(r"^(>=|<=|==|!=|>|<)\s*(.+)$")


def _norm_name(name: str) -> str:
    """大小写不敏感 + extras 内部空白归一（uvicorn[standard]）。"""
    name = name.strip().lower().replace(" ", "")
    name = name.replace("_", "-")
    return name


def parse_requirements_text(text: str) -> dict[str, str]:
    """需求文件 → {归一包名: 说明符}。注释/空行/`-r` 等指令跳过。"""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _REQ.match(line)
        if not m:
            continue
        out[_norm_name(m.group(1))] = m.group(2).replace(" ", "")
    return out


def parse_pyproject_core(text: str) -> dict[str, str]:
    """只取 [project].dependencies 数组（extras 另算，不参与同源断言）。"""
    # 不能用 split("]") — 依赖里的 extras（uvicorn[standard]）自带右括号
    m = re.search(r"\ndependencies = \[(.*?)\n\]", text, re.DOTALL)
    body = m.group(1) if m else ""
    specs: dict[str, str] = {}
    for line in body.splitlines():
        line = line.split("#", 1)[0].strip().rstrip(",").strip()
        if not line or not line.startswith('"'):
            continue
        m = _REQ.match(line.strip('"'))
        if m:
            specs[_norm_name(m.group(1))] = m.group(2).replace(" ", "")
    return specs


def parse_pins(text: str) -> dict[str, str]:
    """钉版清单（`==`）→ {归一包名: 版本}。"""
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _REQ.match(line)
        if not m:
            continue
        ver = m.group(2).strip()
        if ver.startswith("=="):
            pins[_norm_name(m.group(1))] = ver[2:].strip()
    return pins


def _version_tuple(v: str) -> tuple[int, ...]:
    """'2.10.0+cu130' -> (2, 10, 0)；非数字段按 0（预发布/本地版本够用）。"""
    out: list[int] = []
    for seg in v.strip().lstrip("vV").split("."):
        m = re.match(r"\d+", seg)
        out.append(int(m.group(0)) if m else 0)
    return tuple(out)


def _cmp(a: str, b: str) -> int:
    ta, tb = _version_tuple(a), _version_tuple(b)
    n = max(len(ta), len(tb))
    ta += (0,) * (n - len(ta))
    tb += (0,) * (n - len(tb))
    return (ta > tb) - (ta < tb)


def spec_bounds(spec: str) -> tuple[str | None, str | None, bool]:
    """说明符 → (下限, 上限, 上限是否含等号)。"""
    lower: str | None = None
    upper: str | None = None
    inclusive = False
    for part in spec.split(","):
        m = _SPEC_PART.match(part.strip())
        if not m:
            continue
        op, ver = m.group(1), m.group(2)
        if op in (">=", ">"):
            lower = ver
        elif op in ("<", "<="):
            upper, inclusive = ver, op == "<="
    return lower, upper, inclusive


def _runtime_pins() -> dict[str, str]:
    """build_runtime.ps1 里 here-string 的钉版行（`name==ver`）。"""
    text = RUNTIME_PS1.read_text(encoding="utf-8")
    start = text.find('$requirements = @"')
    if start < 0:
        return {}
    block = text[start:].split('"@', 1)[0]
    return parse_pins(block)


def main() -> int:
    if not REQUIREMENTS.exists() or not PYPROJECT.exists():
        print("check_requirements: 缺 requirements.txt / pyproject.toml", file=sys.stderr)
        return 2

    declared = parse_requirements_text(REQUIREMENTS.read_text(encoding="utf-8"))
    core = parse_pyproject_core(PYPROJECT.read_text(encoding="utf-8"))
    problems: list[str] = []

    # 1) 上限 / 钉版
    for name, spec in declared.items():
        lower, upper, _ = spec_bounds(spec)
        if spec.startswith("=="):
            continue
        if upper is None:
            problems.append(f"requirements.txt: {name}{spec} 只有下限, 缺上限（<x）")
        elif lower is not None and _cmp(lower, upper) >= 0:
            problems.append(f"requirements.txt: {name}{spec} 区间为空")

    # 2) requirements.txt <-> pyproject [project].dependencies 同源
    for name, spec in declared.items():
        if name not in core:
            problems.append(f"pyproject.toml 缺核心依赖 {name}{spec}")
        elif core[name] != spec:
            problems.append(
                f"{name}: requirements.txt «{spec}» != pyproject.toml «{core[name]}»")
    for name, spec in core.items():
        if name not in declared:
            problems.append(f"requirements.txt 缺核心依赖 {name}{spec}")

    # 3) 声明区间必须包含发货钉版
    shipped = {"便携 runtime (build_runtime.ps1)": _runtime_pins()}
    if CONTAINER_REQS.exists():
        shipped["Linux 容器 (docker/requirements-linux.txt)"] = parse_pins(
            CONTAINER_REQS.read_text(encoding="utf-8"))
    for where, pins in shipped.items():
        for name, ver in pins.items():
            spec = declared.get(name)
            if spec is None:
                continue
            lower, upper, inclusive = spec_bounds(spec)
            if lower is not None and _cmp(ver, lower) < 0:
                problems.append(f"{where}: {name}=={ver} 低于声明下限 {spec}")
            if upper is not None:
                ok = _cmp(ver, upper) <= 0 if inclusive else _cmp(ver, upper) < 0
                if not ok:
                    problems.append(f"{where}: {name}=={ver} 超出声明上限 {spec}")

    if problems:
        for p in problems:
            print(f"check_requirements: MISMATCH {p}", file=sys.stderr)
        print(f"check_requirements: {len(problems)} 处漂移", file=sys.stderr)
        return 1

    print(f"check_requirements: {len(declared)} 条核心依赖带上限, "
          f"与 pyproject 同源, 覆盖发货钉版 "
          f"({', '.join(f'{k}: {len(v)} 项' for k, v in shipped.items())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
