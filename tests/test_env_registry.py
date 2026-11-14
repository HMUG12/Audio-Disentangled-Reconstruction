"""环境变量登记门禁 (M1, 批次129)。

``adr/core/settings.ENV_REGISTRY`` 是全仓 ``ADR_*`` 变量的唯一名单。
本测试双向断言, 防止两类漂移:

1. 源码新增了变量但没登记 —— 排查配置问题时在登记表里找不到;
2. 登记表留着已删除的变量 —— 文档与代码不一致。

扫描范围: ``adr/**/*.py`` + ``desktop/src-tauri/src/**/*.rs``
(容器/构建脚本只消费同一批变量, 不额外定义)。
"""
from __future__ import annotations

import re
from pathlib import Path

from adr.core.settings import ENV_REGISTRY

ROOT = Path(__file__).resolve().parent.parent
TOKEN = re.compile(r"\bADR_[A-Z0-9_]+\b")

# 形如 ADR_XXX 但不是环境变量: Rust 侧常量 (端口默认值) 与测试用假名。
NOT_ENV = {
    "ADR_PREFERRED_PORT",   # desktop/src-tauri/src/lib.rs 的 const u16 = 9881
}


def _scan(paths) -> dict[str, str]:
    """token -> 首次出现文件 (相对仓库根), 用于失败信息定位。"""
    found: dict[str, str] = {}
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = p.relative_to(ROOT).as_posix()
        for m in TOKEN.finditer(text):
            found.setdefault(m.group(0), rel)
    return found


def _python_sources():
    return sorted((ROOT / "adr").rglob("*.py"))


def _rust_sources():
    return sorted((ROOT / "desktop" / "src-tauri" / "src").rglob("*.rs"))


def test_registry_covers_python_sources():
    found = _scan(_python_sources())
    missing = sorted(set(found) - set(ENV_REGISTRY) - NOT_ENV)
    assert not missing, (
        "以下 ADR_* 变量在 adr/ 源码中出现但未登记到 settings.ENV_REGISTRY: "
        + ", ".join(f"{n} ({found[n]})" for n in missing)
    )


def test_registry_covers_shell_sources():
    found = _scan(_rust_sources())
    missing = sorted(set(found) - set(ENV_REGISTRY) - NOT_ENV)
    assert not missing, (
        "以下 ADR_* 变量在壳源码中出现但未登记到 settings.ENV_REGISTRY: "
        + ", ".join(f"{n} ({found[n]})" for n in missing)
    )


def test_registry_has_no_dead_entries():
    found = set(_scan(_python_sources())) | set(_scan(_rust_sources()))
    dead = sorted(set(ENV_REGISTRY) - found)
    assert not dead, (
        "settings.ENV_REGISTRY 中存在源码里已不再出现的变量 (删除或改名后漏更新): "
        + ", ".join(dead)
    )


def test_registry_owners_are_known():
    owners = {
        "settings", "config", "server", "engine", "training",
        "console", "import", "bootstrap", "cli", "shell",
    }
    bad = sorted({v for v in ENV_REGISTRY.values() if v not in owners})
    assert not bad, f"ENV_REGISTRY 使用了未定义的归属标签: {bad}"
