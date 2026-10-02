"""ADR 安装验证脚本。

用法:
    python scripts/verify_install.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到 path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def check_python_version() -> bool:
    """检查 Python 版本。"""
    print("  Python version:", sys.version.split()[0])
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print("  [FAIL] 需要 Python 3.10+")
        return False
    print("  [OK] Python 3.10+")
    return True


def check_torch() -> bool:
    """检查 PyTorch。"""
    try:
        import torch

        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
            print(f"  VRAM: {vram:.1f} GB")
        return True
    except ImportError:
        print("  [FAIL] PyTorch 未安装")
        return False


def check_adr_import() -> bool:
    """检查 adr 包可正常导入。"""
    try:
        import adr

        print(f"  adr version: {adr.__version__}")
        print(f"  adr license: {adr.__license__}")
        return True
    except ImportError as e:
        print(f"  [FAIL] adr 导入失败: {e}")
        return False


def check_adr_core() -> bool:
    """检查 adr.core 模块。"""
    try:
        from adr.core import (
            load_config,
            setup_device,
            get_logger,
            REGISTRY,
        )

        # 实际调用一次
        logger = get_logger("test")
        cfg = load_config(preset="vram_6gb")
        device = setup_device(verbose=False)

        print(f"  Config loaded: {cfg.config_preset}")
        print(f"  Device: {device.device}, VRAM: {device.vram_preset}")
        print(f"  Registries: {len(REGISTRY.all())} categories")
        return True
    except Exception as e:
        print(f"  [FAIL] adr.core 出错: {e}")
        return False


def check_registries() -> bool:
    """检查各注册表为空 (M1 Day 2 后会有内容)。"""
    from adr.core import REGISTRY

    all_empty = True
    for name, reg in REGISTRY.all().items():
        n = len(reg)
        status = "empty" if n == 0 else f"{n} items"
        print(f"  {name}: {status}")
    return True


def main() -> int:
    print("=" * 60)
    print("ADR Framework Installation Check")
    print("=" * 60)

    checks = [
        ("Python version", check_python_version),
        ("PyTorch", check_torch),
        ("ADR import", check_adr_import),
        ("ADR core", check_adr_core),
        ("Registries", check_registries),
    ]

    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        results.append(fn())

    print("\n" + "=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(f"[SUCCESS] {n_pass}/{n_total} checks passed")
        return 0
    else:
        print(f"[WARN] {n_pass}/{n_total} checks passed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
