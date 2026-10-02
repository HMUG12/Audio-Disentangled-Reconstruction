"""ADR 框架测试配置。"""

import sys
from pathlib import Path

# 添加 src 到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest


def pytest_addoption(parser):
    """注册自定义 CLI 选项。"""
    parser.addoption(
        "--run-asr",
        action="store_true",
        default=False,
        help="运行 ASR 真实推理测试 (会下载 faster-whisper 模型)",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="运行所有耗时 > 5s 的测试",
    )


@pytest.fixture(scope="session")
def root_dir() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return ROOT / "configs"


@pytest.fixture
def default_config():
    """加载默认配置。"""
    from adr.core import load_config

    return load_config(preset="default")
