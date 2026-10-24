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


@pytest.fixture(autouse=True)
def _isolate_runtime(monkeypatch):
    """Runtime 闭环会直写 os.environ: 全局默认只监控, 并隔离模块状态与分块键。
    test_runtime.py 自己的 fixture 会撤掉 ADR_RUNTIME_AUTOTUNE 以测试微调。"""
    import os
    from collections import deque

    from adr.core import runtime

    monkeypatch.setenv("ADR_RUNTIME_AUTOTUNE", "0")
    for k in runtime.TUNE_KEYS:
        if k in os.environ:
            monkeypatch.setenv(k, os.environ[k])
        else:   # 先 set 再 del: 让 monkeypatch 记下"原本不存在", 测试后被还原
            monkeypatch.setenv(k, "x")
            monkeypatch.delenv(k)
    monkeypatch.setattr(runtime, "_samples", deque(maxlen=runtime.WINDOW))
    monkeypatch.setattr(runtime, "_events", deque(maxlen=20))
    monkeypatch.setattr(runtime, "_applied", {})
    monkeypatch.setattr(runtime, "_known_good", None)
    monkeypatch.setattr(runtime, "_frozen", False)


@pytest.fixture(scope="session")
def root_dir() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return ROOT / "configs"


# ---------------------------------------------------------------------------
# 默认音素词典可用性 (CI 转绿, 2026-10-17 review)
# opencpop-extension.txt (607 音素) 位于 third_party/DiffSinger, 不入检出树 —
# ubuntu 干净检出下 load_default_phoneme_dict() 回退到 6 特殊 token。
# 依赖真实词典内容的测试用 HAS_DEFAULT_DICT 做跳过守卫; 本地/全量环境照常跑。
# ---------------------------------------------------------------------------

def _default_dict_ready() -> bool:
    try:
        from adr.data.phoneme_dict import load_default_phoneme_dict

        return load_default_phoneme_dict().vocab_size > 6
    except Exception:
        return False


HAS_DEFAULT_DICT = _default_dict_ready()


@pytest.fixture
def requires_default_dict():
    """依赖真实音素词典的测试守卫: 词典缺失时跳过。"""
    if not HAS_DEFAULT_DICT:
        pytest.skip("默认音素词典在 third_party/DiffSinger (不入库), 干净检出下不可用")


@pytest.fixture
def default_config():
    """加载默认配置。"""
    from adr.core import load_config

    return load_config(preset="default")
