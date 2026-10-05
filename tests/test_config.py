"""测试 adr.core.config 模块。"""

import os
import pathlib
import types

from adr.core import load_config
from adr.core.config import ADRConfig, DataConfig, TrainConfig, adr_data_dir


def test_default_config_loads():
    """默认配置可以加载。"""
    cfg = load_config(preset="default")
    assert isinstance(cfg, ADRConfig)
    assert cfg.config_preset == "default"
    assert cfg.data.sample_rate == 24000
    assert cfg.train.batch_size == 4


def test_vram_4gb_config():
    """4GB 显存档位配置正确。"""
    cfg = load_config(preset="vram_4gb")
    assert cfg.train.batch_size == 1
    assert cfg.train.use_qlora is True
    assert cfg.train.gradient_accumulation_steps == 16


def test_vram_6gb_config():
    """6GB 显存档位配置正确。"""
    cfg = load_config(preset="vram_6gb")
    assert cfg.train.batch_size == 2
    assert cfg.train.use_qlora is True


def test_vram_8gb_config():
    """8GB 显存档位配置正确。"""
    cfg = load_config(preset="vram_8gb")
    assert cfg.train.batch_size == 4
    assert cfg.train.use_flash_attn is True
    assert cfg.train.use_qlora is False


def test_config_override():
    """override 参数生效。"""
    cfg = load_config(
        preset="default",
        override={"train": {"batch_size": 8, "learning_rate": 5e-5}},
    )
    assert cfg.train.batch_size == 8
    assert cfg.train.learning_rate == 5e-5


def test_config_to_yaml(tmp_path):
    """配置可保存为 YAML。"""
    cfg = load_config(preset="vram_6gb")
    out = tmp_path / "test.yaml"
    cfg.to_yaml(out)
    assert out.exists()

    # 重新加载
    cfg2 = load_config(config_path=out, preset="default")
    assert cfg2.train.batch_size == cfg.train.batch_size


# ---------------------------------------------------------------------------
# adr_data_dir 推导链 (批次28 复审 P3)
# ---------------------------------------------------------------------------


def _pin_os(monkeypatch, name: str) -> None:
    """钉死 config 模块视角的 os.name。

    不动真实 os.name: Python 3.13+ pathlib 按 os.name 解析 flavour,
    Windows 上伪装 posix 会让 PosixPath 实例化直接抛 UnsupportedOperation。
    替换 config 命名空间里的 os 引用则只影响被测函数; environ 传真身,
    保证 monkeypatch.setenv/delenv 继续生效。
    """
    fake_os = types.SimpleNamespace(name=name, environ=os.environ)
    monkeypatch.setattr("adr.core.config.os", fake_os)


def _pin_home(monkeypatch, home) -> None:
    """钉住 HOME/USERPROFILE, 让 Path.home() 落到临时目录。"""
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _pin_legacy_dir(monkeypatch, exists: bool) -> None:
    """把 Path.is_dir 对 F:/ADR_data 的回答钉死, 其余路径走真实判定。"""
    real_is_dir = pathlib.Path.is_dir

    def fake_is_dir(self):
        if self == pathlib.PureWindowsPath("F:/ADR_data"):
            return exists
        return real_is_dir(self)

    monkeypatch.setattr(pathlib.Path, "is_dir", fake_is_dir)


def test_data_dir_env_overrides_all(tmp_path, monkeypatch):
    """ADR_DATA_DIR 非空 → 直接采用, 优先级最高。"""
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path))
    assert adr_data_dir() == tmp_path


def test_data_dir_env_whitespace_ignored(tmp_path, monkeypatch):
    """纯空白 ADR_DATA_DIR 视为未设置, 落到默认推导 (批次27 ISSUE-4 对齐)。"""
    monkeypatch.setenv("ADR_DATA_DIR", "   ")
    _pin_os(monkeypatch, "posix")
    _pin_home(monkeypatch, tmp_path / "home")
    assert adr_data_dir() == tmp_path / "home" / ".adr" / "data"


def test_data_dir_windows_legacy_f(tmp_path, monkeypatch):
    """Windows + F:/ADR_data 已存在 → legacy 路径优先于 LOCALAPPDATA。"""
    _pin_os(monkeypatch, "nt")
    monkeypatch.delenv("ADR_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lapp"))
    _pin_legacy_dir(monkeypatch, exists=True)
    assert adr_data_dir() == pathlib.PureWindowsPath("F:/ADR_data")


def test_data_dir_windows_fallback_localappdata(tmp_path, monkeypatch):
    """Windows + F: 不存在 → %LOCALAPPDATA%\\ADR\\data。"""
    _pin_os(monkeypatch, "nt")
    monkeypatch.delenv("ADR_DATA_DIR", raising=False)
    _pin_legacy_dir(monkeypatch, exists=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lapp"))
    assert adr_data_dir() == tmp_path / "lapp" / "ADR" / "data"


def test_data_dir_windows_localappdata_empty(tmp_path, monkeypatch):
    """Windows + LOCALAPPDATA 未设置 → 回落 ~/AppData/Local。"""
    _pin_os(monkeypatch, "nt")
    monkeypatch.delenv("ADR_DATA_DIR", raising=False)
    _pin_legacy_dir(monkeypatch, exists=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    _pin_home(monkeypatch, tmp_path / "uh")
    assert adr_data_dir() == pathlib.Path.home() / "AppData" / "Local" / "ADR" / "data"


def test_data_dir_posix_skips_legacy(tmp_path, monkeypatch):
    """非 Windows 直接落到 ~/.adr/data, 不探测 F: 盘。"""
    _pin_os(monkeypatch, "posix")
    monkeypatch.delenv("ADR_DATA_DIR", raising=False)
    _pin_home(monkeypatch, tmp_path / "home")
    assert adr_data_dir() == tmp_path / "home" / ".adr" / "data"
