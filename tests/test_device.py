"""测试 adr.core.device 模块。"""

from adr.core import setup_device
from adr.core.device import VRAM_PRESETS, DeviceConfig


def test_setup_device_default():
    """setup_device 默认返回有效配置。"""
    cfg = setup_device(verbose=False)
    assert isinstance(cfg, DeviceConfig)
    assert cfg.device in ("cpu", "cuda", "mps")
    assert cfg.vram_preset in ("4gb", "6gb", "8gb", "12gb+", "default")


def test_force_preset_4gb():
    """强制 4GB 档位。"""
    cfg = setup_device(force_preset="4gb", verbose=False)
    assert cfg.vram_preset == "4gb"
    assert cfg.batch_size == VRAM_PRESETS["4gb"]["batch_size"]
    assert cfg.x_pad == 1


def test_force_preset_8gb():
    """强制 8GB 档位。"""
    cfg = setup_device(force_preset="8gb", verbose=False)
    assert cfg.vram_preset == "8gb"
    assert cfg.batch_size == VRAM_PRESETS["8gb"]["batch_size"]
    assert cfg.x_pad == 3


def test_device_summary_format():
    """设备摘要格式正确。"""
    cfg = DeviceConfig(device="cpu", gpu_name="Test", gpu_vram_gb=8.0)
    summary = cfg.summary()
    assert "cpu" in summary
    assert "8.0 GB" in summary


def test_vram_presets_keys():
    """显存预设包含所有档位。"""
    for key in ["4gb", "6gb", "8gb", "12gb+"]:
        assert key in VRAM_PRESETS
        preset = VRAM_PRESETS[key]
        assert "x_pad" in preset
        assert "batch_size" in preset
        assert "precision" in preset
