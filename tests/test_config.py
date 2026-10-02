"""测试 adr.core.config 模块。"""

from adr.core import load_config
from adr.core.config import ADRConfig, DataConfig, TrainConfig


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
