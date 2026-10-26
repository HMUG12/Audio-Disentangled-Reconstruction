"""批次66: 回归测试 (#0 #1 #2 #6 P17 #45-49)。"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from adr.models.alignment import _maximum_path_single
from adr.training.callbacks import ASRCallback, WavQualityCallback
from adr.training.efficient import has_quantized_modules, quantize_4bit


def _make_dummy_sample(idx: int):
    from adr.data.pipeline import TrainSample
    np.random.seed(idx)
    return TrainSample(
        sample_id=f"dummy_{idx:04d}",
        waveform=np.random.randn(24000).astype(np.float32) * 0.1,
        sample_rate=24000,
        text="你好世界",
        phonemes=["ni3", "hao3"],
        f0=np.random.uniform(80, 300, 100).astype(np.float32),
        mel=np.random.randn(80, 100).astype(np.float32) * 0.1,
        start_sec=0.0, end_sec=1.0,
    )


# ========== #0: MAS N>T ==========
def test_mas_n_greater_than_t_no_crash():
    """#0: MAS 在 N>T 时应退化为均匀单调对齐,不产生末音素吃光全部帧的异常。"""
    logp = np.random.randn(5, 3)
    mask = np.ones((5, 3), dtype=bool)
    path = _maximum_path_single(logp, mask, N=5, T=3)
    durations = path.sum(axis=1).astype(int)
    assert path.sum() == 3.0, f"N>T时path应填满所有帧 (got {path.sum()})"
    assert durations[-1] <= 2, f"N>T退化:末音素时长不应吃光全部帧 (got {durations})"
    assert (durations >= 0).all()


def test_mas_n_equals_t_diagonal():
    """MAS N==T边界正确:对角线每音素取1帧。"""
    logp = np.random.randn(3, 3); mask = np.ones((3, 3), dtype=bool)
    path = _maximum_path_single(logp, mask, N=3, T=3)
    assert path.sum() == 3.0
    assert np.allclose(path, np.eye(3)), f"N==T应得对角线对齐, got\n{path}"


# ========== P17: _QuantizedSelfAttention mask 语义 ==========
def _quant_attn():
    pytest.importorskip("bitsandbytes")
    from adr.training.efficient import _QuantizedSelfAttention
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(16, 4, batch_first=True).eval()
    return _QuantizedSelfAttention(mha).eval()


def test_quantized_attn_key_padding_mask_ignores_pad():
    """P17: bool key_padding_mask True=PAD=忽略 (MHA 约定), 不能被 SDPA 当成"参与"。"""
    q = _quant_attn()
    xv = torch.randn(1, 4, 16)
    xp = torch.cat([xv, 1e3 * torch.ones(1, 2, 16)], dim=1)
    kpm = torch.zeros(1, 6, dtype=torch.bool); kpm[:, 4:] = True
    out_masked, _ = q(xp, xp, xp, key_padding_mask=kpm)
    out_valid, _ = q(xv, xv, xv)
    assert (out_masked[:, :4] - out_valid).abs().max().item() < 1.0


def test_quantized_attn_bool_causal_mask_and_float_mask_combine():
    """P17: bool attn_mask True=禁止; float attn_mask + bool kpm 合并不崩溃。"""
    q = _quant_attn()
    x = torch.randn(1, 5, 16)
    causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    out, _ = q(x, x, x, attn_mask=causal)
    first, _ = q(x[:, :1], x[:, :1], x[:, :1])
    assert torch.allclose(out[:, 0], first[:, 0], atol=1e-3), "因果 mask 下第 0 帧只能看自己"
    kpm = torch.zeros(1, 5, dtype=torch.bool); kpm[:, 4:] = True
    out2, _ = q(x, x, x, attn_mask=torch.zeros(5, 5), key_padding_mask=kpm)
    assert torch.isfinite(out2).all()


# ========== #1: callback 训练结束显存清理 ==========
def test_wav_quality_callback_on_train_end_releases_cache():
    """#1: WavQualityCallback.on_train_end 应释放缓存的 BigVGAN。"""
    cb = WavQualityCallback(n_samples=1)
    # 伪造一个 _vocoder_cache 不释放的场景
    cb._vocoder_cache = "fake_bigvgan"
    from unittest.mock import MagicMock
    cb.on_train_end(trainer=MagicMock())
    assert cb._vocoder_cache is None, "on_train_end 应释放 _vocoder_cache"


def test_asr_callback_on_train_end_releases_all_caches():
    """#1: ASRCallback.on_train_end 应释放 BigVGAN 与所有 Whisper 缓存。"""
    cb = ASRCallback(asr_model_sizes=["tiny", "base"], use_placeholder_vocoder=True)
    cb._vocoder_cache = "fake_bigvgan"
    cb._asr_caches = {"tiny": "fake_whisper_tiny", "base": "fake_whisper_base"}
    cb._asr_cache = "fake_main"
    from unittest.mock import MagicMock
    cb.on_train_end(trainer=MagicMock())
    assert cb._vocoder_cache is None
    assert len(cb._asr_caches) == 0
    assert cb._asr_cache is None


# ========== #2: save_checkpoint 不反量化 self.model ==========
def test_save_checkpoint_4bit_does_not_mutate_training_model():
    """#2: save_checkpoint 反量化 4bit 时应复制模型,不改 self.model。"""
    from adr.models.sovits import SoVITS, SoVITSConfig
    from adr.training import Trainer, TrainerConfig
    from adr.training.dataset import VoiceCloneDataset

    model = SoVITS(SoVITSConfig(
        hidden_dim=16, n_layers=1, n_heads=2, ffn_dim=32,
        vocab_size=607, content_dim=8, timbre_dim=8,
    ))
    quantize_4bit(model)
    assert has_quantized_modules(model), "量化失败"
    ds = VoiceCloneDataset(samples=[_make_dummy_sample(i) for i in range(2)])
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TrainerConfig(
            epochs=1, batch_size=2, use_amp=False, use_gradient_checkpointing=False,
            use_lora=False, output_dir=tmp_dir, val_ratio=0.0,
        )
        trainer = Trainer(model=model, train_data=ds, config=config)
        params_before = {id(p) for p in trainer.model.parameters()}
        trainer.save_checkpoint(Path(tmp_dir) / "test.pt")
        assert has_quantized_modules(trainer.model), "save_checkpoint 后 self.model 应仍是量化状态"
        # optimizer 仍持有活模型的参数 (就地反量化会换掉参数对象, 优化器从此更新孤儿张量)
        assert {id(p) for p in trainer.model.parameters()} == params_before
        ckpt = torch.load(Path(tmp_dir) / "test.pt", map_location="cpu", weights_only=False)
        assert all(v.dtype != torch.uint8 for v in ckpt["model_state"].values()), "ckpt 应为反量化权重"


# ========== #6: BigVGAN import 不永久污染 sys.modules ==========
def _write_fake_bigvgan(d: Path) -> None:
    (d / "alias_free_activation" / "torch").mkdir(parents=True)
    (d / "env.py").write_text("class AttrDict(dict):\n    pass\n", encoding="utf-8")
    (d / "activations.py").write_text("X = 1\n", encoding="utf-8")
    (d / "utils.py").write_text("FAKE_BIGVGAN_UTILS = True\n", encoding="utf-8")
    (d / "alias_free_activation" / "__init__.py").write_text("", encoding="utf-8")
    (d / "alias_free_activation" / "torch" / "__init__.py").write_text("", encoding="utf-8")
    (d / "alias_free_activation" / "torch" / "act.py").write_text("", encoding="utf-8")
    (d / "bigvgan.py").write_text(
        "import utils\nassert utils.FAKE_BIGVGAN_UTILS\nclass BigVGAN:\n    pass\n",
        encoding="utf-8")


def test_bigvgan_import_restores_sys_path_and_modules(tmp_path, monkeypatch):
    """#6: 导入后 sys.path 还原, 原有 utils 模块恢复, 泛名不残留。"""
    import types
    import adr.vocoder.bigvgan as bv
    _write_fake_bigvgan(tmp_path)
    for n in ("env", "activations", "bigvgan"):
        monkeypatch.delitem(sys.modules, n, raising=False)
    host_utils = types.ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils", host_utils)
    path_before = list(sys.path)

    cls = bv._import_bigvgan_module(tmp_path)
    assert cls is not None and cls.__name__ == "BigVGAN"
    assert sys.path == path_before
    assert sys.modules["utils"] is host_utils, "宿主 utils 模块应被恢复"
    for n in ("env", "activations", "bigvgan"):
        assert n not in sys.modules, f"{n} 不应残留在 sys.modules"
    assert bv._ATTRDICT is not None and issubclass(bv._ATTRDICT, dict)


# ========== #45-49: 死代码删除验证 ==========
def test_dead_code_functions_removed():
    """#45-49: 无调用方的函数应已删除。"""
    import adr.training.efficient as eff
    import adr.training.grad_ckpt as gcp
    import adr.training.lora as lora
    # #45 enable_flash_attention / get_sdpa_backends / is_sdpa_available
    assert not hasattr(eff, "enable_flash_attention"), "#45 enable_flash_attention 应已删"
    assert not hasattr(eff, "get_sdpa_backends")
    assert not hasattr(eff, "is_sdpa_available")
    # #46 quantize_8bit
    assert not hasattr(eff, "quantize_8bit"), "#46 quantize_8bit 应已删"
    # #47 print_memory_stats / estimate_model_memory / estimate_inference_memory
    assert not hasattr(eff, "print_memory_stats"), "#47 print_memory_stats 应已删"
    assert not hasattr(eff, "estimate_model_memory")
    assert not hasattr(eff, "estimate_inference_memory")
    # #48 estimate_memory_saved
    assert not hasattr(gcp, "estimate_memory_saved"), "#48 estimate_memory_saved 应已删"
    # #49 freeze_non_lora
    assert not hasattr(lora, "freeze_non_lora"), "#49 freeze_non_lora 应已删"
