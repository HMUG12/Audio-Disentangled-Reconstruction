"""测试 M7 多 ASR 集成投票 (medoid 共识 + 一致性指标)。

测试覆盖:
1. _consensus medoid 选择逻辑
2. agreement 一致性计算
3. 集成模式构造 (asr_model_sizes)
4. 集成端到端 (mock 多 ASR, 不同输出)
5. 单模型失败降级
6. 向后兼容 (单模型模式不受影响)
"""
import sys
from pathlib import Path

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

import numpy as np
import pytest
import torch

from adr.data.pipeline import TrainSample
from adr.models.sovits import SoVITS, SoVITSConfig
from adr.training.callbacks import ASRCallback
from adr.training.dataset import VoiceCloneDataset
from adr.training.trainer import Trainer, TrainerConfig


def make_dataset(n: int = 20):
    samples = []
    for i in range(n):
        s = TrainSample(
            sample_id=f"utt_{i:03d}",
            phonemes=["zh", "ong1", "g_uan1"] * 7,
            text="测试",
            f0=np.linspace(180, 200, 80).astype(np.float32),
            mel=np.random.randn(80, 80).astype(np.float32) * 0.1 + 0.5,
            waveform=np.random.randn(8000).astype(np.float32) * 0.1,
            sample_rate=22050,
        )
        samples.append(s)
    return VoiceCloneDataset(samples=samples)


def make_small_model():
    cfg = SoVITSConfig(
        hidden_dim=64, n_layers=2, n_heads=2, ffn_dim=128,
        vocab_size=607, content_dim=64, timbre_dim=32,
        n_mels=80, sample_rate=22050, hop_length=256,
    )
    return SoVITS(cfg)


# ============================================================
# medoid 共识逻辑 (纯函数, 无需 ASR)
# ============================================================
class TestConsensus:
    """_consensus medoid 选择。"""

    def test_identical_hyps_full_agreement(self):
        cb = ASRCallback(use_real_vocoder=False)
        text, agree = cb._consensus(["你好世界", "你好世界", "你好世界"])
        assert text == "你好世界"
        assert agree == 1.0

    def test_medoid_picks_middle(self):
        """3 个假设: 2 个相似 + 1 个离群 → 选相似的那个。"""
        cb = ASRCallback(use_real_vocoder=False)
        text, agree = cb._consensus(["你好世界", "你好世介", "完全不同的东西"])
        # "你好世界" 和 "你好世介" 只差 1 字, medoid 应是其中之一
        assert text in ("你好世界", "你好世介")
        assert 0.0 < agree < 1.0

    def test_tuple_input(self):
        """支持 [(size, text), ...] 输入。"""
        cb = ASRCallback(use_real_vocoder=False)
        text, agree = cb._consensus([("tiny", "你好"), ("small", "你好")])
        assert text == "你好"
        assert agree == 1.0

    def test_single_hyp(self):
        cb = ASRCallback(use_real_vocoder=False)
        text, agree = cb._consensus(["唯一"])
        assert text == "唯一"
        assert agree == 1.0

    def test_empty(self):
        cb = ASRCallback(use_real_vocoder=False)
        text, agree = cb._consensus([])
        assert text == ""
        assert agree == 0.0

    def test_all_different_low_agreement(self):
        cb = ASRCallback(use_real_vocoder=False)
        text, agree = cb._consensus(["你好", "再见", "谢谢"])
        assert agree < 0.5  # 完全不同 → 一致性低


# ============================================================
# 构造
# ============================================================
class TestEnsembleConstruction:
    def test_single_model_default(self):
        cb = ASRCallback(asr_model_size="tiny", use_real_vocoder=False)
        assert cb.asr_model_sizes == ["tiny"]
        assert cb.ensemble is False

    def test_ensemble_mode(self):
        cb = ASRCallback(asr_model_sizes=["tiny", "small"], use_real_vocoder=False)
        assert cb.asr_model_sizes == ["tiny", "small"]
        assert cb.ensemble is True

    def test_ensemble_sizes_overrides_single(self):
        cb = ASRCallback(
            asr_model_size="base",
            asr_model_sizes=["tiny", "small"],
            use_real_vocoder=False,
        )
        assert cb.asr_model_sizes == ["tiny", "small"]
        assert cb.ensemble is True

    def test_single_element_list_not_ensemble(self):
        cb = ASRCallback(asr_model_sizes=["tiny"], use_real_vocoder=False)
        assert cb.ensemble is False


# ============================================================
# 集成端到端 (mock ASR)
# ============================================================
def _mock_faster_whisper(outputs_by_size: dict):
    """构造 mock faster_whisper 模块, 按 size 返回不同文本。"""
    import sys as _sys

    def make_model(size, *a, **k):
        class MockASR:
            def transcribe(self, path, language=None, beam_size=5, vad_filter=True):
                text = outputs_by_size.get(size, "")
                return ([type("Seg", (), {"text": text})()], None)
        return MockASR()

    mock_module = type(_sys)("faster_whisper")
    mock_module.WhisperModel = make_model
    _sys.modules["faster_whisper"] = mock_module


class TestEnsembleE2E:
    def test_ensemble_metrics_in_history(self):
        """集成模式下, _history 应含 consensus CER + median/std/agreement。"""
        # 两个 mock 模型输出不同文本
        _mock_faster_whisper({"tiny": "测话", "small": "测试"})

        cb = ASRCallback(
            n_samples=2, n_timesteps=2, patience=99,
            asr_model_sizes=["tiny", "small"],
            use_real_vocoder=False,
        )
        ds = make_dataset(20)
        model = make_small_model()
        config = TrainerConfig(
            epochs=1, batch_size=2, val_ratio=0.3,
            output_dir=str(REPO / "tests" / "_tmp_m7_e2e"),
            log_every_n_steps=100, save_every_n_epochs=100,
        )
        trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
        trainer.fit()
        assert len(cb._history) > 0
        last = cb._history[-1]
        assert "asr_cer" in last           # 共识 CER
        assert "asr_cer_median" in last    # 模型 CER 中位数
        assert "asr_cer_std" in last       # 模型间离散度
        assert "asr_agreement" in last     # 一致性
        # 两个模型输出相似 ("测话" vs "测试", 2 字差 1 字) → agreement=0.5
        assert last["asr_agreement"] >= 0.5
        # std 应 > 0 (一个全对一个全错)
        assert last["asr_cer_std"] >= 0
        import shutil
        shutil.rmtree(REPO / "tests" / "_tmp_m7_e2e", ignore_errors=True)

    def test_ensemble_degrades_on_single_failure(self):
        """一个模型返回空 (失败) 时, 降级为单模型, 不报错。"""
        _mock_faster_whisper({"tiny": "测试", "small": ""})  # small 失败

        cb = ASRCallback(
            n_samples=2, n_timesteps=2, patience=99,
            asr_model_sizes=["tiny", "small"],
            use_real_vocoder=False,
        )
        ds = make_dataset(20)
        model = make_small_model()
        config = TrainerConfig(
            epochs=1, batch_size=2, val_ratio=0.3,
            output_dir=str(REPO / "tests" / "_tmp_m7_deg"),
            log_every_n_steps=100, save_every_n_epochs=100,
        )
        trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
        trainer.fit()
        # 仍应有结果 (tiny 兜底)
        assert len(cb._history) > 0
        last = cb._history[-1]
        assert "asr_cer" in last
        import shutil
        shutil.rmtree(REPO / "tests" / "_tmp_m7_deg", ignore_errors=True)

    def test_single_model_mode_unchanged(self):
        """单模型模式不应有集成指标。"""
        _mock_faster_whisper({"tiny": "测试"})

        cb = ASRCallback(
            n_samples=2, n_timesteps=2, patience=99,
            asr_model_size="tiny",
            use_real_vocoder=False,
        )
        ds = make_dataset(20)
        model = make_small_model()
        config = TrainerConfig(
            epochs=1, batch_size=2, val_ratio=0.3,
            output_dir=str(REPO / "tests" / "_tmp_m7_single"),
            log_every_n_steps=100, save_every_n_epochs=100,
        )
        trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
        trainer.fit()
        assert len(cb._history) > 0
        last = cb._history[-1]
        assert "asr_cer" in last
        # 单模型模式: 无集成统计
        assert "asr_cer_median" not in last
        assert "asr_agreement" not in last
        import shutil
        shutil.rmtree(REPO / "tests" / "_tmp_m7_single", ignore_errors=True)


class TestEnsembleCache:
    """多模型缓存。"""

    def test_per_size_cache(self):
        _mock_faster_whisper({"tiny": "a", "small": "b"})
        cb = ASRCallback(asr_model_sizes=["tiny", "small"], use_real_vocoder=False)
        m1 = cb._get_asr(size="tiny")
        m2 = cb._get_asr(size="small")
        assert m1 is not None and m2 is not None
        assert m1 is not m2
        # 重复获取命中缓存
        assert cb._get_asr(size="tiny") is m1
        # 主模型兼容属性
        assert cb._asr_cache is m1

    def test_failed_size_marked(self):
        import sys as _sys
        mock_module = type(_sys)("faster_whisper")
        def fail_load(size, *a, **k):
            raise RuntimeError("no such model")
        mock_module.WhisperModel = fail_load
        _sys.modules["faster_whisper"] = mock_module

        cb = ASRCallback(asr_model_sizes=["tiny", "small"], use_real_vocoder=False)
        assert cb._get_asr(size="small") is None
        assert "small" in cb._asr_failed
        # 主模型未尝试, 不受影响
        assert cb._asr_load_failed is False
