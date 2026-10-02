"""测试 ASRCallback (M4 - ASR 早停) + WER/CER 计算函数。

测试覆盖:
1. WER/CER 纯 Python 计算 (无需 ASR)
2. compute_text_similarity 自动检测中英文
3. ASRCallback 构造 (不依赖 faster-whisper)
4. ASRCallback 在没 val_loader 时安全 skip
5. ASRCallback 触发早停
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
from adr.training.callbacks import (
    ASRCallback,
    _edit_distance,
    compute_cer,
    compute_wer,
    compute_text_similarity,
)
from adr.training.dataset import VoiceCloneDataset
from adr.training.trainer import Trainer, TrainerConfig


# ============================================================
# WER/CER 纯计算测试
# ============================================================
class TestEditDistance:
    """测试 Levenshtein 编辑距离。"""

    def test_identical(self):
        assert _edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0

    def test_empty_ref(self):
        assert _edit_distance([], ["a", "b"]) == 2

    def test_empty_hyp(self):
        assert _edit_distance(["a", "b"], []) == 2

    def test_substitution(self):
        assert _edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1

    def test_insertion(self):
        assert _edit_distance(["a", "c"], ["a", "b", "c"]) == 1

    def test_deletion(self):
        assert _edit_distance(["a", "b", "c"], ["a", "c"]) == 1

    def test_complex(self):
        # kitten -> sitting: k->s, e->i, insert g = 3
        assert _edit_distance(
            list("kitten"), list("sitting")
        ) == 3


class TestCER:
    """测试字符错误率 (CER)。"""

    def test_perfect_match(self):
        assert compute_cer("你好世界", "你好世界") == 0.0

    def test_complete_mismatch(self):
        cer = compute_cer("你好世界", "完全不同的文字")
        assert cer > 0.5

    def test_one_char_error(self):
        # "你好世界" (4 chars), "你好世x" (1 char wrong)
        cer = compute_cer("你好世界", "你好世x")
        assert abs(cer - 0.25) < 0.01

    def test_with_spaces(self):
        # 空格被忽略
        cer = compute_cer("你好 世界", "你好世界")
        assert cer == 0.0

    def test_with_punctuation(self):
        # 标点被忽略
        cer = compute_cer("你好,世界。", "你好世界")
        assert cer == 0.0

    def test_empty_reference(self):
        assert compute_cer("", "") == 0.0
        assert compute_cer("", "abc") == 1.0

    def test_all_wrong(self):
        cer = compute_cer("你好", "世界")
        assert cer == 1.0


class TestWER:
    """测试词错误率 (WER)。"""

    def test_perfect_match(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_one_word_wrong(self):
        # "hello world" (2 words), "hello there" (1 wrong) = 0.5
        wer = compute_wer("hello world", "hello there")
        assert abs(wer - 0.5) < 0.01

    def test_insertion(self):
        # "hello world" (2), "hello big world" (3) = 1/2 = 0.5
        wer = compute_wer("hello world", "hello big world")
        assert abs(wer - 0.5) < 0.01

    def test_deletion(self):
        # ref 3 words -> hyp 2 words, 1 deletion = 1/3 ≈ 0.333
        wer = compute_wer("hello big world", "hello world")
        assert abs(wer - 0.333) < 0.01

    def test_empty(self):
        assert compute_wer("", "") == 0.0
        assert compute_wer("", "abc") == 1.0


class TestComputeTextSimilarity:
    """测试自动语言检测。"""

    def test_chinese_uses_cer(self):
        result = compute_text_similarity("你好世界", "你好世x")
        assert result["mode"] == "cer"
        assert result["rate"] > 0

    def test_english_uses_wer(self):
        result = compute_text_similarity("hello world", "hello there")
        assert result["mode"] == "wer"
        assert result["rate"] > 0

    def test_japanese_uses_cer(self):
        result = compute_text_similarity("こんにちは", "こんばんは")
        assert result["mode"] == "cer"

    def test_explicit_lang(self):
        result = compute_text_similarity("hello", "helo", lang="en")
        assert result["mode"] == "wer"

    def test_return_structure(self):
        result = compute_text_similarity("hello world", "hello there")
        assert "rate" in result
        assert "mode" in result
        assert "n_ref" in result
        assert "dist" in result
        assert result["n_ref"] == 2
        assert result["dist"] == 1


# ============================================================
# ASRCallback 测试 (不依赖 faster-whisper)
# ============================================================
def make_dataset(n: int = 20, with_text: bool = True):
    """构造测试 dataset, 文本为真实中文。"""
    samples = []
    texts = [
        "你好世界",
        "这是一个测试",
        "今天天气真不错",
        "欢迎使用 ADR 框架",
        "声纹克隆很简单",
        "再试一次吧",
        "语音合成很有趣",
        "深度学习改变世界",
        "明天会更好",
        "感谢你的支持",
    ]
    for i in range(n):
        text = texts[i % len(texts)] if with_text else ""
        s = TrainSample(
            sample_id=f"utt_{i:03d}",
            phonemes=["zh", "ong1", "g_uan1"] * 7,
            text=text,
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


class TestASRCallbackConstruction:
    """测试 ASRCallback 构造, 不触发 ASR 加载。"""

    def test_default_construction(self):
        """默认参数应能成功构造。"""
        cb = ASRCallback()
        assert cb.n_samples == 5
        assert cb.n_timesteps == 10
        assert cb.asr_model_size == "tiny"
        assert cb.asr_language == "zh"
        assert cb.metric_name == "asr_cer"
        assert cb.mode == "min"
        assert cb.patience == 2
        assert cb.best_value is None
        assert cb.counter == 0
        assert cb.should_stop is False

    def test_custom_construction(self):
        """自定义参数应被正确保存。"""
        cb = ASRCallback(
            n_samples=10,
            n_timesteps=20,
            asr_model_size="base",
            asr_language="en",
            metric_name="asr_wer",
            patience=3,
            min_delta=0.05,
        )
        assert cb.n_samples == 10
        assert cb.n_timesteps == 20
        assert cb.asr_model_size == "base"
        assert cb.asr_language == "en"
        assert cb.metric_name == "asr_wer"
        assert cb.patience == 3
        assert cb.min_delta == 0.05

    def test_legacy_placeholder_param(self):
        """旧参数 use_placeholder_vocoder 应被兼容。"""
        cb = ASRCallback(use_placeholder_vocoder=True)
        assert cb.use_real_vocoder is False

        cb2 = ASRCallback(use_placeholder_vocoder=False)
        assert cb2.use_real_vocoder is True

    def test_legacy_param_takes_precedence_when_both_passed(self):
        """当同时传 use_real_vocoder 和 use_placeholder_vocoder, 旧参数生效 (向后兼容)。"""
        # 仅传 use_placeholder_vocoder
        cb = ASRCallback(use_placeholder_vocoder=True, use_real_vocoder=True)
        # 旧参数语义生效 → use_real_vocoder=False
        assert cb.use_real_vocoder is False

        cb2 = ASRCallback(use_placeholder_vocoder=False, use_real_vocoder=False)
        assert cb2.use_real_vocoder is True


class TestASRCallbackRunWithoutASR:
    """测试 ASRCallback 在没有 faster-whisper 时的安全行为。"""

    def test_skips_when_asr_unavailable(self, monkeypatch):
        """ASR 不可用时, 回调应 skip 而不抛异常。"""
        # 模拟 faster_whisper 导入失败
        import sys as _sys

        # 替换 ASR 加载为失败
        cb = ASRCallback(
            n_samples=2, n_timesteps=2, patience=2,
            asr_model_size="nonexistent_model_for_test",
        )
        # 强制 _get_asr 失败
        cb._asr_load_failed = True

        ds = make_dataset(20)
        model = make_small_model()
        config = TrainerConfig(
            epochs=2, batch_size=2, val_ratio=0.3,
            output_dir=str(REPO / "tests" / "_tmp_asr_skip"),
            log_every_n_steps=100, save_every_n_epochs=100,
        )
        trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
        trainer.fit()
        # 因为 ASR 加载失败, history 应为空
        assert len(cb._history) == 0
        # 不应 stop
        assert cb.should_stop is False
        # 清理
        import shutil
        shutil.rmtree(REPO / "tests" / "_tmp_asr_skip", ignore_errors=True)

    def test_skips_without_val(self):
        """没有 val_loader 时应 skip。"""
        cb = ASRCallback(
            n_samples=2, n_timesteps=2, patience=2,
            asr_model_size="nonexistent",
        )
        # 模拟 ASR 加载失败
        cb._asr_load_failed = True

        ds = make_dataset(20)
        model = make_small_model()
        config = TrainerConfig(
            epochs=2, batch_size=2, val_ratio=0.0,  # 没 val
            output_dir=str(REPO / "tests" / "_tmp_asr_noval"),
            log_every_n_steps=100, save_every_n_epochs=100,
        )
        trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
        trainer.fit()
        assert len(cb._history) == 0
        assert cb.should_stop is False
        # 清理
        import shutil
        shutil.rmtree(REPO / "tests" / "_tmp_asr_noval", ignore_errors=True)


class TestASRCallbackWithMockASR:
    """用 mock ASR 测试完整流程。"""

    def test_mock_asr_improves_over_time(self, monkeypatch):
        """用 mock ASR 模拟 CER 随 epoch 改善, 验证 callback 能收集样本。"""
        # 创建 mock ASR, 每次返回不同长度错文本
        class MockASR:
            def transcribe(self, path, language=None, beam_size=5, vad_filter=True):
                MockASR.call_count = getattr(MockASR, "call_count", 0) + 1
                # 第一次: 大量错字; 后续: 越来越短 (即更接近 ref)
                n_wrong = max(1, 5 - MockASR.call_count // 2)
                return (
                    [type("Seg", (), {"text": "错" * n_wrong})()],
                    None,
                )

        import sys as _sys
        mock_module = type(_sys)("faster_whisper")
        mock_module.WhisperModel = lambda *a, **k: MockASR()
        _sys.modules["faster_whisper"] = mock_module

        cb = ASRCallback(
            n_samples=2, n_timesteps=2, patience=99,  # 不早停
            asr_model_size="tiny",
            use_real_vocoder=False,
        )

        ds = make_dataset(20)
        model = make_small_model()
        config = TrainerConfig(
            epochs=2, batch_size=2, val_ratio=0.3,
            output_dir=str(REPO / "tests" / "_tmp_asr_mock"),
            log_every_n_steps=100, save_every_n_epochs=100,
        )
        trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
        trainer.fit()
        # 验证 callback 跑通, _history 有数据
        assert len(cb._history) > 0
        # 每个 history 都有 asr_cer 指标
        for entry in cb._history:
            assert "asr_cer" in entry or "asr_wer" in entry
        # 清理
        import shutil
        shutil.rmtree(REPO / "tests" / "_tmp_asr_mock", ignore_errors=True)


class TestASRCallbackEarlyStop:
    """测试早停触发。"""

    def test_early_stop_when_no_improvement(self):
        """当 CER 在 patience 个 epoch 内不变, 应触发早停。"""
        # Mock 一个永远返回相同文本的 ASR (CER 永远 = 1.0)
        class MockASR:
            call_count = 0
            def transcribe(self, path, language=None, beam_size=5, vad_filter=True):
                MockASR.call_count += 1
                return (
                    [type("Seg", (), {"text": "错" * 5})()],
                    None,
                )

        import sys as _sys
        mock_module = type(_sys)("faster_whisper")
        mock_module.WhisperModel = lambda *a, **k: MockASR()
        _sys.modules["faster_whisper"] = mock_module

        cb = ASRCallback(
            n_samples=2, n_timesteps=2, patience=1, min_delta=0.001,
            asr_model_size="tiny",
            use_real_vocoder=False,
        )

        ds = make_dataset(20)
        model = make_small_model()
        config = TrainerConfig(
            epochs=5, batch_size=2, val_ratio=0.3,
            output_dir=str(REPO / "tests" / "_tmp_asr_stop"),
            log_every_n_steps=100, save_every_n_epochs=100,
        )
        trainer = Trainer(model=model, train_data=ds, config=config, callbacks=[cb])
        trainer.fit()
        # patience=1 + CER 不变, 应在 epoch 2 后 stop
        assert cb.should_stop is True
        # 清理
        import shutil
        shutil.rmtree(REPO / "tests" / "_tmp_asr_stop", ignore_errors=True)


class TestASRCudaFallback:
    """测试 CUDA 转录失败时回退 CPU (cublas 缺失场景)。"""

    def test_runtime_error_triggers_cpu_fallback(self):
        """首次 transcribe 抛 RuntimeError 时, 应重载 CPU ASR 并重试成功。"""
        calls = {"n": 0}

        class CudaFailASR:
            def transcribe(self, path, language=None, beam_size=5, vad_filter=True):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("Library cublas64_12.dll is not found")
                # CPU 重载后成功
                return ([type("Seg", (), {"text": "你好"})()], None)

        import sys as _sys
        mock_module = type(_sys)("faster_whisper")
        mock_module.WhisperModel = lambda *a, **k: CudaFailASR()
        _sys.modules["faster_whisper"] = mock_module

        cb = ASRCallback(use_real_vocoder=False, asr_model_size="tiny")
        # 模拟真实临时 wav 文件路径
        import tempfile, soundfile as sf
        wav = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, 16000))).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, wav, 16000)
        try:
            # _get_asr 首次会返回 CudaFailASR (伪装 cuda)
            cb._asr_cache = CudaFailASR()
            text = cb._run_transcribe(cb._asr_cache, tmp)
            assert text == "你好"
            assert calls["n"] == 2  # 第 1 次失败, 回退后第 2 次成功
            assert cb._asr_cpu_fallback is True
        finally:
            import os
            os.unlink(tmp)

    def test_fallback_only_once(self):
        """回退只发生一次, 第二次失败不再重载。"""
        class AlwaysFailASR:
            def transcribe(self, path, language=None, beam_size=5, vad_filter=True):
                raise RuntimeError("still broken")

        import sys as _sys
        mock_module = type(_sys)("faster_whisper")
        mock_module.WhisperModel = lambda *a, **k: AlwaysFailASR()
        _sys.modules["faster_whisper"] = mock_module

        cb = ASRCallback(use_real_vocoder=False, asr_model_size="tiny")
        cb._asr_cache = AlwaysFailASR()
        import tempfile, soundfile as sf
        wav = np.zeros(16000, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, wav, 16000)
        try:
            import pytest as _pytest
            with _pytest.raises(RuntimeError):
                cb._run_transcribe(cb._asr_cache, tmp)
            assert cb._asr_cpu_fallback is True
            # 第二次: 已回退过, _reload_asr_cpu 返回 None → 直接 raise
            with _pytest.raises(RuntimeError):
                cb._run_transcribe(cb._asr_cache, tmp)
        finally:
            import os
            os.unlink(tmp)
