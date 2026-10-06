"""测试数据流水线 (M1 Day 2)。

使用合成音频(无网络依赖) 验证流水线结构与 API。
"""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest


def make_synthetic_wav(
    path: str,
    duration_sec: float = 3.0,
    sample_rate: int = 24000,
    freq: float = 440.0,
) -> str:
    """生成合成 wav 文件 (用于测试)。"""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    # 模拟语音:基频 + 共振峰
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    audio += 0.1 * np.sin(2 * np.pi * freq * 2 * t)
    audio += 0.05 * np.random.randn(len(t)).astype(np.float32) * 0.1
    audio = (audio * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return path


def test_audio_utils():
    """测试音频 I/O。"""
    from adr.utils import load_audio, save_audio, compute_mel, AudioData

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        make_synthetic_wav(tmp, duration_sec=2.0)
        audio = load_audio(tmp, sample_rate=24000)
        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 24000
        assert abs(audio.duration - 2.0) < 0.1
        assert audio.num_channels == 1
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_compute_mel_shape():
    """测试 mel 频谱计算。"""
    from adr.utils import compute_mel

    sr = 24000
    duration = 2.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    mel = compute_mel(audio, sr, n_mels=80, hop_length=256)

    assert mel.shape[0] == 80
    expected_frames = int(sr * duration / 256) + 1
    assert abs(mel.shape[1] - expected_frames) < 10


def test_slice_silence_detection():
    """测试静音检测。"""
    from adr.data.slice import detect_silence_segments
    from adr.utils.audio import AudioData

    sr = 24000
    # 1s 静音 + 2s 音频 + 1s 静音 + 2s 音频
    audio = np.zeros(sr * 6, dtype=np.float32)
    audio[sr : sr * 3] = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, sr * 2))

    segs = detect_silence_segments(audio, sr, threshold_db=-30, min_silence_sec=0.3)
    assert len(segs) >= 1, f"Should detect at least 1 silence, got {len(segs)}"


def test_slice_audio():
    """测试音频切片。"""
    from adr.data.slice import SliceConfig, slice_audio
    from adr.utils.audio import AudioData

    sr = 24000
    # 10 秒音频,有静音
    audio = np.zeros(sr * 10, dtype=np.float32)
    for i in range(0, 10, 2):
        s, e = i * sr, (i + 1) * sr
        audio[s:e] = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))

    data = AudioData(audio, sr)
    slices = slice_audio(data, SliceConfig(min_sec=2.0, max_sec=4.0))
    assert len(slices) >= 2
    for s in slices:
        assert 2.0 <= s.duration <= 5.0


def test_g2p_chinese():
    """测试中文 G2P。"""
    from adr.data.g2p import G2P, G2PConfig, G2P

    g2p = G2P(G2PConfig(backend="pypinyin", with_tone=True))
    result = g2p("你好世界")
    assert len(result) > 0
    assert all(isinstance(p, str) for p in result)


def test_g2p_char_fallback():
    """测试字符级 fallback。"""
    from adr.data.g2p import G2P, G2PConfig

    g2p = G2P(G2PConfig(backend="char"))
    result = g2p("hello")
    assert result == ["h", "e", "l", "l", "o"]


def test_g2p_empty_text():
    """测试空文本。"""
    from adr.data.g2p import G2P

    g2p = G2P()
    assert g2p("") == []
    assert g2p("   ") == []


def test_g2p_language_detection():
    """测试语言检测。"""
    from adr.data.g2p import G2P

    assert G2P.is_chinese("你好") is True
    assert G2P.is_chinese("hello") is False
    assert G2P.is_chinese("你好呀") is True  # 3/3 中文
    assert G2P.is_chinese("你好呀hello") is False  # 3/8 < 50%
    assert G2P.is_chinese("你好呀hello world") is False  # 3/13 < 50%
    assert G2P.is_chinese("啊啊啊hello") is False  # 3/8 < 50%
    assert G2P.is_chinese("中国话") is True
    assert G2P.detect_language("你好世界") == "zh"
    assert G2P.detect_language("hello world") == "en"


def test_f0_pyin():
    """测试 F0 提取 (pyin)。"""
    from adr.data.f0 import F0Extractor, F0Config

    sr = 24000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz 纯音

    extractor = F0Extractor(F0Config(sr=sr))
    f0 = extractor(audio, sample_rate=sr)

    assert f0.shape[0] > 0
    # 440Hz 附近
    voiced = f0[f0 > 0]
    if len(voiced) > 0:
        assert 400 < voiced.mean() < 480


def test_f0_statistics():
    """测试 F0 统计。"""
    from adr.data.f0 import F0Extractor

    extractor = F0Extractor()
    stats = extractor.f0_statistics(np.array([100, 200, 300, 0, 0]))
    assert stats["voiced_ratio"] == 0.6
    assert stats["mean"] == 200
    assert stats["min"] == 100
    assert stats["max"] == 300


def test_pipeline_with_synthetic_audio():
    """端到端: 合成 wav → pipeline。"""
    from adr.data.pipeline import DataPipeline, PipelineConfig
    from adr.data.slice import SliceConfig
    from adr.data.asr import ASRConfig

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # 8 秒合成音频 (足够切成 2-3 个切片)
            make_synthetic_wav(tmp_wav, duration_sec=8.0)

            config = PipelineConfig(
                enable_separation=False,   # 跳过 UVR5
                enable_slicing=True,
                enable_asr=False,          # 跳过 ASR (避免下载模型)
                enable_g2p=True,
                enable_f0=False,           # 跳过 F0 (pyin 慢)
                enable_mel=False,
                slice=SliceConfig(
                    min_sec=2.0, max_sec=4.0,
                    silence_threshold_db=-30,
                    target_sr=24000,
                ),
                output_dir=tmp_dir,
            )
            pipeline = DataPipeline(config)
            result = pipeline.run(tmp_wav)

            # 不强求有 G2P 结果 (无文本),但 pipeline 本身要能跑通
            assert result.metadata["input"] == tmp_wav
        finally:
            Path(tmp_wav).unlink(missing_ok=True)


def test_train_sample_save_load():
    """测试 TrainSample 持久化。"""
    from adr.data.pipeline import TrainSample

    sample = TrainSample(
        sample_id="test_001",
        waveform=np.random.randn(24000).astype(np.float32),
        sample_rate=24000,
        text="你好",
        phonemes=["ni3", "hao3"],
        f0=np.zeros(100, dtype=np.float32),
    )

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp = f.name
    try:
        sample.save(tmp)
        loaded = TrainSample.load(tmp)
        assert loaded.sample_id == "test_001"
        assert loaded.text == "你好"
        assert loaded.phonemes == ["ni3", "hao3"]
        assert loaded.sample_rate == 24000
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_asr_lazy_load():
    """测试 ASR 懒加载 (不实际加载模型)。"""
    from adr.data.asr import ASR, ASRConfig

    asr = ASR(ASRConfig(model_size="tiny"))
    # 不调用 transcribe,只检查构造
    assert asr._model is None
    assert asr.config.model_size == "tiny"


def test_asr_real_inference_opt_in(request):
    """ASR 真实推理 (Day 4 验证)。

    默认跳过,需要显式 --run-asr 才执行 (避免 CI 长时间下载模型)。
    本测试验证 faster-whisper 能正确加载并对合成音频转录 (即使合成音是噪声,
    也应能返回一些非空结果或空结果,只要 API 不报错即可)。
    """
    if not request.config.getoption("--run-asr", default=False):
        pytest.skip("Pass --run-asr to enable (downloads ~75MB model)")

    from adr.data.asr import ASR, ASRConfig

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        # 3 秒合成 "说话声": 多频率叠加模拟人声
        make_synthetic_wav(tmp, duration_sec=3.0, freq=200.0)
        asr = ASR(ASRConfig(model_size="tiny", device="cpu", compute_type="int8"))
        result = asr.transcribe(tmp)
        # API 行为: 返回 ASRResult, 字段存在
        assert hasattr(result, "text")
        assert hasattr(result, "language")
        assert hasattr(result, "confidence")
        print(f"  ASR result: text='{result.text[:50] if result.text else ''}' "
              f"lang={result.language} conf={result.confidence:.2f}")
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_separator_vad_fallback():
    """测试 Separator VAD fallback。"""
    from adr.data.separate import Separator, SeparateConfig
    from adr.utils.audio import AudioData

    sep = Separator(SeparateConfig(method="vad"))
    audio = AudioData(np.random.randn(24000).astype(np.float32), 24000)
    vocals, instr = sep(audio)
    assert vocals is not None
    assert instr is None  # VAD fallback 不返回 instrumental


# ---------------------------------------------------------------------------
# 批次41b: F0/Mel 单样本失败跳过 + sample_id 唯一性 (全 mock, 不加载模型)
# ---------------------------------------------------------------------------


def _make_mock_pipeline(tmp_dir: str):
    """构造全 mock 组件的 pipeline (隔离模型下载, 专测编排逻辑)。

    ASR 恒返回固定文本, G2P 恒返回固定音素; _f0 由各测试自行注入。
    """
    import types

    from adr.data.pipeline import DataPipeline, PipelineConfig
    from adr.data.slice import SliceConfig

    config = PipelineConfig(
        enable_separation=False,
        enable_slicing=True,
        enable_asr=True,
        enable_g2p=True,
        enable_f0=True,
        enable_mel=False,
        slice=SliceConfig(min_sec=2.0, max_sec=4.0, target_sr=24000),
        output_dir=str(tmp_dir),
    )
    pipeline = DataPipeline(config)
    pipeline._ensure_components()
    pipeline._asr = types.SimpleNamespace(
        transcribe=lambda w, sample_rate=None: types.SimpleNamespace(text="你好世界")
    )
    pipeline._g2p = lambda text: ["ni3", "hao3"]
    return pipeline


def test_pipeline_f0_failure_skips_sample(tmp_path):
    """F0 提取全部失败 → 流水线不中断, errors 记录且无样本入库 (批次41b)。"""
    pipeline = _make_mock_pipeline(str(tmp_path))

    def _boom(waveform, sample_rate=None):
        raise RuntimeError("f0 boom")

    pipeline._f0 = _boom

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        make_synthetic_wav(wav_path, duration_sec=8.0)
        result = pipeline.run(wav_path, output_dir=tmp_path / "out1")
        assert result.samples == []          # 失败切片不带空 f0 入库
        assert result.errors                 # 有错误记录
        assert all("F0/Mel" in e for e in result.errors)
    finally:
        Path(wav_path).unlink(missing_ok=True)


def test_pipeline_f0_partial_failure(tmp_path):
    """F0 部分失败: 失败切片跳过, 其余切片正常入库 (批次41b)。"""
    pipeline = _make_mock_pipeline(str(tmp_path))
    state = {"n": 0}

    def _flaky(waveform, sample_rate=None):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("f0 boom")
        return np.zeros(10, dtype=np.float32)

    pipeline._f0 = _flaky

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        make_synthetic_wav(wav_path, duration_sec=10.0)
        result = pipeline.run(wav_path, output_dir=tmp_path / "out2")
        assert len(result.errors) == 1
        assert "F0/Mel" in result.errors[0]
        assert len(result.samples) >= 1      # 其余切片照常入库
    finally:
        Path(wav_path).unlink(missing_ok=True)


def test_pipeline_sample_id_unique_across_paths(tmp_path):
    """不同目录同名 wav → sample_id 不碰撞 (批次41b 路径哈希)。"""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    wav_a = str(dir_a / "voice.wav")
    wav_b = str(dir_b / "voice.wav")
    make_synthetic_wav(wav_a, duration_sec=10.0)
    make_synthetic_wav(wav_b, duration_sec=10.0)

    pipeline = _make_mock_pipeline(str(tmp_path))
    r_a = pipeline.run(wav_a, output_dir=tmp_path / "out_a")
    r_b = pipeline.run(wav_b, output_dir=tmp_path / "out_b")

    ids_a = {s.sample_id for s in r_a.samples}
    ids_b = {s.sample_id for s in r_b.samples}
    assert ids_a and ids_b                   # 两条流水线都有产出
    assert not (ids_a & ids_b)               # 同名文件不再互相覆盖
