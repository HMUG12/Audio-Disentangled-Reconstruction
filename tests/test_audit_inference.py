"""批次66: inference 审计修复回归测试 (#7 #8 #9 #10 #11 #44)。"""
from __future__ import annotations

import builtins
import dataclasses
import logging
import wave

import numpy as np
import pytest
import torch


def _sovits_cfg():
    from adr.models.sovits import SoVITSConfig
    return SoVITSConfig(hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
                        vocab_size=607, content_dim=16, timbre_dim=16)


def _pipe(vocoder=None, **kw):
    from adr.inference.pipeline import InferConfig, InferPipeline
    from adr.models.sovits import SoVITS
    kw.setdefault("device", "cpu")
    kw.setdefault("n_timesteps", 3)
    return InferPipeline(SoVITS(_sovits_cfg()), config=InferConfig(**kw), vocoder=vocoder)


def _ref(tmp_path, sr=22050, dur=1.0):
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    path = tmp_path / "ref.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return str(path)


def _ckpt(tmp_path, **overrides):
    from adr.models.sovits import SoVITS
    cfg = _sovits_cfg()
    data = {"model_state": SoVITS(cfg).state_dict(), "model_class": "SoVITS",
            "backbone_config": dataclasses.asdict(cfg)}
    data.update(overrides)
    path = tmp_path / "ckpt.pt"
    torch.save(data, path)
    return path


# ---- #7 save_wav 原子写 ----

def test_save_wav_write_failure_keeps_original(tmp_path, monkeypatch):
    """元数据写回中途 OSError → 原 WAV 字节不变, 无残留临时文件。"""
    monkeypatch.setenv("ADR_WAV_METADATA", "1")
    pipe = _pipe()
    out = tmp_path / "out.wav"
    wav = (0.1 * np.sin(np.linspace(0, 100, 4000))).astype(np.float32)

    from adr.utils.audio import AudioData, save_audio
    save_audio(AudioData(waveform=wav, sample_rate=22050), str(out))
    original = out.read_bytes()

    real_open = builtins.open

    class _HalfWriter:
        def __init__(self, f):
            self.f = f
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self.f.close()
        def write(self, data):
            self.f.write(data[: len(data) // 2])
            raise OSError("模拟磁盘满")

    def fake_open(file, mode="r", *a, **k):
        f = real_open(file, mode, *a, **k)
        return _HalfWriter(f) if "w" in mode else f

    import adr.inference.pipeline as pl
    # save_audio 先写完整 WAV; 只拦截 save_wav 里的写回
    monkeypatch.setattr(pl, "save_audio", lambda *a, **k: None)
    monkeypatch.setattr(pl, "open", fake_open, raising=False)
    pipe.save_wav(wav, out)

    assert out.read_bytes() == original
    assert not (tmp_path / "out.wav.tmp").exists()


def test_save_wav_marks_metadata(tmp_path, monkeypatch):
    """正常路径仍写入 AI 生成标注且可读回。"""
    monkeypatch.setenv("ADR_WAV_METADATA", "1")
    import soundfile as sf
    out = tmp_path / "out.wav"
    _pipe().save_wav(np.zeros(2205, dtype=np.float32), out)
    assert b"ISFT" in out.read_bytes()
    assert sf.read(str(out))[0].shape[0] == 2205
    assert not (tmp_path / "out.wav.tmp").exists()


# ---- #8 use_placeholder_vocoder=False 不静默退化 ----

class _Vocoder:
    def __init__(self, infer_exc=None, to_exc=None):
        self.infer_exc, self.to_exc = infer_exc, to_exc
    def to(self, device):
        if self.to_exc:
            raise self.to_exc
        return self
    def eval(self):
        return self
    def infer(self, mel):
        if self.infer_exc:
            raise self.infer_exc
        return torch.zeros(1, mel.shape[-1] * 256)


def test_vocoder_infer_failure_raises(tmp_path):
    pipe = _pipe(vocoder=_Vocoder(infer_exc=RuntimeError("OOM")))
    with pytest.raises(RuntimeError, match="Vocoder 推理失败"):
        pipe.synthesize("你好", _ref(tmp_path))


def test_vocoder_placeholder_opt_in_still_works(tmp_path):
    pipe = _pipe(vocoder=_Vocoder(infer_exc=RuntimeError("OOM")),
                 use_placeholder_vocoder=True)
    assert pipe.synthesize("你好", _ref(tmp_path)).ndim == 1


def test_no_vocoder_warns(tmp_path, caplog):
    """未配置 vocoder 仍出占位 (演示路径兼容), 但必须告警。"""
    pipe = _pipe()
    with caplog.at_level(logging.WARNING, logger="adr.inference"):
        pipe.synthesize("你好", _ref(tmp_path))
    assert any("占位" in r.message for r in caplog.records)


def test_from_checkpoint_vocoder_load_failure_raises(tmp_path, monkeypatch):
    from adr.inference.pipeline import InferConfig, InferPipeline
    monkeypatch.setattr(InferPipeline, "_load_vocoder", staticmethod(lambda p: None))
    with pytest.raises(RuntimeError, match="Vocoder 加载失败"):
        InferPipeline.from_checkpoint(_ckpt(tmp_path), config=InferConfig(device="cpu"),
                                      vocoder_path=str(tmp_path / "bigvgan"))
    pipe = InferPipeline.from_checkpoint(
        _ckpt(tmp_path), config=InferConfig(device="cpu", use_placeholder_vocoder=True),
        vocoder_path=str(tmp_path / "bigvgan"))
    assert pipe.vocoder is None


# ---- #9 vocoder 迁移失败 ----

def test_vocoder_to_device_failure_raises():
    with pytest.raises(RuntimeError, match="Vocoder 迁移"):
        _pipe(vocoder=_Vocoder(to_exc=RuntimeError("CUDA OOM")))


# ---- #10 model_class / backbone_config ----

def test_unknown_model_class_raises(tmp_path):
    from adr.inference.pipeline import InferPipeline
    with pytest.raises(ValueError, match="未注册"):
        InferPipeline.from_checkpoint(_ckpt(tmp_path, model_class="RenamedNet"))


def test_bad_backbone_config_raises(tmp_path):
    from adr.inference.pipeline import InferPipeline
    cfg = dataclasses.asdict(_sovits_cfg())
    cfg["no_such_field"] = 1
    with pytest.raises(ValueError, match="backbone_config"):
        InferPipeline.from_checkpoint(_ckpt(tmp_path, backbone_config=cfg))


# ---- #11 G2P 失败留痕 ----

def test_g2p_failure_logged(monkeypatch, caplog):
    import adr.data.phoneme_dict as pdm

    def boom(*a, **k):
        raise ImportError("pypinyin missing")
    monkeypatch.setattr(pdm, "encode_phonemes", boom)
    pipe = _pipe()
    with caplog.at_level(logging.WARNING, logger="adr.inference"):
        ids = pipe.text_to_phoneme_ids("你好")
    assert ids  # 仍退化为字符级
    assert any("G2P" in r.message for r in caplog.records)


# ---- #44 输入上限 ----

def test_text_length_cap(monkeypatch):
    monkeypatch.setenv("ADR_INFER_MAX_TEXT_CHARS", "5")
    pipe = _pipe()
    with pytest.raises(ValueError, match="Text too long"):
        pipe.text_to_phoneme_ids("你好世界你好世界")
    assert pipe.text_to_phoneme_ids("你好")


def test_default_text_cap_generous():
    assert len(_pipe().text_to_phoneme_ids("你好" * 500)) > 0


def test_n_timesteps_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("ADR_INFER_MAX_TIMESTEPS", "4")
    pipe = _pipe()
    with pytest.raises(ValueError, match="n_timesteps"):
        pipe.synthesize("你好", _ref(tmp_path), n_timesteps=5)
    with pytest.raises(ValueError, match="n_timesteps"):
        pipe.synthesize("你好", _ref(tmp_path), n_timesteps=-1)
    assert pipe.synthesize("你好", _ref(tmp_path), n_timesteps=3).ndim == 1
