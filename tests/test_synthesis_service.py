"""批次34: SynthesisService 单一合成服务层测试 (RecordingEngine, 不加载真模型)。

覆盖:
- load_profile 三态 (非法名/不存在/成功) + 错误消息逐字 (wire 兼容)
- profile_fill 显式值优先 + falsy/is None 双语义 (镜像 v2 _resolve_profile)
- validate_params 必填/白名单 + media_type 归一化
- stream_chunks/synthesize_once split_method 兜底 (cut3/cut1) + 全参数透传
- stream_bytes wav 首块 44B 头 + raw 裸 PCM (GSV api_v2 wire 契约)
- frame_bytes v3 逐帧完整 WAV 契约
"""
import json
import struct

import numpy as np
import pytest

from adr.core.exceptions import (
    ProfileInvalidError,
    ProfileNotFoundError,
    SynthesisParamsError,
)
from adr.services import SynthesisService
from adr.services.synthesis import MEDIA_TYPES


class RecordingEngine:
    """记录调用参数的假引擎 (签名对齐 GSVEngine); 每块 16 样本便于字节断言。"""

    def __init__(self, sr=32000):
        self.sr = sr
        self.synth_calls = []
        self.stream_calls = []

    def synthesize(self, text, ref_audio, prompt_text="", text_lang="zh",
                   prompt_lang="zh", speed_factor=1.0, seed=-1,
                   t2s_weights=None, vits_weights=None, split_method="cut1",
                   top_k=15, top_p=1.0, temperature=1.0,
                   fragment_interval=None):
        self.synth_calls.append(dict(
            text=text, ref_audio=ref_audio, prompt_text=prompt_text,
            text_lang=text_lang, prompt_lang=prompt_lang,
            speed_factor=speed_factor, seed=seed, t2s_weights=t2s_weights,
            vits_weights=vits_weights, split_method=split_method,
            top_k=top_k, top_p=top_p, temperature=temperature,
            fragment_interval=fragment_interval))
        return np.zeros(0, np.float32), self.sr

    def synthesize_stream(self, text, ref_audio, prompt_text="", text_lang="zh",
                          prompt_lang="zh", t2s_weights=None, vits_weights=None,
                          split_method="cut3", head_seed=-1,
                          top_k=15, top_p=1.0, temperature=1.0,
                          speed_factor=1.0, fragment_interval=None):
        self.stream_calls.append(dict(
            text=text, ref_audio=ref_audio, prompt_text=prompt_text,
            text_lang=text_lang, prompt_lang=prompt_lang,
            t2s_weights=t2s_weights, vits_weights=vits_weights,
            split_method=split_method, head_seed=head_seed,
            top_k=top_k, top_p=top_p, temperature=temperature,
            speed_factor=speed_factor, fragment_interval=fragment_interval))
        for _ in range(2):
            yield np.full(16, 0.1, np.float32), self.sr


@pytest.fixture
def voice_dir(tmp_path, monkeypatch):
    """假档案库: demo 档案 (ref.wav + meta.json), monkeypatch VOICES_DIR。"""
    from adr.models import voice_library
    vd = tmp_path / "voices"
    (vd / "demo").mkdir(parents=True)
    (vd / "demo" / "ref.wav").write_bytes(b"RIFF0000WAVEfmt ")  # 内容不校验
    (vd / "demo" / "meta.json").write_text(json.dumps({
        "name": "demo", "prompt_text": "参考文本",
        "t2s_weights": "GPT_weights_v2/demo.ckpt",
        "vits_weights": "SoVITS_weights_v2/demo.pth",
        "rvc_weights": None, "rvc_index": None, "style": "平静",
        "created_at": "2026-01-01 00:00:00"}), encoding="utf-8")
    monkeypatch.setattr(voice_library, "VOICES_DIR", vd)
    return vd


# ─── load_profile ───

def test_load_profile_invalid_name():
    with pytest.raises(ProfileInvalidError) as ei:
        SynthesisService.load_profile("../escape")
    assert str(ei.value) == "invalid profile name: ../escape"


def test_load_profile_unknown(voice_dir):
    with pytest.raises(ProfileNotFoundError) as ei:
        SynthesisService.load_profile("nope")
    assert str(ei.value) == "unknown profile: nope"


def test_load_profile_ok(voice_dir):
    meta = SynthesisService.load_profile("demo")
    assert meta["name"] == "demo"
    assert meta["ref_audio"].endswith("ref.wav")


# ─── profile_fill ───

def test_profile_fill_request_priority(voice_dir):
    meta = SynthesisService.load_profile("demo")
    filled = SynthesisService.profile_fill(
        meta, ref_audio_path="custom.wav", prompt_text="自定义",
        t2s_weights="g.ckpt", vits_weights="v.pth")
    assert filled == {"ref_audio_path": "custom.wav", "prompt_text": "自定义",
                      "t2s_weights": "g.ckpt", "vits_weights": "v.pth"}


def test_profile_fill_fallback_semantics(voice_dir):
    # falsy 判定 (ref/prompt): 显式空串/None 也兜底
    # is None 判定 (weights): 显式空串不兜底, 仅 None 兜底 — 镜像 v2 原语义
    meta = SynthesisService.load_profile("demo")
    filled = SynthesisService.profile_fill(
        meta, ref_audio_path="", prompt_text=None,
        t2s_weights=None, vits_weights="")
    assert filled["ref_audio_path"] == meta["ref_audio"]
    assert filled["prompt_text"] == "参考文本"
    assert filled["t2s_weights"] == "GPT_weights_v2/demo.ckpt"  # None → 档案值
    assert filled["vits_weights"] == ""  # 显式空串不兜底


# ─── validate_params ───

def test_validate_params_missing_text():
    with pytest.raises(SynthesisParamsError) as ei:
        SynthesisService.validate_params("", "a.wav")
    assert str(ei.value) == "text is required"


def test_validate_params_missing_ref():
    with pytest.raises(SynthesisParamsError) as ei:
        SynthesisService.validate_params("你好", "")
    assert str(ei.value) == ("ref_audio_path is required "
                             "(or pass profile / set ADR_TTS_DEFAULT_PROFILE)")


def test_validate_params_bad_media_type():
    with pytest.raises(SynthesisParamsError) as ei:
        SynthesisService.validate_params("你好", "a.wav", "flac")
    assert str(ei.value) == ("unsupported media_type: flac, "
                             "must be one of wav/mp3/raw/ogg/aac")


def test_validate_params_media_type_normalized():
    assert SynthesisService.validate_params("你好", "a.wav", "WAV") == "wav"
    assert SynthesisService.validate_params("你好", "a.wav", None) == "wav"


# ─── 引擎调用 ───

def test_stream_chunks_fallback_and_passthrough():
    eng = RecordingEngine()
    chunks = list(SynthesisService.stream_chunks(
        eng, "你好", "a.wav", prompt_text="ref", text_lang="zh",
        t2s_weights="g.ckpt", vits_weights="v.pth", split_method=None,
        head_seed=7, top_k=20, top_p=0.9, temperature=0.8,
        speed_factor=1.2, fragment_interval=0.5))
    assert len(chunks) == 2
    assert all(sr == 32000 for _, sr in chunks)
    call = eng.stream_calls[0]
    assert call["split_method"] == "cut3"  # 流式 fallback 单点
    assert call["prompt_text"] == "ref" and call["text_lang"] == "zh"
    assert call["t2s_weights"] == "g.ckpt" and call["vits_weights"] == "v.pth"
    assert call["head_seed"] == 7
    assert call["top_k"] == 20 and call["top_p"] == 0.9
    assert call["temperature"] == 0.8 and call["speed_factor"] == 1.2
    assert call["fragment_interval"] == 0.5


def test_synthesize_once_fallback_cut1():
    eng = RecordingEngine()
    SynthesisService.synthesize_once(eng, "你好", "a.wav", seed=3)
    call = eng.synth_calls[0]
    assert call["split_method"] == "cut1"  # 非流式 fallback 单点
    assert call["seed"] == 3


# ─── 线格式 ───

def test_stream_bytes_wav_first_block_header():
    eng = RecordingEngine()
    blocks = list(SynthesisService.stream_bytes(
        eng, "你好", "a.wav", media_type="wav"))
    # 首块 44B WAV 头 + 2 × 32B 裸 PCM (16 样本 × 2B s16le)
    assert len(blocks) == 3
    assert blocks[0][:4] == b"RIFF"
    assert len(blocks[0]) == 44
    assert struct.unpack_from("<I", blocks[0], 24)[0] == 32000
    assert all(len(b) == 32 for b in blocks[1:])


def test_stream_bytes_raw_no_header():
    eng = RecordingEngine()
    blocks = list(SynthesisService.stream_bytes(
        eng, "你好", "a.wav", media_type="raw"))
    assert len(blocks) == 2
    assert all(len(b) == 32 for b in blocks)
    assert all(b[:4] != b"RIFF" for b in blocks)


def test_frame_bytes_full_wav():
    chunk = np.full(16, 0.1, np.float32)
    frame = SynthesisService.frame_bytes(chunk, 32000)
    assert len(frame) == 44 + 32
    assert frame[:4] == b"RIFF"
    assert struct.unpack_from("<I", frame, 24)[0] == 32000


def test_media_types_whitelist():
    assert MEDIA_TYPES == ("wav", "mp3", "raw", "ogg", "aac")
