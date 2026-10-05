"""测试 adr.models.voice_library (批次 8 档案绑定风格)。"""

import soundfile as sf
import numpy as np

from adr.models.voice_library import list_voices, load_voice, save_voice


def _mk_wav(tmp_path, name="ref.wav"):
    # 4s 正弦波: 建档防护要求 3~10s 且全零静音会被去静音裁空
    p = tmp_path / name
    t = np.linspace(0, 4, 64000, endpoint=False, dtype="float32")
    sf.write(str(p), (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 16000)
    return str(p)


def test_save_load_roundtrip(tmp_path, monkeypatch):
    """建档 → 加载往返, style 字段保留。"""
    import adr.models.voice_library as vl
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    ref = _mk_wav(tmp_path)
    vl.save_voice("测试音", ref, prompt_text="你好", style="冷淡 果断")
    prof = vl.load_voice("测试音")
    assert prof["prompt_text"] == "你好"
    assert prof["style"] == "冷淡 果断"
    assert prof["ref_audio"].endswith("ref.wav")
    assert "测试音" in vl.list_voices()


def test_save_voice_empty_name_rejected(tmp_path, monkeypatch):
    """空音色名拒绝。"""
    import adr.models.voice_library as vl
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    ref = _mk_wav(tmp_path)
    try:
        vl.save_voice("  ", ref)
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_archive_style_resolves_to_preset(tmp_path, monkeypatch):
    """档案 style 描述 → 风格预设解析 (绑定链路)。"""
    import adr.models.voice_library as vl
    from adr.models.style_presets import resolve_style
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    ref = _mk_wav(tmp_path)
    vl.save_voice("moss音", ref, style="无感情 冷淡 绝对理性")
    style, hit = resolve_style(vl.load_voice("moss音").get("style", ""))
    assert hit and style.key == "moss"
