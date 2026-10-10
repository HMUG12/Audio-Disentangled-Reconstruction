"""测试 adr.models.voice_library (批次 8 档案绑定风格)。"""

import soundfile as sf
import numpy as np
import pytest

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


# ============================================================
# 批次50 (m10 E 节): 权重版本化 — bind 入史 / 回滚 / 退化门结算
# ============================================================

import json


def _meta(vl_root, name):
    return json.loads((vl_root / name / "meta.json").read_text(encoding="utf-8"))


def test_bind_weights_pushes_history(tmp_path, monkeypatch):
    """绑定新权重: 旧绑定入 weights_history, last_sim/绑定时间刷新。"""
    import adr.models.voice_library as vl
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    ref = _mk_wav(tmp_path)
    vl.save_voice("版本音", ref, vits_weights="SoVITS_weights_v2/v1.pth")
    vl.bind_weights("版本音", "SoVITS_weights_v2/v2.pth", sim=0.82,
                    note="续训新版本")
    meta = _meta(tmp_path / "voices", "版本音")
    assert meta["vits_weights"] == "SoVITS_weights_v2/v2.pth"
    assert meta["last_sim"] == 0.82
    assert [h["vits"] for h in meta["weights_history"]] == ["SoVITS_weights_v2/v1.pth"]
    assert meta["weights_history"][0]["note"] == "续训新版本"
    assert meta["weights_bound_at"]


def test_bind_weights_requires_existing_binding(tmp_path, monkeypatch):
    """档案不存在 / 无既有绑定权重 → 拒绝 (首绑走 save_voice)。"""
    import adr.models.voice_library as vl
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    ref = _mk_wav(tmp_path)
    with pytest.raises(ValueError, match="档案不存在"):
        vl.bind_weights("幽灵", "w.pth")
    vl.save_voice("裸档", ref)  # 未绑权重
    with pytest.raises(ValueError, match="无既有绑定"):
        vl.bind_weights("裸档", "w.pth")


def test_rollback_voice_roundtrip(tmp_path, monkeypatch):
    """v1→v2→v3 连续绑定后回滚: v3 现行入史(redo), v2 恢复现行;
    再回滚逐版回退到 v1 (redo 条目不被当作"上一版"弹回)。"""
    import adr.models.voice_library as vl
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    ref = _mk_wav(tmp_path)
    vl.save_voice("版本音", ref, vits_weights="w1.pth")
    vl.bind_weights("版本音", "w2.pth", sim=0.82)
    vl.bind_weights("版本音", "w3.pth", sim=0.80)

    meta = vl.rollback_voice("版本音")
    assert meta["vits_weights"] == "w2.pth"
    assert meta["last_sim"] == 0.82
    assert [h["vits"] for h in meta["weights_history"]] == ["w1.pth", "w3.pth"]
    assert meta["weights_history"][-1]["note"] == "回滚前版本"

    meta = vl.rollback_voice("版本音")
    assert meta["vits_weights"] == "w1.pth"
    assert [h["vits"] for h in meta["weights_history"]] == ["w2.pth", "w3.pth"]

    # 回滚到底后再训新版权重: redo 条目不妨碍 "回到训练前的现行版"
    vl.bind_weights("版本音", "w4.pth", sim=0.90)
    meta = vl.rollback_voice("版本音")
    assert meta["vits_weights"] == "w1.pth"
    assert meta["last_sim"] is None  # w1 从未有门禁结算值


def test_rollback_empty_history_raises(tmp_path, monkeypatch):
    """无历史版本 / 档案不存在 → ValueError。"""
    import adr.models.voice_library as vl
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    ref = _mk_wav(tmp_path)
    vl.save_voice("新档", ref)
    with pytest.raises(ValueError, match="无历史版本"):
        vl.rollback_voice("新档")
    with pytest.raises(ValueError, match="档案不存在"):
        vl.rollback_voice("幽灵")


def test_finalize_bind_sim_degraded_gate(tmp_path, monkeypatch):
    """退化门: 新 sim 比上一版 (history 末条 sim) 低 0.05+ → 置 degraded;
    恰好 0.05 边界不触发; 换版重置标记。"""
    import adr.models.voice_library as vl
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    ref = _mk_wav(tmp_path)
    vl.save_voice("版本音", ref, vits_weights="w1.pth")
    assert vl.finalize_bind_sim("版本音", 0.85) is None  # 首训结算: 无基线只写 last_sim
    vl.bind_weights("版本音", "w2.pth")  # 续训绑定: w1 (sim=0.85) 随之入史

    deg = vl.finalize_bind_sim("版本音", 0.75)
    assert deg == {"new_sim": 0.75, "old_sim": 0.85}
    assert _meta(tmp_path / "voices", "版本音")["degraded"] == deg

    # 边界: 恰差 0.05 不算退化; 换版重置退化标记
    vl.bind_weights("版本音", "w3.pth")
    assert "degraded" not in _meta(tmp_path / "voices", "版本音")
    assert vl.finalize_bind_sim("版本音", 0.70) is None  # 0.70 == 0.75 - 0.05
    assert _meta(tmp_path / "voices", "版本音")["last_sim"] == 0.70


def test_finalize_bind_sim_first_train_no_baseline(tmp_path, monkeypatch):
    """首训无历史基线 → 返回 None, last_sim 照写。"""
    import adr.models.voice_library as vl
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    ref = _mk_wav(tmp_path)
    vl.save_voice("新档", ref, vits_weights="w1.pth")
    assert vl.finalize_bind_sim("新档", 0.77) is None
    assert _meta(tmp_path / "voices", "新档")["last_sim"] == 0.77


def test_voice_history_old_meta_compat(tmp_path, monkeypatch):
    """旧 meta (无 weights_history 字段) → voice_history 返回空列表不报错。"""
    import adr.models.voice_library as vl
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    ref = _mk_wav(tmp_path)
    vl.save_voice("老档", ref)
    assert vl.voice_history("老档") == []
