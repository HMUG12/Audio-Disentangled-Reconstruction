"""批次62: 声音安全三层测试 (m10 方案 C)。

覆盖:
- 审计设施: JSONL 落盘 / ADR_SAFETY_AUDIT=0 关闭 / 写失败静默降级 / text_fp 确定性
- 条款 gate: .terms_agreed_v1 标记 mark/agreed 往返 + 无 UI 三入口
  (--agree-terms / ADR_AGREE_TERMS=1 / --from-shell) 与未同意 exit 2
- 分发侧: inject_wav_info LIST/INFO 注入 (wave.open 往返 + RIFF size 回写)
  + ADR_WAV_METADATA=0 关闭 + 非 WAV 原样返回 + pack_wav API 面注入
- 上传侧: profile_create 未同意 422 + save_voice consent/source 落 meta
- settings 声音安全三函数 (agree_terms / safety_audit_enabled / wav_metadata_enabled)
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adr.core import settings
from adr.core.audit import log_event, text_fp
from adr.core.terms import agreed, flag_path, mark


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """审计日志与条款标记都落在 adr_data_dir(): env 指向临时目录隔离。"""
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path / "adr_data"))
    yield


def _make_wav(seconds: float = 0.2, rate: int = 16000) -> bytes:
    """44B 头 + 16bit PCM 的最小合法 WAV (soundfile/wave 均可读)。"""
    n = int(seconds * rate)
    pcm = (np.sin(np.linspace(0, 440 * 2 * np.pi, n)) * 8000).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ===== settings 声音安全三函数 (批次62) =====

def test_settings_agree_terms(monkeypatch):
    monkeypatch.delenv("ADR_AGREE_TERMS", raising=False)
    assert settings.agree_terms() is False
    for v in ("1", "true", "yes", "on", "ON"):
        monkeypatch.setenv("ADR_AGREE_TERMS", v)
        assert settings.agree_terms() is True
    monkeypatch.setenv("ADR_AGREE_TERMS", "0")
    assert settings.agree_terms() is False


def test_settings_safety_audit_enabled(monkeypatch):
    monkeypatch.delenv("ADR_SAFETY_AUDIT", raising=False)
    assert settings.safety_audit_enabled() is True   # 默认开
    monkeypatch.setenv("ADR_SAFETY_AUDIT", "0")
    assert settings.safety_audit_enabled() is False
    monkeypatch.setenv("ADR_SAFETY_AUDIT", "1")
    assert settings.safety_audit_enabled() is True


def test_settings_wav_metadata_enabled(monkeypatch):
    monkeypatch.delenv("ADR_WAV_METADATA", raising=False)
    assert settings.wav_metadata_enabled() is True   # 默认开
    monkeypatch.setenv("ADR_WAV_METADATA", "0")
    assert settings.wav_metadata_enabled() is False


# ===== 审计设施 =====

def test_text_fp_deterministic():
    assert text_fp("你好世界") == text_fp("你好世界")
    assert text_fp("你好世界") != text_fp("你好世界!")
    assert len(text_fp("任意文本")) == 16


def test_audit_jsonl_written():
    log_event("synthesis", text_fp=text_fp("测试文本"), text_len=4, ref="a.wav")
    audit = flag_path().parent / "logs" / "audit.log"
    assert audit.exists()
    rec = json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["event"] == "synthesis"
    assert rec["text_fp"] == text_fp("测试文本")
    assert rec["text_len"] == 4
    assert "ts" in rec


def test_audit_disabled_by_env(monkeypatch):
    monkeypatch.setenv("ADR_SAFETY_AUDIT", "0")
    log_event("synthesis", text_fp="x")
    audit = flag_path().parent / "logs" / "audit.log"
    assert not audit.exists()


def test_audit_failure_degrades(monkeypatch):
    """写失败 (目录不可创建) 不抛异常 — 审计是旁路, 不能阻断业务。"""
    from adr.core import config
    monkeypatch.setenv("ADR_DATA_DIR", "")
    monkeypatch.setattr(config, "adr_data_dir",
                        lambda: object(), raising=True)  # 返回非 Path → mkdir 必炸
    log_event("synthesis", text_fp="x")  # 不应抛


# ===== 条款标记 =====

def test_terms_mark_and_agreed():
    assert agreed() is False
    assert mark() is True
    assert agreed() is True
    assert mark() is True  # 已存在直接成功
    assert flag_path().name == ".terms_agreed_v1"
    assert flag_path().read_text(encoding="utf-8").startswith("agreed_at=")


# ===== 无 UI 条款 gate (server/__main__._enforce_terms) =====

def _args(**kw) -> argparse.Namespace:
    base = dict(agree_terms=False, from_shell=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_enforce_terms_reject_exit2(capsys):
    from adr.server.__main__ import _enforce_terms
    with pytest.raises(SystemExit) as ei:
        _enforce_terms(_args())
    assert ei.value.code == 2
    io_ = capsys.readouterr()
    assert "使用条款未同意" in io_.out + io_.err


def test_enforce_terms_cli_arg_passes():
    from adr.server.__main__ import _enforce_terms
    _enforce_terms(_args(agree_terms=True))  # 不抛即放行
    assert agreed() is True                  # 写了标记


def test_enforce_terms_env_passes(monkeypatch):
    from adr.server.__main__ import _enforce_terms
    monkeypatch.setenv("ADR_AGREE_TERMS", "1")
    _enforce_terms(_args())
    assert agreed() is True


def test_enforce_terms_from_shell_passes():
    """--from-shell: 壳层条款视图担保, server 不拦截也不写标记。"""
    from adr.server.__main__ import _enforce_terms
    _enforce_terms(_args(from_shell=True))
    assert agreed() is False                 # 不写标记 (prewarm 先于用户同意)


def test_enforce_terms_flag_short_circuit():
    mark()
    from adr.server.__main__ import _enforce_terms
    _enforce_terms(_args())                  # 已有标记直接放行


# ===== 分发侧: WAV 元数据注入 =====

def test_inject_wav_info_roundtrip():
    from adr.utils.audio import inject_wav_info
    raw = _make_wav()
    out = inject_wav_info(raw)
    assert out != raw
    assert out[:4] == raw[:4]                # "RIFF" 魔数不动
    assert out[8:44] == raw[8:44]            # fmt/data 头契约不动 (仅 size 字段回写)
    # RIFF size 回写正确
    assert struct.unpack("<I", out[4:8])[0] == len(out) - 8
    # INFO 子 chunk 在尾部
    assert b"LIST" in out[-200:]
    assert b"INFO" in out[-200:]
    assert b"ISFT" in out[-200:]
    assert b"AI-generated voice cloning output" in out
    # wave.open 仍可解析 (数据帧数不变)
    with wave.open(io.BytesIO(out), "rb") as w:
        assert w.getnframes() == int(0.2 * 16000)
        assert w.getframerate() == 16000


def test_inject_wav_info_disabled(monkeypatch):
    from adr.utils.audio import inject_wav_info
    monkeypatch.setenv("ADR_WAV_METADATA", "0")
    raw = _make_wav()
    assert inject_wav_info(raw) == raw


def test_inject_wav_info_non_wav_passthrough():
    from adr.utils.audio import inject_wav_info
    junk = b"not a riff file at all........"
    assert inject_wav_info(junk) == junk
    short = b"RIFF"
    assert inject_wav_info(short) == short


def test_pack_wav_contains_info():
    """API 面: pack_wav 输出已带 AI 生成标注 (批次62)。"""
    from adr.server.audio_codec import pack_wav
    data = np.full(1600, 0.1, np.float32)
    buf = pack_wav(io.BytesIO(), data, 16000)
    raw = buf.getvalue()
    assert b"AI-generated voice cloning output" in raw


# ===== 上传侧: 声纹单独同意 + 素材来源声明 =====

class FakeEngine:
    def warmup(self, vits_weights=None, t2s_weights=None):
        pass


def test_profile_create_requires_consent(tmp_path):
    """consent_granted 缺失 → 422 (声纹属个人敏感信息, 批次62)。"""
    from adr.server import create_app
    audio = tmp_path / "ref.wav"
    audio.write_bytes(_make_wav())
    c = TestClient(create_app(engine=FakeEngine()))
    r = c.post("/api/adr/v1/profiles/create", json={
        "name": "测试音色", "audio_path": str(audio)})
    assert r.status_code == 422
    assert "同意" in str(r.json())


def test_save_voice_records_source_and_consent(tmp_path, monkeypatch):
    """save_voice 的 source/consent 透传落 meta.json (旧档案缺键, 读侧 .get)。"""
    from adr.models import voice_library
    monkeypatch.setattr(voice_library, "VOICES_DIR", tmp_path / "voices")
    src = tmp_path / "src.wav"
    src.write_bytes(_make_wav(seconds=3.5))
    consent = {"granted": True, "at": "2026-10-24T12:00:00+08:00",
               "statement": "本人素材"}
    vdir = voice_library.save_voice("测试A", str(src), prompt_text="你好",
                                    source="本人录制", consent=consent)
    meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "本人录制"
    assert meta["consent"]["granted"] is True
    assert meta["consent"]["statement"] == "本人素材"


def test_save_voice_meta_backward_compat(tmp_path, monkeypatch):
    """不带 source/consent 时 meta 落 None — 旧读侧 .get 不炸。"""
    from adr.models import voice_library
    monkeypatch.setattr(voice_library, "VOICES_DIR", tmp_path / "voices")
    src = tmp_path / "src.wav"
    src.write_bytes(_make_wav(seconds=3.5))
    vdir = voice_library.save_voice("测试B", str(src))
    meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("source") is None
    assert meta.get("consent") is None
