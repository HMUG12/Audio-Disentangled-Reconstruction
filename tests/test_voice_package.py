"""测试 adr.models.voice_package (批次47 音色档案导入导出)。"""

import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from adr.models import voice_library as vl
from adr.models import voice_package as vp


def _mk_wav(tmp_path, name="ref.wav"):
    # 4s 正弦波 (档案防护要求 3~10s)
    p = tmp_path / name
    t = np.linspace(0, 4, 64000, endpoint=False, dtype="float32")
    sf.write(str(p), (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 16000)
    return str(p)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """隔离的档案库 / GSV / RVC 目录 + 假 validate_weights_file。"""
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(vp, "GSV_DIR", tmp_path / "gsv")
    monkeypatch.setattr(vp, "RVC_DIR", tmp_path / "rvc")
    validated = []
    monkeypatch.setattr("adr.models.gsv_engine.validate_weights_file",
                        lambda p: validated.append(p))
    return tmp_path, validated


def _mk_profile(tmp, name, with_weights=True, with_rvc=False):
    ref = _mk_wav(tmp, f"{name}_ref.wav")
    t2s = vits = rvc = index = None
    if with_weights:
        gsv = tmp / "gsv"
        (gsv / "GPT_weights_v2").mkdir(parents=True, exist_ok=True)
        t2s = gsv / "GPT_weights_v2" / f"{name}.ckpt"
        t2s.write_bytes(b"T2S" * 128)
        (gsv / "SoVITS_weights_v2").mkdir(parents=True, exist_ok=True)
        vits = gsv / "SoVITS_weights_v2" / f"{name}.pth"
        vits.write_bytes(b"VITS" * 128)
    if with_rvc:
        (tmp / "rvc" / "assets" / "weights").mkdir(parents=True, exist_ok=True)
        rvc = tmp / "rvc" / "assets" / "weights" / f"{name}.pth"
        rvc.write_bytes(b"RVC" * 16)
        (tmp / "rvc" / "logs" / name).mkdir(parents=True, exist_ok=True)
        index = tmp / "rvc" / "logs" / name / "added.index"
        index.write_bytes(b"IDX" * 16)
    vl.save_voice(name, ref, prompt_text="你好", style="冷淡 果断",
                  t2s_weights=str(t2s) if t2s else None,
                  vits_weights=str(vits) if vits else None,
                  rvc_weights=rvc.name if rvc else None,
                  rvc_index=str(index) if index else None)
    return {"t2s": t2s, "vits": vits, "rvc": rvc, "index": index}


def test_export_import_roundtrip(env):
    """导出 → 删除档案与权重 → 导入: 权重回位/内容一致/校验被调/元数据保留。"""
    tmp, validated = env
    _mk_profile(tmp, "测试音")
    pkg = vp.export_voice("测试音", str(tmp / "out.adrvoice"),
                          progress=lambda m: None)
    assert pkg.exists()

    with zipfile.ZipFile(pkg) as zf:
        names = set(zf.namelist())
        assert {"manifest.json", "meta.json", "ref.wav"} <= names
        assert any(n.startswith("weights/") for n in names)
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["format"] == vp.FORMAT and manifest["version"] == 1
        pmeta = json.loads(zf.read("meta.json"))
        assert pmeta["vits_weights"] is None          # 可移植副本权重置空
        assert pmeta["style"] == "冷淡 果断"

    # 模拟换机: 档案与权重全部消失
    vl.load_voice("测试音")["ref_audio"]
    (tmp / "voices" / "测试音" / "meta.json").unlink()
    (tmp / "gsv" / "GPT_weights_v2" / "测试音.ckpt").unlink()
    (tmp / "gsv" / "SoVITS_weights_v2" / "测试音.pth").unlink()

    r = vp.import_voice(str(pkg), progress=lambda m: None)
    assert r["name"] == "测试音"
    prof = vl.load_voice("测试音")
    assert prof["prompt_text"] == "你好" and prof["style"] == "冷淡 果断"
    assert prof["t2s_weights"] == str(tmp / "gsv" / "GPT_weights_v2" / "测试音.ckpt")
    assert prof["vits_weights"].endswith("测试音.pth")
    assert (tmp / "gsv" / "GPT_weights_v2" / "测试音.ckpt").read_bytes() == b"T2S" * 128
    assert (tmp / "gsv" / "SoVITS_weights_v2" / "测试音.pth").read_bytes() == b"VITS" * 128
    assert len(validated) == 2                        # t2s + vits 都过校验


def test_import_conflict_rejected_then_overwrite(env):
    """同名档案默认拒绝导入, overwrite 放行。"""
    tmp, _ = env
    _mk_profile(tmp, "冲突音")
    pkg = vp.export_voice("冲突音", str(tmp / "c.adrvoice"))
    with pytest.raises(ValueError, match="档案已存在"):
        vp.import_voice(str(pkg))
    vp.import_voice(str(pkg), overwrite=True)
    assert "冲突音" in vl.list_voices()


def test_import_name_override(env):
    """--name 换名导入。"""
    tmp, _ = env
    _mk_profile(tmp, "原名")
    pkg = vp.export_voice("原名", str(tmp / "n.adrvoice"))
    r = vp.import_voice(str(pkg), name="新名")
    assert r["name"] == "新名" and "新名" in vl.list_voices()


def test_export_missing_weight_rejected(env):
    """档案引用的权重文件缺失 → 导出报错 (引导 adr doctor)。"""
    tmp, _ = env
    _mk_profile(tmp, "断链音")
    (tmp / "gsv" / "SoVITS_weights_v2" / "断链音.pth").unlink()
    with pytest.raises(ValueError, match="权重文件不存在"):
        vp.export_voice("断链音", str(tmp / "x.adrvoice"))


def test_bare_profile_roundtrip(env):
    """零样本裸档案 (无权重) 往返。"""
    tmp, validated = env
    _mk_profile(tmp, "裸音", with_weights=False)
    pkg = vp.export_voice("裸音", str(tmp / "b.adrvoice"))
    with zipfile.ZipFile(pkg) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["weights"] is None
    vl.load_voice("裸音")  # 存在性确认
    (tmp / "voices" / "裸音" / "meta.json").unlink()
    vp.import_voice(str(pkg))
    assert "裸音" in vl.list_voices()
    assert validated == []


def test_zip_slip_rejected(env):
    """manifest 权重路径带穿越成分 → 拒绝导入。"""
    tmp, _ = env
    _mk_profile(tmp, "安全音")
    pkg = vp.export_voice("安全音", str(tmp / "s.adrvoice"))
    evil = tmp / "evil.adrvoice"
    with zipfile.ZipFile(pkg) as zin, \
            zipfile.ZipFile(evil, "w") as zout:
        for item in zin.namelist():
            data = zin.read(item)
            if item == "manifest.json":
                m = json.loads(data)
                m["weights"]["vits"] = "weights/../../evil.pth"
                data = json.dumps(m, ensure_ascii=False).encode()
            zout.writestr(item, data)
    with pytest.raises(ValueError, match="非法"):
        vp.import_voice(str(evil), overwrite=True)   # 越过冲突检查, 直击穿越校验


def test_rvc_assets_roundtrip(env):
    """rvc_weights (纯文件名) + rvc_index (路径) 导出导入, meta 惯例保持。"""
    tmp, _ = env
    _mk_profile(tmp, "rvc音", with_weights=False, with_rvc=True)
    pkg = vp.export_voice("rvc音", str(tmp / "r.adrvoice"))
    vl.load_voice("rvc音")
    (tmp / "voices" / "rvc音" / "meta.json").unlink()
    vp.import_voice(str(pkg))
    prof = vl.load_voice("rvc音")
    assert prof["rvc_weights"] == "rvc音.pth"          # 纯文件名 (weight_root 口径)
    assert prof["rvc_index"].endswith("added.index")   # 本机绝对路径
    assert (tmp / "rvc" / "assets" / "weights" / "rvc音.pth").exists()
