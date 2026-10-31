"""批次66 审计回归: voice_library / voice_package 并发写与 zip-bomb 防护。

P9/#30: meta 读-改-写并发丢更新, 固定 tmp 名互相截断, Windows os.replace PermissionError。
#31: import_voice 冲突检查与落位间 TOCTOU。
#38: import_voice 解压前未限制成员解压后大小 (zip-bomb)。
"""
import json
import threading
import time
import zipfile

import numpy as np
import pytest
import soundfile as sf

from adr.models import voice_library as vl
from adr.models import voice_package as vp


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(vl, "VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(vp, "GSV_DIR", tmp_path / "gsv")
    monkeypatch.setattr(vp, "RVC_DIR", tmp_path / "rvc")
    (tmp_path / "voices").mkdir()
    return tmp_path


def _seed(vdir):
    vdir.mkdir(parents=True)
    (vdir / "meta.json").write_text(json.dumps(
        {"name": vdir.name, "vits_weights": "seed.pth",
         "t2s_weights": "seed.ckpt", "created_at": "2026-01-01 00:00:00"}),
        encoding="utf-8")


def _run_threads(n, fn):
    errors, start = [], threading.Barrier(n)

    def worker(i):
        start.wait()
        try:
            fn(i)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return errors


def test_concurrent_bind_rollback_no_lost_update(env):
    """P9: 并发 bind 不丢更新、不抛 PermissionError, 且不残留 tmp 文件。"""
    vdir = vl.VOICES_DIR / "v"
    _seed(vdir)
    errors = _run_threads(30, lambda i: vl.bind_weights(
        "v", vits_weights=f"v{i}.pth", note=f"b{i}"))
    assert not errors, errors[:3]
    meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["weights_history"]) == 30
    assert sorted(p.name for p in vdir.iterdir()) == ["meta.json"]


def test_concurrent_finalize_and_bind_consistent(env):
    """P9: bind 与 finalize_bind_sim 交错时 meta 始终是合法 JSON 且 history 完整。"""
    vdir = vl.VOICES_DIR / "v"
    _seed(vdir)

    def fn(i):
        if i % 2:
            vl.finalize_bind_sim("v", 0.5)
        else:
            vl.bind_weights("v", vits_weights=f"v{i}.pth")
    assert not _run_threads(20, fn)
    meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["weights_history"]) == 10


def test_write_meta_unique_tmp_and_permission_retry(env, monkeypatch):
    """#30: tmp 名唯一; os.replace 短暂 PermissionError 时重试成功。"""
    vdir = vl.VOICES_DIR / "v"
    vdir.mkdir()
    real_replace, seen, fails = vl.os.replace, [], [2]

    def flaky(src, dst):
        seen.append(src)
        if fails[0]:
            fails[0] -= 1
            raise PermissionError(13, "denied")
        return real_replace(src, dst)
    monkeypatch.setattr(vl.os, "replace", flaky)
    vl._write_meta(vdir, {"a": 1})
    vl._write_meta(vdir, {"a": 2})
    assert json.loads((vdir / "meta.json").read_text(encoding="utf-8")) == {"a": 2}
    assert not any(n.endswith("meta.json.tmp") for n in seen)   # 不再用固定名
    assert len(set(seen)) == 2   # 两次写 tmp 名不同 (重试沿用同名)
    assert sorted(p.name for p in vdir.iterdir()) == ["meta.json"]


def _mk_wav(path, seconds=4.0, sr=16000):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False, dtype="float32")
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr)
    return path


def _mk_pkg(path, name, ref_bytes, extra=None):
    manifest = {"format": vp.FORMAT, "version": vp.VERSION, "name": name}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(vp._MANIFEST, json.dumps(manifest))
        zf.writestr(vp._META, json.dumps({"name": name, "prompt_text": path.stem}))
        zf.writestr(vp._REF, ref_bytes)
        for rel, data in (extra or {}).items():
            zf.writestr(rel, data)
    return path


def test_import_same_name_concurrent_toctou(env, monkeypatch):
    """#31: 两个同名导入并发, 只能一个成功, 另一个报"档案已存在"; ref 与 meta 同源。"""
    tmp = env
    ref = _mk_wav(tmp / "r.wav").read_bytes()
    pkgs = [_mk_pkg(tmp / f"pkg{i}.adrvoice", "same", ref + bytes([i]))
            for i in range(2)]
    real_copy = vp._copy_member

    def slow_copy(zf, rel, dst):   # 放大检查→落位窗口
        time.sleep(0.2)
        return real_copy(zf, rel, dst)
    monkeypatch.setattr(vp, "_copy_member", slow_copy)
    errors = _run_threads(2, lambda i: vp.import_voice(str(pkgs[i])))
    assert len(errors) == 1 and "档案已存在" in errors[0], errors
    vdir = vl.VOICES_DIR / "same"
    winner = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))["prompt_text"]
    assert (vdir / "ref.wav").read_bytes()[-1] == int(winner[-1])


def test_import_rejects_oversized_manifest_before_read(env, monkeypatch):
    """#38: manifest 解压后超上限时在 zf.read 之前拒绝。"""
    monkeypatch.setattr(vp, "_MAX_INMEM_BYTES", 1 << 20)
    pkg = env / "bomb.adrvoice"
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(vp._MANIFEST, b" " * (2 << 20))   # 2MB 空白, 压缩后仅 KB 级
        zf.writestr(vp._META, "{}")
        zf.writestr(vp._REF, b"x")
    monkeypatch.setattr(zipfile.ZipFile, "read",
                        lambda *a, **k: pytest.fail("超限成员不应被读入内存"))
    with pytest.raises(ValueError, match="ADR_IMPORT_MAX_INMEM_MB"):
        vp.import_voice(str(pkg))


def test_import_rejects_total_uncompressed_over_limit(env, monkeypatch):
    """#38: 成员解压后总量超上限时在解压任何权重前拒绝, 不落任何文件。"""
    monkeypatch.setattr(vp, "_MAX_TOTAL_BYTES", 1 << 20)
    pkg = _mk_pkg(env / "big.adrvoice", "big", b"x",
                  {"weights/a.pth": b"\0" * (2 << 20)})
    with pytest.raises(ValueError, match="ADR_IMPORT_MAX_TOTAL_MB"):
        vp.import_voice(str(pkg))
    assert not (env / "gsv").exists()
    assert not (vl.VOICES_DIR / "big").exists()


def test_import_normal_package_still_works(env):
    """默认上限下正常包可导入, ref 流式落盘内容一致。"""
    ref = _mk_wav(env / "r.wav").read_bytes()
    pkg = _mk_pkg(env / "ok.adrvoice", "ok", ref)
    out = vp.import_voice(str(pkg))
    assert out["name"] == "ok"
    assert (vl.VOICES_DIR / "ok" / "ref.wav").read_bytes() == ref
