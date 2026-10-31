"""批次66 审计回归: cli_misc 组 (P13 hub 校验 / P14-P15 doctor 裁剪 / P16 list 遮蔽 / #51-#52 死导入)。"""
import ast
import builtins
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

import adr.core.doctor as doctor
from adr.models.hub import PRETRAINED_REGISTRY, PretrainedHub

REPO = Path(__file__).resolve().parent.parent


class _Resp(io.BytesIO):
    def __init__(self, data, headers):
        super().__init__(data)
        self.headers = headers


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(doctor, "VOICES", tmp_path / "voices")
    monkeypatch.setattr(doctor, "LOGS", tmp_path / "logs")


# ---- P13 ----------------------------------------------------------------
def test_registry_all_have_sha256():
    for n, i in PRETRAINED_REGISTRY.items():
        assert i.sha256 and len(i.sha256) == 64, n


def test_download_rejects_unverifiable_body(tmp_path, monkeypatch):
    name = "bigvgan_22khz_80band"
    monkeypatch.setattr(PRETRAINED_REGISTRY[name], "sha256", None)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(b"x" * 10, {}))
    h = PretrainedHub(cache_dir=tmp_path / "c")
    with pytest.raises(RuntimeError, match="Unverifiable"):
        h.download(name, progress=False)
    target = h.cache_dir / PRETRAINED_REGISTRY[name].filename
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".part").exists()


def test_download_accepts_matching_sha(tmp_path, monkeypatch):
    import hashlib
    name = "wavlm_base"
    body = b"model-bytes" * 300
    monkeypatch.setattr(PRETRAINED_REGISTRY[name], "sha256", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(body, {}))
    h = PretrainedHub(cache_dir=tmp_path / "c")
    assert h.download(name, progress=False).read_bytes() == body


# ---- P14 / P15 ------------------------------------------------------------
def _profile(tmp_path, secs=12.0, sr=16000, subtype="FLOAT"):
    d = tmp_path / "voices" / "spk"
    d.mkdir(parents=True)
    t = np.arange(int(secs * sr)) / sr
    sf.write(str(d / "ref.wav"), (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"),
             sr, subtype=subtype)
    (d / "meta.json").write_text(json.dumps({"prompt_text": "原句"}), encoding="utf-8")
    return d


def test_doctor_fix_backs_up_and_flags_meta(tmp_path):
    d = _profile(tmp_path)
    orig = (d / "ref.wav").read_bytes()
    report = doctor.run_checks(fix=True)
    v = {c["id"]: c for c in report["checks"]}["voices"]
    assert v["status"] == "fixed"
    info = sf.info(str(d / "ref.wav"))
    assert info.frames / info.samplerate <= 8.0 + 1e-6
    assert info.subtype == "FLOAT"  # 不降级为 PCM_16
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["prompt_text"] == "原句"
    bak = d / meta["ref_clipped"]["backup"]
    assert bak.read_bytes() == orig
    assert meta["ref_clipped"]["prompt_text_stale"] is True
    assert not list(d.glob("*.tmp"))


def test_doctor_fix_write_failure_isolated(tmp_path, monkeypatch):
    d = _profile(tmp_path, subtype="PCM_16")
    before = (d / "ref.wav").read_bytes()

    def boom(*a, **k):
        raise PermissionError(13, "file in use")

    monkeypatch.setattr(sf, "write", boom)
    report = doctor.run_checks(fix=True)
    ids = {c["id"]: c for c in report["checks"]}
    assert ids["voices"]["status"] == "warn" and "裁剪失败" in ids["voices"]["detail"]
    assert "stop_files" in ids and "disk" in ids and "service" in ids
    assert (d / "ref.wav").read_bytes() == before
    assert sorted(p.name for p in d.iterdir()) == ["meta.json", "ref.wav"]


# ---- P16 ------------------------------------------------------------------
def test_cli_does_not_shadow_builtin_list():
    import adr.cli as cli
    assert getattr(cli, "list", builtins.list) is builtins.list
    assert "list" in cli.main.commands  # 子命令名不变


@pytest.mark.parametrize("args", [["model", "info", "nope"], ["train", "-d", "x", "-b", "nope"]])
def test_cli_unknown_name_exits_1(args, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from adr.cli import main
    r = CliRunner().invoke(main, args)
    assert r.exit_code == 1, r.output


def test_cli_list_command_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from adr.cli import main
    r = CliRunner().invoke(main, ["list"])
    assert r.exit_code == 0 and "Registered plugins" in r.output


# ---- #51 / #52 ------------------------------------------------------------
@pytest.mark.parametrize("rel,names", [
    ("adr/models/base.py", {"field", "Path", "Any"}),
    ("adr/models/vae.py", {"dataclass", "F"}),
])
def test_dead_imports_removed(rel, names):
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    imported = {a.asname or a.name.split(".")[-1]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names}
    assert not (imported & names)
