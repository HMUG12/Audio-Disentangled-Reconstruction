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


# ---- 批次92: doctor 自检扩展 (修复指引 / 依赖完整性 / 引擎失败原因) ----------
def test_doctor_new_checks_and_hints(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "VOICES", tmp_path / "voices")
    monkeypatch.setattr(doctor, "LOGS", tmp_path / "logs")
    rep = doctor.run_checks()
    ids = {c["id"]: c for c in rep["checks"]}
    for cid in ("gpu_adapt", "multilang", "nltk_data", "vc_runtime",
                "data_write", "disk"):
        assert cid in ids, cid
        assert "hint" in ids[cid]
    # 面向用户: warn/fail 必须给出下一步动作 (此处用 multilang 验证结构不空)
    assert ids["multilang"]["status"] in ("ok", "warn")
    assert rep["env"]["version"] and rep["env"]["python"]
    assert rep["env"]["platform"]


def test_doctor_engine_check_surfaces_startup_error(monkeypatch):
    from adr.models import gsv_engine as g
    from adr.server import console as c

    monkeypatch.setattr(g, "stage", lambda: "failed")
    monkeypatch.setattr(g, "stage_error",
                        lambda: "Traceback...\nValueError: 权重文件损坏")
    monkeypatch.setattr(g, "is_stalled", lambda: False)
    monkeypatch.setattr(g, "degraded_seconds", lambda: 0.0)
    monkeypatch.setattr(g, "is_ready", lambda: False)
    monkeypatch.setattr(g, "watchdog_grace_sec", lambda: 300.0)
    rep = c._doctor_engine_check()
    assert rep["status"] == "fail"
    assert "权重文件损坏" in rep["detail"] and rep["hint"]


def test_doctor_endpoint_merges_engine_check(monkeypatch):
    import asyncio

    from adr.models import gsv_engine as g
    from adr.server import console as c

    monkeypatch.setattr("adr.core.doctor.run_checks",
                        lambda fix=False: {"checks": [], "summary": {},
                                           "generated_at": "", "env": {}})
    monkeypatch.setattr(g, "stage", lambda: "ready")
    monkeypatch.setattr(g, "stage_error", lambda: "")
    monkeypatch.setattr(g, "is_stalled", lambda: False)
    monkeypatch.setattr(g, "degraded_seconds", lambda: 0.0)
    monkeypatch.setattr(g, "is_ready", lambda: True)
    monkeypatch.setattr(g, "watchdog_grace_sec", lambda: 300.0)
    rep = asyncio.run(c.system_doctor(fix=False, _guard=None))
    assert [x["id"] for x in rep["checks"]] == ["engine"]
    assert rep["summary"] == {"ok": 1}


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


# ---- #24 (批次89): adr train 接入质量回调, 默认关闭 ----
def _run_train(tmp_path, monkeypatch, extra_args):
    """跑 adr train, 用假 Trainer 捕获构造参数 (不真正训练)。"""
    import adr.cli as cli
    import adr.training as training
    from adr.core import registry as registry_mod

    class _Model:
        pass

    class _Backbones:
        def get(self, name):
            return _Model

        def keys(self):
            return ["sovits"]

    captured = {}

    class _Trainer:
        def __init__(self, **kw):
            captured.update(kw)

        def fit(self):
            return {"val/loss": 0.1}

    monkeypatch.setattr(registry_mod, "BACKBONE_REGISTRY", _Backbones())
    monkeypatch.setattr(training, "Trainer", _Trainer)
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli.main, ["train", "-d", str(tmp_path / "data"),
                                      *extra_args])
    assert r.exit_code == 0, r.output
    return captured["callbacks"]


def test_train_no_quality_callbacks_by_default(tmp_path, monkeypatch):
    """默认不加任何质量回调 → 既有训练行为/耗时不变。"""
    assert _run_train(tmp_path, monkeypatch, []) == []


def test_train_wires_asr_and_warm_restart(tmp_path, monkeypatch):
    from adr.training.callbacks import ASRCallback, WarmRestartCallback

    cbs = _run_train(tmp_path, monkeypatch,
                     ["--eval-asr", "--eval-samples", "2", "--warm-restart"])
    assert [type(c) for c in cbs] == [ASRCallback, WarmRestartCallback]
    assert cbs[0].n_samples == 2
    assert cbs[1].metric_name == "asr_cer" and cbs[1].mode == "min"


def test_train_wires_wav_quality_and_prefers_it_over_loss(tmp_path, monkeypatch):
    from adr.training.callbacks import (
        ASRCallback,
        WarmRestartCallback,
        WavQualityCallback,
    )

    cbs = _run_train(tmp_path, monkeypatch,
                     ["--eval-wav-quality", "--eval-asr", "--warm-restart"])
    assert [type(c) for c in cbs] == [WavQualityCallback, ASRCallback,
                                     WarmRestartCallback]
    # ASR 指标优先 (排在最后注册的指标即实际使用)
    assert cbs[-1].metric_name == "asr_cer"

    cbs = _run_train(tmp_path, monkeypatch, ["--eval-wav-quality", "--warm-restart"])
    assert [type(c) for c in cbs] == [WavQualityCallback, WarmRestartCallback]
    assert cbs[-1].metric_name == "wav_quality" and cbs[-1].mode == "max"


def test_train_warm_restart_alone_falls_back_to_val_loss(tmp_path, monkeypatch):
    from adr.training.callbacks import WarmRestartCallback

    cbs = _run_train(tmp_path, monkeypatch, ["--warm-restart"])
    assert [type(c) for c in cbs] == [WarmRestartCallback]
    assert cbs[0].metric_name == "val/loss" and cbs[0].mode == "min"
