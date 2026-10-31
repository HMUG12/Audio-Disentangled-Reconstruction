"""批次66 审计回归: engines 组 (P11/P12/P21/P22/#53)。"""
from __future__ import annotations

import contextlib
import tempfile
import threading

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path / "data"))


# ---- P11: ONNX 导出失败 → 回退 torch BERT; 并发惰性初始化只建一次 ----

class _Out:
    def __init__(self, hs):
        self.hidden_states = hs


class _TinyBert:
    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None,
                 output_hidden_states=False):
        import torch
        b, s = input_ids.shape
        return _Out([None, torch.full((b, s, 4), 7.0), None, None])


def test_onnx_bert_falls_back_to_torch_when_export_fails(tmp_path, monkeypatch):
    import torch
    import transformers

    from adr.models import gsv_engine

    loads = []
    lock = threading.Lock()

    def _from_pretrained(path, *a, **k):
        with lock:
            loads.append(path)
        return _TinyBert()

    def _boom(self, m):
        cache.write_bytes(b"half")   # 模拟半截导出
        raise ModuleNotFoundError("No module named 'onnxscript'")

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", _from_pretrained)
    monkeypatch.setattr(gsv_engine._OnnxBertFeat, "_export", _boom)
    cache = tmp_path / "bert_feat_cpu.onnx"
    obj = gsv_engine._OnnxBertFeat("stub", cache)
    ids = torch.ones(1, 5, dtype=torch.long)

    results = []
    ts = [threading.Thread(target=lambda: results.append(
        obj(input_ids=ids, attention_mask=ids, token_type_ids=ids)))
        for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert len(results) == 4
    for r in results:
        assert r["hidden_states"][0].shape == (1, 5, 4)
        assert float(r["hidden_states"][0][0, 0, 0]) == 7.0   # 取倒数第 3 层
    assert len(loads) == 1, "并发首调应只加载/导出一次"
    assert not cache.exists(), "半截导出文件应被清理"


def test_onnx_bert_fallback_prefers_given_model(tmp_path, monkeypatch):
    """_lazy_init 传入 GSV 原 BERT: ORT 缺失时直接复用, 不再加载第二份。"""
    import sys

    import torch
    import transformers

    from adr.models import gsv_engine

    cache = tmp_path / "bert.onnx"
    cache.write_bytes(b"exists")   # 缓存已在: 不走导出
    monkeypatch.setitem(sys.modules, "onnxruntime", None)   # import 即失败
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        lambda *a, **k: pytest.fail("不应重复加载 BERT"))
    obj = gsv_engine._OnnxBertFeat("stub", cache, fallback=_TinyBert())
    ids = torch.ones(2, 3, dtype=torch.long)
    r = obj(input_ids=ids, attention_mask=ids, token_type_ids=ids)
    assert r["hidden_states"][0].shape == (2, 3, 4)
    assert cache.exists(), "非本次导出的缓存不应被删"


# ---- P12: cleanup_cache 真正删掉 .part 文件 ----

def test_cleanup_cache_removes_part_file(tmp_path, monkeypatch):
    from adr.models import gsv_bootstrap

    part = tmp_path / "x.zip.part"
    part.write_bytes(b"residue")
    monkeypatch.setattr(gsv_bootstrap, "_MANIFEST",
                        [{"extract_to": tmp_path, "zip_name": "x.zip"},
                         {"extract_to": tmp_path, "zip_name": "absent.zip"}])
    gsv_bootstrap.cleanup_cache()
    assert not part.exists()


# ---- P21: vc_single 返回 None/NaN 音频时显式报错, 不静默输出 NaN ----

class _FakeVC:
    def __init__(self, audio):
        self.audio = audio

    def vc_single(self, *a, **k):
        return "info-msg", (40000, self.audio)


def _rvc(audio):
    from adr.models.rvc_engine import RVCEngine
    eng = RVCEngine()
    eng._rvc_context = lambda: contextlib.nullcontext()
    eng._vc = _FakeVC(audio)
    eng._loaded_model = "m"
    return eng


@pytest.mark.parametrize("audio", [None, np.array([0.1, np.nan], np.float32),
                                   np.zeros(0, np.float32)])
def test_rvc_convert_rejects_invalid_audio(tmp_path, audio):
    with pytest.raises(RuntimeError, match="RVC"):
        _rvc(audio).convert(str(tmp_path / "in.wav"), "m")


def test_rvc_convert_normal_path_unchanged(tmp_path):
    pcm = np.array([16384, -16384], dtype=np.int16)
    audio, sr = _rvc(pcm).convert(str(tmp_path / "in.wav"), "m")
    assert sr == 40000
    np.testing.assert_allclose(audio, [0.5, -0.5])


# ---- P22: sing_like 的 RVC 临时 wav 成功/失败都清理 ----

@pytest.mark.parametrize("fail", [True, False])
def test_sing_like_cleans_temp_wav(tmp_path, monkeypatch, fail):
    import adr.data.f0 as f0mod
    import adr.models.melody_bridge as mb
    import adr.models.rvc_engine as rvc
    import adr.models.voice_library as vl
    from adr.models.diffsinger_engine import DiffSingerEngine

    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmpdir))

    class _F0:
        def extract_with_sr(self, ref):
            return np.zeros(10, np.float32), 24000

    class _R:
        def convert(self, path, *a, **k):
            from pathlib import Path
            assert Path(path).parent == tmpdir and Path(path).exists()
            if fail:
                raise RuntimeError("rvc fail")
            return np.zeros(8, np.float32), 40000

    monkeypatch.setattr(f0mod, "F0Extractor", _F0)
    monkeypatch.setattr(mb, "build_word_level_input",
                        lambda t, f0, sample_rate: {"text": t, "notes": "C4",
                                                    "notes_duration": "0.5"})
    monkeypatch.setattr(vl, "load_voice", lambda n: {"rvc_weights": "w.pth"})
    monkeypatch.setattr(rvc, "get_rvc_engine", lambda: _R())
    eng = DiffSingerEngine()
    monkeypatch.setattr(eng, "synthesize_word",
                        lambda *a, **k: (np.zeros(240, np.float32), 24000))

    if fail:
        with pytest.raises(RuntimeError):
            eng.sing_like("la", "ref.wav", voice_name="me")
    else:
        assert eng.sing_like("la", "ref.wav", voice_name="me")[1] == 40000
    assert list(tmpdir.glob("*.wav")) == []
