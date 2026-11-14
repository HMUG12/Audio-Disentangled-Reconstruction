"""批次66 审计回归: engines 组 (P11/P12/P21/P22/#53)。"""
from __future__ import annotations

import contextlib
import os
import tempfile
import threading
import time

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


# ---- 批次83 (#15): 下载 sha256 校验 / 唯一临时名 / 残留包拒绝复用 ----

class _FakeResp:
    def __init__(self, data):
        self._buf = data
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        chunk, self._buf = self._buf[:n], self._buf[n:]
        return chunk


class _FakeOpener:
    def __init__(self, data):
        self.data = data

    def open(self, req, timeout=60):
        return _FakeResp(self.data)


def _patch_download_body(monkeypatch, body):
    from adr.models import gsv_bootstrap

    monkeypatch.setattr(gsv_bootstrap.urllib.request, "build_opener",
                        lambda *a, **k: _FakeOpener(body))
    monkeypatch.setattr(gsv_bootstrap.urllib.request, "getproxies", lambda: {})


def test_download_rejects_sha_mismatch_and_cleans_part(tmp_path, monkeypatch):
    """内容与清单 sha256 不符 → 报错且不落 dest, 不留 .part。"""
    import hashlib

    from adr.models import gsv_bootstrap

    body = b"model-bytes" * 512
    _patch_download_body(monkeypatch, body)
    dest = tmp_path / "pretrained.zip"
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        gsv_bootstrap._download("http://x", dest,
                                sha256=hashlib.sha256(body + b"!").hexdigest())
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_download_accepts_matching_sha(tmp_path, monkeypatch):
    """sha256 一致时落位成功, 临时名不是固定 .part。"""
    import hashlib

    from adr.models import gsv_bootstrap

    body = b"model-bytes" * 512
    _patch_download_body(monkeypatch, body)
    dest = tmp_path / "pretrained.zip"
    gsv_bootstrap._download("http://x", dest,
                            sha256=hashlib.sha256(body).hexdigest())
    assert dest.read_bytes() == body
    assert list(tmp_path.glob("*.part")) == []


def test_download_resumes_from_part_with_range(tmp_path, monkeypatch):
    """批次95: 中断残片存在时带 Range 续传, 拼接后 sha256 仍一致。"""
    import hashlib

    from adr.models import gsv_bootstrap

    body = b"0123456789" * 100
    prefix = body[:300]
    part = tmp_path / "m.zip.part"
    part.write_bytes(prefix)
    seen = {}

    class _Resp206:
        def __init__(self):
            self.headers = {
                "Content-Length": str(len(body) - len(prefix)),
                "Content-Range": f"bytes {len(prefix)}-{len(body) - 1}/{len(body)}",
            }
            self._buf = body[len(prefix):]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def getcode(self):
            return 206

        def read(self, n=-1):
            chunk, self._buf = self._buf[:n], self._buf[n:]
            return chunk

    class _Opener:
        def open(self, req, timeout=60):
            seen["range"] = req.get_header("Range")
            return _Resp206()

    monkeypatch.setattr(gsv_bootstrap.urllib.request, "build_opener",
                        lambda *a, **k: _Opener())
    monkeypatch.setattr(gsv_bootstrap.urllib.request, "getproxies", lambda: {})
    dest = tmp_path / "m.zip"
    gsv_bootstrap._download("http://x", dest,
                            sha256=hashlib.sha256(body).hexdigest())
    assert seen["range"] == f"bytes={len(prefix)}-"
    assert dest.read_bytes() == body
    assert not part.exists()


def test_download_restarts_when_server_ignores_range(tmp_path, monkeypatch):
    """源忽略 Range 返回 200 → 从头写, 不拼接脏残片。"""
    from adr.models import gsv_bootstrap

    body = b"full-body" * 200
    (tmp_path / "m.zip.part").write_bytes(b"stale")
    _patch_download_body(monkeypatch, body)
    dest = tmp_path / "m.zip"
    gsv_bootstrap._download("http://x", dest)
    assert dest.read_bytes() == body


def test_download_416_discards_part_and_restarts(tmp_path, monkeypatch):
    """本地残片比远端长 (416) → 丢弃重来, 不阻塞换源/重试。"""
    import urllib.error

    from adr.models import gsv_bootstrap

    body = b"payload" * 100
    (tmp_path / "m.zip.part").write_bytes(b"too-long")
    calls = {"n": 0}

    class _Opener:
        def open(self, req, timeout=60):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError("http://x", 416, "range", {}, None)
            return _FakeResp(body)

    monkeypatch.setattr(gsv_bootstrap.urllib.request, "build_opener",
                        lambda *a, **k: _Opener())
    monkeypatch.setattr(gsv_bootstrap.urllib.request, "getproxies", lambda: {})
    dest = tmp_path / "m.zip"
    gsv_bootstrap._download("http://x", dest)
    assert dest.read_bytes() == body
    assert calls["n"] == 2


def test_download_keeps_part_on_network_error(tmp_path, monkeypatch):
    """网络类错误保留 .part, 供下次续传; 不能一失败就删。"""
    from adr.models import gsv_bootstrap

    part = tmp_path / "m.zip.part"
    part.write_bytes(b"xyz")

    class _BoomOpener:
        def open(self, req, timeout=60):
            raise OSError("net down")

    monkeypatch.setattr(gsv_bootstrap.urllib.request, "build_opener",
                        lambda *a, **k: _BoomOpener())
    with pytest.raises(OSError):
        gsv_bootstrap._download("http://x", tmp_path / "m.zip")
    assert part.read_bytes() == b"xyz"


def test_download_rate_limit_env_and_throttle(monkeypatch):
    """限速旋钮解析 + 纯函数补觉时长。"""
    from adr.models import gsv_bootstrap

    monkeypatch.delenv("ADR_DOWNLOAD_RATE_MBPS", raising=False)
    assert gsv_bootstrap._rate_limit_mbps() == 0
    monkeypatch.setenv("ADR_DOWNLOAD_RATE_MBPS", " 2.5 ")
    assert gsv_bootstrap._rate_limit_mbps() == 2.5
    for bad in ("abc", "-1", "0", "0.05", "99999"):
        monkeypatch.setenv("ADR_DOWNLOAD_RATE_MBPS", bad)
        assert gsv_bootstrap._rate_limit_mbps() == 0

    mb = 1024 * 1024
    assert gsv_bootstrap._throttle_delay(2 * mb, 0.0, 2.0) == pytest.approx(1.0)
    assert gsv_bootstrap._throttle_delay(2 * mb, 5.0, 2.0) == 0.0
    assert gsv_bootstrap._throttle_delay(0, 0.0, 2.0) == 0.0
    assert gsv_bootstrap._throttle_delay(2 * mb, 0.0, 0) == 0.0


def _zip_bytes(member="good/keep.txt", data=b"ok"):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, data)
    return buf.getvalue()


def test_residual_zip_with_bad_sha_redownloaded(tmp_path, monkeypatch):
    """残留 zip 的 sha256 不符时丢弃并重新下载 (旧实现只看能否打开)。"""
    from adr.models import gsv_bootstrap

    entry = {"key": "k", "label": "l", "zip_name": "p.zip",
             "urls": {"hf": "http://x"}, "extract_to": tmp_path,
             "marker": tmp_path / "good", "sha256": "0" * 64}
    (tmp_path / "p.zip").write_bytes(_zip_bytes(data=b"stale"))
    body = _zip_bytes()
    calls = []

    def fake_download(url, dest, sha256="", no_proxy=False):
        calls.append(sha256)
        dest.write_bytes(body)

    monkeypatch.setattr(gsv_bootstrap, "_download", fake_download)
    gsv_bootstrap._acquire(entry)
    assert calls == ["0" * 64]          # 坏残留被丢弃, 重新下载
    assert (tmp_path / "good" / "keep.txt").read_bytes() == b"ok"
    assert not (tmp_path / "p.zip").exists()


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


# ---- 批次77 (#6): 权重指纹 —— 同路径文件被覆盖必须重新预检 + 重载 ----

def test_weights_stamp_notices_mtime_and_missing(tmp_path):
    from adr.models import gsv_engine

    p = tmp_path / "w.pth"
    p.write_bytes(b"v1")
    s1 = gsv_engine.weights_stamp(str(p))
    os.utime(p, (1_600_000_000, 1_600_000_000))
    s2 = gsv_engine.weights_stamp(str(p))
    assert s2 != s1 and s2[2] == s1[2]        # 仅 mtime 变也感知
    missing = gsv_engine.weights_stamp(str(tmp_path / "nope.pth"))
    assert missing[1:] == (0, 0)


def test_ensure_weights_reloads_when_same_path_replaced(tmp_path, monkeypatch):
    from adr.models import gsv_engine

    monkeypatch.setattr(gsv_engine.GSVEngine, "_gsv_context",
                        lambda self: contextlib.nullcontext())
    validated, loaded = [], []
    monkeypatch.setattr(gsv_engine, "validate_weights_file", validated.append)

    class _TTS:
        def init_vits_weights(self, path):
            loaded.append(("vits", path))

        def init_t2s_weights(self, path):
            loaded.append(("t2s", path))

    eng = gsv_engine.GSVEngine()
    eng._tts = _TTS()
    weights = tmp_path / "s2G.pth"
    weights.write_bytes(b"v1")

    eng._ensure_weights(str(weights), None)
    eng._ensure_weights(str(weights), None)      # 指纹未变 → 幂等跳过
    assert len(validated) == 1 and len(loaded) == 1

    weights.write_bytes(b"v2-replaced-longer")   # 音色包导入 / 训练产物替换
    eng._ensure_weights(str(weights), None)
    assert len(validated) == 2 and len(loaded) == 2


# ---- 批次87 (SEC-02): 权重安全校验按指纹缓存, 不再每请求整文件反序列化 ----

def test_validate_weights_file_cached_by_stamp(tmp_path, monkeypatch):
    import torch

    from adr.models import gsv_engine

    p = tmp_path / "cached.pth"
    p.write_bytes(b"fake")
    calls = []

    def _fake_load(path, **kwargs):
        calls.append((str(path), kwargs.get("weights_only")))
        return {"w": 1}

    monkeypatch.setattr(torch, "load", _fake_load)
    gsv_engine._WEIGHTS_CHECKED.clear()

    gsv_engine.validate_weights_file(str(p))
    gsv_engine.validate_weights_file(str(p))          # 指纹未变 → 缓存命中
    assert len(calls) == 1 and calls[0][1] is True

    os.utime(p, (1_600_000_000, 1_600_000_000))       # 文件被换掉 → 重新校验
    gsv_engine.validate_weights_file(str(p))
    assert len(calls) == 2
    gsv_engine._WEIGHTS_CHECKED.clear()


# ---- 批次87 (SEC-03): 受限反序列化 —— 白名单外 GLOBAL 一律拒绝 ----

def _weights_zip(payload: bytes) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", payload)
    return buf.getvalue()


class _ForeignGlobal:
    """非白名单类型: pickle GLOBAL 指向本测试模块, 用于验证拒绝/显式放行。"""


_FOREIGN = (__name__, "_ForeignGlobal")


def test_restricted_scan_rejects_foreign_global(tmp_path):
    import pickle

    from adr.models import gsv_engine

    p = tmp_path / "evil.pth"
    p.write_bytes(_weights_zip(pickle.dumps({"x": _ForeignGlobal})))
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        gsv_engine._restricted_weights_scan(str(p))


def test_restricted_scan_allows_explicit_global(tmp_path):
    import pickle

    from adr.models import gsv_engine

    p = tmp_path / "allowed.pth"
    p.write_bytes(_weights_zip(pickle.dumps({"x": _ForeignGlobal})))
    # 白名单内按 stub 化走完, 顶层仍须是 dict
    gsv_engine._restricted_weights_scan(str(p), allow_globals=(_FOREIGN,))


def test_safe_torch_load_roundtrip_and_allow_globals(tmp_path):
    import pickle

    import torch

    from adr.models import gsv_engine
    from adr.training.trainer import TrainerConfig

    plain = tmp_path / "plain.pth"
    torch.save({"w": torch.zeros(2)}, plain)
    got = gsv_engine.safe_torch_load(str(plain))      # 快路径 weights_only=True
    assert isinstance(got, dict) and got["w"].shape == (2,)

    ckpt = tmp_path / "ckpt.pt"
    torch.save({"config": TrainerConfig(epochs=1), "w": torch.zeros(2)}, ckpt)
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        gsv_engine.safe_torch_load(str(ckpt))         # 未放行本仓 dataclass → 拒绝
    got = gsv_engine.safe_torch_load(
        str(ckpt), allow_globals=(("adr.training.trainer", "TrainerConfig"),))
    assert isinstance(got["config"], TrainerConfig) and got["config"].epochs == 1


# ---- 批次89 (#26): 权重预检的截断/截头 zip 与 CUDA OOM 判定 ----

def test_validate_weights_file_rejects_truncated_zip(tmp_path):
    """截断的 zip 权重: 快路径失败 → 慢路径 BadZipFile → 干净 ValueError。"""
    import pickle

    from adr.models import gsv_engine

    raw = _weights_zip(pickle.dumps({"w": 1}))
    p = tmp_path / "truncated.pth"
    p.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(ValueError, match="rejected by safe loader"):
        gsv_engine.validate_weights_file(str(p))


def test_validate_weights_file_heals_headerless_zip(tmp_path):
    """截头 zip (丢 2 字节 'PK'): 慢路径补头后正常放行 (与 GSV 生态自愈一致)。"""
    import torch

    from adr.models import gsv_engine

    p = tmp_path / "plain.pth"
    torch.save({"w": torch.zeros(2)}, p)
    raw = p.read_bytes()
    assert raw[:2] == b"PK"
    headerless = tmp_path / "headerless.pth"
    headerless.write_bytes(raw[2:])
    gsv_engine.validate_weights_file(str(headerless))   # 不抛即通过
    assert gsv_engine.weights_stamp(str(headerless)) in gsv_engine._WEIGHTS_CHECKED


def test_is_cuda_oom_uses_type_name_and_module():
    """真实 torch.cuda.OutOfMemoryError 命中; 同名异模块/普通 RuntimeError 不命中。"""
    import torch

    from adr.models import gsv_engine

    assert gsv_engine.is_cuda_oom(torch.cuda.OutOfMemoryError("out of memory"))
    assert not gsv_engine.is_cuda_oom(RuntimeError("out of memory"))

    class OutOfMemoryError(RuntimeError):   # 同名但非 torch 模块
        pass

    assert not gsv_engine.is_cuda_oom(OutOfMemoryError("boom"))


# ---- 批次89 (#26): bootstrap.ensure 并发只执行一次下载/解压 ----

def test_bootstrap_ensure_serializes_workers(tmp_path, monkeypatch):
    """多线程并发 ensure(): 进程级文件锁串行, 每个 entry 只解一次。"""
    from adr.models import gsv_bootstrap

    marker = tmp_path / "extract" / "marker.dir"
    entry = {
        "key": "unit", "zip_name": "unit.zip", "urls": [],
        "sha256": "", "extract_to": tmp_path / "extract", "marker": marker,
    }
    calls, state = [], {"cur": 0, "max": 0}
    lock = threading.Lock()

    def _fake_acquire(e):
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.15)
        with lock:
            state["cur"] -= 1
        calls.append(e["key"])
        marker.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(gsv_bootstrap, "_MANIFEST", [entry])
    monkeypatch.setattr(gsv_bootstrap, "_acquire", _fake_acquire)
    monkeypatch.setattr(gsv_bootstrap, "pending", lambda: True)
    monkeypatch.setattr(gsv_bootstrap, "GSV_DIR", tmp_path)

    errors = []

    def worker():
        try:
            gsv_bootstrap.ensure()
        except Exception as e:      # pragma: no cover - 失败时把原因带回主线程
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not errors
    assert state["max"] == 1, "并发 ensure() 未被锁串行化"
    assert len(calls) == 1, "marker 就位后不应重复解压"


# ---- 批次122: 强制 cuda 但 CUDA 不可用时必须回落 cpu ----
# CPU 包 + 设置里选 "CUDA" 时, GSV 以 map_location=cfg.device 反序列化权重,
# torch 直接抛 "Attempting to deserialize object on a CUDA device"。

def test_resolve_runtime_device_falls_back_when_cuda_unavailable(monkeypatch):
    import torch
    from adr.models.gsv_engine import resolve_runtime_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_runtime_device("cuda") == "cpu"


def test_resolve_runtime_device_keeps_cuda_when_available(monkeypatch):
    import torch
    from adr.models.gsv_engine import resolve_runtime_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_runtime_device("cuda") == "cuda"


def test_resolve_runtime_device_never_upgrades_cpu(monkeypatch):
    import torch
    from adr.models.gsv_engine import resolve_runtime_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_runtime_device("cpu") == "cpu"
