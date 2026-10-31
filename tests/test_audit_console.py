"""批次66 审计回归: console 训练控制面 / 上传探测 / 建档日志。"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from collections import deque

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adr.server import console


class FakeProc:
    def __init__(self, lines=(), rc=0, alive=False, term_raises=False):
        self.stdout = iter(lines)
        self.rc = rc
        self.alive = alive
        self.term_raises = term_raises
        self.terminated = self.killed = False
        self.pid = 4242

    def poll(self):
        return None if self.alive else self.rc

    def wait(self, timeout=None):
        if self.alive:
            raise subprocess.TimeoutExpired("x", timeout or 0)
        return self.rc

    def terminate(self):
        self.terminated = True
        if self.term_raises:
            raise PermissionError("denied")
        self.alive = False

    def kill(self):
        self.killed = True
        if self.term_raises:
            raise PermissionError("denied")
        self.alive = False


@pytest.fixture(autouse=True)
def _iso_train(tmp_path, monkeypatch):
    """_train 是模块级全局: 用例前后保存/恢复, 并隔离输出目录。"""
    saved = {**console._train, "logs": deque(console._train["logs"], maxlen=400)}
    monkeypatch.setattr(console, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(console, "UPLOAD_DIR", tmp_path / "output" / "uploads")
    console._train.update(state="idle", proc=None, gate=None, exp="",
                          exit_code=None, bind_voice="", reader=None)
    real_kill_tree = console._kill_tree

    def safe_kill_tree(p, graceful=False):  # FakeProc 的 pid 是假的, 绝不能 taskkill
        if isinstance(p, FakeProc):
            return p.terminate() if graceful else p.kill()
        return real_kill_tree(p, graceful)

    monkeypatch.setattr(console, "_kill_tree", safe_kill_tree)
    yield
    rid = max(console._train["run_id"], saved["run_id"]) + 1
    console._train.clear()
    console._train.update(saved, run_id=rid)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    app = FastAPI()
    app.include_router(console.router)
    return TestClient(app)


def _audio(tmp_path):
    p = tmp_path / "in.wav"
    p.write_bytes(b"RIFF0000WAVEfmt ")
    return str(p)


# ─── #12 / #29: 旧 reader 不碰新一轮 ───

def test_stale_reader_does_not_touch_new_run():
    new_gate = FakeProc(alive=True)
    old_gate = FakeProc(alive=True)
    console._train.update(state="running", proc=FakeProc(alive=True),
                          gate=new_gate, exit_code=None, run_id=7)
    console._train["logs"].clear()
    console._train_reader(FakeProc(lines=["old line\n"], rc=1), old_gate, run_id=6)
    assert console._train["state"] == "running"
    assert console._train["exit_code"] is None
    assert console._train["gate"] is new_gate and not new_gate.terminated
    assert old_gate.terminated                      # 只收自己的门禁
    assert "old line" not in list(console._train["logs"])


def test_reader_finishes_own_run():
    gate = FakeProc(alive=True)
    console._train.update(state="running", gate=gate, run_id=3)
    console._train_reader(FakeProc(lines=["a\n"], rc=0), gate, run_id=3)
    assert console._train["state"] == "done"
    assert console._train["exit_code"] == 0
    assert console._train["gate"] is None and gate.terminated


def test_start_rejected_while_stopped_proc_still_alive(client, tmp_path):
    console._train.update(state="stopped", proc=FakeProc(alive=True))
    r = client.post("/api/adr/v1/train/start",
                    json={"audio_path": _audio(tmp_path), "gate_on": False})
    assert r.status_code == 409


# ─── #13: stop 失败不能报 stopped ───

def test_stop_failure_reports_error_and_keeps_running(client):
    p = FakeProc(alive=True, term_raises=True)
    console._train.update(state="running", proc=p, run_id=1)
    r = client.post("/api/adr/v1/train/stop")
    assert r.status_code == 500
    assert console._train["state"] == "running"


def test_stop_success(client):
    p = FakeProc(alive=True)
    console._train.update(state="running", proc=p, run_id=1)
    r = client.post("/api/adr/v1/train/stop")
    assert r.status_code == 200 and r.json() == {"state": "stopped"}
    assert p.terminated and console._train["state"] == "stopped"


# ─── #14: TRAIN_TIMEOUT 真正生效 ───

def test_train_timeout_kills_hung_process(monkeypatch):
    monkeypatch.setattr(console, "TRAIN_TIMEOUT", 0.3)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            stdout=subprocess.PIPE, text=True)
    console._train.update(state="running", proc=proc, run_id=5)
    t0 = time.time()
    console._train_reader(proc, None, run_id=5)
    assert time.time() - t0 < 30
    assert console._train["state"] == "failed"
    assert any("[timeout]" in ln for ln in console._train["logs"])


# ─── #15: 门禁启动失败可见 / 日志句柄关闭 / 回滚收割 ───

def _fake_popen(monkeypatch, gate_fail=False, train_fail=False, seen=None):
    seen = seen if seen is not None else {}

    def fake(cmd, **kw):
        if any("train_gate.py" in str(c) for c in cmd):
            if gate_fail:
                raise OSError("no interpreter")
            seen["gate_fh"] = kw.get("stdout")
            seen["gate"] = FakeProc(alive=True)
            return seen["gate"]
        if train_fail:
            raise OSError("spawn failed")
        return FakeProc(lines=[], rc=0)

    monkeypatch.setattr(console.subprocess, "Popen", fake)
    return seen


def _wait_idle():
    for _ in range(100):
        if console._train["state"] != "running":
            return
        time.sleep(0.02)


def test_gate_spawn_failure_logged(client, tmp_path, monkeypatch, caplog):
    _fake_popen(monkeypatch, gate_fail=True)
    with caplog.at_level(logging.WARNING, logger="adr.server.console"):
        r = client.post("/api/adr/v1/train/start",
                        json={"audio_path": _audio(tmp_path), "gate_on": True})
    assert r.status_code == 200
    assert set(r.json()) == {"state", "exp", "epochs", "recipe"}
    _wait_idle()
    assert any("门禁启动失败" in ln for ln in console._train["logs"])
    assert "门禁启动失败" in caplog.text


def test_gate_log_handle_closed(client, tmp_path, monkeypatch):
    seen = _fake_popen(monkeypatch)
    r = client.post("/api/adr/v1/train/start",
                    json={"audio_path": _audio(tmp_path), "gate_on": True})
    assert r.status_code == 200
    _wait_idle()
    assert seen["gate_fh"].closed


def test_train_spawn_failure_reaps_gate(client, tmp_path, monkeypatch):
    seen = _fake_popen(monkeypatch, train_fail=True)
    r = client.post("/api/adr/v1/train/start",
                    json={"audio_path": _audio(tmp_path), "gate_on": True})
    assert r.status_code == 500
    assert seen["gate"].terminated and seen["gate"].poll() is not None
    assert seen["gate_fh"].closed
    assert console._train["state"] != "running"


# ─── #16: load_voice 错误分类 ───

def _resume(client, tmp_path, name):
    return client.post("/api/adr/v1/train/start", json={
        "audio_path": _audio(tmp_path), "gate_on": False, "resume_from": name})


def test_resume_corrupt_meta_not_reported_missing(client, tmp_path, monkeypatch):
    from adr.models import voice_library
    vd = tmp_path / "voices"
    (vd / "坏档").mkdir(parents=True)
    (vd / "坏档" / "meta.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(voice_library, "VOICES_DIR", vd)
    r = _resume(client, tmp_path, "坏档")
    assert r.status_code == 400
    assert "损坏" in r.json()["detail"]
    assert "不存在" not in r.json()["detail"]


def test_resume_missing_still_not_found(client, tmp_path, monkeypatch):
    from adr.models import voice_library
    monkeypatch.setattr(voice_library, "VOICES_DIR", tmp_path / "voices")
    r = _resume(client, tmp_path, "幽灵")
    assert r.status_code == 400 and "续训档案不存在" in r.json()["detail"]


def test_resume_invalid_name(client, tmp_path, monkeypatch):
    from adr.models import voice_library
    monkeypatch.setattr(voice_library, "VOICES_DIR", tmp_path / "voices")
    r = _resume(client, tmp_path, "..")
    assert r.status_code == 400 and "非法" in r.json()["detail"]


# ─── #18: _probe_audio 异常归一为 ValueError → 上传 400 且删文件 ───

def test_probe_timeout_becomes_value_error(tmp_path, monkeypatch):
    p = tmp_path / "x.m4a"
    p.write_bytes(b"garbage")
    monkeypatch.setattr(console.shutil, "which", lambda n: "ffprobe")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("ffprobe", 30)

    monkeypatch.setattr(console.subprocess, "run", boom)
    with pytest.raises(ValueError):
        console._probe_audio(p)


def test_probe_missing_soundfile_falls_back(tmp_path, monkeypatch):
    p = tmp_path / "x.wav"
    p.write_bytes(b"garbage")
    monkeypatch.setitem(sys.modules, "soundfile", None)  # import → ImportError
    monkeypatch.setattr(console.shutil, "which", lambda n: None)
    monkeypatch.setattr(console, "OUTPUT_DIR", tmp_path / "a" / "b" / "output")
    assert console._probe_audio(p)["via"] == "none"


def test_upload_probe_oserror_is_400_and_cleans(client, tmp_path, monkeypatch):
    monkeypatch.setattr(console.shutil, "which", lambda n: "ffprobe")

    def boom(*a, **k):
        raise OSError("exec format error")

    monkeypatch.setattr(console.subprocess, "run", boom)
    r = client.post("/api/adr/v1/train/upload",
                    files={"file": ("a.m4a", b"not audio", "audio/mp4")})
    assert r.status_code == 400
    up = tmp_path / "output" / "uploads"
    assert not up.exists() or not any(up.iterdir())


# ─── #19: 建档失败保留堆栈 ───

def test_profile_worker_logs_traceback(monkeypatch, caplog):
    from adr.models import voice_library

    def boom(*a, **k):
        raise RuntimeError("engine OOM")

    monkeypatch.setattr(voice_library, "save_voice_auto", boom)
    with caplog.at_level(logging.ERROR, logger="adr.server.console"):
        console._profile_worker("n", "a.wav", "t", "")
    assert console._profile_job["state"] == "failed"
    assert any(r.exc_info and "engine OOM" in str(r.exc_info[1])
               for r in caplog.records)


# ─── 批次66: #14/#29 major — 看门狗 + stop 整树杀, 孙进程不拖住管道 ───

def test_timeout_kills_grandchild_and_reader_finishes(monkeypatch, tmp_path):
    """批次66: 看门狗必须整树杀 — 父 Popen(shell=True) 孙, 只杀父时孙仍握管道
    reader 等不到 EOF。此测真起孙进程验证整树 kill 能让 reader 完成。"""
    monkeypatch.setattr(console, "TRAIN_TIMEOUT", 1.0)
    monkeypatch.setattr(console, "OUTPUT_DIR", tmp_path / "output")
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', 'import time;time.sleep(60)'], shell=True)\n",
        encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    console._train.update(state="running", proc=proc, run_id=8)
    t0 = time.time()
    console._train_reader(proc, None, run_id=8)
    elapsed = time.time() - t0
    # 阈值 30s: 全量套件负载下实测 16.3s (spawn+taskkill 慢), 单跑 ~3s;
    # 判别本质是 "不等孙进程自然退出 (60s)" — 30s 两侧余量充足
    assert elapsed < 30, "reader 不该等孙进程自然退出 (60s)"
    assert console._train["state"] == "failed"
    assert any("[timeout]" in ln for ln in console._train["logs"])


def test_train_stop_kills_tree_not_only_parent(monkeypatch, tmp_path, client):
    """批次66: /train/stop 必须整树杀, 不然孙进程继续占卡且 reader 等不到 EOF;
    start 会拒绝 (reader 仍 alive)。真起孙进程验证。"""
    monkeypatch.setattr(console, "OUTPUT_DIR", tmp_path / "output")
    script = tmp_path / "p.py"
    script.write_text(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', 'import time;time.sleep(90)'], shell=True)\n",
        encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    reader = threading.Thread(target=console._train_reader, args=(proc, None, 9),
                              daemon=True)
    console._train.update(state="running", proc=proc, run_id=9, reader=reader)
    reader.start()
    time.sleep(0.1)
    r = client.post("/api/adr/v1/train/stop")
    assert r.status_code == 200
    # 20s: stop 整树杀正常亚秒级收尾, 全量负载下放宽 (实测 <5s);
    # 孙进程 90s 自然退出 — 杀树失效时 reader 仍不会在 20s 内结束, 断言保持判别力
    reader.join(timeout=20)
    assert not reader.is_alive(), "reader 该在 stop 后收尾, 不该等孙进程自然退"
    assert console._train["state"] == "stopped"


def test_start_rejected_while_reader_alive(client, tmp_path, monkeypatch):
    """批次66: reader 活着 → 仍有孙进程可能握管道/占卡 → 拒绝新训练。"""
    monkeypatch.setattr(console, "OUTPUT_DIR", tmp_path / "output")
    alive_thread = threading.Thread(target=lambda: time.sleep(10), daemon=True)
    alive_thread.start()
    console._train.update(state="idle", proc=None, reader=alive_thread)
    r = client.post("/api/adr/v1/train/start",
                    json={"audio_path": _audio(tmp_path), "gate_on": False})
    assert r.status_code == 409
    alive_thread.join(timeout=0.01)  # 快速清场


# ─── 批次66: minor — TRAIN_TIMEOUT 环境变量解析健壮 ───

def test_timeout_env_bad_value_fallback(monkeypatch, caplog):
    """批次66: ADR_TRAIN_TIMEOUT=2h 等非法值 → 回落默认值并告警, 不让导入失败。"""
    with caplog.at_level(logging.WARNING, logger="adr.server.console"):
        v = console._parse_timeout("2h")
    assert v == 7200.0
    assert "非法" in caplog.text


def test_timeout_env_negative_fallback(caplog):
    with caplog.at_level(logging.WARNING, logger="adr.server.console"):
        v = console._parse_timeout("-1")
    assert v == 7200.0
    assert "非法" in caplog.text


def test_timeout_env_zero_fallback(caplog):
    with caplog.at_level(logging.WARNING, logger="adr.server.console"):
        v = console._parse_timeout("0")
    assert v == 7200.0


def test_timeout_env_valid_returns_value():
    assert console._parse_timeout("3600") == 3600.0
    assert console._parse_timeout(None) == 7200.0


# ─── 批次66: minor — timeout 日志写在 kill 之后 ───

def test_timeout_log_after_kill_success(monkeypatch):
    """批次66: 「已强杀」写在 kill 成功后, 不在 kill 之前承诺。"""
    monkeypatch.setattr(console, "TRAIN_TIMEOUT", 0.2)
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"],
                            stdout=subprocess.PIPE, text=True, start_new_session=True)
    console._train.update(state="running", proc=proc, run_id=11)
    console._train_reader(proc, None, run_id=11)
    logs = list(console._train["logs"])
    assert any("已强杀" in ln for ln in logs), "成功 kill 后该写「已强杀」"


def test_timeout_log_kill_failure(monkeypatch):
    """批次66: kill 抛异常时日志写「强杀失败」, 不写「已强杀」承诺。"""
    monkeypatch.setattr(console, "TRAIN_TIMEOUT", 0.2)

    class Hung(FakeProc):
        def __init__(self):
            super().__init__(rc=1, alive=True, term_raises=True)
            self.stdout = self._lines()

        def _lines(self):  # 惰性生成器: 读时卡 0.6s 后 EOF, 期间看门狗触发
            time.sleep(0.6)
            self.alive = False
            yield from ()

        def wait(self, timeout=None):
            return self.rc

    console._train.update(state="running", proc=None, run_id=12)
    console._train_reader(Hung(), None, run_id=12)
    logs = list(console._train["logs"])
    assert not any("已强杀" in ln for ln in logs), "kill 失败不该写「已强杀」"
    assert any("强杀失败" in ln for ln in logs)
