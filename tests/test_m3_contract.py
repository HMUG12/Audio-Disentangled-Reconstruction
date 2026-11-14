"""M3 契约门禁测试 (M3-a 错误码/媒体类型收口 + M3-b 路由断言/流式参数单源)。

覆盖 docs/architecture/refactoring-roadmap.md 的 M3 验收项:
1. wire 错误码单点: 协议文件不再写 code 字面量, 取值全部 ∈ WIRE_CODES;
2. 媒体类型错误在 v2 / native 两面逐字一致 (native 复用 v2 处理链);
3. media_type → Content-Type 判断只在服务层 (SynthesisService.content_type);
4. OpenAI FORMATS 的取值必须是服务层白名单的子集;
5. 启动期路由冲突断言: create_app() 不冲突, 重复注册直接 RuntimeError;
6. CACHE_VERSION bump 后旧 key / 旧盘上文件不再命中;
7. 流式切片三参数只允许 adr/core/settings.py 读 env (协议层禁止裸读)。
"""
from __future__ import annotations

import ast
import json
from collections import OrderedDict
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from adr.server import create_app, openai_compat, tts_cache
from adr.server.app import assert_unique_routes
from adr.server.exceptions import INVALID_PARAMS, WIRE_CODES
from adr.services.synthesis import MEDIA_TYPES, SynthesisService

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "adr" / "server"


# ─── 1. wire 错误码单点 ───

def _code_literals(path: Path) -> list[tuple[int, str]]:
    """取源码中所有 ``{"code": "<字面量>"}`` 的 (行号, 值)。

    用 AST 而非文本匹配: docstring/注释里的示例不属于 wire 取值。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and k.value == "code"
                    and isinstance(v, ast.Constant) and isinstance(v.value, str)):
                out.append((node.lineno, v.value))
    return out


def test_wire_codes_are_centralized():
    """协议模块里的 code 字面量必须都在 WIRE_CODES 表内 (M3-a 收口)。"""
    offenders = []
    for path in sorted(SERVER_DIR.glob("*.py")):
        if path.name == "exceptions.py":  # 单点表本身
            continue
        for lineno, code in _code_literals(path):
            if code not in WIRE_CODES:
                offenders.append(f"{path.name}:{lineno} -> {code!r}")
    assert not offenders, "未登记的 wire 错误码: " + ", ".join(offenders)


def test_wire_codes_unique_values():
    """WIRE_CODES 是 frozenset, 语义键复用时值必须复用 (unsupported_media → invalid_params)。"""
    assert INVALID_PARAMS in WIRE_CODES
    assert "unsupported_media" not in WIRE_CODES


# ─── fixtures (模式对齐 test_error_protocol) ───

class _FakeEngine:
    """恒抛错的占位引擎: 本文件只测校验/契约面, 不进合成。"""

    sr = 32000

    def synthesize(self, *args, **kwargs):
        raise RuntimeError("boom")

    def synthesize_stream(self, *args, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    return TestClient(create_app(engine=_FakeEngine()))


@pytest.fixture
def voice_dir(tmp_path, monkeypatch):
    """假档案库: 仅 demo 档案 (native 面需要 profile 存在才能走到参数校验)。"""
    from adr.models import voice_library
    vd = tmp_path / "voices"
    (vd / "demo").mkdir(parents=True)
    (vd / "demo" / "ref.wav").write_bytes(b"RIFF0000WAVEfmt ")
    (vd / "demo" / "meta.json").write_text(json.dumps({
        "name": "demo", "prompt_text": "参考文本",
        "t2s_weights": None, "vits_weights": None,
        "created_at": "2026-01-01 00:00:00"}), encoding="utf-8")
    monkeypatch.setattr(voice_library, "VOICES_DIR", vd)
    return vd


# ─── 2. 媒体类型错误: 两面同源 ───

def test_media_error_consistent_v2_and_native(client, voice_dir):
    """v2 与 native 对同一非法 media_type 回同一 body (native 复用 tts_handle)。"""
    r_v2 = client.post("/api/v2/tts", json={
        "text": "你好", "ref_audio_path": "a.wav", "media_type": "flac"})
    r_native = client.post("/api/adr/v1/tts", json={
        "text": "你好", "profile": "demo", "media_type": "flac"})
    assert r_v2.status_code == 400
    assert r_native.status_code == 400
    assert r_v2.json() == r_native.json()
    assert r_v2.json()["code"] == INVALID_PARAMS
    assert r_v2.json()["message"].startswith("unsupported media_type: flac")


# ─── 3. 服务层给 Content-Type ───

def test_service_content_type():
    assert SynthesisService.content_type("wav") == "audio/wav"
    assert SynthesisService.content_type("mp3") == "audio/mp3"
    assert SynthesisService.content_type("raw") == "audio/raw"
    # 未知值不抛错, 回通用二进制类型
    assert SynthesisService.content_type("flac") == "application/octet-stream"


# ─── 4. OpenAI 面只请求白名单内的 media_type ───

def test_openai_formats_subset_of_media_types():
    assert set(openai_compat.FORMATS.values()) <= set(MEDIA_TYPES)


# ─── 5. 启动期路由冲突断言 ───

def test_create_app_has_no_route_conflict():
    """native 与 console 共用 /api/adr/v1; 装配期不得有同 method+path 重复。"""
    create_app(engine=_FakeEngine())  # 内部 assert_unique_routes, 冲突会抛


def test_duplicate_route_raises():
    app = FastAPI()
    for _ in range(2):
        router = APIRouter()

        async def _dup():  # pragma: no cover - 不会被调用
            return {}

        router.add_api_route("/dup", _dup, methods=["GET"])
        app.include_router(router)
    with pytest.raises(RuntimeError, match="duplicate route registration"):
        assert_unique_routes(app)


# ─── 6. CACHE_VERSION bump 失效旧缓存 ───

def test_cache_version_bump_invalidates_old_key(tmp_path, monkeypatch):
    req = {"text": "你好", "ref_audio_path": "a.wav", "media_type": "wav",
           "speed_factor": 1.0, "fragment_interval": 0.3}

    old_key = tts_cache.make_key(req)
    old_file = tmp_path / f"{tts_cache._PREFIX}{old_key}.wav"
    old_file.write_bytes(b"old-audio")

    new_version = tts_cache.CACHE_VERSION + 1
    monkeypatch.setattr(tts_cache, "CACHE_VERSION", new_version)
    monkeypatch.setattr(tts_cache, "_PREFIX", f"v{new_version}-")

    new_key = tts_cache.make_key(req)
    assert new_key != old_key                      # key 参与哈希
    assert tts_cache._fname(new_key, "wav").startswith(f"v{new_version}-")

    # 扫描: 旧版本文件被丢弃, 且不登记进内存 LRU
    monkeypatch.setattr(tts_cache, "_cache", OrderedDict())
    monkeypatch.setattr(tts_cache, "_total_bytes", 0)
    monkeypatch.setattr(tts_cache, "_loaded", False)
    monkeypatch.setattr(tts_cache, "_dir", lambda: tmp_path)
    tts_cache._load_disk_once()

    assert old_key not in tts_cache._cache
    assert not old_file.exists()


# ─── 7. 流式切片参数: env 只在 settings.py 读 ───

_STREAM_ENV_KEYS = {
    "ADR_TTS_STREAM_HEAD", "ADR_TTS_SPLIT_MAX", "ADR_TTS_FRAGMENT_INTERVAL",
}


def _env_read_keys(path: Path) -> set[str]:
    """收集文件里通过 os.environ.get / os.getenv / os.environ[...] 读取的键。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            is_environ_get = (
                isinstance(fn, ast.Attribute) and fn.attr == "get"
                and isinstance(fn.value, ast.Attribute)
                and fn.value.attr == "environ")
            is_getenv = isinstance(fn, ast.Attribute) and fn.attr == "getenv"
            if (is_environ_get or is_getenv) and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(arg.value)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr == "environ":
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    found.add(sl.value)
    return found & _STREAM_ENV_KEYS


def test_stream_split_params_only_read_in_settings():
    """stream_head / split_max / fragment_interval 只允许 settings.py 读 env。"""
    settings_py = (REPO_ROOT / "adr" / "core" / "settings.py").resolve()
    offenders = []
    for path in sorted((REPO_ROOT / "adr").rglob("*.py")):
        if path.resolve() == settings_py:
            continue
        keys = _env_read_keys(path)
        if keys:
            offenders.append(f"{path.relative_to(REPO_ROOT)} -> {sorted(keys)}")
    assert not offenders, "协议/引擎层裸读流式切片 env: " + "; ".join(offenders)
