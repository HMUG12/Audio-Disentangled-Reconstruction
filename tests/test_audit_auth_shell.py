"""批次66 审计回归: auth_shell 组 (P2 / P23 / P24 壳与 call 页 key 透传)。"""
import re
from pathlib import Path

import pytest

from adr.core import settings
from adr.server import __main__ as server_main

REPO = Path(__file__).resolve().parents[1]
CALL_HTML = REPO / "adr" / "server" / "static" / "call.html"
LIB_RS = REPO / "desktop" / "src-tauri" / "src" / "lib.rs"
LAUNCHER = REPO / "desktop" / "src" / "launcher.html"


# ===== P2: has_api_key 与 api_keys 同口径, 全空 key 时 0.0.0.0 警告必须打 =====
@pytest.mark.parametrize("raw", [" , ", ",", " ,, , ", "   "])
def test_has_api_key_agrees_with_api_keys_blank(monkeypatch, raw):
    monkeypatch.setenv("ADR_TTS_API_KEY", raw)
    assert settings.api_keys() == []
    assert settings.has_api_key() is False


def test_warn_fires_when_all_keys_blank(monkeypatch, capsys):
    monkeypatch.setenv("ADR_TTS_API_KEY", " , ")
    server_main._warn_unauthenticated_listen("0.0.0.0")
    assert "安全警告" in capsys.readouterr().err


def test_warn_silent_with_real_key(monkeypatch, capsys):
    monkeypatch.setenv("ADR_TTS_API_KEY", " , k1")
    assert settings.has_api_key() is True
    server_main._warn_unauthenticated_listen("0.0.0.0")
    assert "安全警告" not in capsys.readouterr().err


# ===== P23: 壳预热轮询走免鉴权的 /health =====
def test_shell_prewarm_polls_health():
    src = LIB_RS.read_text(encoding="utf-8")
    fn = src[src.index("fn fetch_engine_stage_info"):]
    fn = fn[: fn.index("\n}\n")]
    assert '"{base}/api/adr/v1/health"' in fn


# ===== P24: call.html 透传 api_key =====
def test_call_html_reads_and_forwards_key():
    src = CALL_HTML.read_text(encoding="utf-8")
    assert 'new URLSearchParams(location.search).get("api_key")' in src
    # 所有 API fetch 都经 api() 包装 (无裸路径 / 裸变量)
    assert not re.findall(r'fetch\("/api/', src)
    assert "fetch(url" not in src
    assert 'fetch(api("/api/adr/v1/profiles"))' in src
    assert "fetch(api(url)" in src


def test_call_page_api_calls_pass_with_query_key(monkeypatch, tmp_path):
    """call.html 用 ?api_key= 透传 — 服务端须接受该形态 (与 pro/easy 同口径)。"""
    from fastapi.testclient import TestClient
    from adr.server.app import create_app

    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ADR_TTS_CACHE", "0")
    monkeypatch.setenv("ADR_TTS_API_KEY", "k1, k2")

    class _Engine:
        sr = 32000

    c = TestClient(create_app(engine=_Engine()))
    assert c.get("/api/adr/v1/profiles").status_code == 401
    assert c.get("/api/adr/v1/profiles?api_key=k1").status_code == 200


# ===== P24: 静态脚本 (<script src>) 启用 key 后仍免鉴权, 否则页面配色/语言瘫痪 =====
def test_static_scripts_exempt_with_key_set(monkeypatch, tmp_path):
    """theme.js / i18n.js 经 <script src> 加载, 浏览器不带 key; 必须免鉴权。"""
    from fastapi.testclient import TestClient
    from adr.server.app import create_app

    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ADR_TTS_CACHE", "0")
    monkeypatch.setenv("ADR_TTS_API_KEY", "k1")

    class _Engine:
        sr = 32000

    c = TestClient(create_app(engine=_Engine()))
    # 设了 key 时业务接口仍需凭据
    assert c.get("/api/adr/v1/profiles").status_code == 401
    # 纯静态脚本放行 (不是 401)
    assert c.get("/theme.js").status_code == 200
    assert c.get("/i18n.js").status_code == 200


def test_theme_js_is_exempt_unit():
    from adr.server.auth import _is_exempt

    assert _is_exempt("/theme.js") is True
    assert _is_exempt("/api/adr/v1/profiles") is False


# ===== P24: 壳导航控制台页带 api_key; launcher 拉档案带 key =====
def test_shell_navigation_appends_key():
    src = LIB_RS.read_text(encoding="utf-8")
    assert "fn console_key_query()" in src and "fn key_query(raw: &str)" in src
    assert "console_key_query()" in src[src.index("async fn enter_console"):src.index("fn start_server(")]
    assert '"&api_key=' in src or "&api_key={enc}" in src


def test_launcher_profiles_fetch_forwards_key():
    src = LAUNCHER.read_text(encoding="utf-8")
    assert '"?api_key=" + encodeURIComponent(key)' in src
    assert 'fillProfiles(lan?.url || "", cfg.api_key)' in src
