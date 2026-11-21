"""控制台全局设置 API (/api/adr/v1/settings) 测试。

覆盖: GET 结构完整 / PUT 部分更新 / 非法值 400 / 未知字段忽略 /
损坏 JSON 备份回落 / DELETE 重置 / 鉴权沿用现有中间件。
"""
import json

import pytest
from fastapi.testclient import TestClient

from adr.core import settings_store
from adr.server import create_app


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """隔离落盘目录 + 默认不鉴权 (与 test_server_api 的 client 口径一致)。"""
    monkeypatch.setattr(settings_store, "adr_data_dir", lambda: tmp_path / "data")
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)


@pytest.fixture
def client():
    # 注入假引擎跳过预热线程; 本组用例不触达引擎
    return TestClient(create_app(engine=object()))


def _path(tmp_path):
    return tmp_path / "data" / "settings.json"


def test_get_defaults_structure(client):
    r = client.get("/api/adr/v1/settings")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"schema_version", "synth", "voice", "ui", "security", "updated_at"}
    assert body["schema_version"] == 1
    assert body["updated_at"] is None            # 无文件
    assert body["synth"]["top_k"] == 15
    assert body["synth"]["streaming_mode"] == "off"
    assert body["voice"] == {"default_profile": "", "language": "zh"}
    assert body["ui"] == {"theme": "default", "lang": "zh", "motion": True,
                          "onboarding_done": False}


def test_put_partial_deep_merge(client, tmp_path):
    r = client.put("/api/adr/v1/settings",
                   json={"synth": {"top_k": 30}, "ui": {"theme": "classic"}})
    assert r.status_code == 200
    body = r.json()
    assert body["synth"]["top_k"] == 30
    assert body["synth"]["temperature"] == 1.0   # 未涉及字段保持默认
    assert body["ui"]["theme"] == "classic"
    assert body["ui"]["lang"] == "zh"
    assert isinstance(body["updated_at"], str)
    # 已落盘, GET 一致
    assert client.get("/api/adr/v1/settings").json()["synth"]["top_k"] == 30
    assert _path(tmp_path).exists()


@pytest.mark.parametrize("payload,field", [
    ({"synth": {"top_k": 0}}, "top_k"),
    ({"synth": {"top_p": 2.0}}, "top_p"),
    ({"synth": {"temperature": 0.01}}, "temperature"),
    ({"synth": {"speed_factor": 3}}, "speed_factor"),
    ({"synth": {"seed": -5}}, "seed"),
    ({"synth": {"repetition_penalty": 2.5}}, "repetition_penalty"),
    ({"synth": {"media_type": "flac"}}, "media_type"),
    ({"synth": {"streaming_mode": "yes"}}, "streaming_mode"),
    ({"voice": {"language": "ja"}}, "language"),
    ({"ui": {"theme": "neon"}}, "theme"),
    ({"ui": {"motion": "yes"}}, "motion"),
])
def test_put_invalid_400(client, payload, field):
    r = client.put("/api/adr/v1/settings", json=payload)
    assert r.status_code == 400
    body = r.json()
    assert field in body["detail"]
    assert body["message"] == body["detail"]     # 全站统一 message 字段
    # 非法更新不生效
    assert client.get("/api/adr/v1/settings").json()["synth"]["top_k"] == 15


def test_put_non_object_body_400(client):
    r = client.put("/api/adr/v1/settings", json=["not", "an", "object"])
    assert r.status_code == 400
    assert "JSON 对象" in r.json()["detail"]


def test_put_group_not_object_400(client):
    r = client.put("/api/adr/v1/settings", json={"synth": "wav"})
    assert r.status_code == 400
    assert "synth" in r.json()["detail"]


def test_put_unknown_fields_ignored(client):
    r = client.put("/api/adr/v1/settings",
                   json={"synth": {"future_param": 1}, "future_group": {}})
    assert r.status_code == 200
    body = r.json()
    assert "future_param" not in body["synth"]
    assert "future_group" not in body


def test_corrupt_file_backed_up_then_defaults(client, tmp_path):
    assert client.put("/api/adr/v1/settings",
                      json={"ui": {"theme": "classic"}}).status_code == 200
    path = _path(tmp_path)
    path.write_text("{ broken json", encoding="utf-8")

    body = client.get("/api/adr/v1/settings").json()
    assert body["ui"]["theme"] == "default"
    assert body["updated_at"] is None            # 坏文件已备份, 无现役文件
    assert not path.exists()
    assert len(list(path.parent.glob("settings.json.bad.*"))) == 1


def test_delete_resets(client, tmp_path):
    client.put("/api/adr/v1/settings", json={"synth": {"top_k": 77}})
    r = client.delete("/api/adr/v1/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["synth"] == settings_store.DEFAULT_SETTINGS["synth"]
    assert body["voice"] == settings_store.DEFAULT_SETTINGS["voice"]
    assert body["ui"] == settings_store.DEFAULT_SETTINGS["ui"]
    assert client.get("/api/adr/v1/settings").json()["synth"]["top_k"] == 15
    data = json.loads(_path(tmp_path).read_text(encoding="utf-8"))
    assert data["schema_version"] == 1


def test_settings_require_api_key(tmp_path, monkeypatch):
    """鉴权沿用现有中间件: 设 key 后无凭据 401, 带 key 正常。"""
    monkeypatch.setenv("ADR_TTS_API_KEY", "k1")
    c = TestClient(create_app(engine=object()))
    assert c.get("/api/adr/v1/settings").status_code == 401
    assert c.put("/api/adr/v1/settings",
                 json={"synth": {"top_k": 20}}).status_code == 401

    h = {"X-API-Key": "k1"}
    body = c.get("/api/adr/v1/settings", headers=h).json()
    assert body["synth"]["top_k"] == 15
    r = c.put("/api/adr/v1/settings", json={"synth": {"top_k": 20}}, headers=h)
    assert r.status_code == 200 and r.json()["synth"]["top_k"] == 20
    assert c.delete("/api/adr/v1/settings", headers=h).status_code == 200


def test_settings_page_served(client):
    """独立设置页 /settings 同源静态返回, 并带上页面标识与运行时依赖。"""
    r = client.get("/settings")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'data-adr-page="settings"' in r.text
    for src in ("/theme.js", "/i18n.js", "/i18n-settings.js",
                "/app-shell.css", "/app-shell.js"):
        assert src in r.text


def test_settings_i18n_dict_served(client):
    r = client.get("/i18n-settings.js")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/javascript")
    assert "settings.save" in r.text
    assert "settings.err_401" in r.text


def test_settings_page_and_dict_exempt_from_api_key(tmp_path, monkeypatch):
    """启用 API key 后静态页/语言包仍可加载 (脚本子请求不带凭据), 数据接口仍鉴权。"""
    monkeypatch.setenv("ADR_TTS_API_KEY", "k1")
    c = TestClient(create_app(engine=object()))
    assert c.get("/settings").status_code == 200
    assert c.get("/i18n-settings.js").status_code == 200
    assert c.get("/api/adr/v1/settings").status_code == 401

