"""文档中心 API 回归 (2026-11-14 深审 A1/A6 补测)。

覆盖三件事:
1. 白名单 8 篇在源码树里全部可读 (文档改名/移路径会在这里红, 而不是装完才发现)
2. 非白名单 slug 一律 404, 不做路径穿越
3. 安装包漏拷 docs/ 时 /check 必须给出 local=0 的可识别信号, 而不是"已是最新"
"""
import pytest
from fastapi.testclient import TestClient

from adr.server import docs_api
from adr.server.app import create_app
from adr.server.docs_api import DOC_CATALOG


class _Engine:
    sr = 32000


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("ADR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ADR_TTS_CACHE", "0")
    monkeypatch.delenv("ADR_TTS_API_KEY", raising=False)
    monkeypatch.delenv("ADR_ALLOW_QUERY_API_KEY", raising=False)
    return TestClient(create_app(engine=_Engine()))


def test_index_lists_whole_whitelist(client):
    r = client.get("/api/adr/v1/docs")
    assert r.status_code == 200
    body = r.json()
    assert {d["slug"] for d in body["docs"]} == set(DOC_CATALOG)
    for d in body["docs"]:
        assert d["title"] and d["group"] and d["summary"]
        assert d["bytes"] > 0
    assert body["online"] is True


@pytest.mark.parametrize("slug", sorted(DOC_CATALOG))
def test_every_slug_reads_locally(client, slug):
    r = client.get(f"/api/adr/v1/docs/{slug}")
    assert r.status_code == 200
    body = r.json()
    assert body["slug"] == slug
    assert body["source"] == "local"
    assert body["markdown"].strip()
    assert body["title"].strip()


@pytest.mark.parametrize("slug", ["nope", "..%2FREADME.md", "README", "docs"])
def test_unknown_or_traversal_slug_is_404(client, slug):
    assert client.get(f"/api/adr/v1/docs/{slug}").status_code == 404


def test_check_degrades_to_offline(monkeypatch, client):
    """官方仓库不可达 → online=False + error, 绝不抛 500。"""
    monkeypatch.setenv("ADR_DOCS_ONLINE_BASE", "http://127.0.0.1:9/")
    r = client.get("/api/adr/v1/docs/check")
    assert r.status_code == 200
    body = r.json()
    assert body["online"] is False
    assert body["updates"] == []
    # 提前在断网处返回, local 只统计到已读到的篇数 (1..8), 不是全量
    assert 1 <= body["local"] <= len(DOC_CATALOG)
    assert body["error"]


def test_check_reports_missing_bundled_docs(monkeypatch, tmp_path, client):
    """「装完没有文档」的可识别信号 (深审 A1): local=0, 前端据此提示未随包。"""
    monkeypatch.setattr(docs_api, "REPO", tmp_path)
    body = client.get("/api/adr/v1/docs/check").json()
    assert body["local"] == 0
    assert body["checked"] == 0
    assert body["updates"] == []
    # 索引同样应为空, 而不是报错
    assert client.get("/api/adr/v1/docs").json()["docs"] == []
