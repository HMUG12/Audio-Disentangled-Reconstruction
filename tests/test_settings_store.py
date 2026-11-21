"""控制台全局设置存储 (adr/core/settings_store.py) 测试。

重点:
- 默认值与 schema_version
- 部分更新深合并 / 未知字段忽略
- 非法值抛 ValueError 且带字段名, 不静默夹取
- 损坏 JSON / 顶层非对象 → 备份 .bad.<时间戳> 并回落默认
- 原子写不残留 .tmp
- 旧 schema (缺 schema_version) 合并迁移
- get_settings 返回深拷贝
"""
import copy
import json

import pytest

from adr.core import settings_store as ss


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """隔离落盘目录, 绝不写真实 ADR_data。"""
    monkeypatch.setattr(ss, "adr_data_dir", lambda: tmp_path / "data")


def _path(tmp_path):
    return tmp_path / "data" / "settings.json"


def test_defaults_without_file(tmp_path):
    assert ss.get_settings() == ss.DEFAULT_SETTINGS
    assert ss.updated_at() is None
    assert not _path(tmp_path).exists()


def test_default_values_exact():
    s = ss.get_settings()
    assert s["synth"] == {
        "top_k": 15, "top_p": 1.0, "temperature": 1.0, "speed_factor": 1.0,
        "seed": -1, "repetition_penalty": 1.35, "media_type": "wav",
        "streaming_mode": "off"}
    assert s["voice"] == {"default_profile": "", "language": "zh"}
    assert s["ui"] == {"theme": "default", "lang": "zh", "motion": True,
                       "onboarding_done": False}


def test_partial_update_deep_merge(tmp_path):
    out = ss.update_settings({"synth": {"top_k": 30},
                              "ui": {"theme": "classic"}})
    assert out["synth"]["top_k"] == 30
    assert out["synth"]["top_p"] == 1.0        # 未涉及字段保留默认
    assert out["voice"] == ss.DEFAULT_SETTINGS["voice"]
    assert out["ui"]["theme"] == "classic"
    assert out["ui"]["lang"] == "zh"
    # 落盘内容与返回值一致 + schema_version
    data = json.loads(_path(tmp_path).read_text(encoding="utf-8"))
    assert data["schema_version"] == ss.SCHEMA_VERSION
    assert data["synth"]["top_k"] == 30
    # GET 再次读取走文件
    assert ss.get_settings() == out


@pytest.mark.parametrize("patch,field", [
    ({"synth": {"top_k": 0}}, "synth.top_k"),
    ({"synth": {"top_k": 101}}, "synth.top_k"),
    ({"synth": {"top_k": 15.5}}, "synth.top_k"),
    ({"synth": {"top_k": True}}, "synth.top_k"),
    ({"synth": {"top_p": 0.04}}, "synth.top_p"),
    ({"synth": {"temperature": 2.01}}, "synth.temperature"),
    ({"synth": {"speed_factor": 0.49}}, "synth.speed_factor"),
    ({"synth": {"seed": -2}}, "synth.seed"),
    ({"synth": {"seed": 2**31}}, "synth.seed"),
    ({"synth": {"seed": "1"}}, "synth.seed"),
    ({"synth": {"repetition_penalty": 0.99}}, "synth.repetition_penalty"),
    ({"synth": {"media_type": "flac"}}, "synth.media_type"),
    ({"synth": {"streaming_mode": "auto2"}}, "synth.streaming_mode"),
    ({"voice": {"language": "ja"}}, "voice.language"),
    ({"voice": {"default_profile": 7}}, "voice.default_profile"),
    ({"ui": {"theme": "dark"}}, "ui.theme"),
    ({"ui": {"lang": "fr"}}, "ui.lang"),
    ({"ui": {"motion": 1}}, "ui.motion"),
    ({"ui": {"onboarding_done": "true"}}, "ui.onboarding_done"),
    ({"synth": "not-an-object"}, "synth"),
])
def test_invalid_values_raise_with_field(patch, field, tmp_path):
    with pytest.raises(ValueError) as ei:
        ss.update_settings(patch)
    assert field in str(ei.value)
    # 失败不落盘, 下次读取仍是默认值 (不静默夹取)
    assert not _path(tmp_path).exists()
    assert ss.get_settings() == ss.DEFAULT_SETTINGS


@pytest.mark.parametrize("patch", [
    {"synth": {"top_k": 1}},
    {"synth": {"top_k": 100, "top_p": 0.05}},
    {"synth": {"temperature": 0.05, "speed_factor": 2.0}},
    {"synth": {"seed": -1, "repetition_penalty": 1.0}},
    {"synth": {"seed": 2**31 - 1, "media_type": "ogg",
               "streaming_mode": "auto"}},
    {"voice": {"language": "en"}, "ui": {"lang": "en", "motion": False}},
])
def test_boundary_values_accepted(patch):
    ss.update_settings(patch)
    assert ss.get_settings()  # 无异常即通过


def test_unknown_fields_ignored():
    out = ss.update_settings({
        "synth": {"future_param": 7},
        "future_group": {"x": 1},
        "schema_version": 99,
    })
    assert "future_param" not in out["synth"]
    assert "future_group" not in out
    assert out["synth"]["top_k"] == 15


def test_non_dict_patch_raises():
    with pytest.raises(ValueError):
        ss.update_settings(["not", "a", "dict"])


def test_corrupt_json_backed_up_and_defaults(tmp_path):
    path = _path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{ this is not json", encoding="utf-8")

    assert ss.get_settings() == ss.DEFAULT_SETTINGS
    assert not path.exists()
    bad = list(path.parent.glob("settings.json.bad.*"))
    assert len(bad) == 1
    assert bad[0].read_text(encoding="utf-8").startswith("{ this")


def test_top_level_non_object_backed_up(tmp_path):
    path = _path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('["not", "an", "object"]', encoding="utf-8")

    assert ss.get_settings() == ss.DEFAULT_SETTINGS
    assert not path.exists()
    assert len(list(path.parent.glob("settings.json.bad.*"))) == 1


def test_invalid_stored_value_backed_up(tmp_path):
    """文件里已有的非法值同样不静默夹取: 备份 + 回落默认。"""
    path = _path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "synth": {"top_k": 999}}),
                    encoding="utf-8")

    assert ss.get_settings()["synth"]["top_k"] == 15
    assert not path.exists()
    assert len(list(path.parent.glob("settings.json.bad.*"))) == 1


def test_atomic_write_leaves_no_tmp(tmp_path):
    ss.update_settings({"synth": {"top_k": 20}})
    names = [p.name for p in _path(tmp_path).parent.iterdir()]
    assert names == ["settings.json"]
    assert not any(n.endswith(".tmp") for n in names)


def test_reset_restores_defaults(tmp_path):
    ss.update_settings({"ui": {"theme": "classic"},
                        "synth": {"top_k": 88}})
    assert ss.reset_settings() == ss.DEFAULT_SETTINGS
    assert ss.get_settings() == ss.DEFAULT_SETTINGS
    data = json.loads(_path(tmp_path).read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["ui"]["theme"] == "default"
    assert data["synth"]["top_k"] == 15


def test_updated_at_tracks_file(tmp_path):
    assert ss.updated_at() is None
    ss.update_settings({})
    stamp = ss.updated_at()
    assert isinstance(stamp, str) and "T" in stamp


def test_schema_version_missing_merges_defaults(tmp_path):
    """旧文件 (缺 schema_version / 缺字段) 读取时按默认值合并迁移。"""
    path = _path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"synth": {"top_k": 20}}), encoding="utf-8")

    s = ss.get_settings()
    assert s["synth"]["top_k"] == 20
    assert s["synth"]["top_p"] == 1.0
    assert s["voice"]["language"] == "zh"
    # 下次写入补回 schema_version
    ss.update_settings({"ui": {"motion": False}})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == ss.SCHEMA_VERSION


def test_get_settings_returns_deep_copy():
    got = ss.get_settings()
    got["synth"]["top_k"] = 99
    got["ui"]["theme"] = "classic"
    assert ss.get_settings() == ss.DEFAULT_SETTINGS
    assert ss.get_settings()["synth"]["top_k"] == 15


def test_default_settings_not_mutated_by_update():
    before = copy.deepcopy(ss.DEFAULT_SETTINGS)
    ss.update_settings({"synth": {"top_k": 42}})
    assert ss.DEFAULT_SETTINGS == before
