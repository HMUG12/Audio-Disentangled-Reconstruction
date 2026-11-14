"""批次96: gsv_bootstrap 解压原子落位 (同卷临时目录 + rename)。

覆盖点:
- 成功解压后载荷在最终路径, 且不留 `.extract-*` / `.old-*` 残迹;
- 已存在的同名目录被整体替换而非合并;
- 老版 G2PW 包 G2PWModel_1.1 → G2PWModel 兼容改名仍在;
- 解压失败不污染目标目录 (marker 不出现, 临时目录被清理)。
"""
import zipfile

import pytest

from adr.models import gsv_bootstrap


def _make_zip(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _entry(tmp_path, *, zip_name="tiny.zip", marker_rel="pkg"):
    extract_to = tmp_path / "target"
    return {
        "key": "k",
        "label": "l",
        "zip_name": zip_name,
        "urls": {"hf": "http://x"},
        "extract_to": extract_to,
        "marker": extract_to / marker_rel,
        "sha256": "",
    }


def _leftovers(extract_to):
    return ([p.name for p in extract_to.glob(".extract-*")]
            + [p.name for p in extract_to.glob("*.old-*")])


def test_extract_lands_payload_and_cleans_temp(tmp_path):
    entry = _entry(tmp_path)
    _make_zip(entry["extract_to"] / entry["zip_name"],
              {"pkg/keep.txt": b"ok", "top.txt": b"hi"})
    gsv_bootstrap._extract(entry)
    assert (entry["extract_to"] / "pkg" / "keep.txt").read_bytes() == b"ok"
    assert (entry["extract_to"] / "top.txt").read_bytes() == b"hi"
    assert _leftovers(entry["extract_to"]) == []
    assert not (entry["extract_to"] / entry["zip_name"]).exists()


def test_extract_replaces_existing_dir_not_merge(tmp_path):
    entry = _entry(tmp_path)
    stale = entry["extract_to"] / "pkg"
    stale.mkdir(parents=True)
    (stale / "stale.txt").write_bytes(b"old")
    _make_zip(entry["extract_to"] / entry["zip_name"], {"pkg/keep.txt": b"new"})
    gsv_bootstrap._extract(entry)
    assert (entry["extract_to"] / "pkg" / "keep.txt").read_bytes() == b"new"
    assert not (entry["extract_to"] / "pkg" / "stale.txt").exists()
    assert _leftovers(entry["extract_to"]) == []


def test_extract_renames_legacy_g2pw_package(tmp_path):
    entry = _entry(tmp_path, marker_rel="G2PWModel")
    _make_zip(entry["extract_to"] / entry["zip_name"],
              {"G2PWModel_1.1/config.py": b"x"})
    gsv_bootstrap._extract(entry)
    assert (entry["extract_to"] / "G2PWModel" / "config.py").read_bytes() == b"x"
    assert not (entry["extract_to"] / "G2PWModel_1.1").exists()
    assert _leftovers(entry["extract_to"]) == []


def test_extract_failure_leaves_target_untouched(tmp_path):
    entry = _entry(tmp_path)
    zip_path = entry["extract_to"] / entry["zip_name"]
    zip_path.parent.mkdir(parents=True)
    zip_path.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        gsv_bootstrap._extract(entry)
    assert not entry["marker"].exists()
    assert _leftovers(entry["extract_to"]) == []
