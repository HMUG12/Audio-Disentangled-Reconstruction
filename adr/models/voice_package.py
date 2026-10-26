"""音色档案导入导出 (M10 批次47): .adrvoice zip 容器, 跨机迁移声音档案。

容器结构:
    manifest.json   格式/版本/档案名/权重清单/sha256 完整性
    ref.wav         参考音频 (3~10s, 与档案库一致)
    meta.json       可移植副本 (权重字段置 null — 真实路径由导入方按本机布局重写)
    weights/        原始权重文件 (保留原文件名)

权重路径解析口径 (与引擎/既有档案一致):
- t2s/vits: 绝对路径 / 仓库根相对 / GSV 根相对 (doctor 用 Path(w).exists() 口径)
- rvc_weights: weight_root (third_party/rvc/assets/weights) 下的纯文件名
- rvc_index: 任意可解析路径 (既有档案为绝对路径)

导入落位 (重写为本机绝对路径, 与既有档案写法一致):
- t2s  → third_party/gpt_sovits/GPT_weights_v2/
- vits → third_party/gpt_sovits/SoVITS_weights_v2/
- rvc  → third_party/rvc/assets/weights/ (meta 存纯文件名, 同既有惯例)
- rvc_index → third_party/rvc/logs/<档案名>/
落盘后 t2s/vits 过 validate_weights_file (批次43 两段式放行资产)。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

from adr.models import voice_library
from adr.models.voice_library import is_safe_name

REPO_ROOT = voice_library.REPO_ROOT
GSV_DIR = REPO_ROOT / "third_party" / "gpt_sovits"
RVC_DIR = REPO_ROOT / "third_party" / "rvc"

FORMAT = "adr-voice-profile"
VERSION = 1
_MANIFEST = "manifest.json"
_META = "meta.json"
_REF = "ref.wav"
_WPREFIX = "weights/"
_CHUNK = 1 << 20

# 批次66: zip-bomb 防护 — 按 ZipInfo.file_size (解压后声明大小) 在解压前拦截。
# zipfile 读成员时不会吐出超过 file_size 的字节 (超出即 CRC/长度报错), 声明值可作上界。
# 正常包: 权重合计 < 1GB, manifest/meta 仅 KB 级; 默认值远高于正常用量。
_MAX_TOTAL_BYTES = int(os.getenv("ADR_IMPORT_MAX_TOTAL_MB", "4096")) << 20
# manifest/meta 会整块读进内存, 单独限小
_MAX_INMEM_BYTES = int(os.getenv("ADR_IMPORT_MAX_INMEM_MB", "16")) << 20

_IMPORT_LOCK = threading.Lock()   # 批次66: 进程内串行导入 (见 import_voice)

Progress = Optional[Callable[[str], None]]

# kind → meta 字段名
_FIELDS = {"t2s": "t2s_weights", "vits": "vits_weights",
           "rvc": "rvc_weights", "rvc_index": "rvc_index"}


def _log(p: Progress, msg: str) -> None:
    if p:
        p(msg)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_member(zf: zipfile.ZipFile, rel: str) -> str:
    h = hashlib.sha256()
    with zf.open(rel) as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_member(zf: zipfile.ZipFile, rel: str, dst: Path) -> None:
    with zf.open(rel) as f, open(dst, "wb") as out:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            out.write(chunk)


# ---------------------------------------------------------------- 解析

def _resolve_t2s_vits(w: str) -> Path:
    cands = [Path(w), REPO_ROOT / w, GSV_DIR / w]
    for c in cands:
        if c.exists():
            return c
    raise ValueError(f"权重文件不存在: {w} (可运行 adr doctor 检查档案完整性)")


def _resolve_rvc(w: str) -> Path:
    cands = [RVC_DIR / "assets" / "weights" / w, Path(w), REPO_ROOT / w]
    for c in cands:
        if c.exists():
            return c
    raise ValueError(f"RVC 权重不存在: {w} (weight_root 下未找到)")


def _resolve_index(w: str) -> Path:
    cands = [Path(w), REPO_ROOT / w, RVC_DIR / w]
    for c in cands:
        if c.exists():
            return c
    raise ValueError(f"RVC 索引不存在: {w}")


_RESOLVERS = {"t2s": _resolve_t2s_vits, "vits": _resolve_t2s_vits,
              "rvc": _resolve_rvc, "rvc_index": _resolve_index}


def _dst_dir(kind: str, profile_name: str) -> Path:
    if kind == "t2s":
        return GSV_DIR / "GPT_weights_v2"
    if kind == "vits":
        return GSV_DIR / "SoVITS_weights_v2"
    if kind == "rvc":
        return RVC_DIR / "assets" / "weights"
    return RVC_DIR / "logs" / profile_name


# ---------------------------------------------------------------- 导出

def export_voice(name: str, out_path: Optional[str] = None,
                 progress: Progress = None) -> Path:
    """导出档案为 .adrvoice 包, 返回包路径。权重文件缺失直接报错。"""
    if not is_safe_name(name):
        raise ValueError(f"音色档案名非法: {name!r}")
    if name not in voice_library.list_voices():
        raise ValueError(f"档案不存在: {name} (现有: {voice_library.list_voices()})")
    meta = voice_library.load_voice(name)
    vdir = voice_library.VOICES_DIR / name

    srcs: dict[str, Path] = {}
    for kind, field in _FIELDS.items():
        w = meta.get(field)
        if w:
            srcs[kind] = _RESOLVERS[kind](str(w))
            _log(progress, f"权重 {kind}: {srcs[kind].name}")

    out = Path(out_path) if out_path else Path(f"{name}.adrvoice")
    out.parent.mkdir(parents=True, exist_ok=True)

    zip_names: dict[str, str] = {}
    integrity: dict[str, str] = {_REF: _sha256_file(vdir / _REF)}
    used: set[str] = set()
    for kind, src in srcs.items():
        rel = _WPREFIX + src.name
        if rel in used:   # 两种权重同名的极端情况: 加 kind 前缀消歧
            rel = _WPREFIX + f"{kind}_{src.name}"
        used.add(rel)
        zip_names[kind] = rel
        integrity[rel] = _sha256_file(src)

    portable = {k: v for k, v in meta.items() if k != "ref_audio"}
    for field in _FIELDS.values():
        portable[field] = None
    manifest = {
        "format": FORMAT, "version": VERSION,
        "name": meta.get("name") or name,
        "created_at": meta.get("created_at"),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exported_by": f"adr voice_package v{VERSION}",
        "weights": zip_names or None,
        "integrity": integrity,
    }

    _log(progress, f"写入 {out.name} (权重 {len(srcs)} 个)...")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=1) as zf:   # pth 压缩率极低, 取速度
        zf.writestr(_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr(_META, json.dumps(portable, ensure_ascii=False, indent=2))
        zf.write(str(vdir / _REF), _REF)
        for kind, src in srcs.items():
            zf.write(str(src), zip_names[kind])
    _log(progress, f"导出完成: {out} ({out.stat().st_size / (1 << 20):.0f} MB)")
    return out


# ---------------------------------------------------------------- 导入

def import_voice(pkg_path: str, name: Optional[str] = None,
                 overwrite: bool = False, progress: Progress = None) -> dict:
    """导入 .adrvoice 包, 返回 {name, weights, ref}。冲突默认拒绝。"""
    pkg = Path(pkg_path)
    if not pkg.exists():
        raise ValueError(f"包不存在: {pkg}")
    # 批次66: 锁包住整个导入含失败回滚, 修冲突检查与落位间的 TOCTOU。
    # _IMPORT_LOCK 串行全部导入 (权重目录/暂存名跨档案共享, 同名权重会互相覆盖);
    # held 内再挂 profile_lock(目标名), 与 save/bind/rollback 等同名写者互斥。
    with _IMPORT_LOCK, contextlib.ExitStack() as held:
        return _import_voice(pkg, name, overwrite, progress, held)


def _import_voice(pkg: Path, name: Optional[str], overwrite: bool,
                  progress: Progress, held: contextlib.ExitStack) -> dict:
    created: list[Path] = []   # 本次暂存的临时文件 (失败回滚只删这些, 不碰既有档案)
    try:
        with zipfile.ZipFile(pkg) as zf:
            names = set(zf.namelist())
            for must in (_MANIFEST, _META, _REF):
                if must not in names:
                    raise ValueError(f"包缺少 {must}, 不是有效的音色档案包")
            # 批次66: 解压前按声明大小拦 zip-bomb (任何 read/解压之前)
            total = sum(i.file_size for i in zf.infolist())
            if total > _MAX_TOTAL_BYTES:
                raise ValueError(
                    f"包解压后总大小 {total >> 20}MB 超上限 "
                    f"{_MAX_TOTAL_BYTES >> 20}MB (ADR_IMPORT_MAX_TOTAL_MB 可调)")
            for must in (_MANIFEST, _META):
                if zf.getinfo(must).file_size > _MAX_INMEM_BYTES:
                    raise ValueError(
                        f"包内 {must} 解压后超上限 {_MAX_INMEM_BYTES >> 20}MB "
                        f"(ADR_IMPORT_MAX_INMEM_MB 可调)")
            manifest = json.loads(zf.read(_MANIFEST))
            if manifest.get("format") != FORMAT:
                raise ValueError("manifest.format 不符, 不是 ADR 音色档案包")
            ver = int(manifest.get("version", 0))
            if ver > VERSION:
                raise ValueError(f"容器版本过新: v{ver} > 本机支持 v{VERSION}")

            # 完整性校验 (清单里登记且包内存在的文件全查)
            for rel, want in (manifest.get("integrity") or {}).items():
                if rel in names and _sha256_member(zf, rel) != want:
                    raise ValueError(f"完整性校验失败: {rel} (包损坏)")

            target = (name or manifest.get("name") or "").strip()
            if not is_safe_name(target):
                raise ValueError(f"音色名非法: {target!r}")
            held.enter_context(voice_library.profile_lock(target))   # 批次66
            vdir = voice_library.VOICES_DIR / target
            # 冲突判据 = 已有真实档案 (meta.json); 残留半截目录 (仅 ref) 放行续用
            if (vdir / "meta.json").exists() and not overwrite:
                raise ValueError(f"档案已存在: {target} (--overwrite 覆盖或 --name 换名)")

            # 权重清单: 只允许 weights/<单段文件名>, 防 zip-slip
            plan: dict[str, str] = {}
            for kind, rel in (manifest.get("weights") or {}).items():
                if kind not in _FIELDS:
                    continue
                if (not rel or not rel.startswith(_WPREFIX)
                        or "/" in rel[len(_WPREFIX):] or ".." in rel
                        or rel not in names):
                    raise ValueError(f"manifest 权重路径非法或文件缺失: {rel!r}")
                plan[kind] = rel

            # 冲突预检 (同 hash 视为已存在可复用)
            placements: dict[str, Path] = {}
            for kind, rel in plan.items():
                dst = _dst_dir(kind, target) / Path(rel).name
                if dst.exists() and not overwrite:
                    if _sha256_file(dst) != _sha256_member(zf, rel):
                        raise ValueError(
                            f"权重文件冲突: {dst} (--overwrite 覆盖)")
                    _log(progress, f"{kind}: 本机已存在同 hash 权重, 复用 {dst.name}")
                placements[kind] = dst

            # 两段式提交 (批次62): 先把权重/ref/meta 全部写成临时文件并校验,
            # meta 解析也在替换之前完成; 全部通过后才一次性 os.replace 落位。
            # 任何中途失败只删临时文件 (created), 既有档案的 ref.wav/权重不受损;
            # overwrite 走临时文件替换, 修复旧实现 "overwrite 不覆盖旧权重字节"。
            commits: list[tuple[str, Path, Path]] = []   # (kind, tmp, final)
            for kind, rel in plan.items():
                dst = placements[kind]
                if dst.exists() and not overwrite:
                    continue   # 同 hash 复用 (前置预检已保证字节一致)
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".adrimport.tmp")
                _copy_member(zf, rel, tmp)
                created.append(tmp)
                commits.append((kind, tmp, dst))
                _log(progress, f"{kind} → {dst} (暂存)")

            from adr.models.gsv_engine import validate_weights_file
            for kind, tmp, _dst in commits:
                if kind in ("t2s", "vits"):
                    _log(progress, f"校验 {kind} 权重...")
                    validate_weights_file(str(tmp))

            vdir.mkdir(parents=True, exist_ok=True)
            ref_tmp = vdir / (_REF + ".adrimport.tmp")
            created.append(ref_tmp)
            _copy_member(zf, _REF, ref_tmp)   # 批次66: 流式落盘, 不整块读进内存

            pmeta = json.loads(zf.read(_META))   # 损坏包在此抛错, 既有档案此刻完好
            pmeta["name"] = target
            for kind, field in _FIELDS.items():
                if kind not in placements:
                    pmeta[field] = None
                elif kind == "rvc":
                    pmeta[field] = placements[kind].name   # weight_root 下纯文件名
                else:
                    pmeta[field] = str(placements[kind])   # 本机绝对路径
            meta_tmp = vdir / "meta.json.tmp"
            meta_tmp.write_text(json.dumps(pmeta, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            created.append(meta_tmp)

            # ---- 提交: 全部校验/解析通过后才做破坏性替换 ----
            for _kind, tmp, dst in commits:
                os.replace(str(tmp), str(dst))
            os.replace(str(ref_tmp), str(vdir / _REF))
            os.replace(str(meta_tmp), str(vdir / "meta.json"))
    except Exception:
        for f in created:
            f.unlink(missing_ok=True)
        raise

    _log(progress, f"导入完成: {target}")
    return {"name": target,
            "ref": str(voice_library.VOICES_DIR / target / _REF),
            "weights": {k: str(p) for k, p in placements.items()}}
