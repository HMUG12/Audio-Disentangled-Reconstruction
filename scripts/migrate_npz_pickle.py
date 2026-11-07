"""旧版 object 数组 npz 迁移 (批次90 / SEC-03)。

背景: 训练样本 npz 的 ``sample_id`` / ``text`` / ``phonemes`` 曾被写成 object
dtype — object 成员在 .npy 里就是 pickle 流, 加载时可执行任意代码, 因此
``adr.data.pipeline.TrainSample.load`` 现在一律 ``allow_pickle=False``,
旧文件会被明确拒绝 (提示重跑预处理)。

本工具是**显式**的一次性迁移: 只有你主动运行它, 才会以 allow_pickle=True
读取指定目录下的 npz (仅限你自己的训练数据), 把上述三列改写为 numpy 字符串
数组后原子覆盖原文件; waveform / mel / f0 等其余数组原样保留 (解压-重压缩)。

用法:
    python scripts/migrate_npz_pickle.py data/opencpop_npz --dry-run   # 预演
    python scripts/migrate_npz_pickle.py data/opencpop_npz             # 迁移

退出码: 0 = 全部成功或无需迁移; 1 = 有文件失败。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# TrainSample 读取的字符串列 (旧版写 object, 新版写 dtype=str)
_STRING_KEYS = ("sample_id", "text", "phonemes")


def _legacy_keys(path: Path) -> list[str]:
    """返回该 npz 里仍是 object dtype 的字符串列 (空列表 = 无需迁移)。"""
    bad: list[str] = []
    with np.load(path, allow_pickle=False) as zf:
        names = set(zf.files)
        for key in _STRING_KEYS:
            if key not in names:
                continue
            try:
                zf[key]   # 惰性读取: object 成员在这里抛 ValueError
            except ValueError:
                bad.append(key)
    return bad


def _migrate(path: Path) -> None:
    """object 字符串列 → dtype=str, 原子覆盖原文件。"""
    with np.load(path, allow_pickle=True) as zf:
        data = {k: zf[k] for k in zf.files}
    for key in _STRING_KEYS:
        arr = data.get(key)
        if arr is not None and arr.dtype == object:
            data[key] = np.array(arr, dtype=str)

    # 临时名必须以 .npz 结尾: np.savez_compressed 会给没有该后缀的名字追加
    # ".npz", 否则 os.replace 找不到真正的临时文件 (实测 WinError 2)。
    tmp = path.with_name(path.stem + ".migrating.npz")
    try:
        np.savez_compressed(tmp, **data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把旧版 object 数组 npz 迁移为字符串数组 "
                    "(训练路径已禁用 allow_pickle)")
    ap.add_argument("root", help="npz 目录 (递归) 或单个 .npz 文件")
    ap.add_argument("--dry-run", action="store_true", help="只列出待迁移文件")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(root.glob("**/*.npz"))
    else:
        print(f"路径不存在: {root}")
        return 1
    if not files:
        print(f"{root} 下没有 .npz 文件")
        return 0

    migrated = skipped = failed = 0
    for f in files:
        try:
            keys = _legacy_keys(f)
        except Exception as e:  # noqa: BLE001 - 单个坏文件不中断整批迁移
            print(f"[!] 读取失败 {f}: {e}")
            failed += 1
            continue
        if not keys:
            skipped += 1
            continue
        label = ", ".join(keys)
        if args.dry_run:
            print(f"[dry-run] 需迁移 {f} ({label})")
            migrated += 1
            continue
        try:
            _migrate(f)
        except Exception as e:  # noqa: BLE001 - 同上
            print(f"[!] 迁移失败 {f}: {e}")
            failed += 1
            continue
        print(f"[ok] {f} ({label})")
        migrated += 1

    print(f"\n完成: 迁移 {migrated} / 无需迁移 {skipped} / 失败 {failed} "
          f"(共 {len(files)} 个文件)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
