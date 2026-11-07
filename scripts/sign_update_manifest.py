#!/usr/bin/env python
"""给更新清单 update.json 加 Ed25519 签名 (批次76 #3)。

用法::

    # 首次: 生成密钥对, 私钥写 ~/.adr/update_signing_key.pem, 打印公钥
    python scripts/sign_update_manifest.py --genkey

    # 签名 (就地重写 update.json, 保留其余字段)
    python scripts/sign_update_manifest.py update.json

    # 只校验签名, 不改文件
    python scripts/sign_update_manifest.py --check update.json

签名对象是 `signing_payload()` 生成的逐字节文本, 与 Rust 端
`desktop/src-tauri/src/update.rs::signing_payload` 必须完全一致:

    adr-update-v1\\n
    {version}\\n
    sha256(notes utf-8)\\n
    {variant}|{name}|{url}|{fallback_url}|{sha256 小写}|{size}\\n   (每个包一行, 顺序按文件)

改了任何一行都会让所有老客户端拒收清单, 因此格式已由 Rust 单测钉死
(`signing_payload_is_byte_stable`)。

私钥绝不入库: 默认路径 ~/.adr/update_signing_key.pem (可用 ADR_UPDATE_SIGNING_KEY
或 --key 覆盖)。发布机丢了私钥 = 只能发新版壳换公钥, 请自行备份。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:  # pragma: no cover - 发布机环境缺依赖时给明确指引
    sys.exit("需要 cryptography: D:\\pyhon\\python.exe -m pip install cryptography")

REPO = Path(__file__).resolve().parents[1]
RUST_UPDATE = REPO / "desktop" / "src-tauri" / "src" / "update.rs"
DEFAULT_KEY = Path(os.path.expanduser("~")) / ".adr" / "update_signing_key.pem"


def key_path(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    env = os.environ.get("ADR_UPDATE_SIGNING_KEY")
    return Path(env).expanduser() if env else DEFAULT_KEY


def load_key(path: Path) -> Ed25519PrivateKey:
    if not path.is_file():
        sys.exit(f"私钥不存在: {path}\n先跑 --genkey, 或把发布机的私钥放回该路径")
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def pubkey_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()


def embedded_pubkey() -> str | None:
    """从 update.rs 读客户端内置公钥, 用于防止签错钥。"""
    try:
        src = RUST_UPDATE.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'UPDATE_PUBKEY_HEX:\s*&str\s*=\s*"([0-9a-fA-F]{64})"', src)
    return m.group(1).lower() if m else None


def signing_payload(manifest: dict) -> bytes:
    notes_hash = hashlib.sha256(str(manifest.get("notes") or "").encode("utf-8")).hexdigest()
    lines = ["adr-update-v1", str(manifest.get("version") or "").strip(), notes_hash]
    for p in manifest.get("packages") or []:
        lines.append(
            "|".join(
                [
                    str(p.get("variant") or "").strip(),
                    str(p.get("name") or "").strip(),
                    str(p.get("url") or "").strip(),
                    str(p.get("fallback_url") or "").strip(),
                    str(p.get("sha256") or "").strip().lower(),
                    str(int(p.get("size") or 0)),
                ]
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="给 update.json 加 Ed25519 签名")
    ap.add_argument("manifest", nargs="?", help="update.json 路径 (--genkey 时可省)")
    ap.add_argument("--key", help=f"私钥路径 (默认 {DEFAULT_KEY})")
    ap.add_argument("--genkey", action="store_true", help="生成密钥对 (已有则只打印公钥)")
    ap.add_argument("--check", action="store_true", help="只校验, 不写回")
    args = ap.parse_args()

    kp = key_path(args.key)
    if args.genkey:
        if kp.exists():
            key = load_key(kp)
            print(f"私钥已存在, 复用: {kp}")
        else:
            kp.parent.mkdir(parents=True, exist_ok=True)
            key = Ed25519PrivateKey.generate()
            kp.write_bytes(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
            print(f"已生成私钥: {kp}  (注意备份, 不要提交到仓库)")
        print(f"公钥 hex: {pubkey_hex(key)}")
        return 0

    if not args.manifest:
        ap.error("需要 update.json 路径")
    path = Path(args.manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    key = load_key(kp)
    pk = pubkey_hex(key)
    want = embedded_pubkey()
    if want and want != pk:
        sys.exit(
            "私钥与客户端内置公钥不匹配, 签出来的清单全部客户端都会拒收。\n"
            f"  客户端 update.rs: {want}\n"
            f"  当前私钥:        {pk}\n"
            "要么用对的私钥, 要么换钥并重新编译壳 (公钥常量在 update.rs)。"
        )

    sig = key.sign(signing_payload(manifest)).hex()
    if args.check:
        got = str(manifest.get("signature") or "").strip().lower()
        ok = got == sig
        print("签名一致" if ok else "签名不一致 (清单被改动, 需要重新签名)")
        print(f"  文件内: {got[:16]}…\n  期望:   {sig[:16]}…" if got else "  文件内: (缺失)")
        return 0 if ok else 1

    manifest["signature"] = sig
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已签名: {path}")
    print(f"  公钥: {pk}")
    print(f"  签名: {sig[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
