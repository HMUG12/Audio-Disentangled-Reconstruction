"""`python -m adr.server` 入口: 启动 ADR TTS 服务。

默认 0.0.0.0:9881 — 与 GPT-SoVITS api_v2 / api_neko 端口一致,
N.E.K.O 无需改端口配置。

安全提示 (批次26): 默认监听 0.0.0.0 意味着局域网可达; 未设 ADR_TTS_API_KEY
时所有 API 完全无鉴权 (见 app.py 中间件)。启动时检测该组合并打醒目警告 —
桌面壳 expose=true 也走此入口 (壳传 --addr 0.0.0.0), 同样被覆盖。
"""
from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from adr.server.app import create_app


def _warn_unauthenticated_listen(addr: str) -> None:
    """监听地址对外 + 无 API Key 时打醒目警告 (批次26, P2-a)。

    create_app 不知道 bind host (uvicorn.run 在本入口传参),
    所以只有这里能同时看到 host 与鉴权配置 — 警告只打这一处, 不在 app.py 重复。
    """
    if addr not in ("0.0.0.0", "::"):
        return
    if os.environ.get("ADR_TTS_API_KEY", "").strip():
        return
    print(
        "\n" + "=" * 64,
        "⚠  安全警告: 服务监听 0.0.0.0 且未设置 ADR_TTS_API_KEY",
        "   局域网内任何设备均可无鉴权调用合成 API。",
        "   如需暴露到局域网, 请设置环境变量后重启:",
        '     set ADR_TTS_API_KEY=<你的密钥>        (Windows CMD)',
        '     $env:ADR_TTS_API_KEY="<你的密钥>"     (PowerShell)',
        "   仅本机使用可改用: python -m adr.server --addr 127.0.0.1",
        "=" * 64 + "\n",
        sep="\n", file=sys.stderr, flush=True,
    )


def main():
    ap = argparse.ArgumentParser(
        description="ADR TTS 服务 (GSV api_v2 兼容层 + ADR 原生档案 API)")
    ap.add_argument("-p", "--port", type=int, default=9881,
                    help="监听端口 (默认 9881, 与 GSV api_v2 一致)")
    ap.add_argument("-a", "--addr", default="0.0.0.0",
                    help="监听地址 (默认 0.0.0.0)")
    args = ap.parse_args()
    _warn_unauthenticated_listen(args.addr)
    uvicorn.run(create_app(), host=args.addr, port=args.port)


if __name__ == "__main__":
    main()
