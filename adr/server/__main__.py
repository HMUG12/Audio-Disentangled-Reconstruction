"""`python -m adr.server` 入口: 启动 ADR TTS 服务。

默认 0.0.0.0:9881 — 与 GPT-SoVITS api_v2 / api_neko 端口一致,
N.E.K.O 无需改端口配置。
"""
from __future__ import annotations

import argparse

import uvicorn

from adr.server.app import create_app


def main():
    ap = argparse.ArgumentParser(
        description="ADR TTS 服务 (GSV api_v2 兼容层 + ADR 原生档案 API)")
    ap.add_argument("-p", "--port", type=int, default=9881,
                    help="监听端口 (默认 9881, 与 GSV api_v2 一致)")
    ap.add_argument("-a", "--addr", default="0.0.0.0",
                    help="监听地址 (默认 0.0.0.0)")
    args = ap.parse_args()
    uvicorn.run(create_app(), host=args.addr, port=args.port)


if __name__ == "__main__":
    main()
