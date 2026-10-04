"""ADR 对外 TTS 服务层 (批次11)。

- /api/v2/*      GPT-SoVITS api_v2 兼容层 — N.E.K.O 等 GSV 生态零改造接入
- /api/adr/v1/*  ADR 原生档案 API — 音色档案一等公民
- 启动: `python -m adr.server -p 9881`
- 对接规范: docs/tts-api-spec.md
"""
from adr.server.app import create_app

__all__ = ["create_app"]
