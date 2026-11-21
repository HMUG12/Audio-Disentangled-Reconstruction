"""控制台全局设置 API (批次: 控制台全局设置后端)。

- GET    /api/adr/v1/settings  完整设置 + updated_at (文件 mtime, 无文件为 null)
- PUT    /api/adr/v1/settings  部分嵌套对象更新 (校验失败 400, detail 中文可读)
- DELETE /api/adr/v1/settings  恢复默认

鉴权沿用 APIKeyMiddleware / OriginGuardMiddleware, 不额外开白名单。
这些值只是控制台前端的默认值, 不改变 /api/adr/v1/tts 对未传参请求的行为。
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from adr.core import settings_store
from adr.server.origin_guard import require_browser_safe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/adr/v1", tags=["adr-settings"])


def _body(settings: dict[str, Any]) -> dict[str, Any]:
    """设置 + schema_version + updated_at (mtime 在写盘之后现取)。"""
    settings.setdefault("schema_version", settings_store.SCHEMA_VERSION)
    settings["updated_at"] = settings_store.updated_at()
    return settings


@router.get("/settings", summary="控制台全局设置 (推理参数默认值)")
async def read_settings():
    return _body(settings_store.get_settings())


@router.put("/settings", summary="更新控制台全局设置 (部分嵌套对象)")
async def write_settings(payload: Any = Body(...),
                         _: None = Depends(require_browser_safe)):
    if not isinstance(payload, dict):
        raise HTTPException(400, "请求体必须是 JSON 对象")
    try:
        updated = settings_store.update_settings(payload)
    except ValueError as exc:
        raise HTTPException(400, f"设置校验失败: {exc}") from exc
    return _body(updated)


@router.delete("/settings", summary="恢复控制台全局默认设置")
async def reset_settings(_: None = Depends(require_browser_safe)):
    return _body(settings_store.reset_settings())
