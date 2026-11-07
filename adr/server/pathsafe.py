"""路径安全 helper (Track B 收口): 用户可控路径点统一清洗。

规则:
- resolve_within(root, candidate): candidate 解析后必须落在 root 目录内,
  否则返回 None — 用于"目录内取文件"场景 (档案参考音频等);
- is_safe_name(name): 名称类参数必须是单段纯名 (无路径分隔符/..),
  用于档案名等直接拼路径的字符串。

批次89: 两个函数已下沉到 adr.core.naming (与 models/voice_library 共用同一实现),
本模块仅保留再导出, 既有 import 路径不变。
"""
from __future__ import annotations

from adr.core.naming import is_safe_name, resolve_within

__all__ = ["is_safe_name", "resolve_within"]
