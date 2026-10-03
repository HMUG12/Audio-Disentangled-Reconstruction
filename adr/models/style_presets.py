"""说话风格人设预设 — 把"人设描述"映射到合成参数组合 (批次 7)。

动机: GSV 微调锁音色不锁"性格" (韵律人设主要来自 ref 音频 + GPT 采样
参数)。本模块让用户用自然语言描述想要的感觉 (如"冷淡果断无感情的 AI"),
自动映射到 {temperature, top_k, speed_factor} 组合 + 录制指引, 使合成
尽量贴合人设, 而不必理解每个参数。

用法:
    from adr.models.style_presets import resolve_style, list_styles
    preset, matched = resolve_style("冷淡 果断 无感情 绝对理性")
    wav, sr = eng.synthesize(text, ref, temperature=preset.temperature,
                             top_k=preset.top_k, speed_factor=preset.speed)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StylePreset:
    key: str
    name: str                 # WebUI 下拉显示名
    desc: str                 # 一句话人设说明
    temperature: float        # GPT AR 采样温度: 低=平稳健宕, 高=起伏多样
    top_k: int                # 采样候选: 低=保守机械, 高=自然活泼
    speed: float              # 语速因子
    record_guide: str         # 参考音频录制指引 (人设的主要杠杆!)
    keywords: tuple = ()      # 描述匹配关键词 (全小写)


_BUILTIN = (
    StylePreset(
        "natural", "自然对话 (默认)", "日常说话, 抑扬自然",
        1.0, 15, 1.0,
        "用平时聊天的语气录制参考音频, 自然即可。",
        ("自然", "日常", "normal", "default"),
    ),
    StylePreset(
        "moss", "理性 AI (MOSS 式)", "无感情/冷淡/果断/绝对理性/机械冷静",
        0.65, 10, 1.0,
        "平板语调录制: 语调压平、语速均匀、句尾不上扬不下沉、情绪归零, "
        "像朗读系统日志一样念参考文本。",
        ("moss", "ai", "机器人", "机械", "冷淡", "冷漠", "无感情", "感情",
         "冷静", "冷酷", "理性", "果断", "平板", "平直", "严肃", "系统",
         "人工智能", "assistant", "siri", "管家"),
    ),
    StylePreset(
        "news", "新闻播报", "正式/庄重/字正腔圆",
        0.8, 15, 1.05,
        "用播新闻的腔调录制: 咬字清晰、节奏规整、句尾沉稳收束。",
        ("新闻", "播报", "播音", "正式", "庄重", "新闻联播", "主持",
         "news", "broadcast"),
    ),
    StylePreset(
        "gentle", "温柔陪伴", "轻柔/亲切/治愈",
        1.0, 15, 0.95,
        "用轻声细语录制: 音量放低、语速放缓、尾音柔和下沉。",
        ("温柔", "亲切", "轻柔", "软", "暖", "治愈", "陪伴", "柔和",
         "耐心", "gentle", "asmr"),
    ),
    StylePreset(
        "calm", "沉稳低语", "低沉/磁性/慵懒",
        0.75, 12, 0.92,
        "压低声音录制: 语速放慢、气息沉稳、句尾留白。",
        ("沉稳", "低沉", "磁性", "慵懒", "大叔", "低语", "深沉",
         "酷", "高冷", "calm"),
    ),
    StylePreset(
        "energetic", "活力元气", "活泼/热情/兴奋",
        1.1, 20, 1.05,
        "用元气满满的语气录制: 语速稍快、音调起伏大、带笑意。",
        ("活泼", "元气", "兴奋", "热情", "开朗", "活力", "激情",
         "少年", "少女", "energetic"),
    ),
)

_PRESETS = {p.key: p for p in _BUILTIN}


def list_styles() -> list[tuple[str, str]]:
    """(key, 显示名) 列表 — WebUI 下拉用。"""
    return [(p.key, f"{p.name} — {p.desc}") for p in _BUILTIN]


def get_style(key: str) -> StylePreset:
    return _PRESETS.get(key, _PRESETS["natural"])


def resolve_style(desc: str) -> tuple[StylePreset, bool]:
    """自由文本人设描述 → (预设, 是否精确命中)。

    关键词计数打分 (大小写不敏感, 子串匹配), 最高分为胜; 无命中回退
    natural。不引入模型语义理解 — 关键词表对 MVP 足够且零依赖。
    """
    d = (desc or "").strip().lower()
    if not d:
        return _PRESETS["natural"], False
    best, best_score = _PRESETS["natural"], 0
    for p in _BUILTIN:
        score = sum(1 for kw in p.keywords if kw and kw in d)
        if score > best_score:
            best, best_score = p, score
    return best, best_score > 0


def style_guide(key_or_desc: str = "") -> str:
    """人设描述 → 参数摘要 + 录制指引 (给用户的一条龙文案)。"""
    p, hit = (resolve_style(key_or_desc) if key_or_desc
              else (_PRESETS["natural"], True))
    head = f"匹配风格: {p.name}" if hit else "未命中特定风格, 使用自然对话"
    return (f"{head}\n"
            f"合成参数: temperature={p.temperature}, top_k={p.top_k}, "
            f"speed={p.speed}\n"
            f"录制指引: {p.record_guide}")
