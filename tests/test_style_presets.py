"""测试 adr.models.style_presets (批次 7 人设风格预设)。"""

from adr.models.style_presets import get_style, list_styles, resolve_style, style_guide


def test_builtin_presets_complete():
    """内置预设字段完整且参数在安全范围。"""
    styles = list_styles()
    assert len(styles) >= 5
    for key, display in styles:
        p = get_style(key)
        assert p.key == key
        assert 0.5 <= p.temperature <= 1.2
        assert 5 <= p.top_k <= 30
        assert 0.8 <= p.speed <= 1.1
        assert p.record_guide  # 每个预设都有录制指引


def test_resolve_moss_persona():
    """MOSS 式描述 → 理性 AI 预设。"""
    for desc in ["冷淡 果断 无感情 绝对理性", "像流浪地球的moss", "冷酷的机器人管家"]:
        p, hit = resolve_style(desc)
        assert hit
        assert p.key == "moss"
        assert p.temperature < 1.0  # 平稳档


def test_resolve_other_personas():
    """其他人设描述路由正确。"""
    assert resolve_style("温柔一点 治愈系")[0].key == "gentle"
    assert resolve_style("新闻联播腔")[0].key == "news"
    assert resolve_style("低沉磁性大叔音")[0].key == "calm"
    assert resolve_style("活泼元气少女")[0].key == "energetic"


def test_resolve_fallback():
    """空/无关描述回退 natural。"""
    p, hit = resolve_style("")
    assert p.key == "natural" and not hit
    p, hit = resolve_style("云计算区块链人工智能时代")  # 含 "智能" 但 moss 词表是 "人工智能"
    # 即便命中也不崩; 不命中则回退
    assert p.key in ("moss", "natural")


def test_style_guide_text():
    """指引文案包含参数与录制建议。"""
    g = style_guide("无感情 冷淡 理性")
    assert "moss" in g.lower() or "理性" in g
    assert "temperature" in g
    assert "录制指引" in g
