"""测试批次 9: 流式首包优化 (首段切分收紧 + 热换幂等)。"""

from adr.models.gsv_engine import GSVEngine


def test_head_split_short_text_kept_whole():
    """≤14 字短句不切 (整句即首段)。"""
    text = "收到，马上出发。"
    head, rest = GSVEngine._stream_head_split(text)
    assert head == text and rest == ""


def test_head_split_comma_level():
    """16~20 字短句在逗号处切出短首段 (批次 9 收紧)。"""
    text = "检测到异常访问，防御协议已启动。"   # 16 字, 逗号在 7
    head, rest = GSVEngine._stream_head_split(text)
    assert head == "检测到异常访问，" and rest == "防御协议已启动。"


def test_head_split_long_text_sentence_end():
    """长文本句末标点优先切分。"""
    text = "人工智能技术正在快速发展。语音合成已经能够生成自然流畅的音频效果，非常惊人。"
    head, rest = GSVEngine._stream_head_split(text)
    assert head.endswith("。") and rest.startswith("语音")
    assert len(head) <= 14


def test_ensure_weights_skip_same_path():
    """同路径热换幂等跳过 (稳态不再重复加载权重)。"""
    eng = GSVEngine.__new__(GSVEngine)  # 跳过 __init__, 手工装状态
    eng._loaded = {"t2s": None, "vits": None}
    calls = []

    class FakeTTS:
        def init_vits_weights(self, p):
            calls.append(("vits", p))

        def init_t2s_weights(self, p):
            calls.append(("t2s", p))

    class FakeCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    eng._tts = FakeTTS()
    eng._gsv_context = FakeCtx

    eng._ensure_weights("/w/a.pth", "/w/b.ckpt")
    assert len(calls) == 2
    eng._ensure_weights("/w/a.pth", "/w/b.ckpt")   # 同路径 → 跳过
    assert len(calls) == 2
    eng._ensure_weights("/w/other.pth", None)      # 换 vits → 只热换 vits
    assert [sys_name for sys_name, _ in calls] == ["vits", "t2s", "vits"]
    # 传入路径经 resolve 归一, 相同真实文件只加载一次
    assert calls[2][1] != calls[0][1]              # other.pth ≠ a.pth
