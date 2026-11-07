"""批次23: 流式句级缓存 + 并发排队可视化 单元测试。

GSVEngine 轻量构造 (懒加载/上下文/热换全部旁路), 注入 FakeTTS 计数前向调用。
断言:
- 同段二发命中缓存 (前向次数不增, int16 PCM 字节一致, sr 一致)
- 不同文本不误命中; ADR_SEG_CACHE=0 关闭; LRU 条数上限逐出
- 排队/合成计数时序: 并发时 queue_depth>=1 可观测, 结束后归零
- 模块级 queue_depth/synth_busy 透传 (_ENGINE 为 None 时 0)
"""
import contextlib
import threading
import time

import numpy as np
import pytest

from adr.models import gsv_engine, gsv_runtime
from adr.models.gsv_engine import GSVEngine


class FakeTTS:
    """最小 TTS 替身: run(inputs) 逐块 yield (sr, chunk), 记录每段文本。"""

    def __init__(self, sr: int = 32000, blocks: int = 2, delay: float = 0.0):
        self.sr = sr
        self.blocks = blocks
        self.delay = delay
        self.calls: list = []

    def run(self, inputs):
        self.calls.append(inputs.get("text", ""))
        for i in range(self.blocks):
            if self.delay:
                time.sleep(self.delay)
            chunk = np.full(160, float(i + 1) / 10.0, dtype=np.float32)
            yield self.sr, chunk


@pytest.fixture(autouse=True)
def _clean_seg_cache():
    gsv_engine._seg_cache_clear()
    yield
    gsv_engine._seg_cache_clear()


@pytest.fixture
def eng(monkeypatch) -> GSVEngine:
    """轻量 GSVEngine: _lazy_init / _gsv_context / _ensure_weights 全旁路。"""
    e = GSVEngine()
    monkeypatch.setattr(e, "_lazy_init", lambda: None)
    monkeypatch.setattr(e, "_gsv_context", contextlib.nullcontext)
    monkeypatch.setattr(e, "_ensure_weights", lambda v, t: None)
    return e


def _collect(gen):
    return [(c, sr) for c, sr in gen]


def test_stream_seg_cache_hit(eng):
    """同文本二发: 零前向, PCM 字节与 sr 一致 (int16 往返)。"""
    tts = FakeTTS()
    eng._tts = tts
    kw = dict(ref_audio="ref.wav", prompt_text="参考文本")
    a = _collect(eng.synthesize_stream("你好世界。", **kw))
    assert tts.calls == ["你好世界。"]
    b = _collect(eng.synthesize_stream("你好世界。", **kw))
    assert tts.calls == ["你好世界。"]          # 第二次零前向
    # 缓存回放是段级整块 (chunk 边界可与首发不同), 拼接后字节必须一致
    pcm_a = b"".join((np.clip(c, -1, 1) * 32767).astype(np.int16).tobytes()
                     for c, _ in a)
    pcm_b = b"".join((np.clip(c, -1, 1) * 32767).astype(np.int16).tobytes()
                     for c, _ in b)
    assert pcm_a == pcm_b
    assert {sr for _, sr in a} == {sr for _, sr in b}


def test_stream_seg_cache_diff_text_miss(eng):
    """不同文本 key 不同 → 两次都前向, 不误命中。"""
    tts = FakeTTS()
    eng._tts = tts
    kw = dict(ref_audio="ref.wav")
    _collect(eng.synthesize_stream("甲文本。", **kw))
    _collect(eng.synthesize_stream("乙文本。", **kw))
    assert tts.calls == ["甲文本。", "乙文本。"]


def test_stream_seg_cache_disable(eng, monkeypatch):
    """ADR_SEG_CACHE=0 → 完全旁路, 不写不读。"""
    monkeypatch.setenv("ADR_SEG_CACHE", "0")
    tts = FakeTTS()
    eng._tts = tts
    kw = dict(ref_audio="ref.wav")
    _collect(eng.synthesize_stream("重复文本。", **kw))
    _collect(eng.synthesize_stream("重复文本。", **kw))
    assert tts.calls == ["重复文本。", "重复文本。"]
    assert not gsv_engine._SEG_CACHE


def test_seg_cache_lru_evict_items(eng, monkeypatch):
    """条数上限逐出: max=2 时第三条挤掉最早一条。"""
    monkeypatch.setenv("ADR_SEG_CACHE_MAX", "2")
    monkeypatch.delenv("ADR_SEG_CACHE_MB", raising=False)
    tts = FakeTTS()
    eng._tts = tts
    kw = dict(ref_audio="ref.wav")
    for t in ("一。", "二。", "三。"):
        _collect(eng.synthesize_stream(t, **kw))
    assert len(gsv_engine._SEG_CACHE) == 2
    n = len(tts.calls)
    _collect(eng.synthesize_stream("一。", **kw))   # 最早已逐出 → 重新前向
    assert len(tts.calls) == n + 1


def test_seg_cache_lru_touch(eng, monkeypatch):
    """命中会 move_to_end: 访问旧条目后, 逐出顺序改变。"""
    monkeypatch.setenv("ADR_SEG_CACHE_MAX", "2")
    monkeypatch.delenv("ADR_SEG_CACHE_MB", raising=False)
    tts = FakeTTS()
    eng._tts = tts
    kw = dict(ref_audio="ref.wav")
    _collect(eng.synthesize_stream("一。", **kw))
    _collect(eng.synthesize_stream("二。", **kw))
    _collect(eng.synthesize_stream("一。", **kw))   # 触碰一。 → 二。变最旧
    n = len(tts.calls)
    _collect(eng.synthesize_stream("三。", **kw))   # 挤掉二。
    _collect(eng.synthesize_stream("一。", **kw))   # 一。仍命中
    assert len(tts.calls) == n + 1                  # 只有三。前向


def test_seg_cache_params_in_key(eng):
    """采样参数变化 → key 不同 → 重新前向 (speed_factor 等)。"""
    tts = FakeTTS()
    eng._tts = tts
    _collect(eng.synthesize_stream("参数文本。", ref_audio="ref.wav",
                                   speed_factor=1.0))
    _collect(eng.synthesize_stream("参数文本。", ref_audio="ref.wav",
                                   speed_factor=1.2))
    assert tts.calls == ["参数文本。", "参数文本。"]


def test_queue_depth_timing(eng):
    """并发排队时序: t1 排队期间 queue_depth>=1 可观测, 结束后归零。"""
    tts = FakeTTS(delay=0.05, blocks=3)
    eng._tts = tts
    seen = {"queue": 0, "busy": 0}

    def worker(idx):
        _collect(eng.synthesize_stream(f"并发文本{idx}。", ref_audio="r.wav"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    threads[0].start()
    time.sleep(0.02)                       # t0 已持锁进入合成
    threads[1].start()
    t_end = time.time() + 2.0
    while time.time() < t_end:
        seen["busy"] = max(seen["busy"], eng.synth_busy())
        if eng.queue_depth() >= 1:
            seen["queue"] = 1
            break
        time.sleep(0.005)
    for t in threads:
        t.join(timeout=5)
    assert seen["queue"] == 1
    assert seen["busy"] >= 1
    assert eng.queue_depth() == 0
    assert eng.synth_busy() == 0


def test_module_level_queue_funcs(eng, monkeypatch):
    """模块级透传: _ENGINE 为 None 返回 0, 注入后读实例计数。"""
    monkeypatch.setattr(gsv_engine, "_ENGINE", None)
    assert gsv_engine.queue_depth() == 0
    assert gsv_engine.synth_busy() == 0
    monkeypatch.setattr(gsv_engine, "_ENGINE", eng)
    assert gsv_engine.queue_depth() == 0
    assert gsv_engine.synth_busy() == 0


def test_busy_counted_during_synthesis(eng):
    """单请求: 合成期间 busy=1, 完整消费后归零 (含生成器 finally)。"""
    tts = FakeTTS(delay=0.03, blocks=2)
    eng._tts = tts
    gen = eng.synthesize_stream("忙碌文本。", ref_audio="r.wav")
    first = next(gen)                       # 迭代一次 → 已进锁
    assert eng.synth_busy() == 1
    for _ in gen:                           # 消费完剩余
        pass
    assert eng.synth_busy() == 0
    assert first[1] == 32000


def test_seg_cache_ref_version_miss(eng, tmp_path):
    """批次85 (#18): 同名参考音频被覆盖 (size 变化) → 段缓存不命中。"""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"v1")
    tts = FakeTTS()
    eng._tts = tts
    _collect(eng.synthesize_stream("版本文本。", ref_audio=str(ref)))
    ref.write_bytes(b"v2-longer")           # 同名覆盖 → 指纹变化
    _collect(eng.synthesize_stream("版本文本。", ref_audio=str(ref)))
    assert tts.calls == ["版本文本。", "版本文本。"]


def test_seg_cache_head_seed_first_seg_only(eng):
    """批次85 (#18): head_seed 变化只影响首段; 后续段不随种子分片。"""
    tts = FakeTTS()
    eng._tts = tts
    text = "第一句实验文本。第二句实验文本。"
    kw = dict(ref_audio="ref.wav")
    _collect(eng.synthesize_stream(text, head_seed=1, **kw))
    assert len(tts.calls) == 2                       # 两段
    _collect(eng.synthesize_stream(text, head_seed=1, **kw))
    assert len(tts.calls) == 2                       # 同 seed 全命中
    _collect(eng.synthesize_stream(text, head_seed=7, **kw))
    assert len(tts.calls) == 3                       # 仅首段重新前向


def test_seg_cache_key_includes_weights_stamp():
    """批次88 (CACHE-01): 权重 (路径, mtime, size) 指纹入段 key。"""
    base = dict(seg="文本。", ref_audio="r.wav", prompt_text="",
                text_lang="zh", prompt_lang="zh", split_method="cut1",
                top_k=15, top_p=1.0, temperature=1.0, speed_factor=1.0,
                fragment_interval=0.3, t2s_weights="t.ckpt",
                vits_weights="v.pth")
    k0 = gsv_runtime._seg_cache_key(**base)
    same = gsv_runtime._seg_cache_key(
        **base, t2s_stamp=("t.ckpt", 1, 10), vits_stamp=("v.pth", 2, 20))
    assert k0 != same
    # 同指纹稳定命中
    assert gsv_runtime._seg_cache_key(
        **base, t2s_stamp=("t.ckpt", 1, 10), vits_stamp=("v.pth", 2, 20)
    ) == same
    # 同路径换权重 (mtime/size 变化) → key 必须变化
    assert gsv_runtime._seg_cache_key(
        **base, t2s_stamp=("t.ckpt", 1, 10), vits_stamp=("v.pth", 3, 20)
    ) != same
    assert gsv_runtime._seg_cache_key(
        **base, t2s_stamp=("t.ckpt", 2, 10), vits_stamp=("v.pth", 2, 20)
    ) != same


def test_ensure_weights_change_clears_seg_cache(eng, monkeypatch, tmp_path):
    """批次88 (CACHE-01): 换权后段缓存清空; 同指纹幂等时不误清。"""
    from adr.models import gsv_runtime as gr

    calls = []

    class _TTS:
        def init_vits_weights(self, p):
            calls.append(("vits", p))

        def init_t2s_weights(self, p):
            calls.append(("t2s", p))

    eng._tts = _TTS()
    monkeypatch.setattr(eng, "_ensure_weights",
                        GSVEngine._ensure_weights.__get__(eng))
    monkeypatch.setattr(gsv_engine, "validate_weights_file", lambda p: None)
    w = tmp_path / "v.pth"
    w.write_bytes(b"v1")

    gr._seg_cache_put("k", b"\x00\x00" * 8, 32000)
    assert gr._SEG_CACHE
    eng._ensure_weights(str(w), None)
    assert calls == [("vits", str(w.resolve()))]
    assert not gr._SEG_CACHE               # 换权 → 旧 PCM 必须作废

    gr._seg_cache_put("k2", b"\x00\x00" * 8, 32000)
    eng._ensure_weights(str(w), None)      # 同指纹 → 不重载、不清缓存
    assert calls == [("vits", str(w.resolve()))]
    assert "k2" in gr._SEG_CACHE


def test_seg_cache_put_same_key_no_double_count(monkeypatch):
    """批次85 (#26): 同 key 重复写入按最新值计字节, 不重复累加。"""
    monkeypatch.setenv("ADR_SEG_CACHE_MAX", "10")
    monkeypatch.setenv("ADR_SEG_CACHE_MB", "1")
    pcm = b"\x00\x00" * 100                          # 200 字节
    gsv_engine._seg_cache_put("k", pcm, 32000)
    assert gsv_engine._SEG_CACHE["k"] == (pcm, 32000)
    assert gsv_runtime._SEG_BYTES == len(pcm)
    gsv_engine._seg_cache_put("k", pcm[:100], 32000)  # 覆盖为更短
    assert gsv_engine._SEG_CACHE["k"] == (pcm[:100], 32000)
    assert gsv_runtime._SEG_BYTES == 100
    assert len(gsv_engine._SEG_CACHE) == 1
