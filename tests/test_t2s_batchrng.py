"""批次 65 第二子批: t2s per-seq RNG 批采样桥等价性测试。

CPU (tiny Text2SemanticDecoder, eager):
A. G3 同种子位等价: bs=1 桥 (行 generator manual_seed=S) vs stock
   batch_infer (global manual_seed=S) → tokens 全等 (每步恰一次
   exponential_ [1,V] 抽取, CPU 同种子同流)
B. G1 确定性: bs=2 同种子两跑 tokens 全等 (同形状同算子序 → CPU 确定)
C. G2 行隔离 + EOS 逐出: merge(S_a,S_b) vs merge(S_a,S_c) → 行 a 全等
   (某行提前逐出后余行采样流不受扰动); 多种子对中至少一对两行停步不同
   (逐出路径真实踩到); 行间种子不同 → 行输出不同 (RNG 在用 sanity)
D. row_sampling 行参数覆盖: top_k=1 贪心 → 输出与行种子无关 (行参数
   真达采样器)
E. 分派: row_seeds 缺席 → fallback, 与 stock 全局 RNG 行为位等

RNG 纪律: 桥路径每行独立 Generator(manual_seed); stock 路径前
manual_seed; 输入构造用显式 Generator, 不动全局流。
"""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
GSV_DIR = REPO / "third_party" / "gpt_sovits" / "GPT_SoVITS"
for _p in (str(REPO), str(GSV_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adr.models.adr_t2s_batchrng as br  # noqa: E402
from AR.models.t2s_model import Text2SemanticDecoder  # noqa: E402

CONFIG = {"model": {
    "hidden_dim": 32, "embedding_dim": 32, "head": 2, "n_layer": 2,
    "vocab_size": 24, "phoneme_vocab_size": 32, "dropout": 0.1, "EOS": 23,
}}
X_LENS = [13, 9]   # 行长不同 → 解码快慢不同 → EOS 逐出路径可踩
PROMPT = 5
EARLY_STOP = 60


def _model():
    torch.manual_seed(42)
    return Text2SemanticDecoder(CONFIG).eval()


def _rows_data(seed=7):
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randint(0, 32, (L,), generator=g) for L in X_LENS]
    x_lens = torch.LongTensor(X_LENS)
    prompts = torch.randint(0, 23, (2, PROMPT), generator=g)
    berts = [torch.randn(1024, L, generator=g) for L in X_LENS]
    return xs, x_lens, prompts, berts


def _bridge_call(model, xs, x_lens, prompts, berts, row_seeds,
                 early_stop_num=EARLY_STOP, row_sampling=None):
    fn = br.install_batched_rng_infer_panel(model, None)
    with torch.no_grad():
        return fn(xs, x_lens, prompts, berts, top_k=15, top_p=1.0,
                  temperature=1.0, early_stop_num=early_stop_num,
                  repetition_penalty=1.35, row_seeds=row_seeds,
                  row_sampling=row_sampling)


def _stock_call(model, xs, x_lens, prompts, berts, early_stop_num=EARLY_STOP):
    with torch.no_grad():
        return model.infer_panel_batch_infer(
            xs, x_lens, prompts, berts, top_k=15, top_p=1.0, temperature=1.0,
            early_stop_num=early_stop_num, repetition_penalty=1.35)


def test_a_g3_stock_equivalence_bs1():
    """bs=1 桥 (行 generator seed=S) == stock (global manual_seed=S)。"""
    model = _model()
    xs, x_lens, prompts, berts = _rows_data()
    torch.manual_seed(123)
    y_stock, idx_stock = _stock_call(model, [xs[0]], torch.LongTensor([X_LENS[0]]),
                                     prompts[:1], [berts[0]])
    y_bridge, idx_bridge = _bridge_call(model, [xs[0]], torch.LongTensor([X_LENS[0]]),
                                        prompts[:1], [berts[0]], [123])
    assert torch.equal(y_stock[0], y_bridge[0])
    assert idx_stock == idx_bridge


def test_b_g1_determinism_bs2():
    """同种子两跑 tokens 全等 (CPU 同形状同算子序确定)。"""
    model = _model()
    xs, x_lens, prompts, berts = _rows_data()
    y1, idx1 = _bridge_call(model, xs, x_lens, prompts, berts, [11, 22])
    y2, idx2 = _bridge_call(model, xs, x_lens, prompts, berts, [11, 22])
    assert all(torch.equal(a, b) for a, b in zip(y1, y2))
    assert idx1 == idx2


def test_c_g2_row_isolation_and_eviction():
    """行 a 种子固定、行 b 种子扫变 → 行 a 输出全等; 逐出路径真实踩到。"""
    model = _model()
    xs, x_lens, prompts, berts = _rows_data()
    ref_row0 = None
    eviction_seen = False
    for s2 in (22, 33, 44, 55, 66, 77, 88, 99):
        y, idx = _bridge_call(model, xs, x_lens, prompts, berts, [11, s2])
        cur0 = y[0]
        if ref_row0 is None:
            ref_row0 = cur0
        else:
            assert torch.equal(ref_row0, cur0)
        if idx[0] != idx[1]:
            eviction_seen = True
    assert eviction_seen, "8 对种子无一命中两行停步不同, 逐出路径未被踩到"
    # 行间种子不同 → 行输出不同 (RNG 真在被消费的 sanity)
    y_a, _ = _bridge_call(model, xs, x_lens, prompts, berts, [11, 22])
    y_b, _ = _bridge_call(model, xs, x_lens, prompts, berts, [11, 23])
    assert not torch.equal(y_a[1], y_b[1])


def test_d_row_sampling_override():
    """top_k=1 贪心行 → 输出与行种子无关 (行参数真达采样器)。"""
    model = _model()
    xs, x_lens, prompts, berts = _rows_data()
    greedy = [{"top_k": 1}, {"top_k": 1}]
    y1, _ = _bridge_call(model, xs, x_lens, prompts, berts, [11, 22], row_sampling=greedy)
    y2, _ = _bridge_call(model, xs, x_lens, prompts, berts, [11, 33], row_sampling=greedy)
    assert torch.equal(y1[0], y2[0])
    assert torch.equal(y1[1], y2[1])


def test_e_dispatch_fallback_no_row_seeds():
    """row_seeds 缺席 → fallback (stock 全局 RNG 行为位等)。

    生产中 TTS.py 每请求先把 infer_panel 重赋值为 batch_infer 再挂桥 —
    测试同样先重赋, install 捕获的 fallback 才是 batch_infer (类的 L1013
    infer_panel 是 naive 委托桩, 只收单行)。"""
    model = _model()
    model.infer_panel = model.infer_panel_batch_infer
    xs, x_lens, prompts, berts = _rows_data()
    fn = br.install_batched_rng_infer_panel(model, None)
    torch.manual_seed(7)
    y1, i1 = fn(xs, x_lens, prompts, berts, top_k=15, top_p=1.0,
                temperature=1.0, early_stop_num=EARLY_STOP,
                repetition_penalty=1.35)
    torch.manual_seed(7)
    y2, i2 = _stock_call(model, xs, x_lens, prompts, berts)
    assert all(torch.equal(a, b) for a, b in zip(y1, y2))
    assert i1 == i2


def test_f_derive_row_seeds():
    """行种子派生: 段 0 = 请求种子; set_seed 重播的全局 random 下请求内确定。"""
    import random
    random.seed(9)
    s1 = br.derive_row_seeds(3, 100)
    assert s1[0] == 100 and len(s1) == 3 and len(set(s1[1:])) == 2
    random.seed(9)
    assert br.derive_row_seeds(3, 100) == s1
    assert br.derive_row_seeds(1, 55) == [55]  # 单段只有请求种子


def test_g_settings_batchrng_and_batchsize(monkeypatch):
    from adr.core import settings
    monkeypatch.delenv("ADR_T2S_BATCH_RNG", raising=False)
    assert settings.t2s_batch_rng() is False
    monkeypatch.setenv("ADR_T2S_BATCH_RNG", "1")
    assert settings.t2s_batch_rng() is True
    monkeypatch.setenv("ADR_T2S_BATCH_RNG", " 1 ")
    assert settings.t2s_batch_rng() is True

    monkeypatch.delenv("ADR_TTS_BATCH_SIZE", raising=False)
    assert settings.tts_batch_size() == 1
    monkeypatch.setenv("ADR_TTS_BATCH_SIZE", "4")
    assert settings.tts_batch_size() == 4
    monkeypatch.setenv("ADR_TTS_BATCH_SIZE", "abc")
    assert settings.tts_batch_size() == 1
    monkeypatch.setenv("ADR_TTS_BATCH_SIZE", "0")
    assert settings.tts_batch_size() == 1
