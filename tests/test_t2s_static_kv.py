"""批次 63: t2s 静态 KV Cache 桥等价性测试 (CPU 小模型, 逐 token 位等)。

与原路径 (decode 每步 torch.cat 重建 KV) 对比:
1. 单步存储层: _decode_step vs T2SBlock.decode_next_token (masked/unmasked)
2. 端到端非流式: _static_kv_batch1 vs infer_panel_batch_infer @ bs=1
3. install 分派: batch_infer 绑定 → batch1; naive_batched 绑定 → naive 包装
4. 端到端流式: bridge(streaming) vs infer_panel_naive 逐 chunk 位等
5. 流式 mute 分支: mute_emb_sim_matrix 路径逐 chunk 位等

RNG 纪律: sample() 每步恰一次 multinomial, 两条路径前各自 torch.manual_seed
重置全局流 — 每步输入位等 + RNG 状态一致 → samples 位等 → y 位等。
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

import adr.models.adr_t2s_static_kv as skv  # noqa: E402
from AR.models.t2s_model import Text2SemanticDecoder  # noqa: E402

# embedding_dim 必须 == hidden_dim (bert_proj/embedding 输出直接进 d_model 维 block)
CONFIG = {"model": {
    "hidden_dim": 32, "embedding_dim": 32, "head": 2, "n_layer": 2,
    "vocab_size": 24, "phoneme_vocab_size": 32, "dropout": 0.1, "EOS": 23,
}}

MAX_LEN = 16   # x_len 13 → 左 pad 到 16 (覆盖 left-pad 分支)
X_LEN = 13
PREFIX = 5     # prompt 长度


@pytest.fixture()
def model():
    torch.manual_seed(42)
    m = Text2SemanticDecoder(CONFIG)
    m.eval()
    return m


def _inputs(seed=7):
    """固定输入: phoneme ids / lens / prompt / bert (TTS.py 真实形态)。"""
    g = torch.Generator().manual_seed(seed)
    ids0 = torch.randint(0, 32, (X_LEN,), generator=g)
    prompt = torch.randint(0, 23, (1, PREFIX), generator=g)
    bert0 = torch.randn(1024, X_LEN, generator=g)
    return ids0, torch.LongTensor([X_LEN]), prompt, [bert0]


def test_decode_step_bitwise(model):
    """_decode_step vs T2SBlock.decode_next_token: 输出与 KV 三者位等。"""
    block = model.t2s_transformer.blocks[0]
    D, H = model.model_dim, model.num_head
    src_len = 7
    g = torch.Generator().manual_seed(1)
    k_cache = torch.randn(1, src_len, D, generator=g)
    v_cache = torch.randn(1, src_len, D, generator=g)
    x = torch.randn(1, 1, D, generator=g)

    mask = torch.zeros(1, H, 1, src_len + 1, dtype=torch.bool)
    mask[:, :, :, ::3] = True
    for attn_mask in (None, mask):
        x_ref, k_ref, v_ref = block.decode_next_token(x, k_cache, v_cache, attn_mask)

        k_buf = torch.zeros(1, src_len + 1, D)
        v_buf = torch.zeros(1, src_len + 1, D)
        k_buf[:, :src_len] = k_cache
        v_buf[:, :src_len] = v_cache
        x_new = skv._decode_step(block, x, k_buf, v_buf, src_len, attn_mask)

        assert torch.equal(x_ref, x_new)
        assert torch.equal(k_ref, k_buf)   # 写入位置/值与 cat 一致
        assert torch.equal(v_ref, v_buf)


def test_batch1_bitwise(model):
    """_static_kv_batch1 vs infer_panel_batch_infer @ bs=1: y/idx 位等。"""
    ids0, x_lens, prompt, bert = _inputs()
    kw = dict(top_k=15, top_p=1.0, early_stop_num=-1, temperature=1.0,
              repetition_penalty=1.35, max_len=MAX_LEN)

    torch.manual_seed(42)
    y_ref, idx_ref = model.infer_panel_batch_infer([ids0], x_lens, prompt, bert, **kw)

    skv._last_kv = None
    torch.manual_seed(42)
    y_new, idx_new = skv._static_kv_batch1(model, [ids0], x_lens, prompt, bert, **kw)

    assert idx_new == idx_ref
    assert y_ref[0].dim() == 1 and torch.equal(y_ref[0], y_new[0])

    # _last_kv 观测点: buffer 实际使用长度 = src_len + 终止 idx
    assert skv._last_kv is not None
    k_bufs, v_bufs, kv_len = skv._last_kv
    assert kv_len == MAX_LEN + PREFIX + idx_ref[0]
    assert k_bufs[0].shape == (1, MAX_LEN + PREFIX + 1500, 32)
    assert v_bufs[-1].shape == (1, MAX_LEN + PREFIX + 1500, 32)


def test_dispatch_batch_infer(model):
    """install 分派: fallback 为 batch_infer 绑定 → 非流式走 batch1 分支。"""
    ids0, x_lens, prompt, bert = _inputs()
    model.infer_panel = model.infer_panel_batch_infer   # 模拟 TTS.py 重绑
    bridge = skv.install_static_kv_infer_panel(model, None)

    kw = dict(top_k=15, top_p=1.0, early_stop_num=-1, temperature=1.0,
              repetition_penalty=1.35, max_len=MAX_LEN)
    torch.manual_seed(42)
    y_ref, idx_ref = model.infer_panel_batch_infer([ids0], x_lens, prompt, bert, **kw)
    torch.manual_seed(42)
    y_new, idx_new = bridge([ids0], x_lens, prompt, bert, **kw)

    assert idx_new == idx_ref
    assert torch.equal(y_ref[0], y_new[0])


def test_dispatch_naive_batched(model):
    """install 分派: fallback 为 naive_batched 绑定 → 非流式走 naive 包装
    (next() 取终值, 元素去 batch 维)。"""
    ids0, x_lens, prompt, bert = _inputs()
    model.infer_panel = model.infer_panel_naive_batched   # 模拟朴素模式重绑
    bridge = skv.install_static_kv_infer_panel(model, None)

    kw = dict(top_k=15, top_p=1.0, early_stop_num=-1, temperature=1.0,
              repetition_penalty=1.35, max_len=MAX_LEN)
    torch.manual_seed(42)
    y_ref, idx_ref = model.infer_panel_naive_batched([ids0], x_lens, prompt, bert,
                                                     top_k=15, top_p=1.0,
                                                     early_stop_num=-1,
                                                     temperature=1.0,
                                                     repetition_penalty=1.35)
    torch.manual_seed(42)
    y_new, idx_new = bridge([ids0], x_lens, prompt, bert, **kw)

    assert idx_new == idx_ref
    assert y_new[0].dim() == 1   # 对齐 naive_batched: y_list.append(y[0])
    assert torch.equal(y_ref[0], y_new[0])


def _collect_chunks(gen):
    return [(None if c is None else c.clone(), bool(last)) for c, last in gen]


def _assert_chunks_equal(chunks_ref, chunks_new):
    assert len(chunks_ref) == len(chunks_new)
    for (c_ref, last_ref), (c_new, last_new) in zip(chunks_ref, chunks_new):
        assert last_new == last_ref
        if c_ref is None:
            assert c_new is None
        else:
            assert c_new is not None and torch.equal(c_ref, c_new)


def test_streaming_chunks_bitwise(model):
    """流式等价: bridge(streaming) vs infer_panel_naive 逐 chunk 位等。"""
    ids0, x_lens, prompt, bert = _inputs()
    kw = dict(top_k=15, top_p=1.0, early_stop_num=-1, temperature=1.0,
              repetition_penalty=1.35)

    torch.manual_seed(42)
    gen_ref = model.infer_panel_naive(ids0.unsqueeze(0), x_lens, prompt,
                                      bert[0].unsqueeze(0),
                                      streaming_mode=True, chunk_length=8, **kw)
    chunks_ref = _collect_chunks(gen_ref)

    torch.manual_seed(42)
    bridge = skv.install_static_kv_infer_panel(model, None)
    # TTS.py 流式调用点形态: ids[0].unsqueeze(0) → list[tensor(1,T)]
    gen_new = bridge([ids0.unsqueeze(0)], x_lens, prompt, [bert[0]],
                     streaming_mode=True, chunk_length=8, **kw)
    chunks_new = _collect_chunks(gen_new)

    _assert_chunks_equal(chunks_ref, chunks_new)
    assert chunks_ref[-1][1] is True   # 生成器正常耗尽, 末块 is_last


def test_streaming_mute_bitwise(model):
    """流式 mute 分支: mute_emb_sim_matrix 路径逐 chunk 位等。"""
    ids0, x_lens, prompt, bert = _inputs()
    g = torch.Generator().manual_seed(9)
    mute = torch.rand(24, generator=g)   # vocab_size=24
    kw = dict(top_k=15, top_p=1.0, early_stop_num=-1, temperature=1.0,
              repetition_penalty=1.35)

    torch.manual_seed(42)
    gen_ref = model.infer_panel_naive(ids0.unsqueeze(0), x_lens, prompt,
                                      bert[0].unsqueeze(0),
                                      streaming_mode=True, chunk_length=6,
                                      mute_emb_sim_matrix=mute,
                                      chunk_split_thershold=0.3, **kw)
    chunks_ref = _collect_chunks(gen_ref)

    torch.manual_seed(42)
    bridge = skv.install_static_kv_infer_panel(model, None)
    gen_new = bridge([ids0.unsqueeze(0)], x_lens, prompt, [bert[0]],
                     streaming_mode=True, chunk_length=6,
                     mute_emb_sim_matrix=mute, chunk_split_thershold=0.3, **kw)
    chunks_new = _collect_chunks(gen_new)

    _assert_chunks_equal(chunks_ref, chunks_new)
    assert chunks_ref[-1][1] is True
