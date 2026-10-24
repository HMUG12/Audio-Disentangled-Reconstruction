"""批次 63 次子批: t2s 分桶 CUDA Graph runner 等价性测试。

CPU (小模型, eager 走分桶视图):
A. ladder=[2500] vs 官方 eager 循环逐行转写 (_ref_eager_decode) — y 位等
   (满档前缀视图 == 全量 buffer, 验证循环转写与视图接线)
B. 子档 ladder 结构断言: dispatch_log 单调不减 / 落档 / 跨档 / 首档
C. 子档视图实际改变注意力 (零填充假质量被裁掉) — 分桶生效的直接证据
D. parse_buckets / _bucket_for 边界; _PrefixView 写真 buffer + 返回前缀
GPU (skipif 无 CUDA):
E. 官方 CUDAGraphRunner vs ours ladder=[2500] graph 模式 — y 位等
F. 子档 graph 模式: graphs 键 == 实际用过的档集合

RNG 纪律: Sampler 每步恰一次 exponential_, 两路径前各自 manual_seed。
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
GSV_DIR = REPO / "third_party" / "gpt_sovits" / "GPT_SoVITS"
for _p in (str(REPO), str(GSV_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adr.models.adr_t2s_bucketed as bk  # noqa: E402
from AR.models.structs_cudagraph import T2SRequest, T2SSession  # noqa: E402
from AR.models.t2s_model_cudagraph import KVCacheNHD, T2SDecoder  # noqa: E402

CONFIG = {"model": {
    "hidden_dim": 32, "embedding_dim": 32, "head": 2, "n_layer": 2,
    "vocab_size": 24, "phoneme_vocab_size": 32, "dropout": 0.1, "EOS": 23,
}}
X_LEN = 13
PREFIX = 5
PREFILL = X_LEN + PREFIX


def _decoder(seed=42):
    torch.manual_seed(seed)
    return T2SDecoder(CONFIG, max_batch_size=1).eval()


def _request(seed=7, early_stop_num=30, use_cuda_graph=False, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    ids0 = torch.randint(0, 32, (X_LEN,), generator=g).to(device)
    prompt = torch.randint(0, 23, (1, PREFIX), generator=g).to(device)
    bert0 = torch.randn(1024, X_LEN, generator=g).to(device)
    return T2SRequest([ids0], torch.LongTensor([X_LEN]).to(device), prompt,
                      [bert0], valid_length=1, top_k=15, top_p=1.0,
                      temperature=1.0, early_stop_num=early_stop_num,
                      repetition_penalty=1.35, use_cuda_graph=use_cuda_graph)


@torch.no_grad()
def _ref_eager_decode(decoder, request, device=torch.device("cpu")):
    """官方 CUDAGraphRunner._handle_request eager 分支逐行转写 (全量 KV, 无 graph)。"""
    kv_cache = decoder.init_cache(1)
    with device:
        session = T2SSession(decoder, request, device=device, dtype=torch.float32)
        input_pos = session.input_pos.clone()
        for idx in range(1500):
            if idx == 0:
                xy_dec = decoder.h.prefill(session.xy_pos, session.attn_mask_nested, kv_cache)
                xy_dec = torch.stack([t[[-1]] for t in xy_dec.unbind()])
            else:
                xy_dec = decoder.h.forward(input_pos, session.xy_pos, kv_cache)
            logits = decoder.ar_predict_layer(xy_dec[:, -1])
            input_pos.add_(1)
            if idx == 0:
                logits[:, -1] = float("-inf")
            samples = session.sampler.sample(
                logits=logits, previous_tokens=session.y, top_k=request.top_k,
                top_p=request.top_p, repetition_penalty=request.repetition_penalty,
                temperature=request.temperature)
            session.y = torch.cat([session.y, samples], dim=1)
            argmax_token = torch.argmax(logits, dim=-1)
            EOS_mask = (argmax_token == decoder.EOS) | (samples.squeeze(1) == decoder.EOS)
            newly = (EOS_mask & (~session.completed)).nonzero()
            if newly.numel() > 0:
                session.y_results[newly[0]] = session.y[newly[0], session.y_len:-1].squeeze(0)
                session.completed[newly] = True
            if torch.all(session.completed).item():
                break
            if (request.early_stop_num != -1
                    and (session.y.size(1) - session.y_len) > request.early_stop_num) \
                    or idx == 1499:
                if not session.completed[0].item():
                    session.y_results[0] = session.y[0, session.y_len:]
                    session.completed[0] = True
                break
            y_emb = decoder.ar_audio_embedding(session.y[:, -1:])
            session.xy_pos = decoder.ar_audio_position.forward(input_pos - session.x_lens, y_emb)
        return session.y_results[: request.valid_length]


def _run(runner, request, seed=42):
    torch.manual_seed(seed)
    with torch.no_grad():
        r = runner.generate(request)
    assert r.status == "Success", r.traceback
    return r.result


# ---------- CPU ----------

def test_full_ladder_matches_reference_cpu():
    dec = _decoder()
    torch.manual_seed(42)
    y_ref = _ref_eager_decode(dec, _request())
    runner = bk.BucketedGraphRunner(dec, buckets=[2500])
    y_new = _run(runner, _request())
    assert torch.equal(y_ref[0], y_new[0])
    assert set(runner.dispatch_log) == {2500}
    assert runner.graphs == {}   # CPU 不捕获


def test_sub_ladder_dispatch_cpu():
    dec = _decoder()
    runner = bk.BucketedGraphRunner(dec, buckets=[24, 40, 2500])
    y = _run(runner, _request(early_stop_num=30))
    log = runner.dispatch_log
    n_steps = len(log)
    assert n_steps >= 1 and y[0].numel() >= 1
    assert log == sorted(log)                      # 单调不减
    assert set(log) <= {24, 40, 2500}
    assert log[0] == 24                            # PREFILL+1=19 ≤ 24
    for k, b in enumerate(log, start=1):           # 每步最小可容档
        assert b == runner._bucket_for(PREFILL + k) and b >= PREFILL + k
    if n_steps > 24 - PREFILL:
        assert len(set(log)) >= 2                  # 跨档


@torch.no_grad()
def test_prefix_view_changes_attention():
    """子档视图裁掉零填充槽 → 与全量 SDPA 输出不同; 满档视图与全量位等。"""
    dec = _decoder()
    caches = dec.init_cache(1)
    g = torch.Generator().manual_seed(3)
    L = 10
    for c in caches:
        c.k_cache[:, :L] = torch.randn(1, L, 2, 16, generator=g)
        c.v_cache[:, :L] = torch.randn(1, L, 2, 16, generator=g)
    x = torch.randn(1, 1, 32, generator=g)
    pos = torch.tensor([L + 1], dtype=torch.int32)

    def run(prefix):
        cs = copy.deepcopy(caches)
        views = cs if prefix is None else [bk._PrefixView(c, prefix) for c in cs]
        return dec.h.forward(pos, x, views)

    full = run(None)
    assert torch.equal(full, run(2500))
    assert not torch.allclose(full, run(16))


def test_parse_buckets_and_bucket_for():
    default = [256, 384, 512, 768, 1024, 1280, 1536, 2048, 2500]
    for s in (None, "", "1", "default", " DEFAULT "):
        assert bk.parse_buckets(s) == default
    assert bk.parse_buckets("512,256,256") == [256, 512, 2500]
    assert bk.parse_buckets("2500") == [2500]
    with pytest.raises(ValueError):
        bk.parse_buckets("8")
    with pytest.raises(ValueError):
        bk.parse_buckets("3000")
    with pytest.raises(ValueError):
        bk.parse_buckets("abc")

    runner = bk.BucketedGraphRunner(_decoder(), buckets=[256, 512])
    assert runner.buckets == [256, 512, 2500]
    assert runner._bucket_for(1) == 256
    assert runner._bucket_for(256) == 256
    assert runner._bucket_for(257) == 512
    assert runner._bucket_for(2500) == 2500
    assert runner._bucket_for(9999) == 2500        # 钳末档


def test_prefix_view_update():
    c = KVCacheNHD(1, 32, 2, 16)
    c.k_cache.normal_()
    c.v_cache.normal_()
    before_k = c.k_cache.clone()
    k_val = torch.full((1, 1, 2, 16), 7.0)
    v_val = torch.full((1, 1, 2, 16), -7.0)
    k, v = bk._PrefixView(c, 8).update(torch.tensor([5], dtype=torch.int32), k_val, v_val)
    assert k.shape == (1, 8, 2, 16) and v.shape == (1, 8, 2, 16)
    assert torch.equal(k[:, 4], k_val[:, 0]) and torch.equal(v[:, 4], v_val[:, 0])
    assert k.data_ptr() == c.k_cache.data_ptr()    # 视图, 非拷贝
    assert torch.equal(c.k_cache[:, 4], k_val[:, 0])   # 真 buffer 被写
    assert torch.equal(c.k_cache[:, :4], before_k[:, :4])
    assert torch.equal(c.k_cache[:, 5:], before_k[:, 5:])


# ---------- GPU ----------

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")


@cuda_only
def test_full_ladder_matches_official_graph_gpu():
    from AR.models.t2s_model_cudagraph import CUDAGraphRunner
    dev = torch.device("cuda")
    dec = _decoder()
    ref = CUDAGraphRunner(copy.deepcopy(dec), dev, torch.float32)
    ours = bk.BucketedGraphRunner(copy.deepcopy(dec), dev, torch.float32, [2500])
    import AR.models.t2s_model_cudagraph as cg
    old = cg.tqdm
    cg.tqdm = type("T", (), {"__call__": lambda s, it, **k: it,
                             "write": staticmethod(lambda *a, **k: None)})()
    try:
        y_ref = _run(ref, _request(use_cuda_graph=True, device=dev))
    finally:
        cg.tqdm = old
    y_new = _run(ours, _request(use_cuda_graph=True, device=dev))
    assert torch.equal(y_ref[0], y_new[0])
    assert set(ours.graphs) == {2500}


@cuda_only
def test_sub_ladder_graph_gpu():
    dev = torch.device("cuda")
    runner = bk.BucketedGraphRunner(_decoder(), dev, torch.float32, [24, 40, 2500])
    _run(runner, _request(early_stop_num=30, use_cuda_graph=True, device=dev))
    log = runner.dispatch_log
    assert log == sorted(log)
    assert set(runner.graphs) == set(log)
    # 第二次请求复用已捕获 graph, 不重捕
    ids = {b: id(g) for b, g in runner.graphs.items()}
    _run(runner, _request(seed=8, early_stop_num=30, use_cuda_graph=True, device=dev))
    for b, i in ids.items():
        assert id(runner.graphs[b]) == i
