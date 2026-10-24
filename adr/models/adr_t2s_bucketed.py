"""批次 63 次子批: t2s 分桶 CUDA Graph runner (批次 10 单 graph 漂移的正解)。

批次 10 缺陷: 官方 CUDAGraphRunner 单 graph 绑定全量 (1, 2500, H, D) KV,
decode 每步 SDPA 对 2500 行全跑且无 mask — [valid_len, 2500) 的零填充槽各贡献
exp(0)=1 的假 softmax 质量, 实测 AR 分布漂移 (同输入 45→254 token)。

分桶: 按 KV 有效长度捕获多张 graph, 每步 dispatch 到最小可容 bucket B, 假质量
(2500-L) → (B-L); 不是新近似, 而是向 masked 精确 softmax 收敛。

实现要点 (不碰模型结构):
- SDPA 读的是 KVCacheNHD.update() 的返回值 → 捕获期给每层套 _PrefixView 代理,
  返回 k_cache[:, :B] 前缀视图, 捕获的 SDPA kernel 永久只读前 B 行;
  replay 不经过 Python, 零运行时开销
- 单份 2500 buffer 全 bucket 共享: 新档前缀 [L, B) 本就是请求开头 empty() 的零
  → bucket 间零拷贝, 显存与单 graph 一致, 升档免费
- graph 懒捕获: 只为实际用到的档捕获
- dispatch: idx=k≥1 时写后有效行数 = prefill_len + k, 取最小 B ≥ 该值
- 最大档 2500 的 kernel 序列与官方单 graph 位同 → ladder=[2500] 与
  CUDAGraphRunner 输出位等 (测试锚点)

_handle_request 逐行对齐官方 t2s_model_cudagraph.CUDAGraphRunner (去 tqdm 输出;
RNG 纪律: Sampler 每步恰一次 exponential_)。
"""
from __future__ import annotations

import bisect
import time
import traceback
from typing import Dict, List, Optional, Sequence

import torch
from torch.cuda.graphs import CUDAGraph

from AR.models.structs_cudagraph import T2SRequest, T2SResult, T2SSession

_MAX_SEQ_LENGTH = 2500   # 官方 T2SDecoder 默认 max_seq_length
_DECODE_CAP = 1500       # 官方循环上限
_MIN_BUCKET = 16
DEFAULT_BUCKETS = "256,384,512,768,1024,1280,1536,2048,2500"


def _normalize_buckets(seq: Sequence[int], max_len: int = _MAX_SEQ_LENGTH) -> List[int]:
    out = sorted({int(b) for b in seq})
    for b in out:
        if not (_MIN_BUCKET <= b <= max_len):
            raise ValueError(f"bucket {b} 越界, 须在 [{_MIN_BUCKET}, {max_len}]")
    if not out or out[-1] != max_len:
        out.append(max_len)   # 末档强制全长, 保证任意长度可容
    return out


def parse_buckets(spec: Optional[str], max_len: int = _MAX_SEQ_LENGTH) -> List[int]:
    """ADR_T2S_CUDAGRAPH_BUCKETS 解析: 空/"1"/"default" → 默认档; 否则逗号分隔整数。"""
    s = (spec or "").strip().lower()
    if s in ("", "1", "default"):
        s = DEFAULT_BUCKETS
    return _normalize_buckets([int(p) for p in s.split(",") if p.strip()], max_len)


class _PrefixView:
    """KVCacheNHD 代理: 原地写入真 buffer, 返回前 prefix 行视图供 SDPA 读。"""

    __slots__ = ("cache", "prefix")

    def __init__(self, cache, prefix: int):
        self.cache = cache
        self.prefix = prefix

    def update(self, input_pos, k_val, v_val):
        k_out, v_out = self.cache.update(input_pos, k_val, v_val)
        return k_out[:, : self.prefix], v_out[:, : self.prefix]


class BucketedGraphRunner:
    def __init__(
        self,
        decoder_model,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        buckets: Optional[Sequence[int]] = None,
    ) -> None:
        assert dtype in {torch.float16, torch.bfloat16, torch.float32}
        self.device = device
        self.dtype = dtype
        self.decoder_model = decoder_model.to(device, dtype)
        max_len = int(decoder_model.h.max_seq_length)
        self.buckets = _normalize_buckets(
            buckets if buckets is not None else parse_buckets(None, max_len), max_len)
        self.graphs: Dict[int, CUDAGraph] = {}
        self.views: Dict[int, list] = {}
        self.dispatch_log: List[int] = []
        self.xy_pos_ = torch.rand(
            (1, 1, decoder_model.embedding_dim), device=device).to(dtype)
        self.xy_dec_ = torch.rand(
            (1, 1, decoder_model.embedding_dim), device=device).to(dtype)
        self.kv_cache = decoder_model.init_cache(1)   # 单份全长 buffer, 全档共享
        # 官方为 .cuda() 硬编码; 此处 .to(device) 以便 CPU 测试
        self.input_pos = torch.tensor([10]).int().to(device)

    def _bucket_for(self, valid_len: int) -> int:
        i = bisect.bisect_left(self.buckets, valid_len)
        return self.buckets[min(i, len(self.buckets) - 1)]

    def _views_for(self, bucket: int) -> list:
        v = self.views.get(bucket)
        if v is None:
            v = [_PrefixView(c, bucket) for c in self.kv_cache]
            self.views[bucket] = v
        return v

    def _handle_request(self, request: T2SRequest):
        with self.device:
            for i in self.kv_cache:
                i.empty()
            self.dispatch_log = []

            decoder = self.decoder_model
            session = T2SSession(decoder, request, device=self.device, dtype=self.dtype)
            assert session.bsz == 1, "分桶 graph 绑定 bs=1"
            self.input_pos.copy_(session.input_pos)
            prefill_len = int(session.prefill_len[0].item())
            use_graph = request.use_cuda_graph and torch.cuda.is_available() \
                and self.device.type == "cuda"

            t1 = 0.0
            infer_speed = 0.0
            bsz = session.y.size(0)

            for idx in range(_DECODE_CAP):
                if idx == 0:
                    xy_dec = decoder.h.prefill(
                        session.xy_pos, session.attn_mask_nested, self.kv_cache)
                    xy_dec = torch.stack([t[[-1]] for t in xy_dec.unbind()])
                else:
                    bucket = self._bucket_for(prefill_len + idx)
                    self.dispatch_log.append(bucket)
                    views = self._views_for(bucket)
                    if use_graph and bucket not in self.graphs:
                        self.xy_pos_.copy_(session.xy_pos)
                        self.graphs[bucket] = decoder.capture(
                            self.input_pos, self.xy_pos_, self.xy_dec_,
                            kv_caches=views)
                    graph = self.graphs.get(bucket) if use_graph else None
                    if graph is not None:
                        self.xy_pos_.copy_(session.xy_pos)
                        graph.replay()
                        xy_dec = self.xy_dec_.clone()
                    else:
                        xy_dec = decoder.h.forward(
                            self.input_pos, session.xy_pos, views)

                logits = decoder.ar_predict_layer(xy_dec[:, -1])
                self.input_pos.add_(1)

                if idx == 0:
                    logits[:, -1] = float("-inf")

                samples = session.sampler.sample(
                    logits=logits,
                    previous_tokens=session.y,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    temperature=request.temperature,
                )

                session.y = torch.cat([session.y, samples], dim=1)

                argmax_token = torch.argmax(logits, dim=-1)
                sample_token = samples.squeeze(1)
                EOS_mask = (argmax_token == decoder.EOS) | (sample_token == decoder.EOS)

                newly_done_mask = EOS_mask & (~session.completed)
                newly_done_indices = newly_done_mask.nonzero()

                if newly_done_indices.numel() > 0:
                    session.y_results[newly_done_indices[0]] = session.y[
                        newly_done_indices[0], session.y_len: -1
                    ].squeeze(0)
                    session.completed[newly_done_indices] = True

                if torch.all(session.completed).item():
                    if session.y.size(1) == 0:
                        session.y = torch.cat(
                            [session.y, torch.zeros_like(samples)], dim=1)
                    elif t1 > 0:
                        infer_speed = (idx - 1) / (time.perf_counter() - t1)
                    break

                if (
                    request.early_stop_num != -1
                    and (session.y.size(1) - session.y_len) > request.early_stop_num
                ) or idx == _DECODE_CAP - 1:
                    for i in range(bsz):
                        if not session.completed[i].item():
                            session.y_results[i] = session.y[i, session.y_len:]
                            session.completed[i] = True
                    break

                y_emb = decoder.ar_audio_embedding(session.y[:, -1:])
                session.xy_pos = decoder.ar_audio_position.forward(
                    self.input_pos - session.x_lens, y_emb)

                if idx == 2:
                    t1 = time.perf_counter()

                if idx % 100 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return session.y_results[: request.valid_length], infer_speed

    def generate(self, request: T2SRequest) -> T2SResult:
        try:
            result, infer_speed = self._handle_request(request)
            return T2SResult(result=result, infer_speed=infer_speed, status="Success")
        except Exception as e:
            return T2SResult(status="Error", exception=e,
                             traceback=traceback.format_exc())

    @staticmethod
    def load_decoder(weights_path, max_batch_size=1):
        from AR.models.t2s_model_cudagraph import CUDAGraphRunner
        return CUDAGraphRunner.load_decoder(weights_path, max_batch_size)
