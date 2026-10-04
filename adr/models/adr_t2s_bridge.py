"""ADR t2s CUDA Graph 桥 (批次 10): 把官方 CUDAGraphRunner 挂进 TTS_infer_pack。

实测 (output 探针, 2026-10-04): AR 41→153.6 it/s, 3.77x — 2s 级音频整体
合成 2.53s→~1.8s, 达成 "2 秒内" 目标。官方 CUDAGraphRunner 只接在
inference_webui (独立加载 decoder, 逐 token 循环 replay, KV cache 预分配
静态 shape), TTS_infer_pack 引擎路径没有; 本桥按官方 webui 同款用法补齐。

约束与防御:
- 仅 bs=1 走 CUDA graph (graph capture 绑定 bsz; 引擎每段单独 run 天然 bs=1);
  bs>1 回退原 infer_panel_batch_infer, 保证批量路径正确性
- runner 按 (ckpt, dtype) 缓存单例, 热换权重自动重建
- 流式模式: 完整 AR 后按 chunk_length 切块 yield — AR 提速 3.77x 后完整生成
  仅 0.3~0.5s, 首包反而比逐 token 切块更短 (实测批次 9 首包 2.17s)
- 官方 _handle_request 循环内 tqdm 会刷屏, install 时静默
"""
import os
import threading

_bridge_lock = threading.Lock()
_bridge_cache = {}   # (ckpt_path, dtype) -> CUDAGraphRunner


def install_cudagraph_infer_panel(model, tts):
    """返回签名兼容 infer_panel 的桥接函数。

    model: T2SModel (含 infer_panel_batch_infer 回退路径);
    tts: TTS_infer_pack.TTS 实例 (取 device/half/权重路径)。
    """
    import torch

    dev = torch.device(tts.configs.device)
    if dev.type != "cuda":
        return model.infer_panel   # 非 GPU 回退
    dtype = torch.float16 if tts.configs.is_half else torch.float32

    from AR.models import t2s_model_cudagraph as _cgmod
    if getattr(_cgmod.tqdm, "__module__", "") != "adr_t2s_bridge":
        class _SilentTqdm:   # 官方逐 token 循环用 tqdm 迭代 + tqdm.write, 全静默
            def __call__(self, it, **k):
                return it
            @staticmethod
            def write(msg, **k):
                pass
        _cgmod.tqdm = _SilentTqdm()

    ckpt = str(tts.configs.t2s_weights_path)
    with _bridge_lock:
        runner = _bridge_cache.get((ckpt, dtype))
        if runner is None:
            from AR.models.t2s_model_cudagraph import CUDAGraphRunner
            runner = CUDAGraphRunner(
                CUDAGraphRunner.load_decoder(ckpt), dev, dtype)
            _bridge_cache[(ckpt, dtype)] = runner

    from AR.models.structs_cudagraph import T2SRequest

    fallback = model.infer_panel   # bs>1 / 异常时的原路径

    def _run(req):
        with torch.no_grad():
            r = runner.generate(req)
        if r.exception is not None:
            raise RuntimeError("CUDAGraph T2S failed") from r.exception
        return r

    def infer_panel_adr(all_phoneme_ids, all_phoneme_lens, prompt,
                        all_bert_features, top_k=15, top_p=1.0,
                        temperature=1.0, early_stop_num=-1, max_len=None,
                        repetition_penalty=1.35, streaming_mode=False,
                        chunk_length=16, **kw):
        # TTS.py 传参形态不一: 非流式可能 list[[T_i]] 或 tensor[B,T];
        # 流式调用点已取 [0].unsqueeze(0) = [1,T]。统一归一为 list[[T_i]]
        if isinstance(all_phoneme_ids, (list, tuple)):
            ids_list = list(all_phoneme_ids)
        else:
            ids_list = [t for t in all_phoneme_ids]
        bsz = len(ids_list)
        if bsz != 1:   # graph 绑定 bs=1, 批量回退
            return fallback(all_phoneme_ids, all_phoneme_lens, prompt,
                            all_bert_features, top_k=top_k, top_p=top_p,
                            temperature=temperature,
                            early_stop_num=early_stop_num, max_len=max_len,
                            repetition_penalty=repetition_penalty,
                            streaming_mode=streaming_mode,
                            chunk_length=chunk_length, **kw)
        ids0 = ids_list[0]
        if ids0.dim() == 2:      # [1,T] (流式调用点) → [T]
            ids0 = ids0.squeeze(0)

        def _bert(b):
            return b.squeeze(0) if b.dim() == 3 else b   # 对齐官方 webui [1024,T]

        # ref-free / prompt_semantic 缺席时 TTS.py 传 None — T2SSession 要求 tensor
        nonlocal_prompt = prompt if prompt is not None else ids0.new_zeros((1, 0))

        if not streaming_mode:
            r = _run(T2SRequest(
                [ids0], all_phoneme_lens, nonlocal_prompt,
                [_bert(all_bert_features[0])], valid_length=1,
                top_k=top_k, top_p=top_p, temperature=temperature,
                early_stop_num=early_stop_num,
                repetition_penalty=repetition_penalty, use_cuda_graph=True))
            toks = [t.flatten() for t in r.result]
            return toks, [t.size(-1) for t in toks]

        def _gen():
            r = _run(T2SRequest(
                [ids0], all_phoneme_lens, nonlocal_prompt,
                [_bert(all_bert_features[0])], valid_length=1,
                top_k=top_k, top_p=top_p, temperature=temperature,
                early_stop_num=early_stop_num,
                repetition_penalty=repetition_penalty, use_cuda_graph=True))
            tokens = r.result[0].flatten()
            total = tokens.size(-1)
            ptr = 0
            while ptr < total:
                nxt = min(ptr + chunk_length, total)
                yield tokens[None, ptr:nxt], False
                ptr = nxt
            yield None, True

        return _gen()

    return infer_panel_adr
