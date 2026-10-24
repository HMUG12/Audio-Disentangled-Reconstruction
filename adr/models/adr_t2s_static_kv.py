"""ADR t2s 静态 KV Cache 桥 (批次 63, v1.3.0 "Autopilot" 内核底座第一子批).

官方 decode_next_token 每步 torch.cat 重建 KV cache — 第 idx 步拷贝
src_len+idx 行, 1500 步 decode 的拷贝总量 O(L^2)。本桥只换 "KV 存储层":
预分配 (1, src_len+1500, D) buffer + 原地写 + narrow 视图, decode 步算子
与 T2SBlock.decode_next_token 逐一对齐 (torch_sdpa 分支, 含 ~attn_mask
取反语义); idx==0 直接调原 jit process_prompt 后 copy_ 进 buffer。bs=1
时 narrow 视图与 cat 的元素寻址一致 (仅 stride[0] 不同, B=1 寻址无效),
sample() 每步恰一次 exponential_ RNG 消耗不变 — 与原路径逐 token 位等。

分派 (TTS.py 每请求先把 infer_panel 重置为原版方法再挂桥, install 时
fallback 捕获的是干净基线或 cudagraph 桥, 无链式套壳):
- bs>1 / prompt 缺席 (ref-free, batch_infer 的转 naive 怪癖路径不复制) → fallback
- streaming_mode=True → _static_kv_naive 流式生成器 (原版 naive 逐行复刻)
- fallback 是 naive_batched (朴素模式) → _static_kv_naive 非流式 + next() 包装
- 其余 (batch_infer / cudagraph 桥) → _static_kv_batch1 (batch_infer 逐行复刻,
  左 pad 语义; tqdm→range(1500), 跳过 EOS 后 k_cache index_select — bs=1
  命中 EOS 必 break, cache 不再被读)

无 CUDA 限制 (纯张量算子, CPU 可跑); 无缓存无锁 (buffer 每次调用新建,
生命周期限定单次合成)。与 ADR_T2S_CUDAGRAPH 不强制互斥: 本补丁排在
CUDA Graph 之后安装, 同开时本桥的 infer_panel 覆盖生效 (非流式走本桥,
bs>1 fallback 给桥)。

_last_kv: 非流式调用结束时的 (k_bufs, v_bufs, kv_len) 观测点, 供测试
断言 buffer 实际使用长度 (流式生成器不被耗尽时不设置, 属预期)。
"""
import os
from typing import List

import torch
import torch.nn.functional as F

_last_kv = None


def _decode_step(block, x, k_buf, v_buf, pos, attn_mask):
    """单层 decode: 与 T2SBlock.decode_next_token 逐算子对齐 (torch_sdpa
    分支), 仅 KV 存储换 buffer 原地写 + narrow 视图。

    block 为 @torch.jit.script T2SBlock 实例 (ScriptObject), 属性可从
    Python 侧访问; mlp.forward 为 scripted 方法, Python 可直调。
    """
    q, k, v = F.linear(x, block.qkv_w, block.qkv_b).chunk(3, dim=-1)

    k_buf[:, pos:pos + 1] = k
    v_buf[:, pos:pos + 1] = v

    bsz, q_len = q.shape[0], q.shape[1]
    kv_len = pos + 1

    q = q.view(bsz, q_len, block.num_heads, -1).transpose(1, 2)
    k = k_buf[:, :kv_len].view(bsz, kv_len, block.num_heads, -1).transpose(1, 2)
    v = v_buf[:, :kv_len].view(bsz, kv_len, block.num_heads, -1).transpose(1, 2)

    attn = F.scaled_dot_product_attention(
        q, k, v, (~attn_mask) if attn_mask is not None else None)

    attn = attn.transpose(1, 2).reshape(bsz, q_len, -1)
    attn = F.linear(attn, block.out_w, block.out_b)

    x = x + attn
    x = F.layer_norm(x, [block.hidden_dim], block.norm_w1, block.norm_b1, block.norm_eps1)
    x = x + block.mlp.forward(x)
    x = F.layer_norm(x, [block.hidden_dim], block.norm_w2, block.norm_b2, block.norm_eps2)
    return x


def _make_buffers(model, xy_pos, src_len):
    tf = model.t2s_transformer
    dev, dt = xy_pos.device, xy_pos.dtype
    k_bufs = [torch.zeros(1, src_len + 1500, model.model_dim, device=dev, dtype=dt)
              for _ in range(tf.num_blocks)]
    v_bufs = [torch.zeros(1, src_len + 1500, model.model_dim, device=dev, dtype=dt)
              for _ in range(tf.num_blocks)]
    return k_bufs, v_bufs


def _copy_prompt_kv(model, k_cache, v_cache, k_bufs, v_bufs, src_len):
    for i in range(model.t2s_transformer.num_blocks):
        k_bufs[i][:, :src_len] = k_cache[i]
        v_bufs[i][:, :src_len] = v_cache[i]


def _decode_pass(model, x, k_bufs, v_bufs, pos, attn_mask):
    tf = model.t2s_transformer
    for i in range(tf.num_blocks):
        x = _decode_step(tf.blocks[i], x, k_bufs[i], v_bufs[i], pos, attn_mask)
    return x


def _static_kv_batch1(model, x, x_lens, prompts, bert_feature, top_k=-100,
                      top_p=100, early_stop_num=-1, temperature=1.0,
                      repetition_penalty=1.35, max_len=None):
    """infer_panel_batch_infer @ bs=1 逐行复刻。差异仅: tqdm→range(1500)、
    KV 存储层换静态 buffer、跳过 k_cache index_select。"""
    from AR.models.utils import sample, make_pad_mask_left

    global _last_kv
    with torch.no_grad():
        if max_len is None:
            max_len = x_lens.max()

        x_item = x[0]
        bert_item = bert_feature[0]
        x_item = model.ar_text_embedding(x_item.unsqueeze(0))
        x_item = x_item + model.bert_proj(bert_item.transpose(0, 1).unsqueeze(0))
        x_item = model.ar_text_position(x_item).squeeze(0)
        x_item = (
            F.pad(x_item, (0, 0, max_len - x_item.shape[0], 0), value=0)
            if x_item.shape[0] < max_len else x_item
        )  ### padding left
        x: torch.Tensor = x_item.unsqueeze(0)

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        stop = False
        ref_free = False

        y_emb = model.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_lens = torch.LongTensor([y_emb.shape[1]] * y_emb.shape[0]).to(x.device)
        y_pos = model.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        ##### create mask #####
        bsz = x.shape[0]
        src_len = x_len + y_len
        y_paddind_mask = make_pad_mask_left(y_lens, y_len)
        x_paddind_mask = make_pad_mask_left(x_lens, max_len)

        # (bsz, x_len + y_len)
        padding_mask = torch.concat([x_paddind_mask, y_paddind_mask], dim=1)

        x_mask = F.pad(
            torch.zeros(x_len, x_len, dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )

        y_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1),
            (x_len, 0),
            value=False,
        )

        causal_mask = torch.concat([x_mask, y_mask], dim=0).view(1, src_len, src_len).repeat(bsz, 1, 1).to(x.device)

        padding_mask = padding_mask.view(bsz, 1, src_len).repeat(1, src_len, 1)

        attn_mask: torch.Tensor = causal_mask.logical_or(padding_mask)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, model.num_head, -1, -1).bool()

        ###### decode #####
        y_list = [None] * y.shape[0]
        batch_idx_map = list(range(y.shape[0]))
        idx_list = [None] * y.shape[0]
        k_bufs = None
        v_bufs = None
        for idx in range(1500):
            if idx == 0:
                xy_dec, k_cache, v_cache = model.t2s_transformer.process_prompt(xy_pos, attn_mask, None)
                k_bufs, v_bufs = _make_buffers(model, xy_pos, src_len)
                _copy_prompt_kv(model, k_cache, v_cache, k_bufs, v_bufs, src_len)
            else:
                xy_dec = _decode_pass(model, xy_pos, k_bufs, v_bufs, src_len + idx - 1, attn_mask)
            logits = model.ar_predict_layer(xy_dec[:, -1])

            if idx == 0:
                attn_mask = F.pad(attn_mask[:, :, -1].unsqueeze(-2), (0, 1), value=False)
            else:
                attn_mask = F.pad(attn_mask, (0, 1), value=False)

            if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
                logits = logits[:, :-1]

            samples = sample(
                logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
            )[0]

            y = torch.concat([y, samples], dim=1)

            ####### 移除batch中已经生成完毕的序列,进一步优化计算量
            tokens = torch.argmax(logits, dim=-1)
            reserved_idx_of_batch_for_y = None
            if (model.EOS in samples[:, 0]) or (model.EOS in tokens):  ###如果生成到EOS，则停止
                l1 = samples[:, 0] == model.EOS
                l2 = tokens == model.EOS
                l = l1.logical_or(l2)
                removed_idx_of_batch_for_y = torch.where(l == True)[0].tolist()
                reserved_idx_of_batch_for_y = torch.where(l == False)[0]
                for i in removed_idx_of_batch_for_y:
                    batch_index = batch_idx_map[i]
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y[i, :-1]

                batch_idx_map = [batch_idx_map[i] for i in reserved_idx_of_batch_for_y.tolist()]

            # 只保留batch中未生成完毕的序列 (bs=1: EOS 命中后必 break,
            # k_cache/v_cache 不再被读 — buffer 无需 index_select)
            if reserved_idx_of_batch_for_y is not None:
                y = torch.index_select(y, dim=0, index=reserved_idx_of_batch_for_y)
                attn_mask = torch.index_select(attn_mask, dim=0, index=reserved_idx_of_batch_for_y)

            if (early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num) or idx == 1499:
                print("use early stop num:", early_stop_num)
                stop = True
                for i, batch_index in enumerate(batch_idx_map):
                    batch_index = batch_idx_map[i]
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y[i, :-1]

            if None not in idx_list:
                stop = True

            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                _last_kv = (k_bufs, v_bufs, src_len + idx)
                break

            ####################### update next step ###################################
            y_emb = model.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * model.ar_audio_position.x_scale + model.ar_audio_position.alpha * model.ar_audio_position.pe[
                :, y_len + idx
            ].to(dtype=y_emb.dtype, device=y_emb.device)

        if None in idx_list:
            for i in range(x.shape[0]):
                if idx_list[i] is None:
                    idx_list[i] = 1500 - 1  ###如果没有生成到EOS，就用最大长度代替

        if ref_free:
            return y_list, [0] * x.shape[0]
        return y_list, idx_list


def _static_kv_naive(model, x, x_lens, prompts, bert_feature, top_k=-100,
                     top_p=100, early_stop_num=-1, temperature=1.0,
                     repetition_penalty=1.35, streaming_mode=False,
                     chunk_length=24, **kwargs):
    """infer_panel_naive @ bs=1 逐行复刻 (生成器)。差异仅: KV 存储层换
    静态 buffer (idx==0 分支内创建), 非流式结束时设 _last_kv 观测点。"""
    from AR.models.utils import sample

    global _last_kv
    mute_emb_sim_matrix = kwargs.get("mute_emb_sim_matrix", None)
    chunk_split_thershold = kwargs.get("chunk_split_thershold", 0.3)
    check_token_num = 2

    with torch.no_grad():
        x = model.ar_text_embedding(x)
        x = x + model.bert_proj(bert_feature.transpose(1, 2))
        x = model.ar_text_position(x)

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False

        k_bufs = None
        v_bufs = None
        ###################  first step ##########################
        if y is not None:
            y_emb = model.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            y_pos = model.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_pos = None
            xy_pos = x
            y = torch.zeros(x.shape[0], 0, dtype=torch.int, device=x.device)
            ref_free = True

        bsz = x.shape[0]
        src_len = x_len + y_len
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),  ###xx的纯0扩展到xx纯0+xy纯1，(x,x+y)
            value=True,
        )
        y_attn_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = (
            torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            .unsqueeze(0)
            .expand(bsz * model.num_head, -1, -1)
            .view(bsz, model.num_head, src_len, src_len)
            .to(device=x.device, dtype=torch.bool)
        )

        token_counter = 0
        curr_ptr = prefix_len
        # Track E (批次33): EOS 旗标设备侧累积, 每 ADR_AR_SYNC_EVERY 步一次
        # D2H 批量结算 — 复刻原实现 (ADR_AR_SYNC_LEGACY=1 回退逐步同步)。
        _every = 1 if os.environ.get("ADR_AR_SYNC_LEGACY") == "1" else max(
            1, int(os.environ.get("ADR_AR_SYNC_EVERY", "8") or "8"))
        _eos_flags: List[torch.Tensor] = []
        for idx in range(1500):
            token_counter += 1
            if xy_attn_mask is not None:
                xy_dec, k_cache, v_cache = model.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)
                k_bufs, v_bufs = _make_buffers(model, xy_pos, src_len)
                _copy_prompt_kv(model, k_cache, v_cache, k_bufs, v_bufs, src_len)
            else:
                xy_dec = _decode_pass(model, xy_pos, k_bufs, v_bufs, src_len + idx - 1, None)

            logits = model.ar_predict_layer(xy_dec[:, -1])

            if idx == 0:
                xy_attn_mask = None
            if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
                logits = logits[:, :-1]

            samples = sample(
                logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
            )[0]

            y = torch.concat([y, samples], dim=1)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            # Track E: EOS 旗标设备侧累积 (0 同步); 终止步/检查点批量 D2H 结算
            _eos_now = (torch.argmax(logits, dim=-1)[0] == model.EOS) | (samples[0, 0] == model.EOS)

            if stop or idx == 1499:
                # 终止步 (早停/1499): 挂起窗口 + 本步旗标一并结算
                _win = torch.stack(_eos_flags + [_eos_now]).cpu()
                _eos_flags.clear()
                if bool(_win.any()):
                    _eos_idx = idx - (_win.shape[0] - 1) + int(torch.nonzero(_win)[0, 0])
                    y = y[:, : prefix_len + _eos_idx]
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                if streaming_mode:
                    yield y[:, curr_ptr:] if curr_ptr < y.shape[1] else None, True
                else:
                    _last_kv = (k_bufs, v_bufs, src_len + idx)
                break

            _eos_flags.append(_eos_now)

            if (idx + 1) % _every:
                ####################### update next step ###################################
                y_emb = model.ar_audio_embedding(y[:, -1:])
                xy_pos = y_emb * model.ar_audio_position.x_scale + model.ar_audio_position.alpha * model.ar_audio_position.pe[
                    :, y_len + idx
                ].to(dtype=y_emb.dtype, device=y_emb.device)
                continue

            # ── 检查点 (每 _every 步): 窗口 EOS 旗标批量结算 — 命中则按首现
            # 位置精确截断, EOS 后的 token 永不外发 ──
            _win = torch.stack(_eos_flags).cpu()
            _eos_flags.clear()
            if bool(_win.any()):
                _eos_idx = idx - (_win.shape[0] - 1) + int(torch.nonzero(_win)[0, 0])
                y = y[:, : prefix_len + _eos_idx]
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                if streaming_mode:
                    yield y[:, curr_ptr:] if curr_ptr < y.shape[1] else None, True
                else:
                    _last_kv = (k_bufs, v_bufs, src_len + idx)
                break

            if streaming_mode and (mute_emb_sim_matrix is not None) and (token_counter >= chunk_length + check_token_num):
                score = mute_emb_sim_matrix[y[0, curr_ptr:]] - chunk_split_thershold
                score[score < 0] = -1
                score[:-1] = score[:-1] + score[1:]  ##考虑连续两个token
                argmax_idx = score.argmax()

                if score[argmax_idx] >= 0 and argmax_idx + 1 >= chunk_length:
                    yield y[:, curr_ptr:], False
                    token_counter -= argmax_idx + 1
                    curr_ptr += argmax_idx + 1

            elif streaming_mode and (mute_emb_sim_matrix is None) and (token_counter >= chunk_length):
                yield y[:, -token_counter:], False
                curr_ptr += token_counter
                token_counter = 0

            ####################### update next step ###################################
            y_emb = model.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * model.ar_audio_position.x_scale + model.ar_audio_position.alpha * model.ar_audio_position.pe[
                :, y_len + idx
            ].to(dtype=y_emb.dtype, device=y_emb.device)

        if not streaming_mode:
            if os.environ.get("ADR_AR_DEBUG_TOKENS") == "1":
                import hashlib
                _gen = y[:, prefix_len:]
                _b = _gen.detach().cpu().numpy().tobytes()
                print(f"[AR-DBG] every={_every} prefix={prefix_len} refree={ref_free} "
                      f"gen={_gen.shape[1]} idx={idx} first16={_gen[:, :16][0].tolist()} "
                      f"sha1={hashlib.sha1(_b).hexdigest()}")
            if ref_free:
                yield y, 0
            yield y, idx


def install_static_kv_infer_panel(model, tts):
    """返回签名兼容 infer_panel 的静态 KV 桥函数。

    model: T2SModel; tts: TTS_infer_pack.TTS 实例 (占位, 与 cudagraph 桥
    签名对齐)。
    """
    fallback = model.infer_panel   # install 时的干净基线 (或 cudagraph 桥)
    _naive_batched = model.infer_panel_naive_batched

    def infer_panel_static_kv(all_phoneme_ids, all_phoneme_lens, prompt,
                              all_bert_features, top_k=15, top_p=1.0,
                              temperature=1.0, early_stop_num=-1, max_len=None,
                              repetition_penalty=1.35, streaming_mode=False,
                              chunk_length=16, **kw):
        # TTS.py 传参形态不一 (同 cudagraph 桥): 非流式 list[[T_i]] 或
        # tensor[B,T]; 流式已取 [0].unsqueeze(0) = [1,T]。归一为 [T]
        if isinstance(all_phoneme_ids, (list, tuple)):
            ids_list = list(all_phoneme_ids)
        else:
            ids_list = [t for t in all_phoneme_ids]
        ids0 = ids_list[0]
        if ids0.dim() == 2:      # [1,T] (流式调用点) → [T]
            ids0 = ids0.squeeze(0)
        if len(ids_list) != 1:   # buffer 绑定 bs=1, 批量回退
            return fallback(all_phoneme_ids, all_phoneme_lens, prompt,
                            all_bert_features, top_k=top_k, top_p=top_p,
                            temperature=temperature,
                            early_stop_num=early_stop_num, max_len=max_len,
                            repetition_penalty=repetition_penalty,
                            streaming_mode=streaming_mode,
                            chunk_length=chunk_length, **kw)
        bert0 = all_bert_features[0]
        bert0 = bert0.squeeze(0) if bert0.dim() == 3 else bert0  # [1024,T]
        if streaming_mode:
            return _static_kv_naive(model, ids0.unsqueeze(0), all_phoneme_lens,
                                    prompt, bert0.unsqueeze(0), top_k, top_p,
                                    early_stop_num, temperature,
                                    repetition_penalty, streaming_mode=True,
                                    chunk_length=chunk_length, **kw)
        if prompt is None:
            # batch_infer 不支持 ref-free; 原版转 naive_batched 的怪癖路径
            # (不透传 repetition_penalty) 不复制 — 回退保真
            return fallback(all_phoneme_ids, all_phoneme_lens, prompt,
                            all_bert_features, top_k=top_k, top_p=top_p,
                            temperature=temperature,
                            early_stop_num=early_stop_num, max_len=max_len,
                            repetition_penalty=repetition_penalty,
                            streaming_mode=streaming_mode,
                            chunk_length=chunk_length, **kw)
        if fallback == _naive_batched:   # 朴素模式 (bound method 用 == 比较)
            y, idx = next(_static_kv_naive(
                model, ids0.unsqueeze(0), all_phoneme_lens, prompt,
                bert0.unsqueeze(0), top_k, top_p, early_stop_num, temperature,
                repetition_penalty, streaming_mode=False, **kw))
            return [y[0]], [idx]   # 对齐 naive_batched: 元素去 batch 维
        return _static_kv_batch1(model, [ids0], all_phoneme_lens, prompt,
                                 [bert0], top_k, top_p, early_stop_num,
                                 temperature, repetition_penalty, max_len=max_len)

    return infer_panel_static_kv
