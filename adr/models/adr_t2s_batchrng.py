"""ADR t2s per-seq RNG 批采样桥 (批次 65 第二子批, N=2 真并行的内核底座).

stock infer_panel_batch_infer 的采样走全局 RNG (utils.sample →
exponential_ 无 generator): 多行合批共享一条 RNG 流 — 行种子不可复现,
行间互相污染, 是跨请求合批 (第三子批) 的前置缺口。本桥只换 "采样层":
decode 循环逐算子对齐 infer_panel_batch_infer (左 pad / mask / EOS 逐出 /
early_stop 全保留), 采样改逐行 — 每行 logits 切片套行参数后用行独立
torch.Generator 抽 exponential_, 行间 RNG 零串扰:
- 同批两行不同种子 → 各行输出与该行单行同种子解码位等 (行隔离);
  某行提前 EOS 逐出后, 余行采样流不受扰动 (逐出不改行 generator)
- 行参数 (top_k/top_p/temperature/repetition_penalty) 支持逐行覆盖 —
  第三子批跨请求合批时每请求可携带自己的采样参数

行种子语义 (TTS.py run() 侧经 derive_row_seeds 生成): 段 0 = 请求种子
(set_seed 返回的 actual_seed, 与 stock 同种子首段位等 — 同为每步恰一次
exponential_ [1,V] 抽取); 其余段 = Python 全局 random (已被 set_seed
重播 → 请求内确定性, 全文+种子可整段复现); -1 → torch.Generator().seed()
(OS 熵, 不吃全局 RNG)。

分派 (TTS.py 每请求先把 infer_panel 重置为原版方法再挂桥, install 时
fallback 捕获干净基线, 无链式套壳):
- row_seeds 缺席 → fallback (stock 全局 RNG 行为, 零改动)
- prompt 缺席 (ref-free) / streaming_mode=True → fallback
  (naive 怪癖路径与流式生成器不复制)
- row_seeds 与批行数不一致 → fallback (防御)

无 CUDA 限制 (纯张量算子; 行 generator 按张量设备创建, CPU/CUDA 皆可)。
与 ADR_T2S_CUDAGRAPH / ADR_T2S_STATIC_KV 不强制互斥: 本补丁排在其后
安装, 同开时本桥的 infer_panel 覆盖生效 (覆盖面仅非流式批路径)。
"""
import os
from typing import Dict, List, Optional

import torch


class _Row:
    """批内一行的采样参数 + 独立 generator (槽位省内存, 批行数 ≤ 两位数)."""

    __slots__ = ("top_k", "top_p", "temperature", "repetition_penalty", "generator")


def derive_row_seeds(n: int, first_seed: int) -> List[int]:
    """行种子派生: 段 0 用请求种子 (set_seed 后的首段与 stock 同种子位等),
    其余段抽 Python 全局 random — run() 内 set_seed(actual_seed) 已重播
    该流, 请求内确定性 (同文本+同种子 → 全部段种子可复现)。"""
    import random

    n = max(1, int(n))
    seeds = [int(first_seed)] + [random.randint(0, 2**32 - 1) for _ in range(n - 1)]
    return seeds


def _build_rows(n: int, row_seeds: List[int],
                row_sampling: Optional[List[Dict]], device: torch.device,
                top_k: int, top_p: float, temperature: float,
                repetition_penalty: float) -> List[_Row]:
    rows: List[_Row] = []
    for i in range(n):
        r = _Row()
        r.top_k, r.top_p = top_k, top_p
        r.temperature, r.repetition_penalty = temperature, repetition_penalty
        if row_sampling is not None:
            ov = row_sampling[i] if i < len(row_sampling) else row_sampling[-1]
            if ov:
                r.top_k = ov.get("top_k", r.top_k)
                r.top_p = ov.get("top_p", r.top_p)
                r.temperature = ov.get("temperature", r.temperature)
                r.repetition_penalty = ov.get("repetition_penalty", r.repetition_penalty)
        gen = torch.Generator(device=device)
        seed = int(row_seeds[i]) if i < len(row_seeds) else -1
        gen.manual_seed(seed if seed >= 0 else torch.Generator().seed())
        r.generator = gen
        rows.append(r)
    return rows


def _sample_rows(logits: torch.Tensor, y: torch.Tensor,
                 local_rows: List[_Row]) -> torch.Tensor:
    """逐行采样: 行 logits 切片 → logits_to_probs(行参数) →
    exponential_(generator=行 generator) → argmax。

    与 stock sample()/multinomial_sample_one_no_sync 逐算子等价 (单行时
    exponential_ [1,V] 抽取顺序一致 → 同种子位等); 输出 dtype 与 stock
    一致 (int32), 由上游 torch.cat 提升。"""
    from AR.models.utils import logits_to_probs

    out = torch.empty(logits.shape[0], 1, dtype=torch.int, device=logits.device)
    for local_i, row in enumerate(local_rows):
        probs = logits_to_probs(
            logits[local_i:local_i + 1],
            previous_tokens=y[local_i:local_i + 1].long(),
            temperature=row.temperature,
            top_k=row.top_k,
            top_p=row.top_p,
            repetition_penalty=row.repetition_penalty,
        )
        q = torch.empty_like(probs).exponential_(1, generator=row.generator)
        out[local_i:local_i + 1] = torch.argmax(probs / q, dim=-1, keepdim=True).to(torch.int)
    return out


def _batch_rng_decode(model, x: List[torch.Tensor], x_lens: torch.Tensor,
                      prompts: torch.Tensor, bert_feature: List[torch.Tensor],
                      rows: List[_Row], top_k: int, top_p: float,
                      early_stop_num: int, temperature: float,
                      repetition_penalty: float, max_len) -> tuple:
    """infer_panel_batch_infer 逐行复刻, 差异仅: tqdm→range、采样换
    _sample_rows (行参数 + 行 generator)、active_row_ids 维护行映射。"""
    import torch.nn.functional as F
    from AR.models.utils import make_pad_mask_left

    max_len = max_len if max_len is not None else x_lens.max()
    x_list = []
    for x_item, bert_item in zip(x, bert_feature):
        x_item = model.ar_text_embedding(x_item.unsqueeze(0))
        x_item = x_item + model.bert_proj(bert_item.transpose(0, 1).unsqueeze(0))
        x_item = model.ar_text_position(x_item).squeeze(0)
        x_item = (
            F.pad(x_item, (0, 0, max_len - x_item.shape[0], 0), value=0)
            if x_item.shape[0] < max_len
            else x_item
        )  ### padding left
        x_list.append(x_item)
    x: torch.Tensor = torch.stack(x_list, dim=0)

    # AR Decoder
    y = prompts

    x_len = x.shape[1]

    k_cache = None
    v_cache = None

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
    active_row_ids = list(range(y.shape[0]))  # 本桥新增: 批位置 → 全局行 id
    idx_list = [None] * y.shape[0]
    stop = False
    for idx in range(1500):
        if idx == 0:
            xy_dec, k_cache, v_cache = model.t2s_transformer.process_prompt(xy_pos, attn_mask, None)
        else:
            xy_dec, k_cache, v_cache = model.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache, attn_mask)
        logits = model.ar_predict_layer(xy_dec[:, -1])

        if idx == 0:
            attn_mask = F.pad(attn_mask[:, :, -1].unsqueeze(-2), (0, 1), value=False)
        else:
            attn_mask = F.pad(attn_mask, (0, 1), value=False)

        if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
            logits = logits[:, :-1]

        samples = _sample_rows(logits, y, [rows[g] for g in active_row_ids])

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
            active_row_ids = [active_row_ids[i] for i in reserved_idx_of_batch_for_y.tolist()]

        # 只保留batch中未生成完毕的序列
        if reserved_idx_of_batch_for_y is not None:
            # index = torch.LongTensor(batch_idx_map).to(y.device)
            y = torch.index_select(y, dim=0, index=reserved_idx_of_batch_for_y)
            attn_mask = torch.index_select(attn_mask, dim=0, index=reserved_idx_of_batch_for_y)
            if k_cache is not None:
                for i in range(len(k_cache)):
                    k_cache[i] = torch.index_select(k_cache[i], dim=0, index=reserved_idx_of_batch_for_y)
                    v_cache[i] = torch.index_select(v_cache[i], dim=0, index=reserved_idx_of_batch_for_y)

        if (early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num) or idx == 1499:
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
            break

        ####################### update next step ###################################
        y_emb = model.ar_audio_embedding(y[:, -1:])
        xy_pos = y_emb * model.ar_audio_position.x_scale + model.ar_audio_position.alpha * model.ar_audio_position.pe[
            :, y_len + idx
        ].to(dtype=y_emb.dtype, device=y_emb.device)

    if None in idx_list:
        for i in range(len(x)):
            if idx_list[i] is None:
                idx_list[i] = 1500 - 1  ###如果没有生成到EOS，就用最大长度代替

    return y_list, idx_list


def install_batched_rng_infer_panel(model, tts):
    """返回签名兼容 infer_panel 的 per-seq RNG 桥函数。

    model: Text2SemanticDecoder; tts: TTS_infer_pack.TTS 实例 (占位, 与
    cudagraph / static_kv 桥签名对齐)。
    """
    fallback = model.infer_panel  # install 时的干净基线 (或前序桥)

    def infer_panel_batch_rng(all_phoneme_ids, all_phoneme_lens, prompt,
                              all_bert_features, top_k=15, top_p=1.0,
                              temperature=1.0, early_stop_num=-1, max_len=None,
                              repetition_penalty=1.35, streaming_mode=False,
                              chunk_length=16, row_seeds=None, row_sampling=None,
                              **kw):
        # stock L611 用 kwargs.get("max_len", x_lens.max()) 取缺省 — 显式
        # 转发 max_len=None 会遮蔽缺省 (L620 None 比较炸), 缺省时不转发
        fw = dict(kw)
        if max_len is not None:
            fw["max_len"] = max_len

        def _fallback():
            return fallback(all_phoneme_ids, all_phoneme_lens, prompt,
                            all_bert_features, top_k=top_k, top_p=top_p,
                            temperature=temperature,
                            early_stop_num=early_stop_num,
                            repetition_penalty=repetition_penalty,
                            streaming_mode=streaming_mode,
                            chunk_length=chunk_length, **fw)

        if (streaming_mode or prompt is None or row_seeds is None):
            return _fallback()
        # TTS.py 非流式传参形态: list[[T_i]] (或 tensor[B,T] 兼容)
        if isinstance(all_phoneme_ids, (list, tuple)):
            ids_list = list(all_phoneme_ids)
        else:
            ids_list = [t for t in all_phoneme_ids]
        if isinstance(all_bert_features, (list, tuple)):
            bert_list = list(all_bert_features)
        else:
            bert_list = [t for t in all_bert_features]
        n = len(ids_list)
        if n != len(bert_list) or n != len(row_seeds) or n < 1:
            return _fallback()
        # 行 tensor 归一 [T] (流式调用点传 [1,T]; 本桥只走非流式, 防御性 squeeze)
        ids_list = [t.squeeze(0) if t.dim() == 2 and t.shape[0] == 1 else t for t in ids_list]
        with torch.no_grad():
            rows = _build_rows(n, row_seeds, row_sampling, prompt.device,
                               top_k, top_p, temperature, repetition_penalty)
            return _batch_rng_decode(model, ids_list, all_phoneme_lens, prompt,
                                     bert_list, rows, top_k, top_p,
                                     early_stop_num, temperature,
                                     repetition_penalty, max_len)

    return infer_panel_batch_rng
