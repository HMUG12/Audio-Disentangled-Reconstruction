"""GPT-SoVITS 跨请求合批 worker (批次 67, m10 N=2 第三子批)。

多请求同签名 (同参考音频 / 同 prompt / 同语速 / 同权重) 时, 把各请求
切分后的段合并进一个池子走 to_batch → t2s AR 批解码 (行种子隔离, 批次
65 内核) → 按请求重组行 → 逐请求 vits 解码, 摊薄 AR 解码的批边际成本。

音质纪律 (字节级位等, 每条都有推演依据):
- t2s 域: 行种子桥 (ADR_T2S_BATCH_RNG) 让每行独立 generator — 行输出
  只依赖 (本行输入, 本行种子), 与同批邻行无关 → 跨请求混批不改变任何
  行的语义 token 序列 (G2 行隔离, test_t2s_batchrng 已证);
- vits 域: 逐请求 cat 解码 — stock method-2 把一个请求的全部段 cat 成
  [1,1,SumT] 连续 decode, 其 randn_like 吃全局流且 dec 卷积/enc_p 注意
  力感受野跨边界; 跨请求 cat 会让 B 段噪声取自 "A 的 set_seed 流的延续"
  并让 A 的尾帧渗进 B 的头帧 → 必须按请求重组后各自 decode (与 stock
  单请求同形状同调用 → 位等自然成立);
- vits 全局流重播: 单请求时序 = set_seed(seed) → [t2s: 行 generator,
  不耗全局 torch RNG] → vits randn。合批中 A/B 请求先后 set_seed 会
  污染彼此的 vits 起点 → 每请求 vits decode 前重播 set_seed(actual_seed)
  (t2s 已全部完成, derive_row_seeds 只耗 Python random 与 torch 流无关)。

失败隔离三级:
1. 预处理抛异常 → 该请求报错, 其余请求不受影响;
2. 预处理返回空 → 该请求 16000 静音 (stock L1197-1199 语义);
3. 合成中途异常 → stock L1551-1562 语义 (traceback / del t2s+vits /
   init_*_weights(configs 路径) 重载 / re-raise) — 该组全部未决请求
   报错 (模型态已不可信, 无法给出 "其余请求不受影响" 的保证)。

vocoder (V3/V4) 请求不走合并路径 (其合成走 using_vocoder_synthesis_*,
与 v1/v2 位等面不同) → 组内逐请求 solo 调 tts.run() (全 stock 路径)。

third_party 零补丁: 本模块是 ADR 侧桥函数, 只调用 TTS 公开成员
(prompt_cache / to_batch / audio_postprocess / set_ref_audio /
_get_ref_spec / text_preprocessor / empty_cache / init_*_weights / run)
与模块级 set_seed — 不改 third_party 任何文件。

worker 模式: 守护线程阻塞 get → ~50ms 合批窗口凑至 ADR_T2S_COALESCE_BATCH →
持 engine._lock 整轮 (与 solo 路径互斥) → 按签名分组 → 逐组 _ensure_weights
+ 合成; 处理期间新到请求留下一轮 (不阻塞提交方)。
"""
from __future__ import annotations

import math
import os
import queue
import threading
import time
import traceback
from concurrent.futures import Future
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from adr.core import settings
from adr.core.audit import log_event

# 合批窗口 (秒): 首请求到达后再等这么久凑批 — 无窗口的 get_nowait 抽干常在
# 第二请求入队前跑空 (GIL 调度 / 提交错峰几 ms), 两请求各占一轮, 合批落空
_COALESCE_WINDOW_S = 0.05


# ---------------------------------------------------------------------------
# 惰性助手 (须在 _gsv_context 内导入 — sys.path 已注入 GSV 目录)
# ---------------------------------------------------------------------------

class _Helpers:
    """TTS 模块级符号缓存: set_seed / NO_PROMPT_ERROR / splits。

    惰性导入 — third_party 的 TTS 模块只有进入 GSV 上下文 (sys.path 注入)
    后才可导入; 首次合成时按需加载。
    """

    def __init__(self):
        self.set_seed = None
        self.no_prompt_error = None
        self.splits = None
        self._lock = threading.Lock()

    def get(self, tts):
        with self._lock:
            if self.set_seed is None:
                import importlib

                mod = importlib.import_module("TTS_infer_pack.TTS")
                seg = importlib.import_module(
                    "TTS_infer_pack.text_segmentation_method")
                self.set_seed = mod.set_seed
                self.no_prompt_error = mod.NO_PROMPT_ERROR
                self.splits = seg.splits
        return self


_HELPERS = _Helpers()


# ---------------------------------------------------------------------------
# 签名与采样参数
# ---------------------------------------------------------------------------

def _signature(inputs: dict) -> tuple:
    """请求签名: 决定能否同组合批。

    进入签名: 影响共享状态的项 — 参考音频 (prompt_cache 装载)、prompt
    (cache 共享)、语速 (vits method-2 vs 串行分叉)、权重 (模型实例)。
    不进签名且安全的原因:
    - top_k/top_p/temperature/repetition_penalty: 行种子桥逐行覆盖
      (row_sampling), 行间零串扰;
    - seed: 逐请求 set_seed + 行种子, 本就逐请求;
    - fragment_interval / text_split_method / batch_size: 逐请求消费
      (audio_postprocess 的 fi / 切分发生在预处理 / 合批恒整池单批
      batch_size=len(data_all), 请求侧上限不进合并路径)。
    """
    prompt_text = inputs.get("prompt_text") or ""
    if prompt_text in [None, ""]:
        norm_prompt = ""
    else:
        # 与 stock L1177-1180 同款规范化 — 规范化后相同才共享 prompt_cache。
        # splits 未加载时 (首窗口分桶发生在请求线程, 不在 GSV 上下文无法
        # 导入) 跳过句尾归一化 — 只会让签名更分散 (少合批, 保守), 不会
        # 错误合并; _round 内 splits 已就绪, 分组仍按规范化签名准确。
        norm_prompt = str(prompt_text).strip("\n")
        if (norm_prompt and _HELPERS.splits is not None
                and norm_prompt[-1] not in _HELPERS.splits):
            norm_prompt += "。" if (inputs.get("prompt_lang") or "zh") != "en" else "."
    return (
        inputs.get("ref_audio_path") or "",
        tuple(inputs.get("aux_ref_audio_paths") or []),
        norm_prompt,
        inputs.get("prompt_lang") or "",
        float(inputs.get("speed_factor", 1.0)),
        inputs.get("t2s_weights"),
        inputs.get("vits_weights"),
    )


def _sig_dbg(sig: Optional[tuple]) -> str:
    """签名调试串 (审计用): 同档案请求签名漂移时, 定位是哪个成分变了。"""
    if sig is None:
        return "none"
    ref, aux, prompt, lang, speed, t2s, vits = sig
    return "|".join([
        Path(ref).parent.name if ref else "-",
        f"aux{len(aux)}",
        f"p{len(prompt)}",
        lang or "-",
        f"x{speed:g}",
        Path(str(t2s)).name if t2s else "-",
        Path(str(vits)).name if vits else "-",
    ])


def _sampling_params(inputs: dict) -> Tuple[int, float, float, float]:
    """请求的 t2s 采样参数 (与 stock run() L1045-1059 默认一致)。"""
    return (
        inputs.get("top_k", 15),
        inputs.get("top_p", 1),
        inputs.get("temperature", 1),
        inputs.get("repetition_penalty", 1.35),
    )


class _Job:
    """一次合批提交。"""

    __slots__ = ("inputs", "future", "submitted")

    def __init__(self, inputs: dict):
        self.inputs = inputs
        self.future: Future = Future()
        self.submitted = time.perf_counter()


# ---------------------------------------------------------------------------
# Coalescer
# ---------------------------------------------------------------------------

class _Coalescer:
    """跨请求合批 worker: submit(inputs) → Future[(sr, np.ndarray int16)]。"""

    def __init__(self, engine):
        self._engine = engine
        self._queue: "queue.Queue[_Job]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._spawn_lock = threading.Lock()

    # -- 提交侧 (引擎请求线程调用, 不持 engine._lock) ------------------

    def submit(self, inputs: dict) -> Future:
        job = _Job(inputs)
        self._queue.put(job)
        self._ensure_thread()
        return job.future

    def _ensure_thread(self):
        with self._spawn_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._loop, name="adr-gsv-coalesce", daemon=True)
                self._thread.start()

    # -- worker 侧 ----------------------------------------------------

    def _loop(self):
        while True:
            job = self._queue.get()  # 阻塞等第一个
            try:
                cap = max(1, settings.t2s_coalesce_batch())
                # 小合批窗口: 首个到达后再等 ~50ms 抽干队列 (不设条数上限 —
                # 池深受调度并发约束, 天然有限); 队列已有后续时 get 立即返回,
                # 窗口零浪费 — 单请求仅多付 50ms 延迟。
                jobs = [job]
                deadline = time.perf_counter() + _COALESCE_WINDOW_S
                while True:
                    remain = deadline - time.perf_counter()
                    if remain <= 0:
                        break
                    try:
                        jobs.append(self._queue.get(timeout=remain))
                    except queue.Empty:
                        break
                # 按签名分桶: 同签名才进同一轮。异签名既无法共享 t2s 批解码,
                # 又引入组间权重热换 — 混合档案下原"一轮多组"每轮吃 2-3 次
                # 热换、每组只剩 1-2 条可合; 分桶后热换收敛为每桶一次,
                # 桶内独占整批 cap (实测 12 请求 3 档案: 80s → 3 轮 ~36s)。
                buckets: Dict[Optional[tuple], List[_Job]] = {}
                for j in jobs:
                    try:
                        sig = _signature(j.inputs)
                    except BaseException:
                        sig = None  # 签名失败单独成桶, 由 _round 报错
                    buckets.setdefault(sig, []).append(j)
                log_event("coalesce_window", jobs=len(jobs), buckets=len(buckets),
                          sigs=[f"{_sig_dbg(k)}x{len(v)}"
                                for k, v in buckets.items()])
                for bucket in buckets.values():
                    for i in range(0, len(bucket), cap):
                        # 同桶超 cap 连续拆片 — 不重开窗口, 无凑批等待
                        chunk = bucket[i:i + cap]
                        try:
                            self._round(chunk)
                        except BaseException:
                            # 兜底: 轮级意外不得吞掉未决 future (请求线程
                            # .result() 会挂死), 逐个回填异常后继续
                            traceback.print_exc()
                            for j in chunk:
                                if not j.future.done():
                                    j.future.set_exception(
                                        RuntimeError("coalesce round failed"))
            except BaseException:  # 兜底: worker 永不因单轮异常死亡
                traceback.print_exc()

    def _round(self, jobs: List[_Job]):
        """一轮: 持引擎锁整轮 → 按签名分组 → 逐组热换权重 + 合成。

        轮级兜底异常: 该轮全部未决请求报错 (分组/收尾阶段的意外)。
        组级异常: 该组未决请求报错 + stock 模型重置, 其余组继续。
        """
        engine = self._engine
        t0 = time.perf_counter()
        groups: Dict[tuple, List[_Job]] = {}
        with engine._lock:
            try:
                engine._lazy_init()
            except BaseException as e:
                # 懒加载失败 (冷启动 OOM 等): 本轮全部未决请求必须拿到
                # 异常 — 若任其逃逸到 _loop 兜底, 请求线程 .result() 挂死
                traceback.print_exc()
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(e)
                log_event("coalesce", requests=len(jobs), ok=0, groups=0,
                          err=type(e).__name__, sig=["init_failed"])
                return
            tts = engine._tts
            helpers = _HELPERS.get(tts)
            with engine._gsv_context():
                # 分组 (签名失败的请求单独报错, 不拖累他人)
                for job in jobs:
                    try:
                        sig = _signature(job.inputs)
                    except BaseException as e:
                        job.future.set_exception(e)
                        continue
                    groups.setdefault(sig, []).append(job)

                try:
                    for gjobs in groups.values():
                        if all(j.future.done() for j in gjobs):
                            continue
                        try:
                            engine._ensure_weights(
                                gjobs[0].inputs.get("vits_weights"),
                                gjobs[0].inputs.get("t2s_weights"))
                            self._synthesize_group(tts, helpers, gjobs)
                        except BaseException as e:
                            traceback.print_exc()
                            log_event("coalesce_group_error",
                                      err=type(e).__name__,
                                      jobs=len(gjobs))
                            self._reset_models(tts)
                            for job in gjobs:
                                if not job.future.done():
                                    job.future.set_exception(e)
                finally:
                    # stock run() finally 语义: empty_cache (实例已打节流补丁)
                    try:
                        tts.empty_cache()
                    except BaseException:
                        pass

        ok_jobs = sum(1 for j in jobs
                      if j.future.done() and j.future.exception() is None)
        log_event("coalesce", requests=len(jobs), ok=ok_jobs,
                  groups=len(groups), wait_ms=int(
                      (t0 - min(j.submitted for j in jobs)) * 1000),
                  sig=[_sig_dbg(k) for k in groups])

    def _reset_models(self, tts):
        """stock run() L1555-1561 语义: 模型态不可信时 del + 按 configs 路径
        重载 (configs.t2s_weights_path/vits_weights_path 是当前生效权重)。
        任一步失败不掩盖原异常 (调用方 set_exception 的是合成异常)。"""
        try:
            del tts.t2s_model
        except BaseException:
            pass
        try:
            del tts.vits_model
        except BaseException:
            pass
        tts.t2s_model = None
        tts.vits_model = None
        try:
            tts.init_t2s_weights(tts.configs.t2s_weights_path)
        except BaseException:
            traceback.print_exc()
        try:
            tts.init_vits_weights(tts.configs.vits_weights_path)
        except BaseException:
            traceback.print_exc()

    # -- 合成一组 ------------------------------------------------------

    def _synthesize_group(self, tts, helpers, gjobs: List[_Job]):
        """同签名一组请求 → 合并池 → t2s 混批 → 按请求 vits → 逐请求回填。

        结构逐段复刻 stock run() (L1136-1549), 差异仅:
        - 池子来自多请求 (行种子派生 + row_sampling 逐行);
        - vits 从 "按批 cat" 改 "按请求重组后 cat" (音质位等, 见模块头);
        - 每请求 vits decode 前重播 set_seed (全局流隔离)。
        回退 (整组逐请求 tts.run, 全 stock 路径):
        - vocoder (V3/V4): 合成走 using_vocoder_synthesis_*, 位等面不同;
        - no_prompt (ref-free): to_batch(prompt_data=None) → 桥收
          prompt=None → 行种子桥 fallback (stock 全局 RNG, 行采样流跨
          请求交叠) → G3-cross 位等不成立 — 音质优先, 不合并。
        """
        base = gjobs[0].inputs
        prompt_text = base.get("prompt_text")
        no_prompt_text = prompt_text in [None, ""]
        if tts.configs.use_vocoder or no_prompt_text:
            self._group_solo(tts, gjobs)
            return

        import numpy as np
        import torch

        top_k, top_p, temperature, repetition_penalty = _sampling_params(base)
        speed_factor = float(base.get("speed_factor", 1.0))
        batch_threshold = base.get("batch_threshold", 0.75)
        batch_rng_on = os.environ.get("ADR_T2S_BATCH_RNG") == "1"

        # -- infer_panel 重装 (与 stock L1068-1110 同序: 先重置再按 env 挂桥;
        #    权重热换会重建 t2s_model.model, 每组必须重放) --
        model = tts.t2s_model.model
        model.infer_panel = model.infer_panel_batch_infer
        if os.environ.get("ADR_T2S_CUDAGRAPH") == "1":
            from adr.models.adr_t2s_bridge import install_cudagraph_infer_panel
            model.infer_panel = install_cudagraph_infer_panel(model, tts)
        if os.environ.get("ADR_T2S_STATIC_KV") == "1":
            from adr.models.adr_t2s_static_kv import install_static_kv_infer_panel
            model.infer_panel = install_static_kv_infer_panel(model, tts)
        if batch_rng_on:
            from adr.models.adr_t2s_batchrng import install_batched_rng_infer_panel
            model.infer_panel = install_batched_rng_infer_panel(model, tts)

        # -- split_bucket: 合批路径恒 False (整池单批, 见下方 to_batch 处;
        #    stock 的 speed/vocoder 自动关闭分支被单批策略覆盖) --
        split_bucket = False

        # -- 逐请求 set_seed + 预处理 + 行种子 → 合并池 --
        data_all: List[dict] = []
        fi_of: List[float] = []           # 请求 → fragment_interval
        seed_of: List[int] = []           # 请求 → actual_seed
        rows_of: List[List[int]] = []     # 请求 → 全局行号列表
        for gi, job in enumerate(gjobs):
            inp = job.inputs
            seed = inp.get("seed", -1)
            seed = -1 if seed in ["", None] else seed
            actual_seed = helpers.set_seed(seed)
            fi_of.append(float(inp.get("fragment_interval", 0.3)))
            seed_of.append(actual_seed)
            try:
                data = self._preprocess(tts, helpers, inp)
            except BaseException as e:
                # 失败隔离第 1 级: 预处理异常 = 该请求报错, 其余请求继续
                job.future.set_exception(e)
                rows_of.append([])
                continue
            if len(data) == 0:
                # stock L1197-1199: 空切分 → 16000 静音 (该请求隔离)
                job.future.set_result(
                    (16000, np.zeros(int(16000), dtype=np.int16)))
                rows_of.append([])
                continue
            if batch_rng_on:
                from adr.models.adr_t2s_batchrng import derive_row_seeds
                for it, s in zip(data, derive_row_seeds(len(data), actual_seed)):
                    it["seed"] = s
            first = len(data_all)
            data_all.extend(data)
            rows_of.append(list(range(first, len(data_all))))

        if not data_all:
            return  # 全部空切分, 静音已回填

        # -- row_sampling 逐行覆盖 (采样参数组内不全同才有) --
        row_sampling_map: Optional[List[Optional[dict]]] = None
        base_params = _sampling_params(base)
        for gi, job in enumerate(gjobs):
            p = _sampling_params(job.inputs)
            if p != base_params:
                if row_sampling_map is None:
                    row_sampling_map = [None] * len(data_all)
                ov = {"top_k": p[0], "top_p": p[1],
                      "temperature": p[2], "repetition_penalty": p[3]}
                for r in rows_of[gi]:
                    row_sampling_map[r] = ov

        # -- 合并池 to_batch: batch_size=整池行数 + split_bucket=False →
        #    顺序分块恰好一批装下全部段 (合批目的所在)。不可沿用请求
        #    batch_size (拆成单行批, 摊薄批边际成本落空), 也不可开
        #    split_bucket (按长度聚类 + score 分桶会把悬殊请求拆散 — 如
        #    451 vs 272 token 比率 0.60 < threshold 0.75 必分两批) --
        data, batch_index_list = tts.to_batch(
            data_all,
            prompt_data=tts.prompt_cache,
            batch_size=len(data_all),
            threshold=batch_threshold,
            split_bucket=split_bucket,
            device=tts.configs.device,
            precision=tts.precision,
        )

        # -- 批循环: t2s 混批解码, 输出按 batch_index_list 还原到全局行槽 --
        rows_semantic = [None] * len(data_all)
        rows_idx = [None] * len(data_all)
        rows_phones = [None] * len(data_all)
        refer_audio_spec = None
        sv_emb = None
        for item, idxs in zip(data, batch_index_list):
            batch_phones = item["phones"]
            if refer_audio_spec is None:
                # 组内签名相同 → refer_spec 组级只装一次 (stock 每批重装,
                # spec.to/compute_embedding3 数值确定 → 位等不受影响)
                refer_audio_spec = []
                sv_emb = [] if tts.is_v2pro else None
                for spec, audio_tensor in tts.prompt_cache["refer_spec"]:
                    spec = spec.to(dtype=tts.precision, device=tts.configs.device)
                    refer_audio_spec.append(spec)
                    if tts.is_v2pro:
                        sv_emb.append(tts.sv_model.compute_embedding3(audio_tensor))
            prompt = tts.prompt_cache["prompt_semantic"].expand(
                len(item["all_phones"]), -1).to(tts.configs.device)
            if row_sampling_map is not None:
                item["row_sampling"] = [row_sampling_map[r] for r in idxs]
            pred_semantic_list, idx_list = model.infer_panel(
                item["all_phones"],
                item["all_phones_len"],
                prompt,
                item["all_bert_features"],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=tts.configs.hz * tts.configs.max_sec,
                max_len=item["max_len"],
                repetition_penalty=repetition_penalty,
                row_seeds=item.get("row_seeds"),
                row_sampling=item.get("row_sampling"),
            )
            for j, r in enumerate(idxs):
                rows_semantic[r] = pred_semantic_list[j]
                rows_idx[r] = idx_list[j]
                rows_phones[r] = batch_phones[j]

        # -- 按请求重组 → vits 解码 → audio_postprocess → 回填 --
        output_sr = tts.configs.sampling_rate
        use_method2 = speed_factor == 1.0
        for gi, job in enumerate(gjobs):
            if job.future.done():
                continue  # 空切分已回填静音
            row_ids = rows_of[gi]
            sem = [rows_semantic[r] for r in row_ids]
            idxs_r = [rows_idx[r] for r in row_ids]
            phs = [rows_phones[r] for r in row_ids]
            # vits randn_like 吃全局 torch 流: 重播本请求种子 + 逐步重放
            # stock t2s 采样对全局流的消耗, 使起点与单请求路径 (set_seed →
            # 逐步采样 → vits) 一致。stock 每步在全局流抽 exponential_
            # (utils.py 无 generator): 步 0-10 形 [1,V-1], 之后 [1,V], 每行
            # K+1 次 (K = 该行 EOS 步号, 含 EOS 步本身); 混批解码中本桥行
            # generator 不耗全局流、桥的抛弃式抽取形状随批行数变化, 均无法
            # 对齐 → 以此处重播为准 (整池单批下逐行重放 = stock batch_size=1
            # 的全局流消耗总量, 与行间顺序无关)
            helpers.set_seed(seed_of[gi])
            _V = model.ar_predict_layer.out_features
            _dt = model.ar_predict_layer.weight.dtype
            for _r in row_ids:
                for _s in range(int(rows_idx[_r]) + 1):
                    torch.empty(1, _V - 1 if _s < 11 else _V,
                                device=tts.configs.device,
                                dtype=_dt).exponential_(1)
            if use_method2:
                batch_audio_fragment = self._vits_method2(
                    tts, sem, idxs_r, phs, refer_audio_spec, sv_emb,
                    speed_factor)
            else:
                batch_audio_fragment = self._vits_serial(
                    tts, sem, idxs_r, phs, refer_audio_spec, sv_emb,
                    speed_factor)
            sr, audio = tts.audio_postprocess(
                [batch_audio_fragment], output_sr, None, speed_factor,
                False, fi_of[gi], False)
            job.future.set_result((sr, audio))

    def _group_solo(self, tts, gjobs: List[_Job]):
        """vocoder (V3/V4) 组回退: 逐请求 tts.run (全 stock 路径 — set_seed/
        桥安装/vocoder 合成/异常重置皆由 run 内部完成)。"""
        for job in gjobs:
            if job.future.done():
                continue
            try:
                sr, audio = None, None
                for sr, audio in tts.run(job.inputs):
                    pass  # 非流式只 yield 一次完整音频
                job.future.set_result((sr, audio))
            except BaseException as e:
                # run 内部已做模型重置 + re-raise (stock L1551-1562)
                job.future.set_exception(e)

    # -- vits 两个分叉 (逐段复刻 stock L1337-1371, 行集来自单请求) ------

    def _vits_method2(self, tts, sem, idxs_r, phs, refer_audio_spec, sv_emb,
                      speed_factor):
        """speed==1.0: 本请求全部段 cat 成一条 decode 再切段 (stock method-2
        同形状同调用 — 位等的关键)。"""
        import torch

        pred_semantic_list = [item[-idx:] for item, idx in zip(sem, idxs_r)]
        upsample_rate = math.prod(tts.vits_model.upsample_rates)
        audio_frag_idx = [
            pred_semantic_list[i].shape[0] * 2 * upsample_rate
            for i in range(0, len(pred_semantic_list))
        ]
        audio_frag_end_idx = [sum(audio_frag_idx[: i + 1])
                              for i in range(0, len(audio_frag_idx))]
        all_pred_semantic = torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(
            0).to(tts.configs.device)
        _batch_phones = torch.cat(phs).unsqueeze(0).to(tts.configs.device)
        _batch_audio_fragment = tts.vits_model.decode(
            all_pred_semantic, _batch_phones, refer_audio_spec,
            speed=speed_factor, sv_emb=sv_emb
        ).detach()[0, 0, :]
        audio_frag_end_idx.insert(0, 0)
        return [
            _batch_audio_fragment[audio_frag_end_idx[i - 1]: audio_frag_end_idx[i]]
            for i in range(1, len(audio_frag_end_idx))
        ]

    def _vits_serial(self, tts, sem, idxs_r, phs, refer_audio_spec, sv_emb,
                     speed_factor):
        """变速: 逐行 decode (stock L1362-1371)。"""
        import torch

        batch_audio_fragment = []
        for i, idx in enumerate(idxs_r):
            phones = phs[i].unsqueeze(0).to(tts.configs.device)
            _pred_semantic = sem[i][-idx:].unsqueeze(0).unsqueeze(0)
            audio_fragment = tts.vits_model.decode(
                _pred_semantic, phones, refer_audio_spec,
                speed=speed_factor, sv_emb=sv_emb
            ).detach()[0, 0, :]
            batch_audio_fragment.append(audio_fragment)
        return batch_audio_fragment

    # -- 预处理 (逐段复刻 stock run() L1136-1196) ----------------------

    def _preprocess(self, tts, helpers, inp: dict) -> List[dict]:
        """单请求前端: ref 装载 + prompt 规范化 + 切分/BERT 提取。

        返回段列表 (可能为空 = 该请求静音)。异常向上抛 = 该请求隔离报错
        (组循环的 try 捕获)。组内同签名 → prompt_cache/ref 装载只发生一次,
        后续请求命中缓存 (stock 同款判断)。"""
        text = inp.get("text", "")
        text_lang = inp.get("text_lang", "")
        ref_audio_path = inp.get("ref_audio_path", "")
        aux_ref_audio_paths = inp.get("aux_ref_audio_paths", [])
        prompt_text = inp.get("prompt_text", "")
        prompt_lang = inp.get("prompt_lang", "")

        no_prompt_text = prompt_text in [None, ""]

        assert text_lang in tts.configs.languages
        if not no_prompt_text:
            assert prompt_lang in tts.configs.languages

        if no_prompt_text and tts.configs.use_vocoder:
            raise helpers.no_prompt_error(
                "prompt_text cannot be empty when using SoVITS_V3")

        if ref_audio_path in [None, ""] and (
            (tts.prompt_cache["prompt_semantic"] is None)
            or (tts.prompt_cache["refer_spec"] in [None, []])
        ):
            raise ValueError(
                "ref_audio_path cannot be empty, when the reference audio "
                "is not set using set_ref_audio()")

        if (ref_audio_path is not None) and (
            ref_audio_path != tts.prompt_cache["ref_audio_path"]
            or (tts.is_v2pro and tts.prompt_cache["refer_spec"][0][1] is None)
        ):
            if not os.path.exists(ref_audio_path):
                raise ValueError(f"{ref_audio_path} not exists")
            tts.set_ref_audio(ref_audio_path)

        aux_ref_audio_paths = aux_ref_audio_paths if aux_ref_audio_paths is not None else []
        paths = set(aux_ref_audio_paths) & set(tts.prompt_cache["aux_ref_audio_paths"])
        if not (len(list(paths)) == len(aux_ref_audio_paths)
                == len(tts.prompt_cache["aux_ref_audio_paths"])):
            tts.prompt_cache["aux_ref_audio_paths"] = aux_ref_audio_paths
            tts.prompt_cache["refer_spec"] = [tts.prompt_cache["refer_spec"][0]]
            for path in aux_ref_audio_paths:
                if path in [None, ""]:
                    continue
                if not os.path.exists(path):
                    print("音频文件不存在，跳过：", path)
                    continue
                tts.prompt_cache["refer_spec"].append(tts._get_ref_spec(path))

        if not no_prompt_text:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in helpers.splits:
                prompt_text += "。" if prompt_lang != "en" else "."
            if tts.prompt_cache["prompt_text"] != prompt_text:
                phones, bert_features, norm_text = \
                    tts.text_preprocessor.segment_and_extract_feature_for_text(
                        prompt_text, prompt_lang, tts.configs.version)
                tts.prompt_cache["prompt_text"] = prompt_text
                tts.prompt_cache["prompt_lang"] = prompt_lang
                tts.prompt_cache["phones"] = phones
                tts.prompt_cache["bert_features"] = bert_features
                tts.prompt_cache["norm_text"] = norm_text

        return tts.text_preprocessor.preprocess(
            text, text_lang, inp.get("text_split_method", "cut1"),
            tts.configs.version)
