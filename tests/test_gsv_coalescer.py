"""批次 67 (m10 N=2 第三子批): 跨请求合批 (gsv_coalescer) 等价性与隔离测试。

真代码面: 真 TTS 类的 to_batch / audio_postprocess / empty_cache (真分桶 /
行种子聚合 / 归一 / 拼接路径) + 真 set_seed + 真 Text2SemanticDecoder
(tiny) + ADR_T2S_BATCH_RNG 行种子桥; 假的只是权重容器与前端 (FakeVits /
FakePreprocessor / ref 装载 stub — 全部显式 generator 派生, 不吃全局
torch 流)。

RNG 纪律 (字节等价断言的根基):
- 全局 torch 流只被两处消费: 真 set_seed (逐请求) 与 FakeVits.decode 的
  torch.rand (无 generator) — "vits decode 前重播 set_seed" 是否正确,
  直接体现为 solo 轮 vs 混批轮的逐请求音频字节是否一致;
- t2s 行采样走行 generator (批次65桥), 与全局流零交集;
- 假前端 (phones/bert/spec/prompt_semantic) 全部显式 generator / 独立
  Random 派生, 同文本跨轮比特相同。

用例:
A. G3-cross 跨请求位等: 混批轮 (A 3段 + B 2段) 每请求音频 == 该请求
   solo 轮; solo A 的 max_len=19 vs 混批 22 (max_len 左移位位等 = 设计
   验证点); 混批轮 vits 恰好 2 次 decode 且 T=各请求 SumT (非 SumT_all
   — vits 渗漏两层破坏的按请求 cat 落地); 混批真发生 (一次 infer_panel
   5 行, 两请求段同批);
B. G1 确定性: 同混批两跑逐请求字节一致;
C. 签名分组: 权重不同 → 两组 _ensure_weights 逐组; 签名失败单请求
   报错不拖累他人;
D. row_sampling 逐行覆盖: 参数异于组基线的请求行带覆盖 dict;
E. 隔离: 空切分 → 16000 静音; 预处理异常单请求报错其余照常; 合成异常
   → 组内全部报错 + init_*_weights 重载 (stock L1551-1562 语义);
F. 回退: no_prompt / vocoder 组整组 solo (tts.run, 不碰 infer_panel);
G. settings: coalesce 默认开; batch 默认 4 / 脏值回落 / 不封顶;
H. worker: submit → future 回填 (多请求分轮不卡死)。
"""
import contextlib
import math
import random
import sys
import threading
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
GSV_ROOT = REPO / "third_party" / "gpt_sovits"          # tools / BigVGAN 层
GSV_DIR = GSV_ROOT / "GPT_SoVITS"                       # TTS_infer_pack / AR 层
# 与 gsv_engine._gsv_context 同款三层注入 (缺中间层 TTS.py 的 tools 导入即炸)
for _p in (str(REPO), str(GSV_ROOT), str(GSV_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adr.models.gsv_coalescer as gc  # noqa: E402
import adr.models.adr_t2s_batchrng as abr  # noqa: E402
from adr.core import settings  # noqa: E402
from AR.models.t2s_model import Text2SemanticDecoder  # noqa: E402

CONFIG = {"model": {
    "hidden_dim": 32, "embedding_dim": 32, "head": 2, "n_layer": 2,
    "vocab_size": 25, "phoneme_vocab_size": 32, "dropout": 0.1, "EOS": 24,
}}
HZ, MAX_SEC = 25, 2        # early_stop_num = 50 (避开 1500 步上限)
SR = 32000
PROMPT_TEXT = "s6"         # 规范化补 "。" 后仍解析为 6 phones


@pytest.fixture(autouse=True)
def _quiet_env(monkeypatch):
    """合批轮运行环境: 行种子桥开 / 桥链关闭 / 审计静默。"""
    monkeypatch.setenv("ADR_T2S_BATCH_RNG", "1")
    monkeypatch.setenv("ADR_T2S_CUDAGRAPH", "0")
    monkeypatch.setenv("ADR_T2S_STATIC_KV", "0")
    monkeypatch.setenv("ADR_SAFETY_AUDIT", "0")
    monkeypatch.delenv("ADR_T2S_COALESCE", raising=False)
    monkeypatch.delenv("ADR_T2S_COALESCE_BATCH", raising=False)


# ---------------------------------------------------------------------------
# 真类惰性导入 + 假前端
# ---------------------------------------------------------------------------

_TTS_CLS = None


def _real_tts_cls():
    """真 TTS 类 (惰性 — 复用其 to_batch / audio_postprocess / empty_cache
    真实现; 合批测试必须踩真分桶/归一/拼接, 不重写)。

    首次导入须 chdir 到 GSV 根: sv.py 顶层 `sys.path.append(cwd +
    "/GPT_SoVITS/eres2net")` 按 cwd 拼路径 (生产 _gsv_context 先 chdir 再
    导入故不炸); 其注入为绝对路径, 导入完成后恢复 cwd 不受影响。"""
    global _TTS_CLS
    if _TTS_CLS is None:
        import os

        old = os.getcwd()
        os.chdir(GSV_ROOT)
        try:
            from TTS_infer_pack.TTS import TTS
        finally:
            os.chdir(old)
        _TTS_CLS = TTS
    return _TTS_CLS


def _seg_seed(seg: str) -> int:
    return zlib.crc32(seg.encode("utf-8"))


def _seg_phones(seg: str, n: int) -> list:
    rng = random.Random(_seg_seed(seg))    # 独立 Random — 不动全局 random
    return [rng.randrange(32) for _ in range(n)]


def _seg_bert(seg: str, n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(_seg_seed(seg))  # 显式 gen — 不动全局流
    return torch.randn(1024, n, generator=g)


class FakePreprocessor:
    """'|' 分段; 段 "s13" → 13 phones; norm_text = "x"*13 (len 进分桶)。
    显式 generator / 独立 Random — 与真 preprocess 不吃全局 torch 流一致。"""

    def __init__(self, fail_texts=()):
        self.fail_texts = set(fail_texts)

    def _extract(self, text: str):
        text = text.rstrip("。.")
        n = int(text[1:]) if (text[:1] == "s" and text[1:].isdigit()) else len(text)
        if n <= 0:
            return None
        return _seg_phones(text, n), _seg_bert(text, n), "x" * n

    def segment_and_extract_feature_for_text(self, text, lang, version):
        r = self._extract(text)
        if r is None:
            return [], torch.zeros(1024, 0), ""
        return r

    def preprocess(self, text, text_lang, split_method, version):
        if text in self.fail_texts:
            raise ValueError(f"boom-preprocess: {text}")
        out = []
        for seg in (text.split("|") if text else []):
            r = self._extract(seg)
            if r is None:
                continue
            phones, bert, norm = r
            out.append({"phones": phones, "bert_features": bert,
                        "norm_text": norm})
        return out


class FakeVits:
    """decode: 输出 [1,1,T*2*upsample]; torch.rand 无 generator 吃全局流 —
    set_seed 重播正确性 → solo vs 混批字节等价可断言; 记录每次调用。"""

    def __init__(self, upsample_rates=(4, 4)):
        self.upsample_rates = list(upsample_rates)
        self.calls = []

    def decode(self, pred_semantic, phones, refer_audio_spec,
               speed=1.0, sv_emb=None):
        t = pred_semantic.shape[-1]
        n = t * 2 * math.prod(self.upsample_rates)
        self.calls.append({"T": t, "n": n, "phones": int(phones.shape[-1])})
        z = torch.rand(1, 1, n)     # 全局流消费点 (位等断言的关键)
        return z + pred_semantic.to(torch.float32).mean()


def _make_fake_tts(use_vocoder=False):
    cls = _real_tts_cls()

    class _FakeTTS(cls):
        """真 TTS 骨架 + 假模型/前端; 权重装载与 ref 装载 stub 化。"""

        def __init__(self):
            self.configs = SimpleNamespace(
                device=torch.device("cpu"), is_half=False, version="v2",
                sampling_rate=SR, hz=HZ, max_sec=MAX_SEC,
                languages=["auto", "zh", "en"], use_vocoder=use_vocoder,
                t2s_weights_path="fake_t2s.ckpt",
                vits_weights_path="fake_vits.pth")
            self.precision = torch.float32
            self.is_v2pro = False
            self.sv_model = None
            self.text_preprocessor = FakePreprocessor()
            self.prompt_cache = {
                "ref_audio_path": None, "prompt_semantic": None,
                "refer_spec": [], "prompt_text": None, "prompt_lang": None,
                "phones": None, "bert_features": None, "norm_text": None,
                "aux_ref_audio_paths": []}
            self.t2s_inits = 0
            self.vits_inits = 0
            self.run_calls = []
            self.init_t2s_weights("fake_t2s.ckpt")
            self.init_vits_weights("fake_vits.pth")

        def init_t2s_weights(self, weights_path):
            self.t2s_inits += 1
            torch.manual_seed(42)
            self.t2s_model = SimpleNamespace(
                model=Text2SemanticDecoder(CONFIG).eval())
            self.configs.t2s_weights_path = weights_path

        def init_vits_weights(self, weights_path):
            self.vits_inits += 1
            self.vits_model = FakeVits()
            self.configs.vits_weights_path = weights_path

        def set_ref_audio(self, ref_audio_path):
            g = torch.Generator().manual_seed(_seg_seed("ref:" + ref_audio_path))
            # 0..22 < EOS(24): 真实语义 token 不落 EOS 槽位 — stock vocab
            # 1025 vs EOS 1024 同款形状 (EOS == vocab-1, 前 10 步 [:, :-1] 禁停)
            self.prompt_cache["prompt_semantic"] = torch.randint(
                0, 23, (6,), generator=g)
            spec = torch.randn(1, 100, 20, generator=g)
            self.prompt_cache["refer_spec"] = [(spec, None)]
            self.prompt_cache["ref_audio_path"] = ref_audio_path

        def run(self, inputs):
            # solo 回退路径 stub: 记录调用 (真 run 的 set_seed/桥安装/异常
            # 重置逻辑是 stock 原样, 零补丁, 不在本测试覆盖面)
            self.run_calls.append(dict(inputs))
            yield SR, np.zeros(16, dtype=np.int16)

    return _FakeTTS()


class FakeEngine:
    def __init__(self, tts):
        self._tts = tts
        self._lock = threading.RLock()
        self.ensured = []

    def _lazy_init(self):
        pass

    @contextlib.contextmanager
    def _gsv_context(self):
        yield

    def _ensure_weights(self, vits_weights, t2s_weights):
        self.ensured.append((vits_weights, t2s_weights))


def _make_ref(tmp_path) -> str:
    """真实存在的空 ref 文件 (预处理有 os.path.exists 检查)。"""
    p = tmp_path / "ref_a.wav"
    p.write_bytes(b"")
    return str(p)


def _inputs(text, seed=11, *, prompt_text=PROMPT_TEXT, speed=1.0,
            t2s_w="t2sA", vits_w="vitsA", ref="ref_a.wav",
            batch_size=8, **kw):
    d = {
        "text": text, "text_lang": "zh", "ref_audio_path": ref,
        "prompt_text": prompt_text, "prompt_lang": "zh",
        "top_k": 15, "top_p": 1.0, "temperature": 1.0,
        "repetition_penalty": 1.35, "speed_factor": speed,
        "seed": seed, "text_split_method": "cut1", "batch_size": batch_size,
        "fragment_interval": 0.3,
        "t2s_weights": t2s_w, "vits_weights": vits_w,
    }
    d.update(kw)
    return d


def _spy_infer_panel(tts, monkeypatch):
    """记录 infer_panel 调用 (kw 含 max_len / row_seeds / row_sampling),
    透传真桥。

    挂点在批次65桥的 install 出口: row_seeds 齐备时桥自走解码循环, 不经
    fallback (infer_panel_batch_infer), 包后者在合批成功路径上永不触发;
    包装 install 返回的 panel 则合批路径必经。monkeypatch 负责还原模块
    属性 (跨测试不泄漏)。"""
    real_install = abr.install_batched_rng_infer_panel
    recs = []

    def spy_install(*a, **kw):
        panel = real_install(*a, **kw)

        def wrapped(*args, **kwd):
            recs.append({"n_rows": len(args[0]), "kw": dict(kwd)})
            return panel(*args, **kwd)

        return wrapped

    monkeypatch.setattr(abr, "install_batched_rng_infer_panel", spy_install)
    return recs


def _run_round(tts, inputs_list):
    """同线程直跑一轮 (绕过 worker 线程): 返回 (engine, jobs), 断言从
    job.future 取结果/异常。"""
    engine = FakeEngine(tts)
    jobs = [gc._Job(i) for i in inputs_list]
    gc._Coalescer(engine)._round(jobs)
    return engine, jobs


# ---------------------------------------------------------------------------
# A. G3-cross 跨请求位等
# ---------------------------------------------------------------------------

def test_a_g3_cross_mixed_vs_solo_byte_equivalence(tmp_path, monkeypatch):
    """混批轮每请求音频 == 该请求 solo 轮 (字节级); max_len 移位位等;
    vits 按请求恰好一次 decode (T=SumT_req); 混批真发生。"""
    ref = _make_ref(tmp_path)
    # A: 3 段 (phones 13/9/11); B: 2 段 (14/16) — 行长跨请求交错
    in_a = _inputs("s13|s9|s11", seed=11, ref=ref)
    in_b = _inputs("s14|s16", seed=22, ref=ref)

    # solo 轮 (单请求 round = solo 参照: set_seed 重播在单请求下是 no-op)
    tts_solo = _make_fake_tts()
    spy = _spy_infer_panel(tts_solo, monkeypatch)
    _, jobs_a = _run_round(tts_solo, [in_a])
    sr_a_solo, audio_a_solo = jobs_a[0].future.result(timeout=30)
    assert sr_a_solo == SR and audio_a_solo.dtype == np.int16
    assert spy[0]["n_rows"] == 3 and spy[0]["kw"]["max_len"] == 19  # 6+13
    dec_solo_a = list(tts_solo.vits_model.calls)          # 1 次, T=SumT_A

    engine_solo, jobs_b = _run_round(tts_solo, [in_b])
    sr_b_solo, audio_b_solo = jobs_b[0].future.result(timeout=30)
    assert spy[1]["n_rows"] == 2 and spy[1]["kw"]["max_len"] == 22  # 6+16
    dec_solo_b = list(tts_solo.vits_model.calls[len(dec_solo_a):])

    # 混批轮: A + B 同签名同轮 → 段池合并 → t2s 混批 → 按请求 vits
    tts_mix = _make_fake_tts()
    spy_m = _spy_infer_panel(tts_mix, monkeypatch)
    engine_mix, jobs = _run_round(tts_mix, [in_a, in_b])
    sr_a_mix, audio_a_mix = jobs[0].future.result(timeout=30)
    sr_b_mix, audio_b_mix = jobs[1].future.result(timeout=30)

    # 字节级位等 (G3-cross 核心: 同请求 solo vs 混批)
    assert sr_a_mix == sr_a_solo
    assert np.array_equal(audio_a_mix, audio_a_solo)
    assert np.array_equal(audio_b_mix, audio_b_solo)

    # 混批真发生: 一次 infer_panel 收 5 行 (两请求段同批), max_len=22
    assert len(spy_m) == 1 and spy_m[0]["n_rows"] == 5
    assert spy_m[0]["kw"]["max_len"] == 22
    assert len(spy_m[0]["kw"]["row_seeds"]) == 5

    # vits 按请求重组: 恰好 2 次 decode, 每请求一次, T=该请求 SumT
    # (非 SumT_all 单次 cat — 渗漏两层破坏的设计验证点)
    dec_mix = tts_mix.vits_model.calls
    assert len(dec_mix) == 2
    assert sorted(c["n"] for c in dec_mix) == sorted(
        c["n"] for c in dec_solo_a + dec_solo_b)
    assert all(c["n"] < sum(x["n"] for x in dec_solo_a + dec_solo_b)
               for c in dec_mix)


def test_b_g1_determinism_mixed_round(tmp_path):
    """同混批两跑 → 逐请求音频字节一致 (G1)。"""
    ref = _make_ref(tmp_path)
    in_a = _inputs("s13|s9|s11", seed=11, ref=ref)
    in_b = _inputs("s14|s16", seed=22, ref=ref)

    tts1 = _make_fake_tts()
    _, jobs1 = _run_round(tts1, [in_a, in_b])
    r11 = jobs1[0].future.result(timeout=30)
    r12 = jobs1[1].future.result(timeout=30)

    tts2 = _make_fake_tts()
    _, jobs2 = _run_round(tts2, [in_a, in_b])
    r21 = jobs2[0].future.result(timeout=30)
    r22 = jobs2[1].future.result(timeout=30)

    assert r11[0] == r21[0] and np.array_equal(r11[1], r21[1])
    assert r12[0] == r22[0] and np.array_equal(r12[1], r22[1])


# ---------------------------------------------------------------------------
# C. 签名分组
# ---------------------------------------------------------------------------

def test_c_signature_grouping_two_weight_sets(tmp_path):
    """不同权重 → 两组, _ensure_weights 逐组调用; 组间互不串。"""
    ref1 = str(tmp_path / "r1.wav")
    ref2 = str(tmp_path / "r2.wav")
    for p in (ref1, ref2):
        Path(p).write_bytes(b"")
    in_a = _inputs("s13|s9", seed=11, ref=ref1, t2s_w="tA", vits_w="vA")
    in_b = _inputs("s14|s16", seed=22, ref=ref2, t2s_w="tB", vits_w="vB")

    tts = _make_fake_tts()
    engine = FakeEngine(tts)
    jobs = [gc._Job(i) for i in (in_a, in_b)]
    gc._Coalescer(engine)._round(jobs)

    assert engine.ensured == [("vA", "tA"), ("vB", "tB")]
    assert all(not j.future.exception() for j in jobs)
    assert len(jobs[0].future.result()[1]) > 0
    assert len(jobs[1].future.result()[1]) > 0


def test_c_signature_failure_isolation(tmp_path):
    """签名构造失败 (speed_factor 非数字) → 仅该请求报错, 其余照常。"""
    ref = _make_ref(tmp_path)
    in_a = _inputs("s13", seed=11, ref=ref)
    in_bad = _inputs("s14", seed=22, ref=ref, speed="fast")

    tts = _make_fake_tts()
    _, jobs = _run_round(tts, [in_a, in_bad])

    with pytest.raises(ValueError):
        jobs[1].future.result(timeout=30)
    sr, audio = jobs[0].future.result(timeout=30)
    assert sr == SR and len(audio) > 0


# ---------------------------------------------------------------------------
# D. row_sampling 逐行覆盖
# ---------------------------------------------------------------------------

def test_d_row_sampling_per_request_override(tmp_path, monkeypatch):
    """采样参数异于组基线的请求 → 其行带覆盖 dict, 基线行为 None。"""
    ref = _make_ref(tmp_path)
    in_a = _inputs("s13|s9|s11", seed=11, ref=ref)              # 基线参数
    in_b = _inputs("s14|s16", seed=22, ref=ref, top_k=1)        # 覆盖参数

    tts = _make_fake_tts()
    spy = _spy_infer_panel(tts, monkeypatch)
    _, jobs = _run_round(tts, [in_a, in_b])

    assert all(not j.future.exception() for j in jobs)
    rs = spy[0]["kw"]["row_sampling"]
    assert rs is not None and len(rs) == 5
    base = {"top_k": 1, "top_p": 1.0, "temperature": 1.0,
            "repetition_penalty": 1.35}
    assert sum(1 for r in rs if r == base) == 2              # B 的 2 段
    assert sum(1 for r in rs if r is None) == 3              # A 的 3 段


# ---------------------------------------------------------------------------
# E. 失败隔离三级
# ---------------------------------------------------------------------------

def test_e_empty_split_silence(tmp_path):
    """空切分请求 → (16000, 16000 静音); 同轮其余请求照常 (隔离第 2 级)。"""
    ref = _make_ref(tmp_path)
    in_empty = _inputs("", seed=11, ref=ref)
    in_b = _inputs("s14|s16", seed=22, ref=ref)

    tts = _make_fake_tts()
    _, jobs = _run_round(tts, [in_empty, in_b])

    sr, audio = jobs[0].future.result(timeout=30)
    assert sr == 16000
    assert np.array_equal(audio, np.zeros(16000, dtype=np.int16))
    sr_b, audio_b = jobs[1].future.result(timeout=30)
    assert sr_b == SR and len(audio_b) > 0


def test_e_preprocess_exception_isolation(tmp_path):
    """预处理异常 → 该请求报错, 同组其余请求音频照常 (隔离第 1 级)。"""
    ref = _make_ref(tmp_path)
    in_a = _inputs("s13|s9|s11", seed=11, ref=ref)   # 正常
    in_bad = _inputs("s14|s16", seed=22, ref=ref)    # preprocess 将抛

    tts = _make_fake_tts()
    tts.text_preprocessor = FakePreprocessor(fail_texts={"s14|s16"})
    _, jobs = _run_round(tts, [in_a, in_bad])

    with pytest.raises(ValueError, match="boom-preprocess"):
        jobs[1].future.result(timeout=30)
    sr_a, audio_a = jobs[0].future.result(timeout=30)
    # A 独占池子 → 与 solo 参照位等 (同种子同前端)
    tts_solo = _make_fake_tts()
    tts_solo.text_preprocessor = FakePreprocessor(fail_texts={"s14|s16"})
    _, j_solo = _run_round(tts_solo, [in_a])
    _, audio_solo = j_solo[0].future.result(timeout=30)
    assert sr_a == SR
    assert np.array_equal(audio_a, audio_solo)


def test_e_synthesis_exception_group_semantics(tmp_path):
    """合成中途异常 → 组内全部未决请求报错 + init_*_weights 重载
    (stock L1551-1562 语义, 隔离第 3 级)。"""
    ref = _make_ref(tmp_path)
    in_a = _inputs("s13|s9|s11", seed=11, ref=ref)
    in_b = _inputs("s14|s16", seed=22, ref=ref)

    tts = _make_fake_tts()
    t2s_inits0, vits_inits0 = tts.t2s_inits, tts.vits_inits

    def boom(*a, **k):
        raise RuntimeError("boom-decode")

    tts.vits_model.decode = boom
    _, jobs = _run_round(tts, [in_a, in_b])

    for j in jobs:
        with pytest.raises(RuntimeError, match="boom-decode"):
            j.future.result(timeout=30)
    # 模型重置: del + 按 configs 路径重载 (计数 +1)
    assert tts.t2s_inits == t2s_inits0 + 1
    assert tts.vits_inits == vits_inits0 + 1


# ---------------------------------------------------------------------------
# F. solo 回退 (no_prompt / vocoder)
# ---------------------------------------------------------------------------

def test_f_no_prompt_group_solo_fallback(monkeypatch):
    """no_prompt 组整组 solo 回退 (行种子桥 fallback 缺口 → 音质优先)。"""
    in_a = _inputs("s13|s9", seed=11, prompt_text="")
    in_b = _inputs("s14|s16", seed=22, prompt_text="")

    tts = _make_fake_tts()
    spy = _spy_infer_panel(tts, monkeypatch)
    jobs = [gc._Job(i) for i in (in_a, in_b)]
    gc._Coalescer(FakeEngine(tts))._round(jobs)

    assert len(tts.run_calls) == 2                      # 逐请求 tts.run
    assert all(i["prompt_text"] == "" for i in tts.run_calls)
    assert spy == []                                    # 不碰 infer_panel
    assert tts.vits_model.calls == []                   # 不碰 vits decode
    assert all(not j.future.exception() for j in jobs)
    for j in jobs:
        sr, audio = j.future.result(timeout=30)
        assert sr == SR and len(audio) == 16


def test_f_vocoder_group_solo_fallback(monkeypatch):
    """vocoder (V3/V4) 组整组 solo 回退 (合成走 using_vocoder_synthesis_*,
    位等面不同)。"""
    in_a = _inputs("s13|s9", seed=11)
    in_b = _inputs("s14|s16", seed=22)

    tts = _make_fake_tts(use_vocoder=True)
    spy = _spy_infer_panel(tts, monkeypatch)
    jobs = [gc._Job(i) for i in (in_a, in_b)]
    gc._Coalescer(FakeEngine(tts))._round(jobs)

    assert len(tts.run_calls) == 2
    assert spy == []
    assert all(not j.future.exception() for j in jobs)


# ---------------------------------------------------------------------------
# G. settings 旋钮
# ---------------------------------------------------------------------------

def test_g_settings_coalesce(monkeypatch):
    monkeypatch.delenv("ADR_T2S_COALESCE", raising=False)
    assert settings.t2s_coalesce_enabled() is True       # 默认开
    monkeypatch.setenv("ADR_T2S_COALESCE", "0")
    assert settings.t2s_coalesce_enabled() is False
    monkeypatch.setenv("ADR_T2S_COALESCE", "1")
    assert settings.t2s_coalesce_enabled() is True
    monkeypatch.setenv("ADR_T2S_COALESCE", "anything")
    assert settings.t2s_coalesce_enabled() is True       # 仅 "0" 关

    monkeypatch.delenv("ADR_T2S_COALESCE_BATCH", raising=False)
    assert settings.t2s_coalesce_batch() == 4            # 默认 4
    for bad in ("abc", "0", "-3", "", "  "):
        monkeypatch.setenv("ADR_T2S_COALESCE_BATCH", bad)
        assert settings.t2s_coalesce_batch() == 4
    monkeypatch.setenv("ADR_T2S_COALESCE_BATCH", "2")
    assert settings.t2s_coalesce_batch() == 2
    monkeypatch.setenv("ADR_T2S_COALESCE_BATCH", "64")
    assert settings.t2s_coalesce_batch() == 64           # 不封顶


# ---------------------------------------------------------------------------
# H. worker 集成
# ---------------------------------------------------------------------------

def test_h_worker_submit_and_cap_rounds(tmp_path, monkeypatch):
    """submit → future 回填; cap=2 时 4 请求分轮处理不卡死。"""
    monkeypatch.setenv("ADR_T2S_COALESCE_BATCH", "2")
    ref = _make_ref(tmp_path)
    tts = _make_fake_tts()
    engine = FakeEngine(tts)
    c = gc._Coalescer(engine)

    futs = [c.submit(_inputs(f"s13|s9|s{i}", seed=100 + i, ref=ref))
            for i in range(4)]
    results = [f.result(timeout=120) for f in futs]
    assert all(sr == SR and a.dtype == np.int16 and len(a) > 0
               for sr, a in results)
    assert len(engine.ensured) >= 1


def test_h2_worker_signature_buckets(tmp_path, monkeypatch):
    """分桶凑轮: 窗口内异签名各自成轮, 同桶一次热换独占整批。

    cap=4, A×2 + B×2 同刻提交 (签名 = 权重集不同):
    - 旧语义一轮 2 组 → 组间热换 2 次, 每组只 2 条;
    - 分桶语义 → 2 轮各 1 组, ensured 恰 2 次且按首现顺序。
    """
    monkeypatch.setenv("ADR_T2S_COALESCE_BATCH", "4")
    ref = _make_ref(tmp_path)
    tts = _make_fake_tts()
    engine = FakeEngine(tts)
    c = gc._Coalescer(engine)

    futs = [c.submit(_inputs(f"s13|s9|s{i}", seed=100 + i, ref=ref,
                             vits_w=f"vits{'A' if i < 2 else 'B'}",
                             t2s_w=f"t2s{'A' if i < 2 else 'B'}"))
            for i in range(4)]
    results = [f.result(timeout=120) for f in futs]
    assert all(sr == SR and a.dtype == np.int16 and len(a) > 0
               for sr, a in results)
    # 每桶一次热换 (桶内同签名不再重换), 桶序 = 首现顺序
    assert engine.ensured == [("vitsA", "t2sA"), ("vitsB", "t2sB")]
