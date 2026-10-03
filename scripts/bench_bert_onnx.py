"""批次4-3: CPU ONNX Runtime 实验 — BERT 特征提取提速实测。

背景 (批次3 profile): 纯 CPU 推理 RTF≈1.9, conv 密集无单点热点;
BERT (roberta-large) 占 CPU 时间 ~1/4。GSV 用倒数第 3 层 hidden states
(TextPreprocessor.get_bert_feature)。
ONNX Runtime 图优化 (LayerNorm/GELU/Attention 融合) 对 transformer
编码器通常有 1.3-2x 收益 — 本脚本实测导出 + 推理提速, 验证后再决定
是否接入 gsv_engine 的 CPU 路径。

用法: python scripts/bench_bert_onnx.py [--seq 64 128 256] [--int8]
输出: output/bench_bert_onnx.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

GSV_BERT = REPO / "third_party" / "gpt_sovits" / "GPT_SoVITS" / \
    "pretrained_models" / "chinese-roberta-wwm-ext-large"
ONNX_OUT = REPO / "output" / "bert_wwm_ext_large_feat.onnx"


class BertFeat(torch.nn.Module):
    """GSV 实际消费面: 倒数第 3 层 hidden states (1024 维)。"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.m(input_ids, attention_mask=attention_mask,
                     token_type_ids=token_type_ids, output_hidden_states=True)
        return out.hidden_states[-3]


def _bench(fn, inputs, n=10, warmup=3):
    for _ in range(warmup):
        fn(*inputs)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*inputs)
        ts.append(time.perf_counter() - t0)
    return min(ts), sum(ts) / len(ts)


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--int8", action="store_true", help="附加 ORT int8 动态量化对比")
    args = ap.parse_args()

    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

    tok = AutoTokenizer.from_pretrained(str(GSV_BERT))
    model = AutoModel.from_pretrained(str(GSV_BERT)).eval()  # CPU fp32
    wrapper = BertFeat(model).eval()
    print(f"[env] torch {torch.__version__}, threads={torch.get_num_threads()}")

    # ---- 导出 (dynamic seq axis) ----
    if not ONNX_OUT.exists():
        dummy = tok("人工智能技术正在以前所未有的速度发展", return_tensors="pt")
        ids = dummy["input_ids"][:1, :16]
        am = dummy["attention_mask"][:1, :16]
        tt = dummy["token_type_ids"][:1, :16]
        print(f"[export] -> {ONNX_OUT.name}")
        torch.onnx.export(
            wrapper, (ids, am, tt), str(ONNX_OUT),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["feat"],
            dynamic_axes={n: {0: "batch", 1: "seq"}
                          for n in ["input_ids", "attention_mask",
                                    "token_type_ids", "feat"]},
            opset_version=17, do_constant_folding=True,
        )
    opts = SessionOptions()
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = torch.get_num_threads()
    sess = InferenceSession(str(ONNX_OUT), opts,
                            providers=["CPUExecutionProvider"])
    print("[export] ORT session ready (ORT_ENABLE_ALL)")

    # int8 动态量化 (可选, 一次性质化; 外部数据模型上 ORT 量化可能失败, 非致命)
    qsess = None
    if args.int8:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            q = REPO / "output" / "bert_wwm_ext_large_feat.int8.onnx"
            if not q.exists():
                quantize_dynamic(str(ONNX_OUT), str(q),
                                 weight_type=QuantType.QInt8)
            qsess = InferenceSession(str(q), opts,
                                     providers=["CPUExecutionProvider"])
            print("[int8] quantized session ready")
        except Exception as e:
            print(f"[int8] 量化失败 (跳过): {type(e).__name__}: {e}")

    result = {"torch_threads": torch.get_num_threads(), "lengths": {}}
    for L in args.seq:
        text = "语音合成低资源测试" * (L // 8 + 1)
        d = tok(text, return_tensors="pt", truncation=True, max_length=L)
        ids, am = d["input_ids"][:1, :L], d["attention_mask"][:1, :L]
        tt = d["token_type_ids"][:1, :L]
        feed = {"input_ids": ids.numpy(), "attention_mask": am.numpy(),
                "token_type_ids": tt.numpy()}

        tmin, tavg = _bench(wrapper, (ids, am, tt))
        omin, oavg = _bench(lambda *a: sess.run(["feat"], feed),
                            (ids, am, tt))
        # 数值一致性 (相对误差)
        ref = wrapper(ids, am, tt).detach().numpy()
        got = sess.run(["feat"], feed)[0]
        err = float(abs(ref - got).max() / (abs(ref).max() + 1e-9))

        row = {"seq": L,
               "torch_min_ms": round(tmin * 1000, 1),
               "ort_min_ms": round(omin * 1000, 1),
               "speedup_x": round(tmin / omin, 2),
               "max_rel_err": round(err, 6)}
        if qsess is not None:
            qmin, _ = _bench(lambda *a: qsess.run(["feat"], feed),
                             (ids, am, tt))
            row["ort_int8_min_ms"] = round(qmin * 1000, 1)
            row["ort_int8_speedup_x"] = round(tmin / qmin, 2)
        result["lengths"][str(L)] = row
        print(f"[L={L}] torch {row['torch_min_ms']}ms | ort "
              f"{row['ort_min_ms']}ms ({row['speedup_x']}x)"
              + (f" | int8 {row.get('ort_int8_min_ms')}ms "
                 f"({row.get('ort_int8_speedup_x')}x)" if args.int8 else "")
              + f" | rel_err {row['max_rel_err']}", flush=True)

    out = REPO / "output" / "bench_bert_onnx.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
