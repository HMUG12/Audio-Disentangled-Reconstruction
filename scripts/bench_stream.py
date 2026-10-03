"""W1: 流式首包延迟实验 — 切句方式 × 文本长度 对首块延迟的影响。

背景: 基准记录流式首块延迟 17.4s (cut1 凑四句一切, 首段太长)。
假设: 首段越短, 语义 token 生成越快, 首包越快。cut5 按标点切应最快。

用法: python scripts/bench_stream.py
产物: output/bench_stream.json
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "output" / "bench_stream.json"

TEXTS = {
    "long_45": "人工智能技术正在以前所未有的速度发展，语音合成系统已经能够生成接近真人水平的自然流畅音频。",
    "long_commas": "声音克隆是一项非常有意思的技术，它只需要几秒钟的参考音频，就能够生成非常接近真人的语音效果，让我们一起来看看吧。",
    "two_20x2": "今天早上我喝了一杯咖啡。然后在公园里散了一会儿步，天气非常不错。",
    "short_12": "您好，请问有什么可以帮您的吗？",
}
METHODS = ["cut0", "cut1", "cut3", "cut5"]


def main():
    import numpy as np
    from adr.models.gsv_engine import GSVEngine
    from adr.models.voice_library import load_voice

    prof = load_voice("我的声音V2")
    ref = prof["ref_audio"]
    vits = prof.get("vits_weights")
    print(f"[bench] 档案权重: {Path(vits).name if vits else '(官方)'}")

    eng = GSVEngine()
    t0 = time.perf_counter()
    eng.warmup()
    print(f"[bench] 引擎预热 {time.perf_counter() - t0:.1f}s")

    rows = []
    for tname, text in TEXTS.items():
        for m in METHODS:
            t0 = time.perf_counter()
            t_first = n_chunks = None
            total_dur = 0.0
            sr = None
            for chunk, sr in eng.synthesize_stream(
                    text, ref, vits_weights=vits, split_method=m):
                if t_first is None:
                    t_first = time.perf_counter() - t0
                n_chunks = (n_chunks or 0) + 1
                total_dur += len(chunk) / sr
            total = time.perf_counter() - t0
            row = {"text": tname, "method": m,
                   "t_first_s": round(t_first, 2),
                   "chunks": n_chunks,
                   "total_s": round(total, 2),
                   "audio_s": round(total_dur, 2),
                   "rtf": round(total / total_dur, 2) if total_dur else None}
            rows.append(row)
            print(f"  {tname:9s} {m}: 首包 {t_first:5.2f}s | 块数 {n_chunks:3d} | "
                  f"总 {total:5.2f}s | 音频 {total_dur:4.2f}s | RTF {row['rtf']}",
                  flush=True)

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[bench] 已写 {OUT_JSON}")


if __name__ == "__main__":
    main()
