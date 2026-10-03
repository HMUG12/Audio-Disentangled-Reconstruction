"""批次3-2: CPU 推理分段 profile — 定位真瓶颈, 决定 ONNX 策略是否值得做。

用法: python scripts/bench_cpu_profile.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    import torch
    from torch.profiler import ProfilerActivity, profile
    from adr.models.gsv_engine import GSVEngine
    from adr.models.voice_library import load_voice

    prof_meta = load_voice("我的声音V2")
    eng = GSVEngine()
    eng.config.device = "cpu"
    eng.config.half = False
    eng.warmup(vits_weights=prof_meta.get("vits_weights"),
               t2s_weights=prof_meta.get("t2s_weights"))

    text = "今天天气不错，我们去公园散步吧。"
    ref = prof_meta["ref_audio"]
    vits = prof_meta.get("vits_weights")
    eng.synthesize(text, ref, vits_weights=vits)  # 预热不计
    print("[cpu-profile] 预热完成, 开始 profile 一次合成")

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        eng.synthesize(text, ref, vits_weights=vits)
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=25, max_name_column_width=55))


if __name__ == "__main__":
    main()
