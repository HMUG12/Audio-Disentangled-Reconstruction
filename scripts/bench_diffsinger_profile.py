"""批次3-3: DiffSinger GPU profile — 定位 33s 固定开销 (与扩散步数无关)。

用法: python scripts/bench_diffsinger_profile.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    import torch
    from torch.profiler import ProfilerActivity, profile
    from adr.models.diffsinger_engine import get_engine

    eng = get_engine()
    trans = REPO / "data" / "opencpop" / "transcriptions.txt"
    utt_id = None
    for line in open(trans, encoding="utf-8"):
        f = line.strip().split("|")
        if len(f) >= 2:
            utt_id = f[0].strip()
            break
    kwargs = eng.load_opencpop_annotation(utt_id)
    kwargs.pop("text", None)

    eng.synthesize(speedup=80, **kwargs)  # 预热不计
    print("[ds-profile] 预热完成, profile speedup=80 一次合成")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True) as prof:
        eng.synthesize(speedup=80, **kwargs)
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=30, max_name_column_width=55))


if __name__ == "__main__":
    main()
