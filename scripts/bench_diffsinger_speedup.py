"""W2: DiffSinger 采样压缩实验 — speedup 步数 vs 合成时间 vs 音频产出。

背景: 基准 27.3s 产 4s 唱歌音频 (1000 步扩散, PNDM speedup=40)。
扫 speedup 40/80/160/240, 时间数据入库, 音频落盘供听感评审。

用法: python scripts/bench_diffsinger_speedup.py
产物: output/ds_speedup_*.wav + output/ds_speedup.json
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "output" / "ds_speedup.json"
WAV_DIR = REPO / "output" / "ds_speedup"
SPEEDUPS = [40, 80, 160, 240]


def main():
    import soundfile as sf
    from adr.models.diffsinger_engine import get_engine

    eng = get_engine()
    # 自动取 OpenCpop 标注表第一个可用句子 (5s 级标注)
    trans = REPO / "data" / "opencpop" / "transcriptions.txt"
    utt_id = None
    for line in open(trans, encoding="utf-8"):
        f = line.strip().split("|")
        if len(f) >= 2:
            utt_id = f[0].strip()
            break
    assert utt_id, "transcriptions.txt 为空"
    kwargs = eng.load_opencpop_annotation(utt_id)
    label = kwargs.pop("text")

    rows = []
    for sp in SPEEDUPS:
        t0 = time.perf_counter()
        wav, sr = eng.synthesize(speedup=sp, **kwargs)
        dt = time.perf_counter() - t0
        audio_s = len(wav) / sr
        out_p = WAV_DIR / f"ds_speedup_{sp}.wav"
        WAV_DIR.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), wav, sr)
        rows.append({"speedup": sp, "gen_s": round(dt, 2),
                     "audio_s": round(audio_s, 2),
                     "rtf": round(dt / audio_s, 2)})
        print(f"  speedup={sp:3d}: {dt:5.1f}s / {audio_s:.2f}s 音频 "
              f"= RTF {dt / audio_s:5.2f}  → {out_p.name}", flush=True)

    OUT_JSON.write_text(json.dumps({"utt": "2069003755", "text": label,
                                    "rows": rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[bench] 已写 {OUT_JSON} (听感自评: 速度 40 vs 240 是否可辨)")


if __name__ == "__main__":
    main()
