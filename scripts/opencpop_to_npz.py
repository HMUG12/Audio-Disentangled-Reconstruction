"""Convert OpenCpop segments to ADR TrainSample npz (M2 方向 A)。

Input:
- data/opencpop/wavs/*.wav  (44100 Hz, utterance-level)
- data/opencpop/transcriptions.txt  (id|text|phonemes|notes|note_dur|phoneme_dur|slur_flags)

Output:
- data/opencpop_npz/samples/{id}.npz
- data/opencpop_npz/metadata.json

核心优化:
- F0 从 notes 推导 (Ground Truth, 1:1 对齐 phoneme + 帧),比 pyin 准
- 多进程 (默认 4 workers)
- 跳过 ASR/G2P (数据已带)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(r"E:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))


# MIDI → Hz
def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def note_name_to_midi(name: str) -> float:
    """'G#4/Ab4' → MIDI 58 (Ab4). 取第一个音名。"""
    if name == "rest":
        return 0.0
    # 形如 'G#4/Ab4', 'E4', 'rest'
    # 取 '/' 前部分
    primary = name.split("/")[0].strip()
    # 解析: 字母 + 可选 #/b + 八度
    if len(primary) < 2:
        return 0.0
    pitch_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    letter = primary[0]
    if letter not in pitch_map:
        return 0.0
    semitone = pitch_map[letter]
    idx = 1
    if idx < len(primary) and primary[idx] == "#":
        semitone += 1
        idx += 1
    elif idx < len(primary) and primary[idx] == "b":
        semitone -= 1
        idx += 1
    try:
        octave = int(primary[idx:])
    except ValueError:
        return 0.0
    return 12 * (octave + 1) + semitone


def parse_transcriptions(path: Path) -> Dict[str, dict]:
    """解析 transcriptions.txt。

    格式: id|text|phonemes|notes|note_dur|phoneme_dur|slur_flags

    Returns:
        {utt_id: {text, phonemes, notes, phoneme_dur}}
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 6:
                continue
            utt_id = parts[0]
            text = parts[1]
            phonemes = parts[2].split() if parts[2] else []
            notes = parts[3].split() if parts[3] else []
            try:
                phoneme_dur = [float(x) for x in parts[5].split()]
            except (ValueError, IndexError):
                continue
            if not phonemes or not notes:
                continue
            mapping[utt_id] = {
                "text": text,
                "phonemes": phonemes,
                "notes": notes,
                "phoneme_dur": phoneme_dur,
            }
    return mapping


def notes_to_f0(notes: List[str], phoneme_dur: List[float], hop_length: int, sr: int) -> np.ndarray:
    """从 notes + phoneme_dur 推导 per-frame F0。

    对齐规则: notes[i] 持续 phoneme_dur[i] 秒,产生 floor(phoneme_dur[i]*sr/hop_length) 帧。
    """
    total_sec = sum(phoneme_dur)
    n_frames = max(1, int(total_sec * sr / hop_length))
    f0 = np.zeros(n_frames, dtype=np.float32)

    cur_frame = 0
    for note, dur in zip(notes, phoneme_dur):
        n_f = max(1, int(dur * sr / hop_length))
        midi = note_name_to_midi(note)
        hz = midi_to_hz(midi) if midi > 0 else 0.0
        f0[cur_frame:cur_frame + n_f] = hz
        cur_frame += n_f

    # 截断到实际帧数
    return f0[:n_frames]


def process_one_utterance(
    utt_id: str,
    info: dict,
    wav_dir: Path,
    out_dir: Path,
    target_sr: int,
    hop_length: int,
) -> Optional[dict]:
    """处理单个 utterance。"""
    wav_path = wav_dir / f"{utt_id}.wav"
    if not wav_path.exists():
        return None

    try:
        # 1. 加载 wav (44100 Hz mono)
        import soundfile as sf
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=-1)

        # 2. 重采样到 target_sr
        if sr != target_sr:
            import librosa
            waveform = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        else:
            waveform = data

        # 3. F0 从 notes 推导 (ground truth, 比 pyin 准且快)
        f0 = notes_to_f0(
            info["notes"], info["phoneme_dur"],
            hop_length=hop_length, sr=target_sr,
        )

        # 4. mel — 必须与 BigVGAN 声码器 convention 严格一致 (v2)
        from adr.utils.audio import compute_mel_bigvgan
        mel = compute_mel_bigvgan(
            waveform, sample_rate=target_sr,
            n_mels=80, hop_length=hop_length,
        )

        # 5. 保存 npz
        out_path = out_dir / f"{utt_id}.npz"
        np.savez_compressed(
            out_path,
            sample_id=np.array([utt_id], dtype=object),
            waveform=waveform.astype(np.float32),
            sample_rate=target_sr,
            text=np.array([info["text"]], dtype=object),
            phonemes=np.array([info["phonemes"]], dtype=object),
            f0=f0,
            mel=mel,
            start_sec=0.0,
            end_sec=len(waveform) / target_sr,
        )

        return {
            "id": utt_id,
            "duration": len(waveform) / target_sr,
            "n_phonemes": len(info["phonemes"]),
            "n_notes": len(info["notes"]),
            "ok": True,
        }
    except Exception as e:
        return {"id": utt_id, "ok": False, "error": str(e)}


def process_wrapper(args):
    """multiprocessing wrapper."""
    utt_id, info, wav_dir, out_dir, sr, hop = args
    return process_one_utterance(utt_id, info, wav_dir, out_dir, sr, hop)


def main():
    parser = argparse.ArgumentParser(description="Convert OpenCpop to ADR TrainSample npz")
    parser.add_argument("--src", default=str(REPO / "data" / "opencpop"))
    parser.add_argument("--dst", default=str(REPO / "data" / "opencpop_npz"))
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    samples_dir = dst / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = src / "wavs"
    trans_path = src / "transcriptions.txt"

    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV dir not found: {wav_dir}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Transcriptions not found: {trans_path}")

    # 1. 解析 transcriptions
    print(f"[1/3] Parsing {trans_path} ...")
    t0 = time.time()
    labels = parse_transcriptions(trans_path)
    print(f"  Loaded {len(labels)} utterances in {time.time()-t0:.1f}s")

    # 2. 准备任务
    tasks = []
    for utt_id, info in labels.items():
        tasks.append((utt_id, info, wav_dir, samples_dir, args.sr, args.hop))

    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"[2/3] Processing {len(tasks)} utterances (workers={args.workers}) ...")

    # 3. 多进程
    t0 = time.time()
    if args.workers <= 1:
        results = [process_wrapper(t) for t in tasks]
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.map(process_wrapper, tasks)

    n_ok = sum(1 for r in results if r and r.get("ok"))
    n_fail = len(results) - n_ok
    elapsed = time.time() - t0
    print(f"  Wav processing: {n_ok} ok, {n_fail} failed in {elapsed:.1f}s")
    if n_ok > 0:
        print(f"  Throughput: {n_ok/elapsed:.1f} samples/sec, {elapsed/n_ok*1000:.0f} ms/sample")

    # 4. 统计
    durations = [r["duration"] for r in results if r and r.get("ok") and "duration" in r]
    n_phonemes = [r["n_phonemes"] for r in results if r and r.get("ok")]
    total_h = sum(durations) / 3600 if durations else 0

    metadata = {
        "source": "opencpop",
        "n_samples": n_ok,
        "n_skipped": n_fail,
        "total_hours": round(total_h, 2),
        "target_sr": args.sr,
        "hop_length": args.hop,
        "phoneme_count_stats": {
            "min": int(np.min(n_phonemes)) if n_phonemes else 0,
            "max": int(np.max(n_phonemes)) if n_phonemes else 0,
            "mean": round(float(np.mean(n_phonemes)), 1) if n_phonemes else 0,
            "median": int(np.median(n_phonemes)) if n_phonemes else 0,
        },
        "duration_stats_sec": {
            "min": round(float(np.min(durations)), 2) if durations else 0,
            "max": round(float(np.max(durations)), 2) if durations else 0,
            "mean": round(float(np.mean(durations)), 2) if durations else 0,
            "median": round(float(np.median(durations)), 2) if durations else 0,
        },
    }

    meta_path = dst / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved {n_ok} samples to {samples_dir}")
    print(f"     Total: {total_h:.2f} hours")
    print(f"     Phonemes/utt: mean={metadata['phoneme_count_stats']['mean']}, "
          f"max={metadata['phoneme_count_stats']['max']}")
    print(f"     Duration: mean={metadata['duration_stats_sec']['mean']}s, "
          f"max={metadata['duration_stats_sec']['max']}s")
    print(f"     Metadata: {meta_path}")


if __name__ == "__main__":
    main()
