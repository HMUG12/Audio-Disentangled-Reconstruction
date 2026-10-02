"""Day 2 端到端验证: 合成 wav → 完整流水线。

会跑 (除 ASR 和 UVR5 外):
    1. 合成一个 8 秒 wav
    2. 静音检测 + 切片
    3. G2P 文本转音素 (手动提供文本)
    4. F0 提取
    5. Mel 频谱
    6. 保存为 npz + metadata.json
"""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np


def make_realistic_speech(
    path: str,
    duration_sec: float = 12.0,
    sample_rate: int = 24000,
) -> str:
    """生成更真实的合成语音: 基频 + 共振峰 + 静音段。"""
    n_samples = int(sample_rate * duration_sec)
    audio = np.zeros(n_samples, dtype=np.float32)

    # 模拟 3 段语音,中间有静音
    segments = [(0, 3.5), (4.0, 7.5), (8.0, 11.0)]
    for start, end in segments:
        s, e = int(start * sample_rate), int(end * sample_rate)
        t = np.linspace(0, end - start, e - s, dtype=np.float32)
        # 基频 200Hz (模拟男声)
        f0 = 200 + 20 * np.sin(2 * np.pi * 2 * t)  # F0 抖动
        # 累积相位
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        seg = 0.3 * np.sin(phase)
        # 共振峰
        seg += 0.15 * np.sin(phase * 2)
        seg += 0.08 * np.sin(phase * 3)
        # 包络 (避免突现)
        env = np.minimum(1.0, np.minimum(t / 0.1, (end - start - t) / 0.1))
        seg *= env
        audio[s:e] = seg

    # 噪声
    audio += 0.01 * np.random.randn(n_samples).astype(np.float32)

    # 归一化
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.9
    audio_int = (audio * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int.tobytes())
    return path


def main():
    print("=" * 60)
    print("Day 2 端到端验证")
    print("=" * 60)

    # 1. 合成 wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    out_dir = tempfile.mkdtemp(prefix="adr_day2_")

    try:
        make_realistic_speech(wav_path, duration_sec=12.0)
        print(f"\n[1/5] 已生成合成 wav: {wav_path}")
        print(f"      12 秒, 24kHz, 单声道")

        # 2. 测试 audio utils
        print(f"\n[2/5] 测试 audio I/O ...")
        from adr.utils import load_audio, compute_mel
        audio = load_audio(wav_path, sample_rate=24000)
        print(f"      ✓ 加载成功: {audio}")
        mel = compute_mel(audio.waveform, 24000, n_mels=80, hop_length=256)
        print(f"      ✓ Mel 频谱: shape={mel.shape}, range=[{mel.min():.2f}, {mel.max():.2f}]")

        # 3. 测试 slice
        print(f"\n[3/5] 测试音频切片 ...")
        from adr.data.slice import slice_audio, SliceConfig
        slices = slice_audio(
            audio,
            SliceConfig(min_sec=2.0, max_sec=4.0,
                        silence_threshold_db=-30, target_sr=24000),
        )
        print(f"      ✓ 切成 {len(slices)} 段:")
        for s in slices[:5]:
            print(f"        [{s.index}] {s.start_sec:.2f}-{s.end_sec:.2f}s ({s.duration:.2f}s)")

        # 4. 测试 G2P
        print(f"\n[4/5] 测试 G2P (pypinyin) ...")
        from adr.data.g2p import G2P, G2PConfig
        g2p = G2P(G2PConfig(backend="pypinyin", with_tone=True))
        for text in ["你好世界", "今天天气真好", "我爱北京天安门"]:
            phonemes = g2p(text)
            print(f"      '{text}' -> {phonemes}")

        # 5. 测试 F0
        print(f"\n[5/5] 测试 F0 提取 (pyin) ...")
        from adr.data.f0 import F0Extractor
        f0_ext = F0Extractor()
        if slices:
            f0 = f0_ext(slices[0].waveform, sample_rate=slices[0].sample_rate)
            stats = f0_ext.f0_statistics(f0)
            print(f"      ✓ Slice 0 F0: shape={f0.shape}, mean={stats['mean']:.1f}Hz, "
                  f"voiced_ratio={stats['voiced_ratio']:.2f}")

        # 6. 完整 pipeline (跳过 ASR, 用预填文本)
        print(f"\n[6/6] 测试 DataPipeline 完整编排 ...")
        from adr.data.pipeline import DataPipeline, PipelineConfig, TrainSample
        from adr.data.slice import SliceConfig

        # 预定义文本 (模拟 ASR 输出)
        test_texts = ["你好世界", "今天天气真好", "我爱北京天安门", "欢迎使用 ADR"]

        config = PipelineConfig(
            enable_separation=False,
            enable_slicing=True,
            enable_asr=False,
            enable_g2p=True,
            enable_f0=True,
            enable_mel=True,
            slice=SliceConfig(min_sec=2.0, max_sec=4.0,
                              silence_threshold_db=-30, target_sr=24000),
            output_dir=out_dir,
        )
        pipeline = DataPipeline(config)
        result = pipeline.run(wav_path)

        # 手动填文本 + 音素 (因为 ASR 跳过,slices 没有 text)
        # 直接构造 TrainSample 展示完整输出
        from adr.data.slice import slice_audio as do_slice
        slices = do_slice(audio, config.slice)
        for i, s in enumerate(slices):
            text = test_texts[i % len(test_texts)]
            phonemes = g2p(text)
            f0 = f0_ext(s.waveform, sample_rate=s.sample_rate)
            mel = compute_mel(s.waveform, s.sample_rate, n_mels=80, hop_length=256)
            ts = TrainSample(
                sample_id=f"synth_{i:04d}",
                waveform=s.waveform,
                sample_rate=s.sample_rate,
                text=text,
                phonemes=phonemes,
                f0=f0,
                mel=mel,
                start_sec=s.start_sec,
                end_sec=s.end_sec,
            )
            result.samples.append(ts)
            ts.save(Path(out_dir) / "samples" / f"{ts.sample_id}.npz")

        # 写 metadata
        import json
        meta = {
            "n_samples": len(result.samples),
            "input": wav_path,
            "duration": result.metadata.get("duration"),
            "skipped": result.skipped,
        }
        with open(Path(out_dir) / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Day 2 验证完成!")
        print(f"  输入: {wav_path}")
        print(f"  输出: {out_dir}")
        print(f"  样本数: {len(result.samples)}")
        print(f"  跳过: {result.skipped}")
        print(f"  错误: {len(result.errors)}")

        # 展示前 3 个样本
        for s in result.samples[:3]:
            print(f"\n  Sample: {s.sample_id}")
            print(f"    文本: {s.text}")
            print(f"    音素: {s.phonemes}")
            print(f"    时长: {s.end_sec - s.start_sec:.2f}s")
            print(f"    F0 shape: {s.f0.shape}, mel shape: {s.mel.shape if s.mel is not None else 'None'}")

        return 0
    finally:
        Path(wav_path).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
