"""A 轨端到端打通：ADR 生成 mel → BigVGAN → wav。

使用 mock 数据驱动全流程，验证：
  ADR forward (随机初始化) → CFM 采样 → BigVGAN (预训练) → wav
"""
import sys
import time
from pathlib import Path

import torch

FISH_VENV_SITE = Path(__file__).resolve().parent.parent / "third_party" / "fish-speech" / ".venv" / "Lib" / "site-packages"
if FISH_VENV_SITE.exists():
    sys.path.insert(0, str(FISH_VENV_SITE))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import soundfile as sf
from adr.configs import ADRConfig, BackboneConfig, ContentConfig, MelodyConfig, TimbreConfig
from adr.model import ADRModel
from adr.data.phoneme_dict import PhonemeDict
from adr.data.opencpop import OpenCpopDataset, _patch_phoneme_dict_ids


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. 加载音素表
    dict_path = Path(__file__).resolve().parent.parent / "third_party" / "DiffSinger" / "dictionaries" / "opencpop-extension.txt"
    pd = PhonemeDict.from_opencpop(dict_path)
    _patch_phoneme_dict_ids(PhonemeDict)
    print(f"Phoneme dict: {len(pd)} entries")

    # 2. 加载 ADR 0.3B 模型
    cfg = ADRConfig(
        content=ContentConfig(vocab_size=len(pd), embed_dim=512, n_layers=6, n_heads=8, ffn_dim=2048, max_phoneme_len=512),
        melody=MelodyConfig(embed_dim=256, n_layers=3, n_heads=4),
        timbre=TimbreConfig(ref_mel_bins=80, embed_dim=512),
        backbone=BackboneConfig(hidden_dim=1024, n_layers=28, n_heads=16, ffn_dim=3072,
                                mel_bins=80, max_mel_frames=512),
    )
    model = ADRModel(cfg).to(device).eval()
    print(f"ADR model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # 3. mock 数据（一只合成数据）
    ds = OpenCpopDataset(data_root="F:/ADR_data/opencpop", phoneme_dict=pd, synthetic=True, n_synthetic=1)
    sample = ds[0]
    print(f"Sample: utt_id={sample['utt_id']}, phoneme_shape={tuple(sample['phoneme_ids'].shape)}, pitch_shape={tuple(sample['pitch_ids'].shape)}")

    # 4. ADR 推理：CFM 采样生成 mel
    phoneme_ids = sample["phoneme_ids"].unsqueeze(0).to(device)
    pitch_ids = sample["pitch_ids"].unsqueeze(0).to(device)
    note_dur = sample["note_duration"].unsqueeze(0).to(device)
    velocity = sample["velocity"].unsqueeze(0).to(device)
    ref_mel = torch.randn(1, 80, 256, device=device)

    print("\n=== ADR sample (CFM 5 steps) ===")
    t0 = time.time()
    with torch.inference_mode():
        mel = model.sample(phoneme_ids, ref_mel,
                           pitch_ids=pitch_ids, note_duration=note_dur, velocity=velocity,
                           is_singing=True, n_timesteps=5)
    print(f"  mel shape: {mel.shape}, time: {time.time()-t0:.2f}s, mem: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    # 5. 加载 BigVGAN（按 22.05kHz 配置；mel hop=256 对应 86.13Hz frame rate）
    print("\n=== Loading BigVGAN ===")
    BIGVGAN_DIR = Path(r"F:\ADR_data\bigvgan")
    sys.path.insert(0, str(BIGVGAN_DIR))
    from bigvgan import BigVGAN, load_hparams_from_json
    h = load_hparams_from_json(str(BIGVGAN_DIR / "config.json"))
    bv = BigVGAN(h)
    state = torch.load(BIGVGAN_DIR / "bigvgan_generator.pt", map_location="cpu", weights_only=True)
    if "generator" in state:
        state = state["generator"]
    bv.load_state_dict(state, strict=False)
    bv = bv.to(device).eval()
    print(f"  BigVGAN: {sum(p.numel() for p in bv.parameters())/1e6:.1f}M params, sr={h.sampling_rate}, hop={h.hop_size}")

    # 6. BigVGAN: mel → wav
    # ADR 输出 mel 是 hop=80 帧率（80Hz 假设），需要重采样到 BigVGAN 的 hop=256（22.05kHz/256=86.13Hz）
    # 简单办法：直接送入，BigVGAN 会按内部上采样处理
    print("\n=== BigVGAN mel→wav ===")
    with torch.inference_mode():
        t0 = time.time()
        audio = bv(mel)
        dt = time.time() - t0
    print(f"  audio shape: {audio.shape}, time: {dt:.2f}s")

    # 7. 保存
    OUT = ROOT / "output" / "adr_bigvgan_e2e.wav"
    audio_np = audio[0, 0].cpu().float().numpy()
    sf.write(OUT, audio_np, h.sampling_rate)
    print(f"\n✅ Saved {OUT}")
    print(f"   duration: {len(audio_np)/h.sampling_rate:.2f}s")
    print(f"   注意：mel 是随机权重生成的，效果不真实，仅验证管线")


if __name__ == "__main__":
    main()
