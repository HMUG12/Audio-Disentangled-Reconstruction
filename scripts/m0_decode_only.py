"""Decode the previously generated codes_0.npy to a wav file using Fish Speech's codec."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "fish-speech"))
from fish_speech.models.text2semantic.inference import load_codec_model  # noqa: E402

CHECKPOINT_DIR = Path("third_party/fish-speech/checkpoints/openaudio-s1-mini")
CODES_NPY = Path("output/codes_0.npy")
OUT_WAV = Path("output/fish_smoke.wav")

device = "cuda"
dtype = torch.bfloat16

print("Loading codec...")
codec = load_codec_model(CHECKPOINT_DIR / "codec.pth", device=device, precision=dtype)
print("codec sample rate:", codec.sample_rate)

print("Loading codes from", CODES_NPY)
codes = np.load(CODES_NPY)
print("codes shape:", codes.shape, "dtype:", codes.dtype)
codes_tensor = torch.from_numpy(codes).long().to(device)

print("Decoding...")
with torch.inference_mode():
    audio = codec.from_indices(codes_tensor[None])[0, 0]

import soundfile as sf

audio_np = audio.cpu().float().numpy()
sf.write(OUT_WAV, audio_np, codec.sample_rate)
print("Saved", OUT_WAV, "duration:", len(audio_np) / codec.sample_rate, "s")
