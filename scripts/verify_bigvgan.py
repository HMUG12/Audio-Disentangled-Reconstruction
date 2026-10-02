"""验证 BigVGAN-base 22kHz 80-band 能从本地权重加载并生成 wav。"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 复用 fish-speech venv
FISH_VENV_SITE = Path(__file__).resolve().parent.parent / "third_party" / "fish-speech" / ".venv" / "Lib" / "site-packages"
if FISH_VENV_SITE.exists():
    sys.path.insert(0, str(FISH_VENV_SITE))

# 把 BigVGAN 源码加到 path
BIGVGAN_DIR = Path(r"F:\ADR_data\bigvgan")
sys.path.insert(0, str(BIGVGAN_DIR))


def main():
    # 加载 config
    from bigvgan import BigVGAN, load_hparams_from_json
    h = load_hparams_from_json(str(BIGVGAN_DIR / "config.json"))
    print("Config: sr=%d, mel=%d, hop=%d" % (h.sampling_rate, h.num_mels, h.hop_size))
    # 加载模型
    model = BigVGAN(h)
    state = torch.load(BIGVGAN_DIR / "bigvgan_generator.pt", map_location="cpu", weights_only=True)
    if "generator" in state:
        state = state["generator"]
    model.load_state_dict(state, strict=False)
    print("BigVGAN loaded OK, params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    # 随机 mel 测一波
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    mel = torch.randn(1, h.num_mels, 200, device=device)
    with torch.inference_mode():
        t0 = time.time()
        audio = model(mel)
        dt = time.time() - t0
    print(f"Generated {audio.shape} in {dt:.2f}s")

    import soundfile as sf
    OUT = Path("output/bigvgan_smoke.wav")
    audio_np = audio[0, 0].cpu().float().numpy()
    sf.write(OUT, audio_np, h.sampling_rate)
    print(f"Saved {OUT}, sr={h.sampling_rate}, duration={len(audio_np)/h.sampling_rate:.2f}s")


if __name__ == "__main__":
    main()
