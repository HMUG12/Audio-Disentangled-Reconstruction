"""下载 WavLM-base 与 BigVGAN-base 预训练权重到 F:\\ADR_data\\。"""
import os
import sys
from pathlib import Path

WAVLM_DIR = Path(r"F:\ADR_data\wavlm")
BIGVGAN_DIR = Path(r"F:\ADR_data\bigvgan")
WAVLM_DIR.mkdir(parents=True, exist_ok=True)
BIGVGAN_DIR.mkdir(parents=True, exist_ok=True)


def dl_wavlm():
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print("Downloading WavLM-base...")
    path = snapshot_download(
        "microsoft/wavlm-base",
        cache_dir=str(WAVLM_DIR / "_cache"),
        local_dir=str(WAVLM_DIR),
        allow_patterns=["*.json", "*.txt", "pytorch_model.bin"],
    )
    print(f"WavLM OK: {path}")


def dl_bigvgan():
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print("Downloading BigVGAN-base 22kHz...")
    # nvidia/bigvgan_22khz_80band 是常用的 22kHz 80-bin mel 配置
    path = snapshot_download(
        "nvidia/bigvgan_22khz_80band",
        cache_dir=str(BIGVGAN_DIR / "_cache"),
        local_dir=str(BIGVGAN_DIR),
    )
    print(f"BigVGAN OK: {path}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("wavlm", "all"):
        try:
            dl_wavlm()
        except Exception as e:
            print(f"WavLM failed: {e}")
    if which in ("bigvgan", "all"):
        try:
            dl_bigvgan()
        except Exception as e:
            print(f"BigVGAN failed: {e}")
