"""下载 WavLM-base 与 BigVGAN-base 预训练权重到数据目录。

用法: python scripts/download_pretrained.py [wavlm|bigvgan|all]
目录由 adr.core.config.adr_data_dir() 推导 (ADR_DATA_DIR > F:/ADR_data legacy
> %LOCALAPPDATA%\\ADR\\data), 不再硬编码 F: 盘 (批次28 复审 P4);
mkdir 移入 __main__, import 本模块不再有目录副作用。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adr.core.config import adr_data_dir

WAVLM_DIR = adr_data_dir() / "wavlm"
BIGVGAN_DIR = adr_data_dir() / "bigvgan"


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
    WAVLM_DIR.mkdir(parents=True, exist_ok=True)
    BIGVGAN_DIR.mkdir(parents=True, exist_ok=True)
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
