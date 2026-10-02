"""下载 OpenCpop 数据集（HF 镜像 Saaaxman/Opencpop，102 downloads）。"""
import os
import sys
from pathlib import Path

TARGET = Path(r"F:\ADR_data\opencpop")
CACHE = TARGET / "_cache"
TARGET.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import snapshot_download
    print("Downloading Saaaxman/Opencpop (HF mirror)...")
    try:
        path = snapshot_download(
            "Saaaxman/Opencpop",
            repo_type="dataset",
            cache_dir=str(CACHE),
            local_dir=str(TARGET),
        )
        print(f"OK: {path}")
    except Exception as e:
        print(f"Saaaxman failed: {e}")
        print("Trying espnet/ace-opencpop-segments...")
        path = snapshot_download(
            "espnet/ace-opencpop-segments",
            repo_type="dataset",
            cache_dir=str(CACHE),
            local_dir=str(TARGET / "_espnet_fallback"),
        )
        print(f"espnet OK: {path}")


if __name__ == "__main__":
    main()
