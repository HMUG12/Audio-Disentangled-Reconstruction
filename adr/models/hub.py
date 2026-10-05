"""预训练模型 Hub (下载/缓存/版本管理)。

设计借鉴:
- Hugging Face transformers: snapshot_download
- GPT-SoVITS: tools/download_pretrained.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from adr.core import get_logger
from adr.core.config import adr_data_dir


@dataclass
class PretrainedInfo:
    """预训练模型元信息。"""
    name: str
    category: str          # backbone / vocoder / content_encoder / ...
    url: str               # 下载 URL
    filename: str          # 保存文件名
    sha256: Optional[str] = None
    size_mb: float = 0.0
    description: str = ""
    version: str = "1.0.0"


# ADR 框架支持的预训练模型清单
PRETRAINED_REGISTRY: dict[str, PretrainedInfo] = {
    # BigVGAN 声码器
    "bigvgan_22khz_80band": PretrainedInfo(
        name="bigvgan_22khz_80band",
        category="vocoder",
        url="https://huggingface.co/nvidia/bigvgan_22khz_80band/resolve/main/bigvgan_generator.pt",
        filename="bigvgan_22khz_80band.pt",
        size_mb=420.0,
        description="NVIDIA BigVGAN 22kHz 80-band (default vocoder)",
        version="1.0.0",
    ),
    "bigvgan_24khz_100band": PretrainedInfo(
        name="bigvgan_24khz_100band",
        category="vocoder",
        url="https://huggingface.co/nvidia/bigvgan_24khz_100band_256x/resolve/main/bigvgan_generator.pt",
        filename="bigvgan_24khz_100band.pt",
        size_mb=450.0,
        description="NVIDIA BigVGAN 24kHz 100-band (high quality)",
        version="1.0.0",
    ),
    # WavLM
    "wavlm_base": PretrainedInfo(
        name="wavlm_base",
        category="content_encoder",
        url="https://huggingface.co/microsoft/wavlm-base/resolve/main/pytorch_model.bin",
        filename="wavlm_base.pt",
        size_mb=380.0,
        description="Microsoft WavLM-Base (content features)",
        version="1.0.0",
    ),
}


class PretrainedHub:
    """预训练模型管理。"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/adr/pretrained"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("adr.models.hub")

    def list(self) -> list[PretrainedInfo]:
        """列出所有可用预训练模型。"""
        return list(PRETRAINED_REGISTRY.values())

    def is_downloaded(self, name: str) -> bool:
        """检查模型是否已下载。"""
        if name not in PRETRAINED_REGISTRY:
            return False
        info = PRETRAINED_REGISTRY[name]
        path = self.cache_dir / info.filename
        return path.exists() and path.stat().st_size > 1024  # 至少 1KB

    def get_path(self, name: str) -> Path:
        """获取本地路径 (不存在则报错)。"""
        if name not in PRETRAINED_REGISTRY:
            raise ValueError(f"Unknown pretrained model: {name}")
        info = PRETRAINED_REGISTRY[name]
        path = self.cache_dir / info.filename
        if not path.exists():
            raise FileNotFoundError(
                f"Pretrained not found: {name}\n"
                f"Expected at: {path}\n"
                f"Run: adr model download {name}"
            )
        return path

    def download(
        self,
        name: str,
        force: bool = False,
        progress: bool = True,
    ) -> Path:
        """下载预训练模型。"""
        if name not in PRETRAINED_REGISTRY:
            raise ValueError(
                f"Unknown pretrained: {name}. "
                f"Available: {list(PRETRAINED_REGISTRY.keys())}"
            )

        info = PRETRAINED_REGISTRY[name]
        target = self.cache_dir / info.filename

        if target.exists() and not force:
            self.log.info(f"  ✓ {name} already downloaded at {target}")
            return target

        self.log.info(f"Downloading {name} from {info.url}")
        self.log.info(f"  Size: ~{info.size_mb:.0f} MB")
        self.log.info(f"  To:   {target}")

        # 用 urllib 下载
        import urllib.request

        try:
            with urllib.request.urlopen(info.url, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB

                with open(target, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress and total > 0:
                            pct = downloaded * 100 / total
                            bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
                            print(f"\r  [{bar}] {pct:.1f}% ({downloaded // 1024 // 1024} MB)", end="")
                print()
        except Exception as e:
            if target.exists():
                target.unlink()
            raise RuntimeError(f"Download failed: {e}") from e

        # 验证 SHA256 (如果有)
        if info.sha256:
            sha = hashlib.sha256()
            with open(target, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha.update(chunk)
            actual = sha.hexdigest()
            if actual != info.sha256:
                target.unlink()
                raise RuntimeError(
                    f"SHA256 mismatch for {name}: "
                    f"expected {info.sha256}, got {actual}"
                )

        self.log.info(f"  ✓ Downloaded {name} ({target.stat().st_size // 1024 // 1024} MB)")
        return target

    def search_paths(self) -> list[Path]:
        """本地权重搜索路径 (批次26: 数据目录统一从 adr_data_dir 推导)。"""
        return [
            self.cache_dir,
            adr_data_dir() / "bigvgan",
            adr_data_dir() / "wavlm",
        ]

    def search_local(self) -> list[tuple[str, Path]]:
        """扫描本地已有的预训练权重 (不限于 ADR 注册的)。

        用于发现用户已下载但 ADR 未注册的权重。
        """
        results = []
        for base in self.search_paths():
            if not base.exists():
                continue
            for pt_file in base.rglob("*.pt"):
                if pt_file.stat().st_size > 1024 * 1024:  # > 1MB
                    results.append((pt_file.stem, pt_file))
        return results

    def __repr__(self) -> str:
        return f"PretrainedHub(cache={self.cache_dir})"


def get_hub() -> PretrainedHub:
    """获取全局 Hub (单例)。"""
    if not hasattr(get_hub, "_cache"):
        get_hub._cache = PretrainedHub()
    return get_hub._cache
