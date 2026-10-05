"""BigVGAN v2 声码器 (M1 默认)。

特性:
- 加载 NVIDIA 官方 BigVGAN 权重 (22kHz / 24kHz)
- 自动 resample 到目标采样率
- 注册到 VOCODER_REGISTRY
- 替换 placeholder 让 pipeline 输出真实可听 wav

参考:
- https://github.com/NVIDIA/BigVGAN
- 权重: F:\\ADR_data\\bigvgan\\bigvgan_generator.pt
- 源码: F:\\ADR_data\\bigvgan\\ (含 bigvgan.py, activations.py, utils.py, env.py, alias_free_activation/)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from adr.core import get_logger, register
from adr.core.config import adr_data_dir
from adr.vocoder.base import BaseVocoder, VocoderConfig


# BigVGAN 源码搜索路径 (批次26: 数据目录统一从 adr_data_dir 推导)
BIGVGAN_DIR_CANDIDATES = [
    adr_data_dir() / "bigvgan",
    Path(__file__).parent.parent.parent / "third_party" / "BigVGAN",
    Path.home() / "BigVGAN",
]


def _find_bigvgan_dir() -> Optional[Path]:
    """搜索 BigVGAN 源码目录。"""
    for cand in BIGVGAN_DIR_CANDIDATES:
        if cand.exists() and (cand / "bigvgan.py").exists():
            return cand
    return None


def _import_bigvgan_module(bigvgan_dir: Path) -> Optional[object]:
    """动态 import BigVGAN 源码 (含 env / activations / alias_free_activation)。"""
    bigvgan_dir = Path(bigvgan_dir).resolve()
    if str(bigvgan_dir) not in sys.path:
        sys.path.insert(0, str(bigvgan_dir))

    try:
        # 先 import env / activations / alias_free_activation (因为 bigvgan.py 依赖它们)
        import env  # noqa: F401
        import activations  # noqa: F401
        import utils  # noqa: F401
        from alias_free_activation.torch import act  # noqa: F401

        # 再 import BigVGAN 主类
        from bigvgan import BigVGAN
        return BigVGAN
    except Exception as e:
        get_logger("adr.vocoder.BigVGANVocoder").error(
            f"Failed to import BigVGAN from {bigvgan_dir}: {e}"
        )
        return None


@register("vocoder", "bigvgan")
class BigVGANVocoder(BaseVocoder):
    """BigVGAN v2 声码器封装 (M1)。

    用法:
        vocoder = BigVGANVocoder.from_pretrained("<BigVGAN 权重目录>")
        wav = vocoder.infer(mel)  # (B, T_wav) 真实音频
    """

    config_class = VocoderConfig

    def __init__(self, config: Optional[VocoderConfig] = None):
        super().__init__(config)
        self.log = get_logger("adr.vocoder.BigVGANVocoder")
        self._model: Optional[nn.Module] = None
        self._h = None  # AttrDict 配置
        self._native_sr: int = self.config.sample_rate
        self._BigVGAN = None  # BigVGAN class

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        config_path: Optional[str | Path] = None,
        target_sr: Optional[int] = None,
        device: str = "auto",
    ) -> "BigVGANVocoder":
        """从本地权重加载。

        Args:
            pretrained_path: bigvgan_generator.pt 路径或 bigvgan 目录
            config_path: config.json 路径 (可选,默认与权重同目录)
            target_sr: 目标采样率 (None 保持原始)
            device: 'cuda' / 'cpu' / 'auto'
        """
        pretrained_path = Path(pretrained_path)

        if pretrained_path.is_dir():
            pt_file = pretrained_path / "bigvgan_generator.pt"
            cfg_file = pretrained_path / "config.json"
            bigvgan_dir = pretrained_path
        else:
            pt_file = pretrained_path
            cfg_file = pretrained_path.parent / "config.json"
            bigvgan_dir = pretrained_path.parent

        if not pt_file.exists():
            raise FileNotFoundError(f"BigVGAN weights not found: {pt_file}")
        if not cfg_file.exists():
            raise FileNotFoundError(f"BigVGAN config not found: {cfg_file}")

        # 读 config
        with open(cfg_file, encoding="utf-8") as f:
            cfg_dict = json.load(f)

        native_sr = cfg_dict.get("sampling_rate", 22050)
        log = get_logger("adr.vocoder.BigVGANVocoder")
        log.info(f"Loading BigVGAN from: {bigvgan_dir}")
        log.info(f"  native_sr={native_sr}, num_mels={cfg_dict.get('num_mels', 80)}")

        # 动态 import BigVGAN 源码
        # 优先用 bigvgan_dir,失败时回退到搜索路径
        if not (bigvgan_dir / "bigvgan.py").exists():
            found = _find_bigvgan_dir()
            if found is None:
                raise FileNotFoundError(
                    f"BigVGAN source code (bigvgan.py) not found. "
                    f"Searched: {[str(c) for c in BIGVGAN_DIR_CANDIDATES]}"
                )
            bigvgan_dir = found
        BigVGAN = _import_bigvgan_module(bigvgan_dir)
        if BigVGAN is None:
            raise RuntimeError("Failed to import BigVGAN source code")

        # 构造 config (注入到 instance)
        config = VocoderConfig(
            name="bigvgan",
            sample_rate=target_sr or native_sr,
            n_mels=cfg_dict.get("num_mels", 80),
            hop_length=cfg_dict.get("hop_size", 256),
            pretrained_path=str(pt_file),
        )

        instance = cls(config)
        instance._BigVGAN = BigVGAN
        instance._native_sr = native_sr
        instance._load_model(pt_file, cfg_dict, device)
        return instance

    def _load_model(self, pt_file: Path, cfg_dict: dict, device: str) -> None:
        """加载 BigVGAN 内部模型。"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        from env import AttrDict
        h = AttrDict(cfg_dict)
        h["use_cuda_kernel"] = False  # 默认 torch 版本,无需 CUDA 编译

        self._h = h
        self._model = self._BigVGAN(h, use_cuda_kernel=False).to(device)

        # 加载权重 — 处理 3 种 ckpt 格式
        ckpt = torch.load(pt_file, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "generator" in ckpt and isinstance(ckpt["generator"], dict):
            # 格式 1: {"generator": OrderedDict}
            state_dict = ckpt["generator"]
            fmt = "dict-with-generator-key"
        elif isinstance(ckpt, dict) and any(k.startswith("generator.") for k in ckpt.keys()):
            # 格式 2: {"generator.xxx": tensor, ...}
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in ckpt.items()
                if k.startswith("generator.")
            }
            fmt = "generator-prefixed"
        else:
            # 格式 3: 裸 state_dict
            state_dict = ckpt
            fmt = "bare-state-dict"

        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            self.log.warning(
                f"BigVGAN load: format={fmt}, missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )
        else:
            self.log.info(f"BigVGAN loaded: format={fmt}, all {len(state_dict)} keys matched")

        self._model.eval()
        self.log.info(
            f"  BigVGAN ready: sr={self._native_sr}, "
            f"mels={self.config.n_mels}, device={device}, "
            f"params={sum(p.numel() for p in self._model.parameters())/1e6:.1f}M"
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel → wav。

        Args:
            mel: (B, n_mels, T_mel) log-mel
        Returns:
            (B, 1, T_wav) wav in [-1, 1]
        """
        if self._model is None:
            self.log.warning("Vocoder not loaded, returning identity (placeholder)")
            # 退化: mel → wav 长度按 hop_size 估算
            return mel[:, 0:1, :] * self.config.hop_length

        return self._model(mel)

    @torch.inference_mode()
    def infer(self, mel: torch.Tensor) -> torch.Tensor:
        """推理 + 自动 resample 到目标 SR。"""
        wav = super().infer(mel)
        # BigVGAN 输出 (B, 1, T_wav),转为 (B, T_wav)
        if wav.ndim == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)

        # Resample 如果需要
        if self.config.sample_rate != self._native_sr:
            import torchaudio
            wav = torchaudio.functional.resample(
                wav, self._native_sr, self.config.sample_rate
            )
        # 限制到 [-1, 1]
        wav = torch.clamp(wav, -1.0, 1.0)
        return wav

    def is_loaded(self) -> bool:
        """是否成功加载了真实模型。"""
        return self._model is not None
