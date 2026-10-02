"""推理流水线: text + ref.wav → wav。

设计:
- 自动 G2P (默认 pypinyin)
- 自动从 ref 提取 mel
- backbone.sample → mel
- vocoder.infer(mel) → wav
- 支持 GPU/CPU 自动切换
- vocoder 加载失败时 fallback 到 mel 第一个 channel (placeholder)

典型用法:
    >>> from adr.inference import InferPipeline
    >>> pipe = InferPipeline.from_checkpoint("output/train/checkpoints/best.pt")
    >>> wav = pipe.synthesize("你好世界", "ref.wav")
    >>> pipe.save_wav(wav, "out.wav")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from adr.core import get_logger
from adr.data.g2p import G2P, G2PConfig
from adr.utils.audio import AudioData, load_audio, save_audio


@dataclass
class InferConfig:
    """推理配置。"""
    # 文本
    g2p_backend: str = "pypinyin"
    g2p_with_tone: bool = True

    # 音频
    ref_sample_rate: int = 22050      # ref 加载时 resample 到此 sr
    n_mels: int = 80
    hop_length: int = 256

    # 推理
    n_timesteps: int = 20
    device: str = "auto"              # auto / cpu / cuda
    output_rms: float = 0.25          # 输出响度归一化 (0=关闭; mel_v2 域输出偏小)

    # 路径
    vocoder_path: Optional[str] = None  # BigVGAN 权重目录,None = 不加载
    use_placeholder_vocoder: bool = False  # vocoder 失败时是否退化 (默认 False)


class InferPipeline:
    """端到端推理流水线。

    主要入口:
        - synthesize(text, ref_audio) -> waveform
        - save_wav(waveform, path)
        - text_to_phoneme_ids(text) -> List[int]
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        config: Optional[InferConfig] = None,
        vocoder: Optional[object] = None,
    ):
        self.config = config or InferConfig()
        self.log = get_logger("adr.inference")

        # Device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Backbone
        self.backbone = backbone.to(self.device)
        self.backbone.eval()
        self.log.info(f"Backbone on {self.device}: "
                     f"{sum(p.numel() for p in self.backbone.parameters())/1e6:.1f}M params")

        # G2P
        self.g2p = G2P(G2PConfig(
            backend=self.config.g2p_backend,
            with_tone=self.config.g2p_with_tone,
        ))

        # Vocoder (optional)
        self.vocoder = vocoder
        if self.vocoder is not None:
            try:
                self.vocoder.to(self.device)
                self.vocoder.eval()
                self.log.info("Vocoder loaded")
            except Exception as e:
                self.log.warning(f"Vocoder move-to-device failed: {e}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        backbone_cls: Optional[type] = None,
        backbone_config: Optional[object] = None,
        config: Optional[InferConfig] = None,
        vocoder: Optional[object] = None,
        vocoder_path: Optional[str] = None,
    ) -> "InferPipeline":
        """从 checkpoint 构造。

        Args:
            checkpoint_path: .pt 路径 (Trainer.save_checkpoint 格式)
            backbone_cls: backbone 类 (None 时从 ckpt['model_class'] 推断)
            backbone_config: backbone config (None 时从 ckpt['backbone_config'] 读取)
            config: 推理配置
            vocoder: 已加载的 vocoder (优先)
            vocoder_path: BigVGAN 权重目录 (vocoder=None 时自动加载)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # 推断 backbone 类型 (从 ckpt['model_class'] 优先)
        if backbone_cls is None:
            model_class_name = ckpt.get("model_class", "SoVITS")
            # 注册表查找 (确保全部内建 backbone 已注册)
            import adr.models.adr2  # noqa: F401
            from adr.core import REGISTRY
            backbone_cls = None
            for name, cls_ in REGISTRY.backbone.items():
                if cls_.__name__ == model_class_name:
                    backbone_cls = cls_
                    break
            if backbone_cls is None:
                # fallback: 用 SoVITS
                from adr.models.sovits import SoVITS
                backbone_cls = SoVITS

        # 推断 backbone config (优先级: ckpt['backbone_config'] > LoRA preset > 默认)
        lora_cfg_saved = ckpt.get("lora_config") or {}
        if backbone_config is None:
            saved_bb_cfg = ckpt.get("backbone_config", None)
            if saved_bb_cfg is not None:
                try:
                    backbone_config = backbone_cls.config_class(**saved_bb_cfg)
                except Exception as e:
                    get_logger("adr.inference").warning(
                        f"Failed to reconstruct backbone config from ckpt: {e}"
                    )
            if backbone_config is None and lora_cfg_saved.get("preset"):
                # LoRA-only ckpt 可能没带 backbone_config, 但 demo 会记 preset
                from adr.models.sovits import SOVITS_PRESETS, SoVITSConfig
                preset = lora_cfg_saved["preset"]
                if preset in SOVITS_PRESETS and backbone_cls.__name__ == "SoVITS":
                    hd, nl, nh, cd, td = SOVITS_PRESETS[preset]
                    backbone_config = SoVITSConfig(
                        hidden_dim=hd, n_layers=nl, n_heads=nh, ffn_dim=hd * 4,
                        content_dim=cd, timbre_dim=td,
                    )
            if backbone_config is None:
                backbone_config = backbone_cls.config_class()

        model = backbone_cls(backbone_config)

        # LoRA-only checkpoint: 先注入 LoRA 结构再加载 LoRA 权重
        is_lora_ckpt = bool(ckpt.get("use_lora")) or ("lora_state" in ckpt)
        if is_lora_ckpt:
            from adr.training.lora import (
                LoRAConfig, apply_lora, load_lora_state_dict,
            )
            trainer_cfg = ckpt.get("config")
            rank = lora_cfg_saved.get("rank", getattr(trainer_cfg, "lora_rank", 8))
            alpha = lora_cfg_saved.get("alpha", getattr(trainer_cfg, "lora_alpha", 16))
            targets = lora_cfg_saved.get(
                "target_modules",
                getattr(trainer_cfg, "lora_target_modules",
                        ("out_proj", "linear1", "linear2")),
            )
            apply_lora(model, LoRAConfig(rank=rank, alpha=alpha,
                                         target_modules=list(targets)))
            lora_sd = ckpt.get("lora_state") or ckpt["model_state"]
            loaded, missing = load_lora_state_dict(model, lora_sd)
            get_logger("adr.inference").info(
                f"LoRA checkpoint loaded: {loaded} params "
                f"(rank={rank}, alpha={alpha})"
                + (f", missing={len(missing)}" if missing else "")
            )
        else:
            try:
                model.load_state_dict(ckpt["model_state"], strict=True)
            except Exception as e:
                get_logger("adr.inference").warning(
                    f"Strict load failed: {e}, retrying with strict=False"
                )
                model.load_state_dict(ckpt["model_state"], strict=False)

        # 加载 vocoder (优先用传入的,其次按路径加载,最后自动找本地 BigVGAN)
        if vocoder is None:
            if vocoder_path is None:
                try:
                    from adr.vocoder.bigvgan import _find_bigvgan_dir
                    found = _find_bigvgan_dir()
                    vocoder_path = str(found) if found else None
                except Exception:
                    vocoder_path = None
            if vocoder_path is not None:
                vocoder = cls._load_vocoder(vocoder_path)

        return cls(model, config=config, vocoder=vocoder)

    @staticmethod
    def _load_vocoder(vocoder_path: str) -> Optional[object]:
        """从路径加载 vocoder (默认尝试 BigVGAN)。"""
        from adr.core import REGISTRY
        log = get_logger("adr.inference")

        # 优先 BigVGAN
        bigvgan_cls = REGISTRY.vocoder.get("bigvgan")
        if bigvgan_cls is not None:
            try:
                vocoder = bigvgan_cls.from_pretrained(vocoder_path, device="auto")
                if vocoder.is_loaded():
                    log.info(f"Vocoder loaded: bigvgan from {vocoder_path}")
                    return vocoder
            except Exception as e:
                log.warning(f"BigVGAN load failed: {e}")

        log.warning("No vocoder loaded — will use placeholder (mel→wav is fake)")
        return None

    def text_to_phoneme_ids(self, text: str) -> list[int]:
        """文本 → 音素 id 列表 (用真实 PhonemeDict,DiffSinger opencpop 607)。

        流程:
        1. 检测语言 (中文 / 英文 / 数字)
        2. 调 G2P 拿音素字符串
        3. 用 PhonemeDict (DiffSinger opencpop 607) 编码

        自动 fallback:
        - 空文本 → 报错
        - G2P 失败 → 用 char 模式
        - OOV 音素 → 映射到 UNK=3
        """
        from adr.data.phoneme_dict import encode_phonemes, load_default_phoneme_dict

        if not text or not text.strip():
            raise ValueError(f"Empty text")

        # 用真实音素字典
        pd = load_default_phoneme_dict()

        # 优先用 PhonemeDict.encode_phonemes (内置 G2P)
        try:
            ids = encode_phonemes(text, phoneme_dict=pd, strip_tone=True)
            if ids and not all(i == pd.encode("<unk>") for i in ids):
                return ids
        except Exception:
            pass

        # Fallback: char-level (英文/数字/pypinyin 失败)
        from adr.data.g2p import G2P, G2PConfig
        char_g2p = G2P(G2PConfig(backend="char"))
        char_phonemes = char_g2p(text)
        if not char_phonemes:
            raise ValueError(f"Could not extract phonemes: '{text}'")

        # 字符级也用字典 encode,OOV 自动 UNK
        return [pd.encode(c) for c in char_phonemes]

    def ref_audio_to_mel(self, ref_audio: Union[str, Path, AudioData]) -> np.ndarray:
        """ref 音频 → log-mel 频谱 (n_mels, T)。"""
        if isinstance(ref_audio, (str, Path)):
            audio = load_audio(
                str(ref_audio),
                sample_rate=self.config.ref_sample_rate,
                mono=True,
            )
        else:
            audio = ref_audio

        from adr.utils.audio import compute_mel_bigvgan
        mel = compute_mel_bigvgan(
            audio.waveform,
            sample_rate=self.config.ref_sample_rate,
            n_mels=self.config.n_mels,
            hop_length=self.config.hop_length,
        )
        self.log.info(f"ref mel shape: {mel.shape}, mean={mel.mean():.2f}")
        return mel

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        ref_audio: Union[str, Path, AudioData],
        n_timesteps: Optional[int] = None,
        f0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """text + ref → wav。

        Args:
            text: 输入文本 (中文/英文)
            ref_audio: 参考音频路径或 AudioData
            n_timesteps: 扩散步数 (None 用 config 默认)
            f0: 可选旋律 F0 (Hz, mel 帧分辨率, SVS 场景)

        Returns:
            (T,) wav 数组, sample_rate = self.config.ref_sample_rate
        """
        n_timesteps = n_timesteps or self.config.n_timesteps

        # 1. 文本 → 音素 ids
        phoneme_ids = self.text_to_phoneme_ids(text)
        phoneme_ids_t = torch.tensor([phoneme_ids], dtype=torch.long, device=self.device)
        phoneme_mask = torch.ones_like(phoneme_ids_t, dtype=torch.bool)

        # 2. ref → mel
        ref_mel_np = self.ref_audio_to_mel(ref_audio)
        ref_mel_t = torch.from_numpy(ref_mel_np).unsqueeze(0).float().to(self.device)

        # 2.5 F0 (SVS 旋律): 显式传入 > backbone 声明 F0_COND 时自动从 ref 提取
        f0_t = None
        if f0 is not None:
            f0_t = torch.from_numpy(np.asarray(f0, dtype=np.float32)).unsqueeze(0).to(self.device)
        else:
            try:
                from adr.models.base import BackboneCapability
                if self.backbone.supports(BackboneCapability.F0_COND):
                    from adr.data.f0 import F0Extractor
                    src = ref_audio if isinstance(ref_audio, (str, Path)) \
                        else ref_audio.waveform
                    f0_np = F0Extractor()(src, self.config.ref_sample_rate)
                    f0_t = torch.from_numpy(
                        np.asarray(f0_np, dtype=np.float32)).unsqueeze(0).to(self.device)
                    self.log.info(f"auto F0 from ref: {f0_np.shape[0]} frames, "
                                  f"voiced={100 * (f0_np > 0).mean():.0f}%")
            except Exception as e:
                self.log.warning(f"auto F0 extraction failed ({e}), 退化为无条件")

        # 3. backbone 推理
        if hasattr(self.backbone, "sample"):
            mel = self.backbone.sample(
                phoneme_ids_t, ref_mel_t, n_timesteps=n_timesteps, f0=f0_t,
            )
        elif hasattr(self.backbone, "forward"):
            mel = self.backbone(phoneme_ids_t, phoneme_mask, ref_mel_t).get("pred_mel", None)
        else:
            raise RuntimeError("Backbone has neither 'sample' nor 'forward'")

        if mel is None:
            raise RuntimeError("Backbone returned None")

        # 4. mel → wav
        if self.vocoder is not None and not self.config.use_placeholder_vocoder:
            try:
                wav_t = self.vocoder.infer(mel)
            except Exception as e:
                self.log.warning(f"Vocoder failed ({e}), using placeholder")
                wav_t = self._placeholder_vocoder(mel)
        else:
            wav_t = self._placeholder_vocoder(mel)

        wav = wav_t.squeeze().cpu().numpy().astype(np.float32)
        # 响度归一化 (mel_v2 域 BigVGAN 输出 RMS 偏小, 人耳验收发现)
        if self.config.output_rms > 0:
            rms = float(np.sqrt((wav ** 2).mean()))
            if rms > 1e-6:
                gain = min(self.config.output_rms / rms, 10.0)  # 限幅防爆音
                wav = (wav * gain).astype(np.float32)
                peak = float(np.abs(wav).max())
                if peak > 0.98:
                    wav = (wav * (0.98 / peak)).astype(np.float32)
        return wav

    def _placeholder_vocoder(self, mel: torch.Tensor) -> torch.Tensor:
        """vocoder 失败时的占位实现: mel[:, 0, :] 当 wav 用。

        注: 这里返回的"wav"实际是mel[0] channel(不真实),只是占位让 pipeline 跑通。
        真实 wav 需要 BigVGAN 等声码器。
        """
        # 反 log-mel 回到线性 mel
        linear = torch.exp(mel)  # (B, n_mels, T)
        # 用 Griffin-Lim 做一个粗略 wav (也用 mel 第一个 channel 当 fallback)
        wav = linear[:, 0, :]  # (B, T)
        # 归一化到 [-1, 1]
        wav = wav / (wav.abs().max() + 1e-8)
        return wav

    def save_wav(
        self,
        waveform: np.ndarray,
        path: Union[str, Path],
        sample_rate: Optional[int] = None,
    ) -> None:
        """保存 wav 到文件。"""
        sr = sample_rate or self.config.ref_sample_rate
        audio = AudioData(waveform=waveform, sample_rate=sr)
        save_audio(audio, str(path))
        self.log.info(f"Saved wav: {path} (sr={sr}, duration={len(waveform)/sr:.2f}s)")
