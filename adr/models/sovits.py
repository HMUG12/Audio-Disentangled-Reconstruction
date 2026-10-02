"""SoVITS 简化版 (M1 默认 backbone)。

设计目标:
- ~50M 参数 (8GB 显存可训练)
- 非自回归, 推理快
- VITS 风格 (VAE + flow + duration)
- 用 LEGACY 旧模型做参考, 但大幅简化

参考:
- VITS (Conditional Variational Autoencoder with Adversarial Learning)
- GPT-SoVITS (s2 部分)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adr.core import get_logger, register
from adr.models.base import BackboneConfig, BaseBackbone
from adr.models.content_encoder import ContentEncoder
from adr.models.timbre_encoder import TimbreEncoder


@dataclass
class SoVITSConfig(BackboneConfig):
    """SoVITS 简化版配置。"""
    name: str = "sovits"
    # 简化参数
    hidden_dim: int = 256          # 256 而非 1024
    n_layers: int = 6              # 6 而非 28
    n_heads: int = 4               # 4 而非 16
    ffn_dim: int = 1024

    # mel
    n_mels: int = 80
    sample_rate: int = 22050       # BigVGAN 默认 22k
    hop_length: int = 256

    # 内容编码器
    vocab_size: int = 607          # DiffSinger 字典
    content_dim: int = 192

    # 音色编码器
    timbre_dim: int = 128

    # CFM
    cfm_n_timesteps: int = 20
    cfm_sigma_min: float = 1e-5

    # 训练
    use_vae: bool = True           # 简化 VAE


# 统一模型规模预设 (hidden_dim, n_layers, n_heads, content_dim, timbre_dim)
# 训练脚本 / 推理 ckpt 恢复 / WebUI 共用, 避免各处硬编码不一致
SOVITS_PRESETS: dict[str, tuple[int, int, int, int, int]] = {
    "small":  (256, 4, 4, 256, 128),
    "medium": (512, 6, 8, 512, 256),
    "x0p3b":  (1024, 24, 16, 1024, 512),
}


def build_sovits_preset(preset: str, vocab_size: int = 607) -> "SoVITS":
    """按预设名构建 SoVITS (preset ∈ small/medium/x0p3b)。"""
    hd, nl, nh, cd, td = SOVITS_PRESETS[preset]
    cfg = SoVITSConfig(
        hidden_dim=hd, n_layers=nl, n_heads=nh, ffn_dim=hd * 4,
        vocab_size=vocab_size, content_dim=cd, timbre_dim=td,
        n_mels=80, sample_rate=22050, hop_length=256,
    )
    return SoVITS(cfg)


class DurationPredictor(nn.Module):
    """时长预测器 (简化版)。"""

    def __init__(self, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        # 用 Conv1d 后接 LayerNorm; 每次 block 转置到 (B, T, D) 再 LN
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T_phoneme, D)
            mask: (B, T_phoneme) bool
        Returns:
            (B, T_phoneme) log duration
        """
        h = x.transpose(1, 2)  # (B, D, T)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h)
            h = h.transpose(1, 2)  # (B, T, D)
            h = norm(h)
            h = h.transpose(1, 2)  # 转回 (B, D, T) 继续 conv
            h = F.gelu(h)
        h = h.transpose(1, 2)  # (B, T, D)
        log_dur = self.proj(h).squeeze(-1)  # (B, T)

        if mask is not None:
            log_dur = log_dur.masked_fill(~mask, 0.0)
        return log_dur


class SoVITS(BaseBackbone):
    """SoVITS 简化版 (~50M 参数)。

    架构:
        1. ContentEncoder: 音素 → 上下文嵌入
        2. TimbreEncoder: 参考 mel → 音色向量
        3. LengthRegulator: 音素 → mel 帧
        4. Decoder: mel 帧 → mel 谱 (简化 CFM)
    """

    config_class = SoVITSConfig

    # M8.6: 能力声明 — TTS+SVS 双模, 接受 F0 条件, 大矩阵全 Linear/MHA 可 LoRA
    from adr.models.base import BackboneCapability as _Cap
    capabilities = _Cap.TTS | _Cap.SVS | _Cap.F0_COND | _Cap.LORA_READY

    def __init__(self, config: Optional[SoVITSConfig] = None):
        super().__init__(config)
        config = self.config  # type: SoVITSConfig

        # 内容编码器
        self.content_encoder = ContentEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.content_dim,
            n_layers=4,
            n_heads=4,
            ffn_dim=1024,
            max_len=1024,
        )

        # 音色编码器
        self.timbre_encoder = TimbreEncoder(
            n_mels=config.n_mels,
            embed_dim=config.timbre_dim,
        )

        # 融合层
        self.fuse = nn.Linear(config.content_dim + config.timbre_dim, config.hidden_dim)

        # F0 注入 (SVS 旋律条件)
        # log1p(f0/700) → hidden_dim, 零初始化:
        # - 旧 checkpoint (无 f0_embed 权重) strict=False 加载后行为与之前完全一致
        # - f0=0 (无声/未提供) 时贡献为 0
        self.f0_embed = nn.Linear(1, config.hidden_dim)
        nn.init.zeros_(self.f0_embed.weight)
        nn.init.zeros_(self.f0_embed.bias)

        # 时长预测器
        self.duration_predictor = DurationPredictor(config.hidden_dim)

        # Decoder (简化 DiT-like)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=config.n_layers,
            enable_nested_tensor=False,  # norm_first=True 与 nested tensor 不兼容
        )

        # 输出投影
        self.out_proj = nn.Linear(config.hidden_dim, config.n_mels)

        self.log.info(
            f"SoVITS initialized: hidden={config.hidden_dim}, "
            f"layers={config.n_layers}, "
            f"vocab={config.vocab_size}"
        )

    def encode_content(
        self,
        phoneme_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """内容编码。"""
        return self.content_encoder(phoneme_ids, mask)

    def encode_timbre(self, ref_mel: torch.Tensor) -> torch.Tensor:
        """音色编码。"""
        return self.timbre_encoder(ref_mel)

    def length_regulate(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """根据预测时长重复 token。

        Args:
            x: (B, T_phoneme, D)
            durations: (B, T_phoneme) int, 每帧重复次数
            max_len: 输出最大长度
        """
        B, T, D = x.shape
        if max_len is None:
            max_len = int(durations.sum(dim=1).max().item())

        output = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
        for b in range(B):
            pos = 0
            for t in range(T):
                d = int(durations[b, t].item())
                if d <= 0:
                    continue
                end = min(pos + d, max_len)
                output[b, pos:end] = x[b, t:t+1].expand(end - pos, -1)
                pos = end
                if pos >= max_len:
                    break
        return output

    def _inject_f0(
        self,
        regulated: torch.Tensor,
        f0: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """把 F0 注入 length-regulated 序列 (mel 帧级)。

        Args:
            regulated: (B, T_mel, D)
            f0: (B, T_f0) Hz, 0 = 无声; None 时原样返回
        """
        if f0 is None:
            return regulated
        T = regulated.size(1)
        f0 = f0[:, :T]
        if f0.size(1) < T:
            f0 = F.pad(f0, (0, T - f0.size(1)))
        f0_feat = torch.log1p(f0.float() / 700.0).unsqueeze(-1)  # (B, T, 1)
        return regulated + self.f0_embed(f0_feat.to(regulated.dtype))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """训练 forward。

        Args:
            batch: {
                'phoneme_ids': (B, T_phoneme),
                'phoneme_mask': (B, T_phoneme) bool,
                'ref_mel': (B, n_mels, T_ref),
                'target_mel': (B, n_mels, T_mel),
                'target_durations': (B, T_phoneme) (可选, ground truth),
                'f0': (B, T_mel) (可选, Hz, 0 = 无声)
            }
        """
        phoneme_ids = batch["phoneme_ids"]
        phoneme_mask = batch.get("phoneme_mask")
        ref_mel = batch["ref_mel"]
        target_mel = batch.get("target_mel")

        # 编码
        content = self.encode_content(phoneme_ids, phoneme_mask)  # (B, T_p, D_c)
        timbre = self.encode_timbre(ref_mel)  # (B, D_t)
        timbre_expanded = timbre.unsqueeze(1).expand(-1, content.size(1), -1)

        # 融合
        fused = self.fuse(torch.cat([content, timbre_expanded], dim=-1))

        # 时长
        log_dur = self.duration_predictor(fused, phoneme_mask)

        # 长度规整
        if "target_durations" in batch:
            # 训练: 用 GT duration
            durations = batch["target_durations"]
        else:
            # 推理: 用 exp(log_dur)
            durations = torch.exp(log_dur).round().long()

        regulated = self.length_regulate(fused, durations)

        # F0 注入 (训练时 teacher forcing 用 GT f0)
        regulated = self._inject_f0(regulated, batch.get("f0"))

        # Decoder
        decoded = self.decoder(regulated)
        pred_mel = self.out_proj(decoded).transpose(1, 2)  # (B, n_mels, T_mel)

        # 损失
        loss = torch.tensor(0.0, device=pred_mel.device)
        if target_mel is not None:
            # 截断到相同长度
            min_T = min(pred_mel.size(-1), target_mel.size(-1))
            mel_loss = F.l1_loss(pred_mel[..., :min_T], target_mel[..., :min_T])
            loss = loss + mel_loss

        # 时长损失 (训练时)
        if "target_durations" in batch:
            log_target = torch.log(batch["target_durations"].float() + 1.0)
            if phoneme_mask is not None:
                log_dur_masked = log_dur[phoneme_mask]
                log_target_masked = log_target[phoneme_mask]
                dur_loss = F.mse_loss(log_dur_masked, log_target_masked)
            else:
                dur_loss = F.mse_loss(log_dur, log_target)
            loss = loss + dur_loss

        return {
            "loss": loss,
            "pred_mel": pred_mel,
            "log_duration": log_dur,
        }

    @torch.inference_mode()
    def sample(
        self,
        phoneme_ids: torch.Tensor,
        ref_mel: torch.Tensor,
        n_timesteps: int = 20,
        min_mel_frames: int = 80,   # ~0.93s @ 22kHz/256
        f0: Optional[torch.Tensor] = None,  # (B, T_mel) Hz, SVS 旋律条件
        **kwargs,
    ) -> torch.Tensor:
        """推理: 音素 + ref_mel → mel。

        Args:
            f0: 可选旋律 F0 (Hz, mel 帧分辨率); None = 无 F0 条件 (TTS 场景)

        Returns:
            (B, n_mels, T_mel)
        """
        B = phoneme_ids.size(0)
        device = phoneme_ids.device

        # DurationPredictor conv kernel=3, 音素序列至少 3 帧 (短文本兜底)
        if phoneme_ids.size(1) < 3:
            phoneme_ids = F.pad(phoneme_ids, (0, 3 - phoneme_ids.size(1)), value=0)

        # 编码
        content = self.encode_content(phoneme_ids)
        timbre = self.encode_timbre(ref_mel)
        timbre_expanded = timbre.unsqueeze(1).expand(-1, content.size(1), -1)
        fused = self.fuse(torch.cat([content, timbre_expanded], dim=-1))

        # 预测时长
        log_dur = self.duration_predictor(fused)
        durations = torch.exp(log_dur).round().long()
        durations = torch.clamp(durations, min=1)

        # 长度规整
        regulated = self.length_regulate(fused, durations)

        # 兜底: 未训练好时 duration 偏小,强制最低帧数 (避免输出 0.01s 噪声)
        if regulated.size(1) < min_mel_frames:
            n_repeat = (min_mel_frames // max(1, regulated.size(1))) + 1
            regulated = regulated.repeat(1, n_repeat, 1)[:, :min_mel_frames]

        # F0 注入 (推理时由外部旋律提供; None = 无条件)
        regulated = self._inject_f0(regulated, f0)

        # Decoder
        decoded = self.decoder(regulated)
        mel = self.out_proj(decoded).transpose(1, 2)

        return mel


# 注册到 REGISTRY
register("backbone", "sovits")(SoVITS)
register("backbone", "sovits_base")(SoVITS)
