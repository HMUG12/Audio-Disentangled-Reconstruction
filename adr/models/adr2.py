"""M9.2: ADR-2 声学模型骨架 (VAE 后验 + flow 先验 + MAS 对齐, 前向可训)。

与一期 SoVITS 的本质区别:
- 一期: 均摊时长 + 单次前向 mel 回归 (对齐错、表达弱)
- ADR-2: MAS 自动对齐 + VAE 后验 q(z|mel) + flow 增强先验 p(z|text)
  → 对齐/表达/概率建模三者都有, 且全部可微端到端

训练目标 (VITS-lite):
  L = L_recon(mel) + λ_kl * L_KL(flow 空间) + L_dur(MAS 时长)

推理:
  文本 → 先验 (mu_p, std_p) → 时长上采样 → flow^{-1}(eps) → mel_head → mel

后续里程碑:
- M9.3: 确定性 DurationPredictor → 随机时长 (flow-based)
- M9.4: mode embedding 双模 (TTS 韵律头 / SVS 旋律条件)
- M9.5: mel_head 占位 → FlowDecoder + 流式推理
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adr.core import get_logger, register
from adr.models.alignment import (
    alignment_from_durations,
    durations_from_alignment,
    gaussian_logp,
    maximum_path,
)
from adr.models.base import BackboneCapability, BackboneConfig, BaseBackbone
from adr.models.content_encoder import ContentEncoder
from adr.models.sovits import DurationPredictor
from adr.models.timbre_encoder import TimbreEncoder
from adr.models.vae import FlowDecoder, PosteriorEncoder, ResidualCouplingFlow


@dataclass
class ADR2Config(BackboneConfig):
    """ADR-2 声学模型配置。"""
    name: str = "adr2"
    vocab_size: int = 607
    n_mels: int = 80
    content_dim: int = 256
    timbre_dim: int = 128
    hidden_dim: int = 256
    z_dim: int = 64               # 潜变量维度 (MAS/VAE 空间)
    posterior_hidden: int = 128
    posterior_layers: int = 6
    n_flow: int = 4               # 先验 flow 耦合层数
    flow_hidden: int = 128
    kl_weight: float = 1.0
    kl_warmup_steps: int = 1      # KL warmup (发散保护)。注意:  warmup 期间
                                  # KL 权重为 0, 后验 z 脱离先验自由膨胀,
                                  # warmup 结束后 MAS 在失控尺度上必崩溃
                                  # → 固定先验 std=1 后不再需要长 warmup
    mas_warmup_steps: int = 500   # MAS 热身: 前 N 步用均摊对齐, 打破零开局对称性
    dur_weight: float = 1.0       # Glow-TTS 惯例: 时长 loss 与 NLL 同量级
    # M9.3 随机时长推理参数
    dur_noise_scale: float = 0.5  # 时长采样噪声 (0=确定性; 长尾会爆时长, 别贪大)
    dur_min_frames: int = 2       # 每音素最少帧数 (防 1 帧糊点)
    dur_max_frames: int = 60      # 每音素最多帧数 (防高斯长尾炸出 1762 帧)
    dur_scale: float = 1.0        # 全局语速缩放 (>1 更慢)
    # M9.5 FlowDecoder (mel 空间条件流, 精确 NLL 训练)
    flowdec_hidden: int = 192
    flowdec_layers: int = 5
    temperature: float = 0.7      # 推理采样温度
    # M9.4 双模 + F0 旋律条件
    use_f0: bool = True           # SVS: F0 注入 FlowDecoder 条件
    f0_dim: int = 32
    default_mode: int = 1         # 0=TTS (无旋律), 1=SVS (歌声, 用 F0)
    sample_rate: int = 22050
    hop_length: int = 256


def _gaussian_log_pdf(x: torch.Tensor, mu: torch.Tensor,
                      log_std: torch.Tensor) -> torch.Tensor:
    """逐帧高斯对数概率 log N(x; mu, std), (B,C,T) → (B,T)。"""
    nll = math.log(2 * math.pi) + 2 * log_std \
        + ((x - mu) ** 2) * torch.exp(-2 * log_std)
    return -0.5 * nll.sum(dim=1)


class StochasticDurationHead(nn.Module):
    """M9.3: 随机时长预测头 (VITS SDP 的高斯简化版)。

    训练: NLL( log(mas_dur) ; mu_d, std_d ) — 学习时长分布而非点估计
    推理: log_dur ~ N(mu_d, (std_d * noise_scale)^2) → exp → round → clamp
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, 2)  # mu_d, log_std_d

    def forward(self, h: torch.Tensor):
        """h (B,N,H) → mu_d, log_std_d (B,N)。"""
        mu, log_std = self.proj(self.net(h)).chunk(2, dim=-1)
        return mu.squeeze(-1), log_std.squeeze(-1).clamp(-5.0, 3.0)

    def nll(self, h: torch.Tensor, log_dur_target: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """训练损失: log_dur_target 在高斯头下的 NLL (每音素均值)。"""
        mu, log_std = self(h)
        nll = log_std + 0.5 * (log_dur_target - mu) ** 2 * torch.exp(-2 * log_std)
        if mask is not None:
            return (nll * mask).sum() / mask.sum().clamp(min=1)
        return nll.mean()

    def sample_durations(self, h: torch.Tensor, noise_scale: float = 1.0,
                         min_frames: int = 2, max_frames: int = 60,
                         dur_scale: float = 1.0) -> torch.Tensor:
        """推理采样: (B,N) int 帧数 (双向 clamp 防长尾)。"""
        mu, log_std = self(h)
        eps = torch.randn_like(mu) * noise_scale
        dur = torch.exp(mu + eps * log_std.exp()) * dur_scale
        return dur.round().long().clamp(min=min_frames, max=max_frames)


class ADR2(BaseBackbone):
    """ADR-2 声学模型骨架。"""

    config_class = ADR2Config
    capabilities = (BackboneCapability.TTS | BackboneCapability.SVS
                    | BackboneCapability.F0_COND | BackboneCapability.LORA_READY)

    def __init__(self, config: Optional[ADR2Config] = None):
        super().__init__(config)
        config = self.config  # type: ADR2Config

        # 文本 → 内容
        self.content_encoder = ContentEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.content_dim,
            n_layers=4, n_heads=4, ffn_dim=1024, max_len=1024,
        )
        # 参考 → 音色全局条件 g
        self.timbre_encoder = TimbreEncoder(
            n_mels=config.n_mels, embed_dim=config.timbre_dim,
        )

        # 先验投影: content → 逐音素高斯 (mu_p, log_std_p) in z 空间
        # 注意: 不要小初始化! 小 init → mu_p 趋同 → MAS 早期退化为
        # "全部帧对一个音素" → 反馈锁死。默认 init + KL warmup 即可。
        self.prior_proj = nn.Linear(config.content_dim, config.z_dim * 2)

        # KL warmup 步数计数 (随 state_dict 保存/恢复)
        self.register_buffer("_kl_step", torch.zeros((), dtype=torch.long))

        # 后验: mel → 逐帧高斯 q(z|mel)
        self.posterior = PosteriorEncoder(
            n_mels=config.n_mels, z_dim=config.z_dim,
            hidden=config.posterior_hidden, n_layers=config.posterior_layers,
        )

        # flow 增强先验 (条件 g = 音色)
        self.flow = ResidualCouplingFlow(
            z_dim=config.z_dim, hidden=config.flow_hidden,
            cond_dim=config.timbre_dim, n_layers=config.n_flow,
        )

        # 时长 (M9.3: 高斯随机头, 训练 NLL / 推理采样)
        self.dur_fuse = nn.Linear(config.content_dim + config.timbre_dim,
                                  config.hidden_dim)
        self.duration_predictor = StochasticDurationHead(config.hidden_dim)

        # M9.4: 模式嵌入 (TTS/SVS 共用 backbone, mode 嵌入并入音色条件)
        self.mode_emb = nn.Embedding(2, config.timbre_dim)

        # M9.4: F0 旋律条件投影 (log1p(f0/700) → f0_dim)
        self.f0_proj = nn.Linear(1, config.f0_dim) if config.use_f0 else None

        # M9.5: FlowDecoder (mel 空间条件流, 逐帧条件 = z + F0 + 音色 g)
        # 替代 mel_head 占位 — 精确 NLL 训练, 推理从噪声生成谐波细节
        f0_extra = config.f0_dim if config.use_f0 else 0
        self.mel_decoder = FlowDecoder(
            n_mels=config.n_mels,
            cond_dim=config.z_dim + config.timbre_dim + f0_extra,
            hidden=config.flowdec_hidden, n_layers=config.flowdec_layers,
        )

    # ---- 内部工具 ----
    def _effective_kl_weight(self) -> float:
        """KL warmup: 训练时线性 0→kl_weight (kl_warmup_steps 步)。"""
        if not self.training:
            return self.config.kl_weight
        self._kl_step += 1
        warm = min(1.0, float(self._kl_step.item()) /
                   max(1, self.config.kl_warmup_steps))
        return self.config.kl_weight * warm

    def _prior_params(self, ids: torch.Tensor):
        """音素 id → 先验 (mu_p, log_std_p): (B,z_dim,N)。

        先验方差固定为 1 (Glow-TTS 惯例): 可学习 std 会让 MAS 的对齐
        被方差项劫持 (某音素靠调大 std "吞掉"全部帧 → 对齐崩溃)。
        固定 std 后 logp 退化为纯 L2 距离, 对齐稳定。
        """
        c = self.content_encoder(ids)                    # (B,N,Cc)
        mu, _ = self.prior_proj(c).chunk(2, dim=-1)
        mu = mu.transpose(1, 2)                          # (B,z,N)
        log_std = torch.zeros_like(mu)                   # 固定 std=1
        return mu, log_std, c

    @staticmethod
    def _upsample(x: torch.Tensor, attn_or_dur: torch.Tensor) -> torch.Tensor:
        """先验参数上采样到帧级。
        attn_or_dur: (B,N,T) 对齐矩阵 或 (B,N) 时长。"""
        if attn_or_dur.dim() == 3:
            return torch.bmm(x, attn_or_dur)             # (B,z,N)x(B,N,T)
        # 时长路径 (推理)
        outs = []
        for b in range(x.size(0)):
            dur = attn_or_dur[b].long().clamp(min=0)
            outs.append(torch.repeat_interleave(x[b], dur, dim=-1))
        max_t = max(o.size(-1) for o in outs)
        outs = [F.pad(o, (0, max_t - o.size(-1))) for o in outs]
        return torch.stack(outs)

    # ---- 训练前向 ----
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ids = batch["phoneme_ids"]                        # (B,N)
        ref_mel = batch["ref_mel"]                        # (B,80,T_ref)
        target_mel = batch["target_mel"]                  # (B,80,T)
        target_mask = batch.get("target_mel_mask")        # (B,T) bool
        B, N = ids.shape
        T = target_mel.size(-1)
        device = ids.device
        if target_mask is None:
            target_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        g = self.timbre_encoder(ref_mel)                  # (B,Ct)
        # M9.4: 模式嵌入 (batch 可指定 mode_id, 默认 config.default_mode)
        mode_id = batch.get("mode_id")
        if mode_id is None:
            mode_id = torch.full((B,), self.config.default_mode,
                                 device=device, dtype=torch.long)
        g = g + self.mode_emb(mode_id)
        g_c = g.unsqueeze(-1)                             # (B,Ct,1) flow 条件

        # 后验 (训练锚点)
        mu_q, log_std_q = self.posterior(target_mel)      # (B,z,T)
        z = self.posterior.sample(mu_q, log_std_q)

        # 先验
        mu_p, log_std_p, c = self._prior_params(ids)      # (B,z,N)

        # flow: z (q 空间) → z_p (先验空间) — 注意必须先算:
        # MAS 与 KL 都必须在【同一空间】(z_p) 进行 (VITS 原文即如此)。
        # 若 MAS 用 raw z 而 KL 推拉 flow(z), 两空间随训练漂移错位,
        # 对齐在中后期必然崩溃 (v4/v5/v6 实测退化根因)。
        z_p, logdet = self.flow(z, g_c)                   # (B,z,T), (B,)

        # 对齐: 前 mas_warmup_steps 步用均摊对齐 (打破零开局对称性),
        # 之后 MAS 精修 (在 z_p 空间)
        with torch.no_grad():
            use_uniform = self.training and \
                int(self._kl_step.item()) < self.config.mas_warmup_steps
            if use_uniform and "target_durations" in batch:
                # 逐样本真实帧数 (不能传 batch 的 T_max: padding 会摊进时长)
                T_per = target_mask.sum(dim=1)              # (B,)
                attn = alignment_from_durations(
                    batch["target_durations"], T_per)       # (B,N,≤T)
                if attn.size(-1) < T:
                    attn = F.pad(attn, (0, T - attn.size(-1)))
            else:
                logp = gaussian_logp(z_p.detach(), mu_p.detach(),
                                     log_std_p.detach())      # (B,N,T)
                attn_mask = target_mask.unsqueeze(1).expand(-1, N, -1)
                attn = maximum_path(logp, attn_mask)          # (B,N,T)
            mas_dur = durations_from_alignment(attn)      # (B,N)

        # 先验上采样到帧级
        mu_p_up = self._upsample(mu_p, attn)              # (B,z,T)
        std_p_up = self._upsample(log_std_p, attn)

        # KL 在 flow 空间 (z_p 已在上方对齐前计算, 与 MAS 同空间)
        log_q = _gaussian_log_pdf(z, mu_q, log_std_q)     # (B,T)
        log_p = _gaussian_log_pdf(z_p, mu_p_up, std_p_up) \
            + logdet.unsqueeze(-1) / T
        kl = ((log_q - log_p) * target_mask).sum() \
            / target_mask.sum().clamp(min=1)

        # mel 重构: FlowDecoder NLL (精确极大似然, 免 GAN)
        # 逐帧条件 = 后验 z + F0 旋律 (M9.4) + 音色 g
        cond = self._make_cond(z, g_c, batch.get("f0"), T)
        recon = self.mel_decoder.nll(target_mel, cond, target_mask)

        # 评估用 pred_mel: 零噪声生成 (分布的"众数"轨迹)
        with torch.no_grad():
            pred_mel = self.mel_decoder.generate(cond.detach(), T,
                                                 temperature=0.0)

        # 时长损失 (MAS 时长 detach 作为目标, 随机头 NLL)
        # 目标截尾: MAS 双峰长尾 (单音素 200+ 帧) 会教坏时长头
        # 输入 detach (Glow-TTS 惯例): 防 recon/KL 大梯度经共享主干干扰时长头
        h_dur = self.dur_fuse(torch.cat([
            c.detach(), g.detach().unsqueeze(1).expand(-1, N, -1)], dim=-1))
        dur_clamped = mas_dur.clamp(min=1, max=self.config.dur_max_frames)
        dur_target = torch.log(dur_clamped.float()).detach()
        ph_mask = batch.get("phoneme_mask")
        dur_loss = self.duration_predictor.nll(h_dur, dur_target, ph_mask)

        loss = recon + self._effective_kl_weight() * kl \
            + self.config.dur_weight * dur_loss

        return {
            "loss": loss, "recon_loss": recon.detach(),
            "kl_loss": kl.detach(), "dur_loss": dur_loss.detach(),
            "pred_mel": pred_mel, "mas_durations": mas_dur,
        }

    # ---- M9.4: 条件构造 (z + F0 + g) ----
    def _make_cond(self, z: torch.Tensor, g_c: torch.Tensor,
                   f0: Optional[torch.Tensor], T: int) -> torch.Tensor:
        """组装 FlowDecoder 逐帧条件。

        Args:
            z: (B, z_dim, T)
            g_c: (B, timbre_dim, 1)
            f0: (B, T_f0) Hz 或 None (None → 零条件 = TTS 模式)
            T: 目标帧数
        """
        parts = [z, g_c.expand(-1, -1, T)]
        if self.f0_proj is not None:
            if f0 is not None:
                if f0.size(-1) != T:
                    f0 = F.interpolate(f0.unsqueeze(1).float(), size=T,
                                       mode="linear", align_corners=False
                                       ).squeeze(1)
                f0f = self.f0_proj(
                    torch.log1p(f0.clamp(min=0).float() / 700.0)
                    .unsqueeze(-1)).transpose(1, 2)       # (B,f0_dim,T)
            else:
                f0f = torch.zeros(z.size(0), self.config.f0_dim, T,
                                  device=z.device, dtype=z.dtype)
            parts.insert(1, f0f)
        return torch.cat(parts, dim=1)

    # ---- 推理 ----
    @torch.inference_mode()
    def sample(
        self,
        phoneme_ids: torch.Tensor,
        ref_mel: torch.Tensor,
        n_timesteps: int = 20,
        f0: Optional[torch.Tensor] = None,
        mode: Optional[object] = None,   # "tts"/"svs"/0/1, None=config.default_mode
        min_mel_frames: int = 80,
        noise_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """文本 + 参考 (+ 可选旋律 F0) → mel (B, n_mels, T)。"""
        B = phoneme_ids.size(0)
        device = phoneme_ids.device
        if phoneme_ids.size(1) < 3:
            phoneme_ids = F.pad(phoneme_ids, (0, 3 - phoneme_ids.size(1)), value=0)

        g = self.timbre_encoder(ref_mel)
        # M9.4: 模式
        if mode is None:
            mode = self.config.default_mode
        if isinstance(mode, str):
            mode = {"tts": 0, "svs": 1}[mode]
        g = g + self.mode_emb(torch.full((B,), int(mode), device=device,
                                         dtype=torch.long))
        g_c = g.unsqueeze(-1)
        mu_p, log_std_p, c = self._prior_params(phoneme_ids)
        N = phoneme_ids.size(1)

        # 随机时长采样 → 上采样 (detach 与训练侧一致)
        h_dur = self.dur_fuse(torch.cat([
            c.detach(), g.detach().unsqueeze(1).expand(-1, N, -1)], dim=-1))
        durations = self.duration_predictor.sample_durations(
            h_dur, noise_scale=self.config.dur_noise_scale,
            min_frames=self.config.dur_min_frames,
            max_frames=self.config.dur_max_frames,
            dur_scale=self.config.dur_scale)              # (B,N)

        mu_up = self._upsample(mu_p, durations)           # (B,z,T)
        std_up = self._upsample(log_std_p, durations)
        T = mu_up.size(-1)
        if T < min_mel_frames:
            pad = min_mel_frames - T
            mu_up = F.pad(mu_up, (0, pad))
            std_up = F.pad(std_up, (0, pad), value=-7.0)

        # 先验采样 → flow^{-1} → FlowDecoder 生成 (条件含 F0 旋律)
        eps = torch.randn_like(mu_up) * noise_scale
        z_p = mu_up + eps * std_up.exp()
        z = self.flow.inverse(z_p, g_c)
        # f0: np/tensor → tensor (B, T_f0); None → TTS 零条件
        if f0 is not None and not torch.is_tensor(f0):
            f0 = torch.from_numpy(np.asarray(f0, dtype=np.float32))
        if f0 is not None:
            f0 = f0.to(device)
            if f0.dim() == 1:
                f0 = f0.unsqueeze(0)
        cond = self._make_cond(z, g_c, f0, z.size(-1))
        mel = self.mel_decoder.generate(cond, z.size(-1),
                                        temperature=self.config.temperature)
        return mel


register("backbone", "adr2")(ADR2)
