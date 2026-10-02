"""ADR 顶层模型 — 组合内容/旋律/音色编码器与 DiT backbone。"""
from typing import Optional

import torch
import torch.nn as nn

from ..configs import ADRConfig
from .content_encoder import ContentEncoder
from .melody_encoder import MelodyEncoder
from .timbre_encoder import TimbreEncoder
from .backbone import SharedBackbone, cfm_loss


class ADRModel(nn.Module):
    """ADR 顶层模型：说话 + 唱歌统一生成。"""

    def __init__(self, config: ADRConfig):
        super().__init__()
        self.config = config
        self.content = ContentEncoder(
            vocab_size=config.content.vocab_size,
            embed_dim=config.content.embed_dim,
            n_layers=config.content.n_layers,
            n_heads=config.content.n_heads,
            ffn_dim=config.content.ffn_dim,
            max_len=config.content.max_phoneme_len,
            dropout=config.content.dropout,
        )
        self.melody = MelodyEncoder(
            n_pitch_bins=config.melody.n_pitch_bins,
            n_velocity_bins=config.melody.n_velocity_bins,
            embed_dim=config.melody.embed_dim,
            n_layers=config.melody.n_layers,
            n_heads=config.melody.n_heads,
            ffn_dim=config.melody.n_layers * 256,
        )
        self.timbre = TimbreEncoder(
            n_mels=config.timbre.ref_mel_bins,
            embed_dim=config.timbre.embed_dim,
        )
        self.backbone = SharedBackbone(
            n_mels=config.backbone.mel_bins,
            hidden_dim=config.backbone.hidden_dim,
            n_layers=config.backbone.n_layers,
            n_heads=config.backbone.n_heads,
            ffn_dim=config.backbone.ffn_dim,
            content_dim=config.content.embed_dim,
            melody_dim=config.melody.embed_dim,
            timbre_dim=config.timbre.embed_dim,
            max_frames=config.backbone.max_mel_frames,
            use_task_modulation=config.backbone.use_task_modulation,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """训练时一步前向 + CFM loss。

        batch dict 期望：
          - phoneme_ids: (B, T_p) int64
          - phoneme_mask: (B, T_p) bool
          - pitch_ids: (B, T_n) int64  说话模式全 0
          - note_duration: (B, T_n) float
          - velocity: (B, T_n) int64
          - is_singing: (B,) bool
          - ref_mel: (B, n_mels, T_mel)
          - target_mel: (B, n_mels, T_mel)
        """
        content = self.content(batch["phoneme_ids"], batch.get("phoneme_mask"))
        melody = self.melody(
            batch["pitch_ids"], batch["note_duration"], batch["velocity"],
            batch["is_singing"],
        )
        timbre = self.timbre(batch["ref_mel"])
        task = None
        if self.config.backbone.use_task_modulation:
            task = batch["is_singing"].long()
        return cfm_loss(self.backbone, batch["target_mel"], content, melody, timbre, task)

    @torch.no_grad()
    def sample(self, phoneme_ids, ref_mel,
               pitch_ids=None, note_duration=None, velocity=None,
               is_singing=False, n_timesteps: int = 20):
        """推理：CFM 采样生成 mel。

        说话模式：pitch/note/velocity 传 None 即可。
        唱歌模式：必须提供对齐到音素的音符序列。
        """
        device = next(self.parameters()).device
        B, T_p = phoneme_ids.shape
        n_mels = self.config.backbone.mel_bins
        # 估算 mel 帧数：简单按音素与音符时长求和（M1 用更精细策略）
        if is_singing and note_duration is not None:
            mel_frames = int((note_duration.sum(dim=1) * 80).max().item())
            mel_frames = max(mel_frames, 100)
        else:
            mel_frames = max(int(T_p * 12), 100)  # 12Hz codec 经验
        mel_frames = min(mel_frames, self.config.backbone.max_mel_frames)

        content = self.content(phoneme_ids.to(device))
        if is_singing:
            melody = self.melody(pitch_ids.to(device), note_duration.to(device),
                                 velocity.to(device),
                                 torch.tensor([True] * B, device=device))
        else:
            # 说话模式：pitch/duration/velocity 用零占位
            T_n = max(T_p, 1)
            melody = self.melody(
                torch.zeros(B, T_n, dtype=torch.long, device=device),
                torch.zeros(B, T_n, device=device),
                torch.zeros(B, T_n, dtype=torch.long, device=device),
                torch.tensor([False] * B, device=device),
            )
        timbre = self.timbre(ref_mel.to(device))
        task = torch.tensor([1 if is_singing else 0] * B,
                            dtype=torch.long, device=device) if self.config.backbone.use_task_modulation else None

        # CFM 采样（midpoint integrator，简化）
        x = torch.randn(B, n_mels, mel_frames, device=device)
        dt = 1.0 / n_timesteps
        for i in range(n_timesteps):
            t_cur = torch.full((B,), i * dt, device=device)
            v = self.backbone(x, t_cur, content, melody, timbre, task)
            x = x + dt * v
        return x  # (B, n_mels, mel_frames)
