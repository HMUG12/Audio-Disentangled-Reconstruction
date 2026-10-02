"""共享 DiT backbone — 内容/旋律/音色三路条件 + 条件流匹配生成声学特征。

设计参照：
- UniVoice: 共享 DiT backbone + 任务调制 token（speech / sing 二选一）
- F5-TTS / CosyVoice: 整流流（CFM）目标
- DiffSinger: 声学特征 → mel 谱（不直接出 wav，由声码器转为波形）
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbed(nn.Module):
    """扩散/CFM 的时间步嵌入。"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        x = t[:, None] * freqs[None]
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class DiTBlock(nn.Module):
    """简化版 DiT block：adaLN-Zero + Self-Attention + FFN。"""
    def __init__(self, hidden_dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, hidden_dim)
        )
        # adaLN-Zero: 由 cond 生成 scale/shift/gate
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, D) 全局条件（time + timbre + task）
        s_ms, s_mg, s_ma, s_fs, s_fg, s_fa = self.adaLN(cond).chunk(6, dim=-1)
        s_ms, s_mg, s_ma = s_ms[:, None], s_mg[:, None], s_ma[:, None]
        s_fs, s_fg, s_fa = s_fs[:, None], s_fg[:, None], s_fa[:, None]
        h = self.norm1(x) * (1 + s_ms) + s_mg
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + s_ma * attn_out
        h = self.norm2(x) * (1 + s_fs) + s_fg
        ffn_out = self.ffn(h)
        x = x + s_fa * ffn_out
        return x


class SharedBackbone(nn.Module):
    """统一 DiT + 条件流匹配 backbone。

    输入：
      noisy_mel: (B, n_mels, T_mel)   # CFM 噪声插值后的 mel
      t:         (B,)                  # CFM 时间步
      content:   (B, T_phoneme, D_c)   # 内容条件
      melody:    (B, T_note, D_m)      # 旋律条件（说话模式已是 null）
      timbre:    (B, D_t)              # 音色条件
      task_emb:  (B, D_t) 或 None      # 任务位
    输出：
      v: (B, n_mels, T_mel)            # 预测的速度场
    """

    def __init__(self, n_mels: int = 80, hidden_dim: int = 1024,
                 n_layers: int = 28, n_heads: int = 16, ffn_dim: int = 3072,
                 content_dim: int = 256, melody_dim: int = 128, timbre_dim: int = 256,
                 max_frames: int = 1500, use_task_modulation: bool = True):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.use_task_modulation = use_task_modulation

        # 输入投影：mel + content + melody 统一到 hidden_dim
        self.mel_proj = nn.Linear(n_mels, hidden_dim)
        self.content_proj = nn.Linear(content_dim, hidden_dim)
        self.melody_proj = nn.Linear(melody_dim, hidden_dim)
        self.timbre_proj = nn.Linear(timbre_dim, hidden_dim)

        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbed(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # 任务调制：speech / sing 两个可学习 token
        if use_task_modulation:
            self.task_embed = nn.Embedding(2, hidden_dim)
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 64, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads, ffn_dim) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, n_mels)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, noisy_mel: torch.Tensor, t: torch.Tensor,
                content: torch.Tensor, melody: torch.Tensor,
                timbre: torch.Tensor, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            noisy_mel: (B, n_mels, T_mel)  注意是 (n_mels, T) 不是 (T, n_mels)
        """
        B, n_mels, T = noisy_mel.shape
        # mel → (B, T, n_mels) → (B, T, D)
        x = self.mel_proj(noisy_mel.transpose(1, 2))
        # 条件聚合：把 content/melody 在时间轴上插值到 mel 帧
        T_c, T_m = content.size(1), melody.size(1)
        # 上采样到 T 帧（线性插值）
        c = F.interpolate(content.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
        m = F.interpolate(melody.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
        x = x + self.content_proj(c) + self.melody_proj(m)
        # 全局条件
        t_emb = self.time_mlp(self.time_embed(t))
        cond = t_emb + self.timbre_proj(timbre)
        if self.use_task_modulation and task is not None:
            cond = cond + self.task_embed(task)
        # 位置编码
        x = x + self.pos_embed[:, :T]
        for block in self.blocks:
            x = block(x, cond)
        x = self.norm_out(x)
        v = self.proj_out(x)  # (B, T, n_mels)
        return v.transpose(1, 2)  # (B, n_mels, T)


def cfm_loss(model: SharedBackbone, mel: torch.Tensor,
             content: torch.Tensor, melody: torch.Tensor,
             timbre: torch.Tensor, task: Optional[torch.Tensor] = None,
             sigma_min: float = 1e-5) -> torch.Tensor:
    """条件流匹配损失（optimal transport CFM）。

    mel: (B, n_mels, T) 真实 mel
    """
    B, n_mels, T = mel.shape
    device = mel.device
    t = torch.rand(B, device=device)  # 均匀采样时间
    noise = torch.randn_like(mel)
    # 插值
    t_b = t.view(B, 1, 1)
    x_t = (1 - (1 - sigma_min) * t_b) * noise + t_b * mel
    # 速度场
    v_pred = model(x_t, t, content, melody, timbre, task)
    v_target = mel - (1 - sigma_min) * noise
    return F.mse_loss(v_pred, v_target)
