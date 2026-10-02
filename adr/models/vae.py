"""M9.2: VAE 后验编码器 + 残差耦合 flow 先验。

学自 VITS/Glow-TTS 的设计原理, 自实现:

- PosteriorEncoder: mel → 逐帧高斯后验 q(z|mel), 训练时提供 MAS 的 "锚点"
- ResidualCouplingFlow: 增强文本先验 p(z|text) 的表达能力
  (可逆仿射耦合层, 带全局条件 g = 音色+模式嵌入)
- 推理时先验采样: z = flow^{-1}(eps, g), eps ~ N(0,1)

QLoRA-ready: 所有可学习变换均为 Conv1d/Linear, 无自定义 CUDA 算子。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PosteriorEncoder(nn.Module):
    """mel → 后验 (mu, log_std), WaveNet-lite 膨胀卷积堆叠。

    Args:
        n_mels: mel 维度 (80)
        z_dim: 潜变量维度
        hidden: 隐层维度
        n_layers: 膨胀卷积层数 (dilation 1,2,4,... 循环)
        kernel: 卷积核大小
    """

    def __init__(self, n_mels: int = 80, z_dim: int = 64, hidden: int = 128,
                 n_layers: int = 6, kernel: int = 5):
        super().__init__()
        self.z_dim = z_dim
        self.pre = nn.Conv1d(n_mels, hidden, 1)
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            dil = 2 ** (i % 4)
            self.convs.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, padding=dil * (kernel - 1) // 2,
                          dilation=dil),
                nn.GELU(),
            ))
        self.proj = nn.Conv1d(hidden, z_dim * 2, 1)

    def forward(self, mel: torch.Tensor):
        """mel (B, n_mels, T) → mu, log_std (B, z_dim, T)。"""
        h = self.pre(mel)
        for conv in self.convs:
            h = conv(h) + h  # 残差
        mu, log_std = self.proj(h).chunk(2, dim=1)
        log_std = log_std.clamp(-7.0, 7.0)
        return mu, log_std

    def sample(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """重参数化采样。"""
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(log_std)


class CouplingLayer(nn.Module):
    """仿射耦合层: 通道分半, 一半经另一半 (+条件 g) 预测仿射参数。

    条件 g 支持两种形态:
    - 全局条件 (B, C, 1): 自动扩展到 T (音色嵌入等)
    - 逐帧条件 (B, C, T): 直接使用 (M9.5 FlowDecoder 的 z 条件)

    forward:  y1 = x1;  y2 = x2 * exp(s(x1, g)) + t(x1, g)
    inverse:  x2 = (y2 - t(y1, g)) * exp(-s(y1, g))
    """

    def __init__(self, z_dim: int, hidden: int, cond_dim: int,
                 kernel: int = 3, n_layers: int = 3):
        super().__init__()
        assert z_dim % 2 == 0, "z_dim 必须为偶数 (通道分半)"
        self.half = z_dim // 2
        in_dim = self.half + cond_dim
        layers: list[nn.Module] = [nn.Conv1d(in_dim, hidden, 1), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Conv1d(hidden, hidden, kernel,
                                 padding=kernel // 2), nn.GELU()]
        self.net = nn.Sequential(*layers)
        # s/t 输出零初始化 → 初始为恒等映射 (训练稳定, VITS 同款技巧)
        self.out = nn.Conv1d(hidden, self.half * 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _expand_g(g: torch.Tensor, T: int) -> torch.Tensor:
        return g.expand(-1, -1, T) if g.size(-1) == 1 else g

    def forward(self, x: torch.Tensor, g: torch.Tensor):
        """x (B, z_dim, T), g (B, cond_dim, 1|T) → y, logdet (B,)。"""
        x1, x2 = x.chunk(2, dim=1)
        h = self.net(torch.cat([x1, self._expand_g(g, x.size(-1))], dim=1))
        s, t = self.out(h).chunk(2, dim=1)
        s = torch.tanh(s)  # 限幅稳定
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)
        logdet = s.sum(dim=(1, 2))
        return y, logdet

    def inverse(self, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=1)
        h = self.net(torch.cat([y1, self._expand_g(g, y.size(-1))], dim=1))
        s, t = self.out(h).chunk(2, dim=1)
        s = torch.tanh(s)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=1)


class ResidualCouplingFlow(nn.Module):
    """N 层耦合 flow + 通道翻转, 支持正逆双向。"""

    def __init__(self, z_dim: int = 64, hidden: int = 128, cond_dim: int = 128,
                 n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            CouplingLayer(z_dim, hidden, cond_dim) for _ in range(n_layers))

    def forward(self, z: torch.Tensor, g: torch.Tensor):
        """z → u (增强先验方向), 返回 (u, logdet)。"""
        logdet_total = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        h = z
        for layer in self.layers:
            h, logdet = layer(h, g)
            h = h.flip(dims=[1])  # 通道翻转, 让两半轮流被变换
            logdet_total = logdet_total + logdet
        return h, logdet_total

    def inverse(self, u: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = u
        for layer in reversed(self.layers):
            h = h.flip(dims=[1])
            h = layer.inverse(h, g)
        return h


def kl_gaussian(mu_q: torch.Tensor, log_std_q: torch.Tensor,
                mu_p: torch.Tensor, log_std_p: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """逐帧 KL( q || p ), 两高斯解析解。

    Args:
        mu_q/log_std_q: (B, C, T) 后验参数
        mu_p/log_std_p: (B, C, T) 先验参数 (MAS 上采样后)
        mask: (B, T) bool 有效帧

    Returns:
        标量 loss (有效帧均值)
    """
    var_q = torch.exp(2 * log_std_q)
    var_p = torch.exp(2 * log_std_p)
    kl = log_std_p - log_std_q + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p) - 0.5
    kl = kl.sum(dim=1)  # (B, T), 对潜变量维求和
    if mask is not None:
        kl = (kl * mask).sum() / mask.sum().clamp(min=1)
    else:
        kl = kl.mean()
    return kl


class FlowDecoder(nn.Module):
    """M9.5: mel 空间条件流解码器 (替代 mel_head 占位)。

    设计: 在 mel 通道 (80) 上做仿射耦合 flow, 逐帧条件 = 潜变量 z + 音色 g。
    - 训练 (inverse): u = D^{-1}(mel; z, g), NLL = 0.5*||u||² - logdet
      精确极大似然, 无 GAN, 无迭代采样
    - 推理 (forward): mel = D(eps; z, g), eps ~ N(0, temperature²)
      随机性来自 eps → 可产生谐波等锐利结构 (确定性回归会被抹平)

    与先验 flow 的区别: 先验 flow 增强 p(z|text) 分布表达;
    FlowDecoder 直接建模 p(mel|z) —— 负责"把潜变量渲染成频谱细节"。
    """

    def __init__(self, n_mels: int = 80, cond_dim: int = 192,
                 hidden: int = 192, n_layers: int = 5):
        super().__init__()
        self.n_mels = n_mels
        self.layers = nn.ModuleList(
            CouplingLayer(n_mels, hidden, cond_dim, kernel=5, n_layers=3)
            for _ in range(n_layers))

    def forward(self, eps: torch.Tensor, cond: torch.Tensor):
        """生成方向: eps (B,80,T) → mel, 返回 (mel, logdet)。"""
        h = eps
        logdet_total = torch.zeros(eps.size(0), device=eps.device,
                                   dtype=eps.dtype)
        for layer in self.layers:
            h, logdet = layer(h, cond)
            h = h.flip(dims=[1])
            logdet_total = logdet_total + logdet
        return h, logdet_total

    def inverse(self, mel: torch.Tensor, cond: torch.Tensor):
        """训练方向: mel → u, 返回 (u, logdet_of_inverse)。"""
        h = mel
        logdet_total = torch.zeros(mel.size(0), device=mel.device,
                                   dtype=mel.dtype)
        for layer in reversed(self.layers):
            h = h.flip(dims=[1])
            # inverse 的 logdet 是 forward 的负值
            x1 = h.chunk(2, dim=1)[0]
            h_net_in = torch.cat([x1, CouplingLayer._expand_g(cond, h.size(-1))], dim=1)
            s, _ = layer.out(layer.net(h_net_in)).chunk(2, dim=1)
            s = torch.tanh(s)
            h = layer.inverse(h, cond)
            logdet_total = logdet_total - s.sum(dim=(1, 2))
        return h, logdet_total

    def nll(self, mel: torch.Tensor, cond: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """训练损失: -log p(mel|cond) (忽略常数项)。"""
        u, logdet = self.inverse(mel, cond)
        nll_frame = 0.5 * u.pow(2).sum(dim=1) - logdet.unsqueeze(-1) / mel.size(-1)
        if mask is not None:
            return (nll_frame * mask).sum() / mask.sum().clamp(min=1)
        return nll_frame.mean()

    def generate(self, cond: torch.Tensor, T: int,
                 temperature: float = 0.7) -> torch.Tensor:
        """推理: 从先验噪声生成 mel (B, n_mels, T)。"""
        B, device = cond.size(0), cond.device
        eps = torch.randn(B, self.n_mels, T, device=device) * temperature
        mel, _ = self.forward(eps, cond)
        return mel
