"""M9.1: MAS 单调对齐搜索 (Monotonic Alignment Search)。

学自 VITS 的设计原理, 自实现 (不依赖其 Cython):

训练时自动学习 "音素 n ↔ mel 帧 t" 的单调对齐, 免去 MFA 外部对齐工具:
- 输入 logp (B, N, T): 每个 (音素, 帧) 对的对数似然 (通常是高斯 NLL:
  logp[n,t] = -0.5 * ||z_t - mu_n||^2 / sigma^2, z 来自 VAE 后验)
- 输出 path (B, N, T): 0/1 对齐矩阵, 满足:
  1. 每帧 t 恰好分配给一个音素 (sum over n = 1)
  2. 单调: 音素顺序不倒退
  3. 边界: 第 0 帧属于音素 0, 最后一帧属于最后一个音素
  4. 每个音素至少 1 帧

DP 递推 (VITS 标准形式):
  Q[n,t] = logp[n,t] + max(Q[n,t-1], Q[n-1,t-1])
  约束 n <= t 且 (N-1-n) <= (T-1-t) (保证剩余音素都有帧可分)
回溯: 从 (N-1, T-1) 出发, Q[n,t-1] >= Q[n-1,t-1] 则停留, 否则前进。

复杂度 O(B*N*T), numpy 实现, 对 N≤100, T≤2000 足够快。
"""
from __future__ import annotations

import numpy as np
import torch


def maximum_path(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """单调对齐搜索 (batch 版)。

    Args:
        logp: (B, N, T) 对数似然, 越大越应对齐
        mask: (B, N, T) bool, True = 有效位置 (padding 区域 False)

    Returns:
        path: (B, N, T) float32 0/1 对齐矩阵
    """
    device, dtype = logp.device, logp.dtype
    logp_np = logp.detach().float().cpu().numpy()
    mask_np = mask.detach().bool().cpu().numpy()
    B, N, T = logp_np.shape
    paths = np.zeros((B, N, T), dtype=np.float32)
    for b in range(B):
        n_valid = int(mask_np[b, :, 0].sum())   # 有效音素数
        t_valid = int(mask_np[b, 0, :].sum())   # 有效帧数
        paths[b] = _maximum_path_single(logp_np[b], mask_np[b], n_valid, t_valid)
    return torch.from_numpy(paths).to(device=device, dtype=dtype)


def _maximum_path_single(logp: np.ndarray, mask: np.ndarray,
                         N: int, T: int) -> np.ndarray:
    """单样本 DP。只考虑前 N 音素 × 前 T 帧 (有效区域)。"""
    NEG = -1e9
    v = np.where(mask[:N, :T], logp[:N, :T], NEG).astype(np.float64)

    # Q[n,t]: 以 (n,t) 为终点的最优路径得分
    Q = np.full((N, T), NEG, dtype=np.float64)
    Q[0, 0] = v[0, 0]
    # 第一列: 只能停留 (音素 0 吃多帧)
    for t in range(1, T):
        Q[0, t] = v[0, t] + Q[0, t - 1]
    # 第一行: 只能前进 (每音素至少一帧, 不允许跳过)
    for n in range(1, N):
        if n < T:  # n <= t 才可能
            Q[n, n] = v[n, n] + Q[n - 1, n - 1]
    # 主体
    for n in range(1, N):
        for t in range(n + 1, T - (N - 1 - n)):
            Q[n, t] = v[n, t] + max(Q[n, t - 1], Q[n - 1, t - 1])

    # 回溯: 终点必须是 (N-1, T-1)
    path = np.zeros((logp.shape[0], logp.shape[1]), dtype=np.float32)
    n, t = N - 1, T - 1
    path[n, t] = 1.0
    while t > 0:
        if n > 0 and Q[n - 1, t - 1] > Q[n, t - 1]:
            n -= 1
        path[n, t - 1] = 1.0
        t -= 1
    return path


def durations_from_alignment(path: torch.Tensor) -> torch.Tensor:
    """对齐矩阵 → 每音素时长 (帧数)。

    Args:
        path: (B, N, T) 0/1 对齐矩阵

    Returns:
        durations: (B, N) int64
    """
    return path.sum(dim=-1).long()


def alignment_from_durations(durations: torch.Tensor,
                             T: "int | torch.Tensor") -> torch.Tensor:
    """时长 → 对齐矩阵 (MAS warmup 用: 打破零开局对称性)。

    Args:
        durations: (B, N) 每音素帧数
        T: 目标帧数。int = 全部样本相同; (B,) tensor = 逐样本真实长度
           (**必须传逐样本长度**, 传 batch  padding 后的 T_max 会把
           padding 帧摊进时长, 时长头直接学飞 — v9 实测 bug)

    Returns:
        path: (B, N, T_max) 0/1 对齐矩阵 (超出样本真实长度的部分为 0)
    """
    B, N = durations.shape
    device = durations.device
    if isinstance(T, int):
        T = torch.full((B,), T, device=device, dtype=torch.long)
    T_max = int(T.max().item())
    path = torch.zeros(B, N, T_max, device=device)
    for b in range(B):
        T_b = int(T[b].item())
        dur = durations[b].float()
        # 归一化到该样本真实长度
        dur = (dur / dur.sum().clamp(min=1) * T_b).round().long().clamp(min=1)
        diff = int(dur.sum().item()) - T_b
        while diff != 0:
            i = int(dur.argmax()) if diff > 0 else int(dur.argmin())
            if diff > 0:
                dur[i] -= 1
                diff -= 1
            else:
                dur[i] += 1
                diff += 1
        cum = torch.cumsum(dur, dim=0).tolist()
        start = 0
        for n, end in enumerate(cum):
            path[b, n, start:end] = 1.0
            start = end
    return path


def gaussian_logp(z: torch.Tensor, mu: torch.Tensor,
                  log_std: torch.Tensor) -> torch.Tensor:
    """MAS 的对数似然: 后验帧 z_t 与先验音素 mu_n 的高斯 NLL。

    Args:
        z: (B, C, T) 后验潜变量 (VAE posterior 采样)
        mu: (B, C, N) 先验均值 (文本编码)
        log_std: (B, C, N) 先验对数标准差

    Returns:
        logp: (B, N, T)
    """
    # -0.5 * sum_c [ log(2pi) + 2*log_std + (z-mu)^2/std^2 ]
    z_e = z.unsqueeze(2)          # (B, C, 1, T)
    mu_e = mu.unsqueeze(-1)       # (B, C, N, 1)
    std_e = log_std.unsqueeze(-1) # (B, C, N, 1)
    nll = 0.5 * (np.log(2 * np.pi) + 2 * std_e
                 + ((z_e - mu_e) ** 2) * torch.exp(-2 * std_e))
    return -nll.sum(dim=1)        # (B, N, T)
