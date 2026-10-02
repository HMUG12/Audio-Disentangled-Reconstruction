"""M9.1: MAS 单调对齐搜索测试。"""
from __future__ import annotations

import numpy as np
import torch

from adr.models.alignment import (
    durations_from_alignment,
    gaussian_logp,
    maximum_path,
)


def _make_logp(N: int, T: int, boundaries: list[int]) -> torch.Tensor:
    """构造已知真值的 logp: 音素 n 在 [boundaries[n], boundaries[n+1]) 帧段高分。"""
    logp = torch.full((1, N, T), -10.0)
    for n in range(N):
        lo = boundaries[n]
        hi = boundaries[n + 1] if n + 1 < len(boundaries) else T
        logp[0, n, lo:hi] = 10.0
    return logp


class TestMaximumPath:
    def test_recovers_known_alignment(self):
        """toy case: 已知分段, MAS 应精确恢复。"""
        N, T = 4, 12
        bounds = [0, 2, 5, 9, 12]
        logp = _make_logp(N, T, bounds)
        mask = torch.ones(1, N, T, dtype=torch.bool)
        path = maximum_path(logp, mask)
        dur = durations_from_alignment(path)[0].tolist()
        assert dur == [2, 3, 4, 3], f"期望 [2,3,4,3], 得到 {dur}"

    def test_monotonic_and_coverage(self):
        """随机 logp: 每帧恰好一个音素, 单调, 每个音素 >=1 帧。"""
        torch.manual_seed(0)
        B, N, T = 2, 7, 40
        logp = torch.randn(B, N, T)
        mask = torch.ones(B, N, T, dtype=torch.bool)
        path = maximum_path(logp, mask)
        # 每帧恰好一个音素
        assert torch.allclose(path.sum(dim=1), torch.ones(B, T))
        # 每个音素 >=1 帧
        dur = durations_from_alignment(path)
        assert (dur >= 1).all()
        # 单调: 每帧分配的音素索引非降
        assign = path.argmax(dim=1)  # (B, T)
        for b in range(B):
            a = assign[b].tolist()
            assert all(a[i] <= a[i + 1] for i in range(T - 1))

    def test_mask_padding(self):
        """mask 截断: 有效 N=3, T=8 (padding 区随机高分也不得被选中)。"""
        N_full, T_full = 5, 12
        logp = torch.randn(1, N_full, T_full)
        logp[0, 3:, :] = 100.0   # padding 音素给超高分
        logp[0, :, 8:] = 100.0   # padding 帧给超高分
        mask = torch.zeros(1, N_full, T_full, dtype=torch.bool)
        mask[0, :3, :8] = True
        path = maximum_path(logp, mask)
        dur = durations_from_alignment(path)[0].tolist()
        assert sum(dur) == 8, "只应覆盖有效帧"
        assert all(d == 0 for d in dur[3:]), "padding 音素不应分到帧"
        assert path[0, :, 8:].sum() == 0, "padding 帧不应被分配"

    def test_equal_split_when_flat(self):
        """平坦 logp: 应近似均分 (不崩溃即可)。"""
        N, T = 3, 9
        logp = torch.zeros(1, N, T)
        mask = torch.ones(1, N, T, dtype=torch.bool)
        dur = durations_from_alignment(maximum_path(logp, mask))[0].tolist()
        assert sum(dur) == T
        assert all(d >= 1 for d in dur)

    def test_gaussian_logp_shape(self):
        """gaussian_logp: 形状与方向 (z 靠近 mu 的位置得分高)。"""
        B, C, N, T = 1, 4, 3, 6
        mu = torch.zeros(B, C, N)
        mu[0, :, 0] = -1.0
        mu[0, :, 1] = 0.0
        mu[0, :, 2] = 1.0
        log_std = torch.zeros(B, C, N)
        z = torch.zeros(B, C, T)
        z[0, :, 1] = -1.0  # 第 1 帧接近音素 0
        z[0, :, 4] = 1.0   # 第 4 帧接近音素 2
        logp = gaussian_logp(z, mu, log_std)
        assert logp.shape == (B, N, T)
        assert logp[0, 0, 1] > logp[0, 1, 1] > logp[0, 2, 1]
        assert logp[0, 2, 4] > logp[0, 1, 4] > logp[0, 0, 4]

    def test_end_to_end_toy(self):
        """gaussian_logp + MAS: 分段高斯应恢复分段边界。"""
        # 3 个音素, 均值 -2/0/2; 帧 0-3 在 -2 附近, 4-7 在 0, 8-11 在 2
        C = 2
        mu = torch.tensor([[[-2.0] * C, [0.0] * C, [2.0] * C]]).transpose(1, 2).float()
        mu = torch.tensor([[-2., -2.], [0., 0.], [2., 2.]]).unsqueeze(0)  # (1,N,C)
        mu = mu.transpose(1, 2)  # (1,C,N)
        log_std = torch.zeros(1, C, 3)
        z = torch.zeros(1, C, 12)
        z[0, :, 0:4] = -2.0
        z[0, :, 8:12] = 2.0
        logp = gaussian_logp(z, mu, log_std)
        mask = torch.ones(1, 3, 12, dtype=torch.bool)
        dur = durations_from_alignment(maximum_path(logp, mask))[0].tolist()
        assert dur == [4, 4, 4], f"期望 [4,4,4], 得到 {dur}"
