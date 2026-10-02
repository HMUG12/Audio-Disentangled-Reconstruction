"""M8.6: backbone 插件接口定型测试。

接口约定 (一期 SoVITS / 二期 ADR-2 / 保底第三方底座共用):
- capabilities 位掩码声明 (TTS/SVS/STREAM/F0_COND/LORA_READY)
- supports(cap) / supports_mode("tts"|"svs")
- sample_stream: 逐 chunk yield mel (默认整段切片回退)
- sample 接受 f0 条件 (声明 F0_COND 时)
"""
from __future__ import annotations

import torch

from adr.models.base import BackboneCapability, BaseBackbone
from adr.models.sovits import SoVITS, SoVITSConfig


def _tiny_model() -> SoVITS:
    cfg = SoVITSConfig(hidden_dim=32, n_layers=1, n_heads=2, ffn_dim=64,
                       content_dim=32, timbre_dim=16)
    return SoVITS(cfg).eval()


class TestBackboneInterface:
    def test_sovits_capabilities(self):
        m = _tiny_model()
        assert m.supports(BackboneCapability.TTS)
        assert m.supports(BackboneCapability.SVS)
        assert m.supports(BackboneCapability.F0_COND)
        assert m.supports(BackboneCapability.LORA_READY)
        assert SoVITS.supports_mode("tts")
        assert SoVITS.supports_mode("svs")

    def test_base_default_tts_only(self):
        """基类默认只声明 TTS。"""
        assert BaseBackbone.capabilities == BackboneCapability.TTS

    def test_sample_stream_chunks(self):
        """默认流式回退: chunks 拼接 == 整段 sample。"""
        m = _tiny_model()
        ids = torch.tensor([[10, 20, 30, 40]])
        ref = torch.randn(1, 80, 120)
        full = m.sample(ids, ref, n_timesteps=2)
        chunks = list(m.sample_stream(ids, ref, n_timesteps=2, chunk_frames=30))
        assert len(chunks) >= 2
        joined = torch.cat(chunks, dim=-1)
        assert joined.shape == full.shape
        assert torch.equal(joined, full)

    def test_sample_accepts_f0(self):
        """F0_COND: sample(f0=...) 路径可用且影响输出。

        注: f0_embed 零初始化 (兼容旧 ckpt), 未训练时贡献恒为 0,
        因此先随机化 f0_embed 模拟训练后状态, 验证管线连通。
        """
        m = _tiny_model()
        with torch.no_grad():
            m.f0_embed.weight.normal_(0, 0.1)
        ids = torch.tensor([[10, 20, 30, 40]])
        ref = torch.randn(1, 80, 120)
        f0 = torch.rand(1, 200) * 300 + 100
        mel_no_f0 = m.sample(ids, ref, n_timesteps=2)
        mel_f0 = m.sample(ids, ref, n_timesteps=2, f0=f0)
        assert mel_f0.dim() == 3
        assert not torch.allclose(mel_f0, mel_no_f0, atol=1e-5)
