"""M9.2: VAE 组件 + ADR2 骨架测试。"""
from __future__ import annotations

import torch

from adr.models.adr2 import ADR2, ADR2Config
from adr.models.base import BackboneCapability
from adr.models.vae import (
    PosteriorEncoder,
    ResidualCouplingFlow,
    kl_gaussian,
)


def _tiny_adr2() -> ADR2:
    cfg = ADR2Config(hidden_dim=32, content_dim=32, timbre_dim=16, z_dim=8,
                     posterior_hidden=16, posterior_layers=2,
                     n_flow=2, flow_hidden=16)
    return ADR2(cfg)


def _tiny_batch(B=2, N=6, T=48, Tref=100, device="cpu"):
    return {
        "phoneme_ids": torch.randint(1, 600, (B, N), device=device),
        "ref_mel": torch.randn(B, 80, Tref, device=device),
        "target_mel": torch.randn(B, 80, T, device=device),
        "target_mel_mask": torch.ones(B, T, dtype=torch.bool, device=device),
    }


class TestVAEComponents:
    def test_posterior_shapes(self):
        pe = PosteriorEncoder(n_mels=80, z_dim=16, hidden=32, n_layers=2)
        mu, log_std = pe(torch.randn(2, 80, 50))
        assert mu.shape == (2, 16, 50)
        assert log_std.shape == (2, 16, 50)
        assert (log_std.abs() <= 7.0).all(), "log_std 应被 clamp"
        z = pe.sample(mu, log_std)
        assert z.shape == mu.shape

    def test_flow_invertibility(self):
        """耦合 flow: inverse(forward(x)) ≈ x; 零初始化时 logdet≈0。"""
        flow = ResidualCouplingFlow(z_dim=8, hidden=16, cond_dim=4, n_layers=3)
        x = torch.randn(2, 8, 30)
        g = torch.randn(2, 4, 1)
        u, logdet = flow(x, g)
        x_rec = flow.inverse(u, g)
        assert torch.allclose(x, x_rec, atol=1e-4), \
            f"重构误差 {((x - x_rec).abs().max()):.2e}"
        # 零初始化 → s=0,t=0 → 各层恒等 (仅通道翻转), logdet≈0
        assert logdet.abs().max() < 1e-5

    def test_kl_gaussian(self):
        """同分布 KL=0; 偏移增大 KL 增大。"""
        mu = torch.zeros(1, 4, 10)
        ls = torch.zeros(1, 4, 10)
        assert kl_gaussian(mu, ls, mu, ls).item() < 1e-6
        mu2 = mu + 1.0
        kl = kl_gaussian(mu, ls, mu2, ls)
        assert kl.item() > 0.5

    def test_kl_mask(self):
        """mask 应排除 padding 帧。"""
        mu_q = torch.zeros(1, 2, 4)
        ls_q = torch.zeros(1, 2, 4)
        mu_p = torch.zeros(1, 2, 4)
        ls_p = torch.zeros(1, 2, 4)
        mu_p[0, :, 3] = 100.0  # padding 帧巨大偏移
        mask = torch.tensor([[True, True, True, False]])
        kl = kl_gaussian(mu_q, ls_q, mu_p, ls_p, mask)
        assert kl.item() < 1e-6, "padding 帧不应贡献 KL"


class TestADR2:
    def test_forward_backward(self):
        """前向产出全部损失且可反向传播。"""
        model = _tiny_adr2()
        out = model(_tiny_batch())
        for k in ("loss", "recon_loss", "kl_loss", "dur_loss"):
            assert torch.isfinite(out[k]), f"{k} 非有限值"
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)

    def test_mas_durations_cover_frames(self):
        """MAS 时长总和 == mel 帧数, 每音素 >=1。"""
        model = _tiny_adr2().eval()
        batch = _tiny_batch()
        out = model(batch)
        dur = out["mas_durations"]
        assert (dur.sum(dim=1) == batch["target_mel"].size(-1)).all()
        assert (dur >= 1).all()

    def test_sample_shape_and_finiteness(self):
        model = _tiny_adr2().eval()
        ids = torch.tensor([[10, 20, 30, 40]])
        ref = torch.randn(1, 80, 120)
        mel = model.sample(ids, ref)
        assert mel.dim() == 3 and mel.size(0) == 1 and mel.size(1) == 80
        assert torch.isfinite(mel).all()

    def test_sample_short_text_no_crash(self):
        """短音素 (长度 1) 不崩 (conv kernel 兜底)。"""
        model = _tiny_adr2().eval()
        mel = model.sample(torch.tensor([[5]]), torch.randn(1, 80, 100))
        assert mel.dim() == 3

    def test_capabilities(self):
        m = _tiny_adr2()
        assert m.supports(BackboneCapability.TTS)
        assert m.supports(BackboneCapability.SVS)
        assert m.supports(BackboneCapability.F0_COND)
        assert m.supports(BackboneCapability.LORA_READY)

    def test_f0_changes_output(self):
        """M9.4: F0 旋律条件应改变采样输出 (SVS)。"""
        m = _tiny_adr2().eval()
        with torch.no_grad():
            for layer in m.mel_decoder.layers:  # 随机化 decoder 输出层模拟训练后
                layer.out.weight.normal_(0, 0.05)
        ids = torch.tensor([[10, 20, 30, 40]])
        ref = torch.randn(1, 80, 120)
        mel_no_f0 = m.sample(ids, ref, noise_scale=0.5)
        mel_f0 = m.sample(ids, ref, noise_scale=0.5,
                          f0=torch.rand(1, 200) * 300 + 100)
        assert mel_f0.dim() == 3
        assert not torch.allclose(mel_f0, mel_no_f0, atol=1e-5)

    def test_mode_embedding_changes_output(self):
        """M9.4: TTS/SVS 模式嵌入应改变输出。"""
        m = _tiny_adr2().eval()
        with torch.no_grad():
            # 模拟训练后状态: 零初始化的耦合层输出打散, 否则 cond 无影响
            for layer in m.mel_decoder.layers:
                layer.out.weight.normal_(0, 0.05)
            m.mode_emb.weight.normal_(0, 0.1)
        ids = torch.tensor([[10, 20, 30, 40]])
        ref = torch.randn(1, 80, 120)
        torch.manual_seed(7)
        mel_tts = m.sample(ids, ref, mode="tts")
        torch.manual_seed(7)  # 固定噪声只留 mode 差异
        mel_svs = m.sample(ids, ref, mode="svs")
        assert not torch.allclose(mel_tts, mel_svs, atol=1e-4)

    def test_forward_with_mode_id(self):
        """M9.4: batch 可指定 mode_id。"""
        m = _tiny_adr2()
        batch = _tiny_batch()
        batch["f0"] = torch.rand(2, 48) * 300
        batch["mode_id"] = torch.ones(2, dtype=torch.long)
        out = m(batch)
        assert torch.isfinite(out["loss"])

    def test_lora_compatible(self):
        """LoRA 能注入 ADR2 (Linear/MHA 矩阵充足)。"""
        from adr.training.lora import LoRALinear, LoRAConfig, apply_lora
        model = _tiny_adr2()
        model = apply_lora(model, LoRAConfig(
            rank=2, target_modules=["linear1", "linear2", "out_proj",
                                    "prior_proj", "dur_fuse"]))
        n = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
        assert n > 0, "LoRA 未注入任何层"

    def test_param_budget(self):
        """medium 配置参数量应在预算内 (≤80M)。"""
        model = ADR2(ADR2Config())
        n = model.num_parameters()
        assert n < 80e6, f"参数量超预算: {n/1e6:.1f}M"
