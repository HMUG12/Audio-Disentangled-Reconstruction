"""批次 51: 探测集收敛模块 (adr.eval.probes) 单测 — 不依赖 GSV/声纹模型。"""
import numpy as np
import soundfile as sf


def test_probes_shape():
    from adr.eval.probes import PROBES
    assert len(PROBES) == 5
    assert all(isinstance(s, str) and s.strip() for s in PROBES)
    assert len(set(PROBES)) == 5  # 无重复句


def test_pretrained_constants_pinned():
    from adr.eval.probes import PRE_S2G, PRE_T2S
    # 钉死官方预训练: 路径必须落在 gsv-v2final-pretrained 下 (防 tts_infer.yaml 漂移)
    assert "gsv-v2final-pretrained" in PRE_T2S and PRE_T2S.endswith(".ckpt")
    assert "gsv-v2final-pretrained" in PRE_S2G and PRE_S2G.endswith("s2G2333k.pth")


def test_clip_ref_short_passthrough(tmp_path):
    from adr.eval.probes import clip_ref
    p = tmp_path / "short.wav"
    sf.write(str(p), np.zeros(16000 * 5, dtype="float32"), 16000)  # 5s ≤ 10s
    assert clip_ref(str(p), "t_short") == str(p)  # 原样返回, 不产生裁剪文件


def test_clip_ref_long_trims_to_8s(tmp_path, monkeypatch):
    import adr.eval.probes as probes_mod
    from adr.eval.probes import clip_ref
    # 输出目录指到 tmp, 避免污染仓库 output/
    monkeypatch.setattr(probes_mod, "REPO_ROOT", tmp_path)
    p = tmp_path / "long.wav"
    sr = 16000
    sf.write(str(p), np.zeros(sr * 14, dtype="float32"), sr)  # 14s > 10s
    out = clip_ref(str(p), "t_long")
    assert out != str(p)
    y, o_sr = sf.read(out)
    assert o_sr == sr and len(y) / o_sr <= 8.01  # 取前 8s


def test_score_probes_uses_same_ref_and_counts(monkeypatch, tmp_path):
    import adr.eval.probes as probes_mod
    from adr.eval.probes import score_probes
    monkeypatch.setattr(probes_mod, "REPO_ROOT", tmp_path)

    seen = []

    class _Eng:
        def synthesize(self, text, ref, t2s_weights=None, vits_weights=None):
            seen.append((text, t2s_weights, vits_weights))
            return np.zeros(1600, dtype="float32"), 16000

    monkeypatch.setattr("adr.eval.speaker_sim.similarity",
                        lambda ref, test: 0.7)

    mean, sims = score_probes(_Eng(), "ref.wav", vits="zeroshot")
    assert len(sims) == 5 and mean == 0.7
    assert all(t2s.endswith(".ckpt") for _, t2s, _ in seen)  # s1 钉死官方预训练
    assert all(v == "zeroshot" for _, _, v in seen)
