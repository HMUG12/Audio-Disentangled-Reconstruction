"""评估探测集 — 长度分布全覆盖的探测文本 + 克隆打分/参考音频裁剪助手。

scripts/train_gate.py (训练门禁) 与 scripts/eval_secs.py (档案评估) 共用,
消除此前 train_gate / eval_heldout 各自复制一份 PROBES 的双源漂移风险。
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GSV_DIR = REPO_ROOT / "third_party" / "gpt_sovits"

# 长/中/短/混合(数字+英文)/叙事 全覆盖 — 单句探测方差大 (短句天然弱,
# 波动 ±0.03+), 均值更稳; 基线与被测条件必须同句集, 否则分数不可比。
PROBES = [
    "人工智能技术正在以前所未有的速度发展，语音合成系统已经能够生成接近真人水平的自然流畅音频。",
    "今天早上我喝了一杯咖啡，然后在公园里散了一会儿步，天气非常不错。",
    "您好，请问有什么可以帮您的吗？",
    "这是一段用于测试声音克隆效果的文本，包含数字一二三四五和英文 Hello World。",
    "夜色渐深，城市的灯火次第亮起，街道上行人渐渐稀少。",
]

# 显式钉死官方预训练 s1/s2: tts_infer.yaml 的 custom 段可能被外部改写指向
# 微调权重 (实测发生过), 隐式默认会让"零样本基线"和打分条件漂移。
PRE_T2S = str(GSV_DIR / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
              / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
PRE_S2G = str(GSV_DIR / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
              / "s2G2333k.pth")


def clip_ref(ref: str, tag: str) -> str:
    """参考音频超 10s (GSV 合成硬限制 3~10s) 时自动裁剪: 去首尾静音后取前 8s。

    实测事故: 控制台把完整训练录音 (--ref output/uploads/xxx) 直接传进
    打分流程, 零样本基线合成即抛 "参考音频在3~10秒范围外" → 进程崩。
    基线与被测条件用同一裁剪 ref, 可比性不变。
    输出 output/clip_ref_<tag>.wav; ≤10s 原样返回。
    """
    import librosa
    import soundfile as sf

    y, sr = librosa.load(ref, sr=None, mono=True)
    if len(y) / sr <= 10.0:
        return ref
    y, _ = librosa.effects.trim(y, top_db=30)          # 去首尾静音
    y = y[: int(8.0 * sr)]                              # 8s < 10s 上限, 留安全边
    out = REPO_ROOT / "output" / f"clip_ref_{tag}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, sr)
    print(f"[probes] ref 超长已裁剪: {Path(ref).name} -> {out.name} (8s, 去静音)", flush=True)
    return str(out)


def _save_tmp(wav, sr) -> Path:
    import soundfile as sf
    p = REPO_ROOT / "output" / "_probe_clone.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), wav, sr)
    return p


def score_probes(eng, ref: str, vits: str = PRE_S2G, t2s: str = PRE_T2S,
                 sents=None):
    """逐句克隆打分, 返回 (均值, 逐句分列表)。

    s1 固定用官方预训练 (与 --skip-s1 训练条件一致), s2 传被测权重 —
    不传任何权重 (PRE_S2G) 时即真零样本基线。
    sents: 自定义句集 — 基线与训练后必须同句集, 否则分数不可比。
    """
    from adr.eval.speaker_sim import similarity
    sims = []
    for text in (sents or PROBES):
        wav, sr = eng.synthesize(text, ref, t2s_weights=t2s, vits_weights=vits)
        sims.append(similarity(ref, str(_save_tmp(wav, sr))))
    return sum(sims) / len(sims), sims
