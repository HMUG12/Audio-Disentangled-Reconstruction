"""音色档案库: 录一次 → 存成命名音色 → 永久复用 (快速克隆的持久化层)。

档案结构: data/voices/<name>/
    ref.wav        参考音频 (3-10 秒干净人声)
    meta.json      {prompt_text, t2s_weights, vits_weights, created_at}

权重字段可指向微调产物 (如 GPT_weights_v2/user_voice-e8.ckpt),
实现"零样本即用 + 微调升级"两档持久化。
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VOICES_DIR = REPO_ROOT / "data" / "voices"


def save_voice(
    name: str,
    ref_audio: str,
    prompt_text: str = "",
    t2s_weights: Optional[str] = None,
    vits_weights: Optional[str] = None,
    rvc_weights: Optional[str] = None,   # D1: RVC 转换模型 (.pth 文件名)
    rvc_index: Optional[str] = None,     # D1: faiss 索引路径
) -> Path:
    """保存音色档案, 返回档案目录。"""
    name = name.strip().replace("/", "_").replace("\\", "_")
    if not name:
        raise ValueError("音色名不能为空")
    vdir = VOICES_DIR / name
    vdir.mkdir(parents=True, exist_ok=True)
    dst = vdir / "ref.wav"
    shutil.copy2(ref_audio, dst)
    meta = {
        "name": name,
        "prompt_text": prompt_text,
        "t2s_weights": t2s_weights,
        "vits_weights": vits_weights,
        "rvc_weights": rvc_weights,
        "rvc_index": rvc_index,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (vdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return vdir


def list_voices() -> list[str]:
    """全部音色名。"""
    if not VOICES_DIR.exists():
        return []
    return sorted(d.name for d in VOICES_DIR.iterdir()
                  if (d / "meta.json").exists() and (d / "ref.wav").exists())


def load_voice(name: str) -> dict:
    """加载音色档案 → {ref_audio, prompt_text, t2s_weights, vits_weights}。"""
    vdir = VOICES_DIR / name
    meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    meta["ref_audio"] = str(vdir / "ref.wav")
    return meta


_PROBE_TEXT = "你好,这是我的声音克隆测试,希望听起来像我。"
_PROBE_LEN_S = 7.0     # 参考段长度
_SWEEP_COUNT = 6       # 扫段数


def save_voice_auto(
    name: str,
    audio_path: str,
    prompt_text: str = "",
    t2s_weights: Optional[str] = None,
    vits_weights: Optional[str] = None,
    progress=None,
):
    """A2: 长音频自动选段建档。

    音频 >15s 时: 均分扫 6 个 7s 段 → 各做零样本探测克隆 → ERes2Net 打分
    → 以最优段建档。短音频直接整段建档。

    progress: 可选回调 fn(msg: str) 用于 WebUI 进度展示。
    返回 (vdir, info_dict)。
    """
    import numpy as np
    import soundfile as sf

    def _log(msg):
        if progress:
            progress(msg)

    wav, sr = sf.read(str(audio_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    dur = len(wav) / sr

    if dur <= 15:
        _log(f"音频 {dur:.1f}s ≤15s, 直接整段建档")
        vdir = save_voice(name, audio_path, prompt_text, t2s_weights, vits_weights)
        return vdir, {"mode": "whole", "duration_s": round(dur, 1)}

    from adr.eval.speaker_sim import similarity
    from adr.models.gsv_engine import get_gsv_engine
    eng = get_gsv_engine()

    starts = np.linspace(2, dur - _PROBE_LEN_S - 2, _SWEEP_COUNT)
    tmp_dir = VOICES_DIR / "_sweep_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    best = (-1.0, None)
    for i, st in enumerate(starts):
        seg = wav[int(st * sr):int((st + _PROBE_LEN_S) * sr)]
        seg_path = tmp_dir / f"seg_{i}.wav"
        sf.write(str(seg_path), seg, sr)
        clone, osr = eng.synthesize(_PROBE_TEXT, str(seg_path))
        clone_path = tmp_dir / f"seg_{i}_clone.wav"
        sf.write(str(clone_path), clone, osr)
        sim = similarity(str(seg_path), str(clone_path))
        _log(f"段{i + 1}/{_SWEEP_COUNT} @{st:.0f}s: 相似度 {sim:.3f}")
        if sim > best[0]:
            best = (sim, seg_path)

    sim, best_seg = best
    vdir = save_voice(name, str(best_seg), prompt_text, t2s_weights, vits_weights)
    for f in tmp_dir.glob("*"):
        f.unlink(missing_ok=True)
    _log(f"建档完成, 最优段相似度 {sim:.3f}")
    return vdir, {"mode": "auto-sweep", "best_sim": round(sim, 3),
                  "duration_s": round(dur, 1), "segments": _SWEEP_COUNT}
