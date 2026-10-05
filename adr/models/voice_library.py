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


def _trim_silence(wav, sr: int, top_db: float = 30.0):
    """首尾静音裁剪 (纯 numpy, 同 librosa.effects.trim 思路, 免重导入)。"""
    import numpy as np
    frame = max(1, int(0.025 * sr))
    n = len(wav) // frame
    if n < 4:
        return wav
    db = 20 * np.log10(
        np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-8)
    loud = np.where(db > db.max() - top_db)[0]
    if len(loud) == 0:
        return wav
    return wav[loud[0] * frame: (loud[-1] + 1) * frame]


def _materialize_ref(src: str, dst: Path) -> dict:
    """ref 落盘 + 时长兜底 (批次18)。

    GSV 合成硬限制 ref 3~10s (TTS.py:815)。此前入库只 copy 不校验,
    超长 ref 直接产出"建档即踩坑"档案 (批次17 事故根源)。
    - ≤10s: 原样 copy
    - >10s: 去首尾静音后取前 8s (留安全边) 重采样写 PCM_16
    - <3s / 无法读取: 报错拒绝 (音频内容不足, 无法自动修复)
    返回 {"action": "copy"|"clipped", "duration_s", "orig_s"?}。
    """
    import numpy as np
    import soundfile as sf
    try:
        info = sf.info(src)
    except Exception as e:
        raise ValueError(f"参考音频无法读取 (损坏或格式不支持): {e}") from e
    dur = info.frames / info.samplerate
    if dur < 3.0:
        raise ValueError(f"参考音频仅 {dur:.1f}s, 至少需要 3s 干净人声")
    if dur <= 10.0:
        shutil.copy2(src, dst)
        return {"action": "copy", "duration_s": round(dur, 2)}

    wav, sr = sf.read(src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = _trim_silence(wav, sr)[: int(8.0 * sr)]
    if len(wav) < int(3.0 * sr):
        # 极端: 去静音后不足 3s (几乎全静音), 兜底取原始前 8s
        wav, sr = sf.read(src, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav[: int(8.0 * sr)]
    sf.write(str(dst), wav, sr, subtype="PCM_16")
    print(f"[voice] ref 超长已自动裁剪: {dur:.1f}s -> {len(wav)/sr:.1f}s "
          f"(GSV 合成限 3~10s)", flush=True)
    return {"action": "clipped", "duration_s": round(len(wav) / sr, 2),
            "orig_s": round(dur, 2)}


def save_voice(
    name: str,
    ref_audio: str,
    prompt_text: str = "",
    t2s_weights: Optional[str] = None,
    vits_weights: Optional[str] = None,
    rvc_weights: Optional[str] = None,   # D1: RVC 转换模型 (.pth 文件名)
    rvc_index: Optional[str] = None,     # D1: faiss 索引路径
    style: str = "",                     # 批次7: 默认说话风格/人设描述
) -> Path:
    """保存音色档案, 返回档案目录。ref 超长自动裁剪 (见 _materialize_ref)。"""
    name = name.strip().replace("/", "_").replace("\\", "_")
    if not name:
        raise ValueError("音色名不能为空")
    vdir = VOICES_DIR / name
    vdir.mkdir(parents=True, exist_ok=True)
    dst = vdir / "ref.wav"
    _materialize_ref(ref_audio, dst)
    # sampling 属档案级手工调优项 (批次20): 重建档案时保留, 不被覆盖
    prev_sampling = None
    try:
        prev_sampling = json.loads(
            (vdir / "meta.json").read_text(encoding="utf-8")).get("sampling")
    except Exception:
        pass
    meta = {
        "name": name,
        "prompt_text": prompt_text,
        "t2s_weights": t2s_weights,
        "vits_weights": vits_weights,
        "rvc_weights": rvc_weights,
        "rvc_index": rvc_index,
        "style": style,
        "sampling": prev_sampling,
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

# 节奏过滤阈值: 静音占比过高 / 长停顿过多 / 尾音塌陷的段不进克隆打分
_RHYTHM_MAX_SILENCE = 0.35   # 帧级静音占比上限
_RHYTHM_MAX_PAUSES = 2       # ≥200ms 停顿次数上限
_RHYTHM_MIN_TAIL_DB = -10.0  # 尾部 400ms 相对全段均值的最小 dB


def _seg_rhythm(seg, sr: int) -> dict:
    """段级节奏特征 (纯 numpy, 毫秒级): 静音占比 / 长停顿数 / 尾音相对能量。

    选段加节奏过滤的依据: 参考段若本身破碎 (多次停顿) 或尾音塌陷,
    克隆输出会逐句模仿该节奏 — 用户听感"顿挫"的根源之一。
    """
    import numpy as np
    frame = int(0.025 * sr)
    n = len(seg) // frame
    if n < 10:
        return {"silence_ratio": 1.0, "pauses": 99, "tail_db": -99.0}
    frames = seg[: n * frame].reshape(n, frame)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-8)
    sil = db < -40.0
    min_pause = max(1, int(0.2 / 0.025))
    pauses = run = 0
    for s in sil:
        run = run + 1 if s else 0
        if run == min_pause:
            pauses += 1
    tail_n = min(n, max(1, int(0.4 / 0.025)))
    tail_db = float(db[-tail_n:].mean() - db.mean())
    return {"silence_ratio": round(float(sil.mean()), 3),
            "pauses": pauses, "tail_db": round(tail_db, 1)}


def _rhythm_ok(r: dict) -> bool:
    return (r["silence_ratio"] <= _RHYTHM_MAX_SILENCE
            and r["pauses"] <= _RHYTHM_MAX_PAUSES
            and r["tail_db"] >= _RHYTHM_MIN_TAIL_DB)


def save_voice_auto(
    name: str,
    audio_path: str,
    prompt_text: str = "",
    t2s_weights: Optional[str] = None,
    vits_weights: Optional[str] = None,
    progress=None,
    style: str = "",
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
        vdir = save_voice(name, audio_path, prompt_text, t2s_weights, vits_weights,
                          style=style)
        return vdir, {"mode": "whole", "duration_s": round(dur, 1)}

    from adr.eval.speaker_sim import similarity
    from adr.models.gsv_engine import get_gsv_engine
    eng = get_gsv_engine()

    starts = np.linspace(2, dur - _PROBE_LEN_S - 2, _SWEEP_COUNT)
    tmp_dir = VOICES_DIR / "_sweep_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    best = (-1.0, None)
    best_any = (-1.0, None)   # 全部被节奏过滤时的兜底
    for i, st in enumerate(starts):
        seg = wav[int(st * sr):int((st + _PROBE_LEN_S) * sr)]
        rhythm = _seg_rhythm(seg, sr)
        seg_path = tmp_dir / f"seg_{i}.wav"
        sf.write(str(seg_path), seg, sr)
        clone, osr = eng.synthesize(_PROBE_TEXT, str(seg_path))
        clone_path = tmp_dir / f"seg_{i}_clone.wav"
        sf.write(str(clone_path), clone, osr)
        sim = similarity(str(seg_path), str(clone_path))
        if sim > best_any[0]:
            best_any = (sim, seg_path)
        ok = _rhythm_ok(rhythm)
        _log(f"段{i + 1}/{_SWEEP_COUNT} @{st:.0f}s: 相似度 {sim:.3f} "
             f"(静音{rhythm['silence_ratio']:.0%} 停顿{rhythm['pauses']} "
             f"尾音{rhythm['tail_db']:+.0f}dB {'✓' if ok else '✗节奏'})")
        if ok and sim > best[0]:
            best = (sim, seg_path)

    if best[1] is None:
        best = best_any
        _log("所有段未过节奏过滤, 取相似度最高段兜底")
    sim, best_seg = best
    vdir = save_voice(name, str(best_seg), prompt_text, t2s_weights, vits_weights,
                      style=style)
    for f in tmp_dir.glob("*"):
        f.unlink(missing_ok=True)
    _log(f"建档完成, 最优段相似度 {sim:.3f}")
    return vdir, {"mode": "auto-sweep", "best_sim": round(sim, 3),
                  "duration_s": round(dur, 1), "segments": _SWEEP_COUNT}
