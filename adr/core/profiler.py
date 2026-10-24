"""批次63 第四子批: 调度层 Profiler — TTS 微基准采样 (RTF/显存)。

v1.3.0 Autopilot 第一级: Profiler 采样 → Policy 配置档 (后者待 N≥2 实现)。
RTF = 耗时/音频时长 (<1 快于实时)。
纪要标定锚点: E5-2667 v3 RTF≈1.7 (CPU) / 4060Ti ≈2G (低配 GPU)。

输出: <data_dir>/profile_<id>.json = {profile_id, timestamp, device_info, profiles[]}
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """单测点结果."""
    name: str       # 测点名称 (如 "default")
    params: dict    # 测点参数快照 (graph=0, top_k=15, ...)
    rtf: float      # 实时率 (wall_sec / audio_sec, <1 即比实时快; 与纪要锚点同口径)
    mem_mb: int     # 峰值显存 MB (CPU 时为 0)
    notes: str = "" # 额外备注 (如 "3 runs median")


def _machine_id() -> str:
    """机器指纹 (脱敏): UUID namespace + MAC 取 sha256 截前 16 字符,
    不泄露设备型号/厂商."""
    try:
        mac = uuid.getnode()
        h = hashlib.sha256(f"{uuid.NAMESPACE_DNS}:{mac}".encode()).hexdigest()
        return h[:16]
    except Exception:
        return "unknown"


def _get_device_info() -> dict:
    """当前设备概要 (不带型号厂商完整名, 避免设备表耦合)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"type": "cpu"}
        # 只取算力 (如 8.9) + 显存 GB, 不取完整 device_name
        cap = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        return {
            "type": "cuda",
            "compute_capability": f"{cap[0]}.{cap[1]}",
            "total_mem_gb": round(props.total_memory / 2**30, 1),
        }
    except Exception:
        return {"type": "unknown"}


def _measure_rtf_mem(
    text: str,
    ref_audio: str,
    n_runs: int = 3,
    **synth_kw,
) -> tuple[float, int]:
    """测单点 RTF + 显存 (median of n_runs, 首次 warmup 不计)。

    Returns: (rtf, mem_mb)
    """
    from adr.models import gsv_engine
    import torch

    eng = gsv_engine.get_gsv_engine()
    is_cuda = torch.cuda.is_available()

    # warmup: 首次合成付 JIT 编译, 不计入统计
    try:
        _ = eng.synthesize(text, ref_audio, **synth_kw)
    except Exception as e:
        log.warning("warmup 失败 (不影响测量): %s", e)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(0)

    rtfs = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            audio, sr = eng.synthesize(text, ref_audio, **synth_kw)
            wall = time.perf_counter() - t0
            audio_sec = len(audio) / sr
            if audio_sec <= 0:
                log.error("测量失败: 合成音频为空")
                return 0.0, 0
            rtfs.append(wall / audio_sec)
        except Exception as e:
            log.error("测量失败: %s", e)
            return 0.0, 0

    mem_mb = 0
    if is_cuda:
        mem_mb = int(torch.cuda.max_memory_allocated(0) / 2**20)

    rtf = sorted(rtfs)[len(rtfs) // 2] if rtfs else 0.0
    return rtf, mem_mb


def run_profile(
    text: str | None = None,
    ref_audio: str | None = None,
    n_runs: int = 3,
) -> dict[str, Any]:
    """运行微基准采样, 返回完整 profile 字典 (未写盘).

    Args:
        text: 测试文本 (默认固定 15 汉字)
        ref_audio: 参考音频 (默认从档案库首个档案取, 无档案则抛异常)
        n_runs: 每测点运行次数 (取中位数)

    Returns:
        {profile_id, timestamp, device_info, profiles: [BenchmarkResult.as_dict()]}
    """
    if text is None:
        text = "青山横北郭,白水绕东城。此地一为别,孤蓬万里征。"
    if ref_audio is None:
        from adr.models.voice_library import list_voices, load_voice
        voices = list_voices()
        if not voices:
            raise ValueError("无音色档案, 无法自动选参考音频; 请传 ref_audio")
        prof = load_voice(voices[0])
        ref_audio = prof["ref_audio"]

    log.info("[Profiler] 开始采样: text=%s 字, ref=%s, n_runs=%d",
             len(text), Path(ref_audio).name, n_runs)

    results: list[BenchmarkResult] = []

    # 测点 1: 默认参数 (graph=0 批次10 因音质漂移默认关, static_kv=0, top_k=15)
    log.info("  [1/1] default (top_k=15, no_graph, no_static_kv)")
    rtf, mem = _measure_rtf_mem(text, ref_audio, n_runs, top_k=15)
    results.append(BenchmarkResult(
        name="default",
        params={"top_k": 15, "cudagraph": 0, "static_kv": 0},
        rtf=round(rtf, 3),
        mem_mb=mem,
        notes=f"{n_runs} runs median",
    ))

    # 未来可扩展更多测点 (graph=1, static_kv=1, buckets, 等 — 音质门通过后)

    return {
        "profile_id": _machine_id(),
        "timestamp": time.time(),
        "device_info": _get_device_info(),
        "profiles": [
            {"name": r.name, "params": r.params, "rtf": r.rtf,
             "mem_mb": r.mem_mb, "notes": r.notes}
            for r in results
        ],
    }


def save_profile(data: dict, output_path: Path | None = None) -> Path:
    """保存 profile JSON."""
    if output_path is None:
        from adr.core.config import adr_data_dir
        output_path = adr_data_dir() / f"profile_{data['profile_id']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("[Profiler] 已保存: %s", output_path)
    return output_path


def load_profile(path: Path | None = None) -> dict | None:
    """读取本机 profile (默认取 <data_dir>/profile_<mid>.json)."""
    if path is None:
        from adr.core.config import adr_data_dir
        mid = _machine_id()
        path = adr_data_dir() / f"profile_{mid}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
