"""音频编码工具 (批次11): 与 api_neko audio_utils.py 字节级对齐的移植。

GSV api_v2 流式协议的字节格式依赖这里的行为:
- pack_wav: 标准库 wave, 单声道 s16le
- pack_ogg / pack_aac: ffmpeg subprocess 转码 (无需 libsndfile)
- pack_raw: 裸 PCM 字节
- wave_header_chunk: 44 字节 WAV 头 (流式 WAV 首块)

ADR 差异: 引擎输出 float32 [-1,1], GSV 字节流是 s16le — 统一经
to_int16() 转换后再打包。
"""
from __future__ import annotations

import shutil
import subprocess
import wave
from contextlib import suppress
from functools import lru_cache
from io import BytesIO

import numpy as np

from adr.core.exceptions import DependencyMissingError
from adr.core.settings import REPO_ROOT
from adr.server.param_norm import env_num


@lru_cache(maxsize=1)
def _resolve_ffmpeg() -> str:
    """定位 ffmpeg 可执行文件 (批次41b)。

    查找顺序: PATH → 项目 bundled 目录 (third_party/gpt_sovits/, 与
    core/doctor.py 的查找口径一致)。都找不到时抛出带安装指引的
    DependencyMissingError (批次84: 依赖缺失类错误, HTTP 面映射 503 而非
    400), 而不是 subprocess 的裸 FileNotFoundError traceback。
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    bundled = REPO_ROOT / "third_party" / "gpt_sovits" / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)
    raise DependencyMissingError(
        "未找到 ffmpeg — OGG/MP3/AAC 音频编码依赖它。解决方式 (任选其一):\n"
        "  1. 安装 ffmpeg 并加入 PATH (https://ffmpeg.org, 或 winget install ffmpeg);\n"
        "  2. 将 ffmpeg.exe 放到项目 third_party/gpt_sovits/ 目录下。\n"
        "若只需 WAV/裸 PCM 输出, 可改用 audio/wav 或 audio/raw 格式 (不依赖 ffmpeg)。"
    )


def _check_ffmpeg(returncode: int, err: bytes) -> None:
    """批次66: ffmpeg 非 0 退出时抛错, 不把空/半截音频当成功返回。"""
    if returncode != 0:
        tail = (err or b"").decode("utf-8", "replace").strip()[-500:]
        raise RuntimeError(f"ffmpeg 编码失败 (exit={returncode}): {tail}")


def _encode_timeout() -> float:
    """批次78 (#14): ffmpeg 转码超时秒数 (ADR_FFMPEG_TIMEOUT_SEC, 默认 120)。"""
    return env_num("ADR_FFMPEG_TIMEOUT_SEC", 120, float)


def _run_ffmpeg(args: list[str], pcm: bytes) -> bytes:
    """起 ffmpeg 按管道转码, 带超时强杀 (批次78 / CODE_REVIEW #14)。

    无超时保护时, 一次卡死的 ffmpeg 会永久占住线程池线程; 累积到线程池耗尽后
    所有请求排队。超时后 ``kill()`` 并回收子进程, 再抛可读错误。
    """
    timeout = _encode_timeout()
    process = subprocess.Popen(
        [_resolve_ffmpeg(), *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = process.communicate(input=pcm, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        with suppress(subprocess.TimeoutExpired, OSError):
            process.communicate(timeout=5)   # 回收管道/句柄, 不留僵尸
        raise RuntimeError(
            f"ffmpeg 编码超时 (>{timeout:.0f}s), 已终止子进程"
            " — 可用 ADR_FFMPEG_TIMEOUT_SEC 调整") from None
    _check_ffmpeg(process.returncode, err)
    return out


def to_int16(data: np.ndarray) -> np.ndarray:
    """float [-1,1] → int16 (clip 防爆音, 与 api_neko 打包逻辑一致)。"""
    if data.dtype == np.int16:
        return data
    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    """编码为 OGG (ffmpeg libvorbis q4)。"""
    io_buffer.write(_run_ffmpeg([
        "-f", "s16le",
        "-ar", str(rate),
        "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "libvorbis",
        "-q:a", "4",
        "-f", "ogg",
        "pipe:1",
    ], to_int16(data).tobytes()))
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    """裸 s16le PCM 字节 (调用方负责先转 int16)。"""
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    """编码为 WAV (标准库 wave, 单声道 16bit)。

    批次62 分发侧: 编码完成后尾部追加 LIST/INFO 标注 AI 生成
    (inject_wav_info); 流式首块 wave_header_chunk 不注入。
    """
    data = to_int16(data)
    with wave.open(io_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())
    # 延迟导入: utils.audio 顶部拉 torch/torchaudio, 避免拖慢本模块导入
    from adr.utils.audio import inject_wav_info
    raw = io_buffer.getvalue()
    out = inject_wav_info(raw)
    if out != raw:
        io_buffer.seek(0)
        io_buffer.truncate(0)
        io_buffer.write(out)
    return io_buffer


def pack_mp3(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    """编码为 MP3 (ffmpeg libmp3lame, 192k) — 批次21 OpenAI 兼容面默认格式。"""
    io_buffer.write(_run_ffmpeg([
        "-f", "s16le",
        "-ar", str(rate),
        "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        "-f", "mp3",
        "pipe:1",
    ], to_int16(data).tobytes()))
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    """编码为 AAC/ADTS (ffmpeg, 192k)。"""
    io_buffer.write(_run_ffmpeg([
        "-f", "s16le",
        "-ar", str(rate),
        "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "aac",
        "-b:a", "192k",
        "-vn",
        "-f", "adts",
        "pipe:1",
    ], to_int16(data).tobytes()))
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str) -> BytesIO:
    """按 media_type 分发编码, 缓冲区指针归零。"""
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "mp3":
        io_buffer = pack_mp3(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def wave_header_chunk(frame_input: bytes = b"", channels: int = 1,
                      sample_width: int = 2, sample_rate: int = 32000) -> bytes:
    """生成 44 字节 WAV 文件头 (流式 WAV 输出的首个 chunk)。

    客户端按 "头 + 裸 s16le PCM" 拼接播放 — GSV api_v2 流式契约。
    """
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()
