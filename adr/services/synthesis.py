"""单一合成服务层 (批次34): 档案解析 / 参数校验 / 引擎调用 / 线格式 收口。

v2_compat / v3_compat / webui 三面此前各自内联同构逻辑, 本模块收为单实现:

- 档案解析+回填: v2 _resolve_profile / v3 _resolve_voice / webui 两处内联 load_voice
- 参数校验: v2 _check_params
- split_method 兜底: 流式 cut3 (v2/v3/webui) + 非流式 cut1 (v2/webui) 共 5 处
- 流式线格式: v2 _stream_generator (wav 首块 44B 头) / v3 _produce (逐帧完整 WAV)

错误消息逐字保留 (wire 兼容): "invalid profile name: X" / "unknown profile: X" /
"text is required" / "ref_audio_path is required (...)" / "unsupported media_type: ..."。

延迟导入说明: adr.server.__init__ → app → v2_compat 构成重链, 本模块若在模块级
import adr.server.pathsafe / adr.server.audio_codec, 则测试侧
`from adr.services.synthesis import ...` 会沿 adr.server.__init__ 触发
v2_compat 循环导入; 故这两个 adr.server 子模块在函数体内延迟导入
(它们自身无反向依赖, 延迟仅为绕开包 __init__ 链)。
"""
from io import BytesIO
from typing import Iterator

from adr.core.exceptions import (
    ProfileInvalidError,
    ProfileNotFoundError,
    SynthesisParamsError,
)
from adr.models import voice_library

# wire 白名单 (批次34 从 v2_compat 收口; 顺序即错误消息中的枚举顺序)
MEDIA_TYPES = ("wav", "mp3", "raw", "ogg", "aac")


class SynthesisService:
    """无状态合成编排 (全 staticmethod; engine 显式传参, 便于测试注入)。"""

    # ── 档案解析 ──

    @staticmethod
    def load_profile(name: str) -> dict:
        """档案名 → meta dict; 非法名/不存在 → ProfileInvalidError/ProfileNotFoundError。"""
        from adr.server import pathsafe  # 延迟导入, 见模块 docstring

        if not pathsafe.is_safe_name(name):
            raise ProfileInvalidError(f"invalid profile name: {name}")
        try:
            return voice_library.load_voice(name)
        except Exception:
            raise ProfileNotFoundError(f"unknown profile: {name}")

    @staticmethod
    def profile_fill(
        meta: dict, *, ref_audio_path=None, prompt_text="",
        t2s_weights=None, vits_weights=None,
    ) -> dict:
        """请求显式值优先, 档案值兜底。

        语义镜像 v2 _resolve_profile: ref/prompt 用 falsy 判定 (显式空串也兜底),
        weights 用 is None 判定 (显式空串不兜底)。
        """
        return {
            "ref_audio_path": ref_audio_path or meta["ref_audio"],
            "prompt_text": prompt_text or meta.get("prompt_text") or "",
            "t2s_weights": (t2s_weights if t2s_weights is not None
                            else meta.get("t2s_weights")),
            "vits_weights": (vits_weights if vits_weights is not None
                             else meta.get("vits_weights")),
        }

    # ── 参数校验 ──

    @staticmethod
    def validate_params(text, ref_audio_path, media_type="wav") -> str:
        """必填项 + media_type 白名单; 返回 lower 后的 media_type。"""
        if not text:
            raise SynthesisParamsError("text is required")
        if not ref_audio_path:
            raise SynthesisParamsError(
                "ref_audio_path is required "
                "(or pass profile / set ADR_TTS_DEFAULT_PROFILE)")
        mt = (media_type or "wav").lower()
        if mt not in MEDIA_TYPES:
            raise SynthesisParamsError(
                f"unsupported media_type: {mt}, "
                f"must be one of wav/mp3/raw/ogg/aac")
        return mt

    # ── 引擎调用 (split_method 兜底单点: 流式 cut3 / 非流式 cut1) ──
    # 注意: ADR 引擎 yield (chunk, sr) — 与 GSV pipeline 的 (sr, chunk) 相反

    @staticmethod
    def stream_chunks(engine, text, ref_audio_path, *, prompt_text="",
                      text_lang="zh", prompt_lang="zh", t2s_weights=None,
                      vits_weights=None, split_method=None, head_seed=-1,
                      top_k=15, top_p=1.0, temperature=1.0, speed_factor=1.0,
                      fragment_interval=None) -> Iterator:
        """流式合成: 透传引擎 synthesize_stream, yield (chunk, sr)。"""
        return engine.synthesize_stream(
            text, ref_audio_path,
            prompt_text=prompt_text or "",
            text_lang=text_lang or "zh",
            prompt_lang=prompt_lang or "zh",
            t2s_weights=t2s_weights,
            vits_weights=vits_weights,
            split_method=split_method or "cut3",
            head_seed=head_seed,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed_factor=speed_factor,
            fragment_interval=fragment_interval,
        )

    @staticmethod
    def synthesize_once(engine, text, ref_audio_path, *, prompt_text="",
                        text_lang="zh", prompt_lang="zh", t2s_weights=None,
                        vits_weights=None, split_method=None, seed=-1,
                        top_k=15, top_p=1.0, temperature=1.0, speed_factor=1.0,
                        fragment_interval=None):
        """非流式合成: 透传引擎 synthesize, 返回 (wav, sr)。"""
        return engine.synthesize(
            text, ref_audio_path,
            prompt_text=prompt_text or "",
            text_lang=text_lang or "zh",
            prompt_lang=prompt_lang or "zh",
            speed_factor=speed_factor,
            seed=seed,
            t2s_weights=t2s_weights,
            vits_weights=vits_weights,
            split_method=split_method or "cut1",
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            fragment_interval=fragment_interval,
        )

    # ── 线格式 ──

    @staticmethod
    def stream_bytes(engine, text, ref_audio_path, *, media_type="wav",
                     **stream_kwargs) -> Iterator[bytes]:
        """GSV api_v2 流式字节契约: wav 首块 44B 头, 之后裸 s16le PCM 块。

        ogg/aac/mp3 逐块独立编码 (块边界即容器边界)。
        其余参数同 stream_chunks。
        """
        from adr.server.audio_codec import (  # 延迟导入, 见模块 docstring
            pack_audio, to_int16, wave_header_chunk)

        mt = media_type
        first = True
        for chunk, sr in SynthesisService.stream_chunks(
                engine, text, ref_audio_path, **stream_kwargs):
            if first and mt == "wav":
                yield wave_header_chunk(sample_rate=sr)
                mt = "raw"
                first = False
            yield pack_audio(BytesIO(), to_int16(chunk), sr, mt).getvalue()

    @staticmethod
    def frame_bytes(chunk, sr) -> bytes:
        """v3 websocket 帧契约: 每句逐帧完整 WAV (44B 头 + s16le PCM)。"""
        from adr.server.audio_codec import (  # 延迟导入, 见模块 docstring
            to_int16, wave_header_chunk)

        return wave_header_chunk(sample_rate=sr) + to_int16(chunk).tobytes()
