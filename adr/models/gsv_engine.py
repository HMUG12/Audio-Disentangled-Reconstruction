"""GPT-SoVITS 官方预训练套壳引擎 (保底路线 - 说话克隆腿)。

定位: 零样本音色克隆 TTS 基线 (5 秒参考音频 → 克隆朗读)。
与 DiffSinger 引擎互补: 那个管唱歌, 这个管说话。

权重 (HF lj1995/GPT-SoVITS):
- s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt (GPT 语义)
- s2G488k.pth / s2D488k.pth (VITS 声学)
- chinese-hubert-base / chinese-roberta-wwm-ext-large (特征)

设计注记 (不能完全抄, 快速克隆优势不能丢):
- 引擎只做推理保底; 数据预处理 / QLoRA 微调 / WebUI 流程仍是 ADR 自己的
- GPT-SoVITS 官方支持 s1/s2 微调, 后续可由 ADR 训练管线接管定制音色
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GSV_DIR = REPO_ROOT / "third_party" / "gpt_sovits"


@dataclass
class GSVEngineConfig:
    """GPT-SoVITS 套壳引擎配置。"""
    gsv_dir: Path = GSV_DIR
    version: str = "v2"           # v1/v2 权重已下载; v3/v4 需另下
    device: str = "auto"
    half: bool = True             # 8GB 卡半精度


class GSVEngine:
    """GPT-SoVITS 推理引擎 (懒加载 + 锁 + cwd 上下文, 同 DiffSingerEngine 模式)。"""

    def __init__(self, config: Optional[GSVEngineConfig] = None):
        self.config = config or GSVEngineConfig()
        self._tts = None
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._tts is not None

    @contextlib.contextmanager
    def _gsv_context(self):
        """进入 GSV 代码上下文 (sys.path + cwd, 其代码用相对路径找权重)。"""
        code_dir = str(self.config.gsv_dir)
        inner = str(self.config.gsv_dir / "GPT_SoVITS")
        old_cwd = os.getcwd()
        added = []
        for p in (code_dir, inner):
            if p not in sys.path:
                sys.path.insert(0, p)
                added.append(p)
        os.chdir(code_dir)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def _lazy_init(self):
        if self._tts is not None:
            return
        # 兼容补丁: NLTK (英文 G2P 依赖) 离线环境下载默认超时 72s/次,
        # 收紧到 5s 快速失败走内置回退发音
        import socket
        socket.setdefaulttimeout(5)
        with self._gsv_context():
            import torch
            from TTS_infer_pack.TTS import TTS, TTS_Config

            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            cfg = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
            cfg.device = device
            cfg.is_half = self.config.half and device == "cuda"
            cfg.version = self.config.version
            self._tts = TTS(cfg)

            # 兼容补丁: 新版 torchaudio.load 依赖 torchcodec (无 Windows 轮子),
            # 用 soundfile shim 替换 (返回 (C,T) float tensor, 与原版一致)
            import soundfile as _sf
            import torch as _torch
            import torchaudio as _ta

            def _sf_load(path, *a, **k):
                data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
                return _torch.from_numpy(data.T), sr

            _ta.load = _sf_load

    def warmup(self):
        """预热: 后台线程调用, 把权重加载从首次合成挪到启动期 (A1)。"""
        with self._lock:
            self._lazy_init()

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """前端符号清洗: '-'/'—'/'~' 在非英文单词内 -> ','(GSV 会把 '-' 读成"减");
        换行 -> ','。英文词内连字符 (GPT-SoVITS) 保留。"""
        import re
        # 任一侧是中文字符即视为分隔符; 两侧皆英文/数字 (GPT-SoVITS) 保留
        text = re.sub(r"(?<=[一-鿿])[-—–~]|[-—–~](?=[一-鿿])", "，", text)
        text = text.replace("\n", "，").replace("\r", "")
        return re.sub(r"，{2,}", "，", text).strip("，")

    def synthesize_stream(
        self,
        text: str,
        ref_audio: str,
        prompt_text: str = "",
        text_lang: str = "zh",
        prompt_lang: str = "zh",
        t2s_weights: Optional[str] = None,
        vits_weights: Optional[str] = None,
        split_method: str = "cut1",
    ):
        """流式合成: 逐块 yield (wav_chunk float32 [-1,1], sr)。

        官方 streaming_mode: 语义 token 分段解码, 首块延迟远小于整段。
        split_method: cut0 不切 / cut1 凑四句一切(默认, 句间停顿最短)
                      / cut2 凑50字一切 / cut3 按中文句号切 / cut5 按标点符号切
        """
        import numpy as np

        text = self._sanitize_text(text)
        ref_audio = str(Path(ref_audio).resolve())
        t2s_weights = str(Path(t2s_weights).resolve()) if t2s_weights else None
        vits_weights = str(Path(vits_weights).resolve()) if vits_weights else None
        with self._lock:
            self._lazy_init()
            inputs = {
                "text": text,
                "text_lang": text_lang,
                "ref_audio_path": ref_audio,
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang,
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "text_split_method": split_method,
                "streaming_mode": True,
                "parallel_infer": True,
            }
            with self._gsv_context():
                if vits_weights:
                    self._tts.init_vits_weights(vits_weights)
                if t2s_weights:
                    self._tts.init_t2s_weights(t2s_weights)
                for sr, chunk in self._tts.run(inputs):
                    chunk = np.asarray(chunk, dtype=np.float32)
                    if np.abs(chunk).max() > 1.5:
                        chunk = chunk / 32768.0
                    yield chunk, int(sr)

    def synthesize(
        self,
        text: str,
        ref_audio: str,
        prompt_text: str = "",
        text_lang: str = "zh",
        prompt_lang: str = "zh",
        speed_factor: float = 1.0,
        seed: int = -1,
        t2s_weights: Optional[str] = None,   # C2: 微调后的 s1 ckpt
        vits_weights: Optional[str] = None,  # C2: 微调后的 s2 pth
        split_method: str = "cut1",          # 切句方式, 见 synthesize_stream
    ) -> "tuple":
        """零样本克隆朗读: (text, 参考音频) → (wav, sr)。

        Args:
            text: 要朗读的文本
            ref_audio: 参考音频路径 (3-10 秒干净人声最佳)
            prompt_text: 参考音频的文本 (可空, 空则弱化文本条件)
            text_lang / prompt_lang: zh/en/ja/ko/yue 等
            speed_factor: 语速 (1.0 原速)
            seed: -1 随机
        """
        import numpy as np

        text = self._sanitize_text(text)
        # 绝对路径: _gsv_context 会 chdir, 相对路径会丢
        ref_audio = str(Path(ref_audio).resolve())
        t2s_weights = str(Path(t2s_weights).resolve()) if t2s_weights else None
        vits_weights = str(Path(vits_weights).resolve()) if vits_weights else None
        with self._lock:
            self._lazy_init()
            inputs = {
                "text": text,
                "text_lang": text_lang,
                "ref_audio_path": str(ref_audio),
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang,
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "speed_factor": speed_factor,
                "seed": seed,
                "text_split_method": split_method,
                "return_fragment": False,
                "streaming_mode": False,
                "parallel_infer": True,
            }
            with self._gsv_context():
                # C2: 微调权重热换 (官方 init_* 方法)
                if vits_weights:
                    self._tts.init_vits_weights(vits_weights)
                if t2s_weights:
                    self._tts.init_t2s_weights(t2s_weights)
                sr, audio = None, None
                for sr, audio in self._tts.run(inputs):
                    pass  # 非流式只 yield 一次完整音频
        if audio is None:
            raise RuntimeError("GPT-SoVITS 未返回音频")
        audio = np.asarray(audio, dtype=np.float32)
        # 官方返回 int16 量程 (rms ~1600), 归一化到 [-1,1]
        if np.abs(audio).max() > 1.5:
            audio = audio / 32768.0
        return audio, int(sr)


_ENGINE: Optional[GSVEngine] = None


def get_gsv_engine() -> GSVEngine:
    """进程级单例。"""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = GSVEngine()
    return _ENGINE
