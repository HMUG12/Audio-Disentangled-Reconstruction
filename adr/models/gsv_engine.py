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

    def warmup(self, vits_weights: Optional[str] = None,
               t2s_weights: Optional[str] = None):
        """预热: 后台线程调用, 把权重加载从首次合成挪到启动期 (A1)。

        传权重时额外热换到目标权重 — 否则首次流式调用要付 ~12s 的
        init_vits_weights 罚金 (实测 bench_stream_variance)。
        """
        with self._lock:
            self._lazy_init()
            if vits_weights or t2s_weights:
                with self._gsv_context():
                    if vits_weights:
                        self._tts.init_vits_weights(str(vits_weights))
                    if t2s_weights:
                        self._tts.init_t2s_weights(str(t2s_weights))

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """前端符号清洗: '-'/'—'/'~' 在非英文单词内 -> ','(GSV 会把 '-' 读成"减");
        换行 -> ','。英文词内连字符 (GPT-SoVITS) 保留。"""
        import re
        # 任一侧是中文字符即视为分隔符; 两侧皆英文/数字 (GPT-SoVITS) 保留
        text = re.sub(r"(?<=[一-鿿])[-—–~]|[-—–~](?=[一-鿿])", "，", text)
        text = text.replace("\n", "，").replace("\r", "")
        return re.sub(r"，{2,}", "，", text).strip("，")

    @staticmethod
    def _stream_head_split(text: str, max_head: int = 30) -> "tuple":
        """W1 首包优化: 从长文本切出短首段 (句末标点优先, 逗号次之)。

        实测 (output/bench_stream.json): 首块延迟 ∝ 首段长度, 45 字长句
        cut1 首包 5.3s / cut0 25.3s。预切 ≤30 字首段后, 任意输入的首包
        与短句看齐 (~2.7s)。返回 (首段, 剩余); 无需切分时剩余为空串。
        """
        if len(text) <= max_head:
            return text, ""
        win = text[:max_head]
        cut = max((i for i, ch in enumerate(win) if ch in "。！？!?；;"), default=-1)
        if cut >= 8:
            return text[: cut + 1], text[cut + 1:]
        cut = next((i for i, ch in enumerate(win) if ch in "，,、：" and i >= 8), -1)
        if cut >= 8:
            return text[: cut + 1], text[cut + 1:]
        return text, ""

    def synthesize_stream(
        self,
        text: str,
        ref_audio: str,
        prompt_text: str = "",
        text_lang: str = "zh",
        prompt_lang: str = "zh",
        t2s_weights: Optional[str] = None,
        vits_weights: Optional[str] = None,
        split_method: str = "cut3",
        head_seed: int = -1,
    ):
        """流式合成: 逐块 yield (wav_chunk float32 [-1,1], sr)。

        官方 streaming_mode: 语义 token 分段解码, 首块延迟远小于整段。
        首包优化: 先合成预切的 ≤30 字短首段, 再合成剩余 (W1, 实测首包
        25.3s→2.7s 级)。split_method: cut3 按句号切(默认, 首包最优)
              / cut1 凑四句一切 / cut0 不切 / cut5 按标点切
        head_seed: ≥0 时固定首段采样种子 — 首包延迟从 2.7~4.9s 方差
              收敛到确定值 (后续段仍随机, 不伤韵律多样性)
        """
        import numpy as np

        text = self._sanitize_text(text)
        ref_audio = str(Path(ref_audio).resolve())
        t2s_weights = str(Path(t2s_weights).resolve()) if t2s_weights else None
        vits_weights = str(Path(vits_weights).resolve()) if vits_weights else None
        head, rest = self._stream_head_split(text)
        segments = [head, rest] if rest else [text]
        with self._lock:
            self._lazy_init()
            inputs = {
                "text": head,
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
                for i, seg in enumerate(segments):
                    inputs["text"] = seg
                    inputs["seed"] = head_seed if (i == 0 and head_seed >= 0) else -1
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
