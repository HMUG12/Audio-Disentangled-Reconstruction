"""DiffSinger v1 官方预训练套壳引擎 (保底路线 B3)。

定位: 高质量 SVS 基线。直接吃 OpenCpop 风格标注
(ph_seq/note_seq/note_dur_seq/is_slur_seq), 输出 24kHz wav。
与 ADR-2 自研管线独立 (自带 NSF-HiFiGAN 声码器, 不经 mel 接口)。

官方权重 (GitHub releases pretrain-model):
- 声学: 0831_opencpop_ds1000 (OpenCpop, 1000 步扩散)
- 声码器: 0109_hifigan_bigpopcs_hop128 (70h 歌声 NSF-HiFiGAN)
- 音高提取: 0102_xiaoma_pe
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DS_CODE_DIR = REPO_ROOT / "third_party" / "diffsinger_v1_code"
DEFAULT_EXP = "0831_opencpop_ds1000"


@dataclass
class DiffSingerEngineConfig:
    """套壳引擎配置。"""
    code_dir: Path = DS_CODE_DIR
    exp_name: str = DEFAULT_EXP
    device: str = "auto"          # auto / cpu / cuda
    # 推理速度: config pndm_speedup=40 → 25 步; 可传 hparams 字符串覆盖


class DiffSingerEngine:
    """DiffSinger v1 推理引擎 (懒加载, 进程内单例语义)。

    v1 代码的 hparams 是模块级全局 + 相对路径 ckpt, 因此:
    - 初始化时把 code_dir 注入 sys.path 并临时 chdir
    - 全程加锁 (Gradio 多线程安全)
    """

    def __init__(self, config: Optional[DiffSingerEngineConfig] = None):
        self.config = config or DiffSingerEngineConfig()
        self._infer = None
        self._hparams = None
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._infer is not None

    @contextlib.contextmanager
    def _ds_context(self):
        """进入 v1 代码上下文 (sys.path + cwd)。"""
        code_dir = str(self.config.code_dir)
        old_cwd = os.getcwd()
        added = code_dir not in sys.path
        if added:
            sys.path.insert(0, code_dir)
        os.chdir(code_dir)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def _lazy_init(self):
        if self._infer is not None:
            return
        with self._ds_context():
            from inference.svs.ds_e2e import DiffSingerE2EInfer
            from utils.hparams import set_hparams, hparams
            self._hparams = set_hparams(
                config=f"checkpoints/{self.config.exp_name}/config.yaml",
                exp_name=self.config.exp_name,
                print_hparams=False,
            )
            self._infer = DiffSingerE2EInfer(hparams)
            device = self.config.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._infer.device = device
            self._infer.model.to(device)
            self._infer.vocoder.to(device)
            if getattr(self._infer, "pe", None) is not None:
                self._infer.pe.to(device)

    def warmup(self):
        """预热: 后台线程调用, 把权重加载从首次合成挪到启动期 (A1)。"""
        with self._lock:
            self._lazy_init()

    def synthesize(
        self,
        ph_seq: str,
        note_seq: str,
        note_dur_seq: str,
        is_slur_seq: str = "",
        text: str = "",
        speedup: Optional[int] = None,
    ) -> "tuple":
        """标注 → 24kHz wav。

        Args:
            ph_seq: 音素序列 "x iao j iu ..."
            note_seq: 音符序列 "C#4/Db4 ..." (每音素一个, 重复)
            note_dur_seq: 每音素音符时长(秒) "0.4 0.4 ..."
            is_slur_seq: 连音标记 "0 0 1 ..." (可空 → 全 0)
            text: 仅用于日志
            speedup: PNDM 加速倍率 (None=配置默认 40)

        Returns:
            (wav: np.ndarray float32, sample_rate: int)
        """
        import numpy as np
        import torch

        with self._lock:
            self._lazy_init()
            n_ph = len(ph_seq.split())
            if not is_slur_seq:
                is_slur_seq = " ".join(["0"] * n_ph)
            inp = {
                "text": text or ph_seq,
                "ph_seq": ph_seq,
                "note_seq": note_seq,
                "note_dur_seq": note_dur_seq,
                "is_slur_seq": is_slur_seq,
                "input_type": "phoneme",
            }
            with self._ds_context(), torch.no_grad():
                hp = self._hparams
                if speedup is not None:
                    hp["pndm_speedup"] = speedup
                wav = self._infer.infer_once(inp)
                sr = hp["audio_sample_rate"]
        return np.asarray(wav, dtype=np.float32), sr

    def synthesize_word(
        self,
        text: str,
        notes: str,
        notes_duration: str,
        speedup: Optional[int] = None,
    ) -> "tuple":
        """word 级输入 (歌词 + 每字音符, '|' 分隔) → wav。"""
        import numpy as np
        import torch

        with self._lock:
            self._lazy_init()
            inp = {
                "text": text,
                "notes": notes,
                "notes_duration": notes_duration,
                "input_type": "word",
            }
            with self._ds_context(), torch.no_grad():
                hp = self._hparams
                if speedup is not None:
                    hp["pndm_speedup"] = speedup
                wav = self._infer.infer_once(inp)
                sr = hp["audio_sample_rate"]
        return np.asarray(wav, dtype=np.float32), sr

    def sing_like(
        self,
        text: str,
        ref_audio: str,
        speedup: Optional[int] = None,
        voice_name: Optional[str] = None,
    ) -> "tuple":
        """B4 桥: 任意歌词 + 参考音频旋律 → 歌声。

        参考音频提供旋律 (F0 量化为音符), 文本决定歌词。
        音色为 OpenCpop 训练歌手 (官方模型无音色克隆能力);
        voice_name 命中档案且档案绑了 RVC 权重时, 追加 RVC 转换 (D1 链:
        你的音色唱歌)。
        """
        from adr.data.f0 import F0Extractor
        from adr.models.melody_bridge import build_word_level_input

        # 批次38 (P0-2): extract_with_sr 保留参考音频原生采样率, 避免 22050
        # 硬编码把 44.1k/48k 参考音频的音符时长放大 2~2.2 倍。
        f0, ref_sr = F0Extractor().extract_with_sr(ref_audio)
        inp = build_word_level_input(text, f0, sample_rate=ref_sr)
        wav, sr = self.synthesize_word(inp["text"], inp["notes"],
                                       inp["notes_duration"], speedup=speedup)

        if voice_name:
            from adr.models.voice_library import load_voice
            prof = load_voice(voice_name)
            rvc_w = prof.get("rvc_weights")
            if rvc_w:
                import tempfile

                import soundfile as sf
                from adr.models.rvc_engine import get_rvc_engine
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_in = f.name
                sf.write(tmp_in, wav, sr)
                wav, sr = get_rvc_engine().convert(
                    tmp_in, rvc_w, index_path=prof.get("rvc_index"))
        return wav, sr
    @staticmethod
    def load_opencpop_annotation(utt_id: str,
                                 transcriptions: Optional[Path] = None,
                                 ) -> dict:
        """从 OpenCpop transcriptions.txt 读一条标注为 synthesize 参数。"""
        transcriptions = transcriptions or (
            REPO_ROOT / "data" / "opencpop" / "transcriptions.txt")
        for line in open(transcriptions, encoding="utf-8"):
            f = line.strip().split("|")
            if f[0] == utt_id:
                return {
                    "text": f[1],
                    "ph_seq": f[2],
                    "note_seq": f[3],
                    "note_dur_seq": f[4],
                    "is_slur_seq": f[6],
                }
        raise KeyError(f"utt_id not found: {utt_id}")


_ENGINE: Optional[DiffSingerEngine] = None


def get_engine() -> DiffSingerEngine:
    """进程级单例 (Gradio 共享)。"""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DiffSingerEngine()
    return _ENGINE
