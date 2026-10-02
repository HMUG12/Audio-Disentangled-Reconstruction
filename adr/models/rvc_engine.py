"""RVC 音色转换引擎 (D1: 歌声/语音 → 目标音色)。

链路定位: DiffSinger/GSV 输出 → RVC 转换 → 用户音色。与档案库绑定
(voice meta 的 rvc_weights/rvc_index 字段)。

注意: faiss 在中文路径下读不了 .index → 引擎自动拷贝索引到 C:\\Temp。
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RVC_DIR = REPO_ROOT / "third_party" / "rvc"


class RVCEngine:
    """RVC 转换引擎 (懒加载单例)。"""

    def __init__(self):
        self._vc = None
        self._lock = threading.Lock()
        self._loaded_model = None

    @contextlib.contextmanager
    def _rvc_context(self):
        old_cwd = os.getcwd()
        if str(RVC_DIR) not in sys.path:
            sys.path.insert(0, str(RVC_DIR))
        # 环境变量须在 infer.* import 前设置 (其 utils 在 import 时捕获)
        os.environ.setdefault("weight_root", str(RVC_DIR / "assets" / "weights"))
        os.environ.setdefault("rmvpe_root", str(RVC_DIR / "assets" / "rmvpe"))
        os.environ.setdefault("index_root", str(RVC_DIR / "logs"))
        os.environ.setdefault("outside_index_root", str(RVC_DIR / "logs"))
        os.chdir(RVC_DIR)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def convert(
        self,
        audio_path: str,
        model_name: str,
        index_path: Optional[str] = None,
        index_rate: float = 0.75,
        pitch: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ) -> "tuple":
        """音频 → 目标音色 (wav float32, sr)。

        model_name: assets/weights 下的 .pth 文件名
        index_path: 检索索引 (.index); 中文路径会自动拷到 C:\\Temp
        """
        import shutil
        import tempfile

        with self._lock, self._rvc_context():
            if self._vc is None:
                from configs.config import Config
                from infer.vc.modules import VC
                self._vc = VC(Config())
            if self._loaded_model != model_name:
                self._vc.get_vc(model_name)
                self._loaded_model = model_name

            # faiss 中文路径 bug: 索引拷到纯 ASCII 临时位
            if index_path and any(ord(c) > 127 for c in str(index_path)):
                tmp = Path(tempfile.gettempdir()) / Path(str(index_path)).name
                shutil.copy2(str(index_path), tmp)
                index_path = str(tmp)

            info, (sr, audio) = self._vc.vc_single(
                0, str(Path(audio_path).resolve()), pitch, "rmvpe",
                index_path or "", index_rate, 0, rms_mix_rate, protect,
            )
        import numpy as np
        audio = np.asarray(audio, dtype=np.float32)
        if np.abs(audio).max() > 1.5:
            audio = audio / 32768.0
        return audio, int(sr)


_ENGINE: Optional[RVCEngine] = None


def get_rvc_engine() -> RVCEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RVCEngine()
    return _ENGINE
