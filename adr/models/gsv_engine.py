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
import io
import logging
import os
import pickle
import sys
import threading
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 批次26: 段缓存拆至 gsv_runtime, 此处 re-export 保持既有访问面不变
# (tests 直接用 gsv_engine._SEG_CACHE / _seg_cache_clear 等)
from adr.models.gsv_runtime import (  # noqa: F401
    _SEG_CACHE,
    _seg_cache_clear,
    _seg_cache_enabled,
    _seg_cache_get,
    _seg_cache_key,
    _seg_cache_put,
)
from adr.core import settings

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GSV_DIR = REPO_ROOT / "third_party" / "gpt_sovits"

# Track B 收口: chdir 是进程级全局状态, GSV third_party 代码用相对路径
# 定位权重, chdir 无法消除, 只能串行化整个 chdir 窗口 — 否则线程 A 刚
# 进入 GSV 上下文, 线程 B 的 finally 恢复 cwd 会把 A 踩进错误目录。
# RLock: synthesize_stream 的 _gsv_context 内嵌套 _ensure_weights 再开
# _gsv_context, 必须可重入。
_CHDIR_LOCK = threading.RLock()


def validate_weights_file(weights_path: str) -> None:
    """权重文件安全预检 (Track B 收口; 批次38 收编为引擎单点): 防恶意 pickle RCE。

    引擎热换最终走 GSV third_party 内部的 torch.load (无 weights_only,
    third_party 不可改), 恶意 pickle 文件会在其加载时执行任意代码。
    两段式校验 (批次43):

    1) 快路径 torch.load(weights_only=True) — 官方预训练等纯 torch 权重
       直接通过;
    2) 慢路径受限 pickle 扫描 — GPT-SoVITS 训练产物把 config 存成第三方
       utils.HParams 对象, torch>=2.6 的 weights_only=True 默认拒绝
       (官方预训练的 config 是纯 dict, 不受影响), 导致安装态 prewarm /
       换权全挂 ("tts failed")。慢路径按 GLOBAL 白名单扫描 pickle 流:
       白名单外一律拒绝 (os.system 等 RCE 载荷进不来), 白名单内对象
       stub 化 — 不 import 模块、不执行类体, storage 一律 stub; 顶层
       仍须为 dict。安全性不降, 训练产物放行。

    唯一强收口点在 _ensure_weights 换权分支内 (覆盖 v2 请求级热换 /
    v3 档案 / native / openai / set_*_weights 全部入口); v2 层另有
    请求入口早检, 仅为了流式请求能在响应头发出前给出干净的 400。
    """
    import torch
    try:
        obj = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as fast_err:
        try:
            _restricted_weights_scan(weights_path)
        except Exception as slow_err:
            raise ValueError(
                f"weights file rejected by safe loader: {type(slow_err).__name__}: "
                f"{slow_err} ({weights_path})") from slow_err
        return  # 慢路径通过: 含 HParams 的 GSV 训练产物
    if not isinstance(obj, dict):
        raise ValueError("weights file top-level must be a dict of tensors")


def _is_trusted_weights_global(module: str, name: str) -> bool:
    """慢路径 GLOBAL 白名单 (批次43, 探针实测真实权重文件 GLOBAL 并集)。

    - collections.OrderedDict: torch.save 顶层容器 (真类, 保顶层 dict 校验)
    - torch.*Storage: zip 权重 persistent_id 元组里的 storage 类型
    - torch._utils._rebuild_*: torch 张量重建函数
    - utils.HParams: GPT-SoVITS 训练产物 config 的第三方类型 (唯一放行的
      非 torch GLOBAL; stub 化, 不 import third_party)
    """
    if module == "collections" and name == "OrderedDict":
        return True
    if module == "torch" and name.endswith("Storage"):
        return True
    if module == "torch._utils" and name in (
        "_rebuild_tensor_v2",
        "_rebuild_tensor",
        "_rebuild_parameter",
        "_rebuild_device_tensor_from_numpy",
    ):
        return True
    return module == "utils" and name == "HParams"


class _WeightsScanStub:
    """GLOBAL 白名单内对象的 no-op 替身: 不 import 模块、不执行任何代码,
    只让 pickle 流走完 (REDUCE / BUILD / SETITEMS / APPENDS 全部吞掉)。"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _WeightsScanStub()

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return None

    def __setstate__(self, state):
        pass

    def append(self, value):
        pass

    def __len__(self):
        return 0


class _WeightsScanUnpickler(pickle.Unpickler):
    """受限 Unpickler: 白名单外 GLOBAL 直接抛 UnpicklingError。"""

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict  # 标准库纯容器, 真类以保顶层 dict 校验
        if _is_trusted_weights_global(module, name):
            return _WeightsScanStub
        raise pickle.UnpicklingError(
            f"forbidden global in weights pickle: {module}.{name}")

    def persistent_load(self, pid):
        # 只验结构不读 storage: zip 权重的张量数据不进入内存
        return _WeightsScanStub()


def _restricted_weights_scan(weights_path: str) -> None:
    """慢路径: 受限 Unpickler 扫描 torch zip 权重的 data.pkl (不加载张量)。"""
    with open(weights_path, "rb") as f:
        raw = f.read()
    if raw[:2] != b"PK":
        # GSV 生态常见的截头 zip 自愈 (与 third_party process_ckpt.load_sovits_new
        # 同款补头); 真 legacy 非 zip 格式会在下方 BadZipFile 拒绝 — 该格式
        # 仅存于 torch 1.x 古董文件, 官方/ADR 产物均为 zip 格式。
        raw = b"PK" + raw
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
            if len(pkl_names) != 1:
                raise pickle.UnpicklingError(
                    f"expected exactly one data.pkl in weights zip, got {len(pkl_names)}")
            data = zf.read(pkl_names[0])
    except zipfile.BadZipFile as e:
        raise pickle.UnpicklingError(
            "not a torch zip weights file (legacy non-zip format not accepted)") from e
    obj = _WeightsScanUnpickler(io.BytesIO(data)).load()
    if not isinstance(obj, dict):
        raise pickle.UnpicklingError(
            "weights file top-level must be a dict of tensors")


def _resolve_fragment_interval(fragment_interval: Optional[float]) -> float:
    """Track E (批次33): 句末静音秒数解析 — None → env ADR_TTS_FRAGMENT_INTERVAL
    (默认 0.3 = GSV 原行为); 显式传入钳到 [0, 2] (负数会使下游 np.zeros 崩)。
    (批次36: env 读取委托 settings 单源)"""
    fi = fragment_interval
    if fi is None:
        fi = settings.fragment_interval_default()
    return max(0.0, min(float(fi), 2.0))


@dataclass
class GSVEngineConfig:
    """GPT-SoVITS 套壳引擎配置。"""
    gsv_dir: Path = GSV_DIR
    version: str = "v2"           # v1/v2 权重已下载; v3/v4 需另下
    device: str = "auto"
    half: bool = True             # 8GB 卡半精度
    bert_onnx: str = "auto"       # auto=仅 CPU 设备启用 / on / off


class _OnnxBertFeat:
    """批次5: ORT 版 BERT 特征提取 (纯 CPU 档 ~1.4x, 实测 bench_bert_onnx.py)。

    模仿 BertForMaskedLM 调用面: res = model(**inputs, output_hidden_states=True)
    GSV 消费面只有 res["hidden_states"][-3] (倒数第 3 层, TextPreprocessor
    get_bert_feature)。其余层不导出 — 一次前向只算一层。
    惰性: 首次调用建 ORT 会话; onnx 模型不存在时自动导出并缓存。
    """

    def __init__(self, bert_dir: str, cache_path: Path):
        self.bert_dir = str(bert_dir)
        self.cache_path = Path(cache_path)
        self._sess = None

    def _ensure_session(self):
        if self._sess is not None:
            return
        import torch
        from transformers import AutoModel
        if not self.cache_path.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            m = AutoModel.from_pretrained(self.bert_dir).eval()

            class _Feat(torch.nn.Module):
                def __init__(self, mm):
                    super().__init__()
                    self.m = mm

                def forward(self, input_ids, attention_mask, token_type_ids):
                    out = self.m(input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 output_hidden_states=True)
                    return out.hidden_states[-3]

            dummy = torch.ones(1, 8, dtype=torch.long)
            torch.onnx.export(
                _Feat(m), (dummy, dummy, dummy), str(self.cache_path),
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["feat"],
                dynamic_axes={n: {0: "b", 1: "s"} for n in
                              ["input_ids", "attention_mask", "token_type_ids",
                               "feat"]},
                opset_version=17, do_constant_folding=True)
        from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
        opts = SessionOptions()
        opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = InferenceSession(str(self.cache_path), opts,
                                      providers=["CPUExecutionProvider"])

    def __call__(self, *, input_ids=None, attention_mask=None,
                 token_type_ids=None, **_kw):
        import torch
        self._ensure_session()
        feed = {"input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
                "token_type_ids": token_type_ids.cpu().numpy()}
        feat = self._sess.run(["feat"], feed)[0]
        # 模仿 hidden_states 元组: GSV 取 [-3:-2] (倒数第 3 层), 3 元素列表
        # 的首位即 [-3]; 其余占位 (只有这一层被消费)
        return {"hidden_states": [torch.from_numpy(feat), None, None]}


class GSVEngine:
    """GPT-SoVITS 推理引擎 (懒加载 + 锁 + cwd 上下文, 同 DiffSingerEngine 模式)。"""

    def __init__(self, config: Optional[GSVEngineConfig] = None):
        self.config = config or GSVEngineConfig()
        self._tts = None
        self._lock = threading.Lock()
        self._loaded = {"t2s": None, "vits": None}   # 已热换权重路径 (批次9幂等跳过)
        # 批次23: 并发排队可视化计数 (_lock 外等待数 + 持锁合成中数)
        self._qlock = threading.Lock()
        self._queue_waiters = 0
        self._busy = 0

    def queue_depth(self) -> int:
        """正在 _lock 外排队等待合成的请求数 (批次23, 供前端可视化)。"""
        with self._qlock:
            return self._queue_waiters

    def synth_busy(self) -> int:
        """正在合成 (持 _lock) 的请求数 (批次23)。"""
        with self._qlock:
            return self._busy

    @property
    def is_ready(self) -> bool:
        return self._tts is not None

    @contextlib.contextmanager
    def _gsv_context(self):
        """进入 GSV 代码上下文 (sys.path + cwd, 其代码用相对路径找权重)。"""
        code_dir = str(self.config.gsv_dir)
        inner = str(self.config.gsv_dir / "GPT_SoVITS")
        # Track B: 全程持锁覆盖 chdir 窗口, 进出后 cwd 恒等于原值
        with _CHDIR_LOCK:
            old_cwd = os.getcwd()
            for p in (code_dir, inner):
                if p not in sys.path:
                    sys.path.insert(0, p)
            os.chdir(code_dir)
            try:
                yield
            finally:
                os.chdir(old_cwd)

    def _lazy_init(self):
        if self._tts is not None:
            return
        # 批次10: t2s CUDA Graph 桥 — AR 提速 3.77x 但官方 SDPA 实现下 AR 分布
        # 漂移 (同输入 45→254 token, 音频时长翻倍), 默认关闭; 需上游修复后
        # 可设 ADR_T2S_CUDAGRAPH=1 启用 (诊断见 docs/m9-phase3-plan.md 批次10)
        os.environ.setdefault("ADR_T2S_CUDAGRAPH", "0")
        # 兼容补丁: NLTK (英文 G2P 依赖) 离线环境下载默认超时 72s/次,
        # 收紧到 5s 快速失败走内置回退发音
        import socket
        socket.setdefaulttimeout(5)
        with self._gsv_context():
            import torch
            # Track E (批次33): AR 解码 tqdm 每句 1500 迭代渲染 stderr,
            # TQDM_DISABLE 对 tqdm 4.67 实测无效 — monkeypatch __init__ 注入
            # disable=True。第三方 `from tqdm import tqdm` 绑定同一类对象,
            # 补类属性即全局生效; 必须在 TTS 导入前打。ADR_TTS_KEEP_TQDM=1 保留。
            if not settings.keep_tqdm():
                try:
                    import tqdm as _tqdm

                    _tqdm_init = _tqdm.tqdm.__init__

                    def _silent_tqdm_init(_tq, *a, **k):
                        k.setdefault("disable", True)
                        _tqdm_init(_tq, *a, **k)

                    _tqdm.tqdm.__init__ = _silent_tqdm_init
                except Exception:
                    pass
            from TTS_infer_pack.TTS import TTS, TTS_Config

            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            # 批次4 yaml 漂移修复: GSV 的 init_vits_weights 会把热换权重写回
            # tts_infer.yaml 的 custom 段 (实测两次进程启动基线 0.812→0.759 漂移,
            # 任何引擎重启都会静默换权重)。对策: 只取 yaml 的 v2 原始段 (预训练
            # 权重) 构造 dict 传入 — 每次进程启动都从干净状态出发; 并把写回
            # 路径重定向到私有 scratch 文件, 共享 yaml 永不被改。
            import yaml as _yaml
            _cfgp = self.config.gsv_dir / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
            _raw = _yaml.safe_load(_cfgp.read_text(encoding="utf-8"))
            cfg = TTS_Config({"v2": _raw["v2"]})
            cfg.device = device
            cfg.is_half = self.config.half and device == "cuda"
            cfg.version = self.config.version
            self._tts = TTS(cfg)
            # 热换权重触发的 save_configs 写到私有文件, 不再污染共享配置
            _scratch = self.config.gsv_dir / "TEMP" / "tts_infer_adr_scratch.yaml"
            _scratch.parent.mkdir(exist_ok=True)
            self._tts.configs.configs_path = str(_scratch)
            # Track E (批次33): run() finally 每句 empty_cache (gc.collect +
            # cuda.empty_cache) — 句间停顿来源之一, 且清缓存后下一步分配要重新
            # cudaMalloc。改为每 16 次调用真清理一次 (防显存碎片兜底保留)。
            # ADR_TTS_KEEP_EMPTY_CACHE=1 回退每句清理。
            if not settings.keep_empty_cache():
                _orig_empty_cache = self._tts.empty_cache
                _ec_calls = {"n": 0}

                def _throttled_empty_cache():
                    # 实时读 env (经 settings): A/B 基准可在同进程切腿
                    # (KEEP_EMPTY_CACHE=1 即旧每句清)
                    if settings.keep_empty_cache():
                        _orig_empty_cache()
                        return
                    _ec_calls["n"] += 1
                    if _ec_calls["n"] % 16 == 0:
                        _orig_empty_cache()

                self._tts.empty_cache = _throttled_empty_cache

            # 批次5: 纯 CPU 档用 ORT 版 BERT (组件级 ~1.4x, 端到端 ~10%)
            if ((device == "cpu" and self.config.bert_onnx == "auto")
                    or self.config.bert_onnx == "on"):
                try:
                    self._tts.bert_model = _OnnxBertFeat(
                        self._tts.bert_model.config._name_or_path,
                        self.config.gsv_dir / "TEMP" / "bert_feat_cpu.onnx")
                except Exception:  # onnxruntime/onnx 缺失等 → 保持 torch 版
                    pass

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
            self._ensure_weights(vits_weights, t2s_weights)

    def _ensure_weights(self, vits_weights: Optional[str],
                        t2s_weights: Optional[str]):
        """热换到目标权重, 同路径幂等跳过 (批次 9)。

        上游 init_vits_weights/init_t2s_weights 无条件完整重载 — 稳态下
        连续合成同一音色每次白付 1~3s 权重重载。调用方须持有 self._lock。
        """
        if not (vits_weights or t2s_weights):
            return
        vw = str(Path(vits_weights).resolve()) if vits_weights else None
        tw = str(Path(t2s_weights).resolve()) if t2s_weights else None
        with self._gsv_context():
            if vw and self._loaded.get("vits") != vw:
                # 批次38: 安全预检收编到唯一换权点 — 恶意 pickle 在 third_party
                # torch.load 前拒绝, 覆盖 v2 请求级 / v3 档案 / set_* 全部入口
                validate_weights_file(vw)
                self._tts.init_vits_weights(vw)
                self._loaded["vits"] = vw
            if tw and self._loaded.get("t2s") != tw:
                validate_weights_file(tw)
                self._tts.init_t2s_weights(tw)
                self._loaded["t2s"] = tw

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
    def _stream_head_split(text: str, max_head: int = 14) -> "tuple":
        """W1 首包优化: 从长文本切出短首段 (句末标点优先, 逗号次之)。

        实测 (output/bench_stream.json): 首块延迟 ∝ 首段长度, 45 字长句
        cut1 首包 5.3s / cut0 25.3s。批次 9 再收紧 30→14 字: 首段 AR 生成
        token 数减半 (首包 4.4s→~2.5s), 逗号层 min 6 字早生效 — 2s 级短句
        (16~20 字) 也能在逗号处切出短首段。返回 (首段, 剩余); 无需切分时
        剩余为空串。
        """
        if len(text) <= max_head:
            return text, ""
        win = text[:max_head]
        cut = max((i for i, ch in enumerate(win) if ch in "。！？!?；;"), default=-1)
        if cut >= 6:
            return text[: cut + 1], text[cut + 1:]
        cut = next((i for i, ch in enumerate(win) if ch in "，,、：" and i >= 6), -1)
        if cut >= 6:
            return text[: cut + 1], text[cut + 1:]
        return text, ""

    @staticmethod
    def _split_long(text: str, max_len: int = 40) -> list:
        """批次20 长句优化: 无句读超长段细分, 救 GSV cut 切不开的最坏情况。

        GSV 内部 split (cut1/3/5) 依赖标点 — 一整段无句号的"单长句"会整段
        一次 AR decode: 耗时线性膨胀且发散 (电音/加字) 概率随长度陡增。
        有句读的段交给 GSV 自切 (保留其凑句/分段优化); 仅对无句读且超
        max_len 的段按停顿标点细分, 完全无标点按 max_len+15 硬切兜底。
        """
        if len(text) <= max_len or any(c in text for c in "。！？!?；;"):
            return [text]
        segs, buf = [], ""
        for ch in text:
            buf += ch
            if len(buf) >= max_len and ch in "，,、：:—…":
                segs.append(buf)
                buf = ""
            elif len(buf) >= max_len + 15:  # 无标点硬切兜底
                segs.append(buf)
                buf = ""
        if buf:
            segs.append(buf)
        return segs

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
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        speed_factor: float = 1.0,
        fragment_interval: Optional[float] = None,
    ):
        """流式合成: 逐块 yield (wav_chunk float32 [-1,1], sr)。

        官方 streaming_mode: 语义 token 分段解码, 首块延迟远小于整段。
        首包优化: 先合成预切的 ≤30 字短首段, 再合成剩余 (W1, 实测首包
        25.3s→2.7s 级)。split_method: cut3 按句号切(默认, 首包最优)
              / cut1 凑四句一切 / cut0 不切 / cut5 按标点切
        head_seed: ≥0 时固定首段采样种子 — 首包延迟从 2.7~4.9s 方差
              收敛到确定值 (后续段仍随机, 不伤韵律多样性)
        top_k / top_p / temperature: GPT 采样参数。默认 (15, 1.0, 1.0)
              为上游流式历史行为; 偏发散, 档案可配保守值压电音 (批次20)
        speed_factor: 语速 (批次21)。仅 ≠1.0 时注入 GSV inputs —
              GSV 流式路径对语速支持不稳, ==1.0 保持历史行为字节级一致
        fragment_interval: 句末静音秒数 (批次33)。None → env
              ADR_TTS_FRAGMENT_INTERVAL (默认 0.3 = GSV 原行为); 置 0 可
              消除句间停顿 (Track E)
        """
        import numpy as np

        text = self._sanitize_text(text)
        ref_audio = str(Path(ref_audio).resolve())
        t2s_weights = str(Path(t2s_weights).resolve()) if t2s_weights else None
        vits_weights = str(Path(vits_weights).resolve()) if vits_weights else None
        head, rest = self._stream_head_split(text)
        segments = [head, *self._split_long(rest)] if rest else [text]
        # 批次23: 排队可视化 — 进锁前计 waiter, 拿到锁转入 busy (entered
        # 标志防懒加载抛异常漏减; 生成器被客户端断连关闭时 finally 同样兜住)
        with self._qlock:
            self._queue_waiters += 1
        entered = False
        try:
            with self._lock:
                with self._qlock:
                    self._queue_waiters -= 1
                    self._busy += 1
                entered = True
                self._lazy_init()
                inputs = {
                    "text": head,
                    "text_lang": text_lang,
                    "ref_audio_path": ref_audio,
                    "prompt_text": prompt_text,
                    "prompt_lang": prompt_lang,
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "text_split_method": split_method,
                    "streaming_mode": True,
                    "parallel_infer": True,
                }
                if speed_factor != 1.0:
                    inputs["speed_factor"] = speed_factor
                # 批次33: 句末静音秒数 — 一直传 (None→env 解析), 0 可消除句间停顿
                fi = _resolve_fragment_interval(fragment_interval)
                inputs["fragment_interval"] = fi
                with self._gsv_context():
                    self._ensure_weights(vits_weights, t2s_weights)
                    for i, seg in enumerate(segments):
                        inputs["text"] = seg
                        inputs["seed"] = head_seed if (i == 0 and head_seed >= 0) else -1
                        # 批次23: 句级缓存 — 命中直接回放 int16, 跳过 GPU 前向
                        ckey = None
                        if _seg_cache_enabled():
                            ckey = _seg_cache_key(
                                seg, ref_audio, prompt_text, text_lang,
                                prompt_lang, split_method, top_k, top_p,
                                temperature, speed_factor, fi, t2s_weights,
                                vits_weights)
                            hit = _seg_cache_get(ckey)
                            if hit is not None:
                                pcm, hit_sr = hit
                                buf = np.frombuffer(pcm, dtype=np.int16)
                                yield buf.astype(np.float32) / 32767.0, int(hit_sr)
                                continue
                        seg_chunks: list = []
                        seg_sr = 32000
                        for sr, chunk in self._tts.run(inputs):
                            chunk = np.asarray(chunk, dtype=np.float32)
                            if np.abs(chunk).max() > 1.5:
                                chunk = chunk / 32768.0
                            seg_chunks.append(chunk)
                            seg_sr = int(sr)
                            yield chunk, int(sr)
                        if ckey is not None and seg_chunks:
                            full = (seg_chunks[0] if len(seg_chunks) == 1
                                    else np.concatenate(seg_chunks))
                            _seg_cache_put(
                                ckey,
                                (np.clip(full, -1.0, 1.0) * 32767).astype(np.int16).tobytes(),
                                seg_sr)
        finally:
            with self._qlock:
                if entered:
                    self._busy -= 1
                else:
                    self._queue_waiters -= 1

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
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        fragment_interval: Optional[float] = None,
    ) -> "tuple":
        """零样本克隆朗读: (text, 参考音频) → (wav, sr)。

        Args:
            text: 要朗读的文本
            ref_audio: 参考音频路径 (3-10 秒干净人声最佳)
            prompt_text: 参考音频的文本 (可空, 空则弱化文本条件)
            text_lang / prompt_lang: zh/en/ja/ko/yue 等
            speed_factor: 语速 (1.0 原速)
            seed: -1 随机
            top_k / top_p / temperature: GPT 采样参数 (短句可收 top_k 降跑偏)
        """
        import numpy as np

        text = self._sanitize_text(text)
        # 绝对路径: _gsv_context 会 chdir, 相对路径会丢
        ref_audio = str(Path(ref_audio).resolve())
        t2s_weights = str(Path(t2s_weights).resolve()) if t2s_weights else None
        vits_weights = str(Path(vits_weights).resolve()) if vits_weights else None
        # 批次23: 排队可视化 (同 synthesize_stream, entered 标志防漏减)
        with self._qlock:
            self._queue_waiters += 1
        entered = False
        try:
            with self._lock:
                with self._qlock:
                    self._queue_waiters -= 1
                    self._busy += 1
                entered = True
                self._lazy_init()
                inputs = {
                    "text": text,
                    "text_lang": text_lang,
                    "ref_audio_path": str(ref_audio),
                    "prompt_text": prompt_text,
                    "prompt_lang": prompt_lang,
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "speed_factor": speed_factor,
                    "seed": seed,
                    "text_split_method": split_method,
                    "return_fragment": False,
                    "streaming_mode": False,
                    "parallel_infer": True,
                    "fragment_interval": _resolve_fragment_interval(fragment_interval),
                }
                self._ensure_weights(vits_weights, t2s_weights)
                with self._gsv_context():
                    sr, audio = None, None
                    for sr, audio in self._tts.run(inputs):
                        pass  # 非流式只 yield 一次完整音频
        finally:
            with self._qlock:
                if entered:
                    self._busy -= 1
                else:
                    self._queue_waiters -= 1
        if audio is None:
            raise RuntimeError("GPT-SoVITS 未返回音频")
        audio = np.asarray(audio, dtype=np.float32)
        # 官方返回 int16 量程 (rms ~1600), 归一化到 [-1,1]
        if np.abs(audio).max() > 1.5:
            audio = audio / 32768.0
        return audio, int(sr)


_ENGINE: Optional[GSVEngine] = None
_ENGINE_LOCK = threading.Lock()  # 批次41a: 单例创建锁 (多线程首次并发防多实例)
_LOADING = False   # 后台预热进行中 (服务启动期, 供控制台显示引擎三态)
_STAGE = ""        # 预热细分阶段: queued/importing/loading/kernel/failed (供前端实时显示)


def is_loading() -> bool:
    """引擎后台预热是否进行中 (进程级, 不触发加载)。"""
    return _LOADING


def is_ready() -> bool:
    """引擎是否已就绪可合成 (_tts 权重已加载, 不触发加载)。"""
    return _ENGINE is not None and _ENGINE._tts is not None


def stage() -> str:
    """预热细分阶段 (人读字符串, 不触发加载): ready 优先, 其余见 _STAGE。"""
    if is_ready():
        return "ready"
    return _STAGE


def stage_text() -> str:
    """downloading 阶段的实时进度文案 (批次44 下载提示), 其余阶段返回空串。

    由 server stats 透传 → 桌面壳轮询 → prewarm 页显示下载进度。
    """
    if _STAGE != "downloading":
        return ""
    try:
        from adr.models import gsv_bootstrap

        st = gsv_bootstrap.status()
    except Exception:
        return "正在下载预训练模型…"
    b = st.get("stage")
    if b == "downloading":
        done, total = st.get("downloaded", 0), st.get("total", 0)
        mb = done / 1048576
        if total > 0:
            return (f"正在下载 {st.get('file', '')} "
                    f"{mb:.1f}/{total / 1048576:.1f} MB ({int(done * 100 / total)}%)")
        return f"正在下载 {st.get('file', '')} {mb:.1f} MB"
    if b == "extracting":
        return f"正在解压 {st.get('file', '')}…"
    return "正在下载预训练模型…"


def queue_depth() -> int:
    """进程级排队数 (批次23, 引擎未初始化时 0)。"""
    return _ENGINE.queue_depth() if _ENGINE is not None else 0


def synth_busy() -> int:
    """进程级合成中请求数 (批次23, 引擎未初始化时 0)。"""
    return _ENGINE.synth_busy() if _ENGINE is not None else 0


def get_gsv_engine() -> GSVEngine:
    """进程级单例 (批次41a: 双重检查锁 — 已初始化后走无锁快路径,
    多线程首次并发调用只创建一个实例)。"""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = GSVEngine()
    return _ENGINE


def prewarm(default_profile: str | None = None) -> None:
    """后台预热入口 (批次26): 阻塞调用, 放后台线程跑。

    状态机 queued→importing→loading→kernel→ready/failed 由本函数独占管理
    (_LOADING/_STAGE 只在模块内部读写, 外部经 is_loading()/stage() 只读观察)。
    服务端 (server/app.py) 启动线程调用; 桌面壳 prewarm 页面轮询 stage() 显示。
    失败不致命 (首次合成时仍会按需加载), 只打日志。
    """
    global _LOADING, _STAGE
    log = logging.getLogger("adr.models.gsv_engine")
    _LOADING = True
    _STAGE = "queued"
    try:
        _STAGE = "importing"  # import torch + GSV 模块 (最耗时可达 20s)
        from adr.models.voice_library import list_voices, load_voice

        prof_name = default_profile
        if not prof_name:
            voices = list_voices()
            prof_name = voices[0] if voices else None
        kw: dict = {}
        prof = None
        if prof_name:
            try:
                prof = load_voice(prof_name)
                kw = {"vits_weights": prof.get("vits_weights"),
                      "t2s_weights": prof.get("t2s_weights")}
                log.info("[prewarm] 用音色档案「%s」的权重预热", prof_name)
            except Exception:
                prof = None  # 档案损坏 → 退回纯预训练权重
        # 批次44: 预训练资源 bootstrap — 便携包不含底模, 首启按需下载+解压
        # (已就位时 pending() 为空, 零开销直接过)
        from adr.models import gsv_bootstrap

        if gsv_bootstrap.pending():
            _STAGE = "downloading"  # 进度经 stage_text() → stats → 壳透传
            gsv_bootstrap.ensure()
        _STAGE = "loading"    # 权重加载进显存
        eng = get_gsv_engine()
        eng.warmup(**{k: v for k, v in kw.items() if v})
        log.info("[prewarm] GPT-SoVITS 引擎就绪")
        # kernel JIT 预热: 否则首次合成再付 ~14s CUDA 编译 (只取首块)
        try:
            _STAGE = "kernel"
            if prof:
                for _ in eng.synthesize_stream(
                        "引擎预热。", prof["ref_audio"],
                        vits_weights=kw.get("vits_weights"),
                        split_method="cut0"):
                    break
                log.info("[prewarm] kernel 预热完成, 首次合成即秒级")
        except Exception as e:
            log.warning("[prewarm] kernel 预热失败 (不影响功能): %s", e)
        _STAGE = "ready"
    except Exception as e:
        _STAGE = "failed"
        log.warning("[prewarm] GSV 预热失败 (首次合成时将现场加载): %s", e)
    finally:
        _LOADING = False
