"""训练回调 (M1 Day 8)。

设计借鉴 HuggingFace Transformers / PyTorch Lightning 的 callback 机制。
轻量级: 不依赖 trainer 内部状态,所有信息通过 kwargs 传入。

M4: 新增 ASRCallback (ASR-based 早停, 用 WER/CER 评估 wav 可懂度)
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class Callback(ABC):
    """回调基类。"""

    def on_train_start(self, trainer: "Trainer", **kwargs) -> None:
        pass

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        pass

    def on_epoch_start(self, trainer: "Trainer", epoch: int, **kwargs) -> None:
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict, **kwargs) -> None:
        pass

    def on_step_end(self, trainer: "Trainer", step: int, metrics: dict, **kwargs) -> None:
        pass


class CheckpointCallback(Callback):
    """定期保存 checkpoint。"""

    def __init__(
        self,
        save_dir: str | Path,
        save_every_n_epochs: int = 1,
        save_best: bool = True,
        metric_name: str = "val/loss",
        mode: str = "min",  # "min" or "max"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode
        self.best_value: Optional[float] = None

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if (epoch + 1) % self.save_every_n_epochs == 0:
            ckpt_path = self.save_dir / f"epoch_{epoch + 1:04d}.pt"
            trainer.save_checkpoint(ckpt_path)
            print(f"  [Checkpoint] Saved {ckpt_path.name}")

        if self.save_best and self.metric_name in metrics:
            value = metrics[self.metric_name]
            if self.best_value is None or (
                (self.mode == "min" and value < self.best_value)
                or (self.mode == "max" and value > self.best_value)
            ):
                self.best_value = value
                best_path = self.save_dir / "best.pt"
                trainer.save_checkpoint(best_path)
                print(f"  [Best] {self.metric_name}={value:.4f} -> {best_path.name}")


class EarlyStoppingCallback(Callback):
    """早停 (验证集 loss 不再下降时停止)。"""

    def __init__(
        self,
        patience: int = 5,
        metric_name: str = "val/loss",
        mode: str = "min",
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self.metric_name not in metrics:
            return
        value = metrics[self.metric_name]
        if self.best_value is None:
            self.best_value = value
            return

        improved = (
            (self.mode == "min" and value < self.best_value - self.min_delta)
            or (self.mode == "max" and value > self.best_value + self.min_delta)
        )
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  [EarlyStop] No improvement for {self.patience} epochs.")


class LoggerCallback(Callback):
    """日志回调 (写 train_log.jsonl)。"""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # 清空旧日志
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        record = {
            "epoch": epoch + 1,
            "timestamp": time.time(),
            **metrics,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ProgressCallback(Callback):
    """打印进度 (轻量, 不依赖 tqdm)。"""

    def __init__(self, print_every_n_steps: int = 10):
        self.print_every_n_steps = print_every_n_steps
        self.epoch_start_time = 0.0

    def on_epoch_start(self, trainer, epoch, **kwargs):
        self.epoch_start_time = time.time()
        print(f"\n=== Epoch {epoch + 1}/{trainer.config.epochs} ===")

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        elapsed = time.time() - self.epoch_start_time
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  Epoch {epoch + 1} done ({elapsed:.1f}s): {metric_str}")


class WavQualityCallback(Callback):
    """Wav 质量监控回调 (M3)。

    设计动机: val/loss 持续下降不代表 TTS wav 质量提升 (loss ↔ wav 质量脱节)。
    此回调在每个 epoch 结束后, 用 N 个验证样本跑推理, 监控 wav 质量指标:
    - wav_rms_mean: 越大越好 (能量足够)
    - wav_centroid_mean: 越接近真实人声 (~4500Hz) 越好

    Args:
        n_samples: 推理样本数 (默认 5)
        n_timesteps: 推理去噪步数 (默认 10, 速度优先)
        ref_audio: 参考音频路径 (用第一个 val sample 的 waveform 兜底)
        metric_name: 主要监控指标 ("wav_rms_mean" 或 "wav_quality")
        mode: "min" 或 "max"
        patience: 多少 epoch 不改善就停
        min_delta: 视为"改善"的最小幅度
        use_placeholder_vocoder: 是否使用 placeholder vocoder (快速但不真实)
        vocoder_path: 真实 vocoder 路径 (None=自动搜索, 需配合 use_placeholder_vocoder=False)

    用法:
        >>> cb = WavQualityCallback(n_samples=5, patience=2)
        >>> trainer = Trainer(model, ds, config, callbacks=[cb])
        >>> trainer.fit()  # 自动监控 wav 质量
    """

    def __init__(
        self,
        n_samples: int = 5,
        n_timesteps: int = 10,
        ref_audio: Optional[str] = None,
        metric_name: str = "wav_quality",
        mode: str = "max",  # wav_quality 越大越好
        patience: int = 2,
        min_delta: float = 1e-3,
        use_placeholder_vocoder: bool = True,
        vocoder_path: Optional[str] = None,
    ):
        self.n_samples = n_samples
        self.n_timesteps = n_timesteps
        self.ref_audio = ref_audio
        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.use_placeholder_vocoder = use_placeholder_vocoder
        self.vocoder_path = vocoder_path
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        self._history: list = []  # 各 epoch 的指标
        self._vocoder_cache = None  # 真实 vocoder 缓存 (lazy load)
        self._vocoder_load_failed = False

    def _extract_f0(self, wav: np.ndarray, sr: int = 22050) -> np.ndarray:
        """从 wav 提取 F0 (用 librosa.pyin, 失败时返回空数组)。"""
        import numpy as _np
        try:
            import librosa
            f0, _, _ = librosa.pyin(
                wav.astype(_np.float32),
                fmin=80, fmax=500,  # 人声基频范围
                sr=sr,
                frame_length=2048,
                hop_length=512,
            )
            if f0 is None:
                return _np.array([])
            return f0[~_np.isnan(f0)]
        except Exception:
            return _np.array([])

    def _compute_wav_metrics(self, trainer) -> dict:
        """在 val set 上跑推理, 计算 wav 指标 (M3.6: 加 F0)。"""
        import numpy as np
        import torch

        trainer.model.eval()
        device = trainer.device
        # 取 N 个 val sample
        if trainer.val_loader is None:
            return {}
        samples_done = 0
        rms_list = []
        centroid_list = []
        f0_pred_means = []  # 生成 wav 的平均 F0
        f0_pred_stds = []   # 生成 wav 的 F0 标准差 (稳定性)
        ref_wav = None
        f0_ref_mean = None

        with torch.no_grad():
            for batch in trainer.val_loader:
                if samples_done >= self.n_samples:
                    break
                # 用第一个 sample 的 waveform 当 ref, 提取 ref F0
                if ref_wav is None and hasattr(batch, "waveform"):
                    ref_wav = batch.waveform[0].cpu().numpy()
                    f0_ref = self._extract_f0(ref_wav)
                    if len(f0_ref) > 5:
                        f0_ref_mean = float(np.mean(f0_ref))
                # 构造 batch dict (移到 device, 带 GT f0 评估 F0 跟随能力)
                batch_dict = {
                    "phoneme_ids": batch.phoneme_ids.to(device),
                    "phoneme_mask": batch.phoneme_mask.to(device),
                    "ref_mel": batch.ref_mel.to(device),
                    "f0": batch.f0.to(device),
                }
                # forward
                try:
                    out = trainer.model(batch_dict)
                    # 模型返回 pred_mel (注意: tensor 不能用 or, 用显式 if)
                    mel = out.get("pred_mel")
                    if mel is None:
                        mel = out.get("mel_pred")
                    if mel is None or samples_done >= self.n_samples:
                        continue
                    # 转为 wav (placeholder 或真实 vocoder)
                    wav = self._mel_to_wav(mel[0])
                    if wav is not None and len(wav) > 100:
                        rms = float(np.sqrt(np.mean(wav ** 2)))
                        if rms > 1e-4:
                            rms_list.append(rms)
                            # Spectral centroid
                            spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
                            freqs = np.fft.rfftfreq(len(wav), 1 / 22050)
                            if spec.sum() > 0:
                                centroid = float(np.sum(freqs * spec) / np.sum(spec))
                                centroid_list.append(centroid)
                            # M3.6: 提取 F0
                            f0 = self._extract_f0(wav)
                            if len(f0) > 5:
                                f0_pred_means.append(float(np.mean(f0)))
                                f0_pred_stds.append(float(np.std(f0)))
                    samples_done += 1
                except Exception as e:
                    import traceback
                    print(f"  [WavQuality] inference failed: {type(e).__name__}: {e}")
                    if "Boolean" in str(e) or "ambiguous" in str(e):
                        traceback.print_exc()
                    continue

        if not rms_list:
            return {}
        rms_mean = float(np.mean(rms_list))
        centroid_mean = float(np.mean(centroid_list)) if centroid_list else 0.0
        # wav_quality = rms * (1 - |centroid - 4500| / 4500)
        # rms 越大越好, 越接近 4500Hz 越好
        centroid_score = max(0.0, 1.0 - abs(centroid_mean - 4500) / 4500)
        wav_quality = rms_mean * centroid_score

        result = {
            "wav_rms_mean": round(rms_mean, 4),
            "wav_centroid_mean": round(centroid_mean, 1),
            "wav_quality": round(wav_quality, 4),
        }

        # M3.6: F0 指标
        if f0_pred_means:
            f0_pred_mean = float(np.mean(f0_pred_means))
            f0_pred_std = float(np.mean(f0_pred_stds))
            result["f0_pred_mean"] = round(f0_pred_mean, 1)
            result["f0_pred_std"] = round(f0_pred_std, 1)
            if f0_ref_mean is not None:
                # F0 偏差 (Hz), 越接近 ref 越好
                f0_bias = abs(f0_pred_mean - f0_ref_mean)
                result["f0_ref_mean"] = round(f0_ref_mean, 1)
                result["f0_bias_hz"] = round(f0_bias, 1)
                # F0 稳定性: std 越低越好 (但人声会有自然变化, 这里用 50Hz 归一化)
                f0_stability = max(0.0, 1.0 - f0_pred_std / 50.0)
                result["f0_stability"] = round(f0_stability, 4)
                # naturalness_score = wav_quality * (1 - f0_bias/200) * f0_stability
                f0_score = max(0.0, 1.0 - f0_bias / 200.0) * f0_stability
                result["f0_score"] = round(f0_score, 4)
                result["naturalness"] = round(wav_quality * f0_score, 4)

        return result

    def _get_vocoder(self):
        """Lazy load + 缓存真实 vocoder (避免每 epoch 重新加载)。"""
        if self._vocoder_cache is not None or self._vocoder_load_failed:
            return self._vocoder_cache
        try:
            from adr.vocoder.bigvgan import BigVGANVocoder
            path = self.vocoder_path or "auto"
            self._vocoder_cache = BigVGANVocoder.from_pretrained(path, device="auto")
            print(f"  [WavQuality] Vocoder loaded: {type(self._vocoder_cache).__name__}")
        except Exception as e:
            self._vocoder_load_failed = True
            print(f"  [WavQuality] Vocoder load failed: {type(e).__name__}: {e}")
            return None
        return self._vocoder_cache

    def _mel_to_wav(self, mel):
        """mel → wav (placeholder 或真实 vocoder, 真实 vocoder 缓存)。"""
        import numpy as np
        if self.use_placeholder_vocoder or mel is None:
            # placeholder: 用 mel 能量做相位重建 (粗略但能给 RMS/centroid 信号)
            mel = mel.detach().cpu().numpy() if hasattr(mel, "detach") else mel
            # 反 STFT 不可行, 用 mel 能量调制白噪声 (给真实 wav 频谱)
            mel = mel.mean(axis=0) if mel.ndim == 2 else mel  # (T,)
            n_samples = len(mel) * 256
            if n_samples < 1000:
                n_samples = 1000
            t = np.linspace(0, len(mel) * 256 / 22050, n_samples)
            # 基础波形: 180Hz 的人声基频 + 谐波
            fundamental = 0.5 * np.sin(2 * np.pi * 180 * t)
            # 调制 mel 能量
            envelope = np.interp(
                np.linspace(0, len(mel), n_samples),
                np.arange(len(mel)),
                mel / (mel.max() + 1e-8),
            )
            wav = fundamental * envelope * 0.3
            return wav.astype(np.float32)
        # 真实 vocoder 调用 (懒加载 + 缓存)
        vocoder = self._get_vocoder()
        if vocoder is None or not vocoder.is_loaded():
            return None
        try:
            mel_t = mel.unsqueeze(0) if mel.ndim == 2 else mel
            wav = vocoder.infer(mel_t)
            return wav.squeeze().cpu().numpy() if hasattr(wav, "cpu") else wav
        except Exception as e:
            print(f"  [WavQuality] vocoder.infer failed: {type(e).__name__}: {e}")
            return None

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self.should_stop:
            return
        wav_metrics = self._compute_wav_metrics(trainer)
        if not wav_metrics:
            print(f"  [WavQuality] epoch {epoch+1}: skipped (no val samples)")
            return

        # 合并到 metrics
        metrics.update(wav_metrics)
        self._history.append(wav_metrics)
        value = wav_metrics.get(self.metric_name)
        if value is None:
            return

        if self.best_value is None:
            self.best_value = value
            print(f"  [WavQuality] epoch {epoch+1}: {wav_metrics} (init)")
            return

        improved = (
            (self.mode == "max" and value > self.best_value + self.min_delta)
            or (self.mode == "min" and value < self.best_value - self.min_delta)
        )
        if improved:
            self.best_value = value
            self.counter = 0
            print(f"  [WavQuality] epoch {epoch+1}: {wav_metrics} (best)")
        else:
            self.counter += 1
            print(
                f"  [WavQuality] epoch {epoch+1}: {wav_metrics} "
                f"(no improve {self.counter}/{self.patience}, best={self.best_value:.4f})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                print(
                    f"  [WavQuality] STOP at epoch {epoch+1}, "
                    f"best {self.metric_name}={self.best_value:.4f}"
                )


# ============================================================
# M4: WER/CER 计算 + ASRCallback
# ============================================================
def _edit_distance(ref_tokens, hyp_tokens) -> int:
    """Levenshtein 编辑距离 (动态规划, 纯 Python)。

    Args:
        ref_tokens: 参考序列 (list of str)
        hyp_tokens: 假设序列 (list of str)

    Returns:
        编辑距离 (替换/删除/插入)
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0:
        return m
    if m == 0:
        return n

    # dp[i][j] = ref[:i] -> hyp[:j] 的最小编辑距离
    # 滚动数组省内存
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # 删除 ref[i-1]
                curr[j - 1] + 1,    # 插入 hyp[j-1]
                prev[j - 1] + cost, # 替换
            )
        prev, curr = curr, prev
    return prev[m]


def _normalize_text(s: str, strip_punct: bool = True) -> list:
    """文本归一化: 去空格/标点, 返回字符列表。

    用于 CER/WER 计算前的预处理。TTS 评估场景下标点通常不算"词",
    因为 TTS 不读出标点。OpenCpop 数据集也没有标点。
    """
    import string
    # 中文 + 英文常见标点
    punct_set = set(string.punctuation) | set("。，！？；：、""''「」『』（）【】《》〈〉…—·")
    out = []
    for c in s:
        if c.isspace():
            continue
        if strip_punct and c in punct_set:
            continue
        out.append(c)
    return out


def compute_cer(reference: str, hypothesis: str) -> float:
    """计算字符错误率 (Character Error Rate, CER)。

    适用于中文等无空格分隔的语言:
    - 将字符串转为字符列表 (去除空格和标点)
    - 计算编辑距离
    - CER = 编辑距离 / 参考字符数

    Args:
        reference: 参考文本 (真实)
        hypothesis: 假设文本 (ASR 输出)

    Returns:
        CER, 范围 [0, +∞), 0 表示完全匹配
    """
    ref_chars = _normalize_text(reference)
    hyp_chars = _normalize_text(hypothesis)

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist = _edit_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def compute_wer(reference: str, hypothesis: str) -> float:
    """计算词错误率 (Word Error Rate, WER)。

    适用于英文等有空格的文本:
    - 按空格分词, 去标点
    - 计算编辑距离
    - WER = 编辑距离 / 参考词数

    Args:
        reference: 参考文本
        hypothesis: 假设文本

    Returns:
        WER, 范围 [0, +∞), 0 表示完全匹配
    """
    import re
    # 简单分词 + 去标点
    ref_words = [w for w in re.split(r"\s+", reference) if w.strip()]
    hyp_words = [w for w in re.split(r"\s+", hypothesis) if w.strip()]

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = _edit_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def compute_text_similarity(reference: str, hypothesis: str, lang: str = "auto") -> dict:
    """计算文本相似度 (自动选 WER/CER)。

    Args:
        reference: 参考文本
        hypothesis: 假设文本
        lang: "zh" 用 CER, "en" 用 WER, "auto" 自动检测 (含中文 → CER)

    Returns:
        dict: {"rate": 0.123, "mode": "cer"/"wer", "n_ref": 10, "dist": 1}
    """
    # 自动检测: 含中文字符 → CER, 否则 WER
    if lang == "auto":
        has_cjk = any("一" <= c <= "鿿" for c in reference) or \
                  any("぀" <= c <= "ヿ" or "가" <= c <= "힯" for c in reference)
        lang = "zh" if has_cjk else "en"

    if lang in ("zh", "ja", "ko"):
        rate = compute_cer(reference, hypothesis)
        mode = "cer"
        n_ref = len(_normalize_text(reference))
    else:
        rate = compute_wer(reference, hypothesis)
        mode = "wer"
        n_ref = len(reference.split())

    if mode == "cer":
        ref_tokens = _normalize_text(reference)
        hyp_tokens = _normalize_text(hypothesis)
    else:
        ref_tokens = [w for w in reference.split() if w.strip()]
        hyp_tokens = [w for w in hypothesis.split() if w.strip()]
    dist = _edit_distance(ref_tokens, hyp_tokens)

    return {
        "rate": round(rate, 4),
        "mode": mode,
        "n_ref": n_ref,
        "dist": dist,
    }


class ASRCallback(Callback):
    """ASR 早停回调 (M4)。

    设计动机: Wav 质量指标 (RMS/centroid/F0) 是信号层面的客观度量,
    但**可懂度 (intelligibility)** 才是 TTS 的终极目标。ASR 转录率 (WER/CER)
    直接衡量"合成 wav 被自动语音识别系统听对的程度",是更贴近用户感知的指标。

    工作流 (每个 epoch 结束):
    1. 取 N 个 val sample, 拿 reference text (来自 dataset.texts)
    2. 跑推理 → pred_mel → vocoder → wav
    3. ASR 转录 wav → hypothesis text
    4. compute_cer(reference, hypothesis) → CER
    5. 若 CER 在 patience 个 epoch 内未改善 (mode="min"), 触发早停

    Args:
        n_samples: 评估样本数 (默认 5)
        n_timesteps: 推理去噪步数 (默认 10)
        asr_model_size: ASR 模型大小 ("tiny"/"base"/"small", 默认 "tiny" 速度快)
        asr_language: ASR 语言 ("zh"/"en"/None=auto)
        use_real_vocoder: 是否用 BigVGAN (False=placeholder)
        vocoder_path: BigVGAN 路径 (None=自动搜索)
        metric_name: 主指标 ("asr_cer" 或 "asr_wer")
        mode: "min" (CER/WER 越小越好)
        patience: 多少 epoch 不改善就停
        min_delta: 视为"改善"的最小幅度
        asr_model_sizes: M7 多模型集成列表 (如 ["tiny", "small"]),
            None → 单模型模式 (用 asr_model_size)。集成模式下:
            - 每个模型独立转录 → medoid 共识 (与其他假设平均编辑距离最小者)
            - 输出 asr_cer_median / asr_cer_std / asr_agreement 统计指标
            - 单模型失败降级为部分集成, 不影响整体评估

    用法:
        >>> cb = ASRCallback(n_samples=5, patience=2, asr_model_size="tiny")
        >>> trainer = Trainer(model, ds, config, callbacks=[cb])
        >>> trainer.fit()

        >>> # M7 集成模式
        >>> cb = ASRCallback(asr_model_sizes=["tiny", "small"], patience=2)

    性能:
        - tiny ASR: ~10s/epoch (CPU, 5 samples)
        - small ASR: ~30s/epoch (CPU, 5 samples)
        - base ASR: ~1min/epoch (CPU, 5 samples)
        - GPU: 1-5s/epoch (small)
    """

    # use_real_vocoder 旧别名, 保留兼容
    _USE_REAL_DEFAULT = True

    def __init__(
        self,
        n_samples: int = 5,
        n_timesteps: int = 10,
        asr_model_size: str = "tiny",
        asr_language: Optional[str] = "zh",
        use_real_vocoder: bool = True,
        vocoder_path: Optional[str] = None,
        metric_name: str = "asr_cer",
        mode: str = "min",
        patience: int = 2,
        min_delta: float = 0.01,
        use_placeholder_vocoder: Optional[bool] = None,
        asr_model_sizes: Optional[list] = None,
    ):
        self.n_samples = n_samples
        self.n_timesteps = n_timesteps
        self.asr_model_size = asr_model_size
        # M7: 多 ASR 集成 (None → 单模型兼容模式)
        self.asr_model_sizes = list(asr_model_sizes) if asr_model_sizes else [asr_model_size]
        self.ensemble = len(self.asr_model_sizes) > 1
        self.asr_language = asr_language
        # 兼容性: use_placeholder_vocoder 是 use_real_vocoder 的旧别名 (取反)
        # 旧代码可能传 use_placeholder_vocoder=True 表示"用 placeholder"
        if use_placeholder_vocoder is not None:
            self.use_real_vocoder = not use_placeholder_vocoder
        else:
            self.use_real_vocoder = use_real_vocoder
        self.vocoder_path = vocoder_path
        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        self._history: list = []
        # 懒加载
        self._vocoder_cache = None
        self._vocoder_load_failed = False
        self._asr_cache = None  # 主模型缓存 (兼容旧属性)
        self._asr_load_failed = False
        # M7: 按模型 size 分别缓存 / 失败标记 / CPU 回退标记
        self._asr_caches: dict = {}
        self._asr_failed: set = set()
        self._asr_cpu_fallback_set: set = set()

    def _get_vocoder(self):
        """懒加载 vocoder (复用 WavQualityCallback 模式)。"""
        if self._vocoder_cache is not None or self._vocoder_load_failed:
            return self._vocoder_cache
        try:
            from adr.vocoder.bigvgan import BigVGANVocoder
            path = self.vocoder_path or "auto"
            self._vocoder_cache = BigVGANVocoder.from_pretrained(path, device="auto")
        except Exception as e:
            self._vocoder_load_failed = True
            print(f"  [ASR] Vocoder load failed: {type(e).__name__}: {e}")
            return None
        return self._vocoder_cache

    def _get_asr(self, force_cpu: bool = False, size: Optional[str] = None):
        """懒加载 ASR (faster-whisper), M7 支持按模型 size 分别缓存。

        Args:
            force_cpu: CUDA 推理失败 (如缺 cublas) 时强制回退 CPU int8
            size: 模型大小 (None=主模型 self.asr_model_sizes[0])
        """
        size = size or self.asr_model_sizes[0]
        if size in self._asr_caches:
            return self._asr_caches[size]
        if size in self._asr_failed:
            return None
        try:
            from faster_whisper import WhisperModel
            import torch
            device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
            compute_type = "float16" if device == "cuda" else "int8"
            model = WhisperModel(size, device=device, compute_type=compute_type)
            self._asr_caches[size] = model
            if size == self.asr_model_sizes[0]:
                self._asr_cache = model  # 兼容旧属性
            print(f"  [ASR] Loaded: {size} on {device}")
        except Exception as e:
            self._asr_failed.add(size)
            if size == self.asr_model_sizes[0]:
                self._asr_load_failed = True
            print(f"  [ASR] Load failed ({size}): {type(e).__name__}: {e}")
            print(f"  [ASR] Install: pip install faster-whisper")
            return None
        return self._asr_caches[size]

    def _reload_asr_cpu(self, size: Optional[str] = None):
        """CUDA 转录失败时, 重新加载 CPU 版 ASR (每个模型一次性回退)。"""
        size = size or self.asr_model_sizes[0]
        if size in self._asr_cpu_fallback_set:
            return None  # 已回退过, 不再重试
        self._asr_cpu_fallback_set.add(size)
        self._asr_caches.pop(size, None)
        if size == self.asr_model_sizes[0]:
            self._asr_cache = None
            self._asr_cpu_fallback = True  # 兼容旧属性
        print(f"  [ASR] CUDA transcribe failed ({size}), fallback to CPU int8")
        return self._get_asr(force_cpu=True, size=size)

    def _mel_to_wav(self, mel):
        """mel → wav。"""
        import numpy as np
        if not self.use_real_vocoder or mel is None:
            # placeholder: 简单正弦 + 调制 (复用 WavQualityCallback 逻辑)
            mel = mel.detach().cpu().numpy() if hasattr(mel, "detach") else mel
            mel = mel.mean(axis=0) if mel.ndim == 2 else mel
            n_samples = max(1000, len(mel) * 256)
            t = np.linspace(0, len(mel) * 256 / 22050, n_samples)
            fundamental = 0.5 * np.sin(2 * np.pi * 180 * t)
            envelope = np.interp(
                np.linspace(0, len(mel), n_samples),
                np.arange(len(mel)),
                mel / (mel.max() + 1e-8),
            )
            return (fundamental * envelope * 0.3).astype(np.float32)
        # 真实 vocoder
        vocoder = self._get_vocoder()
        if vocoder is None or not vocoder.is_loaded():
            return None
        try:
            mel_t = mel.unsqueeze(0) if mel.ndim == 2 else mel
            wav = vocoder.infer(mel_t)
            return wav.squeeze().cpu().numpy() if hasattr(wav, "cpu") else wav
        except Exception as e:
            print(f"  [ASR] vocoder.infer failed: {e}")
            return None

    def _asr_transcribe(self, wav, sr=22050, size: Optional[str] = None) -> str:
        """wav → text (CUDA 失败自动回退 CPU)。

        Args:
            size: ASR 模型 size (None=主模型), M7 集成时逐个模型调用
        """
        asr = self._get_asr(size=size)
        if asr is None:
            return ""
        try:
            import tempfile
            import soundfile as sf
            # faster-whisper 需要 16kHz
            target_sr = 16000
            if sr != target_sr:
                import scipy.signal
                wav_16k = scipy.signal.resample(wav, int(len(wav) * target_sr / sr))
            else:
                wav_16k = wav
            # 写临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                sf.write(tmp_path, wav_16k.astype("float32"), target_sr)
            try:
                text = self._run_transcribe(asr, tmp_path, size=size)
            finally:
                import os
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            return text.strip()
        except Exception as e:
            print(f"  [ASR] transcribe failed ({size or 'primary'}): {type(e).__name__}: {e}")
            return ""

    def _run_transcribe(self, asr, tmp_path: str, size: Optional[str] = None) -> str:
        """执行转录, RuntimeError (如缺 cublas) 时回退 CPU 重试一次。"""
        try:
            segments, info = asr.transcribe(
                tmp_path,
                language=self.asr_language,
                beam_size=3,  # 速度优先
                vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments)
        except RuntimeError as e:
            # cublas/cudnn 缺失等 CUDA 运行时错误 → 回退 CPU
            cpu_asr = self._reload_asr_cpu(size=size)
            if cpu_asr is None:
                raise
            segments, info = cpu_asr.transcribe(
                tmp_path,
                language=self.asr_language,
                beam_size=3,
                vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments)

    def _asr_transcribe_ensemble(self, wav, sr=22050) -> list:
        """M7: 用所有 ASR 模型转录同一 wav, 返回 [(size, text), ...]。

        单个模型失败不影响其他模型 (降级为部分集成)。
        """
        results = []
        for size in self.asr_model_sizes:
            text = self._asr_transcribe(wav, sr=sr, size=size)
            if text:
                results.append((size, text))
        return results

    def _consensus(self, hyps: list) -> tuple:
        """M7: medoid 共识 — 选与其他假设平均归一化编辑距离最小者。

        Args:
            hyps: 假设文本列表 [(size, text), ...] 或 [text, ...]

        Returns:
            (consensus_text, agreement):
                consensus_text: medoid 假设
                agreement: 模型间一致性 [0,1], 1 = 完全一致
        """
        texts = [t for _, t in hyps] if hyps and isinstance(hyps[0], tuple) else list(hyps)
        if not texts:
            return "", 0.0
        if len(texts) == 1:
            return texts[0], 1.0
        n = len(texts)
        norm = [_normalize_text(t) for t in texts]
        avg_dists = []
        for i in range(n):
            total = 0.0
            for j in range(n):
                if i == j:
                    continue
                d = _edit_distance(norm[i], norm[j])
                denom = max(len(norm[i]), len(norm[j]), 1)
                total += d / denom
            avg_dists.append(total / (n - 1))
        best_i = min(range(n), key=lambda i: avg_dists[i])
        agreement = max(0.0, 1.0 - sum(avg_dists) / n)
        return texts[best_i], round(agreement, 4)

    def _compute_cer_metrics(self, trainer) -> dict:
        """在 val set 上跑推理 + ASR, 计算 CER/WER。"""
        import numpy as np
        import torch

        trainer.model.eval()
        device = trainer.device
        if trainer.val_loader is None:
            return {}

        # 收集 val samples 的 (text, mel) 配对
        samples = []
        try:
            for batch in trainer.val_loader:
                if len(samples) >= self.n_samples:
                    break
                # batch.texts 来自 dataset, 是 List[str]
                if not hasattr(batch, "texts"):
                    continue
                for i, text in enumerate(batch.texts):
                    if len(samples) >= self.n_samples:
                        break
                    if not text or not text.strip():
                        continue
                    samples.append({
                        "text": text.strip(),
                        "phoneme_ids": batch.phoneme_ids[i],
                        "phoneme_mask": batch.phoneme_mask[i],
                        "ref_mel": batch.ref_mel[i],
                    })
        except Exception as e:
            print(f"  [ASR] val_loader iteration failed: {e}")
            return {}

        if not samples:
            return {}

        # 推理每个 sample
        cer_list = []
        wer_list = []
        cer_details = []  # 详细对比 (前 3 个)
        ensemble_stats = {}  # M7: 集成统计 (median/std/agreement)

        for idx, s in enumerate(samples[:self.n_samples]):
            try:
                batch_dict = {
                    "phoneme_ids": s["phoneme_ids"].unsqueeze(0).to(device),
                    "phoneme_mask": s["phoneme_mask"].unsqueeze(0).to(device),
                    "ref_mel": s["ref_mel"].unsqueeze(0).to(device),
                }
                with torch.no_grad():
                    out = trainer.model(batch_dict)
                mel = out.get("pred_mel")
                if mel is None:
                    mel = out.get("mel_pred")
                if mel is None:
                    continue
                wav = self._mel_to_wav(mel[0])
                if wav is None or len(wav) < 100:
                    continue

                # ASR (M7: 单模型或集成)
                ref_text = s["text"]
                if self.ensemble:
                    hyps = self._asr_transcribe_ensemble(wav)
                    if not hyps:
                        continue
                    hyp_text, agreement = self._consensus(hyps)
                    # 每个模型的 CER (用于统计离散度)
                    per_model_cer = [
                        compute_text_similarity(ref_text, t, lang="auto")["rate"]
                        for _, t in hyps
                    ]
                else:
                    hyp_text = self._asr_transcribe(wav)
                    agreement = None
                    per_model_cer = None
                sim = compute_text_similarity(ref_text, hyp_text, lang="auto")

                if sim["mode"] == "cer":
                    cer_list.append(sim["rate"])
                else:
                    wer_list.append(sim["rate"])

                # M7: 集成统计
                if self.ensemble and per_model_cer:
                    ensemble_stats.setdefault("median", []).append(
                        float(sorted(per_model_cer)[len(per_model_cer) // 2])
                    )
                    ensemble_stats.setdefault("std", []).append(
                        float(np.std(per_model_cer))
                    )
                    if agreement is not None:
                        ensemble_stats.setdefault("agreement", []).append(agreement)

                if idx < 3:
                    detail = {
                        "ref": ref_text[:40],
                        "hyp": hyp_text[:40],
                        "rate": sim["rate"],
                        "mode": sim["mode"],
                    }
                    if self.ensemble:
                        detail["n_models"] = len(hyps)
                        detail["agreement"] = agreement
                    cer_details.append(detail)
            except Exception as e:
                print(f"  [ASR] sample {idx} failed: {type(e).__name__}: {e}")
                continue

        if not cer_list and not wer_list:
            return {}

        result = {}
        if cer_list:
            result["asr_cer"] = round(float(sum(cer_list) / len(cer_list)), 4)
            result["asr_cer_n"] = len(cer_list)
        if wer_list:
            result["asr_wer"] = round(float(sum(wer_list) / len(wer_list)), 4)
            result["asr_wer_n"] = len(wer_list)
        # 综合 score: 优先用 cer
        primary = result.get(self.metric_name) or result.get("asr_cer") or result.get("asr_wer", 1.0)
        result["asr_primary"] = round(primary, 4)
        # M7: 集成统计指标
        if ensemble_stats.get("median"):
            result["asr_cer_median"] = round(
                float(sum(ensemble_stats["median"]) / len(ensemble_stats["median"])), 4)
        if ensemble_stats.get("std"):
            result["asr_cer_std"] = round(
                float(sum(ensemble_stats["std"]) / len(ensemble_stats["std"])), 4)
        if ensemble_stats.get("agreement"):
            result["asr_agreement"] = round(
                float(sum(ensemble_stats["agreement"]) / len(ensemble_stats["agreement"])), 4)
        if cer_details:
            result["asr_samples"] = cer_details

        return result

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self.should_stop:
            return
        cer_metrics = self._compute_cer_metrics(trainer)
        if not cer_metrics:
            print(f"  [ASR] epoch {epoch+1}: skipped (no val samples or ASR unavailable)")
            return

        # 拆出数值指标 (放 metrics) 和详细列表 (放 _history)
        details = cer_metrics.pop("asr_samples", None)
        metrics.update(cer_metrics)
        history_entry = dict(cer_metrics)
        if details:
            history_entry["asr_samples"] = details
        self._history.append(history_entry)
        value = cer_metrics.get(self.metric_name)
        if value is None:
            value = cer_metrics.get("asr_primary")

        if value is None:
            return

        if self.best_value is None:
            self.best_value = value
            print(f"  [ASR] epoch {epoch+1}: {self.metric_name}={value:.4f} (init)")
            if details:
                for d in details:
                    print(f"         ref='{d['ref']}' -> hyp='{d['hyp']}' ({d['mode']}={d['rate']:.3f})")
            return

        improved = value < self.best_value - self.min_delta
        if improved:
            self.best_value = value
            self.counter = 0
            print(f"  [ASR] epoch {epoch+1}: {self.metric_name}={value:.4f} (best)")
        else:
            self.counter += 1
            print(
                f"  [ASR] epoch {epoch+1}: {self.metric_name}={value:.4f} "
                f"(no improve {self.counter}/{self.patience}, best={self.best_value:.4f})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                print(
                    f"  [ASR] STOP at epoch {epoch+1}, "
                    f"best {self.metric_name}={self.best_value:.4f}"
                )


# ============================================================
# M6: WarmRestartCallback (SGDR 式热重启)
# ============================================================
class WarmRestartCallback(Callback):
    """指标平台期触发的学习率热重启 (M6)。

    设计动机: 训练中质量指标 (CER/wav_quality/val_loss) 常会陷入平台期,
    此时 LR 已衰减到很低, 模型没有足够步长跳出局部最优。SGDR 式
    热重启 (warm restart) 把 LR 重置到衰减后的峰值并重新 warmup,
    给模型一次"再冲刺"的机会。

    与早停的区别:
    - 早停: 指标不改善 → 停止训练 (认输, 保住 best)
    - 热重启: 指标不改善 → LR 重启再战 (最多 max_restarts 次),
      全部重启耗尽后仍不改善, 则由早停 callback 兜底

    工作流 (每个 epoch 结束, 应在质量 callback 之后执行):
    1. 读 metrics[metric_name] (需先由 ASR/WavQuality callback 写入)
    2. patience 个 epoch 未改善 → 触发重启:
       - 新 peak_lr = 当前 lr × lr_decay (不低于 min_lr)
       - 重建 LambdaLR (warmup_cosine), 重新 warmup
       - remaining_steps = 剩余 epoch × steps/epoch
    3. 重启次数达 max_restarts 后不再触发

    Args:
        metric_name: 监控指标 ("asr_cer" / "wav_quality" / "val/loss")
        mode: "min" (CER/loss) 或 "max" (wav_quality)
        patience: 多少 epoch 不改善触发重启
        min_delta: 视为"改善"的最小幅度
        lr_decay: 重启时 LR 衰减系数 (默认 0.5, 即每次减半)
        warmup_steps: 重启后的 warmup 步数
        max_restarts: 最大重启次数 (默认 2)
        min_lr: LR 下限, 低于此值不再重启

    用法:
        >>> asr_cb = ASRCallback(...)          # 先计算指标
        >>> wr_cb = WarmRestartCallback(        # 后消费指标
        ...     metric_name="asr_cer", patience=2, lr_decay=0.5)
        >>> trainer = Trainer(..., callbacks=[asr_cb, wr_cb])

    注意: callbacks 列表中, 本 callback 必须排在产生指标的 callback 之后。
    """

    def __init__(
        self,
        metric_name: str = "asr_cer",
        mode: str = "min",
        patience: int = 2,
        min_delta: float = 0.01,
        lr_decay: float = 0.5,
        warmup_steps: int = 10,
        max_restarts: int = 2,
        min_lr: float = 1e-6,
    ):
        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.lr_decay = lr_decay
        self.warmup_steps = warmup_steps
        self.max_restarts = max_restarts
        self.min_lr = min_lr
        self.best_value: Optional[float] = None
        self.counter = 0
        self.n_restarts = 0
        self._exhausted = False  # min_lr 触底, 永久停用
        self._history: list = []

    def _do_restart(self, trainer, epoch: int) -> None:
        """重建 scheduler: 新峰值 = 当前 lr × lr_decay, 重新 warmup。"""
        from adr.training.optimizer import OptimizerConfig, build_scheduler

        cur_lr = trainer.optimizer.param_groups[0]["lr"]
        new_lr = max(cur_lr * self.lr_decay, self.min_lr)
        # LambdaLR 以 param_groups 的 lr 为 base_lr
        for g in trainer.optimizer.param_groups:
            g["lr"] = new_lr
            g["initial_lr"] = new_lr
        remaining_epochs = max(1, trainer.config.epochs - epoch - 1)
        total_steps = max(1, len(trainer.train_loader) * remaining_epochs)
        cfg = OptimizerConfig(
            lr=new_lr,
            warmup_steps=min(self.warmup_steps, total_steps),
            scheduler="warmup_cosine",
        )
        trainer.scheduler = build_scheduler(trainer.optimizer, cfg, total_steps=total_steps)
        self.n_restarts += 1
        self.counter = 0
        print(
            f"  [WarmRestart] #{self.n_restarts}/{self.max_restarts} "
            f"at epoch {epoch+1}: lr {cur_lr:.2e} -> {new_lr:.2e} "
            f"(re-warmup {cfg.warmup_steps} steps, total {total_steps})"
        )
        self._history.append({
            "epoch": epoch + 1,
            "restart_n": self.n_restarts,
            "lr_before": cur_lr,
            "lr_after": new_lr,
        })

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self._exhausted or self.n_restarts >= self.max_restarts:
            return
        value = metrics.get(self.metric_name)
        if value is None:
            return

        if self.best_value is None:
            self.best_value = value
            return

        improved = (
            (self.mode == "min" and value < self.best_value - self.min_delta)
            or (self.mode == "max" and value > self.best_value + self.min_delta)
        )
        if improved:
            self.best_value = value
            self.counter = 0
            return

        self.counter += 1
        if self.counter >= self.patience:
            cur_lr = trainer.optimizer.param_groups[0]["lr"]
            if cur_lr * self.lr_decay < self.min_lr:
                print(
                    f"  [WarmRestart] skipped: lr {cur_lr:.2e} × {self.lr_decay} "
                    f"< min_lr {self.min_lr:.2e}, disabled"
                )
                self._exhausted = True  # 不再尝试
                return
            self._do_restart(trainer, epoch)
