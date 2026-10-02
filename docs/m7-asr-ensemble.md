# M7 多 ASR 集成投票 (Ensemble Voting)

## 概述

ADR 框架的 **M7 模块** 在 M4 单 ASR 评估的基础上, 引入 **多 ASR 模型集成**:
用多个不同规模的 ASR 模型独立转录同一合成 wav, 通过 **medoid 共识投票**
选出最可信的转录文本, 并输出模型间一致性指标。

| 子模块 | 功能 | 状态 |
|--------|------|------|
| **M7** | `asr_model_sizes` 列表 + per-size 缓存/回退 | ✅ |
| **M7.1** | medoid 共识 + `asr_agreement` 一致性 | ✅ |
| **M7.2** | 集成统计 (`asr_cer_median` / `asr_cer_std`) | ✅ |
| **M7.3** | 单模型失败降级 + CLI `--asr-ensemble` | ✅ |

---

## 1. 动机: 单 ASR 的评估噪声

**问题**: 单个 ASR 模型的转录错误会**污染** CER 指标:

```text
合成 wav (真实文本: "感受停在我发端的指尖")
  ↓
whisper-tiny 转录: "感受听在我发短的之间"  → CER 0.40
whisper-small 转录: "感受停在我发端的指尖"  → CER 0.00
```

CER 0.40 vs 0.00 — **同一个 wav**! 0.40 的差距里有多少是 TTS 的错, 多少是 ASR 的错?
单模型模式下无法区分。

**M7 思路**: 多模型独立转录 →
- 若各模型**意见一致** → 转录可信, CER 主要反映 TTS 质量
- 若各模型**意见分歧** → 音频本身模糊, CER 应谨慎解读

---

## 2. 核心机制

### 2.1 medoid 共识

不用简单多数投票 (字符级投票对长度不一的假设不适用), 而用 **medoid**:
选与其他所有假设平均归一化编辑距离最小的那个:

```text
hyps:  ["感受听在我发短的之间",   (tiny)
        "感受停在我发端的指尖",   (small)  
        "感受停在我发端的手尖"]   (base)]
           ↓ 两两编辑距离
tiny:   0 + 0.40 + 0.45 = 0.425 (avg)
small:  0.40 + 0 + 0.10 = 0.250 (avg) ← medoid
base:   0.45 + 0.10 + 0 = 0.275 (avg)
           ↓
consensus = "感受停在我发端的指尖" (small 的输出)
```

medoid 天然抗离群: 一个模型乱转录不会把共识带偏。

### 2.2 一致性指标 `asr_agreement`

```text
agreement = 1 - mean(所有假设的平均归一化距离)
```

| agreement | 含义 | 行动建议 |
|-----------|------|----------|
| ≥ 0.9 | 模型高度一致, CER 完全可信 | 正常早停/重启 |
| 0.6-0.9 | 基本一致, 有少量分歧 | 正常, 关注趋势 |
| < 0.6 | 严重分歧, 音频模糊或 ASR 不稳定 | CER 仅供参考, 结合 wav 指标判断 |

### 2.3 离散度 `asr_cer_std`

各模型 CER 的标准差。std 大说明 ASR 评估本身不稳定 — 此时即使 mean CER 下降
也可能是噪声, 早停应看 `asr_cer_median` (更稳健)。

---

## 3. 输出指标 (集成模式新增)

| 指标 | 含义 | 计算 |
|------|------|------|
| `asr_cer` | **共识 CER** (主指标) | CER(ref, medoid) |
| `asr_cer_median` | 各模型 CER 中位数 | median(CER_i) |
| `asr_cer_std` | 各模型 CER 标准差 | std(CER_i) |
| `asr_agreement` | 模型间一致性 | 1 - 平均归一化距离 |

`asr_samples` 详情新增 `n_models` 和 `agreement` 字段。

---

## 4. CLI 用法

```bash
python scripts/smoke_train_opencpop.py \
    --data-dir data/opencpop_npz/train \
    --preset medium --n-samples 1000 --epochs 20 \
    --val-ratio 0.1 \
    --vocoder-path F:/ADR_data/bigvgan \
    --asr-wer-patience 3 \
    --asr-ensemble tiny,small        # M7: 双模型集成
```

`--asr-ensemble` 覆盖 `--asr-model-size`。逗号分隔, 支持 2-N 个模型。

**推荐组合**:

| 场景 | ensemble | 每 epoch 开销 (5 samples, CPU) |
|------|----------|-------------------------------|
| smoke / 快速迭代 | `tiny,base` | ~40s |
| 中长时间训练 | `tiny,small` | ~70s |
| 生产精细调优 | `base,small,medium` | ~4min |

**与 M6 联动**: 共识 CER 同样可驱动热重启:

```bash
    --asr-ensemble tiny,small \
    --asr-wer-patience 4 \
    --warm-restart-patience 2 \
    --warm-restart-metric asr_cer
```

---

## 5. 编程接口

```python
from adr.training.callbacks import ASRCallback

# 单模型 (M4 模式, 完全兼容)
cb = ASRCallback(asr_model_size="tiny", patience=2)

# M7 集成模式
cb = ASRCallback(
    asr_model_sizes=["tiny", "small"],
    patience=3, asr_language="zh",
    use_real_vocoder=True,
    vocoder_path="F:/ADR_data/bigvgan",
)
trainer = Trainer(model, ds, config, callbacks=[cb])
trainer.fit()

# 查看集成历史
for h in cb._history:
    print(h["asr_cer"], h.get("asr_agreement"), h.get("asr_cer_std"))
```

### 5.1 降级行为

单个模型加载/转录失败**不影响**其他模型:

```text
[ASR] Loaded: tiny on cuda
[ASR] Load failed (small): RuntimeError: ...    ← small 挂了
[ASR] epoch 1: asr_cer=0.45 (init)              ← tiny 兜底, 正常评估
```

`hyps` 只剩 1 个时, consensus = 该假设, agreement = 1.0 (无分歧可言)。

### 5.2 medoid 共识独立使用

```python
cb = ASRCallback(use_real_vocoder=False)
text, agreement = cb._consensus([
    ("tiny", "感受听在我发短的之间"),
    ("small", "感受停在我发端的指尖"),
    ("base", "感受停在我发端的手尖"),
])
# text = "感受停在我发端的指尖" (与其他两者平均距离最小)
# agreement ≈ 0.83
```

---

## 6. 架构细节

### 6.1 per-size 缓存与回退

M4 的单一 `_asr_cache` 扩展为按 size 分别管理:

```python
self._asr_caches: dict[str, WhisperModel]  # {size: model}
self._asr_failed: set[str]                 # 加载失败的 size
self._asr_cpu_fallback_set: set[str]       # 已回退 CPU 的 size
```

- 每个模型独立懒加载, 首次使用时才下载/初始化
- CUDA cublas 缺失时**按模型**回退 CPU (一个模型回退不影响其他)
- 旧属性 `_asr_cache` / `_asr_load_failed` / `_asr_cpu_fallback`
  保留为主模型 (`asr_model_sizes[0]`) 的别名, 完全向后兼容

### 6.2 工作流

```
每个 val sample:
  ↓
ensemble? 
  ├─ 否: _asr_transcribe(wav) → hyp (M4 原路径)
  └─ 是: _asr_transcribe_ensemble(wav)
           ↓ 逐模型转录 (失败跳过)
         [(size, text), ...]
           ↓ _consensus (medoid)
         consensus_text + agreement
           ↓ 逐模型 CER
         per_model_cer → median/std
  ↓
CER(ref, hyp) → asr_cer
```

---

## 7. 验证与测试

### 7.1 单元测试 (15 tests)

```bash
python -m pytest tests/test_asr_ensemble.py -v
```

| 类别 | 测试 | 验证点 |
|------|------|--------|
| TestConsensus | identical / medoid / tuple / single / empty / all-different | medoid 选择 + agreement 边界 |
| TestEnsembleConstruction | 单模型/集成/覆盖/单元素列表 | 构造与 `ensemble` 标志 |
| TestEnsembleE2E | metrics_in_history / degrades / single_unchanged | 集成指标输出, 降级, 兼容 |
| TestEnsembleCache | per_size_cache / failed_size_marked | 缓存隔离, 失败标记 |

**向后兼容**: M4 的 34 个测试 (`test_asr_callback.py`) 全部通过, 无修改。

### 7.2 实测日志

```text
[ASR] Loaded: tiny on cuda
[ASR] Loaded: small on cuda
[ASR] epoch 1: asr_cer=0.5000 (init)
       ref='测试' -> hyp='测话' (cer=0.500)
Epoch 1 done: asr_cer=0.5000 asr_cer_median=0.5000 
              asr_cer_std=0.2500 asr_agreement=0.5000
```

---

## 8. 性能开销

集成模式下每个 sample 需转录 N 次:

| ensemble | 模型总大小 | 5 samples CPU | 5 samples GPU |
|----------|-----------|---------------|---------------|
| tiny | 39M | ~10s | ~2s |
| tiny,base | 113M | ~40s | ~5s |
| tiny,small | 283M | ~70s | ~8s |
| base,small,medium | 1.1G | ~4min | ~30s |

**建议**: GPU 环境放心用 `tiny,small`; CPU 环境用 `tiny,base` 或保持单 tiny。

---

## 9. 故障排查

### 9.1 `asr_agreement` 持续 < 0.5

**可能**: 合成音频确实模糊 (模型未收敛), 或某个 ASR 模型异常。

**排查**: 看 `asr_samples` 里各模型的 hyp — 若某个模型输出总是空/乱码,
从 ensemble 列表移除它。

### 9.2 集成后 CER 反而更高

**正常**: medoid 是"最中庸"的选择, 不一定是最准的。单模型模式下你可能
碰巧用了较准的那个模型。看 `asr_cer_median` 对比 — 若 median 也更低,
说明集成整体更可靠。

### 9.3 磁盘空间

每个 whisper 模型 39M-3G 不等, 集成 3 个模型约需 1-4GB (`~/.cache/huggingface/`)。

---

## 10. 设计权衡

| 决策 | 选择 | 理由 |
|------|------|------|
| 投票方式 | medoid (非字符级 ROVER) | ROVER 需对齐+加权, 复杂度高收益低; medoid 简洁抗离群 |
| 主指标 | 共识 CER (非 mean) | mean 被离群模型拉偏; 共识代表"多数意见" |
| 失败策略 | 降级 (非整体失败) | 单模型崩溃不应中断训练评估 |
| 缓存粒度 | per-size | 模型独立加载/回退, 互不干扰 |

---

## 11. 相关文件

| 文件 | 作用 |
|------|------|
| [callbacks.py](file:///E:/新创意构思/新建文件夹/ADR/adr/training/callbacks.py) | `asr_model_sizes` + `_consensus` + `_asr_transcribe_ensemble` |
| [smoke_train_opencpop.py](file:///E:/新创意构思/新建文件夹/ADR/scripts/smoke_train_opencpop.py) | CLI `--asr-ensemble` |
| [test_asr_ensemble.py](file:///E:/新创意构思/新建文件夹/ADR/tests/test_asr_ensemble.py) | 15 个集成测试 |
| [m4-asr-wer.md](file:///E:/新创意构思/新建文件夹/ADR/docs/m4-asr-wer.md) | M4 单 ASR 基础 |
| [m6-warm-restart.md](file:///E:/新创意构思/新建文件夹/ADR/docs/m6-warm-restart.md) | M6 热重启 (可消费共识 CER) |
