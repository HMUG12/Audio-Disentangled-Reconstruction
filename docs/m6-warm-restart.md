# M6 学习率热重启 (Warm Restart / SGDR)

## 概述

ADR 框架的 **M6 模块** 实现了 **指标驱动的学习率热重启** (SGDR 式 warm restart):
当质量指标 (CER / wav_quality / val_loss) 陷入平台期时, 自动把 LR 重置到衰减后的峰值
并重新 warmup, 给模型一次"再冲刺"跳出局部最优的机会。

| 子模块 | 功能 | 状态 |
|--------|------|------|
| **M6** | WarmRestartCallback (平台期检测 + scheduler 重建) | ✅ |
| **M6.1** | CLI 集成 (`--warm-restart-*`) + 指标来源校验 | ✅ |

---

## 1. 动机: 平台期 ≠ 终点

**问题**: cosine 衰减后期 LR 趋近 0, 模型即使还没收敛好也"走不动"了:

```text
LR
 │╲
 │ ╲
 │  ╲___        ← cosine 尾段: LR≈0, CER 卡在 0.35 不动
 │      ─────
 └────────────→ epoch
```

此时有两个选择:
- **早停** (M3/M4): 认输, 保住 best checkpoint — 但可能模型还有潜力
- **热重启** (M6): LR 重新拉高再战 — SGDR 论文证明能跳出浅局部最优

```text
LR
 │╲      ╱╲
 │ ╲    ╱  ╲___     ← 重启: peak 减半 + 重新 warmup
 │  ╲__╱       ───
 └────────────────→ epoch
        ↑ CER 平台期触发
```

两者**互补不冲突**: M6 重启次数耗尽后, M3/M4 早停依然兜底。

---

## 2. 与早停的分工

| 机制 | 触发条件 | 动作 | 定位 |
|------|----------|------|------|
| 早停 (M1/M3/M4) | 指标 patience 轮不改善 | 停止训练, 用 best.pt | 防守 (防过拟合) |
| **热重启 (M6)** | 指标 patience 轮不改善 | LR 衰减重启, 继续训练 | 进攻 (榨取潜力) |

**推荐组合**: 先 M6 重启 (max_restarts=2), 耗尽后 M3/M4 早停兜底:

```bash
python scripts/smoke_train_opencpop.py \
    --asr-wer-patience 4 \        # M4 早停 (兜底线, 要比 M6 宽容)
    --warm-restart-patience 2 \   # M6 重启 (更敏感, 先触发)
    --warm-restart-metric asr_cer \
    --warm-restart-decay 0.5 \
    --max-restarts 2
```

**patience 设置原则**: M6 的 patience < 早停的 patience。
上例: CER 2 轮不降 → 重启; 重启后还是连续 4 轮不降 → 早停。

---

## 3. 架构

### 3.1 工作流

```
每个 epoch 结束 (须排在指标产生者之后):
  ↓
读 metrics[metric_name]  (由 ASR/WavQuality callback 先写入)
  ↓
指标缺失? → 安全跳过 (不报错)
  ↓
改善? → 更新 best, counter=0
  ↓
未改善 counter >= patience?
  ↓
min_lr 检查: 当前 lr × decay < min_lr → 永久停用 (_exhausted)
  ↓
触发重启:
  - new_lr = 当前 lr × lr_decay (≥ min_lr)
  - 重建 LambdaLR (warmup_cosine):
      base_lr = new_lr
      warmup = min(warmup_steps, remaining_steps)
      total = 剩余 epoch × steps/epoch
  - 替换 trainer.scheduler
  ↓
n_restarts >= max_restarts → 不再触发
```

### 3.2 scheduler 重建细节

LambdaLR 构造时以 `optimizer.param_groups[i]["lr"]` 为 base_lr, 因此:

1. 先把所有 param group 的 `lr` 和 `initial_lr` 改为 new_lr
2. 再 `build_scheduler(...)` 生成新的 warmup_cosine LambdaLR
3. 新 scheduler 从 step 0 开始 → LR 从 0 爬升到 new_lr (重新 warmup)

训练循环无需改动 — `trainer.scheduler` 是引用, 替换后下一步 `scheduler.step()`
自然走新曲线。

### 3.3 实现位置

[callbacks.py](file:///E:/新创意构思/新建文件夹/ADR/adr/training/callbacks.py) 的
`WarmRestartCallback` 类 (M6 段)。

---

## 4. CLI 用法

```bash
python scripts/smoke_train_opencpop.py \
    --data-dir data/opencpop_npz/train \
    --preset medium --n-samples 1000 --epochs 20 \
    --val-ratio 0.1 \
    --vocoder-path F:/ADR_data/bigvgan \
    --asr-wer-patience 4 \
    --warm-restart-patience 2 \
    --warm-restart-metric asr_cer \
    --warm-restart-decay 0.5 \
    --max-restarts 2
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--warm-restart-patience` | 0 (关) | 指标 N epoch 不改善触发重启 |
| `--warm-restart-metric` | `asr_cer` | 监控指标: `asr_cer` / `wav_quality` / `val/loss` |
| `--warm-restart-decay` | 0.5 | 每次重启 LR 减半 |
| `--max-restarts` | 2 | 最大重启次数 |

**指标来源校验**: 若选 `asr_cer` 但没开 `--asr-wer-patience`, CLI 会打印 WARNING
(热重启不会触发, 因为 metrics 里没有该指标)。

**训练日志示例**:

```text
=== Epoch 5/20 ===
  [ASR] epoch 5: asr_cer=0.3500 (no improve 2/2, best=0.3400)
  [WarmRestart] #1/2 at epoch 5: lr 8.5e-05 -> 4.3e-05 (re-warmup 10 steps, total 300)

=== Epoch 6/20 ===
  [ASR] epoch 6: asr_cer=0.3100 (best)   ← 重启后跳出平台期
```

---

## 5. 编程接口

```python
from adr.training.callbacks import ASRCallback, WarmRestartCallback

asr_cb = ASRCallback(n_samples=5, patience=4, asr_model_size="small")
wr_cb = WarmRestartCallback(
    metric_name="asr_cer",   # 监控 ASR CER
    mode="min",              # CER 越小越好
    patience=2,              # 2 epoch 不降就重启
    lr_decay=0.5,            # 峰值减半
    warmup_steps=10,         # 重启后重新 warmup 10 步
    max_restarts=2,          # 最多重启 2 次
    min_lr=1e-6,             # LR 下限
)

# 注意顺序: 指标产生者在前, 消费者在后
trainer = Trainer(model, ds, config, callbacks=[asr_cb, wr_cb])
trainer.fit()

# 查看重启历史
for h in wr_cb._history:
    print(h)  # {"epoch": 5, "restart_n": 1, "lr_before": 8.5e-5, "lr_after": 4.3e-5}
```

监控 `wav_quality` 时 `mode="max"`:

```python
wr_cb = WarmRestartCallback(metric_name="wav_quality", mode="max", patience=2)
```

---

## 6. 验证与测试

### 6.1 单元测试 (8 tests)

```bash
python -m pytest tests/test_warm_restart.py -v
```

| 测试 | 验证点 |
|------|--------|
| `test_restart_on_flat_metric` | 平台期触发重启, LR 逐次减半 |
| `test_max_restarts_respected` | patience=1 每轮可满足, 但最多重启 N 次 |
| `test_no_restart_when_improving` | 指标持续改善 → 不重启 |
| `test_scheduler_actually_replaced` | scheduler 对象被替换, 重新 warmup |
| `test_min_lr_guard` | LR 触底拒绝重启, `_exhausted` 永久停用 |
| `test_val_loss_metric` | 监控 val/loss 也可触发 |
| `test_skips_when_metric_missing` | 指标缺失安全跳过 |
| `test_history_records_lr` | `_history` 记录重启前后 LR |

### 6.2 CLI 端到端

已验证 (40 samples × 3 epochs, metric=val/loss):
val/loss 持续改善 (1.4943→1.4257) 时热重启**正确不触发**。

---

## 7. 故障排查

### 7.1 热重启从未触发

**排查顺序**:
1. CLI 是否有 `WARNING: metric 来源 callback 未启用`? → 补开对应 `--*-patience`
2. callbacks 顺序: `WarmRestartCallback` 必须排在指标产生者**之后**
3. 指标一直在改善 (好事!) — 看日志里 `(best)` 是否频繁出现
4. `min_delta` 是否太大? 默认 0.01, CER 场景 0.005-0.01 合理

### 7.2 重启后 loss 暴涨

**原因**: warmup_steps 太小, LR 瞬间拉太高。

**解决**: 增大 `warmup_steps` (默认 10), 或减小 `lr_decay` 幅度 (如 0.7)。

### 7.3 `_exhausted` 过早触发

**原因**: `min_lr` 设太高, 第一次重启前 LR 已低于 `min_lr / decay`。

**解决**: 检查初始 lr 与 min_lr 的比例 — 至少留出 `lr × decay^max_restarts > min_lr` 的空间。

---

## 8. 设计权衡

| 决策 | 选择 | 理由 |
|------|------|------|
| 重启峰值 | 当前 lr × decay (而非初始 lr) | SGDR 原版用固定峰值, 但 finetune 场景递减更稳 |
| 触发信号 | 复用 M3/M4 指标 | 不引入新计算开销, 与早停共享评估结果 |
| 停用机制 | `_exhausted` 独立标志 | 与实际重启次数解耦, 语义清晰 |
| scheduler | 直接替换对象 | 训练循环无需改动, 无状态污染 |

---

## 9. 相关文件

| 文件 | 作用 |
|------|------|
| [callbacks.py](file:///E:/新创意构思/新建文件夹/ADR/adr/training/callbacks.py) | WarmRestartCallback |
| [optimizer.py](file:///E:/新创意构思/新建文件夹/ADR/adr/training/optimizer.py) | build_scheduler (warmup_cosine) |
| [smoke_train_opencpop.py](file:///E:/新创意构思/新建文件夹/ADR/scripts/smoke_train_opencpop.py) | CLI `--warm-restart-*` |
| [test_warm_restart.py](file:///E:/新创意构思/新建文件夹/ADR/tests/test_warm_restart.py) | 8 个单元测试 |
| [m4-asr-wer.md](file:///E:/新创意构思/新建文件夹/ADR/docs/m4-asr-wer.md) | M4 ASR 早停 (指标来源) |
| [m3-wav-quality.md](file:///E:/新创意构思/新建文件夹/ADR/docs/m3-wav-quality.md) | M3 wav 早停 (指标来源) |

---

## 10. 参考文献

- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017
- Smith, "Cyclical Learning Rates for Training Neural Networks", WACV 2017
