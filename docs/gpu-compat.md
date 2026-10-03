# GPU 兼容性矩阵 (批次 6-1 审计结论)

> 目标: 明确"NVIDIA/AMD/Intel GPU Windows 兼容"承诺边界。基于全仓 CUDA 依赖点审计 + 实测证据, 非猜测。

## 结论一览

| 功能 | NVIDIA CUDA | AMD / Intel GPU (Windows) | 纯 CPU |
|---|---|---|---|
| 微调训练 (s2/s1) | ✅ 全功能 (4GB 配方实测) | ❌ 不支持 | ❌ 不支持 |
| 数据准备 (切片/ASR/BERT/HuBERT/semantic) | ✅ | ⚠️ 仅切片+ASR(int8) 可跑; BERT/HuBERT half 特征已支持 CPU 降级 (is_half 动态) | ✅ 可跑 (慢, 一次性成本有缓存) |
| 推理合成 (整段/流式) | ✅ (峰值 1410MB) | ✅ 走 CPU 路径 | ✅ (短文本 RTF ≈1.9x; BERT ONNX 后 e2e 63s→6.7s) |
| 流式 TTS | ✅ 首包 ~2.2s | 同 CPU 路径 | ✅ (首包慢于 GPU) |
| 声纹门禁 (ERes2Net) | ✅ | ✅ (全 CPU 实现, 设备无关) | ✅ |
| 一键微调脚本 | ✅ | ⚠️ 数据准备可跑, 到训练步明确 abort 并提示 | 同左 |

## 审计明细 (关键依赖点)

| 依赖点 | 行为 | 状态 |
|---|---|---|
| `adr/core/device.py` | cuda→mps→cpu 自动降级, `capability` 分级提示 | ✅ |
| `scripts/gsv_finetune.py` | ASR 精度 float16→int8 / is_half 动态化 (批次6修复); 训练前 CUDA 守卫 | ✅ 已修 |
| ctranslate2 (faster-whisper) | **无静默回退**: cpu+float16 = LOAD FAIL; cpu+int8 正常 (RTF 0.07) | ✅ 已修 (动态精度) |
| FunASR (中文转写) | `device=cuda if available else cpu` 自动降级 | ✅ |
| `tools/my_utils.load_cudnn()` | 有 `cuda.is_available()` 保护, 无 GPU 跳过 | ✅ |
| GSV 推理引擎 | CPU 全链路可跑 (e2e 冒烟实测); `bert_onnx=auto` 仅 CPU 启用 ORT | ✅ |
| ERes2Net 门禁 | `map_location="cpu"`, 纯 CPU 推理 | ✅ 设备无关 |
| s2/s1 训练 | torch CUDA 训练栈 (amp/fp16_run/16-mixed), 无 AMD/Intel Windows 可靠路径 | ❌ 硬依赖 |

## 为什么不集成 DirectML / Intel XPU

1. **torch-directml**: 版本长期滞后主 torch (2.10 时代无对应版), 且 op 覆盖不全 — GSV 的动态 shape attention/掩码 op 大概率 fallback 或直接不支持, 集成=不可靠承诺, 违背"最小配置可用"原则。
2. **Intel XPU**: torch 2.10 XPU 支持面向特定驱动栈, 而 GSV 依赖链 (ctranslate2 / funasr / onnxruntime 各自的后端) 均无完整 XPU 支持, 单点能跑无意义。
3. **ONNX Runtime DirectML EP**: 理论上可加速 BERT 特征段 (仅此段), 收益局限, 列为未来可选实验, 不排期。

## 非 NVIDIA 用户的现实路径

- **有 NVIDIA 卡 (哪怕 4GB)**: 全功能。`adr check` 会显示 `全功能 (NVIDIA CUDA)`。
- **只有 AMD/Intel 卡**: 本机做 CPU 推理 + 门禁评分 (已实测可用); 训练用任意 NVIDIA 机器跑完把音色档案拷回 (档案=权重+ref, 可迁移), 或用云 GPU。
- 一键脚本在无 NVIDIA 环境跑到训练步会**明确 abort** 并提示 `--skip-to s2` 断点续跑方案 (数据准备产物已缓存, 不重做)。

## 验证方式

```bash
python -m adr.cli check          # 打印设备能力分级
python scripts/bench_lowres.py cpu   # CPU 推理基线 (已实测 RTF≈1.9x)
```

> 局限声明: 本机为 NVIDIA (RTX A2000), AMD/Intel 结论来自代码路径审计 + CPU 路径实测 + 依赖库官方文档, 未在真 AMD/Intel GPU 机器实测。CPU 路径即 AMD/Intel 用户的实际运行路径, 故覆盖等价。
