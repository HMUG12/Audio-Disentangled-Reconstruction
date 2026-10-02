# ADR — Audio-Disentangled Reconstruction

统一说话 + 歌声合成的自研模型框架。

## 当前状态：M0+（骨架已跑通）

- ✅ M0: Fish Speech 推理管线验证（M0 报告 [docs/m0-report.md](m0-report.md)）
- ✅ M0 异常诊断：379s 异常 = 无 prompt 失控，修复脚本 [scripts/m0_rerun_fix.py](../scripts/m0_rerun_fix.py) 已就绪，留待 M1 上云重跑
- ✅ M0+：自研代码骨架 5 模块跑通，验证脚本 [scripts/m0plus_verify.py](../scripts/m0plus_verify.py) ✅ 全部通过

## 包结构

```
adr/
├── __init__.py
├── configs.py                  # 全局配置（Content/Melody/Timbre/Backbone 四组 dataclass）
└── model/
    ├── __init__.py             # ADRModel 顶层 + forward / sample 接口
    ├── content_encoder.py      # 音素编码器
    ├── melody_encoder.py       # 旋律编码器（关键：含 null melody token 设计）
    ├── timbre_encoder.py       # 音色编码器
    └── backbone.py             # 共享 DiT + CFM backbone
```

## 核心设计要点

| 模块 | 借鉴来源 | 自研点 |
|---|---|---|
| **内容编码器** | DiffSinger 音素体系 | 从零训练的轻量 transformer |
| **旋律编码器** | UniVoice 共享架构 | **null melody token** 让说话模式边缘化旋律条件 |
| **音色编码器** | Vevo2 音色解耦 | 当前用 mel+CNN（占位），M1 接 WavLM |
| **Backbone** | F5-TTS 条件流匹配 + DiT 28层 | adaLN-Zero + 任务调制 token |
| **CFM 目标** | Fish Speech / F5-TTS | optimal transport CFM（midpoint integrator） |

## 快速验证

```powershell
# 训练 + 反向
third_party\fish-speech\.venv\Scripts\python.exe scripts\m0plus_verify.py
```

预期输出：
```
Param estimate: ~7M params
Actual params: 5.1M
=== 训练 forward + backward ===
Loss: 1.99
Backward OK
=== 说话模式推理 ===
Speech mel shape: torch.Size([1, 80, 384])
=== 唱歌模式推理 ===
Sing mel shape: torch.Size([1, 80, 270])
✅ 全部验证通过
```

## 当前局限（M1 之前无法解决）

1. **音素字典是占位**（vocab_size=512 随机 embedding）—— M1 需要接 DiffSinger 音素表
2. **音色编码器是简易 CNN**（mel → 嵌入）—— M1 替换为 WavLM 预训练特征
3. **无 vocoder** —— 当前输出 mel，M1 接 HiFi-GAN / BigVGAN 把 mel 转 wav
4. **0.3B 全量模型尚未实现** —— 当前 5.1M 是验证骨架用，全量 backbone 28×1024 需 6-8GB 显存推理 + 40GB+ 训练
5. **无真实数据集** —— 训练走随机 dummy 数据，loss 数字无意义，仅验证流程

## M1 任务清单（需 1×A800 24h）

### 准备（本周可做，无 GPU 需求）
- [ ] 接 DiffSinger 官方音素表（[DiffSinger/phonemes/opencpop](file:///third_party/DiffSinger)）替换 vocab=512 占位
- [ ] 下载 OpenCpop 100 首（中英双语音素+音符标注）作歌声 demo 数据
- [ ] 选型 WavLM-base+ 预训练权重，封装为 `TimbreEncoder` 真实实现
- [ ] 选型 vocoder（建议 BigVGAN-base）下载预训练权重

### 上云冒烟训练（24h 1×A800）
- [ ] 写数据加载器（Emilia 中文 1h 子集 + OpenCpop 1h 唱歌子集）
- [ ] 接 BigVGAN 把 mel → wav 验证训练样本能还原
- [ ] 训练 1000 step 验证 CFM loss 收敛（< 0.5）
- [ ] 合成 5 句说话 + 5 段歌声，标注主观察测

### 架构扩展（如果 M1 顺利）
- [ ] 用预训练 Fish Speech Dual-AR 权重做 backbone 热启动（s2-pro weights 公开）
- [ ] 加 EMA、混合精度、梯度累积
- [ ] 写评测脚本（speaker similarity、UTMOS、MOS）

## 关联文档

- 技术设计：[docs/architecture-design.md](architecture-design.md)
- M0 执行报告：[docs/m0-report.md](m0-report.md)
- 异常诊断：[docs/m0-diagnose.md](m0-diagnose.md)（待补）

## 决策记录

- **不上 llama.py 复用路径**：自研 backbone 用 DiT + CFM，不直接复用 Fish Speech Dual-AR；M1 阶段再做权重热启动评估
- **tokenizer 复用 codec**：声学特征侧用 mel（80 维）而非 RVQ token；M1 评估是否切到 RVQ 离散化（更省显存）
- **歌声侧统一进同一 backbone**：不并行跑 DiffSinger 第二套模型——这是 ADR 的差异化方向
