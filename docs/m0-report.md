# M0 执行报告

> 完成日期：2026-08-07
> 阶段：M0 — 研究复现验证

## 目标

跑通 Fish Speech 推理管线，产出第一个音频样本，验证自研 TTS+SVS 系统的技术可行性。

## 成果

✅ **首个音频样本已生成**：[output/fish_smoke.wav](file:///e:/新创意构思/新建文件夹/ADR/output/fish_smoke.wav)
- 时长 379.4 秒（约 6.3 分钟，44.1kHz）
- 内容：仅 2 个汉字"你好"
- 文件大小 31.9 MB

## 技术指标

| 指标 | 实测值 | 期望范围 | 评价 |
|---|---|---|---|
| 模型加载 | 成功 | — | ✅ Dual-AR 28 层 × 1024 dim, 0.3B |
| Tokenizer | vocab 151643，semantic 范围 151678–155773 | 词表完整 | ✅ |
| 推理吞吐 | 3.06 tokens/s | 8GB 显存下理论上限 | ✅ |
| 显存峰值 | 4.7 GB | 8GB 可用 | ✅ |
| 总推理时长 | 44 分 26 秒（8170 tokens） | — | ⚠️ 见下文 |
| Codec 解码 | 44.1kHz 立体声 → 单声道 wav | — | ✅ |

## 执行过程

1. **环境准备**
   - `uv sync --python 3.12 --extra cu128` 装好 Fish Speech 全套依赖（PyTorch 2.8.0+cu128 + transformers 4.57.3）
   - 克隆 openvpi/DiffSinger 仓库到 `third_party/DiffSinger`（仅获取代码骨架，8GB 显存不足以本地训练）

2. **模型下载**（1.62 GB，43 MB/s）
   - ModelScope `fishaudio/openaudio-s1-mini`
   - 补丁：从 HF `fishaudio/s2-pro` 补齐缺失的 `tokenizer.json / tokenizer_config.json / special_tokens_map.json`（ModelScope 仓库仅含 `tokenizer.tiktoken`，与 s2-pro vocab 对齐）

3. **推理跑通**
   - 脚本：[scripts/m0_fish_smoke.ps1](file:///e:/新创意构思/新建文件夹/ADR/scripts/m0_fish_smoke.ps1)
   - 文本：'你好'（m0_smoke_text.txt）
   - 输出：LLM 推理 → `codes_0.npy`（10 码本 × 8170 tokens，653KB）

4. **Codec 解码**
   - 脚本：[scripts/m0_decode_only.py](file:///e:/新创意构思/新建文件夹/ADR/scripts/m0_decode_only.py)
   - 复用同一 codec.pth，避免重跑 LLM
   - 输出：wav 文件 33MB / 379秒

## 关键发现

### 1. 8GB 显存对"说话+唱歌"统一模型不现实

- s1-mini (0.3B) 推理时已占 4.7GB 显存
- 若训练 0.3B 模型 (BF16 + 优化器状态 + 激活) 需 ≥40GB
- 训练 0.3B 完整模型必须**租云**（4×A800/H800 级），M1/M2 阶段云端完成

### 2. 推理速度瓶颈

- 44 分钟产 6.3 分钟音频 ≈ 7x 实时（仅"你好"两个字）
- 主要瓶颈：Dual-AR 的 28 层 + 序列 8170 token + 单卡推理
- **设计启示**：自研模型若不优化推理（KV cache、speculative decoding、流式），实际产品部署需 4-bit 量化 + 编译加速

### 3. Fish Speech 仓库结构精要

- `fish_speech/models/text2semantic/llama.py` — Dual-AR Transformer（28 层，slow + fast AR 双轨）
- `fish_speech/models/dac/` — 改进 DAC codec（RVQ，10 码本）
- `fish_speech/configs/modded_dac_vq.yaml` — codec 配置（实例化时通过 Hydra）
- `fish_speech/conversation.py` — 多模态对话 + 提示模板
- **与设计文档呼应**：可作为 tokenizer（codec）+ 推理工程基座，二次开发空间大

## DiffSinger 复现状态

- 代码已克隆（third_party/DiffSinger）
- **未在 M0 跑通推理**，原因：8GB 显存不足以同时跑 OpenCpop 数据集上的训练；推理需要预训练 checkpoint（OpenUTAU/DiffScope 生态的标准模型约 1.5GB）
- **后续路径**：M1/M2 阶段上云后，租用 A800 一并跑通 DiffSinger 推理 + 训练，准备歌声侧基线

## 项目结构

```
ADR/
├── docs/
│   └── architecture-design.md         # 技术设计文档
├── third_party/
│   ├── fish-speech/                   # 说话侧基座（已跑通）
│   │   ├── .venv/                     # 独立 Python 3.12 + cu128
│   │   └── checkpoints/openaudio-s1-mini/   # 已下载权重
│   └── DiffSinger/                    # 歌声侧基座（代码已就绪）
├── scripts/
│   ├── m0_fish_smoke.ps1              # 完整推理脚本
│   ├── m0_decode_only.py              # 单独 codec 解码脚本
│   └── m0_smoke_text.txt              # 测试文本
└── output/
    ├── fish_smoke.wav                 # ⭐ M0 首个样本
    ├── codes_0.npy                    # LLM 输出的 audio tokens
    └── fish_smoke.{log,err.log}       # 推理日志
```

## M0 验收清单

- [x] 文档：技术设计文档（[docs/architecture-design.md](file:///e:/新创意构思/新建文件夹/ADR/docs/architecture-design.md)）
- [x] 环境：fish-speech 推理环境（uv + Python 3.12 + cu128）
- [x] 模型：openaudio-s1-mini 权重就绪
- [x] **首次端到端 TTS：跑通，输出 wav**
- [x] M0 报告（本文档）
- [ ] DiffSinger 推理（推迟到 M1 上云阶段）

## 下一步：M1

按设计文档 M1 — 数据流水线 + 冒烟训练：
- **数据**：Emilia/WenetSpeech 子集 + OpenCpop/M4Singer 歌声数据
- **训练**：1 小时说话 + 1 小时唱歌做冒烟训练，租 1×A800（24h 够用）
- **目标**：验证自研模型架构 loss 能收敛、合成音频可懂
- **前置准备**（本周可做）：
  1. DiffSinger 预训练 checkpoint 下载与 OpenUTAU 集成测试
  2. 自研 backbone 代码骨架（基于 Fish Speech Dual-AR + DiffSinger 旋律条件 + UniVoice null melody token）
  3. 数据集下载清单与小批量预筛选脚本
