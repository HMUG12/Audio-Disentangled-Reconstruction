# M0 异常诊断与修复

> 生成日期：2026-08-07
> 关联脚本：[scripts/m0_diagnose.py](../scripts/m0_diagnose.py)、[scripts/m0_make_prompt.py](../scripts/m0_make_prompt.py)、[scripts/m0_rerun_fix.py](../scripts/m0_rerun_fix.py)

## 现象

M0 推理产出了 [output/fish_smoke.wav](file:///output/fish_smoke.wav)（33MB），但**对应的文本只有 2 个汉字"你好"**。

- 预期时长：1–2 秒
- 实际时长：379.4 秒
- 实际 token 数：8170（达到 max_seq_len 8192 的极限）

## 诊断

通过 [scripts/m0_diagnose.py](../scripts/m0_diagnose.py) 分析 codes_0.npy：

| 指标 | 实测 | 期望 |
|---|---|---|
| `<\|im_end\|>` 出现位置 | **未出现** | 必须出现 |
| 最后 token | 583 | 应为 im_end (151645) |
| `codebook 0` 唯一 token 数 | 168 | 应 ≥ 数百 |
| 最常出现 token 占比 | 583 占 42.4% | < 5% |
| 最长无重复连续段 | 18 tokens (0.2%) | > 50% |
| 前 100 vs 后 100 token 匹配 | 0/100 | 任意 |

## 根因

Fish Speech 的停止逻辑（[inference.py:233-234](../third_party/fish-speech/fish_speech/models/text2semantic/inference.py)）：

```python
if cur_token[0, 0, -1] == im_end_id:
    break
```

但模型从未生成 `<|im_end|>`（id 151645），于是循环跑到 `max_seq_len` 极限（8192）。

**根本原因**：`generate_long()` 在没有 `prompt_tokens` 时，system message 退化为：

```python
system_parts = [
    TextPart(text="convert the provided text to speech", cal_loss=False)
]
```

没有 reference audio，模型既不知道用什么音色，也不知道生成多长就该停。最终陷入"复读"。

## 修复方案

1. 生成 1 秒**极低幅度白噪**作为 prompt 音频（既不是纯静音避免触发异常，也不引入语义内容）
2. 重跑时加 `--prompt-audio` + `--prompt-text`
3. 显式传 `--max-new-tokens 2048` 避免再跑满 max_seq_len
4. 去掉 `--half`（bfloat16 在边界 token 上可能引发数值问题）

修复脚本已就绪：[scripts/m0_rerun_fix.py](../scripts/m0_rerun_fix.py)

预估新一次推理时长：5–10 分钟（vs 原 44 分钟）。

## 暂不重跑

- 重跑需约 10 分钟（fp32 + 显式 stop limit）
- 但产物是 1-2 秒 wav，价值有限
- **该工作整体推迟到 M1 上云阶段一并处理**，与 1 小时冒烟训练同时进行
- 当前优先事项：M0+ 自研骨架 + 数据/算力准备

## 经验沉淀（写回 [adr/README.md](../adr/README.md)）

- **生成含 prompt 音频的 TTS 推理**：永远显式传 prompt，否则模型会失控
- **限定 max_new_tokens**：作为安全网，避免无限循环
- **多遍自我诊断**：基于 token 分布的快速诊断能直接定位问题，不必反复重跑
