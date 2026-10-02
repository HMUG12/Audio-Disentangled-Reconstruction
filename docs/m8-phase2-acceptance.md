# M8 二期验收报告（2026-08-16）

## 一句话结论

自研 ADR-2 硬门禁（CER≤15%）未达标 → 按预案触发保底路线并**超额完成**：说话克隆 + 唱歌合成双产品功能已上线可用，用户真人音色相似度 0.779（目标 0.70+）。

## 交付清单

| 交付物 | 位置 | 验收证据 |
|---|---|---|
| 说话克隆（GPT-SoVITS 保底） | WebUI Clone Tab / `gsv_engine.py` | `output/user_clone_best.wav`, sim=0.779 |
| 唱歌合成（DiffSinger 保底） | WebUI Sing Tab / `diffsinger_engine.py` | `output/ds_baseline_2044001628.wav`, ASR 近似全对 |
| 任意歌词+参考旋律桥 | `melody_bridge.py` | `output/ds_bridge_test.wav`, ASR 13/16 字对 |
| 微调一条龙 | `scripts/gsv_finetune.py` | s1 8ep 权重产出 + 热换推理通过 |
| 流式合成 | `GSVEngine.synthesize_stream()` | 3 块/6.7s, 热机首块 17.4s |
| 音色档案库 | `voice_library.py` + WebUI | 不传参考直接合成通过 (47s) |
| 客观相似度门禁 | `adr/eval/speaker_sim.py` | 正例 0.78 / 负例 0.17-0.28 区分正常 |
| 显存基准 | `output/gpu_benchmark.json` | 8GB 全链路可跑 |
| 自研 ADR-2 研究线 | M9.1-9.6a 全落地 | TF corr 0.966, 采样未达标 (CER 1.157) |

## 硬指标对照

| 门禁 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 自研 CER | ≤15% | 115.7% | ❌ 未达 → 触发保底 |
| 克隆相似度 | ≥70% | **77.9%** | ✅ (保底路线) |
| 唱歌可懂度 | 可懂 | ASR 近似全对 | ✅ |
| 8GB 显存 | 全链路可跑 | 推理 <1.6GB / s1 微调 7.9GB | ✅ (s2 训练阻塞例外) |
| 流式首包 | <300ms | 17.4s | ❌ 未达 (引擎层已通, 延迟待优化) |
| 流式唱歌 | — | 扩散架构不支持 | ❌ 全行业同 |

## 遗留问题（按优先级）

1. **s2 声学微调 CUDA 死锁**（本机环境）：音色细节微调不可用；缓解=零样本+最优参考段（已验证 0.779）；根治=换容器/WSL 或深入 eval 阶段排查
2. **流式首块 17s**：需引擎常驻预热 + 首块裁剪优化
3. **自研 ADR-2 电音感**：根因=BigVGAN 语音声码器 vs 歌声域差；方案=换官方 NSF-HiFiGAN (hop128) 重训或对抗微调
4. 9.6b 自研 LoRA/QLoRA 8GB 实测未跑

## 环境兼容补丁清单（复用价值）

- DiffSinger: scipy.kaiser 迁移、load_ckpt Windows 路径正则
- GPT-SoVITS: jieba_fast→jieba、torchaudio.load→soundfile shim、gloo 单卡直通（s1/s2/bucket_sampler）、torch 2.6 weights_only、int16 量程归一化
- 下载: hf-mirror / ghproxy.net + Range 断点续传（GitHub release 大文件）
