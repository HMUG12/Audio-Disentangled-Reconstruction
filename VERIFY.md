# VERIFY — 第三方验证清单

> 每条声明给出可独立执行的验证步骤与预期结果，命令均为实测形态（细节见 [REPRODUCE.md](REPRODUCE.md)）。
> 环境准备：Python 3.10+，`pip install -e .[m1]`；模型资源说明见 REPRODUCE 第 0 节。

| # | 声明 | 验证步骤 | 预期结果 |
| --- | --- | --- | --- |
| 1 | 可安装、可自检 | `python -c "import adr; print(adr.__version__)"`；`adr doctor` | `1.0.0`；doctor 无 FAIL（警告可 `--fix` 清理） |
| 2 | 测试套件全绿 | `python -m pytest --basetemp=<tmp> -p no:cacheprovider` | 实测基线 **431 passed + 1 skipped**（ASR 用例默认跳过，`--run-asr` 开启） |
| 3 | 桌面壳测试全绿 | `cd desktop/src-tauri; cargo test --lib` | **4 passed** |
| 4 | 服务可启动 | `python -m adr.server` 后请求 `/api/adr/v1/health` | HTTP 200，含引擎阶段/就绪状态 |
| 5 | 零样本克隆可用 | REPRODUCE §3 的 `Invoke-RestMethod /v1/audio/speech` | 产出 RIFF 头 wav（音色可任选 `GET /api/v3/voices` 中的档案） |
| 6 | 微调带来正收益 | `python scripts/eval_secs.py --voice <档案> --cpu` | `finetuned.mean > zero_shot.mean`（实测 Δ=+0.0323）；报告落 `output/eval_secs_<voice>.json` |
| 7 | 评估口径自洽 | SECS 报告中锚点分数（真人切片 vs ref） | 锚点 > 两条合成条件（实测 0.9324 vs 0.7729/0.8052），证明打分链路能区分真人/合成 |
| 8 | 盲测协议可信 | `mos_ab_test.py` generate 后检查 `score_sheet.csv` | A/B 顺序随机混淆（无系统规律）；score 阶段才揭晓映射 |
| 9 | 生成可懂 | `python scripts/render_sample_pack.py` | CER ≤ **0.15**（faster-whisper tiny 回译口径） |
| 10 | 训练可复现且防退化 | `gsv_finetune.py` + `train_gate.py` 并行；曲线 `output/train_gate_<exp>.jsonl` | 首行 zero_shot 基线（官方预训练钉死）；达标自动 STOP 早停；退化时档案 ⚠ 标记 + rollback 可回退 |
| 11 | 个人数据不入库 | `git ls-files data/ output/ "*.wav"` | 空（录音/档案/产物全部 ignore；`.gitignore` 头部原则：个人录音为生物特征数据，绝不入库） |
| 12 | 局域网访问强制鉴权 | `python -m adr.server --addr 0.0.0.0`（不设 key） | 启动横幅打无鉴权安全警告；设置 `ADR_TTS_API_KEY` 后无 key 请求被拒 |
| 13 | 安装包卫生 | `python scripts/check_runtime.py`（构建产物上） | exit 0：无根级裸 .pth、无个人数据面、无底模、CPU 包无 nvidia_* 库 |
| 14 | 版本一致 | `import adr` vs `desktop/src-tauri/Cargo.toml` vs `tauri.conf.json` | 四面统一 `1.0.0` |

## 评估口径说明（为何不是 PESQ/ESTOI）

零样本/少样本 TTS 评估没有平行语料（输入文本 ≠ 参考音频内容），PESQ/ESTOI 所需的逐帧对齐不存在。ADR 的三件套各自覆盖一个维度：

- **SECS**（`adr/eval/speaker_sim.py`，ERes2NetV2）：音色保持度。经验区间 `>0.70` 同一人 / `0.55-0.70` 接近 / `<0.45` 不同人；5 句探测取均值（单句波动 ±0.03+）。
- **MOS**（`scripts/mos_ab_test.py`）：盲听人评，零样本 vs 微调随机混淆对比。
- **CER**（faster-whisper 回译）：可懂度下限（合成内容能否被 ASR 还原）。

探测句集（5 句）与预训练钉死常量统一收敛在 [adr/eval/probes.py](adr/eval/probes.py)，训练门禁 / held-out 评估 / 档案评估三处同源，避免双份定义漂移。
