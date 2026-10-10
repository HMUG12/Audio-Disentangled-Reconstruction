# 样张 (samples)

本目录是克隆效果样张的**说明与生成命令**；音频文件本身不入库——个人声音属
生物特征数据，且公共样张有授权成本（方案 A，批次 51 定稿；如需放公共样张见
`.gitignore` 中 `examples/samples` 段注释）。

## 样张构成（每个音色一组三件）

| 文件 | 内容 | 生成方式 |
|------|------|----------|
| `ref.wav` | 参考音频（经 `clip_ref` 裁剪至 ≤8s 的切片） | 克隆流程第 1 步自动产出 |
| `zero_shot_p*.wav` | 官方预训练底模零样本克隆（5 句探测集） | `scripts/eval_secs.py` |
| `finetuned_p*.wav` | 微调后权重克隆（同一探测集） | `scripts/eval_secs.py` |

实测数字（CPU、档案"我的声音V2"，2026-10-10）：锚点 SECS 0.9324 /
零样本 0.7729 / 微调 0.8052（Δ+0.0323）。口径见 [REPRODUCE.md](../../REPRODUCE.md) 第 5 节。

## 本地生成命令

```powershell
# 1) SECS 对比 + 逐句音频落盘 (output/eval_secs_wavs/<音色>/)
python scripts/eval_secs.py --voice 我的声音V2 --cpu

# 2) MOS 盲测包 (无参生成混淆目录; --score 聚合揭晓)
python scripts/mos_ab_test.py --voice 我的声音V2
python scripts/mos_ab_test.py --score <混淆目录名>

# 3) CER 回译样张包 (faster-whisper tiny, 达标线 CER<=0.15)
python scripts/render_sample_pack.py <SoVITS权重> --out output/sample_pack
```

## 在线样张

云盘链接（占位，待发布时替换）：`TODO`
