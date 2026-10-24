# 第三方组件许可声明

ADR 集成或依赖的第三方组件及其许可证。各组件的完整许可文本以其上游仓库/官方发布为准；随包分发的组件，其版权与许可声明一并分发。

## 随包分发 / 内置组件

| 组件 | 用途 | 许可证 | 上游 |
| --- | --- | --- | --- |
| GPT-SoVITS | 语音克隆推理底座 | MIT | https://github.com/RVC-Boss/GPT-SoVITS |
| BigVGAN | 声码器（经 GPT-SoVITS 集成） | MIT | https://github.com/NVIDIA/BigVGAN |
| python-build-standalone | Windows 便携 Python 运行时 | MIT（内含 CPython，PSF-2.0） | https://github.com/astral-sh/python-build-standalone |
| Tauri | 桌面壳框架 | MIT / Apache-2.0 | https://github.com/tauri-apps/tauri |

GPT-SoVITS 自带的上游组件（BigVGAN `incl_licenses/`、AP_BWE 等）遵循其目录内随附的许可文件。

## Python 运行时核心依赖

| 组件 | 许可证 |
| --- | --- |
| torch / torchaudio | BSD-3-Clause |
| numpy | BSD-3-Clause |
| pyyaml | MIT |
| click | BSD-3-Clause |
| rich | MIT |
| tqdm | MPL-2.0 / MIT |
| soundfile | BSD-3-Clause（绑定 libsndfile，LGPL-2.1） |
| librosa | ISC |
| transformers | Apache-2.0 |
| fastapi | MIT |
| uvicorn | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| gradio | Apache-2.0 |
| faster-whisper | MIT（绑定 CTranslate2，MIT） |
| pypinyin | MIT |
| onnxruntime | MIT |
| peft | Apache-2.0 |
| bitsandbytes | MIT |
| pyopenjtalk / jamo / ToJyutping / ko_pron | 各上游许可证（多语言 G2P 可选依赖） |

## 说明

- 本表覆盖 ADR 直接集成与分发的主要组件，随批次持续补充；完整传递依赖清单以各包元数据（`pip show` / Cargo.toml / package.json）为准。
- 预训练底模与用户声音档案不属于本仓库分发物，其使用遵循对应模型上游条款。
