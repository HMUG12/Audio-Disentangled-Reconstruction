# v1.2.0 发布说明（草稿）

> 用途：tag v1.2.0 推送后作为 GitHub Release 正文（`gh release create v1.2.0 --notes-file`）。
> 实测数据已回填：CPU 459.5MB / GPU 1926.4MB；SHA256 见下载节。

## ADR Studio v1.2.0 — 多形态分发

ADR 是运行在你自己电脑上的语音克隆 / TTS 工作台：几分钟参考音频完成声音克隆训练，提供 GPT-SoVITS 兼容的流式合成服务。全部推理与训练均在本地完成，语音与文本数据不上传任何服务器。

### 本版本重点：三种运行形态

引擎与 API 同源，按部署场景选择：

| | 桌面版（Windows 安装包） | 无 UI 服务版（server.bat） | 容器版（Docker） |
| --- | --- | --- | --- |
| 适用场景 | Windows 桌面开箱即用 | Windows 后台常驻 / 无人值守 | Linux 服务器 / NAS / 云端 |
| 启动方式 | 双击 `adr-desktop.exe` | 运行安装目录下 `server.bat` | `docker run` |
| 交互入口 | 壳内启动器 + 控制台 | HTTP API（五套 API 面） | HTTP API（五套 API 面） |
| 变体 | CPU / GPU 双安装包 | 同桌面版 | 镜像 tag `-cpu` / `-cuda` |

### 新特性

- **容器版 Docker 分发** — `Dockerfile` 两阶段构建（GSV 钉版 clone + 兼容补丁自动应用），`docker compose` 单命令起服务；镜像发布至 GitHub Container Registry：

  ```bash
  # CPU
  docker run -d --name adr-server -p 9881:9881 -v adr-data:/data ghcr.io/hmug12/adr-server:1.2.0-cpu
  # GPU（需 NVIDIA Container Toolkit）
  docker run -d --name adr-server --gpus all -p 9881:9881 -v adr-data:/data ghcr.io/hmug12/adr-server:1.2.0-cuda
  ```

- **Windows 无 UI 服务版** — 安装目录自带 `server.bat`（env 驱动：`ADR_DATA_DIR` / `ADR_TTS_API_KEY` / `ADR_TTS_DEVICE` / `ADR_LOG_LEVEL`），后台常驻 / 无人值守场景免壳运行
- **多语言合成** — 中 / 英 / 日 / 粤及中英混说（`text_lang` 参数 + 控制台语言选择）；声音档案可标记语种，合成默认语言自动联动
- **控制台中英文切换** — 控制台四页（索引 / 新手 / 专业 / 调用）一键中英切换，352 条词条，语言偏好自动记忆
- **声音档案导出 / 导入闭环** — `.adrvoice` 包（manifest + 权重 + 参考音频，sha256 完整性校验），换机迁移不再丢声音；导出前明确提示"权重即声音本体"（生物特征数据）
- **续训流程** — easy / pro 控制台"改进已有声音"：追加录音 → 续训 → 质量门自动比对新旧相似度，退化可一键回滚（权重版本链逐版回退）
- **13/14 代 Intel 闪退加固** — Raptor Lake AVX-512 硬件缺陷缓解：CPU 兼容模式（AVX2 上限）**自动启用**（识别 13/14 代即开，也可手动勾选）；native 崩溃时 faulthandler 兜底留痕
- **CUDA 稳定性** — OOM 分流（显存不足返回可读提示而非堆栈）+ 驱动版本诊断日志（驱动过旧导致的静默回退 CPU 有显式警告）
- **日志体系** — `server.log` 超限自动轮转；CLI 统一 `cli.log`；崩溃留痕与"打开日志目录"入口
- **训练安全** — 实验名 Windows 保留名清洗（CON/NUL 等不再导致建目录失败）；训练子进程统一 UTF-8（中文 Windows 乱码根除）

### 已知限制

- **中英混说**清晰度 / 流畅度略欠（底模特性，单语种不受影响）
- **韩语合成**：Windows 版不支持（韩语 G2P 依赖在 Windows Python 3.13 不可用）；**Linux 容器版已补齐** ko 链（`g2pk2` + `python-mecab-ko`）
- 安装包未经充分测试，如遇问题请提 [Issues](https://github.com/HMUG12/Audio-Disentangled-Reconstruction/issues)

### 下载

- **CPU 版**（无独立显卡设备）：`ADR-Studio-1.2.0-x64-setup.exe`（459.5MB）
- **GPU 版**（NVIDIA CUDA 加速）：`ADR-Studio-1.2.0-x64-gpu-setup.exe`（1926.4MB）
- SHA256：
  - CPU `c429beb3348e0997e6841d1ab94a5a0b152c75c430905e96b808f0d1a9f7ad90`
  - GPU `905965a738fe83485ab3adfa03b6deb143ee03d573bafa5b63e780e15bf15feb`
- GitHub Release：https://github.com/HMUG12/Audio-Disentangled-Reconstruction/releases/tag/v1.2.0（附 SHA256 校验值）
- 123 云盘：https://1860775428.share.123pan.com/123pan/5KDqvd-oPPsv（提取码 `ADRS`）

安装说明：per-user 安装无需管理员权限；两版共用同一 AppId，原地升级覆盖旧版，已下载底模与声音档案均保留；首启自动下载预训练底模（约 2GB），此后完全离线。

### 升级 / 兼容性

- v1.0.x / v1.1.0 → v1.2.0：安装包直接覆盖（数据目录兼容）
- N.E.K.O 对接方式不变：provider GPT-SoVITS，地址 `http://127.0.0.1:9881`
- OpenAI 兼容接口不变：`POST /v1/audio/speech`、`GET /v1/models`
- 控制台新增中英切换按钮（右上角）；档案新增语种标记（合成默认语言联动）
