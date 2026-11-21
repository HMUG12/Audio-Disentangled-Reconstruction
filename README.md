# ADR Studio — 本地语音克隆与合成工作台

ADR Studio is a local-first voice cloning and text-to-speech workbench. It runs on your own
machine: build a voice profile from a few seconds of reference audio, fine-tune it with
LoRA/QLoRA if you want a closer match, and serve synthesis over a GPT-SoVITS-compatible HTTP
API. Training, inference and your voice data stay local; nothing is uploaded. The desktop app
ships a portable Python runtime, an offline documentation center, and three browser consoles.

许可证 AGPL-3.0-only · Python 3.10+ · Windows 安装包 / 源码 / Docker 三种形态

## 项目定位与关键能力

ADR（Audio Disentangled Reconstruction）是运行在**你自己电脑上**的语音克隆 / TTS 工作台；推理、训练、声音档案与参考音频均在本地，不上传任何服务器。

- **零样本克隆** — 上传 3–10 秒干净人声即可建档出声，无需训练
- **微调训练** — LoRA / QLoRA 只训少量参数，配梯度检查点与混合精度，降低显存门槛
- **中文语速与风格预设** — `speed_factor` 调速，`style_presets` 提供关键词风格组合；
  情感输入通道尚未实现，风格预设只影响采样与语速（见「已知限制」）
- **局域网 API** — GSV api_v2 / v3、OpenAI `/v1`、ADR 原生 `/api/adr/v1` 四套协议面
- **N.E.K.O 对接** — provider 选 GPT-SoVITS，地址填 `http://127.0.0.1:9881` 即可
- **离线文档** — 控制台内置 10 篇用户文档，中英双语，断网可读
- **桌面端开箱即用** — Tauri 2 壳内置便携 Python 运行时、预热与系统托盘

## 三个控制台与独立设置页

服务启动后，同一套引擎挂载三个入口页与一个设置页（路由见
`adr/server/console.py` 的 `register_pages`）：

| 入口 | 页面 | 适合谁 | 主要能力 |
| --- | --- | --- | --- |
| `/easy` | `adr/server/static/easy.html` | 第一次使用 | 选声音 → 输入文字 → 试听，三步完成，零参数 |
| `/pro` | `adr/server/static/pro.html` | 训练、调参、管模型 | 总览、声音克隆、语音合成、模型库 |
| `/call` | `adr/server/static/call.html` | 接入其他软件 | 双端点调试、参数说明、生成 cURL / Python / JS 代码 |
| `/settings` | `adr/server/static/settings.html` | 设定全局默认值 | 合成默认值、声音与模型、界面、高级、关于 |

`/` 是索引页（`index.html`），只做入口，不参与合成。四个页面顶栏都支持中英文与主题切换，
并有「文档」按钮直达文档中心。

**桌面壳怎么进**：首次启动依次经过条款页 → 三步首启向导 → 预热页（下载约 2 GB 底模并加载
引擎）→ 启动器。启动器里是三张控制台卡片，以及模型入口、设置和关于页；卡片会拉起本地服务
并跳转路由。源码 / 服务形态没有登录页，直接打开 `http://127.0.0.1:9881/easy`。

设置页写入的是**控制台前端默认值**，落盘为数据目录下的 `settings.json`，通过
`GET/PUT/DELETE /api/adr/v1/settings` 读写，不改变未显式传参的外部 API 调用行为；临时改参只对当前一次合成或训练生效。

## 文档

### 内置文档中心

- 入口：各控制台右上角「文档」，或索引页的「文档中心」卡片
- 离线可读：随安装包携带 10 篇用户文档；联网时可「检查更新」与官方仓库逐篇比对
- 中英双语：有译文的篇目可切换 English；缺英文时**回落中文原文并明确提示**，不报错、不空白
- 模糊搜索：拼写近似、中文短句按字符覆盖率兜底；无结果时推荐排障与使用指南。范围只收录
  用户文档 / API 规范 / 发布说明，内部计划与审计材料不在白名单

| 分组（篇数） | 篇目（slug） | 本地文件 |
| --- | --- | --- |
| 入门（3） | quickstart、guide、demos | `README.md`、[docs/en/quickstart.md](docs/en/quickstart.md)、[docs/guide.md](docs/guide.md)、[docs/en/guide.md](docs/en/guide.md)、[docs/demos.md](docs/demos.md) |
| 指南（4） | troubleshooting、gpu、install、update | [docs/troubleshooting.md](docs/troubleshooting.md)、[docs/en/troubleshooting.md](docs/en/troubleshooting.md)、[docs/gpu-compat.md](docs/gpu-compat.md)、[docs/architecture/windows-install-acceptance.md](docs/architecture/windows-install-acceptance.md)、[docs/update-manifest.md](docs/update-manifest.md) |
| 开发（2） | api、arch | [docs/tts-api-spec.md](docs/tts-api-spec.md)、[docs/architecture-design.md](docs/architecture-design.md) |
| 发布（1） | release | [docs/release-notes-v1.2.0.md](docs/release-notes-v1.2.0.md) |

文档 API 前缀为 `/api/adr/v1/docs`：`GET /` 索引、`GET /search?q=` 搜索、`GET /check` 比对
官方仓库、`GET /{slug}?lang=zh|en&source=local|online` 读取单篇。

### 官网文档站

`site/` 是纯静态官网文档站，由 `scripts/build_site.py` 从 `docs/` 与根 `README.md` 生成，
**不要手改**；生成物不入库（已在 `.gitignore`），克隆后先构建。中文页生成到 `site/docs/`，
英文页生成到 `site/en/`，站点约定基址为 `https://adr-studio.pages.dev/`。重建用
`python scripts/build_site.py`，自检用 `python scripts/build_site.py --check`。

## 问题与意见反馈

使用中遇到问题、想要新功能或反馈体验，请填写
[ADR Studio 问题与意见收集表](https://docs.qq.com/form/page/DSkhyWHJnS0ZUa1ZB)（腾讯文档，免登录即可提交）。
官网顶栏「问题反馈」、页脚与首页入口指向同一张表；缺陷类问题也可直接开
[GitHub Issue](https://github.com/HMUG12/Audio-Disentangled-Reconstruction/issues)，
并附上 `adr doctor` 输出、日志片段与复现步骤。

## 快速开始

### 方式一：Windows 安装包

1. 从 [GitHub Releases](https://github.com/HMUG12/Audio-Disentangled-Reconstruction/releases)
   下载安装包：CPU 版用于无独立显卡设备，GPU 版用于 NVIDIA CUDA 加速。两版共用同一 AppId，
   可原地升级覆盖，底模与声音档案保留。
2. 双击安装，按向导接受许可协议（per-user 安装，无需管理员权限）。
3. 首次启动同意使用条款，完成首启向导，等待预热页下载底模并加载引擎，随后进入启动器；
   预热完成后即可完全离线运行。

批量部署可用静默安装：`ADR-Studio-1.3.1-beta.92-x64-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="安装目录"`。

### 方式二：源码运行

环境要求：Python 3.10+；Windows 推荐，其他平台可跑推理。NVIDIA GPU 可选。

```powershell
pip install -e ".[m1]"          # 基础依赖 + TTS 服务
pip install -e ".[m2]"          # 训练加速（LoRA / QLoRA，按需）
pip install -e ".[multilang]"   # 日 / 粤 / 韩 G2P（按需）
python -m adr.server --agree-terms   # 默认 127.0.0.1:9881
```

启动后打开 `http://127.0.0.1:9881/easy`（新手）、`/pro`（专业）、`/call`（调用）、
`/settings`（设置）。

常用命令：

| 命令 | 说明 |
| --- | --- |
| `python -m adr.server -p 9880` | 换端口；`-a 0.0.0.0` 监听局域网（须配 API Key） |
| `adr process -i <音频文件或目录> -o ./output/processed` | 切片 / ASR 转写 / 音素化 |
| `adr train -d <数据目录> --lora --lora-rank 8` | LoRA 微调；`--qlora` 走 4-bit 量化 |
| `adr clone --ref <wav> --text <文本>` | 单句克隆推理 |
| `adr model list / download / info` | 预训练模型管理 |
| `adr doctor [--fix]` | 环境与数据体检 |

### 方式三：Docker

容器版是纯 HTTP 服务镜像，无 Gradio WebUI，入口为
`python -m adr.server --addr 0.0.0.0 --port 9881`：

```bash
cd docker
docker compose up -d --build                                    # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build   # CUDA
curl http://127.0.0.1:9881/api/adr/v1/health
```

两个镜像标签：`adr-server:1.3.1-beta.92-cpu` 与 `adr-server:1.3.1-beta.92-cuda`
（compose 已按标签引用，本地构建会就地覆盖同名镜像）。

CUDA 变体需要 NVIDIA 驱动与 Container Toolkit；无 GPU 时引擎会回退 CPU。详细说明见
[docker/README.md](docker/README.md)。

## TTS API

服务默认只监听 `127.0.0.1:9881`。四套协议面共用同一引擎：

| 协议面 | 端点 | 用途 |
| --- | --- | --- |
| ADR 原生 | `POST` / `GET` `/api/adr/v1/tts` | 按音色档案名合成，档案是一等公民 |
| GSV api_v2 | `POST` / `GET` `/api/v2/tts` | N.E.K.O 与 GPT-SoVITS 生态客户端 |
| GSV api_v3 | `GET /api/v3/voices`、`WS /api/v3/tts/stream-input` | 双向流式；无任务队列，逐句串行 |
| OpenAI 兼容 | `POST /v1/audio/speech`、`GET /v1/models` | OpenAI TTS 兼容客户端 |
| 健康检查 | `GET /api/adr/v1/health` | 探活；`engine_stage` 可区分引擎预热阶段 |

最小示例（字段与 [docs/tts-api-spec.md](docs/tts-api-spec.md) 原生 API 一节一致）：

```bash
# 先确认有哪些音色档案
curl http://127.0.0.1:9881/api/adr/v1/profiles

# 合成一段 wav
curl -X POST http://127.0.0.1:9881/api/adr/v1/tts \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"你好，这是 ADR 合成测试\", \"profile\": \"demo\", \"media_type\": \"wav\"}" \
  --output out.wav
```

```python
import json
import urllib.request

payload = {
    "text": "你好，这是 ADR 合成测试",
    "profile": "demo",          # 也可改用 ref_audio_path 走 v2
    "media_type": "wav",
}
req = urllib.request.Request(
    "http://127.0.0.1:9881/api/adr/v1/tts",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp, open("out.wav", "wb") as f:
    f.write(resp.read())
```

GSV api_v2 的流式约定：`streaming_mode=1/2/3` 时第一块是 44 字节 WAV 头，其后为裸 s16le PCM；
该「头 + 裸块」契约只在 `media_type=wav` 时成立。完整参数、错误码与差异清单见
[docs/tts-api-spec.md](docs/tts-api-spec.md)。

### N.E.K.O 对接

启动桌面端或 `python -m adr.server`，确认监听端口（默认 9881）。N.E.K.O 设置 → 语音合成 →
provider 选 **GPT-SoVITS**，API 地址填 `http://127.0.0.1:9881`，刷新音色列表即可看到 ADR
声音档案；也可用 OpenAI provider 配合 `/v1/audio/speech`。

### 局域网与鉴权

默认仅本机可达。局域网调用须同时设置 API Key，否则服务拒绝启动：
`$env:ADR_TTS_API_KEY="<你的密钥>"; python -m adr.server --addr 0.0.0.0`。

客户端凭据二选一（推荐）：`Authorization: Bearer <key>` 或 `X-API-Key: <key>` 头。
查询参数 `?api_key=<key>` 默认关闭（会随 URL 进访问日志/代理日志）；确需使用时
须额外设置 `ADR_ALLOW_QUERY_API_KEY=1`。
未设置 Key 且确实要监听 `0.0.0.0` 时，需显式设 `ADR_ALLOW_UNAUTHENTICATED=1` 接受风险；
跨站请求默认被拦截，放行用 `ADR_CORS_ORIGINS` / `ADR_ALLOWED_HOSTS`。

## GPU 与显存要求

档位结论以 [docs/gpu-compat.md](docs/gpu-compat.md) 为准：

| 功能 | NVIDIA CUDA | AMD / Intel GPU（Windows） | 纯 CPU |
| --- | --- | --- | --- |
| 微调训练（s2 / s1） | 全功能，4 GB 配方实测 | 不支持 | 不支持 |
| 数据准备（切片 / ASR / 特征） | 支持 | 仅切片 + ASR 可跑 | 可跑，较慢，有缓存 |
| 推理合成（整段 / 流式） | 支持，峰值约 1410 MB | 走 CPU 路径 | 支持，短文本 RTF 约 1.9x |
| 声纹门禁 | 支持 | 支持（纯 CPU 实现） | 支持 |

训练配置预设位于 `configs/`：[default.yaml](configs/default.yaml)、
[vram_4gb.yaml](configs/vram_4gb.yaml)、[vram_6gb.yaml](configs/vram_6gb.yaml)、
[vram_8gb.yaml](configs/vram_8gb.yaml)。没有 NVIDIA 显卡时，可在本机做 CPU 推理与评分，
训练改到任意 NVIDIA 机器或云 GPU 完成后，把音色档案拷回即可（档案 = 权重 + 参考音频）。

## 目录结构

```text
ADR/
├─ adr/             Python 引擎与服务（server/ core/ training/ models/ inference/）
├─ desktop/         Tauri 2 桌面壳（src/ 界面，src-tauri/ Rust）
├─ docs/            用户文档、architecture/ 设计文档、en/ 英文译文
├─ site/            官网静态文档站（由 scripts/build_site.py 生成）
├─ scripts/         构建、训练、打包与站点生成脚本
├─ tests/           pytest 用例
├─ configs/         训练配置预设（default / vram_4gb / vram_6gb / vram_8gb）
└─ docker/          Dockerfile、compose 文件与容器版说明
```

## 开发与测试

```powershell
# 开发依赖与全量测试；必须带 --basetemp（Windows 默认临时目录可能 PermissionError）
pip install -e ".[dev]"
python -m pytest --basetemp=.pytest_tmp
```

- 控制台前端是纯静态页面，位于 `adr/server/static/`（`index/easy/pro/call/settings.html`
  加 `i18n*.js`），由 `register_pages` 挂到服务根；没有 Node 构建步骤
- 桌面壳构建（以 `desktop/package.json` scripts 为准，需要 Rust 工具链）：

```powershell
cd desktop
npm install
npm run dev      # tauri dev
npm run build    # tauri build
```

安装包由 `scripts/build_installer.ps1` 打包。

## 版本规则

- **未正式发布一律 `1.3.1-beta.<提交数>`**，不因功能合入升 minor；正式发布才升 minor
- 版本号三处必须一致：`pyproject.toml`、`desktop/package.json`、
  `desktop/src-tauri/tauri.conf.json`；当前均为 `1.3.1-beta.92`
- 容器镜像 tag 跟随版本号（`-cpu` / `-cuda`），常青 tag 指向最新发布版

## 已知限制与故障排查

- **情感控制未实现**：GSV v2 没有 emotion 输入通道；`style_presets` 只调整采样与语速，
  不产生情感。1.3.0 范围只保留语速可调
- **v3 兼容面无任务队列**：逐句串行合成，`overrides.media_type` 被忽略，binary 帧恒为 WAV
- **流式非 wav 格式**：`ogg` / `aac` / `raw` 流式是每块独立容器，不能直接顺序拼成单流
- **训练需要 NVIDIA CUDA**：AMD / Intel 与纯 CPU 只能做推理与数据准备
- **首次合成有冷启动**：引擎加载约数十秒，之后为流式秒级响应
- **端口冲突**：9881 被占用时用 `-p` 换端口；桌面端在启动器填 0 可自动回退
- **beta 阶段**：尚未正式发布，接口与行为仍可能变化

启动失败、显存不足、合成异常、训练中断、401/413/429、文档中心为空等问题，先看
[docs/troubleshooting.md](docs/troubleshooting.md)；使用流程与设置说明见
[docs/guide.md](docs/guide.md)。

## 风险披露与使用边界

ADR 具备克隆任意人声的技术能力，使用前请确认：

- 克隆、合成任何真实人物的声音，必须事先取得该声音权利人的明确授权
- 严禁用于诈骗、冒充他人、诽谤、伪造证据、侵犯隐私等违法违规用途
- 语音样本含声纹特征，仅在本地数据目录存储与处理；生成内容及用途由使用者自行负责
- 模型输出可能被误用或伪造，请自行评估传播风险并为生成内容标注来源

同意方式：桌面端条款页点击同意；命令行加 `--agree-terms`；服务端 / Docker 设
`ADR_AGREE_TERMS=1`。未同意时服务以退出码 2 拒绝启动。默认不采集设备标识、不上报使用数据。

**源码获取途径（AGPL-3.0 §6）**：本项目源码托管在本仓库，发行版对应同一 tag 的完整源码，
可在 [GitHub Releases](https://github.com/HMUG12/Audio-Disentangled-Reconstruction/releases)
的对应版本附件中下载；安装包与 Docker 镜像均由该源码构建，若你通过二进制分发本软件，
须同时提供本节所述源码获取方式。

## 许可证

本项目以 [GNU Affero General Public License v3.0（AGPL-3.0-only）](LICENSE) 授权发布。
若修改本软件并以其对外提供网络服务，须以 AGPL-3.0-only 向服务使用者提供修改版的完整源码。
第三方组件（GPT-SoVITS、BigVGAN、python-build-standalone、Tauri 等）遵循其原始许可证，
完整清单见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。

维护者：未知之致 · 联系 QQ：3699672176
