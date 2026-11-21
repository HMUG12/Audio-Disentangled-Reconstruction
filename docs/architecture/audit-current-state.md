# ADR 现状架构与增量审计（M0）

- 基线版本：`pyproject.toml` 1.3.1（commit `ac33ffd` 之后的文档批）
- 盘点日期：2026-11-14
- 口径：只读静态盘点 + 源码取证；不含运行时压测
- 上游报告：[audit-2026-10-31.md](../audit-2026-10-31.md)（11 条保留发现）
- 关联 issue：#2（M0 首阶段）

本文件回答三个问题：**现在是什么结构**、**哪些面在对外**、**上一轮审计还剩什么**。目标架构见 [target-architecture.md](target-architecture.md)，改造排期见 [refactoring-roadmap.md](refactoring-roadmap.md)。

---

## 1. 进程与模块关系

```
┌──────────────────────────────────────────────────────────────────┐
│ adr-desktop.exe (Tauri 2 壳, Rust)                                │
│  lib.rs  ~1450 行 / 105KB    ← 窗口·托盘·单实例·配置·进程守护      │
│  update.rs 34KB  账号 account.rs 9KB                             │
│  shell.json  %LOCALAPPDATA%\ADR\shell.json                       │
└───────────────┬──────────────────────────────────────────────────┘
                │ spawn + 环境变量注入 + 180s 健康探活 + 守护重启(≤4)
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ python -m adr.server --addr <127.0.0.1|0.0.0.0> --port <p>       │
│                     [--from-shell]                                │
│  adr/server/app.py  应用工厂 + 6 层中间件 + 5 个路由组             │
│    ├─ v2_compat   /api/v2      GSV api_v2 契约                    │
│    ├─ v3_compat   /api/v3      N.E.K.O voices + WS 双工            │
│    ├─ openai_compat /v1        OpenAI Audio 兼容                  │
│    ├─ native      /api/adr/v1  档案/健康/就绪                      │
│    ├─ console     /api/adr/v1  训练/模型/自检 + 静态控制台页        │
│    └─ 静态页 /, /pro, /easy, /call, /server                       │
│  adr/services/SynthesisService   参数校验·档案解析·流式字节契约     │
│  adr/models  gsv_engine(单例) · gsv_runtime(段缓存)                │
│              gsv_coalescer(跨请求合批) · gsv_bootstrap(底模)        │
│              voice_library · voice_package · rvc/diffsinger        │
│  adr/core    settings.py(ADR_* 单源) · config.py(数据根)           │
│  third_party/gpt_sovits  钉版上游 + 24 补丁                        │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
     数据面：<data>/voices · <data>/tts_cache · <data>/logs
             third_party/gpt_sovits/pretrained_models（首启自引导）
```

容器形态去掉最上层壳：`docker run` 直接以 `python -m adr.server --addr 0.0.0.0`（[Dockerfile](../../docker/Dockerfile#L153)）启动，其余完全相同。Windows 无 UI 服务版走 [server.bat](../../scripts/installer/server.bat)。

## 2. 分层现状

| 层 | 位置 | 职责 | 规模/证据 |
| --- | --- | --- | --- |
| 交付壳 | `desktop/src-tauri/src` | 窗口/托盘/单实例/配置/进程守护/更新/账号 | lib.rs 105KB；update.rs 33.8KB；account.rs 9.3KB |
| 协议适配 | `adr/server/{v2,v3,openai}_compat.py` | 三套对外协议 → 内部 req dict | 每套 100~600 行 |
| 原生 API | `adr/server/native.py` | 档案 CRUD、导入导出、历史回滚 | 前缀 `/api/adr/v1` |
| 管理面 | `adr/server/console.py` | 训练/模型/自检/统计 + 静态页 | 前缀 `/api/adr/v1`，同前缀双 router |
| 服务层 | `adr/services/` | `SynthesisService`：参数校验、档案解析、流式字节契约 | 协议层不直接碰引擎 |
| 引擎层 | `adr/models/` | GSV 单例、段级缓存、跨请求合批、底模引导、档案库、RVC/DiffSinger | 23 个模块 |
| 配置 | `adr/core/{settings,config}.py` | 环境变量单源 / 数据根推导 | settings 394 行 |
| 上游 | `third_party/gpt_sovits` | 钉版 + 补丁 | 不入 git |

结构性判断：**协议层 → 服务层 → 引擎层**的主干已经清楚，问题集中在两端——壳层单文件承载了 12 类职责，协议层有 5 套并行面需要同时演进。

## 3. 对外接口面

### 3.1 HTTP 路由全集

| 组 | 前缀 | 路由 |
| --- | --- | --- |
| GSV v2 兼容 | `/api/v2` | `GET/POST /tts`；`GET/POST /set_gpt_weights`；`GET/POST /set_sovits_weights` |
| GSV v3 兼容 | `/api/v3` | `GET /voices`；`WS /tts/stream-input` |
| OpenAI 兼容 | `/v1` | `POST /audio/speech`；`GET /models` |
| ADR 原生 | `/api/adr/v1` | `GET /health`、`GET /readyz`、`GET /profiles`、`GET /profiles/{name}/history`、`POST /profiles/{name}/rollback`、`GET /profiles/{name}/ref`、`GET /profiles/{name}/export`、`POST /profiles/import`、`POST /tts`、`GET /tts` |
| 控制台 API | `/api/adr/v1` | `GET /system/stats`、`GET/POST /system/doctor`、`POST /train/start`、`GET /train/status`、`POST /train/stop`、`POST /train/upload`、`POST /profiles/create`、`GET /profiles/create/status`、`GET /models`、`POST /models/download`、`GET /models/download/status` |
| 静态页 | 根 | `/`、`/pro`、`/easy`、`/call`、`/server`、`/i18n.js`、`/i18n-{page}.js`、`/theme.js` |

中间件顺序（`add_middleware` 反向即最外层，见 [app.py](../../adr/server/app.py#L241-L284)）：
`SecurityHeaders` → `JSONCharset` → `CORS` → `OriginGuard` → `APIKey` → `BodySizeLimit`。

两处命名冲突风险：`native` 与 `console` **共用** `/api/adr/v1` 前缀（靠路径不重叠维持），OpenAI 的 `/models` 与 console 的 `/models` 分属不同前缀，新增路由时必须先确认落在哪个 router。

### 3.2 Tauri 命令（30 条）

`generate_handler!` 注册清单（[lib.rs](../../desktop/src-tauri/src/lib.rs#L546-L577)）：

| 分类 | 命令 |
| --- | --- |
| 控制台导航 | `launch_console`、`enter_console`、`enter_launcher`、`back_to_launcher`、`on_launcher_ready` |
| 引擎/预热 | `engine_status`、`skip_prewarm`、`retry_prewarm`、`check_models` |
| 配置 | `load_shell_settings`、`save_shell_settings`、`app_paths`、`lan_info`、`open_path` |
| 系统集成 | `autostart_get`、`autostart_set`、`cpu_compat_auto` |
| 条款/向导 | `terms_agreed`、`agree_terms`、`reject_terms`、`setup_done`、`complete_setup` |
| 账号 | `account_info`、`account_set`、`account_login`、`account_logout` |
| 更新 | `check_update`、`download_update`、`install_update`、`package_hint` |

事件面很薄：壳不使用 `emit` 广播，前端通过轮询 `engine_status` + `window.__adrPrewarm` 注入钩子获取状态（[lib.rs](../../desktop/src-tauri/src/lib.rs#L1413-L1421)）。

### 3.3 子进程参数

| 模式 | 命令行 |
| --- | --- |
| `pro` / `easy` / `call` / `settings` | `python -m adr.server --addr <h> --port <p> --from-shell` |

端口决策顺序：`shell.json.service_port` > 9881 > 随机空闲端口（[lib.rs](../../desktop/src-tauri/src/lib.rs#L1269-L1288)）。

### 3.4 环境变量生产/消费

| 变量 | 生产者 | 消费者 |
| --- | --- | --- |
| `ADR_TTS_API_KEY` | 壳（shell.json）/ docker | `settings.api_keys` → `APIKeyMiddleware` |
| `ADR_DATA_DIR` | 壳 / docker / server.bat | `config.adr_data_dir` → 档案/日志/缓存 |
| `ADR_ALLOW_UNAUTHENTICATED` | **仅壳的「对外服务」开关** / compose 显式 | `__main__` 启动校验 |
| `ADR_PORT` | Dockerfile | 容器 CMD 与 HEALTHCHECK 唯一源 |
| `ADR_ALLOW_QUERY_API_KEY` | 运维 | `settings.allow_query_api_key`（默认关） |
| `ADR_TTS_AUDIO_ROOTS` | 容器/局域网部署 | `v2_compat._check_ref_audio` 白名单 |
| `ADR_CORS_ORIGINS` / `ADR_ALLOWED_HOSTS` | 运维 | `app.create_app` → CORS / OriginGuard |
| `ADR_TTS_DEFAULT_PROFILE` | 壳 | `app.state.default_profile` |
| `HF_ENDPOINT` | 壳（镜像开关） | huggingface_hub |
| `NUMBA_CACHE_DIR` | 壳 / 运维 | numba |
| `ADR_TTS_DEVICE` / `ADR_TORCH_THREADS` | 壳 | `gsv_engine` |
| `ADR_GPU_MEM_FRACTION` / `ADR_SCHED_MAX_CONCURRENCY` / `ADR_T2S_COALESCE` | 壳（强度档位） | engine / scheduler |
| `ADR_LOG_LEVEL` / `ADR_S2_NUM_WORKERS` | 壳 | logging / 训练 DataLoader |
| `ADR_DOWNLOAD_RATE_MBPS` | 壳 | bootstrap 下载限速 |

单源结论：**`adr/core/settings.py` 基本成立**——文件头明确列出三类"不在此收编"的例外（`ADR_DATA_DIR`/`ADR_CONFIG`/`ADR_PRESET` 归 `core/config.py`；第三方桥接变量；引擎启动期以 `setdefault` 写默认值的 `ADR_T2S_*`）。`app.py` 已无裸 `os.environ`。例外清单是隐式契约，新增变量容易漏登记。

### 3.5 重复状态源

| 信息 | 出现位置 | 风险 |
| --- | --- | --- |
| API Key | shell.json / 壳进程 env / 子进程 env / URL 查询串 | 改 key 未重启时热通道用新 key、服务端仍用旧 key（`console_key_query` 注释已标注该 shortcut） |
| 端口 | shell.json `service_port` / 探测结果 / `ADR_PORT`（容器） / URL 查询 | 桌面与容器两套口径，桌面无 `ADR_PORT` |
| 数据目录 | shell.json `data_dir` / 壳 `adr_data_dir()` / `ADR_DATA_DIR` / `config.adr_data_dir()` | 三处推导逻辑必须保持同序，当前靠注释与双写维持 |
| 设备 | shell.json `device` / `ADR_TTS_DEVICE` / `EngineConfig.device` | env 仅在 config=="auto" 生效，优先级藏在引擎注释里 |
| 强度档位 | shell.json `strength`（不落盘） → `ADR_SCHED_MAX_CONCURRENCY` + `ADR_T2S_COALESCE` | 映射表只存在于 Rust `env_overrides` |
| 默认档案 | shell.json / `ADR_TTS_DEFAULT_PROFILE` / `app.state.default_profile` | 单向下传，无回读一致性校验 |

## 4. 依赖来源与版本矩阵

五处依赖声明 + 一份对账门禁（[check_requirements.py](../../scripts/check_requirements.py)）：

`pyproject.toml`（区间）→ `requirements.txt`（同源区间）→ `scripts/build_runtime.ps1`（Windows 钉版）→ `docker/requirements-linux.txt`（Linux 钉版）→ `docker/Dockerfile`（torch 单独装）。

| 组件 | 桌面 CPU | 桌面 GPU | Docker CPU | Docker CUDA |
| --- | --- | --- | --- | --- |
| Python | 3.13.16（PBS tag 20261009 + SHA256） | 同左 | 3.11-slim | 3.11-slim |
| torch/torchaudio | 2.10.0（PyPI CPU 轮） | 2.10.0+cu130（download.pytorch.org） | 2.10.0（cpu index） | 2.10.0（cu130 index） |
| GSV 上游 | 本地 `third_party` | 同左 | `GSV_COMMIT=d523079f…` clone + 补丁 | 同左 |

已消解的漂移：PBS 由 `releases/latest` 改为**钉标签 + SHA256**（[build_runtime.ps1](../../scripts/build_runtime.ps1#L74-L103)）；`chardet==7.7.0` 已钉；`pandas==2.3.3` 已进 Linux 清单；Tauri CLI 改用 `desktop/node_modules/.bin/tauri.cmd` 固定版本（[build_installer.ps1](../../scripts/build_installer.ps1#L69-L74)）。

残余维护成本：**同一份版本清单维护 4 份**（Windows 钉版、Linux 钉版、pyproject 区间、CI constraints），`check_requirements.py` 只做区间覆盖断言，不做逐条相等断言——同一包在 Windows 与 Linux 可以合法地钉不同版本。

## 5. 数据路径与生命周期

| 数据类型 | 默认路径 | 覆盖变量 | 生命周期 |
| --- | --- | --- | --- |
| 壳配置 | `%LOCALAPPDATA%\ADR\shell.json` | — | 壳启动加载 + 保存后刷新缓存；坏 JSON 归默认 |
| 数据根 | `%LOCALAPPDATA%\ADR\data`（Linux `~/.adr/data`） | `ADR_DATA_DIR`；历史 `F:/ADR_data` 存在即沿用 | 只推导不创建迁移；改 `data_dir` **仅提示不自动搬** |
| 声音档案 | `<data>/voices/<name>/`：`ref.wav`、`meta.json`、三份权重、`weights_history` | 同上 | `save_voice` 建档；`bind_weights` 追加版本链；`rollback` 回退 |
| TTS 缓存 | `<repo>/data/tts_cache` | `ADR_TTS_CACHE_DIR` / `ADR_TTS_CACHE` / `_MAX` / `_MB` | 32 条 / 256MB LRU；文件名带 `v<版本>-` 前缀，未知前缀启动即删 |
| 段级缓存 | 进程内 | `ADR_SEG_CACHE*` | LRU，重启即失 |
| 底模/预训练 | `third_party/gpt_sovits/pretrained_models` | — | 首启 bootstrap，HF tree API 的 `lfs.oid` 作 SHA256；`.part` 断点续传 |
| 日志 | `ADR_LOG_DIR` > `<data>/logs` | `ADR_LOG_DIR` | 壳侧 `server.log` 10MB 单代轮转 |
| 训练中间产物 | `third_party/gpt_sovits/{GPT,SoVITS}_weights_v2`、`TEMP`、`output` | — | 卸载器可选清理（`[UninstallDelete]`） |

数据与代码混杂点已被打包体检规则覆盖：`check_runtime.py` 拒绝 runtime 内出现 `data/`、`output/`、`logs/`、根级裸 `.pth`、`pretrained_models`（[check_runtime.py](../../scripts/check_runtime.py#L54-L59)）。

## 6. 测试与门禁

| 面 | 现状 |
| --- | --- |
| 测试规模 | `tests/` 60 个文件、888 个 `def test_`（参数化后全量 969 passed / 1 skipped） |
| Linux CI | 3.11：三道门禁 → `pip install -c docker/requirements-linux.txt -e .[all,dev]` → 全量 pytest + `--cov-fail-under=73` |
| Windows CI | 3.11 + 3.13 矩阵，全量 pytest（不装 multilang extras，相关用例自带 skip） |
| Rust CI | `cargo test --locked`（update.rs 9 + lib.rs 10） |
| 发布 CI | `docker-publish.yml`：版本/合规/依赖/语法 gate 后才推镜像 |
| 出包门禁 | `build_installer.ps1` 前置 `check_versions.py` + `check_compliance.py` + `check_runtime.py`，失败 `throw` |
| 覆盖缺口 | 无前端页面自动化、无安装/升级/回滚 E2E、无 docker 镜像冒烟、无性能回归基线 |

## 7. 上一轮审计 11 条复核

| 编号 | 主题 | 状态 | 证据 |
| --- | --- | --- | --- |
| S1 | Docker 默认对外且无鉴权 | 已修复 | compose `127.0.0.1:9881:9881`；`ADR_ALLOW_UNAUTHENTICATED` 由 compose 显式声明；镜像不再写死 |
| S2 | OpenAI 兼容层与官方 SDK 不兼容 | 已修复 | `stream: bool = False`；`_OPENAI_MODELS`/`_OPENAI_VOICES` 白名单跳过标准名 |
| S3 | Tauri 打包配置不完整 | 已修复 | `bundle.active=false` + `icon` 仅 ICO；Inno 为唯一安装器；NSIS 键已删 |
| Q1 | 构建不可复现 | 已修复 | PBS 钉 tag + SHA256；`chardet==7.7.0`；Tauri CLI 走本地 node_modules |
| Q2 | 出包未跑版本/合规门禁 | 已修复 | `build_installer.ps1` L52-67 三道门禁 |
| Q3 | `check_runtime.py` 默认路径不存在 | 已修复 | `_default_runtime()` 探测 cpu/gpu 两候选 |
| Q4 | pandas 隐式依赖 | 已修复 | `docker/requirements-linux.txt` 含 `pandas==2.3.3` |
| Q5 | 缓存键无版本 | 已修复 | `CACHE_VERSION` + 权重内容指纹 + 文件名前缀 |
| Q6 | v3 overrides 被静默忽略 | 已修复 | `_normalize_overrides` 走 `param_norm`，非法值回错误帧；`_produce` 叠加 overrides |
| Q7 | 编码/写盘阻塞事件循环 | 已修复 | `_encode_and_cache` 整体进线程池 |
| Q8 | 合批组内 assert | 已修复 | 改为显式 `ValueError`（[gsv_coalescer.py](../../adr/models/gsv_coalescer.py#L614-L619)） |

11 条全部关闭。需要**运行时复测**而非静态确认的只有两项：S2（用官方 `openai` SDK 打一次真实请求）、Q7（并发下 P99 对比）。

## 8. 现状结论（剩余结构性问题）

代码质量高、分发链路已补齐，剩余问题都是**结构可维护性**而非缺陷：

1. `lib.rs` 105KB 单文件承载 12 类职责，配置/进程/导航/系统集成互相直调，任何一处改动都要通读全文件。
2. 五套 HTTP 面共用一套内部 req dict，参数归一已收口到 `param_norm`，但**协议语义**（错误外壳、流式默认值、字段命名）仍分散在三个 compat 文件。
3. 依赖清单四处并行维护，靠区间断言而非相等断言兜底。
4. 端口/数据目录/设备/Key 各有 3~5 个状态源，一致性靠注释和双写约定。
5. 壳与服务的契约（命令行参数、环境变量、健康探针路径）没有单一文档，散落在 Rust 注释与 Python docstring。
