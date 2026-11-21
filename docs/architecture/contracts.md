# ADR 壳 ↔ 服务 ↔ 数据 契约（M1）

- 状态：**单一事实源文档**。代码变更若触及本文任一条目，必须同批更新本文与 `ENV_REGISTRY`
- 机器可读名单：[settings.py `ENV_REGISTRY`](../../adr/core/settings.py#L63-L152)（`tests/test_env_registry.py` 双向断言）
- 关联：[target-architecture.md](target-architecture.md) 第 5 节、[audit-current-state.md](audit-current-state.md) 第 3 节

---

## 1. 启动契约

| 形态 | 启动方 | 命令行 | 说明 |
| --- | --- | --- | --- |
| 桌面 pro/easy/call/settings | 壳 `start_server` | `python -m adr.server --addr <h> --port <p> --from-shell` | `--from-shell` 表示条款已由壳层 gate，服务不再 exit 2 |
| Windows 无 UI | `server.bat` | `python -m adr.server --addr <h> --port <p>` | 无 `--from-shell`，需 `ADR_AGREE_TERMS` 或已落条款标记 |
| 容器 | Dockerfile `CMD` | `python -m adr.server --addr 0.0.0.0 --port "${ADR_PORT}"` | `exec` 让 python 接管 PID 1 |

壳只传参数与环境变量，**不写服务配置文件**；服务只读环境变量与命令行，**不回写壳配置**。

## 2. 端口推导

| 形态 | 顺序 | 回退提示 |
| --- | --- | --- |
| 桌面 | `shell.json.service_port`（>0 且可用）→ 9881（可用）→ 系统分配空闲端口 | 9881 被占且未指定时，启动器提示"本次改用端口 N（NEKO 侧需手动填写该端口）" |
| 容器 | `ADR_PORT`（默认 9881） | 探针与服务读同一变量（[Dockerfile](../../docker/Dockerfile#L136-L142)） |

约束：探测结果**不回写** `shell.json`；`ADR_PORT` 只用于容器，桌面不设置该变量。

## 3. 健康探针

| 路径 | 语义 | 鉴权 |
| --- | --- | --- |
| `GET /api/adr/v1/health` | HTTP 可达 + 引擎阶段（`engine_stage` / `engine_stage_text` / `engine_error`） | **豁免**（壳轮询不带 key） |
| `GET /api/adr/v1/readyz` | 引擎就绪且未永久降级；未就绪返回 503 | 豁免 |

壳探活超时 180s；容器 `HEALTHCHECK` 用 `/readyz`，`start-period=600s`（首启下载权重）。

## 4. 环境变量

全量名单与归属见 `ENV_REGISTRY`（60+ 项）。按**谁设置**分组：

| 设置方 | 变量 |
| --- | --- |
| 桌面壳（`shell.json` 映射） | `ADR_TTS_API_KEY`、`ADR_DATA_DIR`、`HF_ENDPOINT`、`ADR_TTS_DEFAULT_PROFILE`、`ADR_T2S_CUDAGRAPH`、`NUMBA_CACHE_DIR`、`ADR_TTS_DEVICE`、`ADR_TORCH_THREADS`、`ADR_LOG_LEVEL`、`ADR_S2_NUM_WORKERS`、`ADR_DOWNLOAD_RATE_MBPS`、`ADR_GPU_MEM_FRACTION`、`ADR_TTS_STREAM_HEAD`、`ADR_TTS_SPLIT_MAX`、`ADR_RUNTIME_AUTOTUNE`、`ADR_SAFETY_AUDIT`、`ADR_WAV_METADATA`、`ADR_SCHED_MAX_CONCURRENCY`、`ADR_T2S_COALESCE`、`ADR_ALLOW_UNAUTHENTICATED`；兼容模式另注入 `ONEDNN_MAX_CPU_ISA` / `MKL_ENABLE_INSTRUCTIONS` / `NUMBA_CPU_NAME` |
| 壳自身读取 | `ADR_DESKTOP_PYTHON`、`ADR_LOG_DIR`、`ADR_DATA_DIR`、`ADR_BUILD_VARIANT` |
| 容器 | `ADR_DATA_DIR=/data`、`ADR_TTS_CACHE_DIR`、`HF_HOME`、`ADR_PORT`、`ADR_ALLOW_UNAUTHENTICATED`（compose 显式） |
| 运维 / 局域网 | `ADR_TTS_AUDIO_ROOTS`、`ADR_CORS_ORIGINS`、`ADR_ALLOWED_HOSTS`、`ADR_ALLOW_QUERY_API_KEY` |
| 引擎调优（默认不需要动） | `ADR_TTS_BATCH_SIZE`、`ADR_T2S_BATCH_RNG`、`ADR_T2S_COALESCE_BATCH`、`ADR_T2S_STATIC_KV`、`ADR_T2S_CUDAGRAPH_BUCKETS`、`ADR_AR_*`、`ADR_ENGINE_WATCHDOG_SEC` |
| 缓存 | `ADR_TTS_CACHE`、`ADR_TTS_CACHE_DIR`、`ADR_TTS_CACHE_MAX`、`ADR_TTS_CACHE_MB`、`ADR_SEG_CACHE*` |
| 上限/安全 | `ADR_MAX_BODY_MB`、`ADR_PROFILE_IMPORT_MAX_MB`、`ADR_TTS_MAX_TEXT_CHARS`、`ADR_TTS_MAX_QUEUE`、`ADR_TTS_REF_MAX_MB`、`ADR_TTS_WEIGHTS_MAX_MB`、`ADR_TTS_SPEED_MIN/MAX`、`ADR_FFMPEG_TIMEOUT_SEC` |
| 数据/资产 | `ADR_DATA_DIR`、`ADR_CONFIG`、`ADR_PRESET`、`ADR_MODEL_SOURCE`、`ADR_DOWNLOAD_RATE_MBPS`、`ADR_IMPORT_MAX_*` |

约定：

1. `ADR_*` 的**解析口径**只在 `adr/core/settings.py`（服务/引擎开关）或明确登记的归属模块中出现一次。
2. 新增变量必须登记 `ENV_REGISTRY`；仅写不读（如 `core/policy.py`）也要登记归属。
3. 布尔口径不统一是**有意**的：`=0 关闭`（缓存/安全）与 `=1 开启`（调试/条款）各自沿用历史语义，不得"顺手统一"。

## 5. `shell.json` → 环境变量映射

文件：`%LOCALAPPDATA%\ADR\shell.json`（`LOCALAPPDATA` 缺失时落到数据目录旁）。
映射实现：Rust `env_overrides()` 纯函数（[lib.rs](../../desktop/src-tauri/src/lib.rs#L227-L320)），空串/未设置一律跳过。

| 字段 | 环境变量 | 备注 |
| --- | --- | --- |
| `api_key` | `ADR_TTS_API_KEY` | 逗号分隔多 key |
| `data_dir` | `ADR_DATA_DIR` | 仅提示迁移，不自动搬 |
| `hf_mirror` | `HF_ENDPOINT=https://hf-mirror.com` | 布尔 |
| `default_profile` | `ADR_TTS_DEFAULT_PROFILE` | |
| `cudagraph` | `ADR_T2S_CUDAGRAPH=1` | 音质有损，默认关 |
| `numba_cache_dir` | `NUMBA_CACHE_DIR` | |
| `device` | `ADR_TTS_DEVICE` | 仅 `cpu`/`cuda` 注入，`auto` 不注入 |
| `torch_threads` | `ADR_TORCH_THREADS` | `>0` 才注入 |
| `log_level` | `ADR_LOG_LEVEL` | 白名单 DEBUG/INFO/WARNING/ERROR |
| `s2_num_workers` | `ADR_S2_NUM_WORKERS` | `>0` 才注入 |
| `download_rate` | `ADR_DOWNLOAD_RATE_MBPS` | 0.1~1024 |
| `compat_cpu` | `ONEDNN_MAX_CPU_ISA` / `MKL_ENABLE_INSTRUCTIONS` / `NUMBA_CPU_NAME` | 布尔，或自动检测 Raptor Lake |
| `safety_audit` | `ADR_SAFETY_AUDIT=0` | 仅显式关闭时注入 |
| `wav_metadata` | `ADR_WAV_METADATA=0` | 同上 |
| `gpu_mem_fraction` | `ADR_GPU_MEM_FRACTION` | 0.1~1.0 |
| `stream_head` | `ADR_TTS_STREAM_HEAD` | `>0` 才注入 |
| `split_max` | `ADR_TTS_SPLIT_MAX` | `>0` 才注入 |
| `autotune` | `ADR_RUNTIME_AUTOTUNE=0` | 仅显式关闭时注入 |
| `strength` | `ADR_SCHED_MAX_CONCURRENCY` + `ADR_T2S_COALESCE` | low=1/0，mid=2，balanced=3，strong=4，max=6；不单独落环境变量 |
| `lan_expose` / `service_port` | `ADR_ALLOW_UNAUTHENTICATED` / 命令行 `--addr` `--port` | 不映射成 env |

## 6. 数据目录优先级

| 优先级 | 来源 | 备注 |
| --- | --- | --- |
| 1 | `shell.json.data_dir` | 仅桌面壳 |
| 2 | `ADR_DATA_DIR` | 壳注入、`server.bat`、容器 |
| 3 | `F:/ADR_data` | 历史默认，**目录存在才沿用** |
| 4 | `%LOCALAPPDATA%\ADR\data` | Windows 默认 |
| 5 | `~/.adr/data` | 非 Windows 兜底 |

壳与服务遵循同一顺序（Rust [adr_data_dir](../../desktop/src-tauri/src/lib.rs#L967-L992) / Python [config.py](../../adr/core/config.py#L27-L44)）。`shell.json → ADR_DATA_DIR` 是唯一跨层传递方式；改 `data_dir` **不自动迁移**已有数据。

## 7. 凭据通道

| 通道 | 默认 | 说明 |
| --- | --- | --- |
| `Authorization: Bearer <key>` | 开 | 首选 |
| `X-API-Key: <key>` | 开 | 控制台页面使用 |
| `?api_key=` | **关** | 仅 `ADR_ALLOW_QUERY_API_KEY=1/true` 开启；查询串会留在历史/Referer/反代日志 |
| WS `?token=` / `?api_key=` / 子协议 | 开 | 浏览器 WS 无法自定义头，保留 |

未设置任何 key = 鉴权关闭，此时由 `OriginGuardMiddleware` 拦截跨站请求；`0.0.0.0` + 无 key 且未显式 `ADR_ALLOW_UNAUTHENTICATED=1` 时服务拒绝启动。

## 8. 版本单一事实源

`pyproject.toml` 的 `version` 是唯一事实源（当前 1.3.1）。`check_versions.py` 断言 12 个面：`adr/__init__.py`、`tauri.conf.json`、`Cargo.toml`、`package.json`、`package-lock.json`、`installer.iss`、`launcher.html`、`README.md`、`Dockerfile ARG`、`docker-compose*.yml` 镜像 tag、`docker/README.md`。发布时 `--tag` 额外校验 git tag。

## 9. 变更规则

1. 新增环境变量 → 登记 `ENV_REGISTRY` + 本文第 4 节 + `shell.json` 映射（若由壳设置）。
2. 新增 Tauri 命令 → `generate_handler!` + 本文第 1 节形态表（若涉及启动）。
3. 新增 HTTP 路由 → 确认前缀归属（`/api/v2`、`/api/v3`、`/v1`、`/api/adr/v1`，见第 10 节），并注意 `native` 与 `console` 共用前缀；同 `method + path` 重复注册会被 `assert_unique_routes()` 在启动期拦下。
4. 改端口/数据目录/凭据逻辑 → 必须同时核对第 2、6、7 节与两侧实现。
5. 本文与 `ENV_REGISTRY` 不一致时，以代码 + 门禁测试为准，并当批修文档。

## 10. 协议面与错误码（M3）

| 协议面 | 前缀 | 所属模块 |
| --- | --- | --- |
| GSV api_v2 兼容 | `/api/v2` | `adr/server/v2_compat.py` |
| ADR v3 WebSocket | `/api/v3` | `adr/server/v3_compat.py` |
| OpenAI 兼容 | `/v1` | `adr/server/openai_compat.py` |
| ADR 原生 + 控制台 | `/api/adr/v1` | `native.py`（health/readyz/profiles/tts）+ `console.py`（system/train/models/控制台页） |

`/api/adr/v1` 由两个 router 共同承载。`create_app()` 末尾调用
`assert_unique_routes()`：同一 `method + path` 被注册两次直接 `RuntimeError`，
不会静默覆盖成后注册者。新增路由前按第 9 节第 3 条确认归属。

wire 错误码单点是 `adr/server/exceptions.py` 的 `WIRE_CODES`。v2 / v3 WS /
OpenAI / native / 中间件 / 鉴权 401 只引用常量与 `json_error` / `ws_error`
外壳构造器，协议文件不再写 `code` 字面量。媒体类型错误沿用历史 wire 值
`invalid_params`（语义键 `unsupported_media` 仅登记、不换值），调用方现有的
字符串匹配零改动；`Content-Type` 的唯一出口是
`SynthesisService.content_type()`。门禁：`tests/test_m3_contract.py`。
