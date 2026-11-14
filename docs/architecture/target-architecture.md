# ADR 目标架构（M0）

- 基线：1.3.1
- 约束来源：issue #2 设计红线
- 现状依据：[audit-current-state.md](audit-current-state.md)

本文只定义**边界与数据流**，不定义实施步骤（见 [refactoring-roadmap.md](refactoring-roadmap.md)）。

---

## 1. 不可动的红线

| 红线 | 含义 | 对架构的约束 |
| --- | --- | --- |
| GSV 兼容契约 | `/api/v2/*`、`/api/v3/*` 的路径、字段、字节契约不变 | 协议层可内部重构，wire 格式只能追加不能改 |
| OpenAI 兼容契约 | `/v1/audio/speech` 对齐官方 SDK 默认行为 | `stream` 默认 false 不可回退；标准 model/voice 永不报 400 |
| 离线推理 | 断网可完成合成；只有首次底模下载需要网络 | 任何新组件不得引入运行时联网依赖 |
| 用户数据 | 档案、权重历史、训练产物、缓存 | 首阶段不改目录布局与 `meta.json` schema |
| CPU/GPU 同构 | 不为小功能建独立 Python 环境 | 两包共用一个 runtime 布局，仅 torch 变体不同 |
| 分阶段合并 | 先验收后改造 | 每个里程碑必须有可执行验收清单 |

## 2. 目标分层

```
L0 交付层    desktop shell │ server mode (server.bat) │ container
             职责：进程生命周期、端口、配置持久化、导航、更新
             禁止：业务逻辑、模型加载、档案读写
                          │ 唯一契约：命令行 + 环境变量 + HTTP
                          ▼
L1 协议层    v2_compat │ v3_compat │ openai_compat │ native │ console
             职责：协议解析/序列化、错误外壳、流式语义映射
             禁止：直接加载模型、直接读写档案文件
                          │ 统一内部请求对象（req dict / dataclass）
                          ▼
L2 应用层    SynthesisService │ ProfileService │ TrainService │ ModelService
             职责：参数校验与归一、档案解析、权限/配额、任务编排
             禁止：协议细节、torch 细节
                          │ 领域调用
                          ▼
L3 引擎层    gsv_engine │ gsv_runtime │ gsv_coalescer │ rvc │ diffsinger │ adr2
             职责：模型加载、推理、合批、设备选择、缓存
             禁止：HTTP、文件协议、用户配置
                          │
                          ▼
L4 运行时层  runtime manifest │ bootstrap │ dependency doctor
             职责：解释器/依赖/资产/补丁的声明、校验、引导
L5 数据层    voices │ tts_cache │ logs │ pretrained_models
             职责：路径推导、生命周期、迁移
```

依赖方向**严格向下**：L1 不 import L3 的具体引擎模块（只经 L2），L3 不 import `adr.server`。当前代码中 L1 通过 `_get_engine()` 惰性拿单例，属于允许的"取句柄"而非"取实现"。

## 3. 模块边界

### 3.1 交付层（壳）

| 模块 | 单一职责 | 对外接口 |
| --- | --- | --- |
| `config` | `shell.json` 读写 + 校验 + 变更通知 | `ShellSettings`、`save/load` |
| `sidecar` | 解释器解析、命令行拼装、环境注入、spawn、日志泵 | `start(mode, expose) -> Child` |
| `lifecycle` | 状态机 `Stopped/Starting/Prewarming/Ready`、健康探活、守护重启、孤儿收编 | `phase()`、`ensure_running()` |
| `windows` | 主窗口/导航/下载事件/托盘 | `navigate(page)` |
| `system` | 注册表自启、打开路径、CPU 兼容检测、单实例锁 | 平台命令 |
| `update` / `account` | 更新链 / 本地账号 | 保持现状 |
| `commands` | 仅做参数解包 + 调上述模块 | 30 条 Tauri 命令 |

判定标准：任一模块被删除时，其余模块只需改 import，不需改逻辑。

### 3.2 协议层

每个 compat 文件只做三件事：**解析**（wire → 内部对象）、**映射**（内部结果 → wire）、**错误外壳**。参数归一与区间校验统一走 `adr/server/param_norm.py`（已具备），协议层不得另写一套数值校验。

### 3.3 应用层

`SynthesisService` 是唯一合成入口：`validate_params` → `load_profile` → `synthesize/stream_bytes`。协议层禁止绕过它直接调引擎。

### 3.4 引擎层

`gsv_engine` 单例对外只暴露：`get_gsv_engine()`、`prewarm()`、`synthesize/stream`、`set_weights`、`queue_depth()`、设备解析结果。合批与缓存是引擎内部策略，不向协议层暴露开关以外的细节。

## 4. 关键数据流

### 4.1 非流式合成

```
client → /api/v2|v3|v1 → 协议解析
  → SynthesisService.validate_params（长度/数值/语言/媒体类型）
  → tts_cache.make_key（含 CACHE_VERSION + 权重指纹）
  → 命中？直接返回
  → 未命中：调度准入（queue_depth < ADR_TTS_MAX_QUEUE）
  → gsv_engine（锁内 / 合批 worker）
  → _encode_and_cache（线程池：ffmpeg 编码 + 落盘）
  → 响应
```

关键不变式：**协议层永不持有引擎锁**；编码与写盘永不在事件循环线程。

### 4.2 流式合成

- v2 `streaming_mode=1`：`SynthesisService.stream_bytes` 逐块产出；wav 首块为 44B 头，其后裸 PCM。
- v3 WS `/tts/stream-input`：文本流入 → 句级切片 → 逐句合成 → 二进制音频帧 + JSON 控制帧；断连必须释放引擎锁（已在 `_push_task` finally 实现）。
- OpenAI `stream=true`：映射到 v2 `streaming_mode`。

三条流式路径必须共用同一套切片与字节契约，差异只在帧封装。

### 4.3 档案写入

```
console/native → voice_library.save_voice（ref.wav + meta.json）
  → bind_weights（旧权重入 weights_history，新权重就位）
  → 质量门（可选）→ 退化标记
  → rollback 只回退权重，不动 ref/meta
```

### 4.4 首启引导

```
server 启动 → gsv_bootstrap.ensure()
  → 检查 pretrained_models 完整性（HF tree lfs.oid 作 SHA256）
  → 缺则下载（.part 续传 + 限速 + 多源重试）→ 解压
  → 校验通过才允许引擎加载
```

引导失败必须显式报错到 `/health` 的 `engine_error`，不允许无限挂起。

### 4.5 出包

```
build_runtime.ps1 → build/runtime-{cpu,gpu}
  → check_runtime.py（垃圾/数据面/底模/变体）
  → tauri build --no-bundle（只产 exe）
  → ISCC installer.iss（唯一安装器，载荷 runtime 目录）
  → 门禁：check_versions / check_compliance / check_requirements
```

## 5. 状态源单一化目标

| 信息 | 唯一事实源 | 派生方式 |
| --- | --- | --- |
| 服务端口 | 壳 `service_port`（桌面）/ `ADR_PORT`（容器） | 探测结果只作运行时记录，不回写配置 |
| 数据目录 | `ADR_DATA_DIR` 环境变量 | 壳与服务都用同一推导函数语义；`shell.json.data_dir` 只负责生成该变量 |
| API Key | `shell.json.api_key`（桌面）/ `ADR_TTS_API_KEY`（容器） | 壳启动时注入；热通道读同一值 |
| 推理设备 | `shell.json.device` → `ADR_TTS_DEVICE` | 引擎只读 env，`auto` 时不注入 |
| 默认档案 | `shell.json.default_profile` → `ADR_TTS_DEFAULT_PROFILE` | `app.state.default_profile` 是只读快照 |
| 强度档位 | 壳内映射表 | 展开成显式 env，不再有第二个映射点 |
| 版本 | `pyproject.toml` | `check_versions.py` 断言 12 个面 |

## 6. CPU / CUDA 兼容策略

- **一个运行时布局，两种 torch 变体**：`runtime/BUILD_VARIANT.txt`（`cpu`/`gpu`）是唯一判据；安装器按 `VariantChanged` 决定是否清 `runtime\python`。
- **同一 AppId**：GPU 包装在 CPU 包上 = 原地升级 runtime，不产生双份安装；换包型时才重装 Python 层。
- **设备解析唯一出口**：`gsv_engine.resolve_runtime_device`；CUDA 不可用一律回落 CPU，并在日志中区分"未装 CUDA 轮子"与"驱动过旧"。
- 禁止：为 GPU 单独维护 `requirements-gpu.txt` 或第二份 pyproject；GPU 差异只允许出现在 torch/torchaudio 两行。

## 7. 明确不做

1. 不做目录大搬迁（`adr/models` → `adr/engine/...` 之类）作为第一步。
2. 不做协议层全面重写；只收敛重复的归一与错误外壳。
3. 不为训练/多语言/RVC 各建独立 venv。
4. 不改 `meta.json` schema 与 `.adrvoice` 包格式。
5. 不引入需要联网的运行时组件。
