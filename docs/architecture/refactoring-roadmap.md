# ADR 分层改造路线图（M1–M4）

- 基线：1.3.1
- 依据：[audit-current-state.md](audit-current-state.md) 第 8 节、[target-architecture.md](target-architecture.md)
- 排期原则：**风险升序**。每一阶段结束都必须能出一个可发布的包。

---

## 总览

| 阶段 | 主题 | 风险 | 收益 | 成本 | 可并行 |
| --- | --- | --- | --- | --- | --- |
| M1 | 契约固化 + 状态源收敛 | 低 | 高 | 低 | 部分（互不重叠的文件） |
| M2 | 壳层模块化 | 中 | 高 | 中 | 否（同一 crate） |
| M3 | 协议/服务层收敛 | 中 | 中高 | 中 | 部分（各 compat 文件独立） |
| M4 | Runtime Profile + 模块化安装 | 较高 | 高 | 高 | 否（串行） |

排序理由：M1 只写文档与门禁，不改运行行为，先做可以给后续所有阶段提供验收依据；M2/M3 是纯内部重构，wire 契约不变；M4 触及安装器与分发，必须放在最后，且要有 M1 的验收清单兜底。

---

## M1 契约固化与状态源收敛（低风险）

**目标**：把"壳 ↔ 服务 ↔ 数据"的隐式契约写成唯一文档，并用门禁守住；消除端口/数据目录/Key 的多源推导。

**范围**

1. 新增 `docs/architecture/contracts.md`：命令行参数、环境变量、健康探针路径、端口推导顺序、`shell.json` → env 映射表、数据目录优先级。内容直接来自本批 5 份文档，不新增结论。
2. `settings.py` 的"不在此收编"例外清单升级为显式登记表（模块级常量 + 单测断言），新增 `ADR_*` 变量必须登记。
3. 清点并修正重复状态源：
   - 端口：桌面只认 `service_port`，容器只认 `ADR_PORT`，探测结果不写回；
   - 数据目录：`shell.json.data_dir` 只负责生成 `ADR_DATA_DIR`，壳自身推导改为读取该环境变量（启动时一次性计算后缓存，见 [lib.rs](../../desktop/src-tauri/src/lib.rs#L969-L992) 现有逻辑）；
   - Key：热通道与 sidecar 注入读同一来源，去掉注释里承认的 shortcut。
4. 验收脚本化：`docs/architecture/windows-install-acceptance.md` 的清单转成可勾选记录模板。
5. 补两条缺口测试：OpenAI SDK 端到端（`openai` 指向本地服务，`model="tts-1"`、`voice="alloy"` 断言 200 + 非空音频）；非流式编码线程池的并发回归（并发 8 请求，`/health` 不被阻塞）。

**非目标**：不拆文件、不改 API、不动安装器。

**验收**

- 上述两条测试进入 `tests/`，CI 通过。
- `check_versions.py` / `check_compliance.py` / `check_requirements.py` 全绿。
- 全量 pytest 与基线一致（≥969 passed）。
- 文档中的每条状态源都能在代码里找到唯一实现点。

**风险与回退**：极低。若环境变量登记表导致启动失败，回退为仅告警。

---

## M2 壳层模块化（中风险）

**目标**：把 `lib.rs`（105KB，12 类职责）拆成同 crate 内的 7 个模块，**不换技术方案、不改前端 IPC**。

**范围**

| 新模块 | 从 lib.rs 迁出的内容 |
| --- | --- |
| `config.rs` | `ShellSettings`、`shell_settings_path`、`reload/shell_config`、`env_overrides` |
| `sidecar.rs` | `resolve_runtime`、端口探测、命令行拼装、env 注入、spawn、日志泵/轮转 |
| `lifecycle.rs` | `Phase`、`ServerState`、`wait_healthy`、`supervise`、`kill_current`、孤儿收编 |
| `windows.rs` | 主窗口创建、下载事件、导航、托盘菜单 |
| `system.rs` | 单实例互斥、自启注册表、`open_path`、CPU 兼容检测 |
| `commands.rs` | 30 条 `#[tauri::command]` 的参数解包与转发 |
| `lib.rs` | 只剩 `run()`：装配上述模块 + `generate_handler!` |

约束：

1. 命令名、参数、返回结构**逐字不变**（前端零改动）。
2. `env_overrides` 保持纯函数并沿用现有单测。
3. 每个模块拆完立即 `cargo test --locked`，不攒到最后。
4. 不做异步化改造、不引入 actor/channel 重构。

**验收**

- `cargo test --locked` 通过，测试数量不低于拆前。
- 安装包与本阶段前产物体积差异 < 1%。
- 手工冒烟：启动预热 → 进 pro/easy → 返回启动器 → 托盘退出，四步无异常。
- `lib.rs` 行数 < 300（仅装配）。

**风险与回退**：中等。风险点是 `ServerState` 被命令与守护线程共享，拆分时容易引入锁顺序问题。回退策略：先只拆 `config`/`system`（无共享状态），验证后再拆 `lifecycle`/`sidecar`。

---

## M3 协议层与服务层收敛（中风险）

**目标**：五套 HTTP 面共用一套参数归一与错误外壳，协议文件只剩"编解码 + 语义映射"。

**范围**

1. `param_norm.normalize_params` 已是唯一数值/语言归一入口（v2 与 v3 均调用），本阶段补齐：
   - 错误码映射表集中（`invalid_params` / `unsupported_media` / `profile_not_found` / `payload_too_large` / `busy`），三套协议各自只做外壳转换；
   - 媒体类型白名单与响应头（`Content-Type`）由服务层返回，协议层不重复判断。
2. `native` 与 `console` 共用 `/api/adr/v1` 前缀，增加路由归属注释与启动期冲突断言（同一 method+path 注册两次直接失败）。
3. 缓存治理：`CACHE_VERSION` 已有，补一条"编码参数变更必须 bump 版本"的记录规范（写进 `tts-api-spec.md`）。
4. 流式三路径共用切片参数（`stream_head` / `split_max` / `fragment_interval`），禁止各自读 env。

**非目标**：不合并 v2/v3 路由；不改变任何字段名；不动 OpenAI 默认值。

**验收**

- 三套协议的同一非法参数（如 `top_k=-1`、`text_lang="xx"`）返回结构一致的核心字段（`message` + `code`），外壳各自符合协议。
- 新增路由冲突时服务启动即失败（有单测）。
- 缓存版本变更后旧缓存不再命中（有单测）。
- 全量 pytest 通过，接口契约文档同步更新。

**风险与回退**：中等。错误外壳变动会打破调用方字符串匹配。回退：保留旧 `detail` 字段（当前已保留），只增不改。

---

## M4 Runtime Profile 与模块化安装（较高风险）

**目标**：落地 [runtime-and-module-spec.md](runtime-and-module-spec.md)，让"装什么、缺什么、能不能跑"由声明驱动。

**范围**

1. 新增 `adr/modules.toml` 与 `python -m adr doctor --modules`。
2. 构建脚本生成 `runtime.manifest.json`；`check_runtime.py` 校验清单与模块一致性。
3. `check_requirements.py` 升级为相等断言 + `allowed_drift` 白名单。
4. 安装器读取 `BUILD_VARIANT.txt` 与清单做安装后自检（失败给出明确提示，不回滚安装）。
5. 可选模块降级：缺 `train` / `multilang` / `rvc` 时控制台显示"功能不可用"，而不是请求时报 ImportError。

**非目标**：不做在线依赖安装；不做按需下载 Python 包；不改变 CPU/GPU 双包结构。

**验收**

- 三种环境（桌面 CPU、桌面 GPU、docker-cpu）`doctor --modules` 全绿。
- 手动移除可选包后服务仍可合成，缺失功能有可读提示。
- 出包门禁能拦住"只在 docker 清单新增包"的漂移。
- 按 `windows-install-acceptance.md` 跑完安装/升级/回滚三张表。

**风险与回退**：较高。清单不一致会导致构建失败。回退：清单校验降级为告警（保留 manifest 生成），安装行为不变。

---

## 批次切分建议

| 批次 | 内容 | 门禁 |
| --- | --- | --- |
| M1-a | 契约文档 + 环境变量登记表 | 纯文档 + 单测 |
| M1-b | 状态源收敛 + 两条缺口测试 | 全量 pytest |
| M2-a | 拆 `config` / `system` | `cargo test` + 冒烟 |
| M2-b | 拆 `sidecar` / `lifecycle` | `cargo test` + 安装包冒烟 |
| M2-c | 拆 `windows` / `commands` | 同上 |
| M3-a | 错误码与媒体类型收口 | 全量 pytest + 协议测试 |
| M3-b | 路由冲突断言 + 流式参数统一 | 全量 pytest |
| M4-a | `modules.toml` + doctor | 全量 pytest |
| M4-b | manifest 生成 + 门禁升级 | 双包构建 DryRun |
| M4-c | 安装后自检 + 降级提示 | 三张验收表 |

每条批次单独 commit，`docs/m10-plan.md` 回填背景/范围/验证/是否打包。

## 暂不纳入

- 目录大搬迁、引擎层重写（无收益，风险高）。
- 控制器/事件总线式重构（当前命令式链路足够）。
- 前端框架化（控制台四页为原生 JS，重构成本高于收益）。
- 自动数据迁移（用户数据红线，需单独立项）。
