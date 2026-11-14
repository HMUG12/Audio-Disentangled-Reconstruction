# Windows 安装 / 升级 / 回滚验收清单（M0）

- 适用：ADR Studio 1.3.x 桌面双包（CPU / GPU），安装器为 Inno Setup
- 安装器：[installer.iss](../../scripts/installer/installer.iss)（`AppId {6E1F3C7A-…}`，双包共用）
- 出包脚本：[build_installer.ps1](../../scripts/build_installer.ps1)
- 使用方式：每个发布候选包按表逐项执行，记录"通过 / 失败 / 跳过 + 证据"；任一项失败即该包不可发布

---

## 0. 记录模板

| 字段 | 值 |
| --- | --- |
| 包文件名 | `ADR-Studio-____-x64[-gpu]-setup.exe` |
| SHA256 | |
| 测试机 | Windows 版本 / 显卡 / 内存 |
| 起始状态 | 全新 / 已装 CPU / 已装 GPU / 已装旧版 |
| 执行人 / 日期 | |

判定"通过"的通用证据：截图 + `%LOCALAPPDATA%\ADR\shell.json` + `<安装目录>\runtime\BUILD_VARIANT.txt` + `<数据目录>\logs\server.log` 尾部 50 行。

---

## 1. 安装前（构建侧，每包必做）

| # | 检查 | 命令/位置 | 通过判据 |
| --- | --- | --- | --- |
| 1.1 | 版本面一致 | `python scripts/check_versions.py` | 退出码 0 |
| 1.2 | 合规门禁 | `python scripts/check_compliance.py` | 退出码 0 |
| 1.3 | 依赖漂移 | `python scripts/check_requirements.py` | 退出码 0 |
| 1.4 | runtime 体检 | `python scripts/check_runtime.py --runtime build/runtime-cpu`（GPU 加 `--allow-cuda`） | 无 FAIL |
| 1.5 | 变体标记 | `build/runtime-{cpu,gpu}/BUILD_VARIANT.txt` | 内容与包型一致 |
| 1.6 | 包体积 | 资源管理器 | CPU ≤ 2.2GB 告警线、GPU ≤ 4.6GB 告警线（超限需记录原因） |
| 1.7 | MAX_PATH | `build_installer.ps1` 输出 | "MAX_PATH 预检通过" |
| 1.8 | 包哈希留档 | `Get-FileHash <setup.exe> -Algorithm SHA256` | 记录到第 0 节 |

---

## 2. 全新安装（CPU 包）

前置：干净的 Windows 10/11 x64，未装过 ADR Studio，无 `HKCU\...\Run\ADR Studio` 键。

| # | 步骤 | 预期 |
| --- | --- | --- |
| 2.1 | 双击安装包 | 出现许可页；拒绝则安装中止且无残留 |
| 2.2 | 选择安装目录（默认 `%LOCALAPPDATA%\Programs\ADR Studio`） | 安装进度正常 |
| 2.3 | 等待完成 | 结束时可选"启动 ADR Studio" |
| 2.4 | 首次启动 | 先出现条款页，同意后进入预热页（不是空白窗口） |
| 2.5 | 预热 | 阶段文案推进：torch 导入 → 权重加载 → kernel 预热；首次下载底模有进度 |
| 2.6 | 进入启动器 | 显示三个控制台卡片；无 VBS/JS 报错 |
| 2.7 | 进入 `pro` 控制台 | 页面加载，`/api/adr/v1/health` 返回 200 |
| 2.8 | 合成一句话 | 产出音频可播放；`server.log` 无 traceback |
| 2.9 | 返回启动器 → 托盘 → 显示窗口 | 三次操作均有反馈 |
| 2.10 | 关窗 | 窗口隐藏到托盘，进程仍在（不是退出） |
| 2.11 | 托盘退出 | `adr-desktop.exe` 与 python 子进程全部结束；9881 端口释放 |
| 2.12 | 安装后文件 | `runtime/BUILD_VARIANT.txt = cpu`；`runtime/redist/vc_redist.x64.exe` 存在；`runtime/bin/ffmpeg.exe` 存在 |
| 2.13 | 注册信息 | 控制面板可见 "ADR Studio"；`HKCU` 自启键按用户设置生效 |

---

## 3. 全新安装（GPU 包）

在 2.1–2.13 基础上追加：

| # | 步骤 | 预期 |
| --- | --- | --- |
| 3.1 | 起始状态 | 同 2 节（未装过 CPU 包） |
| 3.2 | 安装后标记 | `runtime/BUILD_VARIANT.txt = gpu` |
| 3.3 | 引擎设备 | 控制台显示 CUDA 设备；日志出现 CUDA 初始化成功，无"回退 CPU" |
| 3.4 | 合成性能 | 首包时间显著低于 CPU 包（记录实测值） |
| 3.5 | 无 GPU 机器 | 若测试机无 NVIDIA 卡，应回落 CPU 并给出可读提示，不得崩溃 |

---

## 4. CPU ↔ GPU 换包（原地升级语义）

| # | 起始 → 目标 | 预期 |
| --- | --- | --- |
| 4.1 | CPU → GPU | 安装器检测到变体不同，删除并重装 `runtime\python`；用户数据（档案、权重、缓存）保留 |
| 4.2 | GPU → CPU | 同上；换包后 `BUILD_VARIANT.txt = cpu` |
| 4.3 | 换包后首次启动 | 无"缺少 DLL / 找不到模块"；引擎设备与新包型匹配 |
| 4.4 | 控制面板项 | 仍只有一个 "ADR Studio"（共用 AppId，不产生双份） |
| 4.5 | 用户数据核对 | 档案数量、默认档案、TTS 缓存命中均不变 |

---

## 5. 版本升级（同包型）

| # | 步骤 | 预期 |
| --- | --- | --- |
| 5.1 | 已装 1.3.0，安装 1.3.1 | 安装器提示已安装版本，走升级而非并行安装 |
| 5.2 | 应用运行中执行升级 | 由 `AppMutex=ADR-Studio-Running` 拦截，提示先关闭程序 |
| 5.3 | 升级完成 | 版本号在启动器 about 弹窗、安装包文件名、`pyproject` 三处一致 |
| 5.4 | 用户数据 | `shell.json` 保留；档案与权重历史完整；`weights_history` 未丢版本 |
| 5.5 | 缓存兼容 | 旧版本缓存文件（`v<旧版本>-*`）被启动扫描丢弃，不返回旧格式音频 |
| 5.6 | 回退能力 | 升级前可回退（见第 6 节），回退后仍能合成 |

---

## 6. 回滚

| # | 场景 | 步骤 | 预期 |
| --- | --- | --- | --- |
| 6.1 | 版本回滚 | 卸载当前版本（保留数据）→ 安装旧版 | 启动成功；档案与默认设置保留；无 schema 报错 |
| 6.2 | 权重回滚 | 控制台 `GET /profiles/{name}/history` | 列出历史版本 |
| 6.3 | 权重回滚执行 | `POST /profiles/{name}/rollback` | 权重回到上一版；`meta.json` 的 `weights_history` 同步前移；再次合成音色变化可听 |
| 6.4 | 回滚幂等 | 连续回滚两次 | 每次退一版，不在相邻版本间振荡 |
| 6.5 | 坏档案回滚 | 手工破坏 `meta.json` 后启动 | 该档案在列表标注异常但不影响其他档案合成 |

---

## 7. 卸载

| # | 场景 | 预期 |
| --- | --- | --- |
| 7.1 | 卸载时选择"保留用户数据" | 程序目录删除；`%LOCALAPPDATA%\ADR\shell.json` 与数据目录保留 |
| 7.2 | 卸载时选择"删除用户数据" | 档案、权重、缓存、日志全部删除；`runtime\data` 等空壳目录被 `dirifempty` 收尾 |
| 7.3 | 卸载后重装 | 7.1 场景下档案自动可见；不存在残留注册表导致启动异常 |
| 7.4 | 卸载时应用在运行 | `AppMutex` 拦截，提示先退出 |
| 7.5 | 残留检查 | `%LOCALAPPDATA%\Programs\ADR Studio` 不存在；无 `ADR Studio` 自启键；无孤儿 python 进程 |

---

## 8. 故障注入

| # | 注入 | 预期 |
| --- | --- | --- |
| 8.1 | 拔网线后首启 | 底模下载失败给出明确网络错误与重试入口；无静默卡死 |
| 8.2 | 9881 被占用 | 回退随机端口并在启动器提示"NEKO 侧需手动填写该端口" |
| 8.3 | 数据目录不可写 | 启动失败并指出路径；不产生半截文件 |
| 8.4 | 磁盘剩余 < 6GB | 安装前或首启时提示空间不足 |
| 8.5 | 缺 VC++ 运行库 | 安装器静默安装 `vc_redist.x64.exe` 后服务可启动 |
| 8.6 | 缺 WebView2 | 安装器检测并提示（Windows 10 旧版） |
| 8.7 | 强杀 `adr-desktop.exe` | 再次启动时孤儿 python 被收编，9881 可重新绑定 |
| 8.8 | 杀软拦截 `runtime\python` 写入 | 预热给出可读错误，而不是无限转圈 |
| 8.9 | 13/14 代 Intel 机器 | 兼容模式自动开启（AVX2 限制），无 native 崩溃 |
| 8.10 | 显存不足 | 引擎报 OOM 并提示调低强度档位或切 CPU，服务不退出 |

---

## 9. 数据完整性核对

| # | 项 | 位置 | 判据 |
| --- | --- | --- | --- |
| 9.1 | 档案完整 | `<data>/voices/<name>/` | `ref.wav` + `meta.json` 存在且可解析 |
| 9.2 | 权重版本链 | `meta.json` `weights_history` | 每项含路径与时间戳；最新版与当前权重一致 |
| 9.3 | 缓存上限 | `<data>/tts_cache` | 条数 ≤ 32、总量 ≤ 256MB（默认配置下） |
| 9.4 | 日志可读 | `<data>/logs/server.log` | UTF-8 中文不乱码；10MB 轮转生效 |
| 9.5 | 无数据混入安装包 | `runtime/` 下 | 不存在 `data/`、`output/`、`logs/`、根级 `.pth`、`pretrained_models` |
| 9.6 | 设置留存 | `%LOCALAPPDATA%\ADR\shell.json` | 升级/回滚后字段与值不变 |

---

## 10. 发布判据

以下全部满足才可发布：

1. 第 1 节 8 项全过。
2. 第 2 节 13 项全过（CPU 包）/ 第 3 节追加 4 项全过（GPU 包）。
3. 第 4、5、6、7 节无"失败"项；"跳过"项须写明原因。
4. 第 8 节至少覆盖 8.1、8.2、8.3、8.5、8.7 五项。
5. 第 9 节 6 项全过。
6. 记录模板填写完整，含包 SHA256 与日志证据路径。
