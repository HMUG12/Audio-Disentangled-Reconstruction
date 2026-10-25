# M10 计划书（草案）— 体验、低配加速与档案能力

> 状态：**草案 v1**（2026-10-07 与用户商定大部分方向；批次顺序未定稿，见文末"待定"）
> 前置：批次 46 已发布 v1.0.0（GitHub Release + 123 云盘安装包）；批次 45 Review 定位的"非本机用不了"卡点由本计划设置中心收编。

## 已确定方向

### A. 桌面壳全面体检（候选批次 47）

- [x] **静态审计**（批次 59）：前端（launcher.html / pro / easy / call 控制台页）全部 `invoke()` / onclick 处理器对照 `lib.rs` 命令注册表，找出未注册/未绑定/无反馈的入口——15 命令全注册、无真死按钮，实际问题 7 项（反馈缺口与跨平台断链）
- [x] **动态点检**（批次 59）：控制台四页（/, /pro, /easy, /call）浏览器自动化点检零 console error；壳内页面经静态服务加载验证（零语法错误 + Tauri 缺失守卫完好）；真壳逐按钮桌面自动化留白（发布前手测覆盖）
- [x] **修复**（批次 59）：死按钮审计确认无（历史踩坑已在前批修复）；本轮修 back 跨平台断链 ×3 + 空路径/空 catch 静默 ×3（详见批次 59）
- [x] **交互反馈统一**（批次 59）：7 项反馈缺口全修——back 兜底、打开目录空路径 toast、prewarm skip 失败提示、剪贴板失败 toast ×2、pExport 完成/失败反馈 + 生物特征确认

### B. 设置中心 v1（批次 48，已交付主体）

- **技术路线（已定）**：壳自带配置文件管理，启动 sidecar 时以**进程环境变量注入**——不污染系统环境变量，CLI 源码用户不受影响
- **落点（已定）**：`%LOCALAPPDATA%\ADR\shell.json`（全字段 Option + 宽松解析，坏文件不炸壳）；注入统一收敛在 start_server()（prewarm 与 launch_console 自动共用）
- [x] **常用层**：
  - [x] 局域网开关 + API Key（`lan_expose`/`api_key` → 注入 `ADR_TTS_API_KEY`；LAN IP 与引擎地址展示 + 复制完整 URL，还批次 45 Review 的债；三卡片 expose 勾选框联动默认值）
  - [x] 声音档案存放位置（`data_dir` → 注入 `ADR_DATA_DIR`，壳 adr_data_dir() 同源收编；迁移引导 = 变更提示 + 打开目录，**自动迁移留白**）
  - [x] 模型下载源切换（`hf_mirror` → 注入 `HF_ENDPOINT`=hf-mirror.com，4.35GB 底模国内首启下载）
  - [ ] 下载体验项（并发 / 限速）——后端无注入面，留白（批次 49/50 顺手）
  - [x] 开机自启（HKCU Run 注册表，reg.exe 零新依赖，删除键不存在幂等）
  - [x] 默认声音档案选择（`default_profile` → 注入 `ADR_TTS_DEFAULT_PROFILE`；引擎就绪时拉 GET /profiles 填 datalist）
  - [x] 打开日志目录 / 更新检查与版本信息入口（open_path 支持目录与 URL 直达 Releases）
- [x] **高级抽屉**：服务端口（`service_port` 覆盖 9881 探测）、NUMBA 缓存目录（`NUMBA_CACHE_DIR`，与批次 47 沙箱根因闭环）、CUDAGraph 开关（`ADR_T2S_CUDAGRAPH`）；**留白**：日志级别、训练 workers（`ADR_S2_NUM_WORKERS`）、推理设备切换——后端均无读取面，批次 49/50 顺手建；低配模式预设联动批次 49 C 节

### C. 低配设备推理加速（批次 49，v1 已交付 + 格局修正）

- **边界（实测结论，不碰）**：单流延迟已近框架层极限（t2s AR ~22-26 it/s 物理地板，首包 0.92s / 2.3s 音频 2.5s）；CUDAGraph 3.77x 但音质崩（上游 SDPA 语义不等价），仅跟进上游
- **⚠ 格局修正（批次 49 profiling 实锤）**：ADR 底模 v2 系的声码在 **VITS decoder 内部**，不走 BigVGAN 外挂（仅 v3/v4 系挂 BigVGAN）→ "BigVGAN→ORT 预期最大收益"不适用当前 v2 路径，降级留白（v3/v4 用户另议）
- [x] **CPU 全链路 profiling**（scripts/bench_cpu_e2e.py 新建：端到端 RTF 计时 + torch.profiler 算子归因）：本机 RTX 台式机实测 **GPU 档 RTF 0.92-1.02 / CPU 档 RTF 1.31-1.43**（20 线程，首包 GPU 1.1s vs CPU 1.5s）——CPU 档远好于预期；归因：**aten::addmm 线性层 35%**（t2s AR 主体 + VITS FFN）+ **VITS decoder mkldnn_conv 22%**（v2 声码真身）+ AR 拼接 cat 7%
- [x] **设备/线程读取面**：`ADR_TTS_DEVICE`（auto/cuda/cpu，仅 config=="auto" 时生效）+ `ADR_TORCH_THREADS`（>0 时 _lazy_init 早期 set_num_threads）——settings.py 单源 + 单测 2 项
- [x] **低配模式预设落设置中心**：高级抽屉"推理设备"下拉 + "CPU 线程上限"，壳注入 ADR_TTS_DEVICE/ADR_TORCH_THREADS（auto/0 不注入，零副作用）；极速模式音质取舍原则保留（默认 auto，用户自选）
- [ ] **t2s int8 CPU 实验**：CPU 格局实测后仍值得试（addmm 小 GEMM 占 35%），但音质门把关，留白后续批次
- [ ] BigVGAN→ORT：留白（见上格局修正；v3/v4 用户可行）

### D. 音色档案导入导出（候选批次 50）

- [x] 导出 = zip 容器（批次 47）：档案元数据 + 参考音频 + 绑定的 t2s/vits/rvc 权重对（风格字段随 meta.json 可移植副本走）+ manifest 清单带版本号与 sha256 完整性
- [x] 导入 = `validate_weights_file` 校验（批次 43 资产复用，批次 47 两段式）+ 重名冲突处理（同 hash 复用 / 覆盖二选一确认）+ 防 zip-slip + 失败回滚
- [x] **导出提示**（批次 59）：权重即声音本体，含生物特征性质数据，导出前 confirm 明确告知（pro.confirm_export_bio，中英双语）
- [x] 解决"声音档案不在安装包内、换机即失"的债（m9 遗留）：.adrvoice 包导出/导入全链路 + pytest 7/7

### E. 续训流程（批次 50 已交付 2026-10-10）

- [x] 底层已通：增量训练 + enc_q 预训练合并（相似度 0.118 事故的修复资产）复用
- [x] 档案版本化：bind_weights 写 weights_history 版本链 + `/profiles/{name}/history` / `/rollback` 端点，redo 逐版回退（续训翻车能退回，且不会在相邻两版间振荡）
- [x] easy/pro 控制台续训入口：easy "改进已有声音" 勾选 / pro 绑定下拉 "↻ 续训并绑回" —— 选已有档案 → 追加音频 → 处理 → 续训 → 质量门比对新旧相似度
- [x] 质量门防退化：训练收尾 finalize_bind_sim 用门禁最优 sim 对比旧版（Δ<-0.05 置退化标记），pro 训练完成弹警告 + 档案选择器 ⚠ 标注 + 一键回滚

### F. 多语言（已定：两者都要；批次位置待定）

- [x] **合成多语言**（批次 53 已交付 2026-10-17）：GSV v2 底模支持中英日粤混说——语言参数暴露（合成 API + 控制台）+ 音色语言标记 + 多语言测试矩阵。落地：① 档案 language 标记全链路（save_voice/save_voice_auto/meta.json/profiles/create API + easy 建档语种选择）；② 语言默认联动档案语种（native /tts 省略 text_lang/prompt_lang → 档案 language，_resolve_profile 收口；v3 双工 init 补 prompt_lang 透传 + _push_task 联动；v3 /voices 暴露 language）；③ easy 合成面板语言下拉（auto 默认/中/英/日/韩/粤）；④ runtime 补 G2P 包（pyopenjtalk/jamo/ToJyutping/ko_pron 入 requirements + check_runtime 第 7 项体检）。**实测矩阵（yui 档案）**：zh/auto(中英混)/en/ja/yue 全通过，产物 output/multilang_test/；**ko 已知限制**：GSV ko 链依赖 g2pk2+eunjeon(mecab-ko)，Windows cp313 不可用——v1.1 宣传口径"中/英/日/粤 + 中英混说"，韩语待 v1.2 容器版（Linux 可装 eunjeon）。微调权重纯中文数据的日粤口音/相似度待用户听感定宣传口径。**用户听感反馈（2026-10-17）**：日/粤可接受；中英混说模式清晰度/流畅度略欠（底模特性，单语种不受影响）——已记入 release notes 已知限制，后续 v1.3 推理加速/引擎更新再看是否缓解
- [x] **UI 界面多语言**（批次 54 已交付 2026-10-17）：控制台四页（index/easy/pro/call）中英文切换。落地：① 运行时 `i18n.js`（I.init/t/apply/setLang，data-i18n 五种属性 + {var} 占位 + localStorage("adr_lang") 持久化 + navigator.language 自动判定 + document.title/html lang 自动跟随）；② 各页独立字典 `i18n-{page}.js`（easy 79 keys / pro 169 keys / call 95 keys / index 9 keys，合计 352 keys，`[中文, English]` 双串格式）；③ 动态串函数化（STAGE_TXT/TRAIN_STATE_TXT 惰性求值）+ 尾部初始渲染挪到 I.init 之后（I.t 在 init 前返回 key 本身）；④ console.py `/i18n.js` + `/i18n-{page}.js` 白名单路由；⑤ 切换按钮自动注入 header 尾部，setLang 触发 onSwitch 回调重拉动态数据。**浏览器实测**：四页双向切换全过（EN↔中文、localStorage 持久化、动态文案 pollEngine/代码片段联动），console 零报错

## H. 下阶段路线图（2026-10-11 规划）

**版本梯子**：v1.0.1（双包发布）→ v1.1（批次 53/54 多语言+i18n）→ v1.2（多形态分发）→ v1.3（推理加速）→ v1.4（训练加速）；B 分发生态/HF Space demo 零散穿插。

### H1. v1.2 多形态分发（用户拍板优先）
- **容器版（Docker，优先）**：标准服务器版——无 UI、CLI/HTTP API 操作；Linux CUDA/CPU torch + adr server + compose；GSV 上游有官方镜像先例（amd64/arm64）。**注（2026-10-17 review）**：现存 `docker/` 目录为项目早期遗留草稿（torch 2.1/py3.10/Gradio 7860 `adr webui` 入口均与现状脱节），v1.2 落地时**重写而非沿用**。**落地（批次 56）**：Dockerfile/compose/README/构建脚本全量重写，GSV 钉版 clone + 24 补丁自动应用，双变体镜像 tag（cpu/cuda），ko 链 Linux 补齐；构建冒烟已实测通过（见批次 56 记录），ghcr.io 镜像发布入批次 60
- **Windows 无 UI 服务版**：同一 runtime 跳过 Tauri 壳，`python -m adr.server` 服务化（批次 60 落地：server.bat + README 三形态对照）
- **Windows 桌面版**：现状即此形态
- 与 v1.3 协同：服务器多用户场景正是连续批处理（continuous batching）的价值面

### H2. v1.3 推理加速（参考项目落地，2026-10-11 调研实锤）
- **GSV-TTS-Lite**（chinokikiss，MIT，PyPI gsv-tts-lite）：V2/V2Pro/V2ProPlus 专用——RTX 3050 TTFT 436→133ms、RTF 0.381→0.108、**显存 1.6→0.8GB**；核心技术 = CUDA Graph + **Nested/静态 KV Cache（原地写入）** + 连续批处理 + FA2
- **Aqua-TTS**（Lucas1479，PyPI aqua-tts）：**上游内存内 overlay 补丁式**（只 patch infer_panel，生成器/流式契约保留）——静态 KV（scatter_ 原写）+ 分桶 CUDA Graph（6 桶 15 键）+ FA2→SDPA→动态三级降级 + **v2/v2Pro/v2ProPlus 可载**；RTX 4070TiS T2S 145→568-645 it/s，TTFP 417→233ms
- **对项目意义**：批次 10 CUDAGraphRunner 音质崩的根因（动态 KV + SDPA 语义不等价）被这两个项目实锤解决——静态 KV 原地写 + 分桶 + 降级路径即正解；Aqua 的 overlay 模式与我们 third_party 钉版 + 补丁架构同构，优先借鉴
- 落地门：音质门把关（同种子位等价或 SECS/MOS 无回退）；流式 cut3/节奏过滤/热切换语义需重验
- 顺带：**v2ProPlus 推理档评估**（官方口径速度≈v2 而相似度超 v4——推理成本不变抬质量上限，训练侧 v2 仍是 4GB 冠军）
- 长线参考：gpt-sovits-onnx-rs / Genie（全 ONNX Rust 跨平台端侧，含安卓）——容器瘦身/端侧另议

### H3. v1.4 训练加速（参考项目评估，不承诺）
- **AMP bf16**（梯度检查点已备，patch #16）：s2 训练 autocast 实验，预期显存 -30~40%，低成本先试
- **Unsloth QLoRA**（2-5x/显存 -30~90%，宣称支持 TTS 微调）：GSV s2 是 VITS+peft LoRA，Unsloth 内核面向 LLM Transformer——兼容性评估批次，达标才转正
- **bitsandbytes-CPU**（纯 CPU LoRA，AVX2/NEON，12-16GB RAM）：差异化卖点"无 GPU 也能训"，但墙钟时间分钟→小时级，预期管理后评估
- D 训练诊断面板与本线合并考虑

### 学习规划（随版本走）
- 近期（批次 53/54）：GSV text_lang 分支与混说边界；vanilla 前端 i18n 方案
- v1.2 前：Docker 多阶段构建 + Linux CUDA 运行时 + compose；Linux 侧 ctranslate2/faiss 差异
- v1.3 前：精读 GSV-TTS-Lite / Aqua-TTS 源码（DeepWiki 已索引）；Windows FA2 可得性（triton-windows 路线）；静态 KV + 分桶设计
- v1.4 前：AMP/bf16 训练机制、QLoRA(NF4) 原理、bitsandbytes CPU 后端

## 待定（继续商量）

1. ~~批次顺序~~ **已定（2026-10-07）**：批次 47 = A 壳体检全扫 + D 音色导入导出 + 性能测量埋点（GPU 利用率曲线提前备料）；后续 48 = B 设置中心 → 49 = C 低配加速 → 50 = E 续训 → F 多语言插入位置待定
2. 多语言 F 拆分（合成 / UI 是否拆两批）
3. ~~设置中心配置文件格式与位置~~ **已定（批次 48）**：JSON、`%LOCALAPPDATA%\ADR\shell.json`
4. 导入导出容器细节：版本兼容策略、是否需要（可选）导出加密
5. 续训的追加录音 UX（复用现有录音/上传通道的方式）
6. 性能测量是否提前：GPU 利用率曲线（训练全程）可作为 A 批次顺带产出，为 C 批次备料

## G. 分发架构（2026-10-10 拍板）

- **实测（v1.0.0 安装包 runtime 3.89GB 构成）**：2.06GB = third_party/gpt_sovits 根级实验临时权重副本（时间戳 .pth ×6，3×585MB 同文件三份，无任何代码/meta 引用）被全量拷贝带入；1.75GB = python + site-packages（torch 2.10.0 **CPU 版** 443MB）；36MB = gpt_sovits 代码 + 文本资源；底模/预训练**本就不在包内**（首启 bootstrap 走 HF）
- [x] **止血（已做）**：删除垃圾 .pth ×12（源 6 + build 6）→ **runtime 3.89→1.78GB**，安装包预计腰斩；`scripts/check_runtime.py` 固化排除规则（根级裸 .pth / 个人数据面 / pretrained_models / nvidia_* 库 / 总量告警），打包前必跑
- [x] **双包拍板**：CPU 包（现状）+ GPU 包（--allow-cuda 构建，CUDA torch +2.5GB）两条安装包线，为未来多平台包铺路；安装态当前为纯 CPU 推理（GPU RTF 0.92 仅开发环境可达）是双包动机
- [x] **双包构建线落地（批次 52）**：`build_runtime.ps1 -AllowCuda`（torch/torchaudio 2.10.0+cu130 自 download.pytorch.org 先行安装，其余钉版依赖后装防解析器拉回 CPU 轮子；冒烟断言 `torch.cuda.is_available()` 防静默回退）；`build_installer.ps1 -Gpu`（打包前自动跑 check_runtime.py 按 --allow-cuda 放行 + ISCC /DGPU_BUILD）；installer.iss 文件名 `ADR-Studio-<ver>-x64-gpu-setup.exe`，双包共用 AppId（GPU 装上 CPU 包 = 原地升级）；顺手修 build_installer.ps1 里 installer.iss 挪入子目录后的失效路径
- [ ] **缓行**：在线依赖档（安装包瘦到 ~200MB + 首启 pip 走国内镜像）——待双包腰斩结果评估是否值得引入首启等待与网络失败面
- 冷备份（防第三方镜像失活的全量冷备 ≈8GB）：**用户自理**（123 云盘现状 + 自选对象存储）
- [x] **公信力批次（批次 51 落地，2026-10-10）**：REPRODUCE.md（7 节复现链全抽实测命令）/ VERIFY.md（14 条声明→验证→预期映射表）/ examples/samples README（方案 A：不入库音频，生成命令 + 云盘占位）/ .github/workflows/smoke.yml（ubuntu 全量 pytest，CPU torch 轮子先行，tests 零依赖 third_party 已甄别）/ scripts/eval_secs.py SECS 口径（实测锚点 0.9324 / 零样本 0.7729 / 微调 0.8052，Δ+0.0323）；**PESQ/ESTOI → SECS/MOS/CER 口径修正完成**（详见 REPRODUCE.md 第 5 节）

## 批次 55（2026-10-17）：全量 review — 基础修复

> v1.1 发布后全仓 review（git/版本/依赖/CI/测试/文档），发现并修复：

- [x] **CI smoke 六连红修复**（批次 51 上线以来 6 次 run 全 failure，7 failed/425 passed）：① 6 个测试依赖 `third_party/DiffSinger/opencpop-extension.txt`（607 音素词典不入检出树）→ conftest 加 `requires_default_dict` 守卫 fixture，干净检出自动 skip（test_phoneme_dict ×3 + test_encode_phonemes_chinese + test_g2p_chinese_nonempty + test_dataset_from_npz_dir）；② `test_data_dir_windows_legacy_f` 跨平台 bug——Linux 上 `PosixPath("F:/ADR_data")` 与 `PureWindowsPath` 永不相等，钉 os 模拟失效 → 改 str 归一比较（`replace("\\","/")`）
- [x] **版本面补齐 1.1.0**：pyproject.toml（1.0.1→1.1.0）+ `adr.__version__`（1.0.0→1.1.0）——v1.1.0 发布时只 bump 了 installer.iss/tauri.conf/Cargo.toml/package.json 五面中的四面；顺带修正 `__license__`（"MIT"→AGPL-3.0，与 pyproject/LICENSE/README 对齐）
- [x] **源码用户 G2P 缺口**：批次 53 四包只进 build_runtime.ps1，pyproject/requirements 均缺 → 源码用户无法日/粤/韩合成 → pyproject 新增 `[multilang]` extra（pyopenjtalk/jamo/ToJyutping/ko_pron 钉版同 runtime）+ README 源码安装注记（不并入 `[all]`，防 CI/全量安装被 pyopenjtalk 源码构建拖挂）
- [x] README 静默安装示例文件名 1.0.1→1.1.0；REPRODUCE.md 补 v1.1.0 安装包实测体积
- 备注：docker/ 遗留草稿注记已入 H1；本地全量 pytest 回归 + 推送后盯 CI 转绿

## 批次 56（2026-10-17）：v1.2 容器版 — Docker 分发落地

> H1 容器版交付物全量重写（旧 docker/ 为早期草稿，torch 2.1/py3.10/Gradio 形态已脱节）：

- [x] **docker/Dockerfile 重写（两阶段）**：python:3.11-slim；`TORCH_VARIANT=cpu|cuda` 构建参数（torch/torchaudio 2.10.0 分别从 cpu / cu130 源安装，与 Windows 双包同款）；venv + `pip install -e .` 保持源码布局（引擎逻辑锚定 REPO_ROOT/third_party，容器内 /app 同层）；GSV **钉版 clone**（见下）+ `apply_compat_patches.py` 自动打补丁；runtime 层 ENV：`ADR_DATA_DIR=/data`、`ADR_TTS_CACHE_DIR=/data/tts_cache`、`HF_HOME=/data/hf-cache`；HEALTHCHECK 探 `/api/adr/v1/health`；入口 `python -m adr.server --addr 0.0.0.0 --port 9881`（纯服务形态，无 WebUI）
- [x] **docker/requirements-linux.txt**：蓝本 = build_runtime.ps1 钉版清单；去 gradio/gradio_client、audioop-lts/standard-aifc（Windows 专用）、torch（按变体单独装）；**补 ko 链 Linux 包**：`g2pk2==0.0.3`（纯 py）+ `python-mecab-ko==1.3.7`（cp311 manylinux 轮子；GSV 官方 requirements 同款 `python_mecab_ko; sys_platform != 'win32'`，词典包自动随拉）——Windows 沿用 eunjeon 不变，Linux 侧韩语链补齐
- [x] **GSV 钉版可复现链**：commit `d523079fc05d9a8028d6085bffe4a2757c32abb6`（无 submodule，干净 clone 即可）；`GSV_PROXY_PREFIX` 构建参数支持 ghproxy 类加速；镜像 tag `adr-server:1.2.0-cpu` / `adr-server:1.2.0-cuda`（pyproject 暂 1.1.0，v1.2 发布时包版本对齐）
- [x] **Track E 补丁收编**：批次 33 GPU→CPU 同步补丁从脚本内嵌字符串迁到 `scripts/patches/track_e_ar_sync.patch`（PATCH_FILES 机制：git apply --whitespace=nowarn + 幂等特征串检测）；修 patch 文件编码（去 UTF-8 BOM + CRLF→LF——PowerShell 重定向产物 git apply 直接失败）；本地已打补丁树上 `git apply --check --reverse` 通过 + 脚本幂等验证（应用 0 / 跳过 24 / 失败 0）
- [x] **compose 双文件**：`docker-compose.yml`（CPU 默认：`./data:/data` bind mount + pretrained_models/G2PWModel 两个 named volume 挂权重自引导目录，不遮蔽 GSV 代码；shm 2g）+ `docker-compose.gpu.yml`（GPU override：cuda 变体 + nvidia 全卡透传 + ADR_TTS_DEVICE=auto）；仓库根 `.dockerignore`（context=.. 决定必须在根：裁 third_party/tests/docs/data，*.md 豁免 README.md）；.gitignore 追加 `docker/data/`
- [x] **docker/README.md 重写 + scripts/build_docker.ps1**：双变体用法/数据卷/端口与 ADR_TTS_API_KEY 鉴权（0.0.0.0 无鉴权警告为容器预期行为）/代理与 GSV 加速/FAQ；构建脚本 `-Cuda`/`-Tag`/`-ProxyPrefix` 封装 docker build
- [x] **容器构建验证 + 冒烟（10-18 实测）**：`adr-server:1.2.0-cpu` 构建成功——两阶段全链走通（apt / torch 2.10.0 cpu 源 / requirements-linux / GSV 钉版 clone d523079f + 补丁应用 21/跳 3/败 0 / venv COPY）。冒烟全过：容器起服务后 `GET /api/adr/v1/health` → 200，响应结构完整（status ok / api_version 1.0 / engine_ready=false + engine_loading=true + engine_stage=downloading「pretrained_models.zip 4351.8 MB 进度」/ weights——engine_ready 置真需容器运行时下载 4.3GB 权重，属运行时自引导非镜像问题）；ko 链 `docker run --rm --entrypoint python … -c "from g2pk2 import G2p"` OK（nltk cmudict 自动下载正常）。环境坑两枚（与镜像无关）：① TRAE 沙箱拦截 Docker Desktop 写自身配置/leveldb → guest init 永久无响应，解法 = 计划任务（Register-ScheduledTask）在沙箱进程树外拉起，引擎 10 秒就绪；② 本机 TUN 代理对容器流量干扰——deb.debian.org:80 间歇超时、清华/中科大 HTTPS SSL unexpected eof、阿里云 HTTP 502，唯 deb.debian.org HTTP 直连窗口期稳定，裸构建重跑即过（buildkit 层缓存使重试代价低）

## 批次 57（2026-10-17）：日志体系 review 修复 + 13/14 代 Intel 闪退加固

> 全量 review 定位四类日志形态（桌面版 server.log / 容器 docker logs / 纯 CLI stderr / 训练内存日志）后修复 + 用户报 13 代 CPU 闪退加固：

- [x] **日志 review 修复**：① 桌面壳 sidecar 补 `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`（Windows pipe 下 Python 默认 GBK，与壳 UTF-8 proc_log 行混排乱码——console.py 起训练子进程时已设、壳漏设）；② `__main__.py` 接 `setup_logging`（此前 root 默认 WARNING，`adr.*` INFO 全被吞，server.log/docker logs 只剩 uvicorn 行；`ADR_LOG_LEVEL` 可覆盖）；③ server.log 超 10MB 启动时轮转 `.old`（单代覆盖）；④ call.html placeholder 反斜杠改正斜杠（中英双语）
- [x] **13/14 代闪退根因**：Raptor Lake AVX-512 高电压硬件缺陷 → torch oneDNN/MKL、numba JIT 计算密集段 native 崩溃（0xC0000005），无 Python 栈、stderr 全空，supervise 有自动重启但退出无痕。缓解 = 全链指令集限到 AVX2（性能略降换稳定）
- [x] **ShellSettings 加 `compat_cpu: Option<bool>`** + env_overrides 注入 `ONEDNN_MAX_CPU_ISA=AVX2` / `MKL_ENABLE_INSTRUCTIONS=AVX2` / `NUMBA_CPU_NAME=generic`（旧 shell.json 缺字段宽松回退，serde 全 Option 模式不变）
- [x] **崩溃留痕**：supervise `try_wait` 检出退出即 proc_log 退出码（native 崩溃场景 server.log 唯一直接证据）；MAX_FAILURES 放弃重启时落盘完整日志路径 + 状态文案指引"设置 → 维护 → 打开日志目录"
- [x] **faulthandler 兜底**：`__main__.py` main() 开头 `faulthandler.enable(file=sys.stderr, all_threads=True)`——native 崩溃瞬间把 Python 侧调用栈写 stderr，随壳泵入 server.log
- [x] **UI**：launcher.html 高级设置加"CPU 兼容模式"勾选框 + load/save 显式带 `compat_cpu`（漏带即保存丢值，ShellSettings serde 教训）
- [x] **CPU 自动兼容（同日追加）**：壳 reg 读 `HKLM\...\CentralProcessor\0` 品牌串（零依赖）——13/14 代 Intel 自动开兼容（`compat_cpu=None` = 自动，Some 手动覆盖）；每次起 sidecar 落日志 `CPU: <品牌>; AVX2=true/false, AVX512=...`，缺 AVX2 醒目警告（官方轮子直接闪退）；`brand_is_raptor_lake` 纯函数单测
- [x] **GPU 诊断日志（同日追加）**：gsv_engine 设备定型后落 `[gpu]` 行——N 卡型号/算力/显存/CUDA runtime；算力 <7.0 警告（本 CUDA 轮子可能不含该架构内核，no kernel image available 即此因）；未走 CUDA 时区分"CUDA 不可用/强制 CPU"与"CPU 版轮子"两种情况
- [x] **兼容性第二批（同日续）— OOM 分流**：gsv_engine 新增 `is_cuda_oom`（不引 torch，按类型名 + 模块前缀判定）与 `handle_cuda_oom`（单点善后：立即 `empty_cache()` 防显存碎片化 + 警告日志 + 用户可读提示）；v2_compat `tts_handle` 非 OOM 之外的异常分流为 503 `{"code":"cuda_oom", "message": 提示}`（此前裸 400 堆栈原文）；流式路径响应头已发只断流 + 落友好日志（native.py tts_get/tts_post 复用 tts_handle 全覆盖）
- [x] **兼容性第二批 — 驱动版本诊断**：gsv_engine `_nvidia_driver_version()`（nvidia-smi `--query-gpu=driver_version` 零依赖）；`[gpu]` 日志行带驱动版本；**"驱动已装但 CUDA 不可用"显式警告**（驱动低于 torch 轮子所需 → 静默回退 CPU 的经典故障）+ 算力 <7.0 老卡警告
- [x] **兼容性第二批 — UI 自动启用提示**：launcher.html `cpu_compat_auto` 命令查询 + 勾选框旁 "（检测到 13/14 代 Intel，已自动启用）"；用户手动设置过则不显示
- [x] 兼容性第二批 — 下载失败引导：gsv_bootstrap 三源全失败错误文案追加网络/代理检查指引（需可达 hf-mirror.com / huggingface.co / modelscope.cn）+ 修复后重启引擎重试
- [x] 验证：cargo test 5/5（env_overrides 单测扩至 12 项断言 + 新增自动语义用例）+ python 导入 faulthandler 断言通过 + 兼容性第二批 python 验证（3 文件 ast + import + is_cuda_oom True/False 判定 + 驱动查询兜底）通过
- [x] **review 修复（双子代理交叉验证 2/2 确认）**：① **launcher 保存固化 compat_cpu（major）**——loadSettings `!!null` 渲染成未勾选 + saveSettings 无条件发 checked，13 代用户保存任意设置即把自动兼容固化成 `Some(false)` 永久压制自动检测（单测语义铁证）；修复 = `_compatTouched` 触碰跟踪（change 监听 + loadSettings 重置 + auto 回调时序防御）+ 保存 `(touched || 此前显式值) ? checked : null` 保住 null 自动语义 + 勾选框回填实际生效状态（自动启用显示已勾选，消除提示与勾选框自相矛盾）；实现时捕获 saveSettings 对象字面量内引用正在构造的 `cfg` 自身（TDZ → 每次保存必 ReferenceError），改模块级 `stCompatLoaded` 记载上次加载真值。② **强制 CPU 驱动警告误报（minor）**——gsv_engine elif 分支只看"驱动已装"就落"请升级驱动"warning，config 显式 cpu（eval_secs/train_gate）与 env 强制 cpu（ADR_TTS_DEVICE）两条主动路径必误报；修复 = `_forced` 标记，主动选择 + 驱动正常落 info"按配置强制 CPU 运行"，警告仅保留 auto 判定回退场景
- [x] review 修复验证：3 文件 ast + gsv_engine import + OOM 判定回归 + 两处修复源码级断言（含 TDZ 残留检查、括号平衡）全绿；cargo 侧无改动

## 批次 58（2026-10-18）：review 遗留五项收尾（低优先延后项清零）

> 批次 57 汇报时列出的 5 个低优先延后项，本轮全部落地：

- [x] **v3_compat websocket OOM 分流**：`_produce` 生产者 except 与外层 stream-input 兜底 except 各加 `is_cuda_oom` 分流——OOM 走 `handle_cuda_oom(e)`（empty_cache + 用户可读提示）+ 帧码 `"cuda_oom"`，与 v2 tts_handle 同模式；非 OOM 保持原 `error_code(e)` 语义，错误帧 code 增量字段 wire 兼容
- [x] **桌面壳 pump 双线程行交错**：`pump_server_log` 由"每线程独立开 append 句柄 + `io::copy` 8KB 块拷贝"改为"调用点单次开句柄 → `Arc<Mutex<File>>` 共享 + `read_until(b'\n')` 整行读 + 每行锁内 write"——块边界劈行与双线程字节交错根除；头行（stream open）由调用方单线程写一次，EOF 各写带 tag（stdout/stderr）的 stream closed 行。整流持锁会死锁（stdout 线程独占句柄时 stderr 永久阻塞），故必须逐行取锁
- [x] **exp 名清洗 Windows 保留名**：新建 `adr/core/naming.py`（`sanitize_exp_name`：控制字符 `unicodedata.category=="Cc"` + `\/:*?"<>|` 删除 + strip + rstrip(". ") + 保留名加 `_` 前缀 + 空回退 my_voice；`is_windows_reserved`：CON/PRN/AUX/NUL/COM1-9/LPT1-9 大小写不敏感含带扩展名形式）；console.py train/start 与 webui `_run_gsv_finetune` 两处旧两行清洗替换调用；voice_library 与 pathsafe 两处 `is_safe_name` 追加保留名拒绝（落盘点 data/voices 与档案名目录需可 mkdir）
- [x] **CLI 文件日志**：`adr/cli.py` main group 统一 `setup_logging(level=ADR_LOG_LEVEL env, log_file=<log_dir>/cli.log)`（log_dir 取 config，配置不可用回退 ./logs）；删除 12 处命令内 `setup_logging(level="INFO")` 重复调用及 import 中的 setup_logging（其 root.handlers.clear() 会重置 group 已挂的 file handler——幂等陷阱）
- [x] 验证：7 文件 py_compile + naming 14 组断言 + is_safe_name 双实现保留名断言 + v3_compat import 与 cuda_oom 分支源码断言 + CLI 冒烟（`python -m adr.cli info` → logs/cli.log 生成含设备日志）+ cargo check 0 警告 + cargo test 5/5 + pytest security/voice_library/server_api 104 passed（pytest 系统临时目录被沙箱拒访，`--basetemp=.pytest_tmp` 绕过，跑后清理）
- [x] **review 修复（双子代理交叉验证 2/2 确认）**：① **webui `sanitize_exp_name` NameError（major）**——import 误落在不使用它的 `_run_train_cmd`（SearchReplace 匹配到两函数雷同 import 块的第一处），使用点 `_run_gsv_finetune` 局部 import 块无该名 → 点击「开始克隆微调」/ REST clone_train 生成器首次迭代即 NameError；修复 = import 移至 `_run_gsv_finetune` 局部 import 块（与 json/subprocess/sys/time 并列）。② **CLI main group setup_logging 无兜底（minor）**——log_dir 指向不可写路径时 makedirs/FileHandler 抛 OSError，click group 回调先于全部子命令执行 → 所有命令（含 doctor）启动即崩；修复 = try/except OSError 降级为仅 stderr 的 setup_logging + click.echo 一行降级警告
- [x] review 修复验证：py_compile + 冒烟（`python -m adr.cli info` exit 0）+ 降级场景 CliRunner 实测（log_file dirname 为普通文件 → FileExistsError → exit 0 + warning 输出）全绿
- [x] **跨平台专项 review（Python 侧 + Rust 侧双子代理）**：Python 10 项（编码/路径/行尾/大小写/保留名/subprocess/asyncio/信号/原子操作/控制台 Unicode）全绿；Rust 8 项（CRLF/编码/mutex 中毒/写失败/路径/cfg 保护/进程终止/线程生命周期）主平台全绿；无 major/minor，info 级发现见当次汇报
- [x] **跨平台修复（review 后追加，用户拍板）**：webui 三处 `Popen`（`_run_train_cmd` 主训练 / `_run_gsv_finetune` 门禁 + 主微调）补注入 `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`（对齐 console.py 同款做法），根除 zh-CN Windows + Python≤3.14 下子进程 GBK 输出经 utf-8/replace 解码乱码 → 早停标记「收到训练门禁早停信号」匹配失败、early_stop 指标静默丢失；两函数局部 import 块补 `import os`。验证：py_compile + 源码断言（2 处 import os + 3 处 PYTHONUTF8）全绿

## 批次 59（2026-10-18）：M10 A 节 UI 质量批次收尾 + D 节音色档案导入导出闭环

> 用户指令"把欠债类修完，然后再将功能做完"——A 节四项全勾，D 节确认批次 47 核心已交付并补齐最后一项 UX 债：

- [x] **静态审计（A-1）**：子代理全量对照——lib.rs 15 命令全注册、控制台四页 fetch 路径全存在、onclick 函数全存在、id 全匹配；结论无真死按钮（历史踩坑 on_download / ACL / fallback 导航均已在前批修复），实际问题 7 项（反馈缺口与跨平台断链）
- [x] **修复（A-3/A-4）**：① **back 跨平台断链 ×3**——easy/pro/call 硬编码 `http://tauri.localhost` 仅 Windows 壳有效（lib.rs TAURI_ORIGIN 非 Windows 为 `tauri://localhost`），改为按 UA 构造壳 origin + 无 Tauri 注入（浏览器直开）兜底回 `/`；② launcher st-open-data/log 空路径静默短路 → toast"目录信息暂不可用"；③ prewarm btnSkip 空 catch → setTip"强制进入失败"；④ launcher 剪贴板 ×2 失败静默 → toast"复制失败, 请手动复制"；⑤ pro pExport 重写（见 D 节条目）
- [x] **动态点检（A-2）**：本地 `python -m adr.server`（9881）+ 浏览器自动化——控制台四页（/、/pro、/easy、/call）加载零 console error，back 默认隐藏、?desktop=1 可见且浏览器直开兜底 href="/" 符合设计；壳 launcher/prewarm 经临时静态服务加载零 SyntaxError（showToast/setTip 就位，Tauri 缺失守卫完好）；真壳逐按钮桌面自动化留白（需 cargo build 完整 Tauri 应用，发布前手测覆盖）
- [x] **D 节闭环**：导出/导入核心批次 47 已交付（voice_package.py：manifest+meta+ref.wav+weights，FORMAT/VERSION 清单；导入防 zip-slip / 同 hash 复用 / validate_weights_file 两段式 / 失败回滚；native.py export/import 端点；pro.html 导入覆盖确认流）+ pytest test_voice_package 7/7——本轮补齐最后一项：**pExport 导出重写**，fetch→blob→a[download]（壳内由 on_download 接手，与 easy rSave 同模式）+ 600s 超时 + 完成/失败 toast 反馈 + 导出前 confirm 生物特征告知（i18n-pro 3 新键：pro.confirm_export_bio"权重即声音本体/仅提供可信对象"、pro.toast_export_done、pro.toast_export_fail）
- [x] 验证：控制台四页 + 壳两页浏览器点检全绿（零 console error）+ 新 i18n 键浏览器实测（{name} 占位符替换正确）+ pytest voice_package 7/7（0.49s）

## 批次 60（规划定稿 2026-10-18）：v1.2 收尾发布

> 用户拍板：方向 = v1.2 收尾发布；无 UI 服务版 = server.bat + README 三形态；容器镜像 = 推 ghcr.io。真壳逐按钮手测按"后面完成后一起测试"落在本批发布前。

- [x] **Windows 无 UI 服务版（H1 收口）**：安装根目录 `server.bat`——调 `runtime\python\python.exe -m adr.server`（PYTHONUTF8=1/PYTHONIOENCODING=utf-8 对齐壳与 webui 做法）；env 直读（ADR_DATA_DIR 缺省 %LOCALAPPDATA%\ADR\data 对齐桌面版、ADR_TTS_API_KEY、ADR_TTS_DEVICE、ADR_LOG_LEVEL），与容器版同为 env 驱动无 UI 形态；README 新增桌面版/无 UI 版/容器版三形态对照小节
- [x] **容器镜像 ghcr.io 发布**：新增 GitHub Actions workflow（tag v* 触发）构建 cpu/cuda 双变体推 ghcr.io（GITHUB_TOKEN packages:write；buildx 层缓存压构建时长，torch 镜像体积大属预期）；README/release notes 补 `docker pull ghcr.io/<owner>/adr-server:1.2.0-cpu|cuda` 用法；镜像 tag 对齐 1.2.0（批次 56 已预留命名）
- [x] **版本面 1.2.0 对齐（批次 55 教训 checklist 化）**：pyproject.toml / adr `__version__` / tauri.conf.json / Cargo.toml（Cargo.lock 随动）/ package.json（package-lock 随动）/ installer.iss 六面 + launcher.html 版本显示 + README 静默安装示例文件名 + REPRODUCE.md 体积注记；新增 `scripts/check_versions.py` 六面一致性断言并纳入 CI smoke，防再犯
- [x] **发布前真壳逐按钮手测**：cargo build 完整 Tauri 壳 → launcher 全按钮走查（设置面板开合/保存/打开数据与日志目录/复制 URL/开机自启/检查更新/QQ 复制）+ prewarm 正常流与 skip + 壳内控制台四页导航与 back + pExport 真实导出→导入闭环 + 一次合成冒烟；发现问题当批即修（用户拍板 pass，后续真实用户手测反馈迭代）
- [x] **双包构建**：build_installer.ps1 CPU + `-Gpu`（check_runtime.py 前置 + GPU 包冒烟断言 torch.cuda.is_available）——CPU 459.5MB / GPU 1926.4MB，GPU runtime 4.00GB 冒烟 CUDA OK（RTX A2000）
- [x] **Release 收尾**：release notes v1.2.0（容器版 + 双包 + 批次 57 日志体系/13-14 代加固 + 批次 58 review 收尾 + 批次 59 UI 质量与档案导出闭环 + **韩语容器版口径兑现**（Windows 限制不变，Linux 容器可装 ko 链）；已知限制：中英混说清晰度/流畅度）→ tag v1.2.0 → GitHub Release 附双安装包 → 123 云盘——Release 已发布（https://github.com/HMUG12/Audio-Disentangled-Reconstruction/releases/tag/v1.2.0，双资产含 SHA256 见 notes）；123 云盘上传为用户自理环节
- [x] plan 收口：H1 勾选 + 本节回填 [x] + 验证记录（见下）

> **验证记录（2026-10-18 会话内）**：① 防重入挡金标准复测通过——新构建单次点击经典卡恰好 +1 spawn（1 kill + 1 spawn + 1 banner + 探活 200），旧构建同场景 +2，Rust CAS 挡结构性免疫双投递；② 合成冒烟通过——GSV 权重缺失走结构化 error+suggestion 降级，SSE 流正常终结无崩溃；③ 音色库路径分裂 bug 证伪——UI choices 实测 3 项，此前系 /config 正则误读，未改代码；④ ~~PS 5.1 GBK 吞包陷阱~~ **归因修正（10-24 会话）**：三重实验证伪 GBK 归因（PSParser 解析层完整 54 行 / tokenizer 内容 Set-Content 写盘完整 / 但 pip 安装日志行号铁证读到缺 8 行文件）——真机制为**构建进程启动时磁盘脚本为旧版**（两次会话间文件状态异常，疑 IDE/云同步回写缓冲）；check_runtime 第⑦项两次拦截坏包价值实证，6cfb825 英文化注释保留为 ASCII 防线；⑤ 本机构建机代理 env 指向未运行代理，重跑需清 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY；⑥ CPU runtime 1847MB smoke 全绿（torch 2.10.0+cpu/adr/GSV imports/pip check），installer 重跑中，GPU 包因时间窗顺延（缓存已就位）。
>
> **验证记录（2026-10-24 会话内，v1.2.0 发布收口）**：⑦ GPU runtime 4092MB smoke 全绿（python 3.13.16 / torch 2.10.0+cu130 / cuda_available True / RTX A2000 Laptop GPU 识别 / adr+GSV imports / pip check）；补装 G2P 四包后 check_runtime 七项体检通过（4.00GB，GPU 上限告警线 4.6GB）；GPU installer 1926.4MB ISCC 全绿；⑧ 双包 SHA256——CPU `c429beb3348e…f7ad90` / GPU `905965a738f8…bf15feb`（全值入 release notes 下载节）；⑨ **防重入修复补提交**（ea49355）：lib.rs Rust CAS + launcher.html launching 标记在 10-18 已构建进双包（修复 15:20 早于 CPU 包构建 17:07），仅漏 git 提交；⑩ 发布链实操：gh CLI 未安装 → git credential manager 取 token + curl 直调 REST API（创建 Release 201 / 资产 uploads 上传）；PS 5.1 ConvertTo-Json 序列化 release notes 触发 GitHub 422（properties/body 校验拒），改用 runtime python json.dumps 生成请求体绕开；⑪ tag v1.2.0 已推，docker-publish.yml 自动触发 ghcr cpu/cuda 双镜像构建；⑫ 代理注意：push 阶段 44444 代理一度未起（直连被 reset），复探测后进程在位恢复推送——构建/push 前先探活代理。

## 规划重排（2026-10-24 会议纪要定稿）：合规门禁 + v1.3 Autopilot + v2.0 路线

> 依据《会议纪要.txt》（302 行：四版本分层 / 合规六提案 / Autopilot 技术线 / 四版本性能表）对照现状差距盘点后重排。**已定稿（2026-10-24 用户拍板，首启条款流取方案 C，会议做记录不阻塞）**：六决策点全部落定，②批次排期与③门禁表按方案 C 一次定稿。文档以 **draft/WIP** 提交至分支，定稿（会议记录确认）后合并回 main。

### ① 决策记录表（2026-10-24）

| # | 决策点 | 结论 | 状态 |
|---|--------|------|------|
| 1 | 版本号口径 | 1.x 全算测试线（含 v1.3 Autopilot）；稳定版另立 **v2.0**，窗口 2026.10–2027.6 | **已定** |
| 2 | Studio Pro+ 联合 max | 物理 6×96G；联合预算上限 **192G**；对外注明**预算≠物理总和**（Server Pro+ 8×96G 维持已确认口径） | 已定（口径表述待会议复核） |
| 3 | 原生二进制编译 | 评估后缓行，**不作发布门禁**；分发以便携运行时为准 | **已定** |
| 4 | 首启条款流 | **方案 C**：条款流放批次 61（最小条款流），安全模块检测/审计随批次 62（三方案对比见③） | **已定（2026-10-24，方案 C）** |
| 5 | Autopilot 合并 | 认可并入 v1.3，命名 **v1.3 "Autopilot"**；"内核底座 + 调度层"两层关系见④ | **已定** |
| 6 | macOS | 纳入 v2.0 窗口（27.6）**末段**；可滑出、不阻塞主线，规则见⑤ | 已定（可滑出） |

### ② 版本与批次排期（按方案 C 定稿）

**1.x 测试线**：

| 批次 | 版本 | 内容 | 状态 |
|------|------|------|------|
| 61 | v1.2.1 | 合规速赢：纪要"立即执行"四项（README 风险披露 / 首启禁止性条款 / 源码随包 / AGPL-3.0-only）+ SPDX + CLA/DCO 拍板 + 门禁清单 v1（③清单 1/2/3/6）+ **最小条款流**（prewarm 前置条款页 + data 标记 `.terms_agreed_v1` + 拒绝退出态，按 2 天增量排；条款文案中文版先行，英文版后补不阻塞 61） | **已完成（2026-10-24，落地记录见下）** |
| 62 | v1.2.2 | 安全模块 MVP：声音安全三层 + WAV 元数据 + 素材授权台账/声纹单独同意 + 无 UI 形态条款（`--agree-terms` / `ADR_AGREE_TERMS` env，**server 形态下默认写日志便于审计**）+ 声音安全设置 + 审计（③清单 4/5） | **已完成（2026-10-24，落地记录见下）** |
| 63+ | v1.3.0 | **v1.3 "Autopilot"**：内核底座（H2 静态 KV / 分桶 CUDA Graph / 连续批处理）+ 调度层，N=1 硬预算起步（见④） | 首子批（t2s 静态 KV）+ 次子批（分桶 CUDA Graph）+ 第三子批（调度层 N=1 硬预算）+ 第四子批（Profiler 微基准）已完成（2026-10-24，记录见下）；**v1.3.0 tag 已推送（指向 062f475，含 Review 修正），Release 与安装包后补**；连续批处理内核 / 准入控制 / Policy / Runtime 待后续子批 |
| 框架线 | — | 自动更新链：launcher"检查更新"现为直开 Releases 页（launcher.html L564），缺版本比对/下载/安装闭环；建议插 62 后或与 63 并行 | 待排期 |

**稳定里程碑 v2.0（2026.10–2027.6）**：验收 = 合规门禁全绿（③）+ Autopilot 五指标达标（④）+ 平台矩阵（Windows 全形态 / Linux 容器 / macOS 窗口末段可滑出⑤）。

### ③ 合规门禁与首启条款流（61 vs 62 细分析）

**门禁清单（61 起逐项落地，新 check_compliance 固化进 CI）**：

1. LICENSE 精确化：`AGPL-3.0-only`（现状 pyproject.toml L10 为模糊 `AGPL-3.0`，LICENSE 全文无 -only/-or-later 声明）+ SPDX 标识 + 第三方许可清单（GSV MIT / 依赖面）
2. 源码随包提供：release 页源码压缩包 + README/条款内源码获取途径（AGPL §6 对应源码义务；third_party 钉版补丁已在仓库，源码包可重构）
3. README 风险披露：声音克隆滥用风险 + 合规使用边界 + 刑事风险分发隔离声明（release notes 同步）
4. 声音安全三层（批次 62）：上传侧素材来源声明 / 合成侧文本审计 / 分发侧 WAV 元数据——**现状 WAV 输出无元数据写入能力（全仓零命中），需新增**；配套素材授权台账与声纹单独同意（纪要"ADRserver 对外可用之前"项，server 已随 v1.2.0 对外，提前至 62）
5. 首启条款 + 声纹个人信息告知（方案见下）+ 设备标识默认关/可选（纪要 L111，隐私面归 62）
6. 发布门禁：check_versions / check_runtime 之外新增 check_compliance（条款资源存在性 + LICENSE/SPDX/风险披露断言）

- 另：CLA/DCO 贡献者治理路径拍板 + CI 许可证扫描（纪要"1.0 稳定版之前"项）→ 归 61 文档面；算法备案/安评离线分发不适用、在线部署场景标识义务提示写入部署文档（归 62）

**首启条款流三方案**：

| 方案 | 内容 | 判定 |
|------|------|------|
| A 全放 61 | 条款流 + 安全检测/审计/元数据一并 | ✗ 检测能力工程量把 v1.2.1 拖成大版本，违背速赢定位 |
| B 全放 62 | 61 只做 LICENSE/README 文档面 | ✗ 61 过轻（半天量）；入口约束晚一批生效；62 混入 UI 改动后"安全模块 MVP"边界不清 |
| **C 拆分（已拍板）** | **61 = 最小条款流**：prewarm 前置条款页（AGPL 义务通俗化 + 禁止性条款 + 声纹个人信息告知，对应纪要首启配置⑫；文案中文版先行，英文后补不阻塞）+ data 标记 `.terms_agreed_v1`（升级/重装不重复弹）+ 拒绝退出态，~2 天增量；**62 = 安全模块**：三层检测/审计 + WAV 元数据 + 无 UI 形态条款（server.bat `--agree-terms` / `ADR_AGREE_TERMS` env，**server 形态默认写日志便于审计**，容器 compose 同步）+ 声音安全设置（纪要首启配置⑬） | ✓ 61 = 合规基线（UI 壳面），62 = 安全能力（模型无关可测），切分干净、两版都不拖 |

方案 C 依据：① prewarm.html 现状为纯步骤展示页（4 步骤条 + setStep/setTip），条款页作前置视图低成本插入，desktop/src 三页（launcher/prewarm/loading）零条款代码（Grep terms/consent/同意 零命中），从零加但量小；② 纪要"立即执行"清单点名"首次启动禁止性条款"，入口约束是 ADRstudio 义务重心（纪要提案二）；③ 拒绝退出态兜底合规姿态。风险：条款文案口径（AGPL 义务通俗化 + 声纹告知措辞）需会议/法务把关，61 内出初稿过目。

**已拍板（2026-10-24）：按方案 C 定稿**——61 = 门禁 1/2/3/6 + 条款流，发 v1.2.1；62 = 门禁 4/5 + 无 UI 条款 + 声音安全设置，发 v1.2.2。条款文案中文版先行（英文后补不阻塞 61）；`--agree-terms` 在 server 形态下默认写日志便于审计。

**批次 61 落地记录（2026-10-24，分支 plan-v2-roadmap）**：

- **门禁 1（LICENSE 精确化 + 第三方清单）**：pyproject.toml / Cargo.toml / package.json 三构建面 license 字段精确为 `AGPL-3.0-only`；`adr/__init__.py` 顺手补 `__license__ = "AGPL-3.0-only"`；新增 THIRD_PARTY_LICENSES.md（随包四组件 GSV/BigVGAN/python-build-standalone/Tauri + Python 依赖面许可表，注明底模与声音档案非分发物）
- **门禁 2（源码随包）**：README License 节重写——源码获取途径（Releases 源码压缩包 + 克隆仓库，AGPL §6）+ 网络服务同许可开放义务 + 附加条款联系方式
- **门禁 3（风险披露）**：README 新增"风险披露与使用边界"节（授权前提 / 禁止用途 / 责任自负 / 声纹个人信息告知 / 数据采集五条，提及首启条款流）；FAQ 商用答同步 AGPL-3.0-only
- **门禁 6（check_compliance 进 CI）**：新增 scripts/check_compliance.py 六组检查（LICENSE 全文 + 三面 license 字段 / README 源码获取 / README 风险披露 / 第三方清单条目 / 条款流 invoke 与命令存在性 / smoke.yml 自证接入），smoke.yml 在 check_versions 步骤后接入
- **最小条款流（方案 C）**：lib.rs 新增 `terms_agreed` / `agree_terms` / `reject_terms` 三命令 + `.terms_agreed_v1` 标记写入 `adr_data_dir()` 根（内容 `agreed_at=<ms>` 便于审计；与 Python 数据目录推导同源，升级/重装不重复弹）；prewarm.html 前置条款视图（`body.terms-mode` 盖住步骤流，中文文案：AGPL 义务通俗化 / 禁止性条款 / 责任边界 / 声纹告知 / 数据采集）；prewarm_flow ready 后导航前插 500ms 轮询 gate（未同意停在预热页，同意后放行导航，预热期间不浪费等待）；skip_prewarm 加 `terms_agreed_flag()` 防御兜底
- **版本面**：v1.2.1 十面对齐（pyproject / `__init__.py` / tauri.conf.json / Cargo.toml / package.json / package-lock.json 两处 / installer.iss / launcher.html fallback / README 四处下载链接）
- **验证**：cargo check 通过（19.85s 无错）/ check_versions 9 项 OK（all faces = 1.2.1）/ check_compliance 6 组 OK / pytest 基线不破
- **遗留**：条款英文版后补（不阻塞）；CLA/DCO 拍板待会议记录确认后补文档；v1.2.1 tag 与 Release 在本批合并 main 后进行

**批次 62 落地记录（2026-10-24，分支 main，发 v1.2.2）**：

- **审计设施（三层底座）**：新增 adr/core/audit.py 单点 — JSONL 追加写 `<adr_data_dir()>/logs/audit.log`；隐私边界：合成文本与素材声明只记 sha256 前 16 位指纹 + 长度（不落明文，声纹/文本属敏感信息，审计日志需可长期留存，参考音频只记文件名）；全 try/except 降级（写失败仅 warning 不阻断业务）；`ADR_SAFETY_AUDIT=0` 全局关闭
- **上传侧（③清单 4：素材声明 + 声纹单独同意）**：train_upload 加 `source_declaration` Form 字段（审计记 ext/size/decl_fp，**不记文件名/路径**，可能含人名）；profile_create 加 `consent_granted` 校验（缺失 422 拒绝建档）+ 同意结构**不信任客户端**，服务端重组 `{granted, at, statement[:200]}` 落 meta；save_voice/save_voice_auto 透传 source/consent（旧档案缺键读侧 .get 兼容）
- **合成侧（文本审计）**：SynthesisService 单点 `_audit_synthesis`（text_fp/text_len/参考音频名），stream_chunks 与 synthesize_once 两入口全覆盖
- **分发侧（③清单 4：WAV 元数据）**：新增 `inject_wav_info`（adr/utils/audio.py，依赖方向 server→utils 正确）— 尾部 LIST/INFO chunk（ISFT "ADR x.y (AI-generated speech)" + ICRD 日期 + ICMT "AI-generated voice cloning output, not a real person"，word-align pad）+ RIFF size 回写；`ADR_WAV_METADATA=0` 关闭；注入点**只有合成输出两条链**（audio_codec.pack_wav 延迟导入 + pipeline.save_wav）——素材预处理链（data/slice、data/separate 走 save_audio）不注入，真人素材不能标 AI 生成；流式 44B 头契约不动（N.E.K.O 按偏移切流）
- **无 UI 形态条款**：__main__ `_enforce_terms` 四级检查（`.terms_agreed_v1` 标记放行 → `--from-shell` 壳担保放行不写标记（prewarm 先于用户同意）→ `--agree-terms`/`ADR_AGREE_TERMS=1` 写标记 + 审计 `terms_agreed` 后放行 → 其余打印条款要点 exit 2）；compose 补注释形态 `ADR_AGREE_TERMS: "1"`（容器默认拒绝启动，标记写挂载 /data 仅需同意一次）
- **声音安全设置 + 设备标识（③清单 5）**：壳 ShellSettings 加 `safety_audit` / `wav_metadata` 两 `Option<bool>`（None=默认开，仅显式关注入 =0，旧 shell.json 缺字段自动默认开）；launcher.html"声音安全"节两开关 + 设备标识说明行（不采集不外发）；settings.py 新增 agree_terms / safety_audit_enabled / wav_metadata_enabled 三函数（单源函数式）
- **门禁**：check_compliance 升 **7 组**（第七组声音安全：audit/terms/条款 gate/inject_wav_info/_audit_synthesis/consent+declaration/compose/lib.rs 十项断言）
- **版本面**：v1.2.2 十一面同法对齐（+installer.iss `#define MyAppVersion`，批次 61 漏记该面）；README 下载链接 4 处 + 条款节补三种同意方式说明（壳条款页 / `--agree-terms` / `ADR_AGREE_TERMS=1`，未同意 exit 2）
- **验证**：check_versions 9 项 OK（all faces = 1.2.2）/ check_compliance 7 组 OK / 新增 tests/test_safety.py 20 项（审计 JSONL 与降级 / 条款 mark 与四级 gate / inject_wav_info wave.open 往返 / pack_wav API 面 / profile_create 422 / consent 落 meta 与向后兼容）/ 全量 pytest 462 passed + 1 skipped / cargo check 通过（12.5s）/ test_server_api 非流式 WAV 断言同步新契约（长度放开 + AI 标注断言）
- **遗留**：v1.2.2 tag 与 Release 后补（用户拍板"tag 后面再发"）；在线部署标识义务提示（③清单 4 附注）随部署文档批次落

**批次 63 首子批落地记录（2026-10-24，分支 main，v1.3.0 起版）**：

- **范围**：v1.3 Autopilot 内核底座第一项——t2s（AR 骨干）静态 KV Cache，decode 每步 `cat` 重建 KV（O(L²) 拷贝）换预分配 buffer + 原地写；与原路径逐 token 位等（等价性测试裁决，非近似）
- **实现（adr/models/adr_t2s_static_kv.py，桥逻辑在 ADR 侧不动 fork）**：`_decode_step` 逐算子对齐 `Text2SemanticDecoder` decode 块（norm/attn/mlp 全内联，KV 直接写 buffer 指定行）；`_make_buffers`（max_seq_len=21+1500 上限预分配）；`_static_kv_batch1`（非流式全等价复刻 batch_infer bs=1 路径：top-k/top-p/temperature/repetition_penalty/EOS 早停同一套 RNG 时序）；`_static_kv_naive`（流式生成器，chunk 切分/mute 帧逻辑逐条对齐 infer_panel_naive）；`install_static_kv_infer_panel` 四路分派（ids/bert 归一 → bs>1 或 prompt None 原路 fallback → streaming 走 naive 桥 → 非 streaming 走 batch1 桥）；模块级 `_last_kv` 保留末次 KV 供调试
- **开关与挂钩**：env `ADR_T2S_STATIC_KV` 默认 0（gsv_engine.py 与 `ADR_T2S_CUDAGRAPH` 同址 setdefault）；TTS.py 挂钩补丁（apply_compat_patches.py 新条目）排在 cudagraph 挂钩之后安装，同开时本桥 infer_panel 覆盖生效；无 CUDA 限制（CPU 可跑）、无缓存无锁（每请求重绑不变）
- **等价性测试（tests/test_t2s_static_kv.py，6 项全绿 41s）**：自建 2 层 32 维 Text2SemanticDecoder（config 纪律 embedding==hidden、EOS==vocab-1）；单步 decode 位等（masked/unmasked 双变体）；batch1 端到端 y 位等 + `_last_kv` 三元组断言（kv_len=MAX_LEN+PREFIX+步数、buffer 形状）；install 分派两路（batch_infer / naive_batched）位等；流式逐块位等（chunk_length=8）；mute 分支位等（chunk_length=6 + mute_emb_sim_matrix）
- **版本面**：v1.3.0 十一面同法对齐（批次 62 口径：pyproject / `__init__.py` / tauri.conf.json / Cargo.toml / package.json / package-lock.json 两处 / installer.iss / launcher.html fallback / README 下载链接）
- **验证**：check_versions 9 项 OK（all faces = 1.3.0）/ check_compliance 7 组 OK / 等价性测试 6 passed（41s）/ 全量 pytest 467 passed + 1 skipped（256.6s）/ cargo check 通过 12.29s（版本面 JSON/TOML 语法核验，adr-desktop v1.3.0 编译）
- **遗留**：v1.3.0 tag 与 Release 随 v1.3 收口（首子批不发版）；~~分桶 CUDA Graph~~（次子批已完成）、连续批处理、调度层（Profiler→Policy→Runtime）待后续子批

**批次 63 次子批落地记录（2026-10-24，分支 main，维持 v1.3.0）**：

- **范围**：分桶 CUDA Graph——修正批次 10 单 graph 缺陷（官方 CUDAGraphRunner 绑全量 `(1,2500,H,D)` KV、SDPA 无 mask，未写零槽各贡献 `exp(0)=1` 假 softmax 质量致 AR 分布漂移，故 `ADR_T2S_CUDAGRAPH` 默认 0）；分桶把假质量从 `(2500-L)` 压到 `(B-L)`，向 masked 精确 softmax 收敛，非新近似
- **实现（adr/models/adr_t2s_bucketed.py，不动 fork）**：`_PrefixView` 代理原地写真 buffer、返回 `[:, :B]` 前缀视图（捕获的 SDPA kernel 永久只读前 B 行，replay 零 Python 开销）；全档共享一份 2500 buffer（档间零拷贝、显存同单 graph）、graph 懒捕获；dispatch：idx=k≥1 取最小 `B ≥ prefill_len+k`（`dispatch_log` 单调不减）；默认档 `256,384,512,768,1024,1280,1536,2048,2500`（`parse_buckets` 去重升序、越界 ValueError、末档自动补 max_len）；`BucketedGraphRunner._handle_request` 逐行对齐官方循环（去 tqdm、bs==1、无 CUDA 走 eager 前缀视图）
- **开关与挂钩**：env `ADR_T2S_CUDAGRAPH_BUCKETS` 默认空（gsv_engine.py 同址 setdefault；settings.py docstring 同步）；在 `ADR_T2S_CUDAGRAPH=1` 已装的桥内选择 runner（非空且非 "0" 用分桶，"1"/"default" 走默认档），**无新增 TTS.py 补丁**；优先级 static_kv > 分桶 > 单 graph
- **等价性测试（tests/test_t2s_bucketed.py，7 项全绿 2.42s，含 2 项 GPU）**：ladder=[2500] 与 eager 参考 CPU 位等 + 与官方 CUDAGraphRunner GPU graph 位等（测试锚点）；子档 dispatch 规则（单调 / 首档 / 最小可容档）；前缀视图确实改变注意力（分桶生效证据）；`parse_buckets`/`_bucket_for` 边界；`_PrefixView` 是视图非拷贝（data_ptr 同、其余行不动）；graph 键 = 用过的档集合、二次请求复用不重捕获
- **验证**：check_versions 9 项 OK（all faces = 1.3.0）/ check_compliance 7 组 OK / 全量 pytest 474 passed + 1 skipped（272.5s）
- **遗留**：分桶仍挂在 `ADR_T2S_CUDAGRAPH=1` 之下（默认关）；真实模型音质/时长回归与开默认值需过落地门音质把关；连续批处理、调度层待后续子批；v1.3.0 tag 后发

**批次 63 第三子批落地记录（2026-10-24，分支 main，维持 v1.3.0）**：

- **范围**：调度层 Runtime 起步——N=1 硬预算（纪要并发落地顺序第一级）。批次 23 以来引擎锁事实上 N=1 但无显式预算、无统计；本子批把并发预算收口为一等公民。准入控制（队列上限/等待超时/拒绝语义）属第四阶段刻意不引入，排队无界不拒绝，线行为与批次 23 完全一致
- **实现（adr/core/scheduler.py 新建，GSVEngine 接线）**：`SynthesisScheduler`（N 预算并发槽：满额条件变量排队、释放唤醒一个；`AdmissionTicket(wait_ms/waiting/busy)`；`stats()` 快照 = max_concurrent/busy/waiting/admitted_total/max_waiting_observed/last_wait_ms/max_wait_ms）；GSVEngine 删批次 23 手工计数（`_qlock/_queue_waiters/_busy` 与 entered 标志舞蹈），两合成入口收口为 `_admit()`（slot + 审计 `schedule` 事件 wait_ms/waiting/busy，log_event 吞异常降级）；`queue_depth()/synth_busy()` 委托调度器——批次 23 观测语义与消费方（native /health、console /system/stats、call/pro 两页）不变
- **开关**：`ADR_SCHED_MAX_CONCURRENCY` 默认 1（settings.py `scheduler_max_concurrency` 单源解析，非数字/小于 1 回落 1）；≥2 为 1.3.x 预留——真并行需等引擎内连续批处理内核，当前 >1 仅放宽准入、合成仍被引擎锁串行
- **测试（tests/test_scheduler.py，7 项全绿）**：N=1 互斥与排队时序（waiting 可观测/预算满不准入/释放准入）；N=2 双槽并发（1.3.x 插槽位）；stats 快照；env 解析回落；引擎接线（非流式 admitted=1 后归零、流式请求级准入多段一槽、env 贯通）
- **验证**：check_versions 9 项 OK / check_compliance 7 组 OK / 全量 pytest 481 passed + 1 skipped（261s）
- **遗留**：真连续批处理内核（t2s bs>1 动态批，per-seq RNG + 音质门）与 N=2 固定份额进 1.3.x；准入控制为并发落地第四阶段；~~Profiler~~（第四子批已完成）→ Policy（画像映射配置档，不查设备表）待后续子批；v1.3.0 tag 后发

**批次 63 第四子批落地记录（2026-10-24，分支 main，维持 v1.3.0）**：

- **范围**：调度层 Profiler（Profiler→Policy→Runtime 第一段）+ 第三子批调度报告对外暴露。Policy 刻意不做：v1.3.0 只有 N=1 一档，加速开关（CUDA Graph/静态 KV/分桶）均默认关、未过落地门音质把关，没有可选档位可映射；N≥2 或加速开关过门后 Policy 才有意义
- **实现（adr/core/profiler.py 新建）**：`run_profile()` 微基准——固定 20 字测试文本 + 档案库首个档案参考音频（可 `--ref` 覆盖），warmup 1 次不计（JIT），测 n 次取中位数；RTF = 耗时/音频时长（<1 快于实时，与纪要锚点 E5-2667 v3 ≈1.7 同口径）；CUDA 下采 `max_memory_allocated` 峰值显存。档 = `{profile_id, timestamp, device_info, profiles[]}` 落 `<data_dir>/profile_<id>.json`；`profile_id` 为 MAC 哈希前 16 位，`device_info` 只取算力 + 显存总量（不记型号，贯彻"不查设备表"）。当前单测点 default（top_k=15，加速全关），加速测点待音质门过后扩展
- **接口**：CLI `adr profile [--ref --runs --force --show]`（已有本机档默认只显示不重采）；`/system/stats` 新增 `sched` 字段（第三子批调度报告，引擎未初始化为 null），模块级 `gsv_engine.sched_stats()` 不触发加载
- **测试（tests/test_profiler.py，7 项全绿，假引擎不依赖模型/GPU）**：档结构与测点参数、warmup+n 次调用计数与参数透传、测量失败归零、无档案报错、存取往返（ADR_DATA_DIR 隔离）、缺档返回 None、机器指纹稳定、模块级 sched_stats 未初始化为 None
- **验证**：check_versions 9 OK / check_compliance 7 组 OK / 全量 pytest 489 passed + 1 skipped（357s）
- **遗留**：真机标定（4060Ti/CPU 两锚点实测入档）；~~Policy~~（v1.3.x 首子批已完成）与壳首启接线；Runtime 闭环监控 + 自愈回退；~~v1.3.0 tag 后发~~（已打于 062f475）

**v1.3.x 首子批落地记录（2026-10-24，分支 main，维持 v1.3.0）**：

- **范围**：调度层 Policy（Profiler→**Policy**→Runtime 第二段），规则集 conservative-v1。原则：音质敏感开关（STATIC_KV / CUDA_GRAPH / 分桶 / INT8）**只推荐不自动开**，音质门通过前默认值不变；不影响音质的流式分块与 torch 线程数可自动调；不查设备表，只用 profile 实测 RTF / 显存 + 本机物理核数
- **实现（adr/core/policy.py 新建）**：`infer(profile, thresholds)` 纯函数 → `{policy_version, rules, generated_at, profile_id, basis, thresholds, overrides, recommendations}`；`policy.json`（`<data_dir>/policy.json`）**只存 overrides + 元数据**，阈值进文件可手改、重新生成时沿用；`apply()` 只应用白名单 `AUTO_KEYS`（ADR_TORCH_THREADS / ADR_TTS_STREAM_HEAD / ADR_TTS_SPLIT_MAX），手写进 overrides 的音质敏感键忽略并告警，`policy_version` 高于代码版本整体不应用；完整快照另做诊断导出 `export_snapshot()`（生效值 + 来源 env/policy/default）
- **规则（阈值默认）**：RTF>1.0 慢机 → 首段 14→10 字、长句细分 40→30 字；CPU 且物理核≥4 → 线程数 = 核数−1；CUDA → 推荐 STATIC_KV=1，显存≥6G 再推荐 CUDAGRAPH=1 + 分桶 default；INT8 推理侧未实现，推荐保持关
- **开关与接线**：优先级 显式 env > policy.json overrides > 代码默认值；引擎构造时 `policy.apply()`（早于首个请求切分与 `_lazy_init` 线程设置），失败不阻断合成；`settings` 新增 `stream_head_chars()` / `split_max_chars()`，流式切分改读这两项（默认值与原字面常量一致）；CLI `adr policy [--show --snapshot]`
- **测试（tests/test_policy.py，11 项全绿）**：CPU 慢机自动项、少核不留核/快机不收短、GPU 门控项只推荐不入 overrides、显存阈值、阈值可配、重新生成沿用用户阈值、apply 白名单 + env 优先 + 自写键可更新、拒绝高版本、缺档/坏档、快照来源、未 apply 进程快照显示待生效值、settings 分块默认与非法回退
- **验证**：check_versions 9 OK / check_compliance 7 组 OK / 全量 pytest 499 passed + 1 skipped（269s，efea18e）；真机（CUDA cc 8.6 / 8G / 20 物理核）`adr profile` → default RTF=1.166、显存 1206MB（3 次中位数）→ `adr policy` 自动生效 STREAM_HEAD=10 / SPLIT_MAX=30，GPU 设备不改线程数，STATIC_KV / CUDAGRAPH / 分桶仅推荐、INT8 保持关，符合 conservative-v1 预期
- **真机修正**：`--snapshot` 在 CLI 进程（未调 apply）把 policy 自动项误标为 default；改为未进 env 的 AUTO_KEYS 按 policy.json overrides 显示（source=policy），显式 env 仍优先，补测试 1 项
- **首启接线（首启配置⑯）**：服务端实现，壳不改（CLI / Docker 同享）。`gsv_engine.prewarm` 到 ready 后调度一次后台线程（每进程至多一次，failed 不触发）：等 10s 空闲期 → 再等引擎无排队 / 无合成 → 本机已有 profile 或档案库为空则跳过（建档后下次启动再跑）→ `run_profile` → `policy.generate` + `apply`，当次进程后续请求即受益；采样走引擎调度槽，期间用户请求排队一次；失败只记日志。测试 tests/test_idle_profile.py 4 项（单次调度 / 已有 profile 跳过 / 无档案跳过 / 采样后生成并应用）
- **遗留**：policy_version 随规则变化递增；音质门通过后另出规则集（如 balanced-v1）才允许自动开加速开关；无档案首启是否内置测定用参考音频（待定，当前跳过）；真机两锚点标定阈值；~~Runtime 闭环~~（v1.3.x 第二子批已完成）

**v1.3.x 第二子批落地记录（2026-10-24，分支 main，维持 v1.3.0）**：

- **范围**：调度层 Runtime（Profiler→Policy→**Runtime** 第三段）闭环监控 + 自愈回退；只动音质中性的流式分块（STREAM_HEAD / SPLIT_MAX），不碰加速开关
- **实现**：新建 `adr/core/runtime.py`。引擎每次非缓存合成上报 `record(ok, elapsed_s, audio_s)`，滑动窗口记 RTF（首请求计时始于 `_lazy_init` 之后，不含模型加载）；窗口 ≥ MIN_SAMPLES 且无失败、RTF 中位数 > policy 阈值 `slow_rtf` → 收短分块至 policy 慢机值，记下调整前快照为已知良好配置并清窗口；之后窗口失败率 ≥ FAIL_RATE_FALLBACK → 回退快照并冻结不再调。显式 env 不覆盖、目标值已生效不重复调；`ADR_RUNTIME_AUTOTUNE=0` 只监控不调。报告挂 `/api/adr/v1/system/stats` 的 `sched.runtime`；`policy.export_snapshot` 中 Runtime 所改键来源显示 `runtime`
- **首启接线修正**：后台采样原只 `run_profile` 不 `save_profile`，导致每次启动重采；已补存盘
- **测试**：tests/test_runtime.py 9 项；全量 pytest 513 passed / 1 skip，check_versions 9 项、check_compliance 7 组 OK
- **真机验收（本机 GPU，RTF≈1.1）**：① 移走 profile/policy 首启 → 后台采样写出两文件，再启跳过；② 有 policy（10/30 已生效）→ 不重复调；③ 无 policy + `AUTOTUNE=0`，6 条 → 只记样本；④ 无 policy + autotune，8 条 → 中位 RTF 1.11 触发 `tune`，分块收至 10/30，记已知良好快照，后续请求正常
- **遗留**：`adr policy --snapshot` 为独立进程，看不到服务进程内 Runtime 改动（以 stats 接口为准）；失败回退路径仅单测覆盖，未在真机制造失败；stats 事件 reason 在 PowerShell 5 显示乱码（客户端解码问题，待确认服务端 charset）

**v1.3.1 安装包实测记录（2026-10-25，分支 main，v1.3.1）**：

- **范围**：CPU / GPU 两个安装包各走一遍"静默安装 → 首启 → 建档触发采样 → Policy 生效"。本机 Xeon E5-2650 v3（20 物理核）+ RTX A2000 Laptop 8G（cc 8.6），数据目录走 legacy `F:\ADR_data`，服务端口 9881
- **CPU 包**（ADR-Studio-1.3.1-x64-setup.exe，459 MB）：首启档案库空 → 跳过采样；建档 `cpufirst` 后当场调度采样，约 3.5 分钟完成 → profile `device=cpu`、default RTF=1.81 → policy overrides STREAM_HEAD=10 / SPLIT_MAX=30 / TORCH_THREADS=19（20 核留 1），日志确认已应用。RTF 与纪要 E5-2667 v3 锚点 ≈1.7 同量级
- **GPU 包**（ADR-Studio-1.3.1-x64-gpu-setup.exe，1926 MB，torch 2.10.0+cu130）：移走 CPU 轮 profile/policy（备份 `*.cpurun_202610251207`）后首启 → 已有档案，后台自动采样约 1 分钟 → profile `device=cuda`、8.0G、default RTF=1.177、峰值显存 1170MB → overrides STREAM_HEAD=10 / SPLIT_MAX=30，不改线程数；STATIC_KV / CUDAGRAPH / 分桶 default 仅推荐，INT8 保持关。安装目录 runtime 跑 `adr policy --snapshot` 来源均为 policy / default，符合 conservative-v1
- **构建问题与修复**：① `build_runtime.ps1` 写 requirements 时 PS 5.1 对中文注释 here-string 写入失败，pip 读到旧文件，CPU runtime 漏 4 个 G2P 包 → 加写回校验（83162d2）；② 旧 `build\runtime-gpu` 为修复前产物（缺 pip / PyYAML，服务起不来）→ 用修复后脚本重建；③ GPU 打包参数是 `-Gpu`，误传 `-AllowCuda` 被静默忽略、去检查 runtime-cpu；④ tauri / ISCC 的 stderr Info 被 PowerShell 转成 CLIXML Error、退出码 1，属误报，以产物时间戳为准
- **~~下一批首项~~（已修复 ea049c5，见下节）**：profile 按机器指纹存档、key 不含设备类型，同机 CPU 包换 GPU 包会命中旧画像、跳过重采，沿用 CPU 画像（含 TORCH_THREADS=19）。本次实测因手工移走文件未触发。修复：最小改法为 `load_profile` 命中后比对 `device_info.type` 与当前设备，不一致视为缺档重采；更彻底为把 type 并入文件 key（`profile_<mid>_<type>.json`），CPU / GPU 画像并存、来回换包不重采；policy.json 同步按来源 profile 失效重新生成
- **遗留**：沙箱环境下 GUI 进程随 shell 回收，GPU 轮未在服务进程内做合成 + stats 的 Runtime 观测（第二子批真机验收已覆盖）；旧版 unins000 静默卸载卡住，用新安装器覆盖安装绕过；会议纪要已 .gitignore（4b48c06）

**画像按设备分档（v1.3.x 下一批首项，已落地，提交 ea049c5）**：

- **profiler**：文件 key 并入设备类型 `profile_<mid>_<type>.json`，`load_profile` 按 `current_device_type()` 取，CPU / GPU 画像并存、来回换包不重采；兼容 v1.3.1 旧文件 `profile_<mid>.json`，仅当其 `device_info.type` 与当前设备一致时沿用，否则视为缺档触发重采
- **policy**：`apply` / `export_snapshot` 比对 `basis.device` 与当前设备，不一致则不应用（CPU 包生成的 TORCH_THREADS=19 不会带进 GPU 包）；来源缺失或当前设备 unknown（torch 不可用）时不拦
- **后台 worker**：当前设备画像已在但 policy.json 来自另一设备（换回已采过的设备）时，跳过采样、按本设备画像就地重新生成并应用
- **测试**：新增画像并存 / 旧文件兼容 / policy 设备不一致不应用 / worker 重新生成 4 项；全量 523 passed, 1 skipped
- **未实测**：未重打安装包做真机换包验证（逻辑由单测覆盖）

---

## 总体 Review — v1.3.1 实测与画像分档（2026-10-23 ~ 10-25）

**时间线**：批次 63 第四子批（Profiler）→ v1.3.x 首子批（Policy）→ v1.3.x 第二子批（Runtime）→ v1.3.1 版本抬升 + 双包实测 → 画像设备分档修复；v1.3.0 tag（062f475）之后 12 次提交

**落地清单**：
1. 调度层三级（Profiler → Policy → Runtime）全链路完成，Policy 保守规则集（音质敏感项只推荐不开）
2. v1.3.1 双包实测：CPU 459 MB（RTF 1.81）/ GPU 1926 MB（RTF 1.177），两条链路验证"首启 → 采样 → Policy 生效"
3. 构建问题修复：requirements 写回校验（83162d2），防 PS 5.1 隐身漏失
4. 画像设备分档：文件 key 并入设备类型 `profile_<mid>_<type>.json`，CPU/GPU 换包不串用旧画像（ea049c5）

**质量快照**（提交 fd8e17f）：
- 测试：523 passed, 1 skipped（v1.3.0 tag 时 489 passed，+34 项）
- 门禁：check_versions 9 OK（all faces = 1.3.1）/ check_compliance 7 组 OK
- 调度层模块：profiler.py 224 行、policy.py 212 行、runtime.py 150 行
- 对应测试：test_profiler.py 135 行、test_policy.py 144 行、test_runtime.py 103 行、test_idle_profile.py 137 行、test_build_profile.py 138 行

**关键设计决策**：
- **不查设备表**：Policy 只用实测 RTF / 显存 + 物理核数，不记型号不查数据库（设备碎片化 + 驱动影响基准）
- **音质敏感项只推荐**：STATIC_KV / CUDA Graph / 分桶 / INT8 默认保持关，音质门通过后另出 balanced-v1 规则集才自动开
- **画像设备分档晚一批**：实测期间读代码发现换包会沿用旧画像（CPU 包的 TORCH_THREADS=19 会带进 GPU 包；本次因手工移走文件未实际触发），补做 ea049c5
- **首启接线后台化**：预热 ready 后等 10s 空闲期且无排队 / 无合成才采样，采样期间用户请求排队（实测 GPU 约 1 分钟、CPU 约 3.5 分钟）
- **Runtime 只调分块**：闭环监控只收短 STREAM_HEAD / SPLIT_MAX 这类音质中性项，不碰加速开关（与 Policy 同一音质原则）

**遗留关闭**：
- 换包沿用旧画像 → ea049c5
- 会议纪要未跟踪 → 4b48c06
- 端口对不上 → 不存在（23366 是临时命令写错）

**环境类遗留**（不进下批次）：
- GPU 轮 Runtime 合成未实测 → 沙箱窗口回收，第二子批真机验收已覆盖
- 旧版卸载器卡住 → 新安装器覆盖安装绕过

**技术债**（远期观察）：
- Policy 规则版本递增（v1 → v2 兼容）
- balanced-v1 规则集（加速开关过音质门后）
- Runtime 失败回退真机验证（当前只单测）
- stats 事件 reason PowerShell 5 乱码（服务端 charset）
- 无档案首启内置测定参考音频（当前跳过）

**验收五指标现状**（纪要只定了指标名，没给量化目标；下表只记现有证据，阈值待定）：

| 指标 | 现有证据 | 缺口 |
|------|---------|------|
| 调参遗憾值 | 无 | v1.3.0 只有 N=1、加速全关，暂无可选档位；N≥2 或加速开关过门后才有意义 |
| 收敛时间 | 首启到 policy 生效：GPU 约 1 分钟、CPU 约 3.5 分钟（单次实测） | 未定目标值，样本仅两台次 |
| 免调参率 | 两台次实测均零手工介入自动生效 | 样本量不足以给比率 |
| 自愈成功率 | 回退路径单测覆盖 | 未在真机制造失败 |
| 流式连续性 | 无专项测量 | Runtime 改分块在请求边界生效（代码层面），未做断流统计 |

**下阶段入口**：
1. 并发落地顺序（纪要）：N=2 固定份额 → 动态份额 + 隔离 → 准入控制；N=2 前置为真连续批处理内核（t2s bs>1，per-seq RNG + 音质门）
2. 加速开关（STATIC_KV / CUDA Graph / 分桶）过落地门音质把关，之后另出规则集让 Policy 自动开
3. 真机标定两锚点（E5-2667 v3 CPU / 4060Ti），校准 conservative-v1 阈值

### ④ Autopilot 与 v1.3 范围说明

- **命名**：v1.3 "Autopilot"——纪要 Autopilot 技术线与原 H2 推理加速线合并为一条
- **两层关系**：
  - **内核底座**（= H2 已规划项）：静态 KV Cache 原地写 + 分桶 CUDA Graph + 连续批处理 + FA2→SDPA→动态三级降级（GSV-TTS-Lite / Aqua-TTS 借鉴；批次 10 CUDAGraphRunner 音质崩的正解，落地门音质把关不变）
  - **调度层**（Autopilot 新增）：Profiler（微基准 20–30s 采样 RTF/显存）→ Policy（画像映射配置档，**不查设备表**，可自动生成也可手动覆盖——纪要首启配置⑯）→ Runtime（闭环监控 + 自愈回退 + 调度报告）
- **并发落地顺序**（纪要）：N=1 硬预算 → N=2 固定份额 → 动态份额 + 隔离 → 准入控制；v1.3.0 交付 N=1，N=2 起进 1.3.x
- **标定锚点**（纪要实测）：E5-2667 v3 RTF≈1.7（CPU 服务器）/ 4060Ti ≈2G 显存（低配 GPU）——Autopilot 标定、预算分配与并发上限推算的两组基准
- **验收五指标**（纪要）：调参遗憾值 / 收敛时间 / 免调参率 / 自愈成功率 / 流式连续性
- **四版本性能口径**（纪要表，Pro+ 两版缓行、口径备案不进近期批次）：ADRstudio 并发 3（双卡 48G 联合 96G）/ ADRserver 并发 100（6×96G）/ Studio Pro+ 并发 10（联合预算 192G，物理 6×96G，**预算≠物理总和**）/ Server Pro+ 并发 1000（8×96G）

### ⑤ macOS 窗口与滑出规则

- **窗口**：v2.0（2026.10–2027.6）**末段**启动——约 27.3 起预研立项（GSV 上游 macOS/MPS 可用性 + python runtime macOS 打包 + Tauri 壳签名链，三项过门才正式进批次）
- **不阻塞主线**：Windows 全形态 + Linux 容器的发布与门禁不等 macOS
- **滑出规则**：v2.0 发布时点若 macOS 未达成"安装包构建 + 合成冒烟 + 三形态文档"三项，则滑出至 v2.0.x patch 补发，v2.0 按 Windows + Linux 容器双平台发布、不延窗口
- 纪要平台矩阵另含鸿蒙/国产 OS（L224/237/273）：列 v2.0 后远期观察项，不进窗口承诺

## 原则

- 能统一的尽量统一，降低使用门槛（用户定调）
- 音质绝对优先的底线 + 用户自选的弹性（极速模式默认关，MOS 门把关）
- 不碰模型结构（框架层项目定位不变），加速走工程路径
