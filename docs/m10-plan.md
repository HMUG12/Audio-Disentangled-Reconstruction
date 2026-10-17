# M10 计划书（草案）— 体验、低配加速与档案能力

> 状态：**草案 v1**（2026-10-07 与用户商定大部分方向；批次顺序未定稿，见文末"待定"）
> 前置：批次 46 已发布 v1.0.0（GitHub Release + 123 云盘安装包）；批次 45 Review 定位的"非本机用不了"卡点由本计划设置中心收编。

## 已确定方向

### A. 桌面壳全面体检（候选批次 47）

- [ ] **静态审计**：前端（launcher.html / pro / easy / call 控制台页）全部 `invoke()` / onclick 处理器对照 `lib.rs` 命令注册表，找出未注册/未绑定/无反馈的入口
- [ ] **动态点检**：控制台四页（/, /pro, /easy, /call）用浏览器自动化逐按钮点检；壳内页面（启动器/托盘/关于页）用桌面自动化逐按钮点检
- [ ] **修复**：死按钮当场修（历史上踩过：命令未注册、下载事件静默取消、fallback 导航缺失）
- [ ] **交互反馈统一**：所有有副作用的按钮统一 loading 态 / 错误提示 / 成功反馈形态

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

- [ ] 导出 = zip 容器：档案元数据 + 参考音频 + 绑定的 t2s/vits 权重对 + 风格预设（清单文件带版本号）
- [ ] 导入 = `validate_weights_file` 校验（批次 43 资产复用）+ 重名冲突处理（重命名/覆盖二选一）
- [ ] **导出提示**：权重即声音本体，含生物特征性质数据，导出时明确告知
- [ ] 解决"声音档案不在安装包内、换机即失"的债（m9 遗留）

### E. 续训流程（批次 50 已交付 2026-10-10）

- [x] 底层已通：增量训练 + enc_q 预训练合并（相似度 0.118 事故的修复资产）复用
- [x] 档案版本化：bind_weights 写 weights_history 版本链 + `/profiles/{name}/history` / `/rollback` 端点，redo 逐版回退（续训翻车能退回，且不会在相邻两版间振荡）
- [x] easy/pro 控制台续训入口：easy "改进已有声音" 勾选 / pro 绑定下拉 "↻ 续训并绑回" —— 选已有档案 → 追加音频 → 处理 → 续训 → 质量门比对新旧相似度
- [x] 质量门防退化：训练收尾 finalize_bind_sim 用门禁最优 sim 对比旧版（Δ<-0.05 置退化标记），pro 训练完成弹警告 + 档案选择器 ⚠ 标注 + 一键回滚

### F. 多语言（已定：两者都要；批次位置待定）

- [x] **合成多语言**（批次 53 已交付 2026-10-17）：GSV v2 底模支持中英日粤混说——语言参数暴露（合成 API + 控制台）+ 音色语言标记 + 多语言测试矩阵。落地：① 档案 language 标记全链路（save_voice/save_voice_auto/meta.json/profiles/create API + easy 建档语种选择）；② 语言默认联动档案语种（native /tts 省略 text_lang/prompt_lang → 档案 language，_resolve_profile 收口；v3 双工 init 补 prompt_lang 透传 + _push_task 联动；v3 /voices 暴露 language）；③ easy 合成面板语言下拉（auto 默认/中/英/日/韩/粤）；④ runtime 补 G2P 包（pyopenjtalk/jamo/ToJyutping/ko_pron 入 requirements + check_runtime 第 7 项体检）。**实测矩阵（yui 档案）**：zh/auto(中英混)/en/ja/yue 全通过，产物 output/multilang_test/；**ko 已知限制**：GSV ko 链依赖 g2pk2+eunjeon(mecab-ko)，Windows cp313 不可用——v1.1 宣传口径"中/英/日/粤 + 中英混说"，韩语待 v1.2 容器版（Linux 可装 eunjeon）。微调权重纯中文数据的日粤口音/相似度待用户听感定宣传口径。**用户听感反馈（2026-10-17）**：日/粤可接受；中英混说模式清晰度/流畅度略欠（底模特性，单语种不受影响）——已记入 release notes 已知限制，后续 v1.3 推理加速/引擎更新再看是否缓解
- [ ] **UI 界面多语言**（批次 54）：控制台页面 i18n（中/英），面向海外发布。后做的理由：批次 53 会新增语言选择 UI，先 i18n 会返工

## H. 下阶段路线图（2026-10-11 规划）

**版本梯子**：v1.0.1（双包发布）→ v1.1（批次 53/54 多语言+i18n）→ v1.2（多形态分发）→ v1.3（推理加速）→ v1.4（训练加速）；B 分发生态/HF Space demo 零散穿插。

### H1. v1.2 多形态分发（用户拍板优先）
- **容器版（Docker，优先）**：标准服务器版——无 UI、CLI/HTTP API 操作；Linux CUDA/CPU torch + adr server + compose；GSV 上游有官方镜像先例（amd64/arm64）
- **Windows 无 UI 服务版**：同一 runtime 跳过 Tauri 壳，`python -m adr.server` 服务化脚本（低工作量）
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

## 原则

- 能统一的尽量统一，降低使用门槛（用户定调）
- 音质绝对优先的底线 + 用户自选的弹性（极速模式默认关，MOS 门把关）
- 不碰模型结构（框架层项目定位不变），加速走工程路径
