# ADR 项目审计报告（第二轮）

- 审计日期：2026-11-07
- 审计对象：`e:\新创意构思\新建文件夹\ADR`（Python 83 个模块 / 20,847 行；Rust 桌面壳 `desktop/src-tauri`）
- 审计方式：静态分析 + 阅读代码 + 现有测试收集（未跑训练、未打包、未压测）
- 编号说明：本轮全部重新编号（`SEC/BUG/CON/PERF/DATA/API/CACHE/OPS`），不沿用上一轮 `CODE_REVIEW.md` 的编号与结论。上轮报告仅作为线索，所有条目均已回到当前代码重新取证并重新投票。

---

## 一、执行摘要

### 1.1 总体结论

当前代码相比上一轮已有明显收敛：流式 `q.put` 阻塞、`tts_cache` 锁内磁盘 IO、训练/推理 mel 算法分叉、桌面更新链验签、`GET /set_*_weights` 守卫、梯度累积余数 flush、`DataLoader` collate、极小数据集划分、权重指纹、ffmpeg 超时、音色包两段式提交等均已落地（见附录 A）。

但仍有 **2 条严重问题**，都会在网络可达时直接打挂服务，且都不是"配置问题"，而是代码路径本身不成立：

1. **未鉴权的 `GET /api/v2/tts` 在参数校验之前就按请求里的权重路径做全量文件校验** —— 任意本地路径/UNC 共享都会被整文件读入内存，可被单请求打爆进程内存，也是跨站/内网触发的入口。
2. **流式合成卡死后的自动恢复链是断的** —— worker 卡住时持有引擎锁、消费端 `q.get()` 无超时、`_mark_stalled()` 只置标记不抢锁、`_lazy_init()` 又需要同一把锁，于是"降级重建"永远执行不到，线程池与准入槽一起泄漏，表现为全站假死。

另有 7 条高危：每请求重复全量校验权重（正常流量即可 OOM）、非 GSV 入口的 `weights_only=False` 反序列化、`0.0.0.0` 无 key 仅告警、小数据集训练静默空转、每请求在全局锁内读盘、async 内同步 ffprobe 阻塞事件循环、桌面端管道无人消费导致子进程阻塞。

### 1.2 各维度统计

| 维度 | 主要产出 | 量化结果 |
|---|---|---|
| Scout 架构 | 模块分层与依赖方向、热点/风险区 | 12 个包，`models`(6,666 行)/`server`(3,628)/`training`(3,215)/`core`(2,456) 占 77% |
| Bugs 缺陷 | 资源与错误处理缺陷 6 条 | `except Exception` 137 处、`except BaseException` 18 处 |
| Security 安全 | 严重 1、高危 3、中危 3 | 未鉴权写/读入口 2 个；`weights_only=False` 5 处 |
| DeadCode 死代码 | 见第五章 | 零引用符号 7 个；不可达回调约 1,000 行；`__init__` 导出 40/56（71%）无包级消费者 |
| TestGaps 测试缺口 | 见第六章 | 59 文件 / 781 个 `def test_` / pytest 收集 854 条 / 13,913 行；测试:源码 ≈ 1.76:1（按公开函数约 3:1） |
| Performance 性能 | 4 条瓶颈 | 每请求一次读盘 + 每次 mel 重建滤波器组 + 逐块 ffmpeg + O(n²) 拼接 |
| Verify 对抗验证 | 33 条候选 → 24 条保留 | 保留率 73%，全部为 3/3 全票；9 条未过半/证据不足已剔除（不列入本报告） |

### 1.3 建议修复顺序

- **P0（当日）**：SEC-01、CON-01。
- **P1（本周）**：SEC-02、SEC-03、SEC-04、PERF-01、PERF-02、BUG-01、BUG-02。
- **P2（迭代内）**：其余中危。
- **P3（顺手清理）**：死代码、低危、文档口径。

---

## 二、审计方法

### 2.1 流程

1. **Scout**：按包统计文件/行数，画出调用方向（`server → services/models`、`training → data/core`、`desktop → server HTTP`），标注高风险区。
2. **逐维取证**：每条发现必须落到「文件 + 行号 + 可复述的代码证据或调用链」。
3. **Verify（3 视角，3/3 才保留）**：
   - **正确性**：代码是否真的按描述执行？有无守卫/兜底使结论不成立？
   - **影响**：触发条件在真实部署（本机 / `0.0.0.0` / 桌面壳）下是否可达？后果是否可观测？
   - **成本**：修复是否可控？有无更便宜的替代（配置/文档）？
   - 任一视角给出"不成立/不可达/不值得"即剔除。
4. **Synthesize**：只写通过验证的条目；推测性内容显式标注 `（推测）`。

### 2.2 明确未做的

- 未运行训练、未打包、未跑 GPU 推理、未做压力测试与浏览器 PNA 实测。
- 覆盖率未用 `coverage.py` 实测，TestGaps 的量化采用「可测函数数 / 被直接引用的风险函数」静态口径，不做百分比伪造。

---

## 三、Scout：架构与风险区

### 3.1 模块规模与职责

| 包 | 文件 | 行数 | 职责 | 主要依赖 |
|---|---:|---:|---|---|
| `adr/models` | 23 | 6,666 | GSV 套壳引擎、自研模型、缓存/合批、引导下载、音色包 | `core`, `third_party/gpt_sovits` |
| `adr/server` | 15 | 3,628 | FastAPI v2/v3/native/openai 兼容层、安全守卫、编码、控制台 | `services`, `models` |
| `adr/training` | 8 | 3,215 | Trainer、Dataset、回调、优化器 | `data`, `models`, `core` |
| `adr/core` | 15 | 2,456 | 配置、运行时自适应、策略、调度器、审计、设备 | 无（被全仓依赖） |
| `adr/data` | 8 | 1,327 | 切片、ASR、G2P、F0、数据流水线 | `utils/audio` |
| `adr/webui` | 1 | 1,206 | Gradio 经典控制台 | `cli`, `models` |
| `adr/inference` | 2 | 496 | 自研推理管线 | `models`, `vocoder` |
| `adr/utils` | 2 | 417 | 音频工具 | 无 |
| `adr/vocoder` | 3 | 359 | BigVGAN | `core` |
| `adr/services` | 2 | 185 | 合成服务门面（分块/编码/参数校验） | `models`, `server/audio_codec` |
| `adr/eval` | 2 | 142 | 评测探针 | `models` |

依赖方向基本单向：`server → services → models`，`training → data → utils`，`core` 为底座。没有循环导入，但存在 **运行时反向依赖**：`models/gsv_engine.py` 顶部 `from adr.core.audit import log_event`、`from adr.core.scheduler import SynthesisScheduler`，而 `core/runtime.record()` 又由引擎调用 —— 这条链上的任何阻塞都会放大成全站阻塞（CON-01、PERF-01 的根因）。

### 3.2 风险区排序

1. **`models/gsv_engine.py`（单文件 ≈ 1,150 行）**：同时持有引擎锁、chdir 全局锁、准入槽、段缓存、线程 worker、降级标记 —— 本项目绝大多数并发缺陷集中于此。
2. **`server/v2_compat.py`**：GET/POST 双入口共用 `tts_handle`，安全守卫只在部分路由上挂载，参数校验与权重校验顺序敏感。
3. **`training/callbacks.py`（≈ 1,100 行）**：产品入口不注册，测试覆盖又偏向 mock，属于"高复杂度 + 低生产触达"的死角。
4. **`desktop/src-tauri/src/lib.rs`**：进程守护逻辑（管道、探活、孤儿收编）无法在 CI 中验证。

---

## 四、严重问题

### SEC-01 · 未鉴权 `GET /tts` 在参数校验前对请求级权重路径做全量文件校验

> **状态：已修复（批次 86，2026-11-07，代码侧；未重出包）**
> GET `/tts` 已挂 `Depends(require_browser_safe)`；权重预检移到 `_check_params()` 之后，
> 且只校验**用户传入**的请求级权重（`_resolve_profile` 回填的档案权重不参与）；
> `validate_weights_file()` 先做 UNC/设备路径拒绝 + `is_file()` + `ADR_TTS_WEIGHTS_MAX_MB`
> （默认 2048MB）大小上限，通过后才 `torch.load` / `f.read()`。回归用例见
> `tests/test_server_security.py::test_audit_sec01_*`。

- **位置**：[v2_compat.py L353-L384](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L353-L384)、[v2_compat.py L244-L263](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L244-L263)、[gsv_engine.py L137-L170](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L137-L170)、[gsv_engine.py L238-L257](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L238-L257)、[native.py L334-L340](file:///e:/新创意构思/新建文件夹/ADR/adr/server/native.py#L334-L340)
- **证据**

  ```python
  # v2_compat.py:353  无 Depends(require_browser_safe)，t2s_weights 来自 query
  @router.get("/tts", summary="TTS 推理 (GET)")
  async def tts_get_endpoint(request: Request, ..., t2s_weights: str = None, ...):

  # v2_compat.py:244  权重校验发生在 _resolve_profile / _check_params 之前
  for _wk in ("t2s_weights", "vits_weights"):
      _wp = req.get(_wk)
      if _wp:
          await asyncio.to_thread(_validate_weights_file, _wp)

  # gsv_engine.py:160  快路径：torch 全量读取
  obj = torch.load(weights_path, map_location="cpu", weights_only=True)
  # gsv_engine.py:238  慢路径：整文件进内存后再解 zip
  with open(weights_path, "rb") as f:
      raw = f.read()
  ```

  调用链：浏览器/任意 HTTP 客户端 → `GET /api/v2/tts?t2s_weights=<任意路径>` → `tts_handle` → `_validate_weights_file` → `torch.load`（失败）→ `_restricted_weights_scan` → `f.read()`。整条链**不要求** `text` 合法、不要求鉴权（未设置 `ADR_TTS_API_KEY` 时）、不需要 Origin（GET 无守卫依赖，见 [origin_guard.py L113-L131](file:///e:/新创意构思/新建文件夹/ADR/adr/server/origin_guard.py#L113-L131) 对"无 Sec-Fetch-Site"一律放行）。
- **影响**
  - 任意本地文件被打开并整读进内存：指向 10GB 文件/网络共享即可打爆 RSS（OOM，进程重启、正在跑的合批全部丢失）。
  - UNC 路径（`\\attacker\share\x`）会让服务主动发起 SMB 连接，构成内网探测/凭据转发面。
  - 无需鉴权的 GET 也意味着**跨站可触发**：`<img src="http://127.0.0.1:9881/api/v2/tts?t2s_weights=...">` 即可。（浏览器 Private Network Access 是否拦截该 no-cors 请求需实测，标 **推测**；非浏览器客户端与局域网暴露场景无此限制。）
  - 错误响应只回显路径与异常类型（[v2 L260-L263](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L260-L263)），**不泄露文件内容** —— 本条不是"任意文件读取泄露"，而是"任意文件打开 + 内存耗尽 + 内网探测"。
- **验证**
  - 正确性 ✓：路由签名确认无守卫依赖；校验顺序在 `_check_params()` 之前（L268）；快/慢路径均全量读盘。
  - 影响 ✓：单请求即可触发；默认 `ADR_TTS_API_KEY` 为空时完全无门槛。
  - 成本 ✓：修复极小 —— GET 路由加 `Depends(require_browser_safe)`，并把权重预检移到 `_check_params` 之后、且只允许档案目录内路径（`resolve_within`）。
  - **3/3 通过。**

---

### CON-01 · 流式卡死后的自动恢复链不成立（线程 + 锁 + 无超时 `get`）

> **状态：已修复（批次 86，2026-11-07，代码侧；未重出包）**
> 消费端 `q.get(timeout=_STREAM_IDLE_SEC=120s)`，静默超时即判定卡死并跳过 30s 排空直接降级；
> 引擎锁改为 `_engine_lock()`（`acquire(timeout=55s)`）；调度准入 `slot(timeout=55s)` 同样有界；
> 超时统一抛 `EngineStalledError`（`code=engine_stalled`）→ HTTP 503 可重试。

- **位置**：[gsv_engine.py L868-L986](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L868-L986)、[gsv_engine.py L531-L556](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L531-L556)、[gsv_engine.py L724-L726](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L724-L726)
- **证据**

  ```python
  # L868  无界队列（批次82 为解 put 阻塞而改）
  q = _queue.Queue()
  def _produce():
      with self._admit(), _RuntimeProbe(stream=True) as probe:
          with self._lock:            # L875  整个 run() 期间持锁
              self._lazy_init()
              ...
              for sr, chunk in self._tts.run(inputs):   # L926  上游可能永久卡住
                  q.put((chunk, int(sr)))
  ...
  item = q.get()                       # L957  消费端无超时阻塞
  ...
  finally:
      deadline = time.monotonic() + _STREAM_DRAIN_SEC   # L970  只约束"清理排空"
      ...
      worker.join(timeout=_STREAM_JOIN_SEC)             # L982
      if worker.is_alive():
          self._mark_stalled(worker)                    # L986
  ```

  ```python
  # L531  注释自己写明了矛盾点
  def _mark_stalled(self, worker):
      """这里不抢 _lock (卡死的 worker 可能正持着它), 只置降级标记;
      下一次 _lazy_init() 在持有 _lock 后丢弃模型并重建。"""
      self._stalled = True
  # L555  但 _lazy_init 的第一步就在 _lock 内
  with self._lock:   # L724 warmup；L875 流式；L1069 synthesize
  ```

  失效链：① worker 卡在 `_tts.run()` 内 → 永不产出 `_DONE` → ② 消费端 `q.get()`（L957）**没有超时**，`finally` 永远不会进入，`_mark_stalled` 根本执行不到；③ 即使被外部取消/断连推进到 `finally`，`_mark_stalled` 置位后，下一次请求要先拿 `_lock` 才能走 `_lazy_init`，而锁仍在卡死 worker 手里 → 重建无法执行；④ 结果：所有后续合成在 `with self._lock`（L875/L1069）排队；准入槽 `AdmissionTicket` 与线程池线程也一直被占。
- **影响**：一次上游卡死 = 全站假死（HTTP 请求永久 hang、`/health` 仍报 ready 见 BUG-05），只能重启进程；桌面壳的守护会看到"进程活着但无响应"，不会自动拉起。
- **验证**
  - 正确性 ✓：`q.get()` 无超时；`_mark_stalled` 只在 `finally` 被调用；`_lazy_init` 需同一把锁。
  - 影响 ✓：触发条件是上游 `run()` 卡住（长音频 AR 解码、CUDA 卡死、chdir 竞争都可能），非理论场景。
  - 成本 ✓：改动集中在 `finally` 与消费循环（给 `q.get()` 加超时 + 超时后主动判 stalled + 用独立"重建锁"或带超时的 `acquire`），不触碰合成主路径。
  - **3/3 通过。**

---

## 五、高危问题

### SEC-02 · 每个请求都做一次全量权重安全校验，且无结果缓存

- **位置**：[v2_compat.py L254-L258](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L254-L258)、[v2_compat.py L451](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L451)、[gsv_engine.py L160](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L160)、[gsv_engine.py L240-L241](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L240-L241)
- **证据**：只要请求带了 `t2s_weights`/`vits_weights`，就在每次请求的线程池里执行 `_validate_weights_file`；传入的是同一个已加载过的路径也照跑。快路径 `torch.load(weights_only=True)` 会把整个权重反序列化一次；慢路径 `raw = f.read()` + `zipfile.BytesIO(raw)` 峰值约 **2× 文件体积**（GSV 权重常见 500MB~1.5GB）。而 `_ensure_weights()`（[L728-L755](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L728-L755)）已有 `(路径, mtime, size)` 指纹做幂等跳过 —— 校验层没有复用这套指纹。
- **影响**：稳态调用（客户端习惯每次带权重路径）下，每个请求白付数百 MB 读盘 + 一次完整反序列化，内存峰值翻倍；并发 2~3 个请求即可 OOM。这同时是一条**无需鉴权的放大攻击面**（配合 SEC-01）。
- **验证**：正确性 ✓（无缓存，且校验点在请求入口）；影响 ✓（高频路径，内存倍数可算）；成本 ✓（复用 `weights_stamp` 做 1 次校验/路径即可）。
- **3/3 通过。**
- **状态：已修复（批次 87）** — `gsv_engine.validate_weights_file()` 先取 `weights_stamp`（路径 + mtime_ns + size）查 `_WEIGHTS_CHECKED`（`OrderedDict` + 锁，FIFO 上限 256），命中直接返回；校验通过后登记指纹，文件被替换即指纹变化自动重校验（与 `_ensure_weights` 幂等分支同语义）。回归用例 `test_validate_weights_file_cached_by_stamp`。

---

### SEC-03 · 非 GSV 入口仍使用 `weights_only=False` / `allow_pickle=True`，构成 RCE 面

- **位置**：[inference/pipeline.py L140](file:///e:/新创意构思/新建文件夹/ADR/adr/inference/pipeline.py#L140)、[training/trainer.py L517](file:///e:/新创意构思/新建文件夹/ADR/adr/training/trainer.py#L517)、[vocoder/bigvgan.py L209](file:///e:/新创意构思/新建文件夹/ADR/adr/vocoder/bigvgan.py#L209)、[data/pipeline.py L93](file:///e:/新创意构思/新建文件夹/ADR/adr/data/pipeline.py#L93)、[webui/__init__.py L1124-L1150](file:///e:/新创意构思/新建文件夹/ADR/adr/webui/__init__.py#L1124-L1150)、[eval/speaker_sim.py L35](file:///e:/新创意构思/新建文件夹/ADR/adr/eval/speaker_sim.py#L35)
- **证据**

  ```python
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)   # inference:140
  ckpt = torch.load(path, map_location="cpu", weights_only=False)              # trainer:517
  ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)           # bigvgan:209
  data = np.load(path, allow_pickle=True)                                      # data:93
  ```

  WebUI 侧 `ckpt_value` 允许用户手输任意路径，`full_ckpt` 只做 `exists()` 检查（[L1132-L1139](file:///e:/新创意构思/新建文件夹/ADR/adr/webui/__init__.py#L1132-L1139)），随后作为 `-c` 传给 `adr.cli clone`，最终落到 `inference/pipeline.py:140` 的 `weights_only=False`。
- **影响**：GSV 入口已有 `validate_weights_file` 收口（安全），但自研引擎/声码器/resume/数据集四条路径没有；一旦受污染权重或 `.npz` 被放到数据目录（音色包导入、训练素材上传、共享盘同步），加载即执行任意代码。属于"信任边界内仍有未统一的口子"。
- **验证**：正确性 ✓（5 处代码明写）；影响 ✓（导入链路确实可达，WebUI 允许任意绝对路径）；成本 ✓（统一到已有的 `_restricted_weights_scan` / `weights_only=True` 兼容层，工作量中等但可分批）。
- **3/3 通过。**
- **状态：已修复（批次 87）** — `gsv_engine` 新增 `safe_torch_load()`（先 `weights_only=True`，失败后受限扫描 pickle GLOBAL 再 `weights_only=False`），白名单新增 `allow_globals` 参数精确放行本仓 dataclass；`inference/pipeline.py`（`TrainerConfig` / `InferConfig`）、`training/trainer.py`、`vocoder/bigvgan.py` 统一改走该入口；`eval/speaker_sim.py` 纯 state_dict 直接 `weights_only=True`；`data/pipeline.py` 的 npz 默认 `allow_pickle=False`（新写入改字符串数组，历史 object 数组文件回落 `allow_pickle=True` 并告警）。WebUI `ckpt_value` 未加路径白名单（仍属信任边界外的"用户手输路径"），但其落地加载已受限。回归用例 `test_restricted_scan_*` / `test_safe_torch_load_roundtrip_and_allow_globals`。

---

### SEC-04 · 监听 `0.0.0.0` 且未配置 API Key 时只告警不拒绝

- **位置**：[server/\_\_main\_\_.py L71-L91](file:///e:/新创意构思/新建文件夹/ADR/adr/server/__main__.py#L71-L91)、[L138](file:///e:/新创意构思/新建文件夹/ADR/adr/server/__main__.py#L138)、[L152](file:///e:/新创意构思/新建文件夹/ADR/adr/server/__main__.py#L152)、[docker/Dockerfile L128](file:///e:/新创意构思/新建文件夹/ADR/docker/Dockerfile#L128)
- **证据**：`_warn_unauthenticated_listen()` 仅 `logger.warning` 后 `return`；默认 `--addr 127.0.0.1`，但 Dockerfile 固定 `CMD ["--addr","0.0.0.0","--port","9881"]`，桌面壳"局域网开关"也直接注入 `0.0.0.0`。
- **影响**：镜像开箱即"全网可写"——任何人可调用 `/tts`（消耗 GPU）、`/set_*_weights`（换模型）、`/profiles/import`（写文件），并可触发 SEC-01/CON-01 这两个打挂服务的面。这是一条"默认部署即可利用"的攻入口。
- **验证**：正确性 ✓（只告警）；影响 ✓（容器默认参数在代码里）；成本 ✓（`0.0.0.0 + 无 key` 时直接退出，或强制 key，改动 3 行）。
- **3/3 通过。**
- **状态：已修复（批次 87）** — `_warn_unauthenticated_listen()` 对 `0.0.0.0` / `::` 且无 `ADR_TTS_API_KEY` 默认打印"拒绝启动"并 `SystemExit(2)`；显式逃生阀 `ADR_ALLOW_UNAUTHENTICATED=1`（`1/true/yes/on`）或 `from_shell=True` 时仍仅告警。`docker/Dockerfile` 显式 `ENV ADR_ALLOW_UNAUTHENTICATED=1` 保留容器默认可用（`--addr 0.0.0.0` 在命名空间内必需），发布到公网仍需设 key；桌面壳在"对外服务"开关打开时注入同一变量。回归用例 `test_refuses_unauthenticated_expose` / `test_warn_fires_when_explicit_opt_out` / `test_warn_fires_from_shell`。

---

### BUG-01 · 训练样本数 ≥2 但训练集 < batch_size 时静默空转

- **位置**：[training/dataset.py L316-L334](file:///e:/新创意构思/新建文件夹/ADR/adr/training/dataset.py#L316-L334)、[training/trainer.py L163-L192](file:///e:/新创意构思/新建文件夹/ADR/adr/training/trainer.py#L163-L192)
- **证据**

  ```python
  # dataset.py:325 只拦 <2
  if len(indices) < 2: raise ValueError(...)
  n_val = min(max(1, int(len(indices) * val_ratio)), len(indices) - 1)
  # trainer.py:169
  drop_last=True
  # trainer.py:191
  total_steps = math.ceil(len(self.train_loader) / grad_accum) * epochs
  ```

  例如 3 条样本、`val_ratio=0.1` → train=2、`batch_size=4` → `len(train_loader)==0` → `total_steps=0`。`build_scheduler` 对 0 步不会报错（[optimizer.py L111-L121](file:///e:/新创意构思/新建文件夹/ADR/adr/training/optimizer.py#L111-L121) 有 `max(1, ...)`），训练循环内层直接不执行 → **跑完 epochs 但一步没更新**，日志里仍显示 epoch 进度。
- **影响**：小样本微调（正好是本项目的目标场景）会得到一个"训练成功但权重没变"的产物，排查成本极高；`dataset.py` 的注释本身已承认这种"静默空转"机制。
- **验证**：正确性 ✓（`drop_last` + 0 steps 无异常）；影响 ✓（3~5 条素材很常见）；成本 ✓（构造 `Trainer` 时断言 `len(train_loader)>0` 并给出明确提示，低成本）。
- **3/3 通过。**
- **状态：已修复（批次 87）** — `Trainer.__init__` 在优化器构建前检查 `len(self.train_loader) == 0`，直接 `ValueError`（含 `samples` / `batch_size` / `drop_last=True` 说明与调整建议），不再跑完 epochs 却零更新。回归用例 `test_trainer_rejects_zero_step_loader`。

---

### BUG-02 · 桌面端 stdout/stderr 管道可能无人消费，子进程写满缓冲后阻塞

- **位置**：[desktop/src-tauri/src/lib.rs L1096-L1130](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/src/lib.rs#L1096-L1130)
- **证据**

  ```rust
  cmd.stdout(std::process::Stdio::piped());   // L1096
  cmd.stderr(std::process::Stdio::piped());   // L1097
  ...
  if let Ok(file) = OpenOptions::new().create(true).append(true).open(&log_path) {
      if let Some(out) = child.stdout.take() { spawn(pump_server_log(out, ...)) }   // L1123
      if let Some(err) = child.stderr.take() { spawn(pump_server_log(err, ...)) }   // L1127
  }   // 打开失败 → 没有任何 take()，也没有 drop 后的读线程
  ```
- **影响**：日志目录不可写（只读安装目录、磁盘满、杀软锁定）时，子进程持有的两个管道没有任何读者；Python 服务端的 stdout 一旦累计到管道缓冲（Windows 约 64KB），`print`/日志/`uvicorn` 访问日志会直接阻塞，表现为"启动卡住且无日志"。这条与 `server_log_path()` 的失败分支叠加，正是最难现场排查的一类问题。
- **验证**：正确性 ✓（`if let Ok` 之外无消费者）；影响 ✓（只读安装目录是常见部署）；成本 ✓（无论日志是否打开都 `take()` 并起泵，泵内写失败则丢弃）。
- **3/3 通过。**
- **状态：已修复（批次 87）** — `pump_server_log()` 的日志句柄改为 `Option<Arc<Mutex<File>>>`，打开失败也能工作（写失败即丢弃）；`start_server()` 在日志文件打开与否都无条件 `take()` stdout/stderr 并各起一个泵，管道恒被抽干，子进程不会再因缓冲写满而阻塞。

---

### PERF-01 · 每请求在运行时全局锁内读盘（`record → _evaluate → policy.load`）

- **位置**：[core/runtime.py L71-L83](file:///e:/新创意构思/新建文件夹/ADR/adr/core/runtime.py#L71-L83)、[core/runtime.py L85-L97](file:///e:/新创意构思/新建文件夹/ADR/adr/core/runtime.py#L85-L97)、[core/policy.py L112-L120](file:///e:/新创意构思/新建文件夹/ADR/adr/core/policy.py#L112-L120)
- **证据**

  ```python
  with _lock:                      # runtime.py:76
      _samples.append(...)
      _evaluate()                  # L80
  # runtime.py:88 默认 ADR_RUNTIME_AUTOTUNE=1，样本累计 ≥ MIN_SAMPLES 后每次都评估
  # policy.py:117
  return json.loads(path.read_text(encoding="utf-8"))
  ```
- **影响**：合成完成的回调在**全局运行时锁**内做文件 IO（策略文件每次重新读），高并发下所有请求的收尾阶段串行等磁盘；策略文件放在网络盘/杀软扫描目录时更明显。这条同时把"每请求一次读盘"放大到"每请求一次全局串行"。
- **验证**：正确性 ✓（锁内调用确实存在，且默认开启）；影响 ✓（每次合成都会走 `record`）；成本 ✓（`policy.load()` 加 mtime 缓存，或把评估移到锁外的独立线程）。
- **3/3 通过。**
- **状态：已修复（批次 87）** — `runtime.record()` 的 `_lock` 只包 `_samples.append`；评估改用独立 `_eval_lock`（`acquire(blocking=False)`，评估中不排队、不拖住请求）；`policy.load()` 新增 `(path, mtime_ns, size)` 缓存，命中不读盘，文件改写自动重读。回归用例 `test_record_evaluates_outside_global_lock` / `test_record_skips_eval_when_another_running` / `test_load_caches_by_stamp`。

---

### PERF-02 · async 处理函数内直接调用同步 ffprobe，最长阻塞事件循环 30 秒

- **位置**：[server/console.py L627](file:///e:/新创意构思/新建文件夹/ADR/adr/server/console.py#L627)、[server/console.py L536-L562](file:///e:/新创意构思/新建文件夹/ADR/adr/server/console.py#L536-L562)
- **证据**

  ```python
  async def train_upload(...):
      ...
      probe = _probe_audio(dest)      # L627  同步函数
  # L557-L561
  r = subprocess.run(["ffprobe", ...], capture_output=True, text=True, timeout=30)
  ```
- **影响**：上传一个损坏/超大的音频时，事件循环被同步子进程占住最长 30s；期间 `/health`、`/tts`、WebSocket 全部无响应，桌面壳会误判服务卡死。项目其它地方已统一用 `asyncio.to_thread`（共 14 处），这里是漏网的一处。
- **验证**：正确性 ✓（`_probe_audio` 是普通 `def`，直接 await 上下文调用）；影响 ✓（上传是不可信输入，超时必现）；成本 ✓（一行 `await asyncio.to_thread(_probe_audio, dest)`）。
- **3/3 通过。**
- **状态：已修复（批次 87）** — `train_upload` 改为 `probe = await asyncio.to_thread(_probe_audio, dest)`，ffprobe 子进程不再占用事件循环。

---

## 六、中危问题

### SEC-05 · 音色包 `overwrite=true` 会替换全局共享权重

- **位置**：[models/voice_package.py L122-L129](file:///e:/新创意构思/新建文件夹/ADR/adr/models/voice_package.py#L122-L129)、[L243-L266](file:///e:/新创意构思/新建文件夹/ADR/adr/models/voice_package.py#L243-L266)、[server/native.py L233-L235](file:///e:/新创意构思/新建文件夹/ADR/adr/server/native.py#L233-L235)
- **证据**：`_dst_dir("t2s")` 固定 `GSV_DIR/GPT_weights_v2`、`"vits"` 固定 `SoVITS_weights_v2`（**所有档案共用**）；非 overwrite 时按 `dst.exists() + sha256` 拒绝冲突，`overwrite=true` 时直接在两段式提交里替换目标文件。
- **影响**：导入一个同名权重的第三方音色包，会把**其它音色正在使用的权重**换掉（音色串台），且不清理 `tts_cache`/段缓存（见 CACHE-01），旧音频继续被命中。恶意包只需与热门权重同名。
- **验证**：正确性 ✓；影响 ✓（共享目录 + 覆盖开关都在代码里）；成本 ✓（覆盖前做 sha 比对并提示"将影响 N 个档案"）。
- **3/3 通过。**

---

### SEC-06 · 请求体总大小无上限，文本长度限制在 JSON 解析之后

- **位置**：[server/v2_compat.py L161-L181](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L161-L181)
- **证据**：`await request.json()` 先完成（Starlette 默认无 body 上限），随后 `cap = max_text_chars()` 才比对 `len(req["text"])`。带 `aux_ref_audio_paths` 等数组字段的请求同样不受体积约束。
- **影响**：单请求可提交数百 MB JSON，解析峰值内存与 SEC-01 的读文件叠加；属于 DoS 面而非数据泄露。
- **验证**：正确性 ✓；影响 ✓（无需鉴权即可打）；成本 ✓（ASGI 中间件按 `Content-Length` 拒绝，或 `receive` 累积计数）。
- **3/3 通过。**

---

### SEC-07 · 更新下载只有连接超时，没有总时长/大小上限

- **位置**：[desktop/src-tauri/src/update.rs L450-L496](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/src/update.rs#L450-L496)、[L331-L333](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/src/update.rs#L331-L333)
- **证据**

  ```rust
  let agent = AgentBuilder::new().timeout_connect(Duration::from_secs(15)).build();  // L458-L460
  let mut got: u64 = 0;
  loop { let n = reader.read(&mut buf)?; ... got += n; }   // 无总超时、无上限
  ```

  安装前的 sha256 校验（[L521-L523](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/src/update.rs#L521-L523)）与 `install_update` 的目录/扩展名/sha 重算（[L659-L682](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/src/update.rs#L659-L682)）是完整的，因此**不是**任意代码执行问题；问题在下盘阶段。
- **影响**：清单指向慢速/无限流（或 `content-length` 缺失）时，客户端可无限写入磁盘、无限挂起更新线程；用户侧表现为"更新卡死在 0%"。
- **验证**：正确性 ✓；影响 ✓（清单可被劫持/服务端异常）；成本 ✓（`timeout_read` + `got > size*1.1` 拒绝，2 处）。
- **3/3 通过。**

---

### BUG-03 · 训练素材上传按"秒级时间戳 + 文件名"落盘，同秒同名互相覆盖

- **位置**：[server/console.py L600-L606](file:///e:/新创意构思/新建文件夹/ADR/adr/server/console.py#L600-L606)
- **证据**：`fname = f"{datetime.now():%Y%m%d_%H%M%S}_{stem}{ext}"`，直接 `dest.open("wb")` 截断写。100ms 内两次上传 `voice.wav` 即写同一路径。
- **影响**：并发/批量上传时静默丢素材；由于回滚逻辑会 `unlink`（L617-L620），后一个失败请求还可能删掉前一个成功请求的文件。
- **验证**：正确性 ✓；影响 ✓（批量导入是真实用法）；成本 ✓（加 `uuid4`/`os.O_EXCL`）。
- **3/3 通过。**

---

### BUG-04 · 档案加载把所有异常统一转成"档案不存在(404)"

- **位置**：[services/synthesis.py L53-L62](file:///e:/新创意构思/新建文件夹/ADR/adr/services/synthesis.py#L53-L62)
- **证据**

  ```python
  try:
      meta = voice_library.load_voice(name)
  except Exception:
      raise ProfileNotFoundError(name)
  ```

  权限被拒、`meta.json` 损坏、JSON 语法错误、软链断裂全部被吞成 404。
- **影响**：运维看到 404 会去检查名字，而真实原因是磁盘/权限/损坏；与 BUG-05 叠加时形成"报不存在但其实坏了"的诊断陷阱。
- **验证**：正确性 ✓；影响 ✓（`load_voice` 会做 IO 与 JSON 解析）；成本 ✓（只捕获 `FileNotFoundError`，其余上抛并映射 500）。
- **3/3 通过。**

---

### BUG-05 · 模块级 `is_ready()` / `stage()` 不看降级标记，卡死后仍报 ready

- **位置**：[models/gsv_engine.py L1108-L1117](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L1108-L1117)、对照 [L508-L512](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L508-L512)
- **证据**

  ```python
  @property
  def is_ready(self): return self._tts is not None and not self._stalled   # L509 正确
  def is_ready():     return _ENGINE is not None and _ENGINE._tts is not None   # L1108 漏判
  ```
- **影响**：CON-01 触发后，`/health`、预热页、桌面壳探活都认为"引擎就绪"，于是持续放流量进来，全部挂住；排障时看不到降级信号。
- **验证**：正确性 ✓（两处判定不一致，静态可见）；影响 ✓（探活是壳的决策依据）；成本 ✓（一行 `and not _ENGINE._stalled`）。
- **3/3 通过。**

---

### CON-02 · 流式非 WAV 格式"每块一次 ffmpeg"，且每块是独立容器

- **位置**：[services/synthesis.py L155-L173](file:///e:/新创意构思/新建文件夹/ADR/adr/services/synthesis.py#L155-L173)、[server/audio_codec.py L96-L171](file:///e:/新创意构思/新建文件夹/ADR/adr/server/audio_codec.py#L96-L171)
- **证据**

  ```python
  mt = media_type
  for chunk, sr in stream_chunks(...):
      if first and mt == "wav": yield wave_header_chunk(sr); mt = "raw"; first = False
      yield pack_audio(BytesIO(), to_int16(chunk), sr, mt).getvalue()   # L173 每块一次
  ```

  只有 WAV 会切成"头 + 裸 PCM"；`ogg/aac/mp3` 每个 chunk 都走一次完整的 `ffmpeg` 子进程，产出各自带容器头的独立文件。
- **影响**：① 每块数百毫秒的进程启动成本，流式首包延迟与 RTF 都被拖高，CPU 占用随块数线性上升；② 下游若把字节流按顺序拼接，得到的是"多个 OGG/MP3 文件首尾相连"，不是合法单流（播放器可能中断/跳帧 —— 具体行为 **推测**，需按客户端实测）。
- **验证**：正确性 ✓（分支与调用都在代码里）；影响 ✓（非 WAV 是默认推荐格式之一，调用量真实）；成本 ✓（改为长驻编码器或先用 WAV 流、由客户端转码）。
- **3/3 通过。**

---

### PERF-03 · `compute_mel_bigvgan` 每次调用都重建 mel 滤波器组与 Hann 窗

- **位置**：[utils/audio.py L332-L336](file:///e:/新创意构思/新建文件夹/ADR/adr/utils/audio.py#L332-L336)、[data/pipeline.py L245-L262](file:///e:/新创意构思/新建文件夹/ADR/adr/data/pipeline.py#L245-L262)
- **证据**

  ```python
  mel_basis = torch.from_numpy(librosa.filters.mel(sr=..., n_fft=..., n_mels=...)).float().to(device)
  hann = torch.hann_window(win_length).to(device)
  ```

  每次调用都重新计算滤波器组并做一次 CPU→GPU 拷贝。`DataPipeline` 对每个 slice 逐个调用（L252 附近）。
- **影响**：一次 30 分钟素材的预处理会重复数百次相同的滤波器构造与 H2D 拷贝；训练数据构建是 CPU/IO 密集，这部分是稳定的白开销（单次约几十 ms 量级，N 次后可观）。
- **验证**：正确性 ✓（无缓存）；影响 ✓（逐 slice 调用）；成本 ✓（按 `(sr,n_fft,n_mels,fmin,fmax,device)` 缓存，`lru_cache` 即可）。
- **3/3 通过。**

---

### PERF-04 · v3 文本缓冲用字符串自增拼接，O(n²)

- **位置**：[server/v3_compat.py L57-L86](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v3_compat.py#L57-L86)
- **证据**：`self._buf += text`（L64）、`current += ch`（L78），且 `extract_sentences()` 每次对**整个缓冲区**重新扫描（L69）后才截断。
- **影响**：长文本按碎片持续追加时，拼接拷贝与全量重扫都是二次复杂度；WebSocket 长会话下延迟随文本长度平方增长。真实会话通常是短句，故列为中危而非高危。
- **验证**：正确性 ✓；影响 ✓（WebSocket 是长连接流式输入，碎片化正是其设计前提）；成本 ✓（`list` + join 或 `bytearray`，局部改动）。
- **3/3 通过。**

---

### DATA-01 · 训练 mel 的 hop 取自 F0 配置，推理取自模型配置；脚本仍用 v1 mel

- **位置**：[data/pipeline.py L252-L256](file:///e:/新创意构思/新建文件夹/ADR/adr/data/pipeline.py#L252-L256)、[inference/pipeline.py L345](file:///e:/新创意构思/新建文件夹/ADR/adr/inference/pipeline.py#L345)、[utils/audio.py L247-L297](file:///e:/新创意构思/新建文件夹/ADR/adr/utils/audio.py#L247-L297)
- **证据**

  ```python
  # data/pipeline.py:255
  s.mel = compute_mel_bigvgan(..., n_mels=80, hop_length=self.config.f0.hop_length)
  # inference/pipeline.py:345
  ... hop_length=self.config.hop_length
  ```

  默认都是 256，但两者来自不同配置对象（`PipelineConfig.f0.hop_length` vs 模型的 `config.hop_length`）。一旦用户改 F0 的 hop（例如为了 pyin 提速），训练 mel 帧格改变而推理不变，声码器域错位且**无任何报错**。另外 `scripts/` 中仍有调用 v1 `compute_mel` 的路径，与生产统一口径不符。
- **影响**：训练成果上线后音质异常、且难以定位（两者都"能跑"）。
- **验证**：正确性 ✓；影响 ✓（配置分叉是显式允许的）；成本 ✓（训练侧改为从模型/声码器配置取 hop，或启动时断言相等）。
- **3/3 通过。**

---

### API-01 · `streaming_mode=3` 与 `2` 完全等价，与文档约定不一致

- **位置**：[server/v2_compat.py L276-L290](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L276-L290)、[docs/call.html L191 附近](file:///e:/新创意构思/新建文件夹/ADR/docs/call.html)
- **证据**

  ```python
  if mode == 2 or mode == 3:      # 同一分支，无任何差异处理
      return StreamingResponse(...)
  ```

  文档/前端把 3 描述为"真流式 + 定长块"，实际与 2 走同一路径（定长块参数 `min_chunk_length` 未被用于 mode 3 语义）。
- **影响**：按文档调 `streaming_mode=3` 的客户端会拿到与 2 相同的块边界，若其播放器依赖定长块会得到抖动/空洞；属于**契约不一致**，不是崩溃类问题。
- **验证**：正确性 ✓；影响 ✓（公开参数 + 文档承诺）；成本 ✓（二选一：实现定长块，或把 3 标为 2 的别名并改文档）。
- **3/3 通过。**

---

### API-02 · `tts_cache` 的 key 有意排除 `seed`，显式不同 seed 返回同一音频

- **位置**：[server/tts_cache.py L7](file:///e:/新创意构思/新建文件夹/ADR/adr/server/tts_cache.py#L7)、[L45-L51](file:///e:/新创意构思/新建文件夹/ADR/adr/server/tts_cache.py#L45-L51)、[L79-L83](file:///e:/新创意构思/新建文件夹/ADR/adr/server/tts_cache.py#L79-L83)
- **证据**：模块 docstring 写明"seed 不入 key — 播报一致性优先"；`_KEY_FIELDS` 不含 `seed`。
- **影响**：GET `/tts?seed=1` 与 `seed=2` 命中同一条缓存，调用方无法通过 seed 获得可复现的差异输出；而段缓存（`_seg_cache_key`）**包含** seed，两套缓存语义相反，换层时行为会突变。
- **验证**：正确性 ✓；影响 ✓（`seed` 是公开参数，A/B 与复现需求直接受影响）；成本 ✓（若确认按"一致性优先"是有意设计，则改为文档声明并删除公开 seed 语义；否则 1 行加入 key）。
- **3/3 通过。**

---

### CACHE-01 · 段缓存换权后不清空，key 只含权重**路径**不含指纹

- **位置**：[models/gsv_runtime.py L38-L51](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_runtime.py#L38-L51)、[models/gsv_runtime.py L83-L86](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_runtime.py#L83-L86)、[server/v2_compat.py L476-L478](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L476-L478)
- **证据**

  ```python
  payload = [seg, ref_audio, prompt_text, ..., t2s_weights, vits_weights, list(ref_stamp), seed]
  ```

  只有参考音频有 `ref_stamp`（路径+mtime+size）；权重只放路径。换权成功处只 `tts_cache.clear()`。`_seg_cache_clear()` 在 `adr/` 生产代码中**零调用**（仅 `gsv_engine` re-export 供测试使用，已全仓 grep 确认）。
- **影响**：训练产物覆盖同路径权重、或音色包 overwrite 后，段缓存（默认 128 条/256MB）继续返回旧人声；重启才消失。流式路径优先命中段缓存（[gsv_engine.py L909-L923](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L909-L923)），所以影响面是流式。
- **验证**：正确性 ✓（key 构成与调用点均可证）；影响 ✓（换权/导入是常规操作）；成本 ✓（`_ensure_weights` 换权分支调用 `_seg_cache_clear()`，一行）。
- **3/3 通过。**

---

## 七、低危问题

### DATA-02 · F0 与 mel 的帧格定义不一致，训练时左对齐拼接

- **位置**：[data/f0.py L85-L92](file:///e:/新创意构思/新建文件夹/ADR/adr/data/f0.py#L85-L92)、[utils/audio.py L338-L345](file:///e:/新创意构思/新建文件夹/ADR/adr/utils/audio.py#L338-L345)、[training/dataset.py L173-L175](file:///e:/新创意构思/新建文件夹/ADR/adr/training/dataset.py#L173-L175)
- **证据**：`librosa.pyin(...)` 未传 `center`（默认 `True`，帧中心在 `t*hop`）；`compute_mel_bigvgan` 显式 `center=False` 并左右 reflect pad `(n_fft-hop)//2 = 384`，帧中心相对原信号偏移约 `384 - 512 ≈ 128 samples`（22050Hz 下约 5.8ms）。collate 时 `f0[i,:T] = s.f0[:T]` 直接左对齐，无重采样/移位。
- **影响**：音高条件与声学帧存在约半帧的系统性偏移；短音素边界上可能轻微错位。幅度小（≈5.8ms），列为低危质量问题，不建议单独排期，随 DATA-01 一起处理即可。
- **验证**：正确性 ✓（两侧帧定义可静态确认）；影响 ✓（系统性、非随机）；成本 ✓（collate 里 `np.interp` 到 mel 帧数并移半帧）。
- **3/3 通过。**

---

### OPS-01 · `audit.log` 每事件打开一次，无轮转、无进程内锁

- **位置**：[core/audit.py L29-L43](file:///e:/新创意构思/新建文件夹/ADR/adr/core/audit.py#L29-L43)
- **证据**：`with open(log_dir / "audit.log", "a", ...)`，无 `RotatingFileHandler`、无 `threading.Lock`（多线程追加依赖 OS 原子性，但不保证多条记录不交错）。
- **影响**：长期运行单文件无限增长；多个 worker/线程并发写可能产生半行；审计失败被 `except Exception` 静默吞掉，磁盘满时"以为有日志其实没有"。
- **验证**：正确性 ✓；影响 ✓（审计默认开启，见 `settings.safety_audit_enabled()`）；成本 ✓（进程内单 `Lock` + 按大小轮转，30 行内）。
- **3/3 通过。**

---

## 八、DeadCode：死代码与简化机会（量化）

### 8.1 量化口径与结果

统计范围：`adr/**/*.py`，83 个文件、20,847 行；AST 提取**模块级公开符号**（`def`/`class`/全大写常量），再对全仓（含 `tests/`、`scripts/`、`desktop/`）做引用计数。

| 指标 | 数值 |
|---|---|
| 模块级公开符号总数 | **438**（函数 260 / 类 94 / 常量 84） |
| 仅在本文件内出现 | 107 |
| 其中装饰器存活（FastAPI 路由 / Click 命令，由框架反射调用） | 44 |
| → 非装饰器、无外部引用的符号 | **63** |
| 全仓零引用（含定义行本身） | 43 |
| 其中装饰器存活 | 37 |
| → **真正无引用的死符号** | **6**（另有 1 个公共 API 见下） |
| 完全零消费者的模块 | **0** |
| 仅被脚本/测试消费的模块 | **3**（`adr/eval/probes.py`、`adr/cli.py`、`adr/server/__main__.py`） |
| `__init__.py` 导出 | 11 个文件共 **56** 个名字，其中 **40 个（71%）** 无包级消费者 |

**零引用符号清单（逐个复核，计数含定义行 = 1）**

| 符号 | 位置 | 判定 |
|---|---|---|
| `get_version` | `adr/__init__.py:37` | 公共便利 API，内部零调用；可保留但应标注为对外 API，否则删除 |
| `get_free_vram_gb` | `adr/core/device.py:258` | 内部死代码，可删 |
| `print_memory_usage` | `adr/core/device.py:266` | 内部死代码，可删 |
| `build_sovits_preset` | `adr/models/sovits.py:69` | 内部死代码，可删 |
| `slice_audio_by_time` | `adr/utils/audio.py:368` | 内部死代码，可删 |
| `pad_or_trim` | `adr/utils/audio.py:380` | 内部死代码，可删 |
| `_USE_REAL_DEFAULT` | `adr/training/callbacks.py:633` | 常量零引用，可删 |

另有 1 个"生产不读、仅测试读"的兼容属性 `_asr_cpu_fallback`（[callbacks.py L744](file:///e:/新创意构思/新建文件夹/ADR/adr/training/callbacks.py#L744)，仅 `tests/test_asr_callback.py:431/458` 断言）——生产逻辑已改用 `_asr_cpu_fallback_set`，该属性属于遗留兼容层。

### 8.2 重复实现

| 重复项 | 位置 | 说明 |
|---|---|---|
| `is_safe_name` | `models/voice_library.py:27` / `server/pathsafe.py:37` | 同一白名单校验两份实现（含 `adr/` 19 处引用、`tests/` 11 处） |
| `_mel_to_wav` | `training/callbacks.py:360` / `:748` | 两个回调类各写一份 |
| mel 算法两套 | `utils/audio.py:247`（v1）/ `:300`（bigvgan） | 生产已统一 bigvgan，v1 仍被脚本/测试引用 |
| 目录包含校验 | `voice_library.py:161-L165` / `pathsafe.resolve_within:17-L34` | 逻辑等价 |
| 缓存淘汰 | `tts_cache` LRU（内存/磁盘两套记账） / `gsv_runtime` LRU | 两套独立实现，语义不同（一个含磁盘条目） |

### 8.3 不可达/近不可达代码

- `training/callbacks.py` 约 **1,000 行**（`WavQualityCallback` L164、`ASRCallback` L586、`WarmRestartCallback` L1072）在产品入口（`adr train` / console）**未注册**：[cli.py L574-L700](file:///e:/新创意构思/新建文件夹/ADR/adr/cli.py#L574-L700) 只挂 Progress/Checkpoint/Logger。这些类仅测试与 `scripts/` 可达。
- 死分支：`v2_compat.py:283-L286`（mode 3 与 2 同一路径，见 API-01）。
- 死常量/死属性：`callbacks.py:633`、`callbacks.py:744`。

### 8.4 其它简化机会

- `except Exception` **137 处**、`except BaseException` **18 处**、`noqa` **15 处**；`TODO/FIXME/HACK` 为 **0**（说明注释治理到位，但也意味着历史欠账没有显式标记）。
- `print(` **53 处**（CLI/脚本可接受，服务路径应统一走 logger）。
- `adr/server/v3_compat.py` 的 `_TextBuffer` 与 `adr/models/gsv_coalescer.py` 的缓冲逻辑高度相似，可合并。

---

## 九、TestGaps：测试覆盖缺口（量化）

### 9.1 规模

| 指标 | 数值 |
|---|---|
| 测试文件 | 59 |
| `def test_` 函数 | **781** |
| pytest 实际收集（含 parametrize 展开，已实跑 `--collect-only`） | **854** |
| 测试代码行 | 13,913 |
| 源码行（`adr/`） | 20,847 |
| 测试/源码行比 | ≈ **1.76 : 1** |
| 测试/公开函数比（781 / 260） | ≈ **3 : 1** |
| `xfail` 标记 | 0 |
| CI 中永不执行的 skip | 2（`test_audit_packaging.py:54`、`test_server_security.py:54`，均 Windows-only，而 CI 仅 ubuntu） |
| Rust 单测 | 19（`update.rs` 9 + `lib.rs` 10），**CI 完全不跑** |

### 9.2 高风险但无直接测试的代码（按风险排序）

| 代码 | 位置 | 缺口 |
|---|---|---|
| 流式 worker 生命周期 / 断连 / 卡死降级 | `gsv_engine.py:868-L986` | 只有 mock 级线程测试（`test_gsv_stream_threading.py:37-L50` 用假 `run`），无"上游永卡"场景，CON-01 因此长期未被发现 |
| `validate_weights_file` 慢路径 | `gsv_engine.py:238-L260` | 无超大文件/UNC/截头 zip 的边界用例 |
| `_seg_cache_key` 的权重维度 | `gsv_runtime.py:38-L51` | 无"同路径换权重"用例（CACHE-01 逃逸） |
| `gsv_bootstrap.ensure` 并发锁 | `gsv_bootstrap.py:149-L291` | 无多线程同时下载/校验失败回滚用例 |
| `voice_package` 完整性失败路径 | `voice_package.py:200-L330` | 仅 happy path + 简单 rollback，无"提交中途断电/部分替换" |
| `_split_long` / `_stream_head_split` | `gsv_engine.py:768-L810` | 无无标点长文、超长英文词用例 |
| `update.rs` 下载/安装链 | `update.rs:450-L682` | 19 个 Rust 单测覆盖纯函数，网络/落盘/安装全未测且 CI 不跑 |
| `is_cuda_oom` / `handle_cuda_oom` | 全仓 | 无真实 OOM 异常样例 |
| `utils/audio` 四个 helper | `audio.py:352-L400` | `get_audio_info`/`slice_audio_by_time`/`pad_or_trim` 无测试 |
| `v2._stream_generator` 客户端断连 | `v2_compat.py:200-L241` | 无"生成器被 cancel"用例 |

### 9.3 "只测 mock 形状"的用例（覆盖假阳性）

- `test_gsv_stream_threading.py:37-L50`：用假 `run` 模拟流式，未覆盖真实 `_lock` 竞争。
- `test_gsv_coalescer.py:295-L301`、`test_audit_engines.py:301-L327`。
- `test_audit_server_compat.py:110-L116`、`test_audit_update_shell.py:29-L34`：直接读源码字符串做断言（"源码里存在某行"），重构即失败、行为回归却可能漏过。
- `test_audit_training.py:87-L94`：只断言配置字段，不验证训练一步。

### 9.4 CI 口径

[.github/workflows/smoke.yml](file:///e:/新创意构思/新建文件夹/ADR/.github/workflows/smoke.yml) 仅 `ubuntu-latest + pytest`：

- 注释仍写"431+ 基线"，实际 781/854 —— 口径过时。
- 不跑 Rust（`cargo test` 缺位）、不跑 Windows、不跑 GPU 路径。
- 覆盖率无门槛（未生成 `coverage.xml`，未设 fail-under）。

---

## 十、优先级行动清单

### P0 —— 立即修（阻断发布）

| # | 条目 | 动作 | 验收 |
|---|---|---|---|
| 1 | SEC-01 ✅ 批次86 | GET `/tts` 挂 `require_browser_safe`；权重预检移到参数校验后；只允许档案目录内路径；对超大文件先做 `stat` 上限 | 无 key 时跨站 GET 返回 403；`?t2s_weights=//server/share/x` 返回 400 而非读盘 |
| 2 | CON-01 ✅ 批次86 | `q.get()` 加超时；超时即判定 stalled 并主动关闭生成器；`_lazy_init` 用带超时的 `acquire` 或独立重建锁 | 模拟上游永卡后，第 2 个请求在 N 秒内返回 503，进程不假死 |

> 批次 86 落地口径：权重校验保留「请求级路径」语义（未收窄到档案目录，避免破坏 `GPT_weights_v2/xxx.ckpt` 与 N.E.K.O 配置契约），改为 UNC/设备路径拒绝 + `is_file()` + 大小上限；CON-01 额外把调度准入 `slot()` 改为可超时，否则 N=1 时第二个请求会卡在准入而非引擎锁上。

### P1 —— 本周

| # | 条目 | 动作 |
|---|---|---|
| 3 | SEC-02 ✅ 批次87 | 用 `weights_stamp` 缓存校验结果（每路径一次），或把校验并入 `_ensure_weights` 幂等分支 |
| 4 | SEC-03 ✅ 批次87 | 统一 5 处 `weights_only=False` / `allow_pickle=True` 到受限加载器；WebUI `ckpt_value` 加白名单 |
| 5 | SEC-04 ✅ 批次87 | `0.0.0.0` 且无 `ADR_TTS_API_KEY` → 拒绝启动（或强制生成临时 key） |
| 6 | BUG-01 ✅ 批次87 | `Trainer` 构造断言 `len(train_loader) > 0`，报错提示调整 `batch_size`/补充样本 |
| 7 | BUG-02 ✅ 批次87 | 无论日志是否可写都 `take()` 并起泵，写失败即丢弃 |
| 8 | PERF-01 ✅ 批次87 | `policy.load()` 加 mtime 缓存；`_evaluate` 移出全局锁 |
| 9 | PERF-02 ✅ 批次87 | `await asyncio.to_thread(_probe_audio, dest)` |

> 批次 87 落地口径：SEC-02 按 `(路径, mtime_ns, size)` 指纹缓存（FIFO 256），换文件自动重校验；SEC-03 新增 `safe_torch_load()` + `allow_globals` 白名单，`data/pipeline.py` 新写入去 object dtype，历史 npz 回落并告警；SEC-04 新增逃生阀 `ADR_ALLOW_UNAUTHENTICATED=1`，Dockerfile 与桌面壳"对外服务"开关显式置位；PERF-01 评估用独立非阻塞 `_eval_lock`，不再串行请求。

### P2 —— 迭代内

| # | 条目 |
|---|---|
| 10 | CACHE-01：换权分支调用 `_seg_cache_clear()`，并把 `weights_stamp` 纳入段 key |
| 11 | SEC-05：音色包覆盖前做 sha 比对 + "影响 N 个档案"确认 |
| 12 | SEC-06：ASGI 层 body 上限 |
| 13 | SEC-07：`timeout_read` + 下载大小校验 |
| 14 | BUG-03：上传名加随机后缀 / `O_EXCL` |
| 15 | BUG-04：只捕获 `FileNotFoundError`，其余映射 500 |
| 16 | BUG-05：模块级 `is_ready()` 判 `_stalled` |
| 17 | CON-02：非 WAV 流改用长驻编码器，或文档限定"前缀块可独立播放" |
| 18 | PERF-03：mel basis/window 缓存 |
| 19 | PERF-04：`_TextBuffer` 改 `list`/`bytearray` |
| 20 | DATA-01 + DATA-02：hop 单源化、F0 重采样到 mel 帧格 |
| 21 | API-01 / API-02：mode 3 实现或改文档；seed 入 key 或删公开语义 |

### P3 —— 清理与治理

| # | 条目 |
|---|---|
| 22 | 删除 6 个零引用符号（保留 `get_version` 则标注为对外 API） |
| 23 | 统一 `is_safe_name`、`_mel_to_wav`、目录包含校验 |
| 24 | 为 `callbacks.py` 三个未注册回调二选一：接入 `adr train`，或移到 `scripts/` 并标注"实验" |
| 25 | CI：跑 `cargo test`；Windows job；覆盖率门槛；更新"431+"注释 |
| 26 | 补 9.2 表中的关键用例（优先 CON-01、CACHE-01、SEC-02） |

---

## 附录 A：本轮复核确认"已修复"的上一轮问题

以下为回代码确认已正确实现，**不计入本轮发现**：

1. 流式 `q.put` 不再阻塞（无界队列，`gsv_engine.py:863-L868`）。
2. `tts_cache` 锁外磁盘 IO（`tts_cache.py:53`）。
3. 训练/推理 mel 统一为 `compute_mel_bigvgan`（`data/pipeline.py:245`、`utils/audio.py:300`）。
4. 桌面更新链：sha256 校验 + 安装目录前缀 + 扩展名 + 重算（`update.rs:521-L523`、`659-L682`）。
5. `GET /set_*_weights` 已有 `require_browser_safe`（`v2_compat.py:484-L502`）。
6. 权重指纹幂等换权（`gsv_engine.py:728-L755`）。
7. ffmpeg 全路径超时（`audio_codec.py:77`）。
8. 梯度累积余数 flush（`trainer.py:401-L412`）。
9. `DataLoader` 绑定方法 collate（`trainer.py:168`）。
10. 极小数据集划分：`<2` 报错且 train ≥1（`dataset.py:325-L334`）——但 train < batch_size 的静默空转仍存在（BUG-01）。
11. 训练数据 `samples/*.npz` 与 dataset 目录层级已修正。
12. `pathsafe.resolve_within` 已用于 profile 引用校验（`native.py:125-L127`）。
13. 音色包两段式提交与回滚（`voice_package.py:268-L330`）。
14. 参考音频原子写 + `ref_stamp` 进入段缓存 key（`gsv_engine.py:897-L899`）。
15. 上传半截文件回滚（`console.py:617-L620`）。
16. 错误契约按 `error_status_code` 分流 400/500/503（`v2_compat.py:465-L471`）。

## 附录 B：范围与局限

- **未运行**：训练、打包、GPU 推理、浏览器 PNA 实测、压力测试、覆盖率实测。
- **静态结论均可复现**：所有行号对应当前工作区；引用方式为 `file:///` 链接。
- **推测标注**：SEC-01 的浏览器跨站可达性（PNA）、CON-02 的 OGG 逐块拼接播放行为，均已在正文标注 `（推测）`，需运行时验证后才能升级为确定结论。
- **已剔除**：Verify 阶段另有 9 条候选未获 3/3（含证据不足、影响不可达、或属有意设计取舍），按要求不列入本报告。

---

*报告结束。全部结论基于 2026-11-07 当前工作区代码。*
