# ADR 项目审计报告（第三轮 · 复核与补盲区）

- 审计日期：2026-11-07
- 审计对象：`e:\新创意构思\新建文件夹\ADR`
- 审计定位：复核第二轮 P0/P1 的落地状态 + 补齐运行时/交付链盲区；**不重复**前两轮已确认条目
- 代码快照：当前工作区（含批次 89/90 未提交改动）。文中行号均对应当前快照；第八节为批次 90 修复回填，5.1 为作者确认的已知约定及其对结论的订正
- 审计方式：静态阅读 + 配置/脚本核对 + 定向实跑（未跑训练、未打包、未压测）

---

## 一、执行摘要

### 1.1 复核结论

第二轮 9 条 P0/P1 中：

| 状态 | 数量 | 条目 |
|---|---:|---|
| 已修（批次 90 闭环） | 9 | SEC-01、CON-01\*、SEC-02、SEC-03、SEC-04、BUG-01、BUG-02、PERF-01、PERF-02 |

\* CON-01 的"进程内自动重建"仍不成立，但"永久 503、无法自恢复"已按作者约定（5.1-1）订正为：**裸跑环境不可恢复，容器/桌面依赖外部 supervise**，并带一个未覆盖前提（R3-CON-02，见 2.1）。

**最关键结论（已按五条约定订正）：CON-01 不再作为"全站永久 503、只能重启"的整体结论成立，但要写清成立范围。**

- **进程内确实无法自愈**：卡死 worker 同时持有 `_lock` 与 `_CHDIR_LOCK`，`_drop_stalled_model` 需 `_lock`、`_lazy_init` 需 `_gsv_context`，两把锁都不会释放。
- **但看门狗能在锁死状态下真的退出进程**：`_watchdog_loop`（`gsv_engine.py:1424-1445`）是独立 daemon 线程，只做 `sleep(15s)` + 无锁读 `_degraded_since` + `on_exit()` + `os._exit(75)`，**不抢引擎锁、不进 `_CHDIR_LOCK`、不占 anyio 线程池**；`on_exit` 的 `_stop_proc` 等待均有上界（`console.py:305-317`）。
- **容器/桌面可被外部拉起**：compose `restart: unless-stopped`（`docker-compose.yml:33`）；桌面壳 `supervise` 轮询重启（`lib.rs:1345-1405`，看门狗 300s 触发时进程存活已远超 `STABLE_UPTIME=60s`，失败计数会重置）。
- **裸跑不自动恢复**：`python -m adr.server` 无守护，退出后不回来；`ADR_ENGINE_WATCHDOG_SEC=0` 可关闭看门狗（约定 1）。
- **未覆盖前提（残余 P2）**：看门狗触发源只有 `_mark_stalled()`（流式 idle 超时，`gsv_engine.py:1230`）。**非流式 solo（默认路径）与合批路径的 `_tts.run` 卡死不会置 `_degraded_since`**，后续请求只是 55s 锁超时 503，`/readyz` 仍 200、看门狗不触发——这条路径上即便容器/桌面也不会自动重启。记为 R3-CON-02。

其余复核要点：SEC-01 慢路径已改 zip 流式中央目录，仅 `data.pkl` 进内存（≤256MB），截头自愈仅 ≤64MB，不再有 2~4 倍文件体积峰值；SEC-03 的 `allow_pickle=True` 回落已删除并新增迁移工具 `scripts/migrate_npz_pickle.py`；SEC-04 镜像已删除 `ENV ADR_ALLOW_UNAUTHENTICATED=1`，改由 compose 显式声明（仅绑回环），`docker run -p` 会按作者意图拒绝启动。三条均转为已修，限值/破坏性变更为有意设计（5.1-3/4/5）。

### 1.2 本轮新发现

| 级别 | 编号 | 条目 | 批次 90 状态 |
|---|---|---|---|
| P1 | R3-DATA-01 | 训练 checkpoint 非原子写入 | 已修复 |
| P2 | R3-CON-01 | 合批路径无界等待 | 已修复 |
| P2 | R3-LIFE-01 | 子进程不随服务退出回收 | 已修复 |
| P2 | R3-OBS-01 | 健康检查不反映永久降级 | 已修复（部署注意见 5.1-2） |
| P2 | R3-PACK-01 | 桌面发货运行时（3.13）未被 CI 覆盖 | 已修复 |
| P3 | R3-PACK-02 | CI 的 torch 未钉版本 | 已修复 |
| P2 | R3-CON-02 | 非流式/合批卡死不置降级标记，看门狗与 `/readyz` 盲区 | **未修（约定核对中新识别的残余）** |

候选 11 条，通过三视角 3/3 的 6 条保留，其余 5 条因影响票或成本票不足已剔除（不列示）；约定核对中新识别 1 条残余（R3-CON-02，同样 3/3）。

### 1.3 实跑验证

- `python -m pytest tests/test_gsv_stream_threading.py tests/test_server_security.py -q`：**71 passed**（覆盖本轮涉及的流式线程、安全守卫回归）
- `python scripts/check_versions.py`：**exit 0**（11 个版本面与 `pyproject.toml` 一致）
- `python scripts/check_requirements.py`：**exit 0**
- 批次 90 回填后全量：**912 passed, 1 skipped, 0 failed**（详见 8.2）

---

## 二、修复状态复核表

| 编号 | 二轮级别 | 当前状态 | 当前证据（文件:行） | 已修的修法 / 残留说明 |
|---|---|---|---|---|
| SEC-01 | P0 | **已修（批次 90）** | `adr/server/v2_compat.py:254-273`、`:358-389`；`adr/models/gsv_engine.py:376-379`、`:388-423`、`:164-193` | 已修：GET `/tts` 挂 `Depends(require_browser_safe)`；权重预检移到 `_check_params()` 之后且只检请求级权重；UNC/设备路径拒绝 + `is_file()` + `ADR_TTS_WEIGHTS_MAX_MB`（默认 2048）。慢路径改为 `zipfile` 流式读磁盘中央目录，仅 `data.pkl` 进内存（超 256MB 拒绝）；截头自愈仅对 ≤64MB 文件；快路径 `torch.load(weights_only=True, mmap=True)`。单请求峰值从约 2~4 倍文件体积降到 data.pkl 体积。256/64MB 与 2GB 文件上限是作者确认的硬约束（5.1-5），不作为缺陷 |
| CON-01 | P0 | **已修（批次 90，带前提）** | `adr/models/gsv_engine.py:60-88`、`:719-737`、`:1190-1230`、`:1323-1336`、`:1390-1445`；`adr/server/__main__.py:177-186` | 已修：`q.get(timeout=120s)`、`_engine_lock` 55s 超时、准入 `slot(timeout=55s)`、超时抛 `EngineStalledError`→503；新增 `/readyz` + 进程级看门狗（降级 300s → `os._exit(75)`），由 Docker restart / 桌面 supervise 拉起。**前提**：触发源仅流式 `_mark_stalled`；非流式 solo/合批 `_tts.run` 卡死不置 `_degraded_since` → 看门狗盲区（R3-CON-02）；裸跑不自动恢复。详见 2.1 |
| SEC-02 | P1 | **已修** | `adr/models/gsv_engine.py:196-208`、`:236-250` | 按 `(路径, mtime_ns, size)` 指纹缓存校验结果，FIFO 上限 256；文件替换后指纹变化自动重校验；回归用例已通过 |
| SEC-03 | P1 | **已修（批次 90）** | `adr/inference/pipeline.py:142-143`；`adr/vocoder/bigvgan.py:210-211`；`adr/training/trainer.py:526`；`adr/eval/speaker_sim.py:35`；`adr/data/pipeline.py:121-146`；`adr/training/dataset.py:261-267`；`scripts/migrate_npz_pickle.py`；`scripts/opencpop_to_npz.py:162` | 4 个入口统一 `safe_torch_load` / `weights_only=True` + GLOBAL 白名单；`TrainSample.load()` 的 `allow_pickle=True` 回落已删除，object 数组直接 `ValueError` 并提示迁移；新增离线迁移工具（目录/单文件、`--dry-run`、逐文件原子替换）；写入端改 `dtype=str`。legacy 数据必须迁移（约定 4），运行期报错已带命令；README/REPRODUCE 未提示，仅作 P3 文档建议 |
| SEC-04 | P1 | **已修（批次 90，有意破坏性变更）** | `adr/server/__main__.py:85-110`；`docker/Dockerfile:129-133`；`docker/docker-compose.yml:32-37`、`:49` | 非容器场景无 key 且绑 `0.0.0.0` 直接拒绝启动；镜像已删除 `ENV ADR_ALLOW_UNAUTHENTICATED=1`，改由 compose 显式声明（仅发布 `127.0.0.1:9881:9881`）。`docker run -p 9881:9881` 升级后会因无 key 拒绝启动——作者确认有意（约定 3），不报缺陷；发布说明需提示 |
| BUG-01 | P1 | **已修** | `adr/training/trainer.py:182-188`；`adr/training/dataset.py:319-328` | `drop_last=True` 导致 0 step 时提前抛错；样本 <2 无法划分时明确报错，不再静默空转 |
| BUG-02 | P1 | **已修** | `desktop/src-tauri/src/lib.rs:1108-1145` | stdout/stderr 一律 `pump_server_log` 抽干，日志文件打不开也继续消费管道，子进程不再因管道写满而阻塞 |
| PERF-01 | P1 | **已修** | `adr/core/policy.py:25-27`、`:117-139`；`adr/core/runtime.py:31`、`:82-95` | `policy.json` 按 `(path, mtime_ns, size)` 缓存解析结果；评估移到全局锁外，不再每请求锁内读盘 |
| PERF-02 | P1 | **已修** | `adr/server/console.py:537`、`:631` | ffprobe 探针改 `await asyncio.to_thread(_probe_audio, dest)`，不再阻塞事件循环 |

### 2.1 CON-01 的收口与前提（结论已按作者约定订正）

"无界等待"早已被压成"有界失败"：

```python
# adr/models/gsv_engine.py:1190
item = q.get(timeout=_STREAM_IDLE_SEC)        # 流式 120s 无输出 → 判卡死
# adr/models/gsv_engine.py:727
if not self._lock.acquire(timeout=_LOCK_WAIT_SEC):   # 55s 拿不到 → 503
```

进程内"自愈"仍不成立：卡死 worker 同时持 `_lock` 与 `_CHDIR_LOCK`（`gsv_engine.py:1323-1336` / `:1157`），`_drop_stalled_model()` 需 `_lock`，`_lazy_init()` 需 `_gsv_context()`。**但批次 90 已不再依赖进程内自愈，改为进程级收口：**

```python
# gsv_engine.py:1424-1445  看门狗线程：独立、无锁、有界
while True:
    time.sleep(15)
    if degraded_seconds() < grace: continue       # 只读 _degraded_since，不抢锁
    ...                                            # log_event("engine_watchdog")
    on_exit()                                      # stop_background_procs（有界）
    os._exit(75)                                   # EX_TEMPFAIL
```

- **锁死状态下能否真的 `os._exit(75)`？能。** 该线程不获取引擎锁、不进入 `_CHDIR_LOCK`、不占 anyio 线程池；`degraded_seconds()` 是无锁读 `_ENGINE._degraded_since`（`gsv_engine.py:1377-1387`）；退出前的 `stop_background_procs()` 内部 `_stop_proc` 的 `wait` 均有上界（10s/5s，`console.py:305-317`），最坏情况也只是延迟退出，不会卡死。
- **外部拉起已就位。** Docker compose `restart: unless-stopped`（`docker/docker-compose.yml:33`）；桌面壳 `supervise` 循环重启（`desktop/src-tauri/src/lib.rs:1345-1405`，`MAX_FAILURES=4`、`STABLE_UPTIME=60s`）。看门狗在 300~315s 触发时进程已存活远超 60s，失败计数会重置，因此可持续拉起。
- **裸跑不恢复。** `python -m adr.server` 无守护，退出后不会自动回来；`ADR_ENGINE_WATCHDOG_SEC=0` 可关闭（约定 1）。

**订正后的结论**：~~"全站永久 503，直到重启进程"~~ → **"进程内不可恢复；容器/桌面依靠外部 supervise 自动恢复；裸跑开发环境不自动恢复。"**

**未覆盖的前提（重要）**：看门狗的唯一触发源是 `_mark_stalled()`（`gsv_engine.py:1230`，流式 idle 超时）。非流式 solo（`:1323-1336`，默认路径）与合批 worker（`gsv_coalescer.py:263-265`）都在锁内直接 `self._tts.run(...)`，没有 idle 看护；卡死时只产生 `engine_lock_timeout`（`:727-733`，**不置 `_degraded_since`**）。此时 `is_ready()` 仍为 true、`is_stalled()` 为 false、`/readyz` 仍 200、看门狗永不触发——**即便在容器/桌面也不会被自动拉起**。这是 R3-CON-02（P2）。

### 2.2 SEC-01 的收口（批次 90）

批次 90 前：慢路径 `raw = f.read()` + `zipfile.BytesIO(raw)`，峰值约 2 倍文件体积。批次 90 后（`adr/models/gsv_engine.py:376-423`）：

- 慢路径 `zipfile.ZipFile(weights_path)` 直接读磁盘中央目录，只有 `data.pkl` 一个成员进内存；超过 `_WEIGHTS_PKL_MAX=256MB` 直接拒绝（防解压炸弹）。
- 截头 zip 自愈（补 `PK` 头）只对 ≤ `_WEIGHTS_PATCH_MAX=64MB` 的文件启用；更大的缺头文件直接拒绝。
- 快路径 `torch.load(weights_only=True, mmap=True)`（老版本 torch 无 `mmap=` 时回落）。
- 文件级上限仍为 `ADR_TTS_WEIGHTS_MAX_MB`（默认 2048）。

单请求峰值从"约 2× 文件体积"降到"data.pkl 体积"。256/64MB 与 2048MB 是作者确认的硬上限（约定 5，正常 GSV 权重远小于此），**不再作为缺陷上报**，仅作为输入约束记录：超限即拒绝属预期行为。

---

## 三、本轮新发现

### R3-DATA-01 · 训练 checkpoint 非原子写入，中断会损坏并覆盖上一份可用权重（P1）— 批次 90 已修复

- **位置**：`adr/training/trainer.py:512`；`adr/training/callbacks.py:70-84`；加载侧 `adr/training/trainer.py:526`
- **证据**

  ```python
  # trainer.py:512  直接写最终路径，无临时文件、无备份
  torch.save(ckpt, path)

  # callbacks.py:70-83  周期性/最优保存都直接覆盖同名文件
  ckpt_path = self.save_dir / f"epoch_{epoch + 1:04d}.pt"
  trainer.save_checkpoint(ckpt_path)
  ...
  best_path = self.save_dir / "best.pt"
  trainer.save_checkpoint(best_path)
  ```

- **影响**：`torch.save` 期间被 Ctrl+C、磁盘写满、断电或 OOM 杀死，`best.pt` / `last.pt` 会留下半截 zip 文件，同时把上一份可用 checkpoint 覆盖掉。`load_checkpoint()` 走 `safe_torch_load`，读到截断文件直接抛错，续训与导出都失败，只能从头重训；磁盘满时"写一半"是确定性行为，不是小概率。
- **验证**
  - 正确性 ✓：写路径确定，无 `.tmp`/`os.replace`/`.bak`。
  - 影响 ✓：桌面/CLI 训练动辄数小时，checkpoint 数百 MB~GB，磁盘余量不足时必现。
  - 成本 ✓：`tmp = path.with_suffix('.tmp')` → `torch.save(tmp)` → `os.replace(tmp, path)`，并保留上一份为 `.bak`；约 5 行。
  - **3/3 通过。**

---

### R3-CON-01 · 合批路径未纳入停滞加固：无界等锁 + 无超时 `result()`（P2，配置相关）— 批次 90 已修复

- **位置**：`adr/models/gsv_coalescer.py:259`；`adr/models/gsv_engine.py:1240`
- **证据**

  ```python
  # gsv_coalescer.py:259  合批 round 直接 with engine._lock，绕过了带超时的 _engine_lock()
  with engine._lock:
      ...

  # gsv_engine.py:1240  请求线程无限等待 future
  sr, audio = self._coalescer.submit(inputs).result()
  ```

  触发条件：`ADR_T2S_BATCH_RNG=1` 且调度并发 ≥2（默认关闭，属文档化的合批调优组合）。
- **影响**：与 CON-01 同一失效模式，但发生在此路径时**不会** 55s 后 503——`_round` 卡在上游时永远不释放 `_lock`，`future` 永不完成，请求线程与准入槽一起永久挂起，anyio 线程池被逐个耗尽。solo/流式路径已经修好，唯独合批路径仍是旧语义。
- **验证**
  - 正确性 ✓：`with engine._lock` 无超时；`.result()` 无 `timeout`；与 `_engine_lock()` 的实现不一致。
  - 影响 ✓：开启合批即命中，属于官方可用的性能选项，不是死配置。
  - 成本 ✓：`_round` 改用 `engine._engine_lock()`；`result(timeout=...)` 超时抛 `EngineStalledError`；两处小改。
  - **3/3 通过。**

---

### R3-LIFE-01 · 服务退出不回收训练/下载子进程，形成孤儿占 GPU（P2）— 批次 90 已修复

- **位置**：`adr/server/console.py:478-504`、`:812`；`adr/server/app.py:143-150`
- **证据**

  ```python
  # console.py:478-481  POSIX 新会话 + Windows 独立进程组
  proc = subprocess.Popen(..., start_new_session=True)
  _train.update(..., proc=proc, gate=gate_proc, ...)

  # app.py:143-150  lifespan 只有 prewarm，yield 后没有任何 shutdown 清理
  @asynccontextmanager
  async def _lifespan(app):
      threading.Thread(target=_prewarm_gsv, ..., daemon=True).start()
      yield
  ```

  `_stop_proc` 只在 `train_stop()` 与 reader 结束时调用；全仓无 `atexit` / signal handler / shutdown hook。
- **影响**：通过控制台启动训练后，关闭服务（Ctrl+C、`systemctl stop`、关闭桌面窗口外的 CLI）不会终止训练/门禁/下载子进程。训练继续写输出目录并占住 GPU 显存，下一次启动再训练会直接 OOM；Windows 任务管理器里留下无主 python 进程。桌面壳的 `taskkill /T` 只覆盖桌面这一种关闭路径。
- **验证**
  - 正确性 ✓：`start_new_session=True` 使子进程脱离父进程组，父进程退出不传导；lifespan 无清理。
  - 影响 ✓：训练是长任务，退出服务是常见操作；GPU 资源冲突后果明确。
  - 成本 ✓：`_lifespan` 的 `finally` 中 `_stop_proc(_train["proc"])`、`_stop_proc(_train["gate"])`、`_stop_proc(_dl["proc"])`。
  - **3/3 通过。**

---

### R3-OBS-01 · 健康检查不反映引擎永久降级，编排器不会重启（P2）— 批次 90 已修复（部署注意见 5.1-2）

- **位置**：`adr/server/native.py:37-50`；`docker/Dockerfile:119-120`；`adr/models/gsv_engine.py:677-699`
- **证据**

  ```python
  # native.py:44-48  状态恒为 ok，就绪度只是附属字段
  return {"status": "ok", "engine_ready": _gsv.is_ready(), ...}

  # Dockerfile:119-120  只看 HTTP 能否 200，不解析 body
  HEALTHCHECK ... CMD python -c "...urlopen('.../api/adr/v1/health'...) || exit 1"
  ```

  CON-01 残留态下 `is_ready()` 为 false（`_stalled=True`），但 `/health` 仍 200。
- **影响**：永久 503 的实例在 Docker/compose/外部探针眼里始终"健康"，restart 策略不会触发；排障时 `/health` 的 `status: ok` 还会误导值班人员。与 2.1 叠加，这是"服务已经不可用但没人知道"的组合故障。
- **验证**
  - 正确性 ✓：状态字段硬编码；HEALTHCHECK 只判断进程退出码。
  - 影响 ✓：容器与桌面是主要交付形态，健康检查是唯一自动恢复触发源。
  - 成本 ✓：增加 `/readyz`（`engine_ready` 且未 `_stalled` 才 200），HEALTHCHECK 指向它；或让 stalled 态直接以非 0 退出，由守护重启。
  - **3/3 通过。**

---

### R3-PACK-01 · 桌面发货运行时是 CPython 3.13，CI 只测 3.11（P2）— 批次 90 已修复

- **位置**：`.github/workflows/smoke.yml:67-74`；`scripts/build_runtime.ps1:66`（及 `:201-206`）；`docker/Dockerfile:83`
- **证据**

  ```yaml
  # smoke.yml:67-74
  test-windows:
    runs-on: windows-latest
    strategy:
      matrix:
        python-version: "3.11"
  ```

  ```powershell
  # build_runtime.ps1:66  桌面便携运行时
  $PbsAsset = "cpython-3.13.16+20261009-x86_64-pc-windows-msvc-install_only.tar.gz"
  ```

  容器基线与 CI 都是 3.11，只有桌面发货件是 3.13。
- **影响**：桌面是主要交付形态，而发货解释器完全没有进入 CI：3.13 移除 `audioop`/`cgi` 等模块、`libsndfile`/`torch` 轮子差异、路径与子进程语义都只在用户机器上首次验证。当前 Windows job 恰好是为"路径/落盘/子进程"加的，却测的是另一个解释器。
- **验证**
  - 正确性 ✓：工作流与打包脚本解释器版本不一致。
  - 影响 ✓：发件平台未被测试，属于交付链真实缺口。
  - 成本 ✓：`test-windows` 增加 `3.13` 矩阵；若个别 extras 无 cp313 轮子，先用 `[m1,m2]`（打包实际使用的子集）跑通。
  - **3/3 通过。**

---

### R3-PACK-02 · CI 的 torch/torchaudio 未钉版本，与容器/便携包存在解析漂移（P3）— 批次 90 已修复

- **位置**：`.github/workflows/smoke.yml:52`、`:80`；`docker/Dockerfile:58`、`:61`；`scripts/build_runtime.ps1:201`、`:206`
- **证据**

  ```yaml
  # smoke.yml:52 / :80
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

  ```
  # Dockerfile:58 / build_runtime.ps1:201,206
  torch==2.10.0 torchaudio==2.10.0
  ```

  `docker/requirements-linux.txt` 不包含 torch（按 CPU/CUDA 变体单独安装），所以 CI 这条 pip 命令没有任何钉版约束。
- **影响**：CI 解析到的是 `pyproject.toml` 允许范围内的最新 torch（含补丁甚至 minor），而容器/便携包固定 2.10.0。`weights_only` 默认值、序列化行为、算子内核都随版本变化，可能出现"CI 绿、发件红"或反之；这也与工作流注释里"CI 与 Docker 共用同一份版本清单"的说法不完全一致（清单只覆盖非 torch 包）。
- **验证**
  - 正确性 ✓：pip 命令无版本；约束文件不含 torch。
  - 影响 ✓：版本面漂移，直接影响权重加载这一核心路径的兼容性。
  - 成本 ✓：两行改为 `torch==2.10.0 torchaudio==2.10.0`。
  - **3/3 通过。**

---

### R3-CON-02 · 非流式/合批卡死不置降级标记，看门狗与 `/readyz` 均盲（P2，约定核对中新识别）

- **位置**：`adr/models/gsv_engine.py:727-733`、`:1323-1336`、`:1157`；`adr/models/gsv_coalescer.py:263-265`
- **证据**

  ```python
  # 1323-1336  solo（默认路径）：持锁直接跑上游，无 idle 看护
  with self._engine_lock():
      self._lazy_init()
      ...
      with self._gsv_context():
          for sr, audio in self._tts.run(inputs):   # 卡死 → _lock 不释放
              pass

  # 727-733  锁超时只记事件，不置 _degraded_since
  log_event("engine_lock_timeout", wait_s=_LOCK_WAIT_SEC)
  raise EngineStalledError(...)

  # 1230  唯一置位点：流式 idle 超时后的 _mark_stalled()
  ```

- **影响**：默认 solo 路径（合批默认关闭）上游 `_tts.run` 卡死（CUDA 驱动挂死、第三方代码死循环）时，请求线程永久持 `_lock`；后续请求 55s 后 503，但 `_stalled=False`、`/readyz` 200、看门狗不触发 → 容器/桌面都不会重启，形成"永久 503 但对编排器始终健康"的组合故障。流式路径有 120s idle 判定，非流式没有，这是两者唯一的行为差。
- **验证**
  - 正确性 ✓：`_degraded_since` 唯一写入点是 `_mark_stalled()`（`gsv_engine.py:751-752`，仅 `:1230` 调用）；`/readyz` 只读 `is_ready() and not is_stalled()`（`native.py:80-82`）。
  - 影响 ✓：solo 是默认路径，卡死场景与 CON-01 同源；后果是"自动恢复机制整体失效"，不是单请求失败。
  - 成本 ✓：给 solo/合批的 `_tts.run` 再套一层与流式同款的 idle 监督，或"连续 N 次 `engine_lock_timeout` 即置 `_degraded_since`"，约 10~20 行。
  - **3/3 通过。**

---

## 四、盲区覆盖说明

### 4.1 并发与生命周期

- 已发现：R3-CON-01（合批无界等待）、R3-LIFE-01（子进程孤儿）、CON-01 残留（双锁不可释放）；批次 90 均已落地，仅剩 R3-CON-02（非流式卡死不置降级标记，见 2.1）。
- 已确认无问题：
  - 流式 worker 异常通过 `__error__` 哨兵回传，不会让消费端空等 120s（`gsv_engine.py` 流式生成器分支）。
  - 合批 `_round` 内部对单个 job 异常做了隔离（失败只落到对应 future），不会污染同批其他请求。
  - 后台线程（prewarm、coalescer、v3 producer、profile worker）均为 daemon，不会阻止进程退出。
  - 准入槽与引擎锁在 solo/流式路径都有 55s 上界，不再出现无界排队。
- 刻意取舍：`_CHDIR_LOCK`（RLock）跨整个合成持锁是为了解决跨线程 chdir/release 的旧死锁（`gsv_engine.py:1009-1014` 注释），代价是卡死时锁不可回收——这正是 2.1 的根因；批次 90 已改为进程级看门狗收口，不再要求进程内解锁。

### 4.2 部署与打包链

- 版本一致性：`scripts/check_versions.py` 覆盖 11 个版本面（pyproject、`adr.__version__`、tauri.conf、Cargo、package.json、installer.iss、Dockerfile ARG、compose 镜像 tag 等），实跑 exit 0，当前统一为 1.3.1。
- 已发现并已修复：R3-PACK-01（CI 3.11 vs 发货 3.13）、R3-PACK-02（torch 漂移）。
- 其他确认：Docker 构建 GSV 通过固定 commit + 补丁链，`check_versions.py` 同时校验 git tag；compose 默认只绑 `127.0.0.1`（`:37`），镜像 tag 由版本脚本联动。

### 4.3 依赖与供应链

- 未形成达到 3/3 的新发现。
- 已核实基线：`gsv_bootstrap` 对下载归档做 SHA-256 全量校验后再解压；`hub.py` 权重下载带 SHA-256；`build_runtime.ps1` 对 torch/依赖精确钉版；`safe_torch_load` 对第三方权重做 GLOBAL 白名单。第三方模型加载的信任边界由"传输完整性 + 反序列化白名单"两层把住。

### 4.4 可观测性与排障

- 已发现并已修复：R3-OBS-01（健康检查失真）。批次 90 新增 `/readyz`（未就绪或 `_stalled` 一律 503），Docker HEALTHCHECK 指向它并给 `start-period=600s`；`/health` 保留纯存活语义。残余仅为部署注意（其它编排器须同步放宽启动宽限期，见 5.1-2）与 R3-CON-02 的盲区。
- 已核实基线：流式失败会按 CUDA OOM / nltk 缺失 / 其他三类分别记可定位日志（`v2_compat.py:231-241`）；引擎锁超时与停滞都会 `log_event`；审计日志已有轮转（第二轮 OPS-01 已修）。
- 刻意取舍：桌面壳的 `server.log` 只在启动时按 10MB 单代轮转（`lib.rs`），运行中不切；纯文本增速低，属已文档化的取舍，不列为发现。

### 4.5 数据与状态一致性

- 已发现并已修复：R3-DATA-01（checkpoint 非原子）。
- 已核实基线：`tts_cache` 磁盘写入走临时文件 + `os.replace` 并清理残片；音色库/档案元数据同样原子替换；`policy.json` 读取带指纹缓存（写入虽非原子，但全仓无 `policy.save()` 调用方，不构成线上路径）。
- 并发读写：`tts_cache`、`_WEIGHTS_CHECKED`、`_LOAD_CACHE` 均有独立锁；未发现"缓存与磁盘不同步"的可复现路径。

---

## 五、作者口径核对（不作为缺陷）

1. `--eval-asr` 与 `--eval-wav-quality` 同时开启时 ASR 优先：代码顺序一致，属设计选型；仅记录，不计缺陷。
2. 覆盖率门槛 73%：`.github/workflows/smoke.yml:62` 确认 `--cov-fail-under=73`，与本地 76% 减 3pt 余量的说明一致。
3. `is_safe_name` 归属：`adr/server/pathsafe.py:9-16` 明确写"批次89 下沉到 `adr.core.naming`，本模块仅再导出"，`adr/core/naming.py:38` 为唯一实现；外部 import 路径不变。与说明一致，不报重复实现/循环依赖。

### 5.1 作者确认的已知约定（五条）

| # | 约定 | 代码/配置证据 | 对第 3 轮结论的影响 |
|---|---|---|---|
| 1 | 看门狗默认 300s + `os._exit(75)`，依赖外部拉起（Docker restart / 桌面壳 supervise）；裸跑开发环境不自动重启，可用 `ADR_ENGINE_WATCHDOG_SEC=0` 关闭 | `gsv_engine.py:82-88`、`:1390-1445`；`__main__.py:177-186`；`docker-compose.yml:33`；`lib.rs:1345-1405` | **修改 CON-01**：从"永久 503、无法自恢复"降级为"裸跑不可恢复，容器/桌面依赖外部 supervise"；并新增前提 R3-CON-02 |
| 2 | `/readyz` 首启预热期返回 503；Docker 已给 `start-period=600s`；其它编排器需同步放宽宽限期 | `native.py:67-92`；`Dockerfile:117-123` | **修改/降级 OBS-01**：原"健康检查不反映永久降级"已修复；转为部署注意事项，不再作为缺陷 |
| 3 | 容器默认无 key 拒绝启动是有意的破坏性变更，`docker run -p 9881:9881` 升级后会直接失败 | `Dockerfile:129-133`；`docker-compose.yml:49`；`__main__.py:85-110` | **修改 SEC-04**：状态改为已修，不再报缺陷；仅需发布说明提示这一破坏性变更 |
| 4 | legacy npz 必须迁移，外部旧数据集需跑 `python scripts/migrate_npz_pickle.py <目录>` | `data/pipeline.py:121-146`；`dataset.py:261-267`；`scripts/migrate_npz_pickle.py` | **修改 SEC-03**：状态改为已修，不再报缺陷；运行期报错已带迁移命令，README/REPRODUCE 未提示 → 仅 P3 文档建议 |
| 5 | `data.pkl` 256MB / 截头 64MB 是硬上限，正常权重远小于此，遇超大权重再评估 | `gsv_engine.py:376-379`、`:405-423` | **修改 SEC-01**：限值属有意设计，不再作为缺陷；只保留"上限是输入约束"的记录 |

逐条核对结果：

1. **维持"进程内不可自愈"的事实，修改整体结论**。看门狗线程不依赖引擎锁/线程池，锁死状态下可完成 `os._exit(75)`；容器与桌面确有外部拉起（`restart: unless-stopped` / `supervise` 循环），裸跑没有。前提是触发源存在——非流式路径不置 `_degraded_since`，见 R3-CON-02。关闭看门狗（`=0`）后回到"永久 503"，属作者已知取舍。
2. **OBS-01 已修复**。`/readyz` 在未就绪/降级时 503，HEALTHCHECK 已指向它；`start-period=600s` 覆盖首启下载 ~2GB 权重的预热窗口。原缺陷结论不再成立，只剩"其它编排器需同步放宽宽限期"的部署提醒：若按默认 30s 探活、3 次重试，首启会被误杀并循环重启。
3. **SEC-04 记录为有意设计，不上报缺陷**。镜像不再写死豁免，compose 显式声明且只发布回环；`docker run -p 9881:9881` 失败是预期行为。仅在发布说明中提示。
4. **SEC-03 记录为已修；迁移提示只作文档建议**。代码路径：object npz → `ValueError`，`dataset.py:267` 的错误信息直接给出迁移命令，因此不会被静默忽略；不足在于 README / REPRODUCE 的训练数据准备章节没有这条命令，外部用户只能在失败后才知道。属 P3 体验补强，不是缺陷。
5. **SEC-01 记录为有意硬上限，不上报缺陷**。256MB 的 `data.pkl` 与 64MB 的截头自愈上限只影响异常超大权重；正常 GSV 权重远小于此，且超限是显式拒绝（有清晰报错），不是崩溃或静默降级。

---

## 六、优先级行动清单

### P0（当日）— 已闭环

1. **CON-01 残留（2.1）**：批次 90 已落地——`/readyz` + Docker HEALTHCHECK 指向它 + 进程级看门狗（降级 300s → `os._exit(75)` → 外部 `restart` / 桌面 `supervise`）。结论已按约定 1 订正；剩余前提见下方 P2-10。

### P1（本周）— 已闭环

2. **R3-DATA-01**：已修，`trainer.py:516-539` 原子写 + `.bak`。
3. **SEC-03 残留**：已修，严格 `allow_pickle=False` + 迁移工具；用户文档补充见 P3-12。

### P2（迭代内）

4. **R3-CON-01**：已修，合批 `_round` 改 `_engine_lock()`，`result(timeout=600s)`。
5. **R3-LIFE-01**：已修，lifespan `finally` + `atexit` 回收子进程。
6. **R3-OBS-01**：已修，`/readyz` + HEALTHCHECK。**部署注意**：其它编排器的启动宽限期需 ≥600s（约定 2）。
7. **R3-PACK-01**：已修，CI 增加 3.13 矩阵。
8. **SEC-01 残留**：已修，zip 流式中央目录；256/64MB 与 2GB 文件上限为有意硬约束（约定 5）。
9. **SEC-04 残留**：已修，镜像不再写死豁免；`docker run -p` 失败属有意破坏性变更（约定 3）。
10. **R3-CON-02（新增残余，唯一未修项）**：非流式 solo/合批 `_tts.run` 卡死不置 `_degraded_since` → 看门狗与 `/readyz` 均盲。建议给非流式路径套同款 idle 监督，或连续 N 次 `engine_lock_timeout` 置降级。

### P3（顺手）

11. **R3-PACK-02**：已修，CI 钉 `torch==2.10.0 torchaudio==2.10.0`。
12. **迁移文档（建议，非缺陷）**：在 README / REPRODUCE 的训练数据准备处补 `python scripts/migrate_npz_pickle.py <目录>`（运行期报错已带命令，属体验补强）。

---

## 七、审计限制

- 未运行训练、未构建桌面安装包、未启动容器、未做并发压测；R3-DATA-01 的损坏路径依据 `torch.save` 的写文件语义与调用链判断，未实际制造断电/磁盘满。
- CON-01 的"进程内不可恢复"依据锁持有关系与 `_lazy_init`/`_drop_stalled_model` 的前置条件静态推导；看门狗"锁死状态下仍能退出"依据 `_watchdog_loop` 的线程实现（只 sleep / 无锁读 / 有界清理）判断，未构造真实上游卡死注入验证实际触发时序。
- 外部拉起能力依据 `docker-compose.yml` 的 `restart: unless-stopped` 与桌面壳 `supervise` 代码（`lib.rs:1345-1405`）判断，未实际启动容器/桌面做重启演练。
- 行号基于当前含批次 89/90 未提交改动的工作区；若后续继续改动，请以代码为准。

---

## 八、修复回填（批次 90，2026-11-07）

> 本轮按第六节行动清单 P0-1、P1-2、P1-3、P2-4~P2-9、P3-10 逐条落地。未打包、未重签、未提交 git。

### 8.1 逐条落地

| 条目 | 级别 | 落地内容 | 关键文件 |
|---|---|---|---|
| P0-1 CON-01 残留 | P0 | ① 新增 `/api/adr/v1/readyz`：`engine_ready and not stalled` 才 200，stalled / 未就绪一律 503 并带 `status`；`/health` 保留为纯存活探针（桌面壳启动探活不受影响）。② `docker/Dockerfile` 的 HEALTHCHECK 指向 `/readyz`（`start-period=600s` 容忍首启预热），编排器/restart 策略由此获得"不可恢复降级"的触发源。③ 进程级看门狗 `gsv_engine.start_watchdog()`：降级持续 `ADR_ENGINE_WATCHDOG_SEC`（默认 300s）后 `os._exit(75)`，退出前先跑 `console.stop_background_procs()`；仅在真实服务入口 `python -m adr.server` 启动，import / 引擎构造 / 测试不启动。④ 鉴权白名单加入 `/readyz`（容器内探针无 key）。 | `adr/models/gsv_engine.py`、`adr/server/native.py`、`adr/server/auth.py`、`adr/server/__main__.py`、`docker/Dockerfile` |
| P1-2 R3-DATA-01 | P1 | `Trainer.save_checkpoint()` 改原子写：同目录 `.tmp` → 旧文件转 `.bak` → `os.replace(tmp, path)`；任一步失败时若目标不存在则回滚 `.bak`，`finally` 清理残留 `.tmp`。`best.pt` / `epoch_XXXX.pt` / `last.pt` 全部经此单点，覆盖写不再产生半截 zip。 | `adr/training/trainer.py` |
| P1-3 SEC-03 残留 | P1 | ① `TrainSample.load()` 删除 `allow_pickle=True` 回落，object 数组直接 `ValueError` 并给出迁移指引。② 新增离线迁移工具 `scripts/migrate_npz_pickle.py`（仅用户主动运行时才 `allow_pickle=True`；支持目录/单文件、`--dry-run`，逐文件原子替换）。③ 修写入端：`scripts/opencpop_to_npz.py` 原先写 `dtype=object`，改为 `sample_id` / `text` / `phonemes` 写 `dtype=str`——否则新生成的数据自己也会被严格加载拒绝。④ `dataset._load_npz_dir()` 全量失败时抛错并提示迁移命令，不再以"空样本"静默继续。 | `adr/data/pipeline.py`、`adr/training/dataset.py`、`scripts/opencpop_to_npz.py`、`scripts/migrate_npz_pickle.py`（新增） |
| P2-4 R3-CON-01 | P2 | 合批轮次 `_round` 由 `with engine._lock` 改为 `engine._engine_lock()`（55s 带超时）；`submit(...).result(timeout=_COALESCE_WAIT_SEC=600s)` 超时转 `EngineStalledError`（→ 503 可重试），不再无限挂起请求线程与准入槽。 | `adr/models/gsv_coalescer.py`、`adr/models/gsv_engine.py` |
| P2-5 R3-LIFE-01 | P2 | `console.stop_background_procs()` 统一回收训练 / 门禁 / 下载子进程（幂等）；`app.py` 的 lifespan 在 `finally` 调用，并注册 `atexit` 兜底。服务退出不再留下占 GPU 的孤儿进程。 | `adr/server/console.py`、`adr/server/app.py` |
| P2-6 R3-OBS-01 | P2 | 同 P0-1：新增 `/readyz`，`/health` 保留存活语义；Docker HEALTHCHECK 改指 `/readyz`。 | `adr/server/native.py`、`docker/Dockerfile` |
| P2-7 R3-PACK-01 | P2 | `smoke.yml` 的 `test-windows` 增加 `python-version: ["3.11", "3.13"]` 矩阵（`fail-fast: false`），与 `build_runtime.ps1` 发货的 CPython 3.13 对齐。 | `.github/workflows/smoke.yml` |
| P2-8 SEC-01 残留 | P2 | 权重校验慢路径改 `zipfile` 流式：直接从磁盘读中央目录，只有 `data.pkl` 一个成员进内存（上限 `_WEIGHTS_PKL_MAX=256MB`，防解压炸弹）；截头 zip 自愈只对 ≤ `_WEIGHTS_PATCH_MAX=64MB` 的文件启用，大文件缺头直接拒绝。单请求峰值从约 2× 文件体积压到 data.pkl 体积。 | `adr/models/gsv_engine.py` |
| P2-9 SEC-04 残留 | P2 | `docker/Dockerfile` 删除 `ENV ADR_ALLOW_UNAUTHENTICATED=1`；改由 `docker/docker-compose.yml` 显式传入（并注明安全前提：ports 只发布 `127.0.0.1:9881:9881`，对外发布须删该行并设 `ADR_TTS_API_KEY`）。直接 `docker run -p 9881:9881` 现在会因无 key 拒绝启动。 | `docker/Dockerfile`、`docker/docker-compose.yml` |
| P3-10 R3-PACK-02 | P3 | CI 两处 `pip install torch torchaudio` 钉到 `torch==2.10.0 torchaudio==2.10.0`（cpu index），与容器/便携包同版本，消除解析漂移。 | `.github/workflows/smoke.yml` |

### 8.2 验证

- 新增/调整用例：
  - `tests/test_server_api.py`：`test_native_readyz_reflects_engine_state`（就绪 200 / stalled 503 / 未就绪 503 且 `/health` 仍 200）、`test_auth_readyz_exempt`。
  - `tests/test_audit_training.py`：`test_save_checkpoint_atomic_keeps_backup`（两次保存后无 `.tmp`、`.bak` 为上一版且可加载）、`test_save_checkpoint_failure_keeps_previous_file`（写入中途抛错时原文件字节不变）。
  - `tests/test_data_pipeline.py`：`test_train_sample_load_rejects_object_arrays`（legacy object 数组 → `ValueError` 且提示 `allow_pickle`）。
  - `tests/test_gsv_stream_threading.py`：`test_coalesce_round_lock_wait_is_bounded`（持锁时合批有界抛 `EngineStalledError`）、`test_watchdog_grace_env_parsing`（默认 300 / `60`→60 / 非法→300 / `0`→关闭且不启线程）。
  - `tests/test_gsv_coalescer.py`：`FakeEngine` 补 `_engine_lock()`（与生产同语义的带超时 contextmanager）。
- 定向：`test_encoding_health / test_gsv_coalescer / test_data_pipeline` **43 passed, 1 skipped**（133.74s）；`test_server_api -k "readyz or health"`、`test_gsv_stream_threading -k "coalesce_round_lock_wait_is_bounded or watchdog_grace"`、`test_audit_training` 全绿。
- 全量：`python -m pytest -q --basetemp=.pytest_tmp` **912 passed, 1 skipped, 0 failed**（427.67s ≈ 7 分 8 秒）。
- `ruff check --select E4,E7,E9,F`：本轮 13 个改动文件仅剩既有告警，无新增。

### 8.3 需要人工拍板的取舍（即 5.1 的五条作者约定）

1. **看门狗默认 300s、退出码 75**：连续降级 5 分钟即 `os._exit(75)`，前提是外部有拉起机制（Docker `restart` 策略 / 桌面壳 `supervise`）。裸跑 `python -m adr.server` 的开发环境没有守护，进程退出后不会自动回来——如不接受可用 `ADR_ENGINE_WATCHDOG_SEC=0` 关闭或调大阈值。轮询 15s，因此实际触发区间约为 300~315s。**注意：触发源只有流式 `_mark_stalled`；非流式 solo/合批路径的盲区见 R3-CON-02。**
2. **`/readyz` 在首启预热期为 503**：这是设计语义（"还不能干活"），Docker HEALTHCHECK 已给 `start-period=600s`；若接入其它编排器需同步放宽启动宽限期，否则会被误判并反复重启。
3. **容器默认无 key 拒绝启动是破坏性变更**：官方 compose 已显式豁免（仅绑回环）。第三方 `docker run -p 9881:9881` 升级后会直接失败——这是有意的（把"无鉴权对外"从默认改成必须显式声明），需在发布说明中提示。
4. **legacy npz 必须迁移**：本机 `data/opencpop_npz/{test,train,samples}` 已全部迁移（**7512 个文件，0 失败**：test 206 / train 3550 / samples 3756）。外部旧数据集需先跑 `python scripts/migrate_npz_pickle.py <目录>`；未迁移时训练会以明确报错中止（不再静默空样本）。
5. **慢路径硬上限 256MB / 64MB**：`data.pkl` 超过 256MB 的权重会被拒（正常 GSV 权重远小于此）；超过 64MB 且缺 zip 头（截头）的文件不再自愈。如遇真实超大权重再评估，不建议为此放宽。
6. **`/readyz` 免鉴权**：容器内探针无 key，且响应不含任何路径信息；与 `/health` 同口径。
