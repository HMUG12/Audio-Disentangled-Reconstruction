# ADR 全仓库代码审查报告

- 审查日期：2026-11-07
- 审查范围：整个仓库（`adr/` 约 93 个 Python 文件、2 万行；另含 `desktop/`、`scripts/`、`docker/`、`.github/`、`tests/`、依赖与配置文件）
- 审查方式：**仅静态审查，未运行训练、未构建安装包、未运行测试，也未做依赖 CVE 扫描**
- 说明：凡无法仅凭静态代码确认的结论均标注「推测」，不应视为已确认事实

## 总体结论

项目已经具备相当完整的工程化基础：分层清晰（core / data / models / training / inference / server / webui）、安全加固痕迹明显（API Key、Origin 守卫、路径安全、缓存上限、队列限流、构建期版本/合规/运行时门禁），测试量约 1 万行。主要风险不在“缺功能”，而在**两套推理/训练栈并存导致的数据域不一致**，以及**若干会在生产环境放大成服务卡死或远程代码执行的边界问题**。

最值得优先修的五点：

1. 流式合成 worker 在“有界队列 + 锁内投递 + 无超时收割”组合下可能永久持锁，拖死全部合成。
2. 训练用 mel 与推理/声码器用 mel 不是同一套算法，属于“训完音色不对”的根因。
3. 桌面更新链把前端回传的包信息当作可信输入，且 CSP 关闭，存在任意程序静默安装链。
4. 默认不设 API Key 时，带副作用的 GET 接口可被任意网页触发。
5. TTS 磁盘缓存在全局锁内做最大 256MB 读写，且跑在事件循环线程上，会拖慢整个服务。

## 建议修复顺序

按“收益 / 改动成本”排序，先做低成本高收益项，再做需要动架构的项。

| 顺序 | 对应问题 | 事项 | 理由 | 预估成本 |
|---|---|---|---|---|
| 1 | #4 | 有副作用接口改 POST + 补无 Origin 校验 | 一处中间件/路由改动即可关闭跨站触发面 | 低 |
| 2 | #6 | 权重幂等改为“路径 + mtime/size”，换权后清 TTS 缓存 | 同时解决安全与“换权无效”的用户可见问题 | 低 |
| 3 | #10、#23 | 修正训练数据目录层级；锁定依赖版本清单 | 纯配置/路径修正，能立即消除“0 样本”和版本漂移 | 低 |
| 4 | #11、#14 | 训练/建档路径白名单；ffmpeg 加超时 | 小改动，消除任意路径读取与线程池耗尽风险 | 低 |
| 5 | #5 | 缓存锁外做磁盘 IO，`get/put` 走线程池 | 解决全服务停顿，收益高 | 中 |
| 6 | #2 | 训练统一 `compute_mel_bigvgan`，mel 与 F0 解耦 | 直接决定微调后的音质与音色一致性 | 中 |
| 7 | #7、#8、#9 | 补梯度累积收尾 flush、修 DataLoader collate、修极小数据集划分 | 训练正确性问题，改动集中在 trainer/dataset | 中 |
| 8 | #1 | 重构流式队列/锁/清理生命周期 | 收益极高，但涉及并发模型，需要配套回归测试 | 中高 |
| 9 | #15、#16、#17 | 下载校验、音色包原子提交、参考音频原子写 | 数据完整性/安全，改动面中等 | 中 |
| 10 | #3、#20、#21 | 更新链签名与校验、安装器清理、禁止 PATH 回落 | 安全收益最高，但涉及发布流程与安装器联调 | 高 |

---

# 严重

## 1. 流式合成可能永久持锁，导致全部合成饿死

**位置**：[gsv_engine.py:804](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L804)、[gsv_engine.py:866](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L866)、[gsv_engine.py:899-903](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L899-L903)

**问题说明**：合成队列 `maxsize=8`；producer 在 `with self._lock:` 与 `_gsv_context()` 内执行 `q.put`。消费端变慢或断连后停止迭代时，producer 阻塞在 put 且仍持有引擎锁。清理段 `q.get()` 没有超时；`worker.join(timeout=30)` 超时后函数直接返回，但 worker 可能仍存活并继续持锁。

**影响**：一个慢客户端或半开连接即可让后续所有合成（包括非流式接口）永久阻塞；表现为服务“假死”，只能重启。

**修改建议**：改为无界队列或在锁外投递；清理使用带 deadline 的 `q.get(timeout=...)`；join 超时后将引擎标记为不可用并强制重建模型，不要继续复用。

## 2. 训练 mel 与推理 mel 算法不一致

**位置**：[data/pipeline.py:230-239](file:///e:/新创意构思/新建文件夹/ADR/adr/data/pipeline.py#L230-L239)、[inference/pipeline.py:340-343](file:///e:/新创意构思/新建文件夹/ADR/adr/inference/pipeline.py#L340-L343)、[audio.py:313-321](file:///e:/新创意构思/新建文件夹/ADR/adr/utils/audio.py#L313-L321)

**问题说明**：训练侧使用 `compute_mel`（`power=1.0`、`center=True`、`fmax=sr/2`、无 slaney norm）；推理/声码器侧使用 `compute_mel_bigvgan`（`fmax=8000`、slaney、`center=False`）。`audio.py` 的注释已明确说明 v1 mel 与 BigVGAN 不匹配。另外 mel 计算被包在 `if self.config.enable_f0` 内，`--no-f0` 时 `mel=None`。

**影响**：训练域与声码器域整体错位，微调后的音色/音质会明显退化；关闭 F0 时样本 mel 为空，可能变成全 0 目标。

**修改建议**：训练侧统一改用 `compute_mel_bigvgan`，统一采样率与 hop；mel 计算与 F0 开关解耦。

## 3. 桌面更新链的信任边界不成立

**位置**：[update.rs:467](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/src/update.rs#L467)、[update.rs:504](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/src/update.rs#L504)、[update.rs:524-526](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/src/update.rs#L524-L526)、[update.rs:26](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/src/update.rs#L26)、[tauri.conf.json:10-13](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/tauri.conf.json#L10-L13)

**问题说明**：`download_update` 接收前端回传的整个 `UpdatePackage`（含 `url`、`sha256`、`name`）；安装时只校验 `is_file()`，随后通过 `.bat` 执行该路径。更新清单仅靠 TLS，没有签名验签；同时 `csp: null` 且 `withGlobalTauri: true`。

**影响**：一旦 WebView 出现 XSS 或清单/CDN 被篡改，攻击者可让客户端下载并静默安装任意程序，形成完整的远程代码执行链。

**修改建议**：包选择、URL、哈希全部由 Rust 端依据签名清单决定；安装路径必须限定在 `updates/` 目录内并重算 SHA256；清单增加 Ed25519/minisign 验签；启用最小 CSP 并评估关闭全局 Tauri 对象。

## 4. 默认无鉴权 + 有副作用 GET 可被跨站触发

**位置**：[origin_guard.py:98-103](file:///e:/新创意构思/新建文件夹/ADR/adr/server/origin_guard.py#L98-L103)、[settings.py:69-72](file:///e:/新创意构思/新建文件夹/ADR/adr/core/settings.py#L69-L72)、[v2_compat.py:433-460](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L433-L460)、[console.py:555-556](file:///e:/新创意构思/新建文件夹/ADR/adr/server/console.py#L555-L556)

**问题说明**：Origin 守卫只在请求带 `Origin` 时校验；默认 `ADR_TTS_API_KEY` 为空。此时 `GET /set_gpt_weights`、`GET /set_sovits_weights`、`GET /system/doctor?fix=true` 均可被浏览器 `<img>` 或顶层导航直接触发。

**影响**：任意网页可让本机服务切换全局模型、裁改文件（`doctor fix`）；虽然读不到响应，但副作用足够造成拒绝服务或状态污染。

**修改建议**：所有有副作用接口改 POST，并校验 `Sec-Fetch-Site`；无 Origin 的请求也拒绝非只读方法；换权/修复接口加一次性 CSRF token。

## 5. TTS 缓存的磁盘 IO 在全局锁和事件循环线程内执行

**位置**：[tts_cache.py:147-190](file:///e:/新创意构思/新建文件夹/ADR/adr/server/tts_cache.py#L147-L190)、[v2_compat.py:299-300](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L299-L300)

**问题说明**：`get` 在 `with _lock` 内 `read_bytes`；`put` 在锁内 `write_bytes` / `os.replace` / `getsize`，单文件可达 256MB。`v2_compat` 又在 `async def` 中同步调用 `tts_cache.get`。

**影响**：一次大缓存写入会阻塞所有缓存命中请求，并占用事件循环线程，`/health` 和流式首包一起停顿。

**修改建议**：锁只保护 `OrderedDict` 和字节账目；磁盘读写移到锁外，`get/put` 用 `asyncio.to_thread`；同 key 加细粒度锁避免重复写。

## 6. 权重热换的校验只在“路径变化”时执行，换权后缓存不失效

**位置**：[gsv_engine.py:687-696](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L687-L696)、[voice_package.py:308-316](file:///e:/新创意构思/新建文件夹/ADR/adr/models/voice_package.py#L308-L316)、[v2_compat.py:433-481](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L433-L481)

**问题说明**：`validate_weights_file` 仅在 `_loaded != vw` 时调用。若音色包导入用 `os.replace` 覆盖同一路径的已加载权重，后续请求会跳过校验并进入第三方 `torch.load`（**推测**：需实际覆盖流程成立）。另外换权成功后没有 `tts_cache.clear()`。

**影响**：安全上可能绕过权重预检；功能上同一文本在换权后仍返回旧人声音频，用户会认为“换权无效”。

**修改建议**：幂等判断改为 `(路径, mtime, size/hash)`；禁止覆盖已加载权重文件；换权成功后清空 TTS 缓存。

---

# 一般

## 7. 梯度累积余数被静默丢弃

**位置**：[trainer.py:336-369](file:///e:/新创意构思/新建文件夹/ADR/adr/training/trainer.py#L336-L369)

**问题说明**：只在 `(step+1) % grad_accum_steps == 0` 时 step；epoch 末尾不足一组时，下个 epoch 开头 `zero_grad()` 直接丢弃这批梯度，但 `epoch_loss` 仍统计它。

**影响**：每个 epoch 最后一组梯度白算；梯度累积步数与数据量不整除时训练更新次数偏少。

**修改建议**：epoch 结束后若仍有累积，补一次 clip + step + scheduler.step。

## 8. Windows 下 `num_workers>0` 会直接失败

**位置**：[trainer.py:165-174](file:///e:/新创意构思/新建文件夹/ADR/adr/training/trainer.py#L165-L174)

**问题说明**：`collate_fn=lambda ...` 在 spawn 模式下不可 pickle，且闭包捕获整个 Trainer。默认 `num_workers=0` 掩盖了该问题。

**影响**：用户一旦调大 `num_workers`，Windows 上训练启动即报 PicklingError。

**修改建议**：改为模块级函数或 `self.train_data.collate` 绑定方法。

## 9. 极小数据集会得到空训练集

**位置**：[dataset.py:316-318](file:///e:/新创意构思/新建文件夹/ADR/adr/training/dataset.py#L316-L318)

**问题说明**：`n_val = max(1, int(len(indices) * val_ratio))`；1 条样本时 val=1、train=0，配合 `drop_last=True` 与 `total_steps=0`，训练直接空转。

**影响**：单样本微调场景静默失败或 scheduler 退化，用户只看到“训练完成”但权重未更新。

**修改建议**：`len < 2` 时报错或回退为“全量训练 + 同集验证”。

## 10. 训练数据目录层级与 CLI 帮助不一致

**位置**：[data/pipeline.py:272-276](file:///e:/新创意构思/新建文件夹/ADR/adr/data/pipeline.py#L272-L276)、[dataset.py:242](file:///e:/新创意构思/新建文件夹/ADR/adr/training/dataset.py#L242)、[cli.py:543-577](file:///e:/新创意构思/新建文件夹/ADR/adr/cli.py#L543-L577)

**问题说明**：数据流水线写入 `<output>/samples/*.npz`；数据集只 `glob("*.npz")` 非递归；而 CLI 帮助示例是 `adr train -d output/processed`。

**影响**：按文档执行会加载 0 个样本，训练无法开始。

**修改建议**：`glob("**/*.npz")`，或在 CLI 统一追加 `samples` 子目录并同步帮助文案。

## 11. 训练/建档接口把任意本地路径当音频

**位置**：[console.py:384-387](file:///e:/新创意构思/新建文件夹/ADR/adr/server/console.py#L384-L387)、[console.py:668-678](file:///e:/新创意构思/新建文件夹/ADR/adr/server/console.py#L668-L678)、[v2_compat.py:137-153](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L137-L153)

**问题说明**：仅 `exists()`，不校验是否位于 `UPLOAD_DIR` 内、是否为普通文件；错误文案还会构成任意路径存在性探针。

**影响**：本机任意进程可把任意文件当训练素材/参考音频送入后续处理；也便于探测磁盘路径。

**修改建议**：统一走 `pathsafe.resolve_within(UPLOAD_DIR, path)` 并检查 `is_file()`。

## 12. v2 兼容层把所有内部异常统一成 400

**位置**：[v2_compat.py:336-340](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L336-L340)、[v2_compat.py:451](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L451)、[v2_compat.py:477](file:///e:/新创意构思/新建文件夹/ADR/adr/server/v2_compat.py#L477)

**问题说明**：除 OOM/NLTK 外全部返回 400 `"tts failed"`；ffmpeg 缺失、磁盘满、权重损坏都被归为参数错误。

**影响**：客户端会把服务端故障当请求错误反复重试，放大负载并掩盖真实原因。

**修改建议**：仅参数错误保留 400，其余改 500/503。

## 13. 错误响应存在三套外壳

**位置**：[app.py:100-104](file:///e:/新创意构思/新建文件夹/ADR/adr/server/app.py#L100-L104)、[auth.py:110-116](file:///e:/新创意构思/新建文件夹/ADR/adr/server/auth.py#L110-L116)

**问题说明**：`HTTPException` 返回 `{"detail","code"}`，鉴权中间件返回 `{"message"}`，422 仍是 FastAPI 默认 `{"detail":[...]}`，而 v2/v3 统一 `{"message","code"}`。

**影响**：客户端按 `message` 取错误文案时，401/422 取不到，错误提示可能为空。

**修改建议**：至少给 401/422 补 `message` 字段，形成单一错误契约。

## 14. 音频转码子进程没有超时

**位置**：[audio_codec.py:79](file:///e:/新创意构思/新建文件夹/ADR/adr/server/audio_codec.py#L79)、[audio_codec.py:132](file:///e:/新创意构思/新建文件夹/ADR/adr/server/audio_codec.py#L132)、[audio_codec.py:157](file:///e:/新创意构思/新建文件夹/ADR/adr/server/audio_codec.py#L157)

**问题说明**：三处 `process.communicate(input=...)` 均无 `timeout`。

**影响**：ffmpeg 卡死会永久占用线程池线程，逐步耗尽工作线程，最终所有请求排队。

**修改建议**：`communicate(timeout=...)` + 超时 `kill()`。

## 15. 预训练权重下载无哈希、固定临时名、无并发锁

**位置**：[gsv_bootstrap.py:100-120](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_bootstrap.py#L100-L120)

**问题说明**：只校验 `Content-Length`，不校验 sha256；临时文件固定为 `dest.part`；`ensure()` 无进程级锁。

**影响**：权重可能被中间人替换后直接解压；并发预热会写坏同一 `.part`（并发场景为**推测**，固定路径可确认）。

**修改建议**：清单内置 sha256；临时名带 uuid；`ensure` 加文件锁。

## 16. 音色包导入不是原子提交

**位置**：[voice_package.py:308-316](file:///e:/新创意构思/新建文件夹/ADR/adr/models/voice_package.py#L308-L316)

**问题说明**：依次 `os.replace` 权重、ref、meta；中途失败只清理未落位的 `.tmp`，已替换的权重/ref 会与旧 meta 混成半新档案。

**影响**：导入失败后音色档案处于不一致状态，可能出现“名字在但音频/权重不匹配”。

**修改建议**：提交前备份旧文件，失败逆序回滚；或先写 meta 再替换并在 finally 恢复。

## 17. 参考音频写入非原子，扫描临时目录被共享

**位置**：[voice_library.py:80](file:///e:/新创意构思/新建文件夹/ADR/adr/models/voice_library.py#L80)、[voice_library.py:443](file:///e:/新创意构思/新建文件夹/ADR/adr/models/voice_library.py#L443)

**问题说明**：`shutil.copy2` 直写最终 `ref.wav`，中断会留半截音频；固定 `_sweep_tmp` 目录被并发建档共享。

**影响**：并发保存音色时可能互相覆盖/删除对方正在使用的临时音频，产生坏档案。

**修改建议**：先写唯一 `.tmp` 再 `os.replace`；`_sweep_tmp` 改为 uuid 子目录并在 `finally` 清理。

## 18. 段缓存 key 不包含参考音频版本与首段种子

**位置**：[gsv_runtime.py:34-43](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_runtime.py#L34-L43)、[gsv_engine.py:842-847](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L842-L847)

**问题说明**：key 只含 `ref_audio` 路径；`gsv_engine` 的 key 也未包含 `head_seed`。

**影响**：同名档案重建 `ref.wav` 后会继续回放旧参考音频生成的缓存；改首段种子不生效。

**修改建议**：key 加入 ref 文件 `mtime+size`（或内容哈希）与 `head_seed`。

## 19. 占位声码器输出时长错误，且 WavQuality 默认使用它

**位置**：[inference/pipeline.py:452-464](file:///e:/新创意构思/新建文件夹/ADR/adr/inference/pipeline.py#L452-L464)、[callbacks.py:198-208](file:///e:/新创意构思/新建文件夹/ADR/adr/training/callbacks.py#L198-L208)

**问题说明**：`wav = linear[:, 0, :]` 返回 mel 帧数而非采样点数，按 22050Hz 保存会短约 256 倍；`WavQualityCallback` 默认 `use_placeholder_vocoder=True`。

**影响**：若启用占位声码器，质量早停会基于假音频计算 RMS/centroid/F0，可能在错误的 epoch 停止。

**修改建议**：占位路径按 hop 上采样或直接拒绝输出；质量回调默认改用真实声码器。

## 20. 安装器双变体共用 AppId 且无任何清理

**位置**：[installer.iss:15-47](file:///e:/新创意构思/新建文件夹/ADR/scripts/installer/installer.iss#L15-L47)、[installer.iss:125](file:///e:/新创意构思/新建文件夹/ADR/scripts/installer/installer.iss#L125)

**问题说明**：同一 AppId 支持 CPU/GPU 包原地升级，但无 `[InstallDelete]`；静默升级后 `skipifsilent` 不会拉起应用。

**影响**：推测 GPU→CPU 覆盖后旧 `torch/nvidia_*` 库残留，`BUILD_VARIANT.txt` 与实际不符；更新完成后用户看到应用关闭且不重启。

**修改建议**：升级时清空 `runtime`（或用 Inno `[Code]` 按变体清理）；更新完成后显式启动应用。

## 21. 打包态 PATH 回落 `python` 存在劫持风险

**位置**：[lib.rs:1591](file:///e:/新创意构思/新建文件夹/ADR/desktop/src-tauri/src/lib.rs#L1591)

**问题说明**：未找到便携 runtime 时回落 `PathBuf::from("python")`，而 `current_dir` 是探测出的仓库根。Windows 会搜索应用目录/cwd，同名 `python.exe` 可能被优先执行。

**影响**：推测：安装目录或当前目录被投放同名 `python.exe` 时，客户端会执行劫持程序。

**修改建议**：打包态找不到 `runtime/python/python.exe` 直接报错，禁止 PATH 回落。

## 22. CI 镜像发布没有测试门禁

**位置**：[docker-publish.yml:3-39](file:///e:/新创意构思/新建文件夹/ADR/.github/workflows/docker-publish.yml#L3-L39)

**问题说明**：`v*` tag 直接构建推送，不依赖 smoke job，也未校验 tag 与 `pyproject.toml` 版本一致。

**影响**：带缺陷的提交可直接产出正式镜像；tag 与包内版本可能不一致。

**修改建议**：镜像发布 `needs: smoke`，并复用 `check_versions.py`。

## 23. 依赖声明与运行时版本漂移

**位置**：[requirements.txt:6-17](file:///e:/新创意构思/新建文件夹/ADR/requirements.txt#L6-L17)

**问题说明**：全部是 `>=` 下限（如 `torch>=2.1.0`），但打包 runtime 使用固定 torch/Python 版本，Docker 与 Windows runtime 的 Python 版本也不同。

**影响**：不同机器/CI 解析到不同版本，出现“本地能跑、装机失败”或行为差异。

**修改建议**：开发依赖给出一份 lock/上限，构建与 CI 复用同一份版本清单。

---

# 建议

> 原始审查中建议项为无序列表，此处按原顺序补编号 24–35，便于追溯。

## 24. 回调 list 被就地污染

**位置**：[trainer.py:211-212](file:///e:/新创意构思/新建文件夹/ADR/adr/training/trainer.py#L211-L212)

**问题说明**：`self.callbacks = callbacks or []` 后继续 append，直接修改调用方传入的 list。

**影响**：同一 list 传给第二个 Trainer 会累积重复回调，指标/保存逻辑重复执行。

**修改建议**：改为 `list(callbacks or [])`。

## 25. 两套早停实现并存且状态不落盘

**位置**：[trainer.py:254](file:///e:/新创意构思/新建文件夹/ADR/adr/training/trainer.py#L254)、[callbacks.py:87-124](file:///e:/新创意构思/新建文件夹/ADR/adr/training/callbacks.py#L87-L124)

**问题说明**：`best_val_loss` 每次 `fit()` 重置；回调侧另有一套 best/counter，判定条件不同，且不写入 checkpoint。

**影响**：续训会丢失早停基准，可能用更差的权重覆盖 best.pt；两种机制行为不一致。

**修改建议**：统一为一种实现，并把 best/counter 写入 checkpoint。

## 26. 段缓存字节重复计数

**位置**：[gsv_runtime.py:61-69](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_runtime.py#L61-L69)

**问题说明**：同 key 覆盖前未扣减旧 `len(pcm)`，直接累加新值。

**影响**：账目虚高，提前逐出仍然有效的缓存条目，缓存命中率下降。

**修改建议**：写入前 `pop` 旧值并回减字节数。

## 27. 缓存目录口径不一致

**位置**：[hub.py:76](file:///e:/新创意构思/新建文件夹/ADR/adr/models/hub.py#L76)

**问题说明**：硬编码 `~/.cache/adr/pretrained`，与 `adr_data_dir()` 的其他用途脱节；`.part` 仍为固定名。

**影响**：用户设置的数据目录不生效，模型缓存分散；并发下载仍可能写坏同一 `.part`。

**修改建议**：统一使用 `adr_data_dir()`，临时名唯一化。

## 28. 实验加速模块默认不可达且占用资源

**位置**：[adr_t2s_static_kv.py:34-224](file:///e:/新创意构思/新建文件夹/ADR/adr/models/adr_t2s_static_kv.py#L34-L224)、[adr_t2s_bridge.py:20-63](file:///e:/新创意构思/新建文件夹/ADR/adr/models/adr_t2s_bridge.py#L20-L63)、[gsv_coalescer.py:377-380](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_coalescer.py#L377-L380)

**问题说明**：模块级 `_last_kv` 长期持有整组 KV buffer；`_bridge_cache` 只增不删；但两者都由环境变量默认关闭。

**影响**：一旦开启，切换音色会持续增加显存占用直至 OOM；默认关闭时又容易让人误以为已生效。

**修改建议**：加 LRU/显式释放；启动日志明确记录“未启用/已启用”。

## 29. Diffsinger 全局 hparams 污染

**位置**：[diffsinger_engine.py:141-169](file:///e:/新创意构思/新建文件夹/ADR/adr/models/diffsinger_engine.py#L141-L169)

**问题说明**：直接修改 `hp["pndm_speedup"]`，不传 `speedup` 时不会恢复默认。

**影响**：后续请求永久沿用上一次的加速设置，跨请求参数污染（legacy webui 路径）。

**修改建议**：记录默认值，每请求显式恢复或构造 hparams 副本。

## 30. 两套推理栈 / 两套 UI 长期并存

**位置**：[inference/pipeline.py](file:///e:/新创意构思/新建文件夹/ADR/adr/inference/pipeline.py)、[gsv_engine.py](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py)、[webui](file:///e:/新创意构思/新建文件夹/ADR/adr/webui/__init__.py)、[console.py](file:///e:/新创意构思/新建文件夹/ADR/adr/server/console.py)

**问题说明**：ADR 自研推理与 GSV 封装各有一套预处理/声码器；Gradio `webui` 与 FastAPI `console` 功能重叠；`diffsinger/rvc/melody_bridge` 仅被 legacy webui 引用。

**影响**：同一问题需要修两处，极易出现只修一条路径的回归。

**修改建议**：明确主路径并标注 experimental，逐步下线未维护分支。

## 31. 配置默认值重复

**位置**：[config.py](file:///e:/新创意构思/新建文件夹/ADR/adr/core/config.py)、[trainer.py:40-81](file:///e:/新创意构思/新建文件夹/ADR/adr/training/trainer.py#L40-L81)

**问题说明**：两处各维护一套 warmup/accum 等默认值，`configs/*.yaml` 只做部分映射。

**影响**：通过不同入口启动会得到不同训练行为，排查困难。

**修改建议**：单一默认源，YAML 做完整映射或在启动时打印最终生效配置。

## 32. 控制台 async 接口内同步 IO

**位置**：[console.py:490-495](file:///e:/新创意构思/新建文件夹/ADR/adr/server/console.py#L490-L495)

**问题说明**：训练状态读取、模型清单磁盘检查均在事件循环线程执行。

**影响**：磁盘慢时会短暂阻塞所有 HTTP 请求。

**修改建议**：改普通 `def`（FastAPI 自动丢线程池）或 `asyncio.to_thread`。

## 33. 停止训练忽略 graceful 语义

**位置**：[console.py:267-275](file:///e:/新创意构思/新建文件夹/ADR/adr/server/console.py#L267-L275)

**问题说明**：`taskkill /T /F` 无条件强杀；模型下载无超时/停止端点。

**影响**：训练进程可能来不及保存检查点；下载卡住后状态永久 `downloading`，后续下载全部 409。

**修改建议**：先温和终止再升级强杀；下载增加超时与停止接口。

## 34. 全局 socket 超时副作用

**位置**：[gsv_engine.py:521](file:///e:/新创意构思/新建文件夹/ADR/adr/models/gsv_engine.py#L521)

**问题说明**：`socket.setdefaulttimeout(5)` 会改变进程内所有 socket/HTTP 客户端的默认超时。

**影响**：其他网络请求可能意外超时，行为难以预测。

**修改建议**：只对 nltk 下载局部设置超时。

## 35. Windows 临时文件写法存在风险（推测）

**位置**：[asr.py:117-119](file:///e:/新创意构思/新建文件夹/ADR/adr/data/asr.py#L117-L119)

**问题说明**：在 `NamedTemporaryFile(delete=False)` 仍打开时向同一路径 `sf.write`。

**影响**：推测：Windows 上可能触发 `PermissionError`，且 `finally` 的 `os.unlink` 也会因占用失败，异常被吞。

**修改建议**：退出 `with` 后再写文件，并在 `finally` 中清理；建议在 Windows CI 上补一条回归测试。
