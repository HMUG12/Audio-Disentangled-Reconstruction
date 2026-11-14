# ADR TTS 服务对接规范

> 批次 11 · 面向 N.E.K.O 与一切需要 TTS 服务的消费方
> 协议蓝本: GPT-SoVITS api_v2 (经 api_neko 验证的 N.E.K.O 消费面)

ADR 训练产物 (微调权重 + 音色档案) 通过 HTTP 服务对外提供。为最大化兼容,
同时暴露两个协议面:

| 协议面 | 前缀 | 适用场景 |
|---|---|---|
| **GSV api_v2 兼容层** | `/api/v2` | N.E.K.O、GPT-SoVITS 生态客户端 — **零改造接入** |
| **ADR 原生 API** | `/api/adr/v1` | 新接入方; 音色档案一等公民, 无需管理音频文件路径 |

## 1. 快速启动

```powershell
# 仓库根目录
python -m adr.server             # 默认 0.0.0.0:9881 (与 GSV api_v2 端口一致)
python -m adr.server -p 9880     # 自定义端口
```

| 环境变量 | 说明 |
|---|---|
| `ADR_TTS_DEFAULT_PROFILE` | 默认音色档案名; 请求未带 `ref_audio_path`/`profile` 时回退使用 |
| `ADR_TTS_API_KEY` | API Key (鉴权), 逗号分隔可配多个; **未设置则不鉴权** (默认, N.E.K.O 零改造) |

首次请求会加载引擎 (~10s 冷启动), 之后常驻; 权重支持请求级热换。
推理串行 (引擎内部锁), 并发请求排队 — 与 GSV api_v2 部署行为一致。

## 2. GSV api_v2 兼容层

### 2.1 `POST /api/v2/tts` (同时支持 GET 查询参数)

请求体 JSON, 与 GPT-SoVITS api_v2 同名同义。

**必填 (二选一):**

| 字段 | 类型 | 说明 |
|---|---|---|
| `ref_audio_path` | str | 参考音频绝对路径 (3-10s 干净人声) |
| `profile` | str | **ADR 扩展**: 音色档案名 (见 §5), 自动解析参考音频与权重 |
| `text` | str | 要合成的文本 (始终必填) |

**协议内支持 (透传引擎):**

| 字段 | 默认 | 说明 |
|---|---|---|
| `prompt_text` | `""` | 参考音频对应文本 |
| `text_lang` / `prompt_lang` | `zh` | 语种 (自动转小写) |
| `media_type` | `wav` | `wav` / `raw` / `ogg` / `aac` (ogg/aac 需 ffmpeg) |
| `streaming_mode` | `false` | `0/1/2/3` 或 bool, 语义见 §2.2 |
| `text_split_method` | `cut5` | 切句方式 (cut0/cut1/cut2/cut3/cut5...) |
| `speed_factor` | `1.0` | 语速 (仅非流式生效) |
| `seed` | `-1` | 随机种子; 流式传给首段 |
| `top_k` / `top_p` / `temperature` | `15/1.0/1.0` | 采样参数 (仅非流式生效) |
| `t2s_weights` / `vits_weights` | null | **ADR 扩展**: 请求级热换 GPT/SoVITS 权重 |

**接受但忽略 (ADR 引擎不支持, 仅为协议兼容):**

`aux_ref_audio_paths`, `batch_size`, `batch_threshold`, `split_bucket`,
`fragment_interval`, `parallel_infer`, `repetition_penalty`, `sample_steps`,
`super_sampling`, `overlap_length`, `min_chunk_length`

### 2.2 `streaming_mode` 语义 (与 api_neko 逐字对齐)

| 值 | 行为 |
|---|---|
| `0` / `false` | 非流式, 返回完整音频 |
| `1` / `true` | 分段流: 按句切分逐块返回 (ADR 流式引擎的本征模式) |
| `2` | 真流式 (ADR 中与 1 行为一致) |
| `3` | **mode 2 的别名** — 官方是"真流式+定长块", ADR 不实现定长切块, 按句切块行为同 2 |
| 其他 | `400 {"message": "the value of streaming_mode must be ..."}` |

注: JSON `true` (bool) 在 Python 中 `True == 1`, 自然落分支 1 — 与
GSV 官方 api_v2 行为一致。

### 2.3 流式响应字节格式 (客户端拼接契约)

```
第 1 块:  44 字节 WAV 头 (单声道/16bit/服务端采样率)
第 2+ 块: 裸 s16le PCM 小端块 (无任何封装)
```

客户端收到后顺序写入同一个 wav 文件 (或直接送播放器) 即可。

Python 拼接示例:

```python
import requests

r = requests.post("http://127.0.0.1:9881/api/v2/tts", json={
    "text": "这是流式合成测试",
    "profile": "my_voice",          # 或 ref_audio_path
    "streaming_mode": 1, "media_type": "wav",
}, stream=True)

with open("out.wav", "wb") as f:
    for chunk in r.iter_content(chunk_size=4096):
        f.write(chunk)              # 头 + 裸 PCM 顺序落盘即是合法 wav
```

> **非 wav 格式的流式限制 (批次88 / CON-02)**: 上述"头 + 裸块"契约只在
> `media_type=wav` 时成立。请求 `ogg` / `aac` / `raw` 等格式时, 当前实现是
> **每块各自独立编码成完整容器**, 顺序拼接不等于一个合法单流; 需要边收边播
> 或落盘成单文件时请用 `wav`, 由客户端自行转码。

### 2.4 错误格式

错误响应统一带 `message` + `code` 两个字段 (批次84)，`detail` 在 HTTPException
与 422 场景保留 (只增不改)：

| 状态码 | 场景 | 示例 |
|---|---|---|
| 400 | 参数/请求数据错误 | `{"message": "text is required", "code": "invalid_params"}` |
| 400 | 媒体类型不在白名单 | `{"message": "unsupported media_type: ...", "code": "invalid_params"}` |
| 401 | API Key 缺失或错误 | `{"detail": "unauthorized", "message": "unauthorized", "code": "unauthorized"}` |
| 404 | 音色档案不存在 | `{"message": "profile not found: xxx", "code": "profile_not_found"}` |
| 413 | 文本/请求体超上限 | `{"message": "text too long: ...", "code": "payload_too_large"}` |
| 422 | 请求体字段校验失败 | `{"detail": [...], "message": "text: Field required", "code": "validation_error"}` |
| 500 | 服务端内部故障 | `{"message": "tts failed", "Exception": "...", "code": "internal_error"}` |
| 503 | 依赖缺失/设备/模型不可用 | `{"message": "tts failed", "Exception": "...", "code": "dependency_missing"}` |
| 503 | 合成队列已满 | `{"message": "busy: synthesis queue full", "code": "busy"}` |

`Exception` 字段仅在 v2 合成失败时出现 (GSV wire 兼容, 逐字保留)。客户端应按
`code` 决定是否重试 —— 500/503 是服务端问题, 不要当成 400 参数错误反复重试。

码值单点：v2 / v3 WS / OpenAI 三套外壳的 `code` 全部取自
`adr/server/exceptions.py` 的 `WIRE_CODES`（M3-a 收口），协议文件不再写字面量。
媒体类型错误沿用历史值 `invalid_params`（语义键 `unsupported_media` 只登记不换值），
调用方按字符串匹配的既有逻辑零改动。

### 2.5 权重切换端点 (全局语义)

```
GET  /api/v2/set_gpt_weights?weights_path=GPT_weights_v2/demo-e8.ckpt
GET  /api/v2/set_sovits_weights?weights_path=SoVITS_weights_v2/demo-e8.pth
POST /api/v2/set_gpt_weights?weights_path=GPT_weights_v2/demo-e8.ckpt
POST /api/v2/set_sovits_weights?weights_path=SoVITS_weights_v2/demo-e8.pth
```

成功 `200 {"message": "success"}`; 路径缺失/安全预检不通过
`400 {"message": "change gpt weight failed", ...}`; 加载阶段的依赖/设备/权重
损坏等内部故障按 §2.4 分流为 500/503 (批次84)。

幂等与缓存 (批次77): 幂等键是 `(绝对路径, mtime_ns, size)` —— 同一路径的
文件被覆盖 (音色包导入 / 训练产物替换) 会重新做安全预检并重载; 换权成功后
自动清空 TTS 缓存, 不会继续返回旧人声。

浏览器防护 (批次77): 这类有副作用的端点默认拒绝浏览器跨站触发 ——
`Sec-Fetch-Site: cross-site` 且 Origin 不在白名单 (`tauri://localhost` /
`http://tauri.localhost` / 本机 localhost 页 / `ADR_CORS_ORIGINS`) 的请求回
`403`。`<img>` / 顶层导航 / no-cors fetch 这类不带 Origin 的跨站 GET 同样被拒。
不带 `Sec-Fetch-*` 头的后端客户端 (N.E.K.O / curl / SDK) 不受影响;
配置了 `ADR_TTS_API_KEY` 时由鉴权兜底, 该防护让位。

### 2.6 缓存版本与失效规则

音频缓存键由 `adr/server/tts_cache.py` 的 `CACHE_VERSION` 参与构造，落盘文件名
也带该前缀；读到其他版本的文件一律丢弃。因此：

1. **任何会改变音频字节的编码参数变更都必须 bump `CACHE_VERSION`** ——
   包括采样率/位深、wav 头写法、mp3/ogg/aac 编码器参数、流式分片拼接方式。
2. 仅改文案、错误消息、日志格式等不影响音频字节的变更 **不要** bump，
   否则会无谓清空用户缓存。
3. bump 后的旧缓存文件在下次扫描时被清理，不会继续命中；已有单测
   （`test_m3_contract.py::test_cache_version_bump_invalidates_old_key`）兜底。
4. `CACHE_VERSION` 是内部实现细节，不属于 HTTP 契约，不需要同步给调用方。

## 3. N.E.K.O 对接步骤

1. 启动本服务 (默认端口即 GSV 标准端口 9881);
2. N.E.K.O 的 TTS 后端类型选 GPT-SoVITS (api_v2), 地址填
   `http://127.0.0.1:9881` — **无需任何改造**;
3. 参考音频二选一:
   - 传统方式: `ref_audio_path` 指向 `data/voices/<name>/ref.wav`;
   - 推荐方式: 请求带 `"profile": "<档案名>"` (无需传音频路径);
4. 权重切换: N.E.K.O 调 `set_gpt_weights`/`set_sovits_weights` 端点时,
   ADR 会预热对应权重 (等价于 GSV 官方行为)。

示例请求 (N.E.K.O 等价发出):

```json
POST /api/v2/tts
{
  "text": "你好世界",
  "text_lang": "zh",
  "ref_audio_path": "data/voices/demo/ref.wav",
  "prompt_text": "参考音频的文本",
  "prompt_lang": "zh",
  "media_type": "wav",
  "streaming_mode": 1
}
```

## 4. ADR 原生 API (`/api/adr/v1`)

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 服务/引擎/当前权重状态 |
| `/profiles` | GET | 档案清单 (名称/ref_audio 绝对路径/prompt/风格/权重/创建时间) |
| `/profiles/{name}/ref` | GET | 下载档案参考音频 (audio/wav) |
| `/tts` | POST+GET | 按档案名合成 |
| `/system/stats` | GET | 控制台总览: GPU/内存/磁盘/引擎状态/训练状态 |
| `/call` | GET | 调用控制台页 (双端点参数调试 + 代码片段生成) |

`/system/stats` 引擎三态字段: `engine_loading` (后台预热中) /
`engine_ready` (_tts 权重就绪可合成) / `engine_stage` 预热细分阶段
`queued → importing → loading → kernel → ready`, 失败为 `failed`。
消费方探活推荐 `/health`; 需要区分"服务活着但引擎没好"时看 `engine_stage`。

原生合成请求:

```json
POST /api/adr/v1/tts
{
  "text": "你好世界",
  "profile": "demo",
  "prompt_text": null,        // 可选, 覆盖档案默认
  "speed_factor": 1.0,
  "seed": -1,
  "media_type": "wav",
  "streaming_mode": false,    // true → 分段流 (§2.3 字节格式)
  "t2s_weights": null,        // 可选请求级热换
  "vits_weights": null,
  "top_k": 15,
  "top_p": 1.0,
  "temperature": 1.0,
  "text_split_method": "cut1" // cut0 不切 / cut1 凑四句 / cut3 按句 / cut5 按标点
}
```

`streaming_mode: true` 映射为 v2 分支 1 (分段流); 原生 API 固定
`media_type` 白名单与错误格式同 §2。

curl 示例:

```bash
curl http://127.0.0.1:9881/api/adr/v1/profiles
curl -OJ http://127.0.0.1:9881/api/adr/v1/profiles/demo/ref
curl -X POST http://127.0.0.1:9881/api/adr/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好", "profile": "demo"}' --output out.wav
```

## 5. 音色档案 (profile) 机制

档案目录: `data/voices/<name>/`

```
ref.wav    参考音频 (3-10s 干净人声)
meta.json  {"prompt_text", "t2s_weights", "vits_weights",
            "rvc_weights", "rvc_index", "style", "created_at"}
```

带 `profile` 请求时: 参考音频取 `ref.wav`; `prompt_text`、微调权重
(`t2s_weights`/`vits_weights`, 可指向 ADR 微调产物) 自动带上 —
请求里显式给出的字段优先于档案值。档案名不存在 → `400 unknown profile`。

建档方式: WebUI 录音建档, 或 `adr.models.voice_library.save_voice()`。

## 6. 与 GPT-SoVITS 官方 api_v2 的差异 (诚实清单)

| 项 | 官方 api_v2 | ADR | 影响 |
|---|---|---|---|
| 忽略参数表 (§2.1) | 生效 | 接受但忽略 | 客户端无感; 批处理/步数类参数无效 |
| `streaming_mode=3` 定长块 | 定长切片 | **mode 2 别名, 按句切块** | 消费方按块播放无差异; 需要定长块请自行切分 |
| 语种校验 | 严格枚举 | 宽松 (透传引擎) | 错语种在引擎层报错 (400 tts failed) |
| v3 task 队列 + 双 WebSocket (`/tts/stream*`) | 有 (api_neko v3 面) | **未实现** | 需要队列/双向流的消费方请用 api_v2 轮询/流式 |
| 权重切换 | 全局重载 | 全局预热, 同路径幂等 | 更快; 语义一致 |
| 端口 | 9880 (GSV 官方) / 9881 (api_neko) | 9881 | N.E.K.O 默认配置直连 |

> N.E.K.O 的消费面是 api_v2 (tts.py), 不依赖 v3 队列/WS — 上述缺口不影响对接。

## 7. 安全与鉴权

### 7.1 默认行为 (局域网信任环境, 与 GSV 生态一致)

不设置 `ADR_TTS_API_KEY` 时**不鉴权**, CORS 全开 — N.E.K.O 及一切
GSV 生态客户端零改造直连。

### 7.2 API Key 鉴权 (推荐公网/半信任环境启用)

```powershell
$env:ADR_TTS_API_KEY = "my-secret-key"          # 单 key
$env:ADR_TTS_API_KEY = "key-a,key-b"            # 多 key (发给不同消费方)
python -m adr.server
```

启用后所有端点要求凭据, 三种携带方式等价:

| 方式 | 示例 |
|---|---|
| Authorization 头 | `Authorization: Bearer my-secret-key` |
| X-API-Key 头 | `X-API-Key: my-secret-key` |
| 查询参数 | `/api/v2/tts?text=...&api_key=my-secret-key` |

- key **精确匹配** (大小写敏感); 未通过 → `401 {"message": "unauthorized"}`
- 豁免路径: `GET /api/adr/v1/health` (监控探活免凭据)
- CORS 预检 (OPTIONS) 不要求凭据, 浏览器前端可正常携带凭据跨域调用

curl 示例:

```bash
curl -H "Authorization: Bearer my-secret-key" \
     -X POST http://127.0.0.1:9881/api/v2/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "你好", "profile": "demo"}' --output out.wav
```

### 7.3 其余注记

`ref_audio_path`/权重切换端点接受服务器本地路径 — 等同于把文件系统
部分读权交给调用方, 请勿对不可信方开放 (公网部署建议: 反向代理 +
HTTPS + 限定 key 消费方)。

边界（批次78 / CODE_REVIEW #11）:

- **v2 `ref_audio_path` 有意保持本地绝对路径**：这是 N.E.K.O 等既有客户端
  的配置契约（零改造红线），仅保留「存在时必须是普通文件 + 大小上限」校验。
- **控制台训练/建档收白名单**：`POST /api/adr/v1/train/start`、
  `POST /api/adr/v1/profiles/create` 的 `audio_path` 必须位于
  `output/uploads/` 内且是普通文件，否则统一返回
  `400 音频不存在或不可用: 请先通过 /api/adr/v1/train/upload 上传 ...`。
  越界 / 不存在 / 非普通文件共用同一文案，不构成磁盘路径存在性探针
  （控制台页面本身先上传再调用，不受影响）。
- ffmpeg 转码子进程带超时（`ADR_FFMPEG_TIMEOUT_SEC`，默认 120 秒），
  超时强杀并返回可读错误，不会永久占用线程池线程。

