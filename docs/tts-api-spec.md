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
| `3` | 真流式+定长块 (ADR 中与 1 行为一致) |
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

### 2.4 错误格式

所有业务错误统一 `400 + JSON`:

```json
{"message": "text is required"}
{"message": "tts failed", "Exception": "..."}
```

### 2.5 权重切换端点 (全局语义)

```
GET /api/v2/set_gpt_weights?weights_path=GPT_weights_v2/demo-e8.ckpt
GET /api/v2/set_sovits_weights?weights_path=SoVITS_weights_v2/demo-e8.pth
```

成功 `200 {"message": "success"}`; 路径缺失/加载失败
`400 {"message": "change gpt weight failed", ...}`。
热换幂等: 同路径重复切换不重复加载。

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
| `/profiles` | GET | 档案清单 (名称/prompt/风格/权重/创建时间) |
| `/profiles/{name}/ref` | GET | 下载档案参考音频 (audio/wav) |
| `/tts` | POST+GET | 按档案名合成 |

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
  "vits_weights": null
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
| `streaming_mode=3` 定长块 | 定长切片 | 按句切块 | 消费方按块播放无差异 |
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
