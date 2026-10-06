# ADR: Audio Disentangled Reconstruction

低资源声音克隆训练框架 — 用几分钟参考音频, 在消费级显卡 (4-8GB VRAM) 上完成声音克隆训练, 并提供 GPT-SoVITS 兼容的流式 TTS 服务。

## 功能特性

- **低资源训练**: LoRA / QLoRA 微调, 只训约 1% 参数, 显存占用降低约 70%; 梯度检查点 + 混合精度
- **GSV 兼容 TTS 服务**: 实现 GPT-SoVITS api_v3 双工流式协议 (`/api/v3/voices` + `/api/v3/tts/stream-input`), 流式返回 WAV 音频块
- **NEKO 零配置对接**: TTS provider 选 GPT-SoVITS, API 地址保持默认 `http://127.0.0.1:9881` 即可; 音色下拉自动列出 ADR 档案
- **数据处理管线**: 音频切片 / ASR 转写 / F0 提取 / 音素化一条龙 (`adr process`)
- **桌面壳**: Tauri 封装, 启动器 / 预热 / 服务守护 / 日志面板

## 快速开始

环境要求: Python 3.10+ (Windows 推荐), NVIDIA GPU 可选 (CPU 可推理, 训练建议 GPU)。

```powershell
# 1. 安装 (M1 基础 + WebUI)
pip install -e .[m1]

# 训练加速 (LoRA/QLoRA, 按需)
pip install -e .[m2]

# 2. 启动 TTS 服务 (默认端口 9881, NEKO 直连)
$env:NUMBA_CACHE_DIR = "E:\adr_numba_cache"   # 任意可写目录, 避免 numba 缓存落只读路径
python -m adr.server

# 3. 或启动 legacy WebUI (Gradio, 7860)
adr webui          # 等价于双击 启动WebUI.bat
```

### 声音克隆训练

```powershell
# 数据处理: 原始音频 -> 切片/转写/音素化 npz
adr process -i <音频文件或目录> -o ./output/processed

# LoRA 微调
adr train -d <数据目录> --lora --lora-rank 8
```

### 常用命令

| 命令 | 说明 |
| --- | --- |
| `python -m adr.server` | 启动 GSV 兼容 TTS 服务 (默认 127.0.0.1:9881, `-p` 改端口 / `-a` 改地址) |
| `adr train -d <dir> --lora` | LoRA 微调训练 |
| `adr process -i <audio>` | 数据处理管线 |
| `adr clone --ref <wav> --text <文本>` | 单句克隆推理 |
| `adr model list / download / info` | 预训练模型管理 |
| `adr doctor [--fix]` | 环境/数据体检 |
| `adr webui` | legacy WebUI (功能冻结期) |

## NEKO 对接

1. 启动 ADR TTS 服务: `python -m adr.server` (确认横幅显示监听端口, 默认 9881)
2. NEKO 设置 -> 语音合成 -> provider 选 **GPT-SoVITS**, API 地址填 `http://127.0.0.1:9881`
3. 刷新音色列表, ADR 档案会自动出现

注意: 首次合成有模型冷启动 (数十秒), 之后为流式秒级响应。若桌面壳提示 "9881 被占用", 说明端口被其他 GSV 服务占用, 改用壳提示的端口。

## 局域网访问与安全

服务默认只监听 `127.0.0.1`（仅本机可达，NEKO 本机对接不受影响）。需要局域网内其他设备调用时：

```powershell
$env:ADR_TTS_API_KEY = "<你的密钥>"     # 必须: 对外暴露必须同时配 API key
python -m adr.server --addr 0.0.0.0
```

- 监听 `0.0.0.0` 且未设置 `ADR_TTS_API_KEY` 时, 局域网内任何设备均可**无鉴权**调用合成 API（启动横幅会打安全警告）。
- API key 支持逗号分隔多个; 客户端凭据三选一: `Authorization: Bearer <key>` / `X-API-Key: <key>` 头 / `?api_key=<key>` 查询参数。

## 测试

```powershell
$env:NUMBA_CACHE_DIR = "E:\adr_numba_cache"
pytest
```

## 文档

- [docs/](docs/) — 设计文档、里程碑计划、批次记录 (m9-phase3-plan.md)
- [adr/configs/](adr/configs/) — 训练配置预设 (default / vram_4gb / vram_8gb / ...)

## License

[MIT](LICENSE)
