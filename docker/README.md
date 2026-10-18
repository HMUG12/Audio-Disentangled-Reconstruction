# ADR 容器版 (Docker)

> v1.2 多形态分发: 纯 HTTP 服务镜像, 无 WebUI。克隆/训练/合成全部通过 HTTP API 或容器内 CLI 完成。

## 形态说明

- 镜像入口: `python -m adr.server --addr 0.0.0.0 --port 9881`, 暴露五套 API 面
  (ADR native `/api/adr/v1`、v2 兼容、v3 兼容、OpenAI 兼容、console)
- **无 Gradio WebUI** (与 GSV 官方 WebUI 无关); 纯服务形态, 面向集成方
- GPT-SoVITS 钉版: 构建时容器内 clone commit
  `d523079fc05d9a8028d6085bffe4a2757c32abb6`, 并自动应用 24 处兼容补丁
  (`scripts/apply_compat_patches.py`)
- 双镜像变体: `adr-server:1.2.0-cpu` (默认) / `adr-server:1.2.0-cuda` (cu130 wheel)

## 系统要求

- Docker Desktop (Windows/macOS) 或 Docker Engine + Compose v2
- CUDA 变体: NVIDIA GPU + 驱动 (Windows 用 Docker Desktop WSL2 后端即内置 GPU
  支持; Linux 需 NVIDIA Container Toolkit)
- 磁盘: 镜像数 GB (CUDA 变体更大) + 权重约 2GB + 业务数据

## 快速开始 (CPU)

```bash
cd docker
docker compose up -d --build      # 首次构建较慢 (拉镜像 + pip 依赖 + GSV clone)
docker compose logs -f            # 等待 "Uvicorn running"
curl http://127.0.0.1:9881/api/adr/v1/health
```

health 返回恒 200, `engine_ready` 字段表示引擎加载状态 (首启下载权重期间为 false)。

## 快速开始 (GPU)

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

镜像 tag 变为 `adr-server:1.2.0-cuda`, 并透传全部 GPU。无 GPU 的机器误用 CUDA
镜像时, 引擎 auto 自检会回退 CPU 运行 (不会崩溃)。

验证 GPU 生效:

```bash
docker compose exec adr python -c "import torch; print(torch.cuda.is_available())"
```

## 已发布镜像 (ghcr.io)

tag `v*` 推送自动触发 GitHub Actions 构建双变体并发布至 ghcr.io，无需本地构建：

```bash
docker pull ghcr.io/hmug12/adr-server:1.2.0-cpu
docker pull ghcr.io/hmug12/adr-server:1.2.0-cuda

# 常青 tag: 始终指向最新发布版
docker pull ghcr.io/hmug12/adr-server:cpu
```

直接运行（环境变量与数据卷语义同本地构建，见下文）：

```bash
docker run -d --name adr-server -p 9881:9881 -v adr-data:/data \
  ghcr.io/hmug12/adr-server:1.2.0-cpu
```

## 数据卷

| 挂载点 | 内容 | 类型 |
|---|---|---|
| `/data` (`ADR_DATA_DIR`) | 档案 / 输出 / TTS缓存 / HF缓存 | bind mount → `docker/data/` |
| `/app/third_party/gpt_sovits/pretrained_models` | GSV 预训练权重 (首启下载 ~1.4GB) | named volume |
| `/app/third_party/gpt_sovits/text/G2PWModel` | G2PW 中文注音模型 (首启下载 ~560MB) | named volume |

- 权重引导幂等: 中断后 `docker compose restart` 续传, named volume 保留已下载部分
- 清理权重: `docker compose down -v` (业务数据在 `docker/data/`, 不受影响)

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `ADR_TTS_API_KEY` | 空 | 设置后所有 API 需鉴权; 暴露到局域网/公网必须设置 |
| `ADR_TTS_DEVICE` | auto | 引擎自检: 有 GPU 用 cuda, 否则 cpu (镜像内不写死) |
| `ADR_PORT` | 9881 | HEALTHCHECK 依据; 自定义容器内端口时需同步设置 |

## 端口与鉴权

容器监听 `0.0.0.0:9881` 且无鉴权时, 启动日志会打警告——容器场景这是预期行为。
仅本机使用可改端口映射为 `127.0.0.1:9881:9881`; 对外暴露务必设置 API key:

```yaml
# 追加到 docker-compose.yml 的 services.adr 下
    environment:
      ADR_TTS_API_KEY: "your-secret"
```

调用方携带 `Authorization: Bearer <ADR_TTS_API_KEY>`。

## 构建脚本 (可选)

```powershell
# 仓库根执行
powershell -File scripts/build_docker.ps1                                    # CPU
powershell -File scripts/build_docker.ps1 -Cuda                              # CUDA
powershell -File scripts/build_docker.ps1 -Tag myreg/adr:1.2.0 -Cuda         # 自定义 tag
powershell -File scripts/build_docker.ps1 -ProxyPrefix "https://ghproxy.net/"  # GSV clone 加速
```

## 网络受限 / 代理

构建过程访问三类源: PyPI (依赖包)、download.pytorch.org (torch 轮子)、github.com
(GSV 钉版 clone)。

- GitHub clone 慢/失败: `-ProxyPrefix "https://ghproxy.net/"` (或任意同类加速前缀)
- apt update 失败 (deb.debian.org 走 HTTP/80, 部分代理不覆盖):
  `-AptMirror "mirrors.tuna.tsinghua.edu.cn"` (compose 里取消
  `APT_MIRROR` 注释)
- pip 慢: 自行改 `docker/Dockerfile` 换镜像源, 或在 Docker Desktop
  Settings → Resources → Proxies 配全局代理

## 常见问题

### Q: 容器内如何跑 CLI (克隆/训练)?

```bash
docker compose exec adr adr --help
```

CLI 子命令详见仓库根 README。

### Q: 想升级 GSV 版本?

修改 `docker/Dockerfile` 的 `GSV_COMMIT` 构建参数 (钉版保证可复现; 新版本可能
需要调整 `scripts/apply_compat_patches.py` 的补丁点)。
