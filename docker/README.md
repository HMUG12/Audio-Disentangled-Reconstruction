# ADR Docker 快速上手

> **Plug-and-play**: 一行命令起 WebUI,无需手动装 CUDA/PyTorch/FFmpeg

## 系统要求

- **Docker**: 20.10+
- **Docker Compose**: v2.0+
- **NVIDIA Container Toolkit**: [安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **GPU**: 任意 NVIDIA 显卡 (4GB+ VRAM)
- **磁盘**: 10GB+ 可用空间

## 一键启动

```bash
# 1. 构建并启动 (首次约 10-20 分钟)
docker-compose up -d --build

# 2. 查看日志
docker-compose logs -f

# 3. 打开浏览器
# http://localhost:7860
```

## 常用命令

```bash
# 进入容器
docker-compose exec adr bash

# 在容器内运行 CLI
docker-compose exec adr adr clone --ref /workspace/ref.wav --text "你好"

# 停止
docker-compose down

# 删除所有数据 (慎用)
docker-compose down -v
```

## 常见问题

### Q1: 构建时下载 PyTorch 太慢

修改 `docker/Dockerfile`,换用国内镜像:
```dockerfile
RUN pip install torch==2.1.0 torchaudio==2.1.0 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: GPU 不可用

检查 NVIDIA Container Toolkit:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Q3: 显存不够

修改 `docker-compose.yml` 环境变量:
```yaml
environment:
  - ADR_PRESET=vram_4gb   # 改为 4GB 档
```

### Q4: 端口冲突

修改 `docker-compose.yml` 端口映射:
```yaml
ports:
  - "8888:7860"  # 主机端口:容器端口
```

## 性能参考

| 显存 | 配置档 | 训练速度 | 推理速度 |
|---|---|---|---|
| 4GB | vram_4gb | ~3x 实时 | ~5x 实时 |
| 6GB | vram_6gb | ~2x 实时 | ~3x 实时 |
| 8GB | vram_8gb | ~1x 实时 | ~2x 实时 |
| 12GB+ | vram_12gb+ | 0.5x 实时 | 1x 实时 |
