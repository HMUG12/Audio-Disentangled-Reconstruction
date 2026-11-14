# Runtime Profile 与模块 Manifest 规范（M0）

- 状态：**规范草案**，M1 只落地"声明 + 校验"，不改变现有安装行为
- 关联：[target-architecture.md](target-architecture.md) 第 2、6 节
- 现有实现：[build_runtime.ps1](../../scripts/build_runtime.ps1)、[requirements-linux.txt](../../docker/requirements-linux.txt)、[check_runtime.py](../../scripts/check_runtime.py)、[check_requirements.py](../../scripts/check_requirements.py)

---

## 1. 要解决什么

当前"一个包能跑"所需的信息分散在：`build_runtime.ps1` 的内嵌钉版清单、`docker/requirements-linux.txt`、`docker/Dockerfile` 的 torch 安装、`pyproject.toml` 的区间、`check_runtime.py` 的体检规则、`BUILD_VARIANT.txt` 的变体标记。缺一份**机器可读的声明**回答：

1. 这个 runtime 是哪个 profile、包含哪些组件、体积预算多少；
2. 某个功能（训练 / 多语言 / RVC / 控制台）依赖哪些 Python 包与资产；
3. 装完之后怎么自检、失败怎么定位。

Profile 与 Manifest 就是补这份声明，**不引入新的安装机制**。

## 2. Runtime Profile

Profile = 一次构建产物的完整描述。首阶段固定五个：

| Profile ID | 用途 | Python | torch | 依赖清单 | 体积预算 |
| --- | --- | --- | --- | --- | --- |
| `desktop-cpu` | Windows 安装包（默认） | 3.13.16 PBS | 2.10.0 (PyPI CPU) | `build_runtime.ps1` Step4 | ≤ 2.2 GB |
| `desktop-gpu` | Windows 安装包（GPU） | 3.13.16 PBS | 2.10.0+cu130 | 同上 + CUDA index | ≤ 4.6 GB |
| `docker-cpu` | 容器 CPU | 3.11-slim | 2.10.0 (cpu index) | `docker/requirements-linux.txt` | 不设限 |
| `docker-cuda` | 容器 CUDA | 3.11-slim | 2.10.0 (cu130 index) | 同上 | 不设限 |
| `dev` | 源码开发 | 任意 ≥3.10 | 任意受支持 | `pyproject.toml` 区间 | 不设限 |

Profile 之间**只允许**在以下字段不同：Python 版本、torch/torchaudio 的版本与 index、平台专属包、体积预算。其余必须在同一份共享清单内，防止"桌面能跑、容器缺包"（历史 Q4 就是 pandas 只存在于开发机）。

### 2.1 字段定义

| 字段 | 说明 | 现有对应物 |
| --- | --- | --- |
| `id` | 上述五个之一 | — |
| `python` | 版本 + 来源 + SHA256 | `build_runtime.ps1` `$Pbs*`；Dockerfile `FROM python:3.11-slim` |
| `torch` | 版本 / 变体 / index-url | 脚本 Step5、Dockerfile L57-63 |
| `requirements` | 钉版清单路径 | 两份清单 |
| `assets` | 需要随包/首启的资产 | `bin/ffmpeg`、`redist/vc_redist`、`pretrained_models`（首启） |
| `patches` | 上游补丁 | `scripts/apply_compat_patches.py` |
| `size_budget_gb` | 体检告警阈值 | `check_runtime.py` `WARN_TOTAL_GB_*` |
| `variant_flag` | 判据文件 | `BUILD_VARIANT.txt` |

## 3. `runtime.manifest.json`

放在构建产物根（`build/runtime-cpu/runtime.manifest.json`、容器内 `/app/runtime.manifest.json`），由构建脚本生成、由体检脚本校验。

```json
{
  "schema": 1,
  "profile": "desktop-cpu",
  "adr_version": "1.3.1",
  "python": {"version": "3.13.16", "source": "python-build-standalone",
             "tag": "20261009", "sha256": "286285F1..."},
  "torch": {"version": "2.10.0", "variant": "cpu",
            "index_url": "https://download.pytorch.org/whl/cpu"},
  "requirements": [{"file": "scripts/build_runtime.ps1#Step4", "pins": 47}],
  "gsv_commit": "d523079fc05d9a8028d6085bffe4a2757c32abb6",
  "modules": ["server-api", "tts-core", "console", "train", "multilang"],
  "assets": [
    {"name": "ffmpeg", "path": "bin/ffmpeg.exe", "required": true},
    {"name": "pretrained_models", "path": "third_party/gpt_sovits/pretrained_models",
     "required": true, "delivery": "bootstrap"}
  ],
  "size_budget_gb": 2.2
}
```

规则：

1. `schema` 是整数，破坏性变更才 +1；新增字段不升版。
2. `modules` 必须与模块 Manifest 中 `required: true` 的集合一致（见第 4 节），否则体检失败。
3. 构建脚本写清单，**不得由体检脚本"推断"**；推断只作为缺失清单时的降级。
4. 清单不含绝对路径、用户名、时间戳，保证可复现比对。

## 4. 模块 Manifest

ADR 内部把能力切成模块，每个模块声明它需要的包与资产。落地形态：单个 `adr/modules.toml`（或 JSON），随包分发。

| 模块 | 必需 | 依赖（Python） | 依赖（资产/外部） | 说明 |
| --- | --- | --- | --- | --- |
| `server-api` | 是 | fastapi、uvicorn、python-multipart | — | 低于此则服务起不来 |
| `tts-core` | 是 | torch、torchaudio、numpy、soundfile、librosa、transformers | `pretrained_models`、`bin/ffmpeg` | 合成主链 |
| `console` | 是 | （复用 server-api） | 静态页资源 | 控制台 API + 页面 |
| `train` | 否 | peft、bitsandbytes（m2 extras） | `scripts/*.py` 训练脚本、ASR 模型 | 缺失时 UI 显式提示不可用 |
| `multilang` | 否 | pyopenjtalk、jamo、ToJyutping、ko_pron | G2P 数据 | 缺失时回退中文 |
| `rvc` | 否 | — | `third_party/rvc` 权重 | 唱歌转换链 |
| `diffsinger` | 否 | — | `third_party/DiffSinger` + 音素词典 | 缺失时相关用例 skip |
| `updater` | 桌面是 / 容器否 | — | 更新清单 URL | 仅壳 |

字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 模块名 |
| `required` | 缺失时是"服务不可用"还是"功能降级" |
| `packages` | 导入名（用于 `importlib.util.find_spec` 自检，不用 pip 名） |
| `assets` | 路径 + 是否必需 |
| `degrade` | 缺失时用户可见的行为（如 `train` → "训练功能不可用"） |

自检入口（M1 落地）：`python -m adr doctor --modules`，输出每模块 OK/缺失/降级，复用 `console._doctor_engine_check` 与 `check_runtime.py` 的判定，不新增第二套探测逻辑。

## 5. 依赖解析规范

### 5.1 三层职责

| 层 | 文件 | 形式 | 用途 |
| --- | --- | --- | --- |
| 声明层 | `pyproject.toml` | 区间（`>=x,<y`） | 开发者 `pip install -e .` |
| 构建层 | `build_runtime.ps1` / `docker/requirements-linux.txt` | 精确 `==` | 出货产物 |
| 断言层 | `check_requirements.py` | 规则 | 区间必须覆盖所有钉版 |

### 5.2 规则

1. **只有 torch/torchaudio 允许跨 profile 不同**，且必须带变体后缀（`+cu130`）或走不同 index；其他任何包在两个 profile 钉不同版本，必须在清单注释里写明原因。
2. `pyproject` 的区间必须**包含**所有 profile 的钉版（现有门禁已做）。
3. M4 起追加**相等断言**：同一包在 `desktop-*` 与 `docker-*` 的钉版必须相等，除非在 `runtime.manifest.json` 的 `allowed_drift` 中登记。
4. 新增依赖必须同时更新：`pyproject` 区间、`requirements.txt`、至少一个 profile 的钉版、`modules.toml` 归属。四项缺一，`check_requirements.py` 失败。
5. 不允许隐式依赖兜底（历史 pandas 问题）：任何 `import` 的第三方包必须能在某个模块的 `packages` 中找到。
6. 可选模块的包只进 extras 与对应 profile；`required: false` 模块缺包时**不得**让服务启动失败。

### 5.3 体量与资产

- `bin/ffmpeg.exe`、`redist/vc_redist.x64.exe` 属于 `assets`，走 SHA256 钉版 + 缓存校验（现有机制）。
- 底模属 `delivery: bootstrap`，不入包；完整性判据是 HF tree API 的 `lfs.oid`。
- 体积预算来自 `runtime.manifest.json`，`check_runtime.py` 只做告警，不阻断（GPU 包天然超限）。

## 6. 渐进落地（不改现有行为）

| 步 | 动作 | 对用户的影响 |
| --- | --- | --- |
| 1 | 新增 `adr/modules.toml` + 自检命令 | 无（只读） |
| 2 | 构建脚本末尾生成 `runtime.manifest.json` | 无（多一个文件） |
| 3 | `check_runtime.py` 增加"清单存在且模块齐全"检查 | 出包更严 |
| 4 | `check_requirements.py` 增加相等断言 + `allowed_drift` | CI 更严 |
| 5 | 安装器读取清单做安装后自检（可选） | 安装最后多一步校验 |

## 7. 验收条件

1. `python -m adr doctor --modules` 在桌面 CPU 包、GPU 包、docker-cpu 三种环境输出全部必需模块 OK。
2. 故意卸载 `peft` 后，`train` 模块报降级，服务仍可合成（返回明确提示，不是 500）。
3. `check_requirements.py` 能对"只在 docker 清单新增包"报错。
4. 两个 profile 的 `runtime.manifest.json` 差异仅限第 2 节允许字段。
5. 清单中不出现绝对路径或用户名（可复现性检查）。
