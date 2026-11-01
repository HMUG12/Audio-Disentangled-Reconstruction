# 更新清单 update.json 规格（批次 73）

> 面向官网/发布维护者。客户端实现见 `desktop/src-tauri/src/update.rs`（`MANIFEST_URL` / `UpdateManifest`）。

## 1. 位置与要求

固定地址（客户端硬编码，不可改路径、不可重定向到 HTML）：

```
https://adr-studio.pages.dev/update.json
```

- 匿名可 GET，无需鉴权 / Cookie / Referer
- 返回 `200`，`Content-Type: application/json; charset=utf-8`
- **UTF-8 无 BOM**（带 BOM 会解析失败 → 客户端提示「更新清单格式错误」）
- 严格 JSON：无注释、无尾逗号
- 建议 `Cache-Control: max-age=300`；**每次发版后 purge 该文件**（客户端另有 24h 本地缓存，CDN 再叠一层会拖到次日）

安装包本身不放这里：Cloudflare Pages 单文件 25MB 上限放不下 1.7GB 包，包体托管 HF（主）+ 123 云盘（备）。

## 2. 格式

```json
{
  "version": "1.4.0",
  "notes": "1. 修复 GPU 包在无 N 卡机器上的启动崩溃\n2. 新增更新失败时的诊断详情",
  "packages": [
    {
      "variant": "cpu",
      "name": "ADR-Studio-1.4.0-x64-setup.exe",
      "url": "https://huggingface.co/<repo>/resolve/main/ADR-Studio-1.4.0-x64-setup.exe?download=true",
      "fallback_url": "https://<123云盘直链>/ADR-Studio-1.4.0-x64-setup.exe",
      "sha256": "7f3a…（64 位小写十六进制）",
      "size": 1834512384
    },
    {
      "variant": "gpu",
      "name": "ADR-Studio-1.4.0-x64-gpu-setup.exe",
      "url": "https://huggingface.co/<repo>/resolve/main/ADR-Studio-1.4.0-x64-gpu-setup.exe?download=true",
      "fallback_url": "https://<123云盘直链>/ADR-Studio-1.4.0-x64-gpu-setup.exe",
      "sha256": "…",
      "size": 4329181440
    }
  ]
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `version` | 是 | `1.4.0` / `v1.4.0` 均可；逐段数字比较，**大于**本机版本才提示更新；缺失或非数字段按 0 处理 |
| `notes` | 否 | 纯文本，多行用 `\n`，原样显示在更新弹窗 |
| `packages[]` | 是 | 每个平台包一项 |
| `packages[].variant` | 是 | 仅 `cpu` / `gpu` |
| `packages[].name` | 否 | 落盘文件名，仅取文件名部分（路径会被剥掉）；建议与安装包真实文件名一致 |
| `packages[].url` | 是 | 主下载直链（HF）。为空则「下载」按钮置灰，只保留 GitHub Releases 跳转 |
| `packages[].fallback_url` | 否 | 备用直链（123 云盘），主源失败自动切换；1.7GB 包强烈建议提供 |
| `packages[].sha256` | **是** | 64 位十六进制，大小写不敏感。**为空直接拒绝下载**（完整性硬门槛），不许留空占位 |
| `packages[].size` | 否 | 字节数，仅作进度条分母；缺失时读 `Content-Length` |

## 3. 下载链接的实测要求

- 必须是直链：`200` + 完整文件流，不能是网页 / 分享页 / JS 跳转
- 无 `Referer`、无登录态也能下载（客户端用裸 HTTP 请求）
- 允许 302/307（HF 会跳转，客户端自动跟随）
- 123 云盘解析直链若有有效期 / 防盗链，发版当天须用无 Referer 的 curl 实测

## 4. 发版顺序（不可颠倒）

1. 上传安装包到 HF 与 123 云盘，确认两条链接都能完整下载
2. 算 SHA256：`certutil -hashfile <安装包> SHA256`，与清单逐位核对
3. **最后**更新 `update.json` 的 `version` / `notes` / `packages`
4. purge CDN 上的 `update.json`

顺序颠倒会让先到的用户下到半成品并被判「校验不符」（客户端会删档重下，不会装坏，但体验差）。

## 5. 客户端行为（供参考，无需官网实现）

- 启动器打开时静默查一次；用户点「检查更新」强制绕过 24h 缓存
- 下载落 `%LOCALAPPDATA%\ADR\updates\`，边下边算哈希，校验通过才解锁「立即安装」
- **不自动安装**：必须用户点「立即安装」才静默执行（避免打断合成 / 训练）
- 先派 detached 批处理等桌面壳退出，再 `/VERYSILENT` 安装（Inno 带 AppMutex，壳在跑时静默装会被直接中止）
- 清单拉不到时退回 GitHub `releases/latest` 页面，不会误装
- `variant` 按本机 `runtime/BUILD_VARIANT.txt` 选择；无对应项时退回第一个带 `url` 的包（别只写一个包又写错 variant）
- 官网页面 HTML 不参与自动更新，改版无影响；下载页建议同时挂 HF 与 123 云盘入口，与清单保持一致
