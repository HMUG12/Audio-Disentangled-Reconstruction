//! 批次73: 客户端更新链 + 包型互提示。
//!
//! 设计口径 (用户拍板):
//! - 更新包托管: HF 主选 + 123 云盘备用 — 清单每个包同时给 `url` 与
//!   `fallback_url`, 主链下载失败自动切备用 (1.7GB 包单源失败率不低)
//! - 安装触发: 下载 + SHA256 校验通过后**用户点「立即安装」**才执行 —
//!   不自动装, 免得在合成/训练中途打断
//! - 清单只放官网静态 JSON (Cloudflare Pages 单文件 25MB 上限, 装不下安装包)
//! - 24h 缓存: 启动器每次打开都查一次会把官网打成热点, 也没有意义
//!
//! 包型互提示: `build_runtime.ps1` 在 runtime 根写 `BUILD_VARIANT.txt`
//! (cpu/gpu), 壳读它 + 探测 NVIDIA 驱动服务, 不匹配时启动器顶部横幅提示换包
//! (GPU 包装到无 N 卡机器 = 起不来; CPU 包装到有 N 卡机器 = 白白慢 10 倍)。

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::Manager;   // get_webview_window (进度 eval 落点)

#[cfg(windows)]
use std::os::windows::process::CommandExt;

/// 更新清单地址 (官网 Cloudflare Pages 静态文件; 结构见下方 `UpdateManifest`)
pub const MANIFEST_URL: &str = "https://adr-studio.pages.dev/update.json";
/// 版本页兜底 (清单拉不到 / 无对应包时「去下载」落点)
pub const RELEASES_URL: &str =
    "https://github.com/HMUG12/Audio-Disentangled-Reconstruction/releases/latest";
/// 清单缓存有效期
const CACHE_TTL_SECS: u64 = 24 * 3600;
/// 单个安装包下载进度回调粒度 (字节)
const PROGRESS_STEP: u64 = 1 << 20;
/// 无 Content-Length / 清单未给 size 时的下载硬上限 (批次88 / SEC-07)。
const MAX_DOWNLOAD_BYTES: u64 = 8 << 30;

// ---------------------------------------------------------------------------
// 清单结构
// ---------------------------------------------------------------------------

/// 清单里的单个安装包。字段全带 default: 官网手改 JSON 漏字段时只跳过该包,
/// 不让整份清单解析失败。
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct UpdatePackage {
    /// "cpu" | "gpu"
    #[serde(default)]
    pub variant: String,
    /// 安装包文件名 (落盘名, 仅取 basename 防目录穿越)
    #[serde(default)]
    pub name: String,
    /// 主下载地址 (HF)
    #[serde(default)]
    pub url: String,
    /// 备用下载地址 (123 云盘)
    #[serde(default)]
    pub fallback_url: Option<String>,
    /// 小写或大写十六进制 SHA256 (校验时忽略大小写)
    #[serde(default)]
    pub sha256: String,
    /// 字节数 (仅用于进度条总量; 缺失时读 Content-Length)
    #[serde(default)]
    pub size: u64,
}

#[derive(Clone, Default, Deserialize, Serialize)]
pub struct UpdateManifest {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub packages: Vec<UpdatePackage>,
    /// 批次76 (#3): 清单 Ed25519 签名 (hex)。签的是 `signing_payload()` 的
    /// 逐字节内容, 落款私钥只在发布机 (~/.adr/update_signing_key.pem)。
    /// **缺失即拒绝** — 没有签名的清单等于让远端任意指定要装哪个 exe。
    #[serde(default)]
    pub signature: String,
}

/// 磁盘缓存: 原文整体存下来, 离线时还能用上次的清单提示包型下载链接。
#[derive(Default, Deserialize, Serialize)]
struct UpdateCache {
    checked_at: u64,
    /// 清单原始 JSON 文本
    manifest: String,
}

// ---------------------------------------------------------------------------
// 纯函数 (单测覆盖)
// ---------------------------------------------------------------------------

/// "v1.3.2" / "1.3.2" / "1.3.2-beta" 归一成数字段。非数字段按 0 处理。
fn version_parts(v: &str) -> Vec<u64> {
    v.trim()
        .trim_start_matches(['v', 'V'])
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u64>().unwrap_or(0))
        .collect()
}

/// a 是否比 b 新 (逐段比较, 缺段补 0: 1.3 == 1.3.0)。
pub fn version_gt(a: &str, b: &str) -> bool {
    let (pa, pb) = (version_parts(a), version_parts(b));
    let n = pa.len().max(pb.len());
    for i in 0..n {
        let x = pa.get(i).copied().unwrap_or(0);
        let y = pb.get(i).copied().unwrap_or(0);
        if x != y {
            return x > y;
        }
    }
    false
}

/// 按包型挑包; 找不到该变体时退回第一个有 url 的包 (官网只发单包时也能用)。
pub fn pick_package<'a>(m: &'a UpdateManifest, variant: &str) -> Option<&'a UpdatePackage> {
    m.packages
        .iter()
        .find(|p| p.variant == variant && !p.url.is_empty())
        .or_else(|| m.packages.iter().find(|p| !p.url.is_empty()))
}

/// 文件名清洗: 只取 basename, 去掉 Windows 非法字符 (清单由人维护, 不可信)。
pub fn safe_file_name(name: &str) -> String {
    let base = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    let cleaned: String = base
        .chars()
        .map(|c| if r#"<>:"/\|?*"#.contains(c) { '_' } else { c })
        .collect();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        "adr_update_setup.exe".to_string()
    } else {
        cleaned
    }
}

/// 从 URL 末段猜文件名 (清单没写 name 时兜底)。
pub fn file_name_from_url(url: &str) -> String {
    let no_query = url.split('?').next().unwrap_or(url);
    let base = no_query.rsplit('/').next().unwrap_or("");
    if base.is_empty() {
        String::new()
    } else {
        base.to_string()
    }
}

/// 包型互提示判定 (纯函数): 返回 "" / "gpu_pkg_no_gpu" / "cpu_pkg_has_gpu"。
pub fn hint_kind(variant: &str, has_nvidia: bool) -> &'static str {
    match (variant, has_nvidia) {
        ("gpu", false) => "gpu_pkg_no_gpu",
        ("cpu", true) => "cpu_pkg_has_gpu",
        _ => "",
    }
}

// ---------------------------------------------------------------------------
// 批次76 (#3): 清单验签 — 更新链的信任根
// ---------------------------------------------------------------------------
//
// 威胁模型: 清单是官网上的一个静态 JSON。若它被篡改 (CDN/账号/中间人),
// 旧实现会照单全收 url + sha256, 前端再整包回传给 install_update —— 等于
// "云端说装什么就装什么", 是一条完整 RCE 链。现在:
//   1. 清单必须带 Ed25519 签名, 缺失/不匹配一律拒绝 (验签失败不降级)
//   2. 包选择 / 包地址 / 哈希全部由 Rust 侧从**已验签**清单里取, 前端不参与
//   3. 安装前重算 SHA256 并与清单比对, 路径必须落在 <数据目录>\updates\ 内
// 私钥只在发布机 (~/.adr/update_signing_key.pem), 签名单生成见
// scripts/sign_update_manifest.py; 公钥下面的 hex 与它成对, 换钥 = 老客户端
// 全部拒绝新清单, 只能随新版本一起换。

/// 清单验签公钥 (Ed25519, 32 字节 hex)
const UPDATE_PUBKEY_HEX: &str =
    "a7afe6393380f2efd5606d70035c87ba06df3cb11e68b77fc2cfcc2e3117c965";

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    if s.is_empty() || s.len() % 2 != 0 {
        return None;
    }
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len() / 2);
    let mut i = 0;
    while i < b.len() {
        let hi = (b[i] as char).to_digit(16)?;
        let lo = (b[i + 1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    Some(out)
}

fn sha256_hex_bytes(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(data))
}

/// 待签名文本 (逐字节稳定, 不受 JSON 排版/字段顺序影响)。
/// 改动这里必须同步 scripts/sign_update_manifest.py, 否则全部清单验签失败。
fn signing_payload(m: &UpdateManifest) -> Vec<u8> {
    let mut s = String::from("adr-update-v1\n");
    s.push_str(m.version.trim());
    s.push('\n');
    s.push_str(&sha256_hex_bytes(m.notes.as_bytes()));
    s.push('\n');
    for p in &m.packages {
        s.push_str(&format!(
            "{}|{}|{}|{}|{}|{}\n",
            p.variant.trim(),
            p.name.trim(),
            p.url.trim(),
            p.fallback_url.as_deref().unwrap_or("").trim(),
            p.sha256.trim().to_ascii_lowercase(),
            p.size
        ));
    }
    s.into_bytes()
}

/// 用指定公钥验签 (单测注入测试公钥; 生产走下面的 `verify_manifest`)。
fn verify_manifest_with(m: &UpdateManifest, pubkey_hex: &str) -> Result<(), String> {
    let sig = m.signature.trim();
    if sig.is_empty() {
        return Err("更新清单未签名, 已拒绝 (安全策略: 见 docs/update-manifest.md §6)".into());
    }
    let sig_bytes = hex_decode(sig).ok_or("更新清单 signature 不是合法十六进制")?;
    let pk = hex_decode(pubkey_hex).ok_or("内置更新公钥损坏")?;
    ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, pk)
        .verify(&signing_payload(m), &sig_bytes)
        .map_err(|_| "更新清单签名校验失败 (清单被篡改或签错钥)".to_string())
}

fn verify_manifest(m: &UpdateManifest) -> Result<(), String> {
    verify_manifest_with(m, UPDATE_PUBKEY_HEX)
}

// ---------------------------------------------------------------------------
// 本机环境: 包型 / 显卡 / 缓存路径
// ---------------------------------------------------------------------------

/// 清单缓存文件: 与 shell.json 同目录 (%LOCALAPPDATA%\ADR)。
fn cache_path() -> PathBuf {
    crate::config::shell_settings_path()
        .parent()
        .map(|p| p.join("update_cache.json"))
        .unwrap_or_else(|| std::env::temp_dir().join("adr_update_cache.json"))
}

/// 当前包型: `<exe目录>/runtime/BUILD_VARIANT.txt` (缺失 → "unknown" 不提示;
/// 旧版本安装包没有该文件, 宁可不提示也别误报)。
/// 开发调试可用 ADR_BUILD_VARIANT=cpu|gpu 覆盖。
pub fn package_variant() -> String {
    if let Ok(v) = std::env::var("ADR_BUILD_VARIANT") {
        let v = v.trim().to_ascii_lowercase();
        if v == "cpu" || v == "gpu" {
            return v;
        }
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let Some(dir) = exe_dir else {
        return "unknown".into();
    };
    let txt = std::fs::read_to_string(dir.join("runtime").join("BUILD_VARIANT.txt"))
        .unwrap_or_default();
    match txt.trim().to_ascii_lowercase().as_str() {
        "cpu" => "cpu".into(),
        "gpu" => "gpu".into(),
        _ => "unknown".into(),
    }
}

/// 是否装了 NVIDIA 显卡: 查内核显示驱动服务 nvlddmkm (注册表, 零外进程开销)。
#[cfg(windows)]
pub(crate) fn has_nvidia_gpu() -> bool {
    let mut cmd = std::process::Command::new("reg");
    cmd.args(["query", r"HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm"]);
    cmd.creation_flags(crate::system::CREATE_NO_WINDOW);
    cmd.output().map(|o| o.status.success()).unwrap_or(false)
}

#[cfg(not(windows))]
pub(crate) fn has_nvidia_gpu() -> bool {
    false
}

// ---------------------------------------------------------------------------
// 清单读写
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn read_cache() -> Option<UpdateCache> {
    let s = std::fs::read_to_string(cache_path()).ok()?;
    serde_json::from_str(&s).ok()
}

fn write_cache(manifest: &str) {
    let c = UpdateCache { checked_at: now_secs(), manifest: manifest.to_string() };
    if let Ok(s) = serde_json::to_string(&c) {
        let p = cache_path();
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(p, s);
    }
}

/// 拉取清单原文 (先看 24h 缓存, force 则强制联网)。
/// 联网失败但手上有旧缓存时退回旧缓存 (离线也能提示有新版 + 给下载链接)。
fn fetch_manifest_text(force: bool) -> Result<(String, bool), String> {
    let cache = read_cache();
    if !force {
        if let Some(c) = &cache {
            if now_secs().saturating_sub(c.checked_at) < CACHE_TTL_SECS
                && !c.manifest.trim().is_empty()
            {
                return Ok((c.manifest.clone(), true));
            }
        }
    }
    // 批次88 (SEC-07): 清单同样加读超时, 且最多只读 512KB —
    // 旧实现 into_string() 会把恶意/异常源的响应全量读进内存。
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(20))
        .build();
    match agent.get(MANIFEST_URL).call() {
        Ok(resp) => {
            let mut body = String::new();
            let read = resp
                .into_reader()
                .take(512 * 1024)
                .read_to_string(&mut body);
            match read {
                Ok(_) if !body.trim().is_empty() => {
                    write_cache(&body);
                    Ok((body, false))
                }
                Ok(_) => Err("更新清单为空".into()),
                Err(e) => Err(format!("读取更新清单失败: {e}")),
            }
        }
        Err(e) => {
            if let Some(c) = cache.filter(|c| !c.manifest.trim().is_empty()) {
                return Ok((c.manifest, true));
            }
            Err(format!("连接更新服务器失败: {e}"))
        }
    }
}

fn parse_manifest(body: &str) -> Result<UpdateManifest, String> {
    serde_json::from_str::<UpdateManifest>(body).map_err(|e| format!("更新清单格式错误: {e}"))
}

/// 备用包地址 (包型提示「去下载另一包」用, 只读缓存不联网)。
fn other_variant_url(variant: &str) -> String {
    let want = if variant == "gpu" { "cpu" } else { "gpu" };
    let url = read_cache()
        .and_then(|c| parse_manifest(&c.manifest).ok())
        // 未验签的缓存不给任何地址 (批次76 #3)
        .filter(|m| verify_manifest(m).is_ok())
        .and_then(|m| {
            m.packages
                .iter()
                .find(|p| p.variant == want && !p.url.is_empty())
                .map(|p| p.url.clone())
        });
    url.unwrap_or_else(|| RELEASES_URL.to_string())
}

/// 拉清单 + 解析 + **验签** (批次76 #3: 唯一合法入口, 调用方拿到的必然是
/// 已签名清单; 未签名/验签失败直接 Err, 不降级)。
fn fetch_verified_manifest(force: bool) -> Result<(UpdateManifest, bool), String> {
    let (body, from_cache) = fetch_manifest_text(force)?;
    let m = parse_manifest(&body)?;
    if m.version.trim().is_empty() {
        return Err("更新清单缺少 version 字段".into());
    }
    verify_manifest(&m)?;
    Ok((m, from_cache))
}

// ---------------------------------------------------------------------------
// 命令: 检查更新
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn check_update(app: tauri::AppHandle, force: Option<bool>) -> Result<serde_json::Value, String> {
    let current = app.package_info().version.to_string();
    let (m, from_cache) = fetch_verified_manifest(force.unwrap_or(false))?;
    let available = version_gt(&m.version, &current);
    let variant = package_variant();
    let pkg = if available { pick_package(&m, &variant) } else { None };
    let pkg_json = pkg
        .map(|p| serde_json::to_value(p).unwrap_or(serde_json::Value::Null))
        .unwrap_or(serde_json::Value::Null);
    let ok = serde_json::json!({
        "available": available,
        "current": current,
        "latest": m.version,
        "notes": m.notes,
        "variant": variant,
        "package": pkg_json,
        // 主备均无 url 时前端「下载」按钮置灰, 只给版本页跳转
        "package_ready": !pkg_json["url"].as_str().unwrap_or("").is_empty(),
        "cached": from_cache,
        "releases_url": RELEASES_URL,
    });
    crate::lifecycle::proc_log("update", &format!(
        "检查更新: 当前 {current} / 清单 {} / 可用 {available}{}",
        m.version, if from_cache { " (缓存)" } else { "" }
    ));
    Ok(ok)
}

// ---------------------------------------------------------------------------
// 命令: 下载 + SHA256 校验
// ---------------------------------------------------------------------------

fn sha256_hex(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path).map_err(|e| format!("打开下载文件失败: {e}"))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("读取下载文件失败: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// 向前端推一次进度 (window.__adrUpdate 由 launcher.html 挂载)。
fn eval_progress(window: &Option<tauri::WebviewWindow>, got: u64, total: u64, attempt: usize) {
    let Some(w) = window else { return };
    let pct = if total > 0 { (got as f64 * 100.0 / total as f64).min(100.0) } else { 0.0 };
    let payload = serde_json::json!({
        "got": got, "total": total, "percent": pct, "attempt": attempt,
    });
    let _ = w.eval(&format!(
        "window.__adrUpdate && window.__adrUpdate({});",
        payload
    ));
}

/// 单源下载 (流式落盘 + 进度)。失败返回 Err, 由调用方切备用源。
fn download_one(
    app: &tauri::AppHandle,
    url: &str,
    dest: &Path,
    total_hint: u64,
    attempt: usize,
) -> Result<u64, String> {
    let window = app.get_webview_window("main");
    // 批次88 (SEC-07): 只设连接超时挡不住"连上后无限慢/无限吐"的源 —
    // 加读超时 (两次 read 之间的空窗) 与体积上限 (声明 size 的 110%)。
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(60))
        .build();
    let resp = agent
        .get(url)
        .call()
        .map_err(|e| format!("连接失败: {e}"))?;
    let total = resp
        .header("content-length")
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(total_hint);
    let cap = if total > 0 {
        total.saturating_add(total / 10)
    } else {
        MAX_DOWNLOAD_BYTES
    };
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(dest).map_err(|e| format!("创建文件失败: {e}"))?;
    let mut buf = vec![0u8; 128 * 1024];
    let mut got: u64 = 0;
    let mut last_emit: u64 = 0;
    let mut last_pct: i64 = -1;
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                drop(file);
                let _ = std::fs::remove_file(dest);
                return Err(format!("下载中断: {e}"));
            }
        };
        file.write_all(&buf[..n]).map_err(|e| format!("写入文件失败: {e}"))?;
        got += n as u64;
        if got > cap {
            drop(file);
            let _ = std::fs::remove_file(dest);
            return Err(format!(
                "下载体积异常 ({got} > {cap} 字节上限), 已中止"
            ));
        }
        let pct = if total > 0 { (got * 100 / total) as i64 } else { 0 };
        if got - last_emit >= PROGRESS_STEP || pct != last_pct {
            last_emit = got;
            last_pct = pct;
            eval_progress(&window, got, total, attempt);
        }
    }
    let _ = file.flush();
    Ok(got)
}

/// 安装包落盘目录; 也是 install_update 唯一允许执行的文件前缀。
fn updates_dir() -> Result<PathBuf, String> {
    crate::config::adr_data_dir()
        .map(|d| d.join("updates"))
        .ok_or_else(|| "数据目录不可用, 无法保存安装包".to_string())
}

/// 下载安装包到 `<数据目录>\updates\`; 已存在且校验通过则秒回 (重试不重下)。
fn download_blocking(
    app: &tauri::AppHandle,
    pkg: &UpdatePackage,
) -> Result<PathBuf, String> {
    let name = if pkg.name.trim().is_empty() {
        let g = file_name_from_url(&pkg.url);
        if g.is_empty() { "adr_update_setup.exe".to_string() } else { g }
    } else {
        pkg.name.clone()
    };
    let dir = updates_dir()?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("创建更新目录失败: {e}"))?;
    let dest = dir.join(safe_file_name(&name));
    let expect = pkg.sha256.trim().to_ascii_lowercase();
    if expect.is_empty() {
        return Err("更新清单缺少 sha256, 已拒绝下载 (完整性校验是硬门槛)".into());
    }

    // 上次已下好: 校验一致直接复用 (用户中断后点重试不必重拉 1.7GB)
    if dest.exists() {
        if let Ok(h) = sha256_hex(&dest) {
            if h == expect {
                eval_progress(&app.get_webview_window("main"), 1, 1, 0);
                return Ok(dest);
            }
        }
        let _ = std::fs::remove_file(&dest);
    }

    let mut sources: Vec<String> = vec![pkg.url.clone()];
    if let Some(f) = pkg.fallback_url.as_deref() {
        if !f.trim().is_empty() && f != pkg.url {
            sources.push(f.to_string());
        }
    }

    let mut errors: Vec<String> = Vec::new();
    for (i, url) in sources.iter().enumerate() {
        crate::lifecycle::proc_log("update", &format!("下载安装包 [{}/{}]: {url}", i + 1, sources.len()));
        match download_one(app, url, &dest, pkg.size, i + 1) {
            Ok(got) => match sha256_hex(&dest) {
                Ok(h) if h == expect => {
                    crate::lifecycle::proc_log("update", &format!("下载完成并校验通过: {got} 字节"));
                    eval_progress(&app.get_webview_window("main"), got, got, i + 1);
                    return Ok(dest);
                }
                Ok(h) => {
                    let _ = std::fs::remove_file(&dest);
                    errors.push(format!("源{} 校验不符 (得到 {h})", i + 1));
                }
                Err(e) => errors.push(format!("源{} {e}", i + 1)),
            },
            Err(e) => errors.push(format!("源{} {e}", i + 1)),
        }
    }
    Err(format!("下载失败: {}", errors.join("; ")))
}

/// 批次76 (#3): 前端只报「当前界面上的清单版本」, 不再回传整个 UpdatePackage
/// (旧实现 pkg 由前端给 → 页面被注入即可指定任意 URL/哈希)。包型、地址、
/// 哈希一律由 Rust 从**已验签**清单里取; 版本对不上说明清单已更新, 让前端
/// 重新检查一次, 避免拿旧页面的版本号下新包再装错。
#[tauri::command]
pub async fn download_update(
    app: tauri::AppHandle,
    version: Option<String>,
) -> Result<serde_json::Value, String> {
    let (m, _) = fetch_verified_manifest(false)?;
    if let Some(v) = version.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        if version_parts(v) != version_parts(&m.version) {
            return Err(format!(
                "更新清单版本已变化 (页面 {v} / 线上 {}), 请重新检查更新",
                m.version
            ));
        }
    }
    let variant = package_variant();
    let p = pick_package(&m, &variant)
        .cloned()
        .ok_or_else(|| "更新清单里没有可用安装包".to_string())?;
    let a = app.clone();
    let path = tauri::async_runtime::spawn_blocking(move || download_blocking(&a, &p))
        .await
        .map_err(|e| format!("内部任务失败: {e}"))??;
    Ok(serde_json::json!({
        "path": path.to_string_lossy(),
        "name": path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
    }))
}

// ---------------------------------------------------------------------------
// 命令: 安装
// ---------------------------------------------------------------------------

/// 静默安装: 先派一个 detached 批处理等本进程退出, 再拉起安装包。
/// 顺序不可颠倒 — Inno 带 AppMutex (批次72), 壳还在跑时 /VERYSILENT 会因
/// 互斥体直接中止 (无 UI, 用户只看到"什么都没发生")。
#[cfg(windows)]
fn spawn_installer(path: &str) -> Result<(), String> {
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    let pid = std::process::id();
    let bat = std::env::temp_dir().join(format!("adr_update_{pid}.bat"));
    // 纯 ASCII: 批处理按系统代码页解析, 中文注释在 GBK/UTF-8 下都可能炸。
    let script = format!(
        "@echo off\r\n\
         setlocal\r\n\
         :wait\r\n\
         tasklist /FI \"PID eq {pid}\" /NH 2>nul | find \"{pid}\" >nul\r\n\
         if not errorlevel 1 (\r\n\
         \x20 ping -n 2 127.0.0.1 >nul\r\n\
         \x20 goto wait\r\n\
         )\r\n\
         start \"\" \"{path}\" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART\r\n\
         del \"%~f0\"\r\n"
    );
    std::fs::write(&bat, script).map_err(|e| format!("写更新助手脚本失败: {e}"))?;
    std::process::Command::new("cmd")
        .arg("/C")
        .arg(&bat)
        .creation_flags(crate::system::CREATE_NO_WINDOW | DETACHED_PROCESS)
        .spawn()
        .map_err(|e| format!("启动安装程序失败: {e}"))?;
    Ok(())
}

#[cfg(not(windows))]
fn spawn_installer(_path: &str) -> Result<(), String> {
    Err("当前平台不支持自动安装, 请手动运行安装包".into())
}

/// 拉起安装程序并退出壳 (退出后助手脚本等 1s 再执行安装)。
///
/// 批次76 (#3): 不再接受前端传入路径。旧实现只校验 `is_file()`, 页面被注入
/// 时随便给个 exe 就会被 /VERYSILENT 静默执行 = 任意代码执行。现在只认:
/// 已验签清单里的包 → `<数据目录>\updates\<清单文件名>` → 规范化后必须仍在
/// 该目录内 → 重算 SHA256 与清单一致 → 才允许执行。
#[tauri::command]
pub fn install_update(app: tauri::AppHandle) -> Result<(), String> {
    let (m, _) = fetch_verified_manifest(false)?;
    let variant = package_variant();
    let pkg = pick_package(&m, &variant)
        .ok_or_else(|| "更新清单里没有可用安装包".to_string())?;
    let expect = pkg.sha256.trim().to_ascii_lowercase();
    if expect.is_empty() {
        return Err("更新清单缺少 sha256, 已拒绝安装".into());
    }
    let name = if pkg.name.trim().is_empty() {
        let g = file_name_from_url(&pkg.url);
        if g.is_empty() { "adr_update_setup.exe".to_string() } else { g }
    } else {
        pkg.name.clone()
    };
    let dir = updates_dir()?;
    let canon_dir = dir
        .canonicalize()
        .map_err(|e| format!("更新目录不可用: {e}"))?;
    let canon = dir
        .join(safe_file_name(&name))
        .canonicalize()
        .map_err(|_| "安装包不存在, 请重新下载".to_string())?;
    if !canon.starts_with(&canon_dir) {
        return Err("安装包路径不在更新目录内, 已拒绝".into());
    }
    if !canon
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("exe"))
        .unwrap_or(false)
    {
        return Err("安装包不是可执行文件, 已拒绝".into());
    }
    let h = sha256_hex(&canon)?;
    if h != expect {
        let _ = std::fs::remove_file(&canon);
        return Err("安装包校验不符 (已删除), 请重新下载".into());
    }
    let path = canon.to_string_lossy().to_string();
    spawn_installer(&path)?;
    crate::lifecycle::proc_log("update", &format!("已派发静默安装, 壳即将退出: {path}"));
    // 留 800ms 让前端把「安装中」提示刷出来, 再退出
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(800));
        app.exit(0);
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// 命令: 包型互提示
// ---------------------------------------------------------------------------

/// 包型与显卡不匹配时给启动器顶部横幅数据 (只读本地, 不联网)。
#[tauri::command]
pub fn package_hint() -> serde_json::Value {
    let variant = package_variant();
    let has_nvidia = has_nvidia_gpu();
    let kind = hint_kind(&variant, has_nvidia);
    serde_json::json!({
        "variant": variant,
        "has_nvidia": has_nvidia,
        "kind": kind,
        "other_url": if kind.is_empty() { String::new() } else { other_variant_url(&variant) },
        "releases_url": RELEASES_URL,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_gt_numeric_segments() {
        assert!(version_gt("1.3.2", "1.3.1"));
        assert!(version_gt("v1.4", "1.3.9"));
        assert!(!version_gt("1.3.1", "1.3.1"));
        assert!(!version_gt("1.2.9", "1.3"));
        // 缺段补 0
        assert!(!version_gt("1.3", "1.3.0"));
        assert!(version_gt("1.3.1", "1.3"));
        assert!(version_gt("2", "1.99.99"));
    }

    #[test]
    fn safe_file_name_strips_and_sanitizes() {
        assert_eq!(safe_file_name("../../evil.exe"), "evil.exe");
        assert_eq!(safe_file_name(r"C:\tmp\a:b.exe"), "a_b.exe");
        assert_eq!(safe_file_name("   "), "adr_update_setup.exe");
        assert_eq!(safe_file_name(".."), "adr_update_setup.exe");
    }

    #[test]
    fn pick_package_prefers_variant_then_first() {
        let m = UpdateManifest {
            version: "1.4.0".into(),
            notes: String::new(),
            packages: vec![
                UpdatePackage { variant: "cpu".into(), url: "u-cpu".into(), ..Default::default() },
                UpdatePackage { variant: "gpu".into(), url: "u-gpu".into(), ..Default::default() },
            ],
            ..Default::default()
        };
        assert_eq!(pick_package(&m, "gpu").unwrap().url, "u-gpu");
        assert_eq!(pick_package(&m, "cpu").unwrap().url, "u-cpu");
        // 未知变体 → 第一个有 url 的包
        assert_eq!(pick_package(&m, "unknown").unwrap().url, "u-cpu");
    }

    #[test]
    fn hint_kind_matrix() {
        assert_eq!(hint_kind("gpu", false), "gpu_pkg_no_gpu");
        assert_eq!(hint_kind("cpu", true), "cpu_pkg_has_gpu");
        assert_eq!(hint_kind("gpu", true), "");
        assert_eq!(hint_kind("cpu", false), "");
        assert_eq!(hint_kind("unknown", true), "");
    }

    #[test]
    fn manifest_parses_with_missing_fields() {
        let body = r#"{"version":"1.4.0","packages":[{"variant":"cpu","url":"u"}]}"#;
        let m = parse_manifest(body).unwrap();
        assert_eq!(m.version, "1.4.0");
        assert_eq!(m.packages.len(), 1);
        assert!(m.packages[0].sha256.is_empty());
    }

    #[test]
    fn file_name_from_url_ignores_query() {
        assert_eq!(file_name_from_url("https://x/y/ADR-Setup.exe?token=1"), "ADR-Setup.exe");
        assert_eq!(file_name_from_url("https://x/"), "");
    }

    // ---- 批次76 (#3): 清单验签 ----

    fn hex_encode(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    use ring::signature::KeyPair;

    fn sample_manifest() -> UpdateManifest {
        UpdateManifest {
            version: "1.4.0".into(),
            notes: "fix: 训练脚本缺失".into(),
            packages: vec![UpdatePackage {
                variant: "cpu".into(),
                name: "ADR-Studio-1.4.0-x64-setup.exe".into(),
                url: "https://example.com/a.exe".into(),
                fallback_url: Some("https://example.com/b.exe".into()),
                sha256: "AB12".into(),
                size: 4096,
            }],
            signature: String::new(),
        }
    }

    #[test]
    fn signing_payload_is_byte_stable() {
        // 固定向量: scripts/sign_update_manifest.py 必须产出逐字节相同结果,
        // 改格式等于把所有已发布清单作废, 这里把格式钉死。
        let s = String::from_utf8(signing_payload(&sample_manifest())).unwrap();
        let notes_hash = sha256_hex_bytes("fix: 训练脚本缺失".as_bytes());
        assert_eq!(
            s,
            format!(
                "adr-update-v1\n1.4.0\n{notes_hash}\n\
                 cpu|ADR-Studio-1.4.0-x64-setup.exe|https://example.com/a.exe|\
                 https://example.com/b.exe|ab12|4096\n"
            )
        );
    }

    #[test]
    fn verify_manifest_rejects_unsigned_tampered_and_wrong_key() {
        let kp = ring::signature::Ed25519KeyPair::from_seed_unchecked(&[7u8; 32]).unwrap();
        let pk = hex_encode(kp.public_key().as_ref());
        let mut m = sample_manifest();

        // 未签名 → 拒绝 (不是"跳过校验")
        assert!(verify_manifest_with(&m, &pk).is_err());

        m.signature = hex_encode(kp.sign(&signing_payload(&m)).as_ref());
        verify_manifest_with(&m, &pk).expect("正确签名必须通过");

        // 篡改内容 (notes 参与签名) → 拒绝
        m.notes.push('!');
        assert!(verify_manifest_with(&m, &pk).is_err());
        m.notes.pop();

        // 篡改包地址 → 拒绝
        m.packages[0].url = "https://evil.example/x.exe".into();
        assert!(verify_manifest_with(&m, &pk).is_err());
        m.packages[0].url = "https://example.com/a.exe".into();

        // 换公钥 → 拒绝
        let other = ring::signature::Ed25519KeyPair::from_seed_unchecked(&[9u8; 32]).unwrap();
        assert!(verify_manifest_with(&m, &hex_encode(other.public_key().as_ref())).is_err());

        // 非法 hex 签名 → 拒绝而不是 panic
        m.signature = "zz".into();
        assert!(verify_manifest_with(&m, &pk).is_err());
    }

    #[test]
    fn embedded_pubkey_is_well_formed() {
        assert_eq!(UPDATE_PUBKEY_HEX.len(), 64);
        assert!(hex_decode(UPDATE_PUBKEY_HEX).is_some());
    }
}
