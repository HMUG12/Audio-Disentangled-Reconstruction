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
// 本机环境: 包型 / 显卡 / 缓存路径
// ---------------------------------------------------------------------------

/// 清单缓存文件: 与 shell.json 同目录 (%LOCALAPPDATA%\ADR)。
fn cache_path() -> PathBuf {
    super::shell_settings_path()
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
fn has_nvidia_gpu() -> bool {
    let mut cmd = super::Command::new("reg");
    cmd.args(["query", r"HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm"]);
    cmd.creation_flags(super::CREATE_NO_WINDOW);
    cmd.output().map(|o| o.status.success()).unwrap_or(false)
}

#[cfg(not(windows))]
fn has_nvidia_gpu() -> bool {
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
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .build();
    match agent.get(MANIFEST_URL).call() {
        Ok(resp) => match resp.into_string() {
            Ok(body) if !body.trim().is_empty() => {
                write_cache(&body);
                Ok((body, false))
            }
            Ok(_) => Err("更新清单为空".into()),
            Err(e) => Err(format!("读取更新清单失败: {e}")),
        },
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
        .and_then(|m| {
            m.packages
                .iter()
                .find(|p| p.variant == want && !p.url.is_empty())
                .map(|p| p.url.clone())
        });
    url.unwrap_or_else(|| RELEASES_URL.to_string())
}

// ---------------------------------------------------------------------------
// 命令: 检查更新
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn check_update(app: tauri::AppHandle, force: Option<bool>) -> Result<serde_json::Value, String> {
    let current = app.package_info().version.to_string();
    let (body, from_cache) = fetch_manifest_text(force.unwrap_or(false))?;
    let m = parse_manifest(&body)?;
    if m.version.trim().is_empty() {
        return Err("更新清单缺少 version 字段".into());
    }
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
    super::proc_log("update", &format!(
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
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
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
    let dir = super::adr_data_dir()
        .ok_or_else(|| "数据目录不可用, 无法保存安装包".to_string())?
        .join("updates");
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
        super::proc_log("update", &format!("下载安装包 [{}/{}]: {url}", i + 1, sources.len()));
        match download_one(app, url, &dest, pkg.size, i + 1) {
            Ok(got) => match sha256_hex(&dest) {
                Ok(h) if h == expect => {
                    super::proc_log("update", &format!("下载完成并校验通过: {got} 字节"));
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

#[tauri::command]
pub async fn download_update(
    app: tauri::AppHandle,
    pkg: UpdatePackage,
) -> Result<serde_json::Value, String> {
    let a = app.clone();
    let p = pkg.clone();
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
    super::Command::new("cmd")
        .arg("/C")
        .arg(&bat)
        .creation_flags(super::CREATE_NO_WINDOW | DETACHED_PROCESS)
        .spawn()
        .map_err(|e| format!("启动安装程序失败: {e}"))?;
    Ok(())
}

#[cfg(not(windows))]
fn spawn_installer(_path: &str) -> Result<(), String> {
    Err("当前平台不支持自动安装, 请手动运行安装包".into())
}

/// 拉起安装程序并退出壳 (退出后助手脚本等 1s 再执行安装)。
#[tauri::command]
pub fn install_update(app: tauri::AppHandle, path: String) -> Result<(), String> {
    let p = PathBuf::from(&path);
    if !p.is_file() {
        return Err("安装包不存在, 请重新下载".into());
    }
    spawn_installer(&path)?;
    super::proc_log("update", &format!("已派发静默安装, 壳即将退出: {path}"));
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
}
