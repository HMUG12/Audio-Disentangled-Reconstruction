//! 本地账号 (批次101): 仅本机登录用, 不联网、不上报、不出机器。
//!
//! - 存储: `<数据目录>\.local_account` (JSON)。口令按 PBKDF2-HMAC-SHA256
//!   (16B 随机盐, 120k 轮) 派生后落盘, 明文永不写盘。
//! - 默认账号: 账号文件不存在时仍保留内置默认 (`ADR` / `Adr12345@`) 作为
//!   旧装机兼容, 但首启向导强制改密; 仍在使用默认口令时禁止进入控制台
//!   (`using_default_password`), 直到用户在设置里改掉。
//! - 会话: 登录态存进程内 (`LOGGED_IN`), 退出应用即失效。控制台启动命令
//!   (`enter_console` / `launch_console`) 校验登录态, 未登录直接拒绝 —
//!   纯前端遮罩会被绕过, 必须壳侧兜一道。

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// 默认账号 (未创建账号文件时生效)。
pub const DEFAULT_USER: &str = "ADR";
pub const DEFAULT_PASS: &str = "Adr12345@";
/// 口令最小长度 (首启向导 / 设置中心同口径)。
pub const MIN_PASS_LEN: usize = 6;
/// PBKDF2 轮数 (桌面端毫秒级, 兼顾抗爆破与登录手感)。
const ITERATIONS: u32 = 120_000;

/// 进程内登录态 (退出应用即失效)。
static LOGGED_IN: AtomicBool = AtomicBool::new(false);

/// 「记住我」免密时长 (天)。
const REMEMBER_DAYS: u64 = 7;

/// 账号文件内容 (口令只存派生值 + 盐)。
#[derive(Serialize, Deserialize)]
struct AccountFile {
    username: String,
    salt: String,
    hash: String,
    iterations: u32,
    updated_at: u64,
}

/// 账号文件落点: `<数据目录>\.local_account` (与条款标记同目录, 覆盖安装保留)。
fn account_path() -> Option<PathBuf> {
    crate::config::adr_data_dir().map(|d| d.join(".local_account"))
}

/// 「记住我」令牌路径 (与账号文件同目录)。
fn session_path() -> Option<PathBuf> {
    crate::config::adr_data_dir().map(|d| d.join(".local_account.session"))
}

/// 写免密令牌: exp + HMAC(账号口令派生值, exp), 到期或被改密即失效。
fn write_session(acc: &AccountFile) -> Result<(), String> {
    let p = session_path().ok_or_else(|| "数据目录不可用".to_string())?;
    let exp = now_ms() + REMEMBER_DAYS * 24 * 3600 * 1000;
    let sig = hmac_sha256(acc.hash.as_bytes(), exp.to_string().as_bytes());
    let body = serde_json::json!({ "exp": exp, "sig": hex_encode(&sig) });
    std::fs::write(&p, body.to_string()).map_err(|e| format!("写入免密令牌失败: {e}"))
}

/// 校验免密令牌 (过期 / 签名不符 / 账号已删都视为无效)。
fn session_valid() -> bool {
    let Some(p) = session_path() else { return false };
    let Ok(raw) = std::fs::read_to_string(p) else { return false };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else { return false };
    let exp = v.get("exp").and_then(|x| x.as_u64()).unwrap_or(0);
    if exp <= now_ms() {
        return false;
    }
    let Some(acc) = load_account() else { return false };
    let expect = hmac_sha256(acc.hash.as_bytes(), exp.to_string().as_bytes());
    match v.get("sig").and_then(|x| x.as_str()).and_then(hex_decode) {
        Some(sig) => ct_eq(&sig, &expect),
        None => false,
    }
}

fn clear_session() {
    if let Some(p) = session_path() {
        let _ = std::fs::remove_file(p);
    }
}

/// 读取账号文件 (坏 JSON / 读不到 → None = 走默认账号)。
fn load_account() -> Option<AccountFile> {
    let p = account_path()?;
    let raw = std::fs::read_to_string(p).ok()?;
    serde_json::from_str::<AccountFile>(&raw).ok()
}

/// 写账号文件 (口令已派生, 明文不出内存到盘)。
fn save_account(acc: &AccountFile) -> Result<(), String> {
    let p = account_path().ok_or_else(|| "数据目录不可用, 无法保存账号".to_string())?;
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("创建数据目录失败: {e}"))?;
    }
    let body = serde_json::to_string_pretty(acc).map_err(|e| e.to_string())?;
    std::fs::write(&p, body).map_err(|e| format!("写入账号文件失败: {e}"))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let val = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let mut i = 0;
    while i < b.len() {
        out.push((val(b[i])? << 4) | val(b[i + 1])?);
        i += 2;
    }
    Some(out)
}

/// HMAC-SHA256 (sha2 只提供 Sha256, HMAC 手写 20 行, 省一个依赖)。
fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        let d = Sha256::digest(key);
        k[..32].copy_from_slice(&d);
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(msg);
    let ih = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(ih);
    let mut out = [0u8; 32];
    out.copy_from_slice(&outer.finalize());
    out
}

/// PBKDF2-HMAC-SHA256 (输出 32B, 单块 — dkLen = hLen 时够用)。
fn pbkdf2_sha256(password: &[u8], salt: &[u8], iterations: u32) -> [u8; 32] {
    let it = iterations.max(1);
    let mut buf = Vec::with_capacity(salt.len() + 4);
    buf.extend_from_slice(salt);
    buf.extend_from_slice(&1u32.to_be_bytes());
    let mut u = hmac_sha256(password, &buf);
    let mut t = u;
    for _ in 1..it {
        u = hmac_sha256(password, &u);
        for i in 0..32 {
            t[i] ^= u[i];
        }
    }
    t
}

/// 常量时间比较 (长度不等直接 false, 但不提前返回内容差异)。
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// 16B 随机盐 (ring 的 SystemRandom, 已在依赖里 — 批次76 引入)。
fn random_salt() -> Result<[u8; 16], String> {
    use ring::rand::{SecureRandom, SystemRandom};
    let mut buf = [0u8; 16];
    SystemRandom::new()
        .fill(&mut buf)
        .map_err(|e| format!("生成随机盐失败: {e}"))?;
    Ok(buf)
}

/// 账号文件是否仍对应内置默认口令 (用户名 = ADR 且口令派生值匹配)。
fn matches_default_password(acc: &AccountFile) -> bool {
    if acc.username != DEFAULT_USER {
        return false;
    }
    let salt = hex_decode(&acc.salt).unwrap_or_default();
    let calc = pbkdf2_sha256(DEFAULT_PASS.as_bytes(), &salt, acc.iterations);
    let stored = hex_decode(&acc.hash).unwrap_or_default();
    ct_eq(&calc, &stored)
}

/// 仍处于内置默认口令状态 (无账号文件 / 文件里就是默认口令)。
/// 壳侧据此拒绝进入控制台, 强制用户先改密。
pub(crate) fn using_default_password() -> bool {
    match load_account() {
        Some(acc) => matches_default_password(&acc),
        None => true,
    }
}

/// 校验用户名 + 口令。账号文件不存在时按内置默认账号比对。
fn verify(username: &str, password: &str) -> bool {
    match load_account() {
        Some(acc) => {
            let u_ok = ct_eq(acc.username.as_bytes(), username.trim().as_bytes());
            let salt = hex_decode(&acc.salt).unwrap_or_default();
            let calc = pbkdf2_sha256(password.as_bytes(), &salt, acc.iterations);
            let stored = hex_decode(&acc.hash).unwrap_or_default();
            let p_ok = ct_eq(&calc, &stored);
            u_ok & p_ok
        }
        None => {
            let u_ok = ct_eq(DEFAULT_USER.as_bytes(), username.trim().as_bytes());
            let p_ok = ct_eq(DEFAULT_PASS.as_bytes(), password.as_bytes());
            u_ok & p_ok
        }
    }
}

/// 前端查询: 当前账号信息 + 是否已登录。
/// username 一律返回真实账号名 (未创建文件时 = 默认账号名), 前端据此预填。
#[tauri::command]
pub fn account_info() -> serde_json::Value {
    // 「记住我」: 令牌有效则直接恢复登录态 (每次启动最多验一次)
    let auto = !LOGGED_IN.load(Ordering::Relaxed) && session_valid();
    if auto {
        LOGGED_IN.store(true, Ordering::Relaxed);
    }
    let acc = load_account();
    serde_json::json!({
        "username": acc.as_ref().map(|a| a.username.clone()).unwrap_or_else(|| DEFAULT_USER.to_string()),
        "customized": acc.is_some(),
        "default_password": acc.as_ref().map_or(true, matches_default_password),
        "logged_in": LOGGED_IN.load(Ordering::Relaxed),
        "auto_login": auto,
        "remember_days": REMEMBER_DAYS,
        "min_pass_len": MIN_PASS_LEN,
    })
}

/// 设置/修改本地账号 (首启向导与设置中心共用): 派生口令后落盘并直接置登录态。
/// 已有账号时 old_password 必填且必须匹配 (忘记密码走 account_reset)。
#[tauri::command]
pub fn account_set(username: String, password: String,
                   old_password: Option<String>) -> Result<(), String> {
    let user = username.trim().to_string();
    if user.is_empty() {
        return Err("用户名不能为空".into());
    }
    if user.chars().count() > 32 {
        return Err("用户名过长 (最多 32 字符)".into());
    }
    if password.chars().count() < MIN_PASS_LEN {
        return Err(format!("密码至少 {MIN_PASS_LEN} 位"));
    }
    if password == DEFAULT_PASS {
        return Err("请勿使用内置默认口令，换一个只有你知道的密码".into());
    }
    // 改密必须验旧密码 (首启/忘记密码时没有账号文件, 不需要)
    if let Some(acc) = load_account() {
        let old = old_password.unwrap_or_default();
        let salt = hex_decode(&acc.salt).unwrap_or_default();
        let calc = pbkdf2_sha256(old.as_bytes(), &salt, acc.iterations);
        let stored = hex_decode(&acc.hash).unwrap_or_default();
        if !ct_eq(&calc, &stored) {
            return Err("旧密码不正确".into());
        }
    }
    let salt = random_salt()?;
    let hash = pbkdf2_sha256(password.as_bytes(), &salt, ITERATIONS);
    let acc = AccountFile {
        username: user,
        salt: hex_encode(&salt),
        hash: hex_encode(&hash),
        iterations: ITERATIONS,
        updated_at: now_ms(),
    };
    save_account(&acc)?;
    clear_session();   // 改密后旧免密令牌立即失效
    LOGGED_IN.store(true, Ordering::Relaxed);
    crate::lifecycle::proc_log("account", &format!("本地账号已更新: {}", acc.username));
    Ok(())
}

/// 登录校验 (失败只回通用文案, 不区分用户名错 / 口令错)。
/// remember = true 时写 7 天免密令牌。
#[tauri::command]
pub fn account_login(username: String, password: String,
                     remember: Option<bool>) -> Result<(), String> {
    if verify(&username, &password) {
        LOGGED_IN.store(true, Ordering::Relaxed);
        if remember.unwrap_or(false) {
            if let Some(acc) = load_account() {
                write_session(&acc)?;
            }
        } else {
            clear_session();
        }
        Ok(())
    } else {
        Err("用户名或密码错误".into())
    }
}

/// 校验当前账号口令 (供「进入设置 / 导出模型」二次验证; 不改变登录态)。
#[tauri::command]
pub fn account_verify(password: String) -> bool {
    match load_account() {
        Some(acc) => {
            let salt = hex_decode(&acc.salt).unwrap_or_default();
            let calc = pbkdf2_sha256(password.as_bytes(), &salt, acc.iterations);
            let stored = hex_decode(&acc.hash).unwrap_or_default();
            ct_eq(&calc, &stored)
        }
        None => ct_eq(DEFAULT_PASS.as_bytes(), password.as_bytes()),
    }
}

/// 退出登录 (清会话态与免密令牌; 账号文件保留)。
#[tauri::command]
pub fn account_logout() {
    LOGGED_IN.store(false, Ordering::Relaxed);
    clear_session();
}

/// 忘记密码: 清除本机账号文件并退出登录态。
/// 只删 `<数据目录>\.local_account`, 不动声音档案、模型与训练产物;
/// 前端随后引导用户立即设置新口令 (不回到内置默认口令)。
#[tauri::command]
pub fn account_reset() -> Result<(), String> {
    let p = account_path().ok_or_else(|| "数据目录不可用, 无法重置账号".to_string())?;
    match std::fs::remove_file(&p) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("清除账号文件失败: {e}")),
    }
    LOGGED_IN.store(false, Ordering::Relaxed);
    clear_session();
    crate::lifecycle::proc_log("account", "用户通过「忘记密码」重置本机账号");
    Ok(())
}

/// 当前登录态 (壳侧命令用, 非前端)。
pub fn logged_in() -> bool {
    LOGGED_IN.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc4231_case1() {
        // RFC 4231 test case 1: key = 0x0b*20, data = "Hi There"
        let key = [0x0bu8; 20];
        let mac = hmac_sha256(&key, b"Hi There");
        assert_eq!(
            hex_encode(&mac),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn pbkdf2_matches_rfc6070_style_vector() {
        // RFC 6070 是 HMAC-SHA1, 这里用自洽性校验: 同输入同输出, 变盐则变输出
        let a = pbkdf2_sha256(b"password", b"salt", 1000);
        let b = pbkdf2_sha256(b"password", b"salt", 1000);
        let c = pbkdf2_sha256(b"password", b"salt2", 1000);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn ct_eq_basic() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"ab"));
    }

    #[test]
    fn hex_roundtrip() {
        let raw = [0u8, 1, 15, 16, 255];
        assert_eq!(hex_decode(&hex_encode(&raw)).unwrap(), raw.to_vec());
        assert!(hex_decode("0g").is_none());
    }
}
