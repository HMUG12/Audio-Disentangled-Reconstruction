//! 本地账号 (批次101): 仅本机登录用, 不联网、不上报、不出机器。
//!
//! - 存储: `<数据目录>\.local_account` (JSON)。口令按 PBKDF2-HMAC-SHA256
//!   (16B 随机盐, 120k 轮) 派生后落盘, 明文永不写盘。
//! - 默认账号: 账号文件不存在时用内置默认 (`ADR` / `Adr12345@`), 开箱即用;
//!   用户在首启向导或设置里改过口令后, 一律以文件为准。
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
    let acc = load_account();
    serde_json::json!({
        "username": acc.as_ref().map(|a| a.username.clone()).unwrap_or_else(|| DEFAULT_USER.to_string()),
        "customized": acc.is_some(),
        "logged_in": LOGGED_IN.load(Ordering::Relaxed),
        "min_pass_len": MIN_PASS_LEN,
    })
}

/// 设置/修改本地账号 (首启向导与设置中心共用): 派生口令后落盘并直接置登录态。
#[tauri::command]
pub fn account_set(username: String, password: String) -> Result<(), String> {
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
    LOGGED_IN.store(true, Ordering::Relaxed);
    crate::lifecycle::proc_log("account", &format!("本地账号已更新: {}", acc.username));
    Ok(())
}

/// 登录校验 (失败只回通用文案, 不区分用户名错 / 口令错)。
#[tauri::command]
pub fn account_login(username: String, password: String) -> Result<(), String> {
    if verify(&username, &password) {
        LOGGED_IN.store(true, Ordering::Relaxed);
        Ok(())
    } else {
        Err("用户名或密码错误".into())
    }
}

/// 退出登录 (清会话态; 账号文件保留)。
#[tauri::command]
pub fn account_logout() {
    LOGGED_IN.store(false, Ordering::Relaxed);
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
