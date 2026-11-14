//! 系统集成 (M2 拆分, 批次130): 单实例互斥体 / 开机自启 / CPU 兼容检测 /
//! 首启条款与向导标记。从 `lib.rs` 原样迁出, 行为不变。
//!
//! 只放"与 Windows / 文件标记打交道"的逻辑; 命令包装仍留在 lib.rs（M2-c 统一收编）。

use std::path::PathBuf;
use std::process::Command;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use crate::config::adr_data_dir;
use crate::lifecycle::proc_log;

/// CPU 品牌串 (批次57): reg 读注册表 (零依赖, 与 autostart 同套路),
/// 用于 13/14 代自动检测与日志留档。
#[cfg(windows)]
pub(crate) fn query_cpu_brand() -> Option<String> {
    let mut cmd = Command::new("reg");
    cmd.args([
        "query",
        r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        "/v", "ProcessorNameString",
    ]);
    cmd.creation_flags(CREATE_NO_WINDOW);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    // 行形如 "    ProcessorNameString    REG_SZ    13th Gen Intel(R) Core(TM) i7-13700K"
    s.lines().find_map(|l| {
        l.split("REG_SZ")
            .nth(1)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    })
}
#[cfg(not(windows))]
pub(crate) fn query_cpu_brand() -> Option<String> {
    None
}

/// Raptor Lake 判定 (纯函数, 单测覆盖): Intel 13/14 代品牌串。
pub(crate) fn brand_is_raptor_lake(brand: &str) -> bool {
    let b = brand.to_ascii_lowercase();
    b.contains("intel") && (b.contains("13th gen") || b.contains("14th gen"))
}

/// CPU 档案诊断 + 13/14 代自动兼容判定 (每次起 sidecar 调用, 结果落 server.log)。
/// 顺带留档 AVX2/AVX512 支持 — 缺失时 torch 官方轮子会直接闪退, 至少有据可查。
#[cfg(windows)]
pub(crate) fn cpu_auto_compat() -> bool {
    let Some(brand) = query_cpu_brand() else {
        return false;
    };
    #[cfg(target_arch = "x86_64")]
    {
        let (avx2, avx512) =
            (is_x86_feature_detected!("avx2"), is_x86_feature_detected!("avx512f"));
        proc_log("server", &format!("CPU: {brand}; AVX2={avx2}, AVX512={avx512}"));
        if !avx2 {
            proc_log(
                "server",
                "警告: CPU 不支持 AVX2, torch 官方轮子可能无法运行 \
                 (需 Intel Haswell 2013+ / AMD Zen 2017+)",
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    proc_log("server", &format!("CPU: {brand}"));
    let raptor = brand_is_raptor_lake(&brand);
    if raptor {
        proc_log(
            "server",
            "检测到 Intel 13/14 代 (Raptor Lake): 未手动设置时自动启用 \
             CPU 兼容模式 (全链限 AVX2)",
        );
    }
    raptor
}
#[cfg(not(windows))]
pub(crate) fn cpu_auto_compat() -> bool {
    false
}

/// 自启注册表值: exe 全路径加引号 (含空格路径必需)。
pub(crate) fn autostart_cmd_value(exe: &std::path::Path) -> String {
    format!("\"{}\"", exe.display())
}

#[cfg(windows)]
pub(crate) const RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";
#[cfg(windows)]
pub(crate) const RUN_VALUE: &str = "ADR Studio";

#[cfg(windows)]
pub(crate) const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// 批次72 (卸载体验): 运行标记互斥体名 — 与 installer.iss 的 [Setup] AppMutex /
/// adr/server/__main__.py 三端**逐字一致** (tests/test_audit_packaging.py 防漂移)。
/// 裸名 = 会话命名空间 (Local\), 与 per-user 安装语义匹配。安装器/卸载器启动时
/// 检测到即提示"关闭程序后再继续", 治"卸载时不检查程序是否关闭"。
#[cfg(windows)]
pub(crate) const APP_MUTEX_NAME: &str = "ADR-Studio-Running";

/// 创建运行标记互斥体 (批次72)。句柄故意不 CloseHandle: 随进程退出 (含被
/// taskkill /F) 由 OS 回收释放。创建失败静默 — 不阻断壳启动, 只是安装器
/// 少一道运行中拦截 (沙箱策略等极端环境)。
#[cfg(windows)]
pub(crate) fn create_app_mutex() {
    #[link(name = "kernel32")]
    extern "system" {
        fn CreateMutexW(lpMutexAttributes: *const core::ffi::c_void,
                        bInitialOwner: i32,
                        lpName: *const u16) -> *mut core::ffi::c_void;
    }
    use std::os::windows::ffi::OsStrExt;
    let name: Vec<u16> = std::ffi::OsStr::new(APP_MUTEX_NAME)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    unsafe {
        CreateMutexW(std::ptr::null(), 0, name.as_ptr());
    }
}

/// 本机默认路由出口 IP (UDP connect 不发包探测, 零依赖);
/// 离线/多网卡异常时返回空串 — 前端降级提示。
pub(crate) fn local_lan_ip() -> Option<String> {
    let s = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("223.5.5.5:80").ok()?; // 阿里公共 DNS: 仅路由决策, 不发数据
    s.local_addr().ok().map(|a| a.ip().to_string())
}

// ---------------------------------------------------------------------------
// 首启条款 (批次61): prewarm 前置条款页 + .terms_agreed_v1 标记 + 拒绝退出态
// ---------------------------------------------------------------------------

/// 条款同意标记路径: <数据目录>\.terms_agreed_v1 (数据目录与 Python adr_data_dir()
/// 同源推导)。覆盖安装保留数据目录 → 升级/重装不重复弹条款。
pub(crate) fn terms_flag_path() -> Option<PathBuf> {
    adr_data_dir().map(|d| d.join(".terms_agreed_v1"))
}

/// 内部查询 (prewarm_flow 门禁与 skip_prewarm 兜底共用): 标记文件存在即视为已同意。
pub(crate) fn terms_agreed_flag() -> bool {
    terms_flag_path().map(|p| p.exists()).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// 首启配置向导 (批次101): 预热前的一次性配置
//
// 会议纪要 L163「设计增加项: 首次启动配置」①-⑯ 共 16 项, 实际落地取可支撑
// 子集 (纪要未逐项排期, 本批按 "先能跑起来" 原则收窄):
//   本向导: ①数据目录 ⑧开机自启 ③推理设备 ④CPU 线程上限 ⑫协议(条款页)
//             + 本地账号 (批次101 新增)
//   设置中心: ⑩定向路径 / ②权重导入 / ⑤⑥显存内存上限 / ⑰强度 / ⑯Autopilot
//             覆盖策略等, 仍在进入壳后的设置中心与后台 Autopilot 自动标定。
// 完成标记: <数据目录>\.setup_done_v1 (覆盖安装保留数据目录 → 不重复弹)。
// ---------------------------------------------------------------------------

/// 首启配置完成标记路径。
pub(crate) fn setup_flag_path() -> Option<PathBuf> {
    adr_data_dir().map(|d| d.join(".setup_done_v1"))
}

/// 内部查询: 标记存在 = 已完成首启配置。
pub(crate) fn setup_done_flag() -> bool {
    setup_flag_path().map(|p| p.exists()).unwrap_or(false)
}
