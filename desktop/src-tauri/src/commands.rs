//! Tauri 命令层 (M2 拆分, 批次131): 前端 `window.__TAURI__.core.invoke` 的入口。
//!
//! 从 `lib.rs` 原样迁出, 签名/返回形状逐字不变 (前端零改动)。命令只做参数解包、
//! 门禁校验与转发, 真正的活儿在 `sidecar` / `lifecycle` / `config` / `system`。
//! `account.rs` / `update.rs` 的命令仍留在各自模块, 在 `lib.rs` 统一注册。

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tauri::{AppHandle, Manager, State};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use crate::account;
use crate::config::{
    adr_data_dir, reload_shell_config, shell_config, shell_settings_path, ShellSettings,
};
use crate::lifecycle::{
    kill_current, prewarm_flow, proc_log, Phase, ServerState, PREWARM_ACTIVE,
};
use crate::sidecar::{
    console_key_query, find_repo_root, launch_blocking, server_log_path,
};
use crate::system::{
    autostart_cmd_value, brand_is_raptor_lake, local_lan_ip, query_cpu_brand, setup_done_flag,
    setup_flag_path, terms_agreed_flag, terms_flag_path,
};
#[cfg(windows)]
use crate::system::{CREATE_NO_WINDOW, RUN_KEY, RUN_VALUE};
use crate::windows::{app_page, navigate, shell_origin_param};

/// 防重入互斥: UIA 单击也可能双投递 (相隔数秒的两个独立事件, JS 层 launching
/// 挡不住)。两次并发 launch_blocking 会双 spawn 引擎: kill_current 的
/// swap→store 窗口让第二个 kill_current 静默通过, 双 spawn 抢探活, 输家流程
/// 被别人引擎的 200 响应误判成功。Rust 侧 CAS 对一切事件来源免疫。
static LAUNCH_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// 启动指定控制台 (阻塞到探活通过并完成导航, 前端转 loading 态)。
/// expose = 调用模式: 服务监听 0.0.0.0 供局域网调用 (默认仅 127.0.0.1)。
#[tauri::command]
pub(crate) async fn launch_console(
    app: AppHandle,
    mode: String,
    expose: Option<bool>,
) -> Result<String, String> {
    if !matches!(mode.as_str(), "legacy" | "pro" | "easy" | "call") {
        return Err(format!("未知控制台模式: {mode}"));
    }
    // 批次101: 本地账号门禁 — 未登录不进控制台 (纯前端遮罩可被绕过, 壳侧兜一道)
    if !account::logged_in() {
        return Err("请先在启动器登录本地账号".into());
    }
    if LAUNCH_IN_PROGRESS
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Err("已有控制台启动流程进行中".into());
    }
    // 探活最长 180s, 放到阻塞线程池, 避免卡住 async runtime
    let result = tauri::async_runtime::spawn_blocking(move || {
        launch_blocking(app, mode, expose.unwrap_or(false))
    })
    .await
    .map_err(|e| format!("内部任务失败: {e}"))?;
    // 完成即复位: 成功后壳已导航离开, 失败时前端要能重试。防重入窗口只需
    // 覆盖启动过程本身 (双投递的第二发落在 600ms + wait_healthy 窗口内)。
    LAUNCH_IN_PROGRESS.store(false, Ordering::Release);
    result
}

/// 进入控制台: pro/easy 优先走热通道 (服务常驻且 expose 一致 → 直接导航, 秒进);
/// 服务不可用 / legacy / expose 切换 → 回落 launch_console 冷启动。
#[tauri::command]
pub(crate) async fn enter_console(
    app: AppHandle,
    mode: String,
    expose: Option<bool>,
) -> Result<String, String> {
    let expose = expose.unwrap_or(false);
    // 批次101: 本地账号门禁 (热通道同样拦)
    if !account::logged_in() {
        return Err("请先在启动器登录本地账号".into());
    }
    if mode == "legacy" {
        return launch_console(app, mode, Some(expose)).await;
    }
    let state: State<ServerState> = app.state();
    let base = state.base_url.lock().unwrap().clone();
    if let Some(base) = base {
        // 热通道前提: 服务进程还活着且监听模式与请求一致
        let alive = ureq::get(&format!("{base}/api/adr/v1/health"))
            .timeout(Duration::from_secs(2))
            .call()
            .map(|r| r.status() == 200)
            .unwrap_or(false);
        if alive && state.expose.load(Ordering::Relaxed) == expose {
            let sop = shell_origin_param(&app.get_webview_window("main"));
            let target = match mode.as_str() {
                "pro" => format!("{base}/pro?desktop=1{sop}"),
                "easy" => format!("{base}/easy?desktop=1{sop}"),
                _ => format!("{base}/call?desktop=1{sop}"),
            };
            let window = app.get_webview_window("main");
            // 批次66: 导航带 api_key (返回值/托盘提示仍用不含 key 的 target)
            navigate(&window, &format!("{target}{}", console_key_query()));
            return Ok(target);
        }
    }
    launch_console(app, mode, Some(expose)).await
}

/// launcher.html 每次加载时调用 (含从控制台返回): 更新托盘态。
/// 服务常驻预热 (启动即拉起), 不再杀进程 — pro/easy 返回启动器后可秒进。
/// 远程控制台页 (http://127.0.0.1:P) 不在 capability 授权范围内, 无法走 IPC,
/// 其返回按钮是纯 location 导航回本页, 因此统一由本页 (本地 origin) 收尾。
#[tauri::command]
pub(crate) fn on_launcher_ready(app: AppHandle) {
    let state: State<ServerState> = app.state();
    let tip = if state.phase() == Phase::Ready {
        "ADR Studio — 引擎已预热 (启动器)"
    } else {
        "ADR Studio — 引擎加载中 (启动器)"
    };
    if let Some(t) = app.tray_by_id("adr-tray") {
        let _ = t.set_tooltip(Some(tip));
    }
}

/// 引擎当前状态 (launcher 顶栏显示): 未启动 / 加载中 / 已预热。
/// 批次37: 按 Phase 状态机映射, 输出形状不变 (前端零改动)。
#[tauri::command]
pub(crate) fn engine_status(app: AppHandle) -> serde_json::Value {
    let state: State<ServerState> = app.state();
    let phase = match state.phase() {
        Phase::Stopped => "stopped",
        Phase::Starting | Phase::Prewarming => "loading",
        Phase::Ready => "ready",
    };
    serde_json::json!({ "phase": phase })
}

/// 用户强制放行 (预热页失败/超时的「仍要进入」): 跳启动器, 引擎继续后台加载。
#[tauri::command]
pub(crate) fn skip_prewarm(app: AppHandle) {
    // 批次61: 条款门禁 — 未同意首启条款时「仍要进入」不放行 (条款视图下该按钮
    // 本就不可达, 此处为纯防御兜底, 防止未来 UI 变动绕过条款流)。
    if !terms_agreed_flag() {
        return;
    }
    // 批次101: 首启配置门禁 — 未完成首启配置时同样不放行 (配置项需在引擎
    // 启动前落定: 设备走向 / 线程上限 / 数据目录都是 sidecar 启动时的环境注入)。
    if !setup_done_flag() {
        return;
    }
    // 批次102: 本机账号登录门禁 — 未登录不放行 (登录已前移到预热页首屏,
    // 「仍要进入」是引擎失败时的兜底, 同样不能绕过登录)。
    if !account::logged_in() {
        return;
    }
    let state: State<ServerState> = app.state();
    // 服务确实在跑 (pid != 0) 才标 Ready (批次40): 服务已被杀/从未启动时
    // 标 Ready 会谎报就绪 (launcher 顶栏/托盘显示失真); 导航保留 —
    // launcher 里 enter_console 按 base_url+探活自行冷启动兜底。
    if state.pid.load(Ordering::Relaxed) != 0 {
        state.set_phase(Phase::Ready);
    }
    let window = app.get_webview_window("main");
    navigate(&window, &app_page(&window, "launcher.html"));
}

/// 预热页「重试」: 杀现服务重新走完整预热流程。
#[tauri::command]
pub(crate) async fn retry_prewarm(app: AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state: State<ServerState> = app.state();
        kill_current(&state); // 置 Stopped (批次37 状态机), 预热流程内再转 Starting
        let window = app.get_webview_window("main");
        navigate(&window, &app_page(&window, "prewarm.html"));
        std::thread::sleep(Duration::from_millis(600));
        prewarm_flow(app);
    })
    .await
    .map_err(|e| format!("内部任务失败: {e}"))
}

/// loading 页「返回启动器」按钮: 停服务 + 导航回 launcher (批次28 复审 P1,
/// 逻辑对齐托盘 home 菜单项)。未注册时 invoke 会 rejected, 前端需自行 catch 兜底。
#[tauri::command]
pub(crate) fn back_to_launcher(app: AppHandle) {
    let state: State<ServerState> = app.state();
    kill_current(&state);
    if let Some(t) = app.tray_by_id("adr-tray") {
        let _ = t.set_tooltip(Some("ADR Studio — 引擎已停止"));
    }
    let window = app.get_webview_window("main");
    navigate(&window, &app_page(&window, "launcher.html"));
}

/// 批次102: 登录成功后进启动器。
/// 用途: 登录已前移到预热页首屏 (壳启动即弹), 但预热流程 (prewarm_flow)
/// 只在"本次启动里"守登录门禁 — 登出后从预热页重新登录时, 该流程早已结束,
/// 需要本命令在登录成功后主动导航到启动器。三道门禁 (条款 / 首启配置 / 登录)
/// 与 prewarm_flow 保持同口径。
#[tauri::command]
pub(crate) fn enter_launcher(app: AppHandle) -> Result<(), String> {
    if !terms_agreed_flag() {
        return Err("请先同意使用条款".into());
    }
    if !setup_done_flag() {
        return Err("请先完成首次启动配置".into());
    }
    if !account::logged_in() {
        return Err("请先登录本机账号".into());
    }
    // 预热流程仍在跑 (引擎就绪后守登录门禁) → 由它统一导航, 这里不再跳一次。
    if PREWARM_ACTIVE.load(Ordering::Relaxed) {
        return Ok(());
    }
    let window = app.get_webview_window("main");
    navigate(&window, &app_page(&window, "launcher.html"));
    Ok(())
}

/// GSV 必需权重 (与 adr/core/doctor.py _GSV_REQUIRED 同口径)。
const GSV_REQUIRED: [&str; 6] = [
    "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "gsv-v2final-pretrained/s2G2333k.pth",
    "chinese-hubert-base/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/tokenizer.json",
    "sv/pretrained_eres2netv2w24s4ep4.ckpt",
];

/// 权重根目录: 便携态 = <exe旁>/runtime, 开发态 = 仓库根 (adr/cli.py 标记)。
fn models_root() -> Option<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(base) = exe.parent() {
            let rt = base.join("runtime");
            if rt.join("third_party").is_dir() {
                return Some(rt);
            }
        }
    }
    find_repo_root()
}

/// 批次102: 首启向导 ⑨ 检查模型 — 只查本地文件是否存在, 不联网不下载。
#[tauri::command]
pub(crate) fn check_models() -> serde_json::Value {
    let dir = models_root().map(|b| {
        b.join("third_party")
            .join("gpt_sovits")
            .join("GPT_SoVITS")
            .join("pretrained_models")
    });
    let missing: Vec<String> = match dir.as_ref() {
        Some(d) => GSV_REQUIRED
            .iter()
            .filter(|f| !d.join(f).is_file())
            .map(|f| f.to_string())
            .collect(),
        None => GSV_REQUIRED.iter().map(|f| f.to_string()).collect(),
    };
    serde_json::json!({
        "dir": dir.as_ref().map(|d| d.display().to_string()).unwrap_or_default(),
        "found": GSV_REQUIRED.len() - missing.len(),
        "total": GSV_REQUIRED.len(),
        "missing": missing,
        "ready": missing.is_empty(),
    })
}

// ---------------------------------------------------------------------------
// 设置中心命令 (批次48): launcher.html 设置模态框经 invoke 调用
// ---------------------------------------------------------------------------

/// 读取设置 (进程内缓存, 启动时已从盘加载)。
#[tauri::command]
pub(crate) fn load_shell_settings() -> ShellSettings {
    shell_config()
}

/// 保存设置: 写 %LOCALAPPDATA%\ADR\shell.json + 刷新缓存。
/// 环境变量注入在下次 sidecar 启动时生效 (前端提示"重启引擎后生效")。
#[tauri::command]
pub(crate) fn save_shell_settings(settings: ShellSettings) -> Result<(), String> {
    let path = shell_settings_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("创建配置目录失败: {e}"))?;
    }
    let body = serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?;
    std::fs::write(&path, body).map_err(|e| format!("写入 shell.json 失败: {e}"))?;
    reload_shell_config();
    proc_log("settings", &format!("shell.json 已保存: {}", path.display()));
    Ok(())
}

/// 局域网调用信息: 出口 IP + 当前服务地址 + 本次 expose 状态。
/// 还批次 45 Review 的债: key 输入框旁展示 LAN IP:端口与可复制完整 URL。
#[tauri::command]
pub(crate) fn lan_info(app: AppHandle) -> serde_json::Value {
    let state: State<ServerState> = app.state();
    serde_json::json!({
        "ip": local_lan_ip().unwrap_or_default(),
        "url": state.base_url.lock().unwrap().clone(),
        "expose": state.expose.load(Ordering::Relaxed),
    })
}

/// 壳常用路径 (设置页展示 + 打开按钮数据源)。
#[tauri::command]
pub(crate) fn app_paths() -> serde_json::Value {
    serde_json::json!({
        "data_dir": adr_data_dir().map(|p| p.to_string_lossy().into_owned()),
        "log_dir": server_log_path().parent().map(|p| p.to_string_lossy().into_owned()),
        "shell_json": shell_settings_path().to_string_lossy(),
    })
}

/// 开机自启状态 (HKCU Run 注册表查询; 非 Windows 恒 false)。
#[tauri::command]
#[cfg(windows)]
pub(crate) fn autostart_get() -> bool {
    let mut cmd = Command::new("reg");
    cmd.args(["query", RUN_KEY, "/v", RUN_VALUE]);
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd.output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[tauri::command]
#[cfg(not(windows))]
pub(crate) fn autostart_get() -> bool {
    false
}

/// 开/关开机自启: reg add / delete HKCU Run 键, 值为壳 exe 全路径。
#[tauri::command]
#[cfg(windows)]
pub(crate) fn autostart_set(enable: bool) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("解析 exe 路径失败: {e}"))?;
    let mut cmd = Command::new("reg");
    if enable {
        cmd.args([
            "add", RUN_KEY, "/v", RUN_VALUE, "/t", "REG_SZ",
            "/d", &autostart_cmd_value(&exe), "/f",
        ]);
    } else {
        cmd.args(["delete", RUN_KEY, "/v", RUN_VALUE, "/f"]);
    }
    cmd.creation_flags(CREATE_NO_WINDOW);
    let out = cmd.output().map_err(|e| format!("reg.exe 执行失败: {e}"))?;
    if !out.status.success() {
        // delete 时键不存在也算成功 (幂等)
        let stderr = String::from_utf8_lossy(&out.stderr);
        if !enable && stderr.contains("unable to find") {
            return Ok(());
        }
        return Err(format!("注册表操作失败: {}", stderr.trim()));
    }
    proc_log("settings", &format!("开机自启 → {enable}"));
    Ok(())
}

#[tauri::command]
#[cfg(not(windows))]
pub(crate) fn autostart_set(_enable: bool) -> Result<(), String> {
    Err("仅支持 Windows".into())
}

/// 用系统默认程序打开目标: 目录 (日志/数据/配置) 或 HTTPS 链接 (Releases 页)。
/// 批次116 (审计 R5-LIB-01): 收紧信任边界 —— URL 只放行 https; 本地目标只放行
/// 已存在的**目录**。原先 https/http 与任意存在的文件都放行, 前端一旦被注入
/// (见 R5-FE-01) 就能借壳执行任意本地可执行文件 (`open::that` 对 .exe 是直接运行)。
#[tauri::command]
pub(crate) fn open_path(path: String) -> Result<(), String> {
    let target = path.trim();
    if target.starts_with("https://") {
        return open::that(target).map_err(|e| format!("打开失败: {e}"));
    }
    if target.contains("://") {
        return Err("仅支持 https 链接".into());
    }
    let p = PathBuf::from(target);
    if !p.is_dir() {
        return Err(format!("路径不存在或不是目录: {target}"));
    }
    open::that(&p).map_err(|e| format!("打开失败: {e}"))
}

/// CPU 自动兼容检测结果 (批次57): launcher 在勾选框旁提示"已自动启用"。
/// 纯查询不落日志 (cpu_auto_compat 起引擎时已留档)。
#[tauri::command]
pub(crate) fn cpu_compat_auto() -> bool {
    query_cpu_brand().map(|b| brand_is_raptor_lake(&b)).unwrap_or(false)
}

/// 前端查询: 是否已同意首启条款 (prewarm.html 加载时调用, 决定展示条款页还是步骤流)。
#[tauri::command]
pub(crate) fn terms_agreed() -> bool {
    terms_agreed_flag()
}

/// 前端「同意」: 写标记文件 (内容含毫秒时间戳, 便于审计追溯)。
#[tauri::command]
pub(crate) fn agree_terms() -> Result<(), String> {
    let p = terms_flag_path()
        .ok_or_else(|| "数据目录不可用, 无法记录条款同意状态".to_string())?;
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    std::fs::write(&p, format!("agreed_at={ts}\n"))
        .map_err(|e| format!("写入条款标记失败: {e}"))
}

/// 前端「拒绝」: 落日志留痕后退出 (禁止性条款, 拒绝即不可用)。
#[tauri::command]
pub(crate) fn reject_terms(app: AppHandle) {
    proc_log("terms", "用户拒绝使用条款, 应用退出");
    app.exit(0);
}

/// 前端查询: 首启配置是否已完成 (prewarm.html 据此决定展示向导还是步骤流)。
#[tauri::command]
pub(crate) fn setup_done() -> bool {
    setup_done_flag()
}

/// 前端「保存并继续」/「跳过使用默认」: 写完成标记 (内容含毫秒时间戳)。
#[tauri::command]
pub(crate) fn complete_setup() -> Result<(), String> {
    let p = setup_flag_path()
        .ok_or_else(|| "数据目录不可用, 无法记录首启配置状态".to_string())?;
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    std::fs::write(&p, format!("done_at={ts}\n"))
        .map_err(|e| format!("写入首启配置标记失败: {e}"))
}
