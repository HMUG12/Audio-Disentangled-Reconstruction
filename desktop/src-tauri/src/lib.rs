//! ADR Studio 桌面壳 (Tauri 2)。
//!
//! 职责:
//! 1. 启动即预热: 窗口先展示 prewarm.html, setup 后台拉起 FastAPI 服务并轮询
//!    引擎预热阶段 (torch 导入 → 权重加载 → kernel 预热), 就绪后才放行进 launcher
//! 2. 启动器: launcher.html 三种控制台
//!    - legacy: ``python -m adr.cli webui`` (Gradio 经典控制台, 按需拉起)
//!    - pro/easy: 复用常驻 FastAPI 服务秒进 (引擎已预热, 不再杀进程)
//! 3. 健康探活: legacy 探 ``GET /``; pro/easy 探 ``GET /api/adr/v1/health``
//! 4. 进程守护: 服务意外退出按同 mode 自动重启 (稳定 60s 重置失败计数, 上限 4 次)
//! 5. 孤儿收编 (批次37): 壳崩溃/强杀会遗留 python 服务孤儿 (占显存/9881 端口),
//!    启动预热前按命令行特征扫描收编 (kill 整棵树)
//! 6. 托盘: 显示窗口 / 返回启动器 / 浏览器打开 / 退出 (退出时 taskkill /T 杀整棵进程树)
//! 7. 关窗 = 隐藏到托盘, 真正退出只走托盘菜单
//!
//! Python 解释器解析优先级:
//! 1. 便携模式: ``<exe目录>/runtime/python/python.exe`` (打包后的绿色运行时)
//! 2. 环境变量 ``ADR_DESKTOP_PYTHON`` (开发调试指定解释器)
//! 3. 仓库虚拟环境 ``<仓库根>/.venv/Scripts/python.exe`` (若存在)
//! 4. PATH 上的 ``python`` (仓库根从 cwd / exe 逐级向上探测)
//!
//! 阶段 2 计划 (见 docs/m9-phase3-plan.md): 前端替换 Gradio、模型管理器、
//! 便携 runtime/ 目录 (python-build-standalone + 离线依赖 + 静态 ffmpeg)。

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use tauri::{
    AppHandle, Manager, State, WebviewWindow,
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

/// 服务 (重) 启动失败上限
const MAX_FAILURES: u32 = 4;
/// 单次启动健康探活超时 (首次导入 torch/CUDA 较慢)
const HEALTH_TIMEOUT: Duration = Duration::from_secs(180);
/// 稳定运行超过该时长后重置失败计数
const STABLE_UPTIME: Duration = Duration::from_secs(60);
/// 启动预热总超时: 超时放行进启动器 (引擎继续后台加载, 控制台页有三态显示)
const PREWARM_TIMEOUT: Duration = Duration::from_secs(240);

/// WebView2 前端资源 origin (Windows)。
/// 导航一律用绝对地址: 当前页面可能停在远端控制台 (http://127.0.0.1:P),
/// 相对路径会指到远端 origin 上导致 404。
#[cfg(windows)]
const TAURI_ORIGIN: &str = "http://tauri.localhost";
#[cfg(not(windows))]
const TAURI_ORIGIN: &str = "tauri://localhost";

/// 服务生命周期状态机 (批次37): 显式四态替代原 prewarm_done 布尔与散落组合。
/// 转换图: Stopped → Starting → Prewarming → Ready → (kill) → Stopped
/// - Stopped:    无服务进程 (初始 / kill_current / 探活失败)
/// - Starting:   进程已 spawn, health 探活未过 (engine_status → "loading")
/// - Prewarming: 服务已 healthy, prewarm_flow 轮询引擎预热中 ("loading")
/// - Ready:      引擎就绪, 或用户强制放行 / 预热超时放行 (秒进通道开放, "ready")
/// 两条不变式保证枚举与进程一致: pid 写入处必置 Starting, pid 清零处必置 Stopped。
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Stopped = 0,
    Starting = 1,
    Prewarming = 2,
    Ready = 3,
}

/// 共享状态: sidecar PID (0 = 未运行)、就绪后的服务地址、当前控制台模式。
struct ServerState {
    pid: AtomicU32,
    base_url: Mutex<Option<String>>,
    /// "" (启动器) | legacy | pro | easy
    mode: Mutex<String>,
    /// 调用模式: true = 服务监听 0.0.0.0 (局域网可调用 API), false = 仅 127.0.0.1
    expose: AtomicBool,
    /// 生命周期状态机 (批次37, 替代原 prewarm_done: AtomicBool)
    phase: AtomicU8,
}

impl ServerState {
    fn phase(&self) -> Phase {
        match self.phase.load(Ordering::Relaxed) {
            1 => Phase::Starting,
            2 => Phase::Prewarming,
            3 => Phase::Ready,
            _ => Phase::Stopped,
        }
    }
    fn set_phase(&self, p: Phase) {
        self.phase.store(p as u8, Ordering::Relaxed);
    }
}

/// 退出标记: 托盘退出 / RunEvent::Exit 时置位, supervise 轮询后收尾。
/// 重启场景 (launch_console 切换模式) 只杀 sidecar, 不动该标记。
static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            app.manage(ServerState {
                pid: AtomicU32::new(0),
                base_url: Mutex::new(None),
                mode: Mutex::new(String::new()),
                expose: AtomicBool::new(false),
                phase: AtomicU8::new(Phase::Stopped as u8),
            });

            // 主窗口: 代码创建 (替代 tauri.conf.json windows), 以便挂 on_download —
            // WebView2 对网页 a[download] 默认静默取消下载 (控制台"下载音频"失败根因),
            // 必须显式注册下载事件并把产物落盘到系统下载目录。
            let initial: tauri::Url = format!("{TAURI_ORIGIN}/prewarm.html")
                .parse()
                .expect("初始 URL 非法");
            tauri::webview::WebviewWindowBuilder::new(
                app,
                "main",
                tauri::WebviewUrl::External(initial),
            )
            .title("ADR Studio")
            .inner_size(1280.0, 820.0)
            .min_inner_size(960.0, 600.0)
            .center()
            .on_download(|webview, event| match event {
                tauri::webview::DownloadEvent::Requested { destination, .. } => {
                    // blob: 下载建议名可能缺失, 统一落系统下载目录; 文件名优先取
                    // WebView2 建议名 (含 a[download] 属性), 缺失时用时间戳兜底
                    let name = destination
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| {
                            let ts = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis())
                                .unwrap_or(0);
                            format!("adr_audio_{ts}.wav")
                        });
                    let dir = webview
                        .app_handle()
                        .path()
                        .download_dir()
                        .unwrap_or_else(|_| std::env::temp_dir());
                    *destination = dir.join(name).into();
                    true // 允许下载
                }
                tauri::webview::DownloadEvent::Finished { .. } => true,
                _ => true, // non_exhaustive 枚举兜底
            })
            .build()?;

            // 系统托盘
            let show_i = MenuItem::with_id(app, "show", "显示主窗口", true, None::<&str>)?;
            let home_i = MenuItem::with_id(app, "home", "返回启动器", true, None::<&str>)?;
            let web_i = MenuItem::with_id(app, "web", "在浏览器打开控制台", true, None::<&str>)?;
            let quit_i = MenuItem::with_id(app, "quit", "退出", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_i, &home_i, &web_i, &quit_i])?;
            let mut tray = TrayIconBuilder::with_id("adr-tray")
                .menu(&menu)
                .tooltip("ADR Studio — 就绪")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "home" => {
                        let state: State<ServerState> = app.state();
                        kill_current(&state);
                        if let Some(t) = app.tray_by_id("adr-tray") {
                            let _ = t.set_tooltip(Some("ADR Studio — 引擎已停止"));
                        }
                        let window = app.get_webview_window("main");
                        navigate(&window, &format!("{TAURI_ORIGIN}/launcher.html"));
                    }
                    "web" => {
                        let state: State<ServerState> = app.state();
                        let url = state.base_url.lock().unwrap().clone();
                        if let Some(url) = url {
                            let _ = open::that(url);
                        }
                    }
                    "quit" => {
                        shutdown_server(app);
                        app.exit(0);
                    }
                    _ => {}
                });
            if let Some(icon) = app.default_window_icon() {
                tray = tray.icon(icon.clone());
            }
            tray.build(app)?;

            // 启动即预热: 后台拉服务 + 轮询引擎阶段, 就绪后自动导航进 launcher
            {
                let h = app.handle().clone();
                std::thread::spawn(move || prewarm_flow(h));
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            launch_console,
            enter_console,
            on_launcher_ready,
            engine_status,
            skip_prewarm,
            retry_prewarm,
            back_to_launcher
        ])
        .on_window_event(|window, event| {
            // 关窗 = 隐藏到托盘; 真正退出走托盘菜单 (保证 sidecar 被清理)
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .build(tauri::generate_context!())
        .expect("tauri 初始化失败")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                shutdown_server(app);
            }
        });
}

// ---------------------------------------------------------------------------
// Tauri commands (前端通过 window.__TAURI__.core.invoke 调用)
// ---------------------------------------------------------------------------

/// 启动指定控制台 (阻塞到探活通过并完成导航, 前端转 loading 态)。
/// expose = 调用模式: 服务监听 0.0.0.0 供局域网调用 (默认仅 127.0.0.1)。
#[tauri::command]
async fn launch_console(app: AppHandle, mode: String, expose: Option<bool>) -> Result<String, String> {
    if !matches!(mode.as_str(), "legacy" | "pro" | "easy" | "call") {
        return Err(format!("未知控制台模式: {mode}"));
    }
    // 探活最长 180s, 放到阻塞线程池, 避免卡住 async runtime
    tauri::async_runtime::spawn_blocking(move || launch_blocking(app, mode, expose.unwrap_or(false)))
        .await
        .map_err(|e| format!("内部任务失败: {e}"))?
}

/// 进入控制台: pro/easy 优先走热通道 (服务常驻且 expose 一致 → 直接导航, 秒进);
/// 服务不可用 / legacy / expose 切换 → 回落 launch_console 冷启动。
#[tauri::command]
async fn enter_console(app: AppHandle, mode: String, expose: Option<bool>) -> Result<String, String> {
    let expose = expose.unwrap_or(false);
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
            let target = match mode.as_str() {
                "pro" => format!("{base}/pro?desktop=1"),
                "easy" => format!("{base}/easy?desktop=1"),
                _ => format!("{base}/call?desktop=1"),
            };
            let window = app.get_webview_window("main");
            navigate(&window, &target);
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
fn on_launcher_ready(app: AppHandle) {
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
fn engine_status(app: AppHandle) -> serde_json::Value {
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
fn skip_prewarm(app: AppHandle) {
    let state: State<ServerState> = app.state();
    // 服务确实在跑 (pid != 0) 才标 Ready (批次40): 服务已被杀/从未启动时
    // 标 Ready 会谎报就绪 (launcher 顶栏/托盘显示失真); 导航保留 —
    // launcher 里 enter_console 按 base_url+探活自行冷启动兜底。
    if state.pid.load(Ordering::Relaxed) != 0 {
        state.set_phase(Phase::Ready);
    }
    let window = app.get_webview_window("main");
    navigate(&window, &format!("{TAURI_ORIGIN}/launcher.html"));
}

/// 预热页「重试」: 杀现服务重新走完整预热流程。
#[tauri::command]
async fn retry_prewarm(app: AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state: State<ServerState> = app.state();
        kill_current(&state); // 置 Stopped (批次37 状态机), 预热流程内再转 Starting
        let window = app.get_webview_window("main");
        navigate(&window, &format!("{TAURI_ORIGIN}/prewarm.html"));
        std::thread::sleep(Duration::from_millis(600));
        prewarm_flow(app);
    })
    .await
    .map_err(|e| format!("内部任务失败: {e}"))
}

/// loading 页「返回启动器」按钮: 停服务 + 导航回 launcher (批次28 复审 P1,
/// 逻辑对齐托盘 home 菜单项)。未注册时 invoke 会 rejected, 前端需自行 catch 兜底。
#[tauri::command]
fn back_to_launcher(app: AppHandle) {
    let state: State<ServerState> = app.state();
    kill_current(&state);
    if let Some(t) = app.tray_by_id("adr-tray") {
        let _ = t.set_tooltip(Some("ADR Studio — 引擎已停止"));
    }
    let window = app.get_webview_window("main");
    navigate(&window, &format!("{TAURI_ORIGIN}/launcher.html"));
}

// ---------------------------------------------------------------------------
// 启动与守护
// ---------------------------------------------------------------------------

/// 与 adr/core/config.py adr_data_dir() 同序推导 (批次27 复审 ISSUE-2 对齐):
/// ADR_DATA_DIR > F:/ADR_data (legacy, 存在即沿用) > %LOCALAPPDATA%\ADR\data。
/// 返回 None 表示交由 Python 侧默认 (~/.adr/data), 不注入子进程。
/// 空串环境变量视为未设置 (批次27 复审 ISSUE-4: PathBuf("") 会落到进程 CWD)。
fn adr_data_dir() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("ADR_DATA_DIR") {
        if !v.trim().is_empty() {
            return Some(PathBuf::from(v));
        }
    }
    let legacy = PathBuf::from("F:/ADR_data");
    if legacy.is_dir() {
        return Some(legacy);
    }
    std::env::var("LOCALAPPDATA")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|la| PathBuf::from(la).join(r"ADR\data"))
}

/// 服务日志目录 (批次26, 批次27 重构): ADR_LOG_DIR > <数据目录>\logs
/// (均失败回落 %TEMP%\adr_server.log)。数据目录经 adr_data_dir() 与 Python 同源 —
/// 老机器 (F:\ADR_data) 日志随数据落 F: 盘, 新机器落 %LOCALAPPDATA%\ADR\data\logs。
/// 追加写, 启动时打时间戳分隔行。
fn server_log_path() -> PathBuf {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("ADR_LOG_DIR") {
        if !v.trim().is_empty() {
            candidates.push(PathBuf::from(v));
        }
    }
    if let Some(d) = adr_data_dir() {
        candidates.push(d.join("logs"));
    }
    for base in candidates {
        if std::fs::create_dir_all(&base).is_ok() {
            return base.join("server.log");
        }
    }
    std::env::temp_dir().join("adr_server.log")
}

/// 后台泵: 把服务 stdout/stderr 追加写入日志文件 (进程退出时 copy 结束)。
fn pump_server_log<R: std::io::Read>(mut reader: R, path: PathBuf) {
    use std::io::Write;
    let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path)
    else {
        return;
    };
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let _ = writeln!(file, "\n===== [server {ts}] stream open =====");
    let _ = std::io::copy(&mut reader, &mut file);
    let _ = writeln!(file, "\n===== [server {ts}] stream closed =====");
}

/// launch_console 的阻塞实现: 杀旧 → 置模式 → 启动+探活+导航 → 交给守护线程。
fn launch_blocking(app: AppHandle, mode: String, expose: bool) -> Result<String, String> {
    let window = app.get_webview_window("main");
    let state: State<ServerState> = app.state();

    // 服务互斥: 切换前杀旧 sidecar, 等端口释放
    kill_current(&state); // 置 Stopped (批次37 状态机); start_server 成功后转 Starting
    *state.mode.lock().unwrap() = mode.clone();
    state.expose.store(expose, Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(600));

    navigate(&window, &format!("{TAURI_ORIGIN}/loading.html"));
    update_status(&window, "正在启动本地引擎…");
    if let Some(t) = app.tray_by_id("adr-tray") {
        let _ = t.set_tooltip(Some("ADR Studio 启动中…"));
    }

    let Some((child, target, started)) = start_and_wait(&app, &mode, &window, &state) else {
        if let Some(t) = app.tray_by_id("adr-tray") {
            let _ = t.set_tooltip(Some("ADR Studio 引擎启动失败"));
        }
        return Err("引擎启动失败或超时 (探活 180s), 请检查 Python 环境后重试".into());
    };

    // 守护线程: 崩溃后按同 mode 自动重启
    let h = app.clone();
    std::thread::spawn(move || supervise(h, mode, child, started));
    Ok(target)
}

/// 按模式启动 sidecar 并探活; 成功后写 base_url/托盘并导航到目标页。
/// 返回 (子进程, 目标 URL, 启动时刻); 失败返回 None (状态已就地更新)。
fn start_and_wait(
    app: &AppHandle,
    mode: &str,
    window: &Option<WebviewWindow>,
    state: &State<ServerState>,
) -> Option<(Child, String, Instant)> {
    let (child, started) = start_server(mode, window, state)?;
    // 探活成功 (start_server 内 wait_healthy 已 200) 才转 Prewarming (批次40):
    // 维持不变式 "Prewarming: 服务已 healthy" — 此前本路径 Phase 停在 Starting。
    // 前端零感知: Starting/Prewarming 在 engine_status 中都映射 "loading"。
    state.set_phase(Phase::Prewarming);
    let base = state.base_url.lock().unwrap().clone().unwrap_or_default();
    // ?desktop=1: 控制台页据此显示"返回启动器" (壳内标记; 浏览器直开不带)
    let target = match mode {
        "legacy" => format!("{base}/"),
        "pro" => format!("{base}/pro?desktop=1"),
        "easy" => format!("{base}/easy?desktop=1"),
        _ => format!("{base}/call?desktop=1"),
    };
    let expose_tip = if state.expose.load(Ordering::Relaxed) { " (对外服务)" } else { "" };
    if let Some(t) = app.tray_by_id("adr-tray") {
        let _ = t.set_tooltip(Some(format!("ADR Studio — {target}{expose_tip}")));
    }
    navigate(window, &target);
    Some((child, target, started))
}

/// 启动 sidecar 并探活 (不导航, 预热流程与 launch_console 共用)。
/// 成功后写 pid/base_url; 返回 (子进程, 启动时刻); 失败返回 None。
fn start_server(
    mode: &str,
    window: &Option<WebviewWindow>,
    state: &State<ServerState>,
) -> Option<(Child, Instant)> {
    let (python, cwd, pythonpath) = resolve_runtime();
    // 调用模式: 0.0.0.0 对局域网开放 API; 默认仅本机 (webview 与探活都走 127.0.0.1)
    let host = if state.expose.load(Ordering::Relaxed) { "0.0.0.0" } else { "127.0.0.1" };
    // pro/easy 承载 TTS 服务: 优先绑定 9881 (NEKO GPT-SoVITS provider 默认端口, 零配置直连);
    // 9881 被占用则回退随机端口并提示。legacy 是 Gradio WebUI, 不承载 TTS, 恒用随机端口。
    let (port, using_preferred) = if mode != "legacy"
        && probe_fixed_port(ADR_PREFERRED_PORT, host)
    {
        (ADR_PREFERRED_PORT, true)
    } else {
        let Some(p) = free_tcp_port() else {
            update_status(window, "端口分配失败, 正在重试…");
            return None;
        };
        (p, false)
    };
    if mode != "legacy" && !using_preferred {
        update_status(window, &format!(
            "9881 被其他程序占用, 本次改用端口 {port} (NEKO 侧需手动填写该端口)"
        ));
    }
    let base = format!("http://127.0.0.1:{port}");

    let mut cmd = Command::new(&python);
    match mode {
        "legacy" => {
            cmd.args([
                "-m", "adr.cli", "webui",
                "--host", host,
                "--port", &port.to_string(),
                "--no-browser",
            ]);
        }
        // pro / easy: 同一个 FastAPI 服务, 静态控制台页由 server 提供
        _ => {
            cmd.args([
                "-m", "adr.server",
                "--addr", host,
                "--port", &port.to_string(),
            ]);
        }
    }
    cmd.current_dir(&cwd);
    if let Some(pp) = &pythonpath {
        cmd.env("PYTHONPATH", pp);
    }
    // 批次27 复审 ISSUE-2: 注入数据目录, 保证子进程 Python (config.adr_data_dir()
    // 优先读此环境变量) 与壳的目录推导永远同源
    if let Some(d) = adr_data_dir() {
        cmd.env("ADR_DATA_DIR", &d);
    }
    // 服务日志落盘 (排错必需: 此前服务崩溃/卡死均无日志可查)
    // 路径见 server_log_path(): ADR_LOG_DIR > <adr_data_dir()>\logs
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            update_status(window, &format!("启动失败: {e}"));
            return None;
        }
    };
    let log_path = server_log_path();
    if let Some(out) = child.stdout.take() {
        let p = log_path.clone();
        std::thread::spawn(move || pump_server_log(out, p));
    }
    if let Some(err) = child.stderr.take() {
        let p = log_path.clone();
        std::thread::spawn(move || pump_server_log(err, p));
    }
    state.pid.store(child.id(), Ordering::Relaxed);
    *state.base_url.lock().unwrap() = Some(base.clone());
    // 不变式 (批次37): pid 写入 → Starting
    state.set_phase(Phase::Starting);

    let health_path = if mode == "legacy" { "/" } else { "/api/adr/v1/health" };
    update_status(window, "引擎加载中… 首次启动导入 torch/CUDA 较慢");
    if !wait_healthy(&base, health_path, window) {
        kill_tree(child.id());
        let _ = child.wait();
        // 所有权校验 (批次28 复审 P2, 对齐 supervise 让位规则):
        // 探活 180s 内 pid 可能被新的 launch_console 接管, 此时不能清掉接管方的状态
        if state.pid.load(Ordering::Relaxed) == child.id() {
            state.pid.store(0, Ordering::Relaxed);
            *state.base_url.lock().unwrap() = None;
            // 不变式 (批次37): pid 清零 → Stopped
            state.set_phase(Phase::Stopped);
        }
        update_status(window, "启动超时, 请检查 Python 环境 (180s 探活超时)");
        return None;
    }
    Some((child, Instant::now()))
}

// ---------------------------------------------------------------------------
// 启动预热流程 (setup 后台拉起, 就绪前窗口停在 prewarm.html)
// ---------------------------------------------------------------------------

/// 驱动 prewarm.html 更新阶段 (页面挂 window.__adrPrewarm 钩子)。
fn eval_prewarm(window: &Option<WebviewWindow>, stage: &str, text: &str) {
    if let Some(w) = window {
        let _ = w.eval(&format!(
            "window.__adrPrewarm && window.__adrPrewarm({}, {});",
            js_quote(stage),
            js_quote(text)
        ));
    }
}

/// 从 stats 拉引擎预热阶段 + 实时进度文案 (批次44: 下载进度透传, 不触发加载)。
fn fetch_engine_stage_info(base: &str) -> Option<(String, String)> {
    let url = format!("{base}/api/adr/v1/system/stats");
    let body = ureq::get(&url)
        .timeout(Duration::from_secs(2))
        .call()
        .ok()?
        .into_string()
        .ok()?;
    let v = serde_json::from_str::<serde_json::Value>(&body).ok()?;
    let stage = v.get("engine_stage").and_then(|x| x.as_str())?.to_string();
    let text = v
        .get("engine_stage_text")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    Some((stage, text))
}

/// 启动预热主流程: 拉服务 → 轮询引擎阶段 → ready 后自动放行进启动器。
/// 失败停在预热页 (「重试/仍要进入」按钮); 超时自动放行 (引擎继续后台加载)。
fn prewarm_flow(app: AppHandle) {
    let window = app.get_webview_window("main");
    let state: State<ServerState> = app.state();

    eval_prewarm(&window, "starting", "正在检查遗留服务进程…");
    // 孤儿收编 (批次37): 壳上次异常退出 (崩溃/强杀/断电) 会遗留 python 服务孤儿,
    // 占显存与 9881 端口 → 预热撞端口被迫回退随机端口。先收编再启动;
    // 查询失败时内部记日志放行, 不阻塞预热。
    reap_orphans();
    eval_prewarm(&window, "starting", "正在启动本地服务…");
    let Some((child, started)) = start_server("pro", &window, &state) else {
        eval_prewarm(&window, "failed", "服务启动失败, 请检查 Python 环境后重试");
        return;
    };
    let base = state.base_url.lock().unwrap().clone().unwrap_or_default();

    // 守护线程: 服务常驻, launcher 的 pro/easy 走秒进通道
    let h = app.clone();
    std::thread::spawn(move || supervise(h, "pro".to_string(), child, started));

    // 服务已 healthy, 转入引擎预热轮询 (批次37 状态机)
    state.set_phase(Phase::Prewarming);

    // 轮询引擎预热阶段 (queued → importing → downloading → loading → kernel → ready)
    let deadline = Instant::now() + PREWARM_TIMEOUT;
    let mut last = String::new();
    loop {
        if SHUTTING_DOWN.load(Ordering::Relaxed) {
            return;
        }
        // 服务被接管 (用户重试/另起控制台) → 本流程让位
        if state.base_url.lock().unwrap().as_deref() != Some(base.as_str()) {
            return;
        }
        match fetch_engine_stage_info(&base) {
            Some((st, _)) if st == "ready" => break,
            Some((st, _)) if st == "failed" => {
                eval_prewarm(&window, "failed", "引擎预热失败, 请查看日志或重试");
                return;
            }
            Some((st, text)) if !st.is_empty() => {
                let changed = st != last;
                // 批次44: downloading 阶段文案为实时下载进度, 每次轮询都刷新;
                // 其余阶段仅阶段变化时 eval 一次 (文案固定)
                if changed {
                    last = st.clone();
                }
                if st == "downloading" && !text.is_empty() {
                    eval_prewarm(&window, &st, &text);
                } else if changed {
                    let text = match st.as_str() {
                        "queued" => "预热排队中, 等待服务初始化…",
                        "importing" => "加载框架 (torch / CUDA, 首次较慢)…",
                        "downloading" => "下载预训练模型 (首次运行)…",
                        "loading" => "加载模型权重…",
                        "kernel" => "预热推理内核 (首句提速)…",
                        _ => "预热中…",
                    };
                    eval_prewarm(&window, &st, text);
                }
            }
            _ => {}
        }
        if Instant::now() >= deadline {
            // 超时放行: 引擎继续后台加载, 控制台页有三态显示兜底
            eval_prewarm(&window, "timeout", "预热超时, 先进启动器 (引擎后台继续加载)");
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    // 就绪 (ready 或超时放行): 秒进通道开放 (批次37 状态机)
    state.set_phase(Phase::Ready);
    if let Some(t) = app.tray_by_id("adr-tray") {
        let _ = t.set_tooltip(Some("ADR Studio — 引擎已预热 (启动器)"));
    }
    navigate(&window, &format!("{TAURI_ORIGIN}/launcher.html"));
}

/// 守护线程: 监控 child, 意外退出按同 mode 自动重启。
/// 所有权规则: state.pid != 本线程 child 的 pid → 已被新 launch_console 接管, 让位退出。
fn supervise(app: AppHandle, mode: String, mut child: Child, mut started: Instant) {
    let window = app.get_webview_window("main");
    let state: State<ServerState> = app.state();
    let mut my_pid = child.id();
    let mut failures = 0u32;

    loop {
        // 轮询而非阻塞 wait, 保证退出路径畅通
        while !SHUTTING_DOWN.load(Ordering::Relaxed) {
            match child.try_wait() {
                Ok(None) => std::thread::sleep(Duration::from_millis(500)),
                Ok(Some(_)) => break, // 服务进程意外退出
                Err(_) => break,
            }
        }
        if SHUTTING_DOWN.load(Ordering::Relaxed) {
            let _ = child.kill(); // 兜底 (正常退出路径 taskkill 已处理)
            return;
        }

        // 所有权丢失 (被新 launch_console / on_launcher_ready 接管) → 让位
        if state.pid.load(Ordering::Relaxed) != my_pid {
            return;
        }
        let _ = child.kill();
        state.pid.store(0, Ordering::Relaxed);
        *state.base_url.lock().unwrap() = None;
        // 不变式 (批次37): pid 清零 → Stopped (随后 start_and_wait 内转 Starting)
        state.set_phase(Phase::Stopped);

        // 稳定运行够久 → 视为一次成功, 重置失败计数
        if started.elapsed() >= STABLE_UPTIME {
            failures = 0;
        }
        failures += 1;
        if failures > MAX_FAILURES {
            if let Some(t) = app.tray_by_id("adr-tray") {
                let _ = t.set_tooltip(Some("ADR Studio 引擎启动失败"));
            }
            update_status(&window, "多次启动失败, 请返回启动器重试。");
            return;
        }

        update_status(&window, "服务断开, 正在自动重启…");
        navigate(&window, &format!("{TAURI_ORIGIN}/loading.html"));
        if let Some(t) = app.tray_by_id("adr-tray") {
            let _ = t.set_tooltip(Some("ADR Studio 引擎重启中…"));
        }
        std::thread::sleep(Duration::from_secs(2));
        match start_and_wait(&app, &mode, &window, &state) {
            Some((c, _, t0)) => {
                child = c;
                my_pid = child.id();
                started = t0;
            }
            None => return, // 重启失败: 状态已就地更新, 用户可回启动器重试
        }
    }
}

/// 停掉当前 sidecar (幂等, 不动退出标记 — 重启场景靠它区分真退出)。
fn kill_current(state: &State<ServerState>) {
    let pid = state.pid.swap(0, Ordering::Relaxed);
    if pid != 0 {
        // 日志留痕 (批次40): 被杀 pid 与 taskkill 结果落服务日志, 排障可查
        let ok = kill_tree(pid);
        proc_log(
            "kill",
            &if ok {
                format!("kill_current: taskkill /T /F pid={pid} 成功")
            } else {
                format!("kill_current: taskkill /T /F pid={pid} 失败 (Access denied / 已退出)")
            },
        );
        *state.base_url.lock().unwrap() = None;
        // 不变式 (批次37): pid 清零 → Stopped
        state.set_phase(Phase::Stopped);
    }
}

/// 真退出路径: 置退出标记 + 杀 sidecar 进程树 (幂等)。
fn shutdown_server(app: &AppHandle) {
    SHUTTING_DOWN.store(true, Ordering::Relaxed);
    let state: State<ServerState> = app.state();
    kill_current(&state);
}

/// Windows 下杀整棵进程树 (uvicorn/gradio 可能带子进程)。
/// 返回 taskkill 是否成功 — 驱动挂死/提权进程会 Access denied (批次37 实测),
/// 收编方据此记日志, 不静默吞掉。
fn kill_tree(pid: u32) -> bool {
    let mut cmd = Command::new("taskkill");
    cmd.args(["/PID", &pid.to_string(), "/T", "/F"]);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd.output().map(|o| o.status.success()).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// 孤儿收编 (批次37): 壳崩溃/强杀/断电时 taskkill 不会执行, python 服务进程
// 成为孤儿 — 占显存、占 9881 端口 (下次预热被迫回退随机端口)。壳启动预热前
// 扫描本机 python 进程, 按命令行特征识别 ADR 服务孤儿并杀整棵树。
// ---------------------------------------------------------------------------

/// 扫描并收编孤儿服务进程 (幂等)。
/// 匹配口径宁窄勿宽: 命令行含 ``-m adr.server`` (pro/easy FastAPI 服务) 或
/// ``-m adr.cli webui`` (legacy Gradio) — 恰是壳拉起服务的两种形状;
/// 刻意不匹配 ``-m adr.cli`` 其他子命令 (如 train), 训练进程绝不能被误杀。
fn reap_orphans() {
    let Some(procs) = list_python_processes() else {
        reap_log("PowerShell 查询失败, 本次跳过收编");
        return;
    };
    let mut killed = 0u32;
    for (pid, cmdline) in procs {
        let mine = cmdline.contains("-m adr.server") || cmdline.contains("-m adr.cli webui");
        if !mine {
            continue;
        }
        if kill_tree(pid) {
            killed += 1;
            reap_log(&format!("收编孤儿 pid={pid}: {cmdline}"));
        } else {
            // 驱动挂死 (CUDA 卡死) / admin 提权进程: 用户态杀不掉, 记日志放行
            reap_log(&format!("收编失败 (taskkill Access denied) pid={pid}: {cmdline}"));
        }
    }
    if killed > 0 {
        reap_log(&format!("共收编 {killed} 个孤儿进程"));
    }
}

/// 列出本机 python* 进程的 (pid, 命令行)。
/// 输出为制表符分隔逐行解析 — 规避 PS 5.1 ConvertTo-Json 单对象/数组形态差异;
/// 输出编码强制 UTF-8 (默认 OEM 代码页, 命令行含中文路径时丢字)。
fn list_python_processes() -> Option<Vec<(u32, String)>> {
    let script = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; \
        Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | \
        ForEach-Object { \"{0}`t{1}\" -f $_.ProcessId, $_.CommandLine }";
    let mut cmd = Command::new("powershell");
    cmd.args(["-NoProfile", "-NonInteractive", "-Command", script]);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut procs = Vec::new();
    for line in text.lines() {
        let Some((pid, cmdline)) = line.split_once('\t') else {
            continue;
        };
        if let Ok(p) = pid.trim().parse::<u32>() {
            procs.push((p, cmdline.to_string()));
        }
    }
    Some(procs)
}

/// 收编日志追加到服务日志 (server_log_path(), 与 server 流日志同文件排障)。
fn reap_log(msg: &str) {
    proc_log("reap", msg);
}

/// 通用进程操作日志 (批次40): 按标签追加到服务日志, 供 kill/收编等留痕。
fn proc_log(tag: &str, msg: &str) {
    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(server_log_path())
    else {
        return;
    };
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let _ = writeln!(f, "===== [{tag} {ts}] {msg}");
}

/// 轮询健康端点直到 200 或超时 (legacy: Gradio 首页; pro/easy: FastAPI health)。
fn wait_healthy(base: &str, path: &str, window: &Option<WebviewWindow>) -> bool {
    let url = format!("{base}{path}");
    let deadline = Instant::now() + HEALTH_TIMEOUT;
    let mut note = Instant::now();
    while Instant::now() < deadline {
        if let Ok(resp) = ureq::get(&url).timeout(Duration::from_secs(2)).call() {
            if resp.status() == 200 {
                return true;
            }
        }
        if note.elapsed() >= Duration::from_secs(6) {
            note = Instant::now();
            update_status(window, "引擎加载中… 首次启动导入 torch/CUDA 较慢");
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    false
}

/// 解析 (python 解释器, 工作目录, 可选 PYTHONPATH)。
fn resolve_runtime() -> (PathBuf, PathBuf, Option<PathBuf>) {
    // 1) 便携模式: exe 旁的 runtime/ (阶段 0 打包产物)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(base) = exe.parent() {
            let runtime = base.join("runtime");
            let py = runtime.join("python").join("python.exe");
            if py.exists() {
                return (py, runtime, Some(PathBuf::from(".")));
            }
        }
    }

    let root = find_repo_root();

    // 2) 显式指定解释器 (开发调试)
    if let Ok(p) = std::env::var("ADR_DESKTOP_PYTHON") {
        if !p.trim().is_empty() {
            return (PathBuf::from(p), root, None);
        }
    }

    // 3) 仓库虚拟环境 (若存在)
    let venv_py = root.join(".venv").join("Scripts").join("python.exe");
    if venv_py.exists() {
        return (venv_py, root, None);
    }

    // 4) PATH python
    (PathBuf::from("python"), root, None)
}

/// 从 cwd / exe 逐级向上探测仓库根 (以 adr/cli.py 为标记)。
fn find_repo_root() -> PathBuf {
    let mut bases: Vec<PathBuf> = Vec::new();
    if let Ok(c) = std::env::current_dir() {
        bases.push(c);
    }
    if let Ok(e) = std::env::current_exe() {
        if let Some(p) = e.parent() {
            bases.push(p.to_path_buf());
        }
    }
    for mut base in bases {
        for _ in 0..8 {
            if base.join("adr").join("cli.py").exists() {
                return base;
            }
            if !base.pop() {
                break;
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// NEKO GPT-SoVITS provider 默认 API 端口 (adr.server 默认端口一致, 零配置直连)。
const ADR_PREFERRED_PORT: u16 = 9881;

/// 探测固定端口当前能否绑定 (绑定后立即释放), 绑定地址与监听 host 一致。
/// 探测与 uvicorn 实际绑定间的 TOCTOU 竞态可接受 (壳守护失败会报错并拉起重试)。
fn probe_fixed_port(port: u16, host: &str) -> bool {
    TcpListener::bind((host, port)).is_ok()
}

/// 取一个随机空闲 TCP 端口 (绑定后立即释放, 与 uvicorn 绑定间的竞态可忽略)。
fn free_tcp_port() -> Option<u16> {
    TcpListener::bind(("127.0.0.1", 0))
        .ok()?
        .local_addr()
        .ok()
        .map(|a| a.port())
}

/// 在加载页上更新状态文本 (JSON 引号转义, 无注入风险)。
fn update_status(window: &Option<WebviewWindow>, text: &str) {
    if let Some(w) = window {
        let _ = w.eval(&format!(
            "window.__adrStatus && window.__adrStatus({});",
            js_quote(text)
        ));
    }
}

/// 把窗口导航到目标页 (必须是绝对 URL, 见 TAURI_ORIGIN 注释)。
fn navigate(window: &Option<WebviewWindow>, url: &str) {
    if let Some(w) = window {
        let _ = w.eval(&format!("location.replace({});", js_quote(url)));
    }
}

fn js_quote(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}
