//! ADR Studio 桌面壳 (Tauri 2)。
//!
//! 职责:
//! 1. 启动器: 窗口先展示 launcher.html, 三种控制台按需拉起
//!    - legacy: ``python -m adr.cli webui`` (Gradio 经典控制台)
//!    - pro/easy: ``python -m adr.server`` (FastAPI, 静态控制台页 /pro /easy)
//! 2. 健康探活: legacy 探 ``GET /``; pro/easy 探 ``GET /api/adr/v1/health``
//! 3. 服务互斥: 同一时间只跑一个 sidecar, 切换控制台 / 返回启动器时杀旧进程 (省显存)
//! 4. 进程守护: 服务意外退出按同 mode 自动重启 (稳定 60s 重置失败计数, 上限 4 次)
//! 5. 托盘: 显示窗口 / 返回启动器 / 浏览器打开 / 退出 (退出时 taskkill /T 杀整棵进程树)
//! 6. 关窗 = 隐藏到托盘, 真正退出只走托盘菜单
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
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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

/// WebView2 前端资源 origin (Windows)。
/// 导航一律用绝对地址: 当前页面可能停在远端控制台 (http://127.0.0.1:P),
/// 相对路径会指到远端 origin 上导致 404。
#[cfg(windows)]
const TAURI_ORIGIN: &str = "http://tauri.localhost";
#[cfg(not(windows))]
const TAURI_ORIGIN: &str = "tauri://localhost";

/// 共享状态: sidecar PID (0 = 未运行)、就绪后的服务地址、当前控制台模式。
struct ServerState {
    pid: AtomicU32,
    base_url: Mutex<Option<String>>,
    /// "" (启动器) | legacy | pro | easy
    mode: Mutex<String>,
    /// 调用模式: true = 服务监听 0.0.0.0 (局域网可调用 API), false = 仅 127.0.0.1
    expose: AtomicBool,
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
            });

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

            // 启动即展示 launcher.html, 服务按需拉起 (launch_console command)
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![launch_console, on_launcher_ready])
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
    if !matches!(mode.as_str(), "legacy" | "pro" | "easy") {
        return Err(format!("未知控制台模式: {mode}"));
    }
    // 探活最长 180s, 放到阻塞线程池, 避免卡住 async runtime
    tauri::async_runtime::spawn_blocking(move || launch_blocking(app, mode, expose.unwrap_or(false)))
        .await
        .map_err(|e| format!("内部任务失败: {e}"))?
}

/// launcher.html 每次加载时调用 (含从控制台返回): 收尾上一个控制台的 sidecar。
/// 远程控制台页 (http://127.0.0.1:P) 不在 capability 授权范围内, 无法走 IPC,
/// 其返回按钮是纯 location 导航回本页, 因此统一由本页 (本地 origin) 收尾。
#[tauri::command]
fn on_launcher_ready(app: AppHandle) {
    let state: State<ServerState> = app.state();
    kill_current(&state);
    if let Some(t) = app.tray_by_id("adr-tray") {
        let _ = t.set_tooltip(Some("ADR Studio — 引擎已停止"));
    }
}

// ---------------------------------------------------------------------------
// 启动与守护
// ---------------------------------------------------------------------------

/// launch_console 的阻塞实现: 杀旧 → 置模式 → 启动+探活+导航 → 交给守护线程。
fn launch_blocking(app: AppHandle, mode: String, expose: bool) -> Result<String, String> {
    let window = app.get_webview_window("main");
    let state: State<ServerState> = app.state();

    // 服务互斥: 切换前杀旧 sidecar, 等端口释放
    kill_current(&state);
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
    let Some(port) = free_tcp_port() else {
        update_status(window, "端口分配失败, 正在重试…");
        return None;
    };
    let (python, cwd, pythonpath) = resolve_runtime();
    let base = format!("http://127.0.0.1:{port}");
    // 调用模式: 0.0.0.0 对局域网开放 API; 默认仅本机 (webview 与探活都走 127.0.0.1)
    let host = if state.expose.load(Ordering::Relaxed) { "0.0.0.0" } else { "127.0.0.1" };

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
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            update_status(window, &format!("启动失败: {e}"));
            return None;
        }
    };
    state.pid.store(child.id(), Ordering::Relaxed);

    let health_path = if mode == "legacy" { "/" } else { "/api/adr/v1/health" };
    update_status(window, "引擎加载中… 首次启动导入 torch/CUDA 较慢");
    if !wait_healthy(&base, health_path, window) {
        kill_tree(child.id());
        let _ = child.wait();
        state.pid.store(0, Ordering::Relaxed);
        update_status(window, "启动超时, 请检查 Python 环境 (180s 探活超时)");
        return None;
    }

    *state.base_url.lock().unwrap() = Some(base.clone());
    // ?desktop=1: 控制台页据此显示"返回启动器" (壳内标记; 浏览器直开不带)
    let target = match mode {
        "legacy" => format!("{base}/"),
        "pro" => format!("{base}/pro?desktop=1"),
        _ => format!("{base}/easy?desktop=1"),
    };
    let expose_tip = if state.expose.load(Ordering::Relaxed) { " (对外服务)" } else { "" };
    if let Some(t) = app.tray_by_id("adr-tray") {
        let _ = t.set_tooltip(Some(format!("ADR Studio — {target}{expose_tip}")));
    }
    navigate(window, &target);
    Some((child, target, Instant::now()))
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
        kill_tree(pid);
        *state.base_url.lock().unwrap() = None;
    }
}

/// 真退出路径: 置退出标记 + 杀 sidecar 进程树 (幂等)。
fn shutdown_server(app: &AppHandle) {
    SHUTTING_DOWN.store(true, Ordering::Relaxed);
    let state: State<ServerState> = app.state();
    kill_current(&state);
}

/// Windows 下杀整棵进程树 (uvicorn/gradio 可能带子进程)。
fn kill_tree(pid: u32) {
    let mut cmd = Command::new("taskkill");
    cmd.args(["/PID", &pid.to_string(), "/T", "/F"]);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let _ = cmd.output();
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
