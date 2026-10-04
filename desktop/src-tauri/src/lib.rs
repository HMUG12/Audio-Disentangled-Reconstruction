//! ADR Studio 桌面壳 (Tauri 2)。
//!
//! 职责:
//! 1. sidecar 托管: 启动 ``python -m adr.cli webui --host 127.0.0.1 --port <随机空闲端口> --no-browser``
//!    (Gradio Web 控制台, 与命令行 ``adr webui`` 同一入口)
//! 2. 健康探活: 轮询 ``GET /`` 通过后把窗口导航到本地 Web 控制台
//! 3. 进程守护: 服务意外退出自动重启 (稳定运行 60s 后重置失败计数)
//! 4. 托盘: 显示窗口 / 浏览器打开 / 退出 (退出时 taskkill /T 杀整棵服务进程树)
//! 5. 关窗 = 隐藏到托盘, 真正退出只走托盘菜单
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
use std::process::Command;
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

/// 共享状态: sidecar PID (0 = 未运行) 与就绪后的服务地址。
struct ServerState {
    pid: AtomicU32,
    base_url: Mutex<Option<String>>,
}

/// 退出标记: 托盘退出 / RunEvent::Exit 时置位, supervisor 轮询后收尾
static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            app.manage(ServerState {
                pid: AtomicU32::new(0),
                base_url: Mutex::new(None),
            });

            // 系统托盘
            let show_i = MenuItem::with_id(app, "show", "显示主窗口", true, None::<&str>)?;
            let web_i = MenuItem::with_id(app, "web", "在浏览器打开控制台", true, None::<&str>)?;
            let quit_i = MenuItem::with_id(app, "quit", "退出", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_i, &web_i, &quit_i])?;
            let mut tray = TrayIconBuilder::with_id("adr-tray")
                .menu(&menu)
                .tooltip("ADR Studio 启动中…")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
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

            // sidecar 监管线 (后台线程, 不阻塞 UI)
            let handle = app.handle().clone();
            std::thread::spawn(move || supervisor(handle));

            Ok(())
        })
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

/// 停掉 sidecar 进程树 (幂等)。
fn shutdown_server(app: &AppHandle) {
    SHUTTING_DOWN.store(true, Ordering::Relaxed);
    let state: State<ServerState> = app.state();
    let pid = state.pid.swap(0, Ordering::Relaxed);
    if pid != 0 {
        kill_tree(pid);
    }
}

/// Windows 下杀整棵进程树 (uvicorn 可能带子进程)。
fn kill_tree(pid: u32) {
    let mut cmd = Command::new("taskkill");
    cmd.args(["/PID", &pid.to_string(), "/T", "/F"]);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let _ = cmd.output();
}

/// sidecar 监管线: 启动 → 探活 → 导航 → 监控 → 崩溃重启。
fn supervisor(app: AppHandle) {
    let window = app.get_webview_window("main");
    let state: State<ServerState> = app.state();
    let mut failures = 0u32;

    while failures <= MAX_FAILURES && !SHUTTING_DOWN.load(Ordering::Relaxed) {
        update_status(&window, "正在启动本地引擎…");
        let Some(port) = free_tcp_port() else {
            std::thread::sleep(Duration::from_secs(2));
            failures += 1;
            continue;
        };
        let (python, cwd, pythonpath) = resolve_runtime();
        let url = format!("http://127.0.0.1:{port}");

        let mut cmd = Command::new(&python);
        cmd.args([
            "-m", "adr.cli", "webui",
            "--host", "127.0.0.1",
            "--port", &port.to_string(),
            "--no-browser",
        ])
        .current_dir(&cwd);
        if let Some(pp) = &pythonpath {
            cmd.env("PYTHONPATH", pp);
        }
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);

        match cmd.spawn() {
            Ok(mut child) => {
                state.pid.store(child.id(), Ordering::Relaxed);
                if wait_healthy(&url, &window) {
                    let started = Instant::now();
                    *state.base_url.lock().unwrap() = Some(url.clone());
                    if let Some(t) = app.tray_by_id("adr-tray") {
                        let _ = t.set_tooltip(Some(format!("ADR Studio — {url}")));
                    }
                    navigate(&window, &url);

                    // 监控运行 (轮询而非阻塞 wait, 保证退出路径畅通)
                    while !SHUTTING_DOWN.load(Ordering::Relaxed) {
                        match child.try_wait() {
                            Ok(None) => std::thread::sleep(Duration::from_millis(500)),
                            Ok(Some(_)) => break, // 服务进程意外退出 → 重启
                            Err(_) => break,
                        }
                    }
                    let _ = child.kill(); // 兜底 (正常退出路径 taskkill 已处理)
                    state.pid.store(0, Ordering::Relaxed);
                    *state.base_url.lock().unwrap() = None;
                    if let Some(t) = app.tray_by_id("adr-tray") {
                        let _ = t.set_tooltip(Some("ADR Studio 引擎重启中…"));
                    }
                    // 稳定运行够久 → 视为一次成功, 重置失败计数
                    if started.elapsed() >= STABLE_UPTIME {
                        failures = 0;
                    }
                } else {
                    // 探活超时: 杀掉重试
                    kill_tree(child.id());
                    let _ = child.wait();
                    state.pid.store(0, Ordering::Relaxed);
                    update_status(&window, "启动超时, 正在重试…");
                }
                failures += 1;
                std::thread::sleep(Duration::from_secs(2));
            }
            Err(e) => {
                update_status(&window, &format!("启动失败: {e}"));
                failures += 1;
                std::thread::sleep(Duration::from_secs(3));
            }
        }
    }

    if !SHUTTING_DOWN.load(Ordering::Relaxed) {
        if let Some(t) = app.tray_by_id("adr-tray") {
            let _ = t.set_tooltip(Some("ADR Studio 引擎启动失败"));
        }
        update_status(&window, "多次启动失败, 请检查 Python 环境后重启应用。");
    }
}

/// 轮询 Web 控制台首页 (/) 直到 200 或超时 (Gradio 就绪即全站可用)。
fn wait_healthy(base: &str, window: &Option<WebviewWindow>) -> bool {
    let url = format!("{base}/");
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

/// 把窗口导航到就绪后的本地控制台。
fn navigate(window: &Option<WebviewWindow>, url: &str) {
    if let Some(w) = window {
        let _ = w.eval(&format!("location.replace({});", js_quote(url)));
    }
}

fn js_quote(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}
