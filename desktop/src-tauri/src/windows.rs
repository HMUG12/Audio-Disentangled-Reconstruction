//! 主窗口 / 托盘 / 页面导航 (M2 拆分, 批次131)。
//!
//! 从 `lib.rs` 原样迁出, 行为不变: 主窗口代码创建 (替代 tauri.conf.json windows),
//! 以便挂 `on_download` — WebView2 对网页 a[download] 默认静默取消下载
//! (控制台"下载音频"失败根因); 托盘菜单提供 显示/返回启动器/浏览器打开/退出。

use tauri::{
    App, Manager, State, WebviewWindow,
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
};

use crate::lifecycle::{kill_current, shutdown_server, ServerState};

/// WebView2 前端资源 origin (Windows)。
/// 导航一律用绝对地址: 当前页面可能停在远端控制台 (http://127.0.0.1:P),
/// 相对路径会指到远端 origin 上导致 404。
#[cfg(windows)]
const TAURI_ORIGIN: &str = "http://tauri.localhost";
#[cfg(not(windows))]
const TAURI_ORIGIN: &str = "tauri://localhost";

/// 创建主窗口 (prewarm.html 首屏) 并注册下载事件。
pub(crate) fn create_main_window(app: &App) -> tauri::Result<()> {
    // 批次100: 用 App 相对路径而不是硬编码 origin — dev 下 tauri dev 会把
    // frontendDist 挂到内置 dev server 并注入 build.devUrl (此时嵌入资源表
    // 为空, 访问 tauri protocol 会 "asset not found: prewarm.html"),
    // WebviewUrl::App 由 tauri 按 devUrl/tauri protocol 自动解析。
    tauri::webview::WebviewWindowBuilder::new(
        app,
        "main",
        tauri::WebviewUrl::App("prewarm.html".into()),
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
    Ok(())
}

/// 系统托盘 (显示主窗口 / 返回启动器 / 浏览器打开 / 退出)。
pub(crate) fn create_tray(app: &App) -> tauri::Result<()> {
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
                navigate(&window, &app_page(&window, "launcher.html"));
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
    Ok(())
}

/// 在加载页上更新状态文本 (JSON 引号转义, 无注入风险)。
pub(crate) fn update_status(window: &Option<WebviewWindow>, text: &str) {
    if let Some(w) = window {
        let _ = w.eval(&format!(
            "window.__adrStatus && window.__adrStatus({});",
            js_quote(text)
        ));
    }
}

/// 应用前端页面的 origin (批次100)。
///
/// `tauri dev` 在 frontendDist 为目录时会把它挂到内置 dev server 上, 并把
/// ``build.devUrl`` 注入编译期配置 (TAURI_CONFIG) — 这种情况下嵌入资源表为空,
/// 硬编码 tauri protocol origin 会 "asset not found"。运行时以配置里的 devUrl
/// 为准, 没有 devUrl (打包 / 无 dev server) 时回退 tauri protocol origin。
pub(crate) fn app_origin(window: &Option<WebviewWindow>) -> String {
    let dev_url = window
        .as_ref()
        .and_then(|w| w.config().build.dev_url.clone());
    match dev_url {
        Some(u) => u.as_str().trim_end_matches('/').to_string(),
        None => TAURI_ORIGIN.to_string(),
    }
}

/// 应用页面 (prewarm/loading/launcher) 的绝对地址, 见 [`app_origin`]。
pub(crate) fn app_page(window: &Option<WebviewWindow>, page: &str) -> String {
    format!("{}/{}", app_origin(window), page.trim_start_matches('/'))
}

/// 控制台页「返回启动器」用的壳 origin 查询串 (批次101)。
/// 控制台页是远程 origin (127.0.0.1:P), 拿不到壳的 IPC, 只能靠查询参数回传
/// 壳地址: dev 下壳在内置 dev server (http://127.0.0.1:1645), 打包态在
/// tauri protocol origin。控制台页硬编码 tauri.localhost 在 dev 下会命中空
/// 资源表 → "asset not found: launcher.html"。
/// 非 http(s) 形态 (macOS/Linux 的 tauri://) 返回空串 → 控制台页回落到原逻辑。
pub(crate) fn shell_origin_param(window: &Option<WebviewWindow>) -> String {
    let origin = app_origin(window);
    if !(origin.starts_with("http://") || origin.starts_with("https://")) {
        return String::new();
    }
    let enc: String = origin
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b':' | b'/' => {
                (b as char).to_string()
            }
            _ => format!("%{:02X}", b),
        })
        .collect();
    format!("&shell_origin={enc}")
}

/// 把窗口导航到目标页 (必须是绝对 URL, 见 TAURI_ORIGIN 注释)。
pub(crate) fn navigate(window: &Option<WebviewWindow>, url: &str) {
    if let Some(w) = window {
        let _ = w.eval(&format!("location.replace({});", js_quote(url)));
    }
}

pub(crate) fn js_quote(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}
