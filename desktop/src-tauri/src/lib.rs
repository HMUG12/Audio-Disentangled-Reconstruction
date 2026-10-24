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
//! 8. 设置中心 (批次48): 壳自带 shell.json (%LOCALAPPDATA%\ADR\shell.json),
//!    启动 sidecar 时以**进程环境变量**注入 (ADR_TTS_API_KEY / ADR_DATA_DIR /
//!    HF_ENDPOINT / ADR_TTS_DEFAULT_PROFILE / ADR_T2S_CUDAGRAPH / NUMBA_CACHE_DIR /
//!    端口覆盖) — 不污染系统环境变量, CLI 源码用户不受影响; 开机自启走
//!    HKCU Run 注册表 (reg.exe, 零新依赖)
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
use std::sync::{Arc, Mutex};
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
/// - Ready:      引擎就绪, 或预热页用户点「仍要进入」强制放行 (秒进通道开放, "ready")
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

// ---------------------------------------------------------------------------
// 设置中心 (批次48): shell.json 配置 + sidecar 环境变量注入
// ---------------------------------------------------------------------------

/// 设置中心配置 (launcher.html 设置模态框读写, sidecar 启动时注入)。
/// 全字段 Option: 旧版本文件缺字段/坏 JSON 一律回退默认, 不炸壳。
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct ShellSettings {
    // --- 常用层 ---
    /// 局域网开关默认值 (launcher 卡片勾选框可本次覆盖)
    lan_expose: Option<bool>,
    /// ADR_TTS_API_KEY — 逗号分隔多 key, 空串 = 鉴权关闭
    api_key: Option<String>,
    /// ADR_DATA_DIR 覆盖 (声音档案存放位置); 空串 = 用推导默认
    data_dir: Option<String>,
    /// true → HF_ENDPOINT=https://hf-mirror.com (国内模型下载镜像)
    hf_mirror: Option<bool>,
    /// 开机自启 (HKCU Run 注册表, 与本文件无关的独立持久化)
    autostart: Option<bool>,
    /// ADR_TTS_DEFAULT_PROFILE — /tts 缺省档案
    default_profile: Option<String>,
    // --- 高级抽屉 ---
    /// 服务端口覆盖; 0/None = 默认逻辑 (9881 探测 → 随机回退)
    service_port: Option<u16>,
    /// NUMBA_CACHE_DIR — numba JIT 缓存重定向 (批次47: 沙箱/杀软禁写
    /// site-packages 时预热会卡死, 重定向到可写目录即解)
    numba_cache_dir: Option<String>,
    /// ADR_T2S_CUDAGRAPH=1 — t2s CUDA Graph 加速 (音质有损, 默认关)
    cudagraph: Option<bool>,
    /// ADR_TTS_DEVICE — 推理设备走向 (批次49 低配预设): auto/cuda/cpu;
    /// auto/空 = 不注入 (后端按可用性探测)
    device: Option<String>,
    /// ADR_TORCH_THREADS — torch CPU 线程上限 (批次49 低配预设);
    /// 0/None = 不注入 (PyTorch 自按物理核)
    torch_threads: Option<u16>,
    /// 兼容模式 (批次57): 13/14 代 Intel (Raptor Lake) 计算密集段闪退缓解 —
    /// 全链指令集限到 AVX2 (oneDNN/MKL/numba), 绕开 AVX-512 电压缺陷路径。
    /// None = 壳自动检测 (Raptor Lake 自动开); Some = 用户手动覆盖
    compat_cpu: Option<bool>,
    // --- 声音安全 (批次62) ---
    /// ADR_SAFETY_AUDIT — 审计日志 (JSONL, data_dir/logs/audit.log); 默认开
    safety_audit: Option<bool>,
    /// ADR_WAV_METADATA — 分发 WAV 尾部 LIST/INFO 标注 AI 生成; 默认开
    wav_metadata: Option<bool>,
}

static SHELL_CFG: std::sync::OnceLock<Mutex<ShellSettings>> = std::sync::OnceLock::new();

fn shell_cell() -> &'static Mutex<ShellSettings> {
    SHELL_CFG.get_or_init(|| Mutex::new(ShellSettings::default()))
}

/// shell.json 落点: %LOCALAPPDATA%\ADR\shell.json (LOCALAPPDATA 缺失 → 数据目录旁)。
fn shell_settings_path() -> PathBuf {
    if let Ok(la) = std::env::var("LOCALAPPDATA") {
        if !la.trim().is_empty() {
            return PathBuf::from(la).join(r"ADR\shell.json");
        }
    }
    adr_data_dir().unwrap_or_else(std::env::temp_dir).join("shell.json")
}

/// 从盘加载配置并刷新进程内缓存 (启动时 / 保存后调用; 平时读缓存避免
/// 每条 proc_log 都碰文件系统)。坏 JSON / 读取失败 → 默认值。
fn reload_shell_config() -> ShellSettings {
    let cfg = std::fs::read_to_string(shell_settings_path())
        .ok()
        .and_then(|s| serde_json::from_str::<ShellSettings>(&s).ok())
        .unwrap_or_default();
    *shell_cell().lock().unwrap() = cfg.clone();
    cfg
}

/// 进程内缓存读取 (未加载过 → 读盘)。
fn shell_config() -> ShellSettings {
    shell_cell().lock().unwrap().clone()
}

/// shell.json → sidecar 环境变量映射 (纯函数, 单测覆盖)。
/// 只影响子进程, 不写壳进程自身环境; 空串字段一律跳过 (= 未配置)。
/// auto_compat: CPU 自动检测结果 (批次57) — compat_cpu=None 时按它定夺, Some 优先。
fn env_overrides(c: &ShellSettings, auto_compat: bool) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = Vec::new();
    if let Some(k) = c.api_key.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        v.push(("ADR_TTS_API_KEY".into(), k.into()));
    }
    if let Some(d) = c.data_dir.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        v.push(("ADR_DATA_DIR".into(), d.into()));
    }
    if c.hf_mirror.unwrap_or(false) {
        v.push(("HF_ENDPOINT".into(), "https://hf-mirror.com".into()));
    }
    if let Some(p) = c.default_profile.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        v.push(("ADR_TTS_DEFAULT_PROFILE".into(), p.into()));
    }
    if c.cudagraph.unwrap_or(false) {
        v.push(("ADR_T2S_CUDAGRAPH".into(), "1".into()));
    }
    if let Some(d) = c.numba_cache_dir.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        v.push(("NUMBA_CACHE_DIR".into(), d.into()));
    }
    // 批次49 低配预设: 设备走向 (auto/空 = 不注入, 后端按可用性探测);
    // 线程上限 (>0 才注入, 0 = PyTorch 自按物理核)
    if let Some(d) = c.device.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        if d == "cpu" || d == "cuda" {
            v.push(("ADR_TTS_DEVICE".into(), d.into()));
        }
    }
    if let Some(n) = c.torch_threads.filter(|n| *n > 0) {
        v.push(("ADR_TORCH_THREADS".into(), n.to_string()));
    }
    // 批次57 兼容模式: Intel 13/14 代 (Raptor Lake) AVX-512 高电压硬件缺陷可致
    // 计算密集段 native 崩溃 (0xC0000005, 闪退无 Python 栈)。全链指令集限到 AVX2
    // (oneDNN/MKL/numba), 性能略降换稳定。
    if c.compat_cpu.unwrap_or(auto_compat) {
        v.push(("ONEDNN_MAX_CPU_ISA".into(), "AVX2".into()));
        v.push(("MKL_ENABLE_INSTRUCTIONS".into(), "AVX2".into()));
        v.push(("NUMBA_CPU_NAME".into(), "generic".into()));
    }
    // 批次62 声音安全: 两项默认开, 仅用户显式关闭时注入 =0
    // (审计日志 / WAV AI 生成标识, 后端 settings.py 单源口径)
    if !c.safety_audit.unwrap_or(true) {
        v.push(("ADR_SAFETY_AUDIT".into(), "0".into()));
    }
    if !c.wav_metadata.unwrap_or(true) {
        v.push(("ADR_WAV_METADATA".into(), "0".into()));
    }
    v
}

/// CPU 品牌串 (批次57): reg 读注册表 (零依赖, 与 autostart 同套路),
/// 用于 13/14 代自动检测与日志留档。
#[cfg(windows)]
fn query_cpu_brand() -> Option<String> {
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
fn query_cpu_brand() -> Option<String> {
    None
}

/// Raptor Lake 判定 (纯函数, 单测覆盖): Intel 13/14 代品牌串。
fn brand_is_raptor_lake(brand: &str) -> bool {
    let b = brand.to_ascii_lowercase();
    b.contains("intel") && (b.contains("13th gen") || b.contains("14th gen"))
}

/// CPU 档案诊断 + 13/14 代自动兼容判定 (每次起 sidecar 调用, 结果落 server.log)。
/// 顺带留档 AVX2/AVX512 支持 — 缺失时 torch 官方轮子会直接闪退, 至少有据可查。
#[cfg(windows)]
fn cpu_auto_compat() -> bool {
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
fn cpu_auto_compat() -> bool {
    false
}

/// 自启注册表值: exe 全路径加引号 (含空格路径必需)。
fn autostart_cmd_value(exe: &std::path::Path) -> String {
    format!("\"{}\"", exe.display())
}

#[cfg(windows)]
const RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";
#[cfg(windows)]
const RUN_VALUE: &str = "ADR Studio";

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

            // 设置中心: 启动即加载 shell.json (数据目录推导/日志路径/后续注入全依赖)
            reload_shell_config();

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
            terms_agreed,
            agree_terms,
            reject_terms,
            skip_prewarm,
            retry_prewarm,
            back_to_launcher,
            load_shell_settings,
            save_shell_settings,
            lan_info,
            app_paths,
            autostart_get,
            autostart_set,
            open_path,
            cpu_compat_auto
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

/// 防重入互斥: UIA 单击也可能双投递 (相隔数秒的两个独立事件, JS 层 launching
/// 挡不住)。两次并发 launch_blocking 会双 spawn 引擎: kill_current 的
/// swap→store 窗口让第二个 kill_current 静默通过, 双 spawn 抢探活, 输家流程
/// 被别人引擎的 200 响应误判成功。Rust 侧 CAS 对一切事件来源免疫。
static LAUNCH_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// 启动指定控制台 (阻塞到探活通过并完成导航, 前端转 loading 态)。
/// expose = 调用模式: 服务监听 0.0.0.0 供局域网调用 (默认仅 127.0.0.1)。
#[tauri::command]
async fn launch_console(app: AppHandle, mode: String, expose: Option<bool>) -> Result<String, String> {
    if !matches!(mode.as_str(), "legacy" | "pro" | "easy" | "call") {
        return Err(format!("未知控制台模式: {mode}"));
    }
    if LAUNCH_IN_PROGRESS
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Err("已有控制台启动流程进行中".into());
    }
    // 探活最长 180s, 放到阻塞线程池, 避免卡住 async runtime
    let result = tauri::async_runtime::spawn_blocking(move || launch_blocking(app, mode, expose.unwrap_or(false)))
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
    // 批次61: 条款门禁 — 未同意首启条款时「仍要进入」不放行 (条款视图下该按钮
    // 本就不可达, 此处为纯防御兜底, 防止未来 UI 变动绕过条款流)。
    if !terms_agreed_flag() {
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
// 设置中心命令 (批次48): launcher.html 设置模态框经 invoke 调用
// ---------------------------------------------------------------------------

/// 读取设置 (进程内缓存, 启动时已从盘加载)。
#[tauri::command]
fn load_shell_settings() -> ShellSettings {
    shell_config()
}

/// 保存设置: 写 %LOCALAPPDATA%\ADR\shell.json + 刷新缓存。
/// 环境变量注入在下次 sidecar 启动时生效 (前端提示"重启引擎后生效")。
#[tauri::command]
fn save_shell_settings(settings: ShellSettings) -> Result<(), String> {
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

/// 本机默认路由出口 IP (UDP connect 不发包探测, 零依赖);
/// 离线/多网卡异常时返回空串 — 前端降级提示。
fn local_lan_ip() -> Option<String> {
    let s = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("223.5.5.5:80").ok()?; // 阿里公共 DNS: 仅路由决策, 不发数据
    s.local_addr().ok().map(|a| a.ip().to_string())
}

/// 局域网调用信息: 出口 IP + 当前服务地址 + 本次 expose 状态。
/// 还批次 45 Review 的债: key 输入框旁展示 LAN IP:端口与可复制完整 URL。
#[tauri::command]
fn lan_info(app: AppHandle) -> serde_json::Value {
    let state: State<ServerState> = app.state();
    serde_json::json!({
        "ip": local_lan_ip().unwrap_or_default(),
        "url": state.base_url.lock().unwrap().clone(),
        "expose": state.expose.load(Ordering::Relaxed),
    })
}

/// 壳常用路径 (设置页展示 + 打开按钮数据源)。
#[tauri::command]
fn app_paths() -> serde_json::Value {
    serde_json::json!({
        "data_dir": adr_data_dir().map(|p| p.to_string_lossy().into_owned()),
        "log_dir": server_log_path().parent().map(|p| p.to_string_lossy().into_owned()),
        "shell_json": shell_settings_path().to_string_lossy(),
    })
}

/// 开机自启状态 (HKCU Run 注册表查询; 非 Windows 恒 false)。
#[tauri::command]
#[cfg(windows)]
fn autostart_get() -> bool {
    let mut cmd = Command::new("reg");
    cmd.args(["query", RUN_KEY, "/v", RUN_VALUE]);
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd.output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[tauri::command]
#[cfg(not(windows))]
fn autostart_get() -> bool {
    false
}

/// 开/关开机自启: reg add / delete HKCU Run 键, 值为壳 exe 全路径。
#[tauri::command]
#[cfg(windows)]
fn autostart_set(enable: bool) -> Result<(), String> {
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
fn autostart_set(_enable: bool) -> Result<(), String> {
    Err("仅支持 Windows".into())
}

/// 用系统默认程序打开目标: 目录 (日志/数据/迁移引导) 或 URL (检查更新直达
/// Releases 页)。URL 不做存在性校验, 本地路径必须存在。
#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    if path.starts_with("http://") || path.starts_with("https://") {
        return open::that(&path).map_err(|e| format!("打开失败: {e}"));
    }
    let p = PathBuf::from(&path);
    if !p.exists() {
        return Err(format!("路径不存在: {path}"));
    }
    open::that(&p).map_err(|e| format!("打开失败: {e}"))
}

/// CPU 自动兼容检测结果 (批次57): launcher 在勾选框旁提示"已自动启用"。
/// 纯查询不落日志 (cpu_auto_compat 起引擎时已留档)。
#[tauri::command]
fn cpu_compat_auto() -> bool {
    query_cpu_brand().map(|b| brand_is_raptor_lake(&b)).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// 启动与守护
// ---------------------------------------------------------------------------

/// 与 adr/core/config.py adr_data_dir() 同序推导 (批次27 复审 ISSUE-2 对齐):
/// shell.json data_dir (批次48 设置中心, 最高优先) > ADR_DATA_DIR env >
/// F:/ADR_data (legacy, 存在即沿用) > %LOCALAPPDATA%\ADR\data。
/// 返回 None 表示交由 Python 侧默认 (~/.adr/data), 不注入子进程。
/// 空串环境变量视为未设置 (批次27 复审 ISSUE-4: PathBuf("") 会落到进程 CWD)。
fn adr_data_dir() -> Option<PathBuf> {
    // 批次48: 设置中心显式指定的档案目录最高优先 (读进程内缓存, 不碰盘)
    if let Some(v) = shell_config().data_dir {
        if !v.trim().is_empty() {
            return Some(PathBuf::from(v));
        }
    }
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

// ---------------------------------------------------------------------------
// 首启条款 (批次61): prewarm 前置条款页 + .terms_agreed_v1 标记 + 拒绝退出态
// ---------------------------------------------------------------------------

/// 条款同意标记路径: <数据目录>\.terms_agreed_v1 (数据目录与 Python adr_data_dir()
/// 同源推导)。覆盖安装保留数据目录 → 升级/重装不重复弹条款。
fn terms_flag_path() -> Option<PathBuf> {
    adr_data_dir().map(|d| d.join(".terms_agreed_v1"))
}

/// 内部查询 (prewarm_flow 门禁与 skip_prewarm 兜底共用): 标记文件存在即视为已同意。
fn terms_agreed_flag() -> bool {
    terms_flag_path().map(|p| p.exists()).unwrap_or(false)
}

/// 前端查询: 是否已同意首启条款 (prewarm.html 加载时调用, 决定展示条款页还是步骤流)。
#[tauri::command]
fn terms_agreed() -> bool {
    terms_agreed_flag()
}

/// 前端「同意」: 写标记文件 (内容含毫秒时间戳, 便于审计追溯)。
#[tauri::command]
fn agree_terms() -> Result<(), String> {
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
fn reject_terms(app: AppHandle) {
    proc_log("terms", "用户拒绝使用条款, 应用退出");
    app.exit(0);
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

/// 后台泵: 把服务 stdout/stderr 逐行追加写入共享日志文件。
/// 双线程共享同一句柄 (Arc<Mutex<File>>) 且整行读取 (read_until b'\n')，
/// 避免 io::copy 8KB 块拷贝导致的行劈裂与双线程字节交错。
fn pump_server_log<R: std::io::Read>(reader: R,
                                     file: Arc<Mutex<std::fs::File>>,
                                     tag: &str) {
    use std::io::{BufRead, BufReader, Write};
    let mut reader = BufReader::new(reader);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let mut f = file.lock().unwrap();
                let _ = f.write_all(&line);
            }
        }
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut f = file.lock().unwrap();
    let _ = writeln!(f, "\n===== [server {tag} {ts}] stream closed =====");
}

/// server.log 简易轮转: 超 10MB → rename server.log.old (单代覆盖, 不累加)。
/// 启动 sidecar 前调用一次; 运行中不再切 — append 增速低 (纯文本行), 可接受。
fn rotate_server_log() {
    let path = server_log_path();
    if let Ok(meta) = std::fs::metadata(&path) {
        if meta.len() > 10 * 1024 * 1024 {
            let old = path.with_extension("log.old");
            let _ = std::fs::remove_file(&old); // Windows rename 不覆盖已存在目标
            let _ = std::fs::rename(&path, old);
        }
    }
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
    // 批次48: 设置中心 service_port > 0 时最高优先 (仍被占则回退默认逻辑)。
    let cfg = shell_config();
    let cfg_port = cfg.service_port.unwrap_or(0);
    let (port, using_preferred) = if mode != "legacy" && cfg_port > 0 && probe_fixed_port(cfg_port, host)
    {
        (cfg_port, true)
    } else if mode != "legacy" && probe_fixed_port(ADR_PREFERRED_PORT, host) {
        (ADR_PREFERRED_PORT, true)
    } else {
        let Some(p) = free_tcp_port() else {
            update_status(window, "端口分配失败, 正在重试…");
            return None;
        };
        (p, false)
    };
    if mode != "legacy" && !using_preferred && cfg_port == 0 {
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
    // server.log 编码统一 (批次56 review): Windows pipe 下 Python 默认 ANSI(GBK)
    // 编码中文, 与壳的 UTF-8 proc_log 行混排乱码 — 与 console.py 起训练子进程对齐
    cmd.env("PYTHONIOENCODING", "utf-8");
    cmd.env("PYTHONUTF8", "1");
    // 简易轮转: server.log 超 10MB → rename .old (启动时一次, 单代覆盖)
    rotate_server_log();
    if let Some(pp) = &pythonpath {
        cmd.env("PYTHONPATH", pp);
    }
    // 批次27 复审 ISSUE-2: 注入数据目录, 保证子进程 Python (config.adr_data_dir()
    // 优先读此环境变量) 与壳的目录推导永远同源
    if let Some(d) = adr_data_dir() {
        cmd.env("ADR_DATA_DIR", &d);
    }
    // 批次48: 设置中心 shell.json → 进程环境变量注入 (API Key / HF 镜像 /
    // 默认档案 / CUDAGraph / numba 缓存目录 / data_dir 覆盖)。
    // 只写子进程环境, 不污染系统; 同 key 后写覆盖前写 (data_dir 双写值一致)
    for (k, v) in env_overrides(&cfg, cpu_auto_compat()) {
        cmd.env(k, v);
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
    // 单次打开句柄共享给 stdout/stderr 两泵; 头行由调用方单线程写一次
    if let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        use std::io::Write;
        let file = Arc::new(Mutex::new(file));
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        {
            let mut f = file.lock().unwrap();
            let _ = writeln!(f, "\n===== [server {ts}] stream open =====");
        }
        if let Some(out) = child.stdout.take() {
            let file = file.clone();
            std::thread::spawn(move || pump_server_log(out, file, "stdout"));
        }
        if let Some(err) = child.stderr.take() {
            std::thread::spawn(move || pump_server_log(err, file, "stderr"));
        }
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
/// 失败停在预热页 (「重试/仍要进入」按钮)。无总超时: v1.0.1 拍板取消超时
/// 自动放行 — 首启要下载 ~2GB 底模, 超时放行会让用户进面板才发现引擎没好。
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
    // 无总超时 (v1.0.1): downloading 阶段可能持续十几分钟 (底模 ~2GB), 必须等就绪才放行
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
        std::thread::sleep(Duration::from_millis(500));
    }

    // 就绪 (ready): 秒进通道开放 (批次37 状态机)
    state.set_phase(Phase::Ready);
    if let Some(t) = app.tray_by_id("adr-tray") {
        let _ = t.set_tooltip(Some("ADR Studio — 引擎已预热 (启动器)"));
    }
    // 批次61: 首启条款门禁 — 未同意条款前停在预热页不放行导航。
    // 引擎已在 setup 阶段照常预热 (本函数即预热流程), gate 只拦导航:
    // 用户点「同意」写入标记后, 下轮 500ms 轮询放行; 条款阅读期间不浪费预热等待。
    if !terms_agreed_flag() {
        eval_prewarm(&window, "kernel", "引擎已就绪, 请先阅读并同意使用条款");
        loop {
            if SHUTTING_DOWN.load(Ordering::Relaxed) {
                return;
            }
            if state.base_url.lock().unwrap().as_deref() != Some(base.as_str()) {
                return;
            }
            if terms_agreed_flag() {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
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
                Ok(Some(st)) => {
                    // 批次57: 退出码留痕 — 13/14 代闪退 (0xC0000005 等 native 崩溃) 无
                    // Python 栈, stderr 往往全空, 这一行是 server.log 里唯一的直接证据。
                    proc_log(
                        "server",
                        &format!("pid={my_pid} 退出, code={}", st.code().unwrap_or(-1)),
                    );
                    break; // 服务进程意外退出
                }
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
            // 批次57: 放弃重启时把完整日志路径落盘, 避免用户报障时无从下手
            let log_hint = format!(
                "连续 {failures} 次启动失败, 放弃自动重启; 完整日志: {}",
                server_log_path().display()
            );
            proc_log("server", &log_hint);
            if let Some(t) = app.tray_by_id("adr-tray") {
                let _ = t.set_tooltip(Some("ADR Studio 引擎启动失败"));
            }
            update_status(&window, "多次启动失败, 已停止自动重启。请到 设置 → 维护 → 打开日志目录 查看原因后重试。");
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

// ---------------------------------------------------------------------------
// 单元测试 (批次48): cargo test --lib
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_overrides_maps_all_fields() {
        let c = ShellSettings {
            lan_expose: Some(true),          // 不进环境变量 (launch_console 参数语义)
            api_key: Some(" k1,k2 ".into()), // trim 保留 key 内部
            data_dir: Some(" D:/voices ".into()),
            hf_mirror: Some(true),
            autostart: Some(true),           // 独立注册表持久化, 不进环境变量
            default_profile: Some(" my_voice ".into()),
            service_port: Some(9900),        // 端口走 start_server 探测, 不进环境变量
            numba_cache_dir: Some(" F:/nc ".into()),
            cudagraph: Some(true),
            device: Some(" cpu ".into()),    // 批次49 低配预设
            torch_threads: Some(4),
            compat_cpu: Some(true),          // 批次57 13/14 代兼容模式
        };
        let m: std::collections::HashMap<String, String> = env_overrides(&c, false).into_iter().collect();
        assert_eq!(m.get("ADR_TTS_API_KEY").unwrap(), "k1,k2");
        assert_eq!(m.get("ADR_DATA_DIR").unwrap(), "D:/voices");
        assert_eq!(m.get("HF_ENDPOINT").unwrap(), "https://hf-mirror.com");
        assert_eq!(m.get("ADR_TTS_DEFAULT_PROFILE").unwrap(), "my_voice");
        assert_eq!(m.get("NUMBA_CACHE_DIR").unwrap(), "F:/nc");
        assert_eq!(m.get("ADR_T2S_CUDAGRAPH").unwrap(), "1");
        assert_eq!(m.get("ADR_TTS_DEVICE").unwrap(), "cpu");
        assert_eq!(m.get("ADR_TORCH_THREADS").unwrap(), "4");
        // 兼容模式: 全链限 ISA 到 AVX2 (oneDNN/MKL/numba)
        assert_eq!(m.get("ONEDNN_MAX_CPU_ISA").unwrap(), "AVX2");
        assert_eq!(m.get("MKL_ENABLE_INSTRUCTIONS").unwrap(), "AVX2");
        assert_eq!(m.get("NUMBA_CPU_NAME").unwrap(), "generic");
        assert_eq!(m.len(), 11);
    }

    #[test]
    fn env_overrides_default_and_empty_skip() {
        // 全默认 → 零注入 (鉴权关闭/HF 官方源等后端原语义)
        assert!(env_overrides(&ShellSettings::default(), false).is_empty());
        // 空串 = 未配置, 一律跳过
        let c = ShellSettings {
            api_key: Some("   ".into()),
            data_dir: Some("".into()),
            default_profile: Some("".into()),
            numba_cache_dir: Some("".into()),
            ..Default::default()
        };
        assert!(env_overrides(&c, false).is_empty());
        // hf_mirror=false / cudagraph=false → 不注入
        let c = ShellSettings { hf_mirror: Some(false), cudagraph: Some(false), ..Default::default() };
        assert!(env_overrides(&c, false).is_empty());
    }

    #[test]
    fn env_overrides_compat_auto_semantics() {
        // 批次57: compat_cpu=None → 按 auto_compat 自动; Some 手动优先
        let m: std::collections::HashMap<String, String> =
            env_overrides(&ShellSettings::default(), true).into_iter().collect();
        assert_eq!(m.get("ONEDNN_MAX_CPU_ISA").unwrap(), "AVX2");
        assert_eq!(m.get("MKL_ENABLE_INSTRUCTIONS").unwrap(), "AVX2");
        assert_eq!(m.get("NUMBA_CPU_NAME").unwrap(), "generic");
        assert_eq!(m.len(), 3);
        // 手动关 → 即便自动检出 13 代也不注入
        let c = ShellSettings { compat_cpu: Some(false), ..Default::default() };
        assert!(env_overrides(&c, true).is_empty());
        // 品牌串判定: Intel 品牌串固定带 "13th/14th Gen" 前缀
        assert!(brand_is_raptor_lake("13th Gen Intel(R) Core(TM) i7-13700K"));
        assert!(brand_is_raptor_lake("14th Gen Intel(R) Core(TM) i9-14900KS"));
        assert!(!brand_is_raptor_lake("AMD Ryzen 9 7950X"));
        assert!(!brand_is_raptor_lake("12th Gen Intel(R) Core(TM) i7-12700K"));
    }

    #[test]
    fn shell_settings_serde_roundtrip_and_defaults() {
        let c = ShellSettings { api_key: Some("k".into()), hf_mirror: Some(true), ..Default::default() };
        let s = serde_json::to_string(&c).unwrap();
        let back: ShellSettings = serde_json::from_str(&s).unwrap();
        assert_eq!(back.api_key.as_deref(), Some("k"));
        assert_eq!(back.hf_mirror, Some(true));
        // 旧版本文件缺字段 / 含未知字段 → 宽松解析不炸
        let legacy: ShellSettings = serde_json::from_str(r#"{"api_key":"x","unknown_field":1}"#).unwrap();
        assert_eq!(legacy.api_key.as_deref(), Some("x"));
        assert_eq!(legacy.service_port, None);
        // 坏 JSON → 默认 (reload_shell_config 同口径)
        assert!(serde_json::from_str::<ShellSettings>("{oops").is_err());
    }

    #[test]
    fn autostart_value_quotes_path() {
        assert_eq!(
            autostart_cmd_value(std::path::Path::new("C:/Program Files/ADR/adr-desktop.exe")),
            r#""C:/Program Files/ADR/adr-desktop.exe""#
        );
    }
}
