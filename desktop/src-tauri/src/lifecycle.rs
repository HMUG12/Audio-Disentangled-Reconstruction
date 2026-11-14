//! 服务生命周期 (M2 拆分, 批次130): 状态机 / 守护重启 / 引擎预热轮询 /
//! 进程树结束与孤儿收编 / 通用进程日志。
//!
//! 从 `lib.rs` 原样迁出, **行为不变**。sidecar 启动与解释器解析在
//! [`crate::sidecar`], 窗口辅助 (navigate/app_page/update_status/js_quote) 在 crate 根。

use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager, State, WebviewWindow};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use crate::account;
use crate::sidecar::{server_log_path, start_and_wait, start_server};
use crate::system::{setup_done_flag, terms_agreed_flag};
#[cfg(windows)]
use crate::system::CREATE_NO_WINDOW;
use crate::windows::{app_page, js_quote, navigate, update_status};

/// 服务 (重) 启动失败上限
const MAX_FAILURES: u32 = 4;
/// 单次启动健康探活超时 (首次导入 torch/CUDA 较慢)
const HEALTH_TIMEOUT: Duration = Duration::from_secs(180);
/// 稳定运行超过该时长后重置失败计数
const STABLE_UPTIME: Duration = Duration::from_secs(60);

/// 服务生命周期状态机 (批次37): 显式四态替代原 prewarm_done 布尔与散落组合。
/// 转换图: Stopped → Starting → Prewarming → Ready → (kill) → Stopped
/// - Stopped:    无服务进程 (初始 / kill_current / 探活失败)
/// - Starting:   进程已 spawn, health 探活未过 (engine_status → "loading")
/// - Prewarming: 服务已 healthy, prewarm_flow 轮询引擎预热中 ("loading")
/// - Ready:      引擎就绪, 或预热页用户点「仍要进入」强制放行 (秒进通道开放, "ready")
/// 两条不变式保证枚举与进程一致: pid 写入处必置 Starting, pid 清零处必置 Stopped。
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Stopped = 0,
    Starting = 1,
    Prewarming = 2,
    Ready = 3,
}

/// 共享状态: sidecar PID (0 = 未运行)、就绪后的服务地址、当前控制台模式。
pub(crate) struct ServerState {
    pub(crate) pid: AtomicU32,
    pub(crate) base_url: Mutex<Option<String>>,
    /// "" (启动器) | legacy | pro | easy
    pub(crate) mode: Mutex<String>,
    /// 调用模式: true = 服务监听 0.0.0.0 (局域网可调用 API), false = 仅 127.0.0.1
    pub(crate) expose: AtomicBool,
    /// 生命周期状态机 (批次37, 替代原 prewarm_done: AtomicBool)
    pub(crate) phase: AtomicU8,
}

impl ServerState {
    pub(crate) fn phase(&self) -> Phase {
        match self.phase.load(Ordering::Relaxed) {
            1 => Phase::Starting,
            2 => Phase::Prewarming,
            3 => Phase::Ready,
            _ => Phase::Stopped,
        }
    }
    pub(crate) fn set_phase(&self, p: Phase) {
        self.phase.store(p as u8, Ordering::Relaxed);
    }
}

/// 退出标记: 托盘退出 / RunEvent::Exit 时置位, supervise 轮询后收尾。
/// 重启场景 (launch_console 切换模式) 只杀 sidecar, 不动该标记。
static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

/// 批次102: 预热流程是否仍在运行 (含"等登录"门禁等待期)。
/// 登录前移后出现两个导航来源: prewarm_flow (引擎就绪 + 已登录 → 跳启动器)
/// 与前端 enter_launcher (登出后重新登录 → 主动跳启动器)。用本标志让两者互斥,
/// 避免同一页面被连续导航两次 (启动器白闪一次重载)。
pub(crate) static PREWARM_ACTIVE: AtomicBool = AtomicBool::new(false);

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

/// 从 stats 拉引擎预热阶段 + 实时进度文案 + 失败原因摘要
/// (批次44: 下载进度透传; 批次73: failed 附带 traceback 摘要, 不触发加载)。
fn fetch_engine_stage_info(base: &str) -> Option<(String, String, String)> {
    // 读 /health 而非 /system/stats: 两者都含 engine_stage/engine_stage_text,
    // 但启用 ADR_TTS_API_KEY 后 /system/stats 要鉴权, 壳轮询不带 key → 401 →
    // 预热永久卡住; /health 在 auth 豁免名单内, 永远可达。
    let url = format!("{base}/api/adr/v1/health");
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
    let error = v
        .get("engine_error")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    Some((stage, text, error))
}

/// 驱动预热页失败详情 (展开区 / 复制诊断, 批次73)。
fn eval_prewarm_error(window: &Option<WebviewWindow>, error: &str) {
    if error.trim().is_empty() {
        return;
    }
    if let Some(w) = window {
        let _ = w.eval(&format!(
            "window.__adrPrewarmError && window.__adrPrewarmError({});",
            js_quote(error)
        ));
    }
}

/// traceback 摘要取末行当标题 (通常是 "XxxError: 具体原因"), 过长截断。
fn error_headline(error: &str) -> String {
    let last = error
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .next_back()
        .unwrap_or("");
    if last.is_empty() {
        return "引擎预热失败".into();
    }
    if last.chars().count() > 200 {
        let t: String = last.chars().take(200).collect();
        return format!("{t}…");
    }
    last.to_string()
}

/// 启动预热主流程: 拉服务 → 轮询引擎阶段 → ready 后自动放行进启动器。
/// 失败停在预热页 (「重试/仍要进入」按钮)。无总超时: v1.0.1 拍板取消超时
/// 自动放行 — 首启要下载 ~2GB 底模, 超时放行会让用户进面板才发现引擎没好。
pub(crate) fn prewarm_flow(app: AppHandle) {
    // 批次102: 标记预热流程在跑 (登录前移后与 enter_launcher 互斥导航)。
    PREWARM_ACTIVE.store(true, Ordering::Relaxed);
    prewarm_flow_inner(app);
    PREWARM_ACTIVE.store(false, Ordering::Relaxed);
}

fn prewarm_flow_inner(app: AppHandle) {
    let window = app.get_webview_window("main");
    let state: State<ServerState> = app.state();

    // 批次101: 首启配置向导前置 — 未完成配置前不启动引擎。配置项 (数据目录 /
    // 推理设备 / CPU 线程上限 / 开机自启) 都是在 sidecar 启动那一刻以环境变量
    // 注入的, 引擎起来之后再让用户选 = 选了不生效, 必须挡在引擎启动之前。
    if !setup_done_flag() {
        eval_prewarm(&window, "queued", "等待完成首次启动配置…");
        loop {
            if SHUTTING_DOWN.load(Ordering::Relaxed) {
                return;
            }
            if setup_done_flag() {
                break;
            }
            std::thread::sleep(Duration::from_millis(300));
        }
    }

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
            Some((st, _, _)) if st == "ready" => break,
            Some((st, _, err)) if st == "failed" => {
                // 批次73: 失败不再只显示固定文案 — 带上 traceback 末行做标题,
                // 完整摘要推给预热页展开区 (含"复制诊断")
                eval_prewarm(&window, "failed", &format!("引擎预热失败: {}", error_headline(&err)));
                eval_prewarm_error(&window, &err);
                return;
            }
            Some((st, text, _)) if !st.is_empty() => {
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
    // 批次102: 本机账号登录门禁 — 登录前移到预热页首屏 (壳一启动就弹),
    // 登录成功前不放行导航; 引擎已在后台预热, 登录期间不浪费时间。
    if !account::logged_in() {
        eval_prewarm(&window, "kernel", "引擎已就绪, 请登录本机账号后进入启动器");
        loop {
            if SHUTTING_DOWN.load(Ordering::Relaxed) {
                return;
            }
            if state.base_url.lock().unwrap().as_deref() != Some(base.as_str()) {
                return;
            }
            if account::logged_in() {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }
    navigate(&window, &app_page(&window, "launcher.html"));
}

/// 守护线程: 监控 child, 意外退出按同 mode 自动重启。
/// 所有权规则: state.pid != 本线程 child 的 pid → 已被新 launch_console 接管, 让位退出。
pub(crate) fn supervise(app: AppHandle, mode: String, mut child: Child, mut started: Instant) {
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
        navigate(&window, &app_page(&window, "loading.html"));
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
pub(crate) fn kill_current(state: &State<ServerState>) {
    let pid = state.pid.swap(0, Ordering::Relaxed);
    if pid != 0 {
        // 日志留痕 (批次40): 被杀 pid 与 taskkill 结果落服务日志, 排障可查。
        // 批次72: 三分类 — 杀成 / 已退出 (竞态, 目标已达成) / 失败 (Access denied 等)
        let note = match kill_tree(pid) {
            KillOutcome::Killed =>
                format!("kill_current: taskkill /T /F pid={pid} 成功"),
            KillOutcome::AlreadyGone =>
                format!("kill_current: taskkill /T /F pid={pid} 进程已退出 (无需杀)"),
            KillOutcome::Failed =>
                format!("kill_current: taskkill /T /F pid={pid} 失败 (Access denied / 驱动挂死), \
                         可能遗留孤儿, 下次启动由收编逻辑处理"),
        };
        proc_log("kill", &note);
        *state.base_url.lock().unwrap() = None;
        // 不变式 (批次37): pid 清零 → Stopped
        state.set_phase(Phase::Stopped);
    }
}

/// 真退出路径: 置退出标记 + 杀 sidecar 进程树 (幂等)。
pub(crate) fn shutdown_server(app: &AppHandle) {
    SHUTTING_DOWN.store(true, Ordering::Relaxed);
    let state: State<ServerState> = app.state();
    kill_current(&state);
}

/// taskkill 结果三分类 (批次72): 杀成 / 进程已退出 / 失败。
pub(crate) enum KillOutcome {
    Killed,
    /// taskkill 退出码 128 = 进程不存在 — kill_current 的 swap→taskkill 窗口内
    /// 进程自然退出 (探活失败自愈 / supervise 抢先收割) 的竞态, 目标已达成,
    /// 不算失败 (此前与 Access denied 混记一条"失败"日志, 误导排障)。
    AlreadyGone,
    /// Access denied (驱动挂死/提权进程, 批次37 实测) 或其他非 0 退出码。
    Failed,
}

/// taskkill 退出码 → 结果分类 (纯函数, 单测覆盖)。
/// None (理论上 Windows taskkill 不出现) 保守归 Failed。
pub(crate) fn classify_taskkill_code(code: Option<i32>) -> KillOutcome {
    match code {
        Some(0) => KillOutcome::Killed,
        Some(128) => KillOutcome::AlreadyGone,
        _ => KillOutcome::Failed,
    }
}

/// Windows 下杀整棵进程树 (uvicorn/gradio 可能带子进程)。
/// 返回 taskkill 结果分类 — 驱动挂死/提权进程会 Access denied (批次37 实测),
/// 收编方据此记日志, 不静默吞掉。
pub(crate) fn kill_tree(pid: u32) -> KillOutcome {
    let mut cmd = Command::new("taskkill");
    cmd.args(["/PID", &pid.to_string(), "/T", "/F"]);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd.output()
        .map(|o| classify_taskkill_code(o.status.code()))
        .unwrap_or(KillOutcome::Failed)
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
        match kill_tree(pid) {
            KillOutcome::Killed => {
                killed += 1;
                reap_log(&format!("收编孤儿 pid={pid}: {cmdline}"));
            }
            KillOutcome::AlreadyGone =>
                // 批次72: 枚举→taskkill 窗口内自然退出, 目标已达成, 不算失败
                reap_log(&format!("孤儿已自行退出 pid={pid}: {cmdline}")),
            KillOutcome::Failed =>
                // 驱动挂死 (CUDA 卡死) / admin 提权进程: 用户态杀不掉, 记日志放行
                reap_log(&format!("收编失败 (taskkill Access denied) pid={pid}: {cmdline}")),
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
pub(crate) fn proc_log(tag: &str, msg: &str) {
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
pub(crate) fn wait_healthy(base: &str, path: &str, window: &Option<WebviewWindow>) -> bool {
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
