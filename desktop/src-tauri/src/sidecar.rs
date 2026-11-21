//! sidecar 进程管理 (M2 拆分, 批次130): 服务日志泵/轮转、启动与健康探活、
//! Python 解释器解析、端口分配。
//!
//! 从 `lib.rs` 原样迁出, **行为不变**。进程守护/状态机在 [`crate::lifecycle`],
//! 窗口辅助 (navigate/app_page/update_status) 仍在 crate 根。

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager, State, WebviewWindow};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use crate::config::{adr_data_dir, env_overrides, shell_config};
use crate::lifecycle::{
    kill_current, kill_tree, proc_log, supervise, wait_healthy, Phase, ServerState,
};
use crate::system::cpu_auto_compat;
#[cfg(windows)]
use crate::system::CREATE_NO_WINDOW;
use crate::windows::{app_page, navigate, shell_origin_param, update_status};

/// 服务日志目录 (批次26, 批次27 重构): ADR_LOG_DIR > <数据目录>\logs
/// (均失败回落 %TEMP%\adr_server.log)。数据目录经 adr_data_dir() 与 Python 同源 —
/// 老机器 (F:\ADR_data) 日志随数据落 F: 盘, 新机器落 %LOCALAPPDATA%\ADR\data\logs。
/// 追加写, 启动时打时间戳分隔行。
pub(crate) fn server_log_path() -> PathBuf {
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
/// 批次87 (BUG-02): 日志句柄改为可选 — 打不开日志时仍必须把管道抽干,
/// 否则子进程写满 stdout/stderr 缓冲区 (4~64KB) 后会阻塞, 表现为启动卡死。
fn pump_server_log<R: std::io::Read>(reader: R,
                                     file: Option<Arc<Mutex<std::fs::File>>>,
                                     tag: &str) {
    use std::io::{BufRead, BufReader, Write};
    let mut reader = BufReader::new(reader);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                if let Some(file) = &file {
                    let mut f = file.lock().unwrap();
                    let _ = f.write_all(&line);
                }
            }
        }
    }
    if let Some(file) = &file {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut f = file.lock().unwrap();
        let _ = writeln!(f, "\n===== [server {tag} {ts}] stream closed =====");
    }
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
pub(crate) fn launch_blocking(app: AppHandle, mode: String, expose: bool) -> Result<String, String> {
    let window = app.get_webview_window("main");
    let state: State<ServerState> = app.state();

    // 服务互斥: 切换前杀旧 sidecar, 等端口释放
    kill_current(&state); // 置 Stopped (批次37 状态机); start_server 成功后转 Starting
    *state.mode.lock().unwrap() = mode.clone();
    state.expose.store(expose, Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(600));

    navigate(&window, &app_page(&window, "loading.html"));
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
pub(crate) fn start_and_wait(
    app: &AppHandle,
    mode: &str,
    window: &Option<WebviewWindow>,
    state: &State<ServerState>,
) -> Option<(Child, String, Instant)> {
    let (child, started) = start_server(window, state)?;
    // 探活成功 (start_server 内 wait_healthy 已 200) 才转 Prewarming (批次40):
    // 维持不变式 "Prewarming: 服务已 healthy" — 此前本路径 Phase 停在 Starting。
    // 前端零感知: Starting/Prewarming 在 engine_status 中都映射 "loading"。
    state.set_phase(Phase::Prewarming);
    let base = state.base_url.lock().unwrap().clone().unwrap_or_default();
    // ?desktop=1: 控制台页据此显示"返回启动器" (壳内标记; 浏览器直开不带)
    // shell_origin: 把壳地址回传给控制台页 — 控制台页是远程 origin, 拿不到壳的
    // IPC, 硬编码 tauri.localhost 在 dev 下命中空资源表 ("asset not found:
    // launcher.html"), 批次101 起由壳显式告知。
    let sop = shell_origin_param(window);
    let target = match mode {
        "pro" => format!("{base}/pro?desktop=1{sop}"),
        "easy" => format!("{base}/easy?desktop=1{sop}"),
        _ => format!("{base}/call?desktop=1{sop}"),
    };
    let expose_tip = if state.expose.load(Ordering::Relaxed) { " (对外服务)" } else { "" };
    if let Some(t) = app.tray_by_id("adr-tray") {
        let _ = t.set_tooltip(Some(format!("ADR Studio — {target}{expose_tip}")));
    }
    // 批次66: 控制台页导航带 api_key (页面从 ?api_key= 读 key)
    let key = console_key_query();
    navigate(window, &format!("{target}{key}"));
    Some((child, target, started))
}

/// 批次66: 控制台页 api_key 透传 — 启用鉴权后页面从 ?api_key= 读 key, 壳不带则全部 401。
/// 口径同 sidecar 注入: shell.json api_key 优先, 否则继承壳进程 ADR_TTS_API_KEY;
/// 取首个非空逗号项, URL 编码; 无 key 返回空串。
/// shortcut: 读当前设置而非 sidecar 启动时快照, 改 key 未重启引擎时热通道会带新 key; 需要时改为启动时记录。
pub(crate) fn console_key_query() -> String {
    let raw = shell_config()
        .api_key
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("ADR_TTS_API_KEY").ok())
        .unwrap_or_default();
    key_query(&raw)
}

/// 纯函数 (单测覆盖): 逗号分隔 key 串 → "&api_key=<首个非空项, 百分号编码>" 或空串。
pub(crate) fn key_query(raw: &str) -> String {
    let Some(k) = raw.split(',').map(str::trim).find(|k| !k.is_empty()) else {
        return String::new();
    };
    let mut enc = String::new();
    for b in k.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
            enc.push(b as char);
        } else {
            enc.push_str(&format!("%{b:02X}"));
        }
    }
    format!("&api_key={enc}")
}

/// 启动 sidecar 并探活 (不导航, 预热流程与 launch_console 共用)。
/// 成功后写 pid/base_url; 返回 (子进程, 启动时刻); 失败返回 None。
pub(crate) fn start_server(
    window: &Option<WebviewWindow>,
    state: &State<ServerState>,
) -> Option<(Child, Instant)> {
    // 批次76 (#21): 解析失败 = 硬失败 (不再回落 PATH 上的裸 "python")
    let (python, cwd, pythonpath) = match resolve_runtime() {
        Ok(v) => v,
        Err(e) => {
            proc_log("runtime", &format!("解释器解析失败: {e}"));
            update_status(window, &format!("启动失败: {e}"));
            return None;
        }
    };
    // 调用模式: 0.0.0.0 对局域网开放 API; 默认仅本机 (webview 与探活都走 127.0.0.1)
    let host = if state.expose.load(Ordering::Relaxed) { "0.0.0.0" } else { "127.0.0.1" };
    // 承载 TTS 服务: 优先绑定 9881 (NEKO GPT-SoVITS provider 默认端口, 零配置直连);
    // 9881 被占用则回退随机端口并提示。
    // 批次48: 设置中心 service_port > 0 时最高优先 (仍被占则回退默认逻辑)。
    let cfg = shell_config();
    let cfg_port = cfg.service_port.unwrap_or(0);
    let (port, using_preferred) = if cfg_port > 0 && probe_fixed_port(cfg_port, host) {
        (cfg_port, true)
    } else if probe_fixed_port(ADR_PREFERRED_PORT, host) {
        (ADR_PREFERRED_PORT, true)
    } else {
        let Some(p) = free_tcp_port() else {
            update_status(window, "端口分配失败, 正在重试…");
            return None;
        };
        (p, false)
    };
    if !using_preferred && cfg_port == 0 {
        update_status(window, &format!(
            "9881 被其他程序占用, 本次改用端口 {port} (NEKO 侧需手动填写该端口)"
        ));
    }
    let base = format!("http://127.0.0.1:{port}");

    let mut cmd = Command::new(&python);
    // pro / easy / call: 同一个 FastAPI 服务, 静态控制台页由 server 提供
    cmd.args([
        "-m", "adr.server",
        "--addr", host,
        "--port", &port.to_string(),
        // 批次99 (首启实测): 全新安装时 .terms_agreed_v1 尚不存在, 而壳
        // 在条款页之前就要起服务预热 — 不带此参数 server 会 exit 2,
        // 预热直接卡死 "服务启动失败"。条款由壳层 prewarm 门禁担保
        // (未同意不放行导航), server 侧不再重复拦截且不写标记。
        "--from-shell",
    ]);
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
    // 批次87 (SEC-04): 服务在 0.0.0.0 且未设 key 时默认拒绝启动。桌面壳的
    // 「对外服务」开关是用户在设置页的显式选择 (UI 已提示风险), 这里把这份
    // 知情显式传给服务端, 保留原有局域网能力。
    if state.expose.load(Ordering::Relaxed) {
        cmd.env("ADR_ALLOW_UNAUTHENTICATED", "1");
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
    // 批次87 (BUG-02): 句柄打不开也要继续, 由泵把管道抽干 (见 pump_server_log)
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .ok()
        .map(|file| {
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
            file
        });
    if let Some(out) = child.stdout.take() {
        let file = log_file.clone();
        std::thread::spawn(move || pump_server_log(out, file, "stdout"));
    }
    if let Some(err) = child.stderr.take() {
        let file = log_file.clone();
        std::thread::spawn(move || pump_server_log(err, file, "stderr"));
    }
    state.pid.store(child.id(), Ordering::Relaxed);
    *state.base_url.lock().unwrap() = Some(base.clone());
    // 不变式 (批次37): pid 写入 → Starting
    state.set_phase(Phase::Starting);

    let health_path = "/api/adr/v1/health";
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

/// 解释器来源 (纯函数决策的输出, 单测覆盖)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PySource {
    /// exe 旁 runtime/python/python.exe (打包态)
    Portable,
    /// ADR_DESKTOP_PYTHON 显式指定
    Explicit,
    /// 仓库 .venv/Scripts/python.exe
    Venv,
    /// 开发态扫 PATH 得到的绝对路径
    PathScan,
}

/// 纯函数: 决定用哪个解释器 (批次76 #21)。
///
/// 铁律: **打包态 (in_repo=false) 绝不走 PathScan**。Windows 解析裸
/// "python" 时搜索顺序含 exe 目录与 cwd, 攻击者在应用目录放一个同名
/// python.exe 即可以用户权限执行任意代码。打包态找不到随包解释器 = 报错,
/// 让用户重装, 而不是降级到不可信来源。
///
/// 参数:
/// - portable_found: exe 旁 runtime/python/python.exe 存在
/// - explicit: ADR_DESKTOP_PYTHON 指定值 (None/空 = 未指定)
/// - explicit_exists: 指定值指向的文件确实存在
/// - venv_found / path_found: 仓库 .venv / PATH 扫描命中
/// - in_repo: 是否处于仓库内 (开发态)
pub(crate) fn decide_python_source(
    portable_found: bool,
    explicit: Option<&str>,
    explicit_exists: bool,
    venv_found: bool,
    in_repo: bool,
    path_found: bool,
) -> Result<PySource, String> {
    if portable_found {
        return Ok(PySource::Portable);
    }
    if let Some(p) = explicit.map(str::trim).filter(|s| !s.is_empty()) {
        // 显式指定但文件不存在 → 硬失败 (不静默降级)
        if !explicit_exists {
            return Err(format!("ADR_DESKTOP_PYTHON 指向的解释器不存在: {p}"));
        }
        return Ok(PySource::Explicit);
    }
    if !in_repo {
        // 打包态且无随包解释器 → 硬失败, 无 PATH 兜底
        return Err(
            "未找到随包 Python 运行时 (runtime/python/python.exe), 请重新安装完整安装包".into(),
        );
    }
    if venv_found {
        return Ok(PySource::Venv);
    }
    if path_found {
        return Ok(PySource::PathScan);
    }
    Err("PATH 中未找到 python.exe, 请安装 Python 或设置 ADR_DESKTOP_PYTHON".into())
}

/// 解析 (python 解释器, 工作目录, 可选 PYTHONPATH)。
///
/// 批次76 (#21): 打包态**禁止 PATH 回落**。开发态兜底也返回 **绝对路径**,
/// 不用裸名字, 避免同样的搜索顺序问题。
fn resolve_runtime() -> Result<(PathBuf, PathBuf, Option<PathBuf>), String> {
    // 1) 便携模式: exe 旁的 runtime/ (阶段 0 打包产物)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(base) = exe.parent() {
            let runtime = base.join("runtime");
            let py = runtime.join("python").join("python.exe");
            if py.is_file() {
                return Ok((py, runtime, Some(PathBuf::from("."))));
            }
        }
    }

    // 开发态判定: exe 逐级向上找到 adr/cli.py 即仓库内 (cargo run 的 exe 在
    // target/debug/ 下, 向上 8 层可到仓库根)。**不看 cwd** —— 打包态 cwd 由
    // 启动者控制, 放一份 adr/cli.py 就能把壳诱进开发态再吃 PATH 回落。
    let root = find_repo_root();
    // 工作目录: 仓库根 (开发态) 或当前目录 (兜底, 仅当解释器显式指定时用)
    let cwd = root
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let explicit = std::env::var("ADR_DESKTOP_PYTHON").ok();
    let explicit_path = explicit.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from);
    let explicit_exists = explicit_path.as_ref().map(|p| p.is_file()).unwrap_or(false);
    let venv_found = root
        .as_ref()
        .map(|r| r.join(".venv").join("Scripts").join("python.exe").is_file())
        .unwrap_or(false);
    let path_hit = find_python_on_path();

    let source = decide_python_source(
        false,
        explicit.as_deref(),
        explicit_exists,
        venv_found,
        root.is_some(),
        path_hit.is_some(),
    )?;
    match source {
        PySource::Portable => unreachable!("便携分支已在上方返回"),
        PySource::Explicit => Ok((explicit_path.unwrap(), cwd, None)),
        PySource::Venv => {
            let root = root.unwrap();
            Ok((root.join(".venv").join("Scripts").join("python.exe"), root, None))
        }
        PySource::PathScan => {
            let root = root.unwrap();
            Ok((path_hit.unwrap(), root, None))
        }
    }
}

/// 从 exe 逐级向上探测仓库根 (以 adr/cli.py 为标记)。找不到 → None。
/// 只认 exe 侧: cwd 在打包态不可信 (见 resolve_runtime 注释)。
pub(crate) fn find_repo_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let mut base = exe.parent()?.to_path_buf();
    for _ in 0..8 {
        if base.join("adr").join("cli.py").is_file() {
            return Some(base);
        }
        if !base.pop() {
            break;
        }
    }
    None
}

/// 扫 PATH 找 python.exe, 返回**绝对路径**。Windows 上 ``Command::new("python")``
/// 会先搜 exe 目录/cwd, 这里只认 PATH 里的显式目录并回全路径, 避免劫持。
fn find_python_on_path() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        let cand = dir.join("python.exe");
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
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
