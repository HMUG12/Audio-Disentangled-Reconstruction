//! 设置中心配置 (M2 拆分, 批次130): shell.json 读写 + 环境变量映射 + 数据目录推导。
//!
//! 从 `lib.rs` 原样迁出, **行为与解析顺序不变**。对外只暴露 `pub(crate)` 接口:
//! - `ShellSettings` / `shell_settings_path` / `reload_shell_config` / `shell_config`
//! - `env_overrides`（纯函数，单测覆盖）
//! - `adr_data_dir`（与 `adr/core/config.py` 同序推导）
//!
//! 契约文档: docs/architecture/contracts.md 第 5、6 节。

use std::path::PathBuf;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// 设置中心 (批次48): shell.json 配置 + sidecar 环境变量注入
// ---------------------------------------------------------------------------

/// 设置中心配置 (launcher.html 设置模态框读写, sidecar 启动时注入)。
/// 全字段 Option: 旧版本文件缺字段/坏 JSON 一律回退默认, 不炸壳。
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub(crate) struct ShellSettings {
    // --- 常用层 ---
    /// 局域网开关默认值 (launcher 卡片勾选框可本次覆盖)
    pub(crate) lan_expose: Option<bool>,
    /// ADR_TTS_API_KEY — 逗号分隔多 key, 空串 = 鉴权关闭
    pub(crate) api_key: Option<String>,
    /// ADR_DATA_DIR 覆盖 (声音档案存放位置); 空串 = 用推导默认
    pub(crate) data_dir: Option<String>,
    /// true → HF_ENDPOINT=https://hf-mirror.com (国内模型下载镜像)
    pub(crate) hf_mirror: Option<bool>,
    /// ADR_DOWNLOAD_RATE_MBPS — 预训练资源下载限速 (MB/s, 批次95);
    /// 0/None = 不限速
    pub(crate) download_rate: Option<f32>,
    /// 开机自启 (HKCU Run 注册表, 与本文件无关的独立持久化)
    pub(crate) autostart: Option<bool>,
    /// ADR_TTS_DEFAULT_PROFILE — /tts 缺省档案
    pub(crate) default_profile: Option<String>,
    // --- 高级抽屉 ---
    /// 服务端口覆盖; 0/None = 默认逻辑 (9881 探测 → 随机回退)
    pub(crate) service_port: Option<u16>,
    /// NUMBA_CACHE_DIR — numba JIT 缓存重定向 (批次47: 沙箱/杀软禁写
    /// site-packages 时预热会卡死, 重定向到可写目录即解)
    pub(crate) numba_cache_dir: Option<String>,
    /// ADR_T2S_CUDAGRAPH=1 — t2s CUDA Graph 加速 (音质有损, 默认关)
    pub(crate) cudagraph: Option<bool>,
    /// ADR_TTS_DEVICE — 推理设备走向 (批次49 低配预设): auto/cuda/cpu;
    /// auto/空 = 不注入 (后端按可用性探测)
    pub(crate) device: Option<String>,
    /// ADR_TORCH_THREADS — torch CPU 线程上限 (批次49 低配预设);
    /// 0/None = 不注入 (PyTorch 自按物理核)
    pub(crate) torch_threads: Option<u16>,
    /// ADR_LOG_LEVEL — 服务/训练日志级别 (批次93): DEBUG/INFO/WARNING/ERROR;
    /// 空/None = 不注入 (后端默认 INFO)
    pub(crate) log_level: Option<String>,
    /// ADR_S2_NUM_WORKERS — 训练 DataLoader 进程数 (批次93);
    /// 0/None = 不注入 (主进程加载, Windows 最稳)
    pub(crate) s2_num_workers: Option<u16>,
    /// 兼容模式 (批次57): 13/14 代 Intel (Raptor Lake) 计算密集段闪退缓解 —
    /// 全链指令集限到 AVX2 (oneDNN/MKL/numba), 绕开 AVX-512 电压缺陷路径。
    /// None = 壳自动检测 (Raptor Lake 自动开); Some = 用户手动覆盖
    pub(crate) compat_cpu: Option<bool>,
    // --- 声音安全 (批次62) ---
    /// ADR_SAFETY_AUDIT — 审计日志 (JSONL, data_dir/logs/audit.log); 默认开
    pub(crate) safety_audit: Option<bool>,
    /// ADR_WAV_METADATA — 分发 WAV 尾部 LIST/INFO 标注 AI 生成; 默认开
    pub(crate) wav_metadata: Option<bool>,
    // --- 首启向导补全 (批次102, 会议纪要 ①-⑯ 可落地子集) ---
    /// ADR_GPU_MEM_FRACTION — 单进程显存上限占比 (0.1~1.0); None = 不限制
    pub(crate) gpu_mem_fraction: Option<f32>,
    /// 强度档位: auto/low/mid/balanced/strong/max — 映射调度并发上限
    /// (ADR_SCHED_MAX_CONCURRENCY) 与跨请求合批 (ADR_T2S_COALESCE), 不单独落环境变量
    pub(crate) strength: Option<String>,
    /// ADR_TTS_STREAM_HEAD — 流式首段最大字数; 0/None = 后端默认 (14)
    pub(crate) stream_head: Option<u16>,
    /// ADR_TTS_SPLIT_MAX — 无句读长段细分字数; 0/None = 后端默认 (40)
    pub(crate) split_max: Option<u16>,
    /// ADR_RUNTIME_AUTOTUNE=false — 关闭 Autopilot 自动微调; None/true = 默认开
    pub(crate) autotune: Option<bool>,
}

static SHELL_CFG: std::sync::OnceLock<Mutex<ShellSettings>> = std::sync::OnceLock::new();

pub(crate) fn shell_cell() -> &'static Mutex<ShellSettings> {
    SHELL_CFG.get_or_init(|| Mutex::new(ShellSettings::default()))
}

/// shell.json 落点: %LOCALAPPDATA%\ADR\shell.json (LOCALAPPDATA 缺失 → 数据目录旁)。
pub(crate) fn shell_settings_path() -> PathBuf {
    if let Ok(la) = std::env::var("LOCALAPPDATA") {
        if !la.trim().is_empty() {
            return PathBuf::from(la).join(r"ADR\shell.json");
        }
    }
    adr_data_dir().unwrap_or_else(std::env::temp_dir).join("shell.json")
}

/// 从盘加载配置并刷新进程内缓存 (启动时 / 保存后调用; 平时读缓存避免
/// 每条 proc_log 都碰文件系统)。坏 JSON / 读取失败 → 默认值。
pub(crate) fn reload_shell_config() -> ShellSettings {
    let cfg = std::fs::read_to_string(shell_settings_path())
        .ok()
        .and_then(|s| serde_json::from_str::<ShellSettings>(&s).ok())
        .unwrap_or_default();
    *shell_cell().lock().unwrap() = cfg.clone();
    cfg
}

/// 进程内缓存读取 (未加载过 → 读盘)。
pub(crate) fn shell_config() -> ShellSettings {
    shell_cell().lock().unwrap().clone()
}

/// shell.json → sidecar 环境变量映射 (纯函数, 单测覆盖)。
/// 只影响子进程, 不写壳进程自身环境; 空串字段一律跳过 (= 未配置)。
/// auto_compat: CPU 自动检测结果 (批次57) — compat_cpu=None 时按它定夺, Some 优先。
pub(crate) fn env_overrides(c: &ShellSettings, auto_compat: bool) -> Vec<(String, String)> {
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
    // 批次93: 日志级别 (白名单, 防止任意值进 logging.setLevel 静默回落);
    // 训练 DataLoader 进程数 (>0 才注入)
    if let Some(l) = c.log_level.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        let up = l.to_ascii_uppercase();
        if matches!(up.as_str(), "DEBUG" | "INFO" | "WARNING" | "ERROR") {
            v.push(("ADR_LOG_LEVEL".into(), up));
        }
    }
    if let Some(n) = c.s2_num_workers.filter(|n| *n > 0) {
        v.push(("ADR_S2_NUM_WORKERS".into(), n.to_string()));
    }
    // 批次95: 预训练资源下载限速 (MB/s), 0.1~1024 之外视为未配置
    if let Some(r) = c.download_rate.filter(|r| *r >= 0.1 && *r <= 1024.0) {
        v.push(("ADR_DOWNLOAD_RATE_MBPS".into(), format!("{r}")));
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
    // 批次102 首启向导补全 (会议纪要 L163-180 ①-⑯ 中可落地的部分):
    // 资源上限 / 流式参数 / Autopilot 开关 / 强度档位。
    // 显存上限 (⑤⑥): 0.1~1.0 之外视为未配置 (后端不设上限)。
    if let Some(f) = c.gpu_mem_fraction.filter(|f| *f >= 0.1 && *f <= 1.0) {
        v.push(("ADR_GPU_MEM_FRACTION".into(), format!("{f}")));
    }
    // 流式首段字数 (⑪) / 长句细分上限 (⑪): 0 = 后端默认, 不注入。
    if let Some(n) = c.stream_head.filter(|n| *n > 0) {
        v.push(("ADR_TTS_STREAM_HEAD".into(), n.to_string()));
    }
    if let Some(n) = c.split_max.filter(|n| *n > 0) {
        v.push(("ADR_TTS_SPLIT_MAX".into(), n.to_string()));
    }
    // Autopilot 自动标定 (⑯): 默认开, 仅显式关闭时注入 =0。
    if c.autotune == Some(false) {
        v.push(("ADR_RUNTIME_AUTOTUNE".into(), "0".into()));
    }
    // 强度档位 (⑦): 映射调度并发上限 + 跨请求合批开关, 不是伪开关。
    // low = 串行且不合批 (最省资源); mid/balanced/strong/max = 逐级放宽并发,
    // 合批沿用后端默认 (开)。auto/未知 = 零注入 (后端默认 N=1 + 合批开)。
    match c.strength.as_deref().map(str::trim).unwrap_or("") {
        "low" => {
            v.push(("ADR_SCHED_MAX_CONCURRENCY".into(), "1".into()));
            v.push(("ADR_T2S_COALESCE".into(), "0".into()));
        }
        "mid" => v.push(("ADR_SCHED_MAX_CONCURRENCY".into(), "2".into())),
        "balanced" => v.push(("ADR_SCHED_MAX_CONCURRENCY".into(), "3".into())),
        "strong" => v.push(("ADR_SCHED_MAX_CONCURRENCY".into(), "4".into())),
        "max" => v.push(("ADR_SCHED_MAX_CONCURRENCY".into(), "6".into())),
        _ => {}
    }
    v
}

/// 与 adr/core/config.py adr_data_dir() 同序推导 (批次27 复审 ISSUE-2 对齐):
/// shell.json data_dir (批次48 设置中心, 最高优先) > ADR_DATA_DIR env >
/// F:/ADR_data (legacy, 存在即沿用) > %LOCALAPPDATA%\ADR\data。
/// 返回 None 表示交由 Python 侧默认 (~/.adr/data), 不注入子进程。
/// 空串环境变量视为未设置 (批次27 复审 ISSUE-4: PathBuf("") 会落到进程 CWD)。
pub(crate) fn adr_data_dir() -> Option<PathBuf> {
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
