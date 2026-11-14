//! 壳单元测试 (M2 批次131 从 `lib.rs` 迁出): `cargo test --lib`。
//!
//! 只测纯函数 (环境变量映射 / 配置解析 / 解释器决策矩阵 / taskkill 分类),
//! 不启动 Tauri 运行时。

use crate::config::{env_overrides, ShellSettings};
use crate::lifecycle::{classify_taskkill_code, KillOutcome};
use crate::sidecar::{decide_python_source, key_query, PySource};
use crate::system::{autostart_cmd_value, brand_is_raptor_lake};

#[test]
fn env_overrides_maps_all_fields() {
    let c = ShellSettings {
        lan_expose: Some(true),          // 不进环境变量 (launch_console 参数语义)
        api_key: Some(" k1,k2 ".into()), // trim 保留 key 内部
        data_dir: Some(" D:/voices ".into()),
        hf_mirror: Some(true),
        download_rate: Some(2.5),        // 批次95 下载限速 (MB/s)
        autostart: Some(true),           // 独立注册表持久化, 不进环境变量
        default_profile: Some(" my_voice ".into()),
        service_port: Some(9900),        // 端口走 start_server 探测, 不进环境变量
        numba_cache_dir: Some(" F:/nc ".into()),
        cudagraph: Some(true),
        device: Some(" cpu ".into()),    // 批次49 低配预设
        torch_threads: Some(4),
        log_level: Some(" debug ".into()),   // 批次93 日志级别 (白名单归一)
        s2_num_workers: Some(2),           // 批次93 训练 DataLoader 进程数
        compat_cpu: Some(true),          // 批次57 13/14 代兼容模式
        // 批次62 声音安全字段 (默认开 = 不注入, 本用例不断言)
        gpu_mem_fraction: Some(0.75),    // 批次102 显存上限 75%
        strength: Some("strong".into()), // 批次102 强度档位 → 并发上限 4
        stream_head: Some(18),           // 批次102 流式首段字数
        split_max: Some(60),             // 批次102 长句细分上限
        autotune: Some(false),           // 批次102 关闭 Autopilot
        ..Default::default()
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
    assert_eq!(m.get("ADR_LOG_LEVEL").unwrap(), "DEBUG");
    assert_eq!(m.get("ADR_S2_NUM_WORKERS").unwrap(), "2");
    assert_eq!(m.get("ADR_DOWNLOAD_RATE_MBPS").unwrap(), "2.5");
    // 兼容模式: 全链限 ISA 到 AVX2 (oneDNN/MKL/numba)
    assert_eq!(m.get("ONEDNN_MAX_CPU_ISA").unwrap(), "AVX2");
    assert_eq!(m.get("MKL_ENABLE_INSTRUCTIONS").unwrap(), "AVX2");
    assert_eq!(m.get("NUMBA_CPU_NAME").unwrap(), "generic");
    // 批次102: 首启向导补全项
    assert_eq!(m.get("ADR_GPU_MEM_FRACTION").unwrap(), "0.75");
    assert_eq!(m.get("ADR_TTS_STREAM_HEAD").unwrap(), "18");
    assert_eq!(m.get("ADR_TTS_SPLIT_MAX").unwrap(), "60");
    assert_eq!(m.get("ADR_RUNTIME_AUTOTUNE").unwrap(), "0");
    assert_eq!(m.get("ADR_SCHED_MAX_CONCURRENCY").unwrap(), "4");
    assert_eq!(m.len(), 19);
}

#[test]
fn env_overrides_batch102_strength_and_limits() {
    // 强度档位 low = 串行 + 关合批 (最省资源); strong = 并发 4 (合批走默认开)
    let m: std::collections::HashMap<String, String> =
        env_overrides(&ShellSettings { strength: Some("low".into()), ..Default::default() }, false)
            .into_iter()
            .collect();
    assert_eq!(m.get("ADR_SCHED_MAX_CONCURRENCY").unwrap(), "1");
    assert_eq!(m.get("ADR_T2S_COALESCE").unwrap(), "0");
    let m: std::collections::HashMap<String, String> =
        env_overrides(&ShellSettings { strength: Some("strong".into()), ..Default::default() }, false)
            .into_iter()
            .collect();
    assert_eq!(m.get("ADR_SCHED_MAX_CONCURRENCY").unwrap(), "4");
    assert!(m.get("ADR_T2S_COALESCE").is_none());
    // auto / 空 / 未知档位 = 后端默认, 零注入
    for s in ["auto", "", "bogus"] {
        let c = ShellSettings { strength: Some(s.into()), ..Default::default() };
        assert!(env_overrides(&c, false).is_empty());
    }
    // 显存上限越界 / 流式参数为 0 → 不注入 (后端默认)
    let c = ShellSettings {
        gpu_mem_fraction: Some(0.05),
        stream_head: Some(0),
        split_max: Some(0),
        ..Default::default()
    };
    assert!(env_overrides(&c, false).is_empty());
    // Autopilot 显式开 = 不注入 (后端默认开)
    let c = ShellSettings { autotune: Some(true), ..Default::default() };
    assert!(env_overrides(&c, false).is_empty());
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
        log_level: Some("   ".into()),
        ..Default::default()
    };
    assert!(env_overrides(&c, false).is_empty());
    // 非法日志级别 → 白名单丢弃
    let c = ShellSettings { log_level: Some("VERBOSE".into()), ..Default::default() };
    assert!(env_overrides(&c, false).is_empty());
    // 限速 0 / 越界 → 不注入 (0.1~1024 之外)
    for bad in [0.0_f32, 0.05, 5000.0] {
        let c = ShellSettings { download_rate: Some(bad), ..Default::default() };
        assert!(env_overrides(&c, false).is_empty());
    }
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
fn key_query_first_nonempty_encoded() {
    // 批次66: 控制台导航 api_key 透传
    assert_eq!(key_query(""), "");
    assert_eq!(key_query(" , "), "");
    assert_eq!(key_query(" ,k1, k2"), "&api_key=k1");
    assert_eq!(key_query("a b&c=d/é"), "&api_key=a%20b%26c%3Dd%2F%C3%A9");
}

#[test]
fn autostart_value_quotes_path() {
    assert_eq!(
        autostart_cmd_value(std::path::Path::new("C:/Program Files/ADR/adr-desktop.exe")),
        r#""C:/Program Files/ADR/adr-desktop.exe""#
    );
}

/// 批次72: taskkill 退出码三分类 — 128 (进程不存在) 是竞态不是失败。
#[test]
fn taskkill_exit_code_classification() {
    assert!(matches!(classify_taskkill_code(Some(0)), KillOutcome::Killed));
    assert!(matches!(classify_taskkill_code(Some(128)), KillOutcome::AlreadyGone));
    assert!(matches!(classify_taskkill_code(Some(1)), KillOutcome::Failed));
    assert!(matches!(classify_taskkill_code(None), KillOutcome::Failed));
}

/// 批次72: 互斥体名单源常量 — 与 installer.iss AppMutex / __main__.py 一致
/// (跨语言无法编译期共享, 靠名字常量 + Python 侧防漂移测试双重锁定)。
#[cfg(windows)]
#[test]
fn app_mutex_name_matches_spec() {
    assert_eq!(crate::system::APP_MUTEX_NAME, "ADR-Studio-Running");
}

/// 批次76 (#21): 打包态禁止 PATH 回落 —— 决策矩阵全覆盖。
/// 参数序: (portable_found, explicit, explicit_exists, venv_found, in_repo, path_found)
#[test]
fn decide_python_source_blocks_path_fallback_when_packaged() {
    // 打包态 + 有随包解释器 → Portable
    assert_eq!(
        decide_python_source(true, None, false, false, false, true).unwrap(),
        PySource::Portable
    );
    // 打包态 + 无随包解释器 + PATH 里有 python → **必须报错**, 不得降级
    let err = decide_python_source(false, None, false, false, false, true).unwrap_err();
    assert!(err.contains("runtime/python/python.exe"), "错误信息应指向缺失的随包解释器: {err}");
    // 打包态 + PATH 也没有 → 同样报错
    assert!(decide_python_source(false, None, false, false, false, false).is_err());
    // 打包态 + venv 恰好存在 (仓库被误放到安装目录) → 依然不得走 venv
    assert!(decide_python_source(false, None, false, true, false, true).is_err());
}

/// 批次76 (#21): 开发态兜底顺序 — venv 优先于 PATH, 且显式指定最高。
#[test]
fn decide_python_source_dev_mode_order() {
    // 开发态: venv 优先
    assert_eq!(
        decide_python_source(false, None, false, true, true, true).unwrap(),
        PySource::Venv
    );
    // 开发态: 无 venv → PATH 扫描兜底
    assert_eq!(
        decide_python_source(false, None, false, false, true, true).unwrap(),
        PySource::PathScan
    );
    // 开发态: 全都没有 → 报错
    assert!(decide_python_source(false, None, false, false, true, false).is_err());
    // 显式指定 (存在) → 最高优先, 打包态也认 (运维显式覆盖)
    assert_eq!(
        decide_python_source(false, Some("D:/pyhon/python.exe"), true, false, false, true).unwrap(),
        PySource::Explicit
    );
    // 显式指定但文件不存在 → 硬失败, 不静默降级
    let err = decide_python_source(false, Some("D:/nope/python.exe"), false, false, false, true)
        .unwrap_err();
    assert!(err.contains("ADR_DESKTOP_PYTHON"), "{err}");
}
