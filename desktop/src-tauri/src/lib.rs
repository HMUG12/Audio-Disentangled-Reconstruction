//! ADR Studio 桌面壳 (Tauri 2)。
//!
//! 职责:
//! 1. 启动即预热: 窗口先展示 prewarm.html, setup 后台拉起 FastAPI 服务并轮询
//!    引擎预热阶段 (torch 导入 → 权重加载 → kernel 预热), 就绪后才放行进 launcher
//! 2. 启动器: launcher.html 三种控制台 (pro / easy / call)
//!    - 复用常驻 FastAPI 服务秒进 (引擎已预热, 不再杀进程)
//! 3. 健康探活: 单一探 ``GET /api/adr/v1/health``
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
//! 2. 环境变量 ``ADR_DESKTOP_PYTHON`` (开发调试指定解释器, 仅开发态)
//! 3. 仓库虚拟环境 ``<仓库根>/.venv/Scripts/python.exe`` (仅开发态)
//! 4. 开发态兜底: 扫 PATH 取 python.exe **绝对路径** (不用裸 "python" 名字)
//!
//! 批次76 (#21): 打包态 (``runtime/`` 目录存在) 找不到解释器 = 直接报错,
//! 绝不回落 PATH —— Windows 解析裸 "python" 会先搜 exe 目录与 cwd, 同名
//! python.exe 被替换即等于任意代码执行。开发态判定只看 exe 向上探测到的
//! 仓库根, 不看 cwd (cwd 在打包态由启动者控制, 可被投放 adr/cli.py 伪造)。
//!
//! 模块划分 (M2 批次130/131, 本文件只剩装配):
//! - [`config`]   设置中心 shell.json 读写 / 环境变量映射 / 数据目录推导
//! - [`system`]   单实例互斥体 / 开机自启 / CPU 兼容检测 / 条款与向导标记
//! - [`sidecar`]  Python 解释器解析 / 端口分配 / 拉起服务 / 服务日志泵
//! - [`lifecycle`] Phase 状态机 / 预热轮询 / 守护重启 / 孤儿收编
//! - [`windows`]  主窗口 (下载事件) / 托盘菜单 / 页面导航
//! - [`commands`] 30 条 Tauri 命令 (参数解包与转发)
//! - [`account`]  本地账号 (本机登录)
//! - [`update`]   客户端更新链 (清单检查 / 下载校验 / 静默安装)

mod update;

mod account;

mod config;
mod system;
mod sidecar;
mod lifecycle;
mod windows;
mod commands;

#[cfg(test)]
mod tests;

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};

use tauri::Manager;
use tauri_plugin_notification::NotificationExt;

/// 首次最小化到任务栏时提示一次 (引擎仍在后台, 后续最小化不再打扰)
static MIN_NOTIFIED: AtomicBool = AtomicBool::new(false);

use config::reload_shell_config;
use lifecycle::{prewarm_flow, shutdown_server, Phase, ServerState};

#[cfg(windows)]
use system::create_app_mutex;

pub fn run() {
    // 批次72: 尽早持锁 — 安装器/卸载器 (AppMutex) 据此提示先关闭程序
    #[cfg(windows)]
    create_app_mutex();
    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
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

            windows::create_main_window(app)?;
            windows::create_tray(app)?;

            // 启动即预热: 后台拉服务 + 轮询引擎阶段, 就绪后自动导航进 launcher
            let h = app.handle().clone();
            std::thread::spawn(move || prewarm_flow(h));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::launch_console,
            commands::enter_console,
            commands::on_launcher_ready,
            commands::engine_status,
            commands::terms_agreed,
            commands::agree_terms,
            commands::reject_terms,
            commands::setup_done,
            commands::complete_setup,
            commands::env_check,
            commands::factory_reset,
            account::account_info,
            account::account_set,
            account::account_login,
            account::account_logout,
            account::account_reset,
            account::account_verify,
            commands::enter_launcher,
            commands::check_models,
            commands::skip_prewarm,
            commands::retry_prewarm,
            commands::back_to_launcher,
            commands::load_shell_settings,
            commands::save_shell_settings,
            commands::lan_info,
            commands::app_paths,
            commands::autostart_get,
            commands::autostart_set,
            commands::open_path,
            commands::cpu_compat_auto,
            update::check_update,
            update::download_update,
            update::install_update,
            update::package_hint
        ])
        .on_window_event(|window, event| {
            match event {
                // 关窗 = 隐藏到托盘; 真正退出走托盘菜单 (保证 sidecar 被清理)
                tauri::WindowEvent::CloseRequested { api, .. } => {
                    let _ = window.hide();
                    api.prevent_close();
                }
                // 首次最小化提示一次: 引擎仍在后台, 从托盘恢复
                tauri::WindowEvent::Resized(_) => {
                    let minimized = window.is_minimized().unwrap_or(false);
                    if minimized && !MIN_NOTIFIED.swap(true, Ordering::Relaxed) {
                        let _ = window
                            .app_handle()
                            .notification()
                            .builder()
                            .title("ADR Studio 仍在后台运行")
                            .body("合成引擎不会停止，可从托盘图标恢复窗口。")
                            .show();
                    }
                }
                _ => {}
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
