// 发布版隐藏控制台窗口 (GUI 子系统)
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    adr_desktop_lib::run();
}
