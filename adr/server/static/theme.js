/* theme.js — 控制台/桌面壳共用 UI 主题层。
 *
 * 用法: <head> 内、<style> 之前同步引入 (防闪烁):
 *   服务端页: <script src="/theme.js"></script>
 *   壳内页:   <script src="theme.js"></script>   (desktop/src/theme.js 为本文件逐字副本)
 *
 * 主题: <html data-theme="default|classic">
 *   default — 浅蓝手绘风 (与新图标一致), 缺省值
 *   classic — 各页原有配色 (各页 :root 变量取原值)
 * 存储: localStorage("adr.theme"); 非法/缺失 → "default"。
 * API:  window.adrTheme.get() / set(name) / toggle() / mountSwitch(host) / relabel() / onChange(fn)
 *       切换时派发 window 事件 "adr-theme-change" (canvas 等 JS 绘制按需重绘)。
 * 切换按钮文案: 有 i18n.js 时走 I.t("common.theme_*"), 否则用下方同格式兜底字典。 */
"use strict";
(function () {
  const KEY = "adr.theme", NAMES = ["default", "classic"], FALLBACK = "default";
  // 与 i18n.js I18N_COMMON 同键同文案 (壳内页无 i18n.js 时兜底)
  const LABELS = {
    "common.theme_default": ["主题：默认", "Theme: Default"],
    "common.theme_classic": ["主题：经典", "Theme: Classic"],
    "common.theme_title": ["切换配色主题（默认 / 经典）", "Switch color theme (Default / Classic)"],
  };
  const root = document.documentElement;

  function read() {
    try {
      const s = localStorage.getItem(KEY);
      return NAMES.indexOf(s) >= 0 ? s : FALLBACK;
    } catch (_) { return FALLBACK; }
  }
  let cur = read();
  root.setAttribute("data-theme", cur);

  // 切换按钮外观变量: classic 下不定义 → 走 var() 兜底 = 原语言按钮样式
  const st = document.createElement("style");
  st.textContent =
    '[data-theme="default"]{--adr-pill-bg:#FFFFFF;--adr-pill-txt:#2B6CB0;' +
    '--adr-pill-line:#CFE3FA;--adr-pill-hover:#5EAAF7;--adr-pill-hover-txt:#1A202C}';
  (document.head || root).appendChild(st);

  function lang() {
    if (typeof I === "object" && I && I.lang) return I.lang;
    try {
      const s = localStorage.getItem("adr_lang");
      if (s === "zh" || s === "en") return s;
    } catch (_) {}
    // 无 i18n 的壳内页: 跟随页面自身 <html lang> (壳页为中文)
    return (root.lang || "zh").toLowerCase().indexOf("en") === 0 ? "en" : "zh";
  }
  function t(k) {
    if (typeof I === "object" && I && typeof I.t === "function") {
      const s = I.t(k);
      if (s && s !== k) return s;
    }
    const e = LABELS[k];
    return e ? (lang() === "en" ? e[1] : e[0]) : k;
  }
  function relabel() {
    const b = document.getElementById("themeBtn");
    if (!b) return;
    b.textContent = t("common.theme_" + cur);
    b.title = t("common.theme_title");
    b.setAttribute("aria-label", b.title);
  }
  function set(name) {
    if (NAMES.indexOf(name) < 0) return;
    cur = name;
    root.setAttribute("data-theme", name);
    try { localStorage.setItem(KEY, name); } catch (_) {}
    relabel();
    try { window.dispatchEvent(new CustomEvent("adr-theme-change", { detail: { theme: name } })); } catch (_) {}
  }
  // 与 i18n.js 语言按钮同款胶囊样式 (classic 取原值)
  function pill(b) {
    b.style.cssText = "background:var(--adr-pill-bg,#fff);color:var(--adr-pill-txt,#6b7280);"
      + "border:1.5px solid var(--adr-pill-line,#e5e9f2);border-radius:999px;"
      + "padding:8px 16px;font-size:13px;cursor:pointer;font-family:inherit;font-weight:600;"
      + "margin-left:10px;flex:none;transition:.15s";
    b.onmouseenter = () => {
      b.style.borderColor = "var(--adr-pill-hover,#10b981)";
      b.style.color = "var(--adr-pill-hover-txt,#1f2937)";
    };
    b.onmouseleave = () => {
      b.style.borderColor = "var(--adr-pill-line,#e5e9f2)";
      b.style.color = "var(--adr-pill-txt,#6b7280)";
    };
  }
  // 无 header 的页面: 右上角固定容器 (语言按钮也挂这里)
  function fixedHost() {
    let h = document.getElementById("adrSwitches");
    if (!h) {
      h = document.createElement("div");
      h.id = "adrSwitches";
      h.style.cssText = "position:fixed;top:16px;right:18px;z-index:98;display:flex;align-items:center";
      document.body.appendChild(h);
    }
    return h;
  }
  function mountSwitch(host) {
    let b = document.getElementById("themeBtn");
    if (!b) {
      b = document.createElement("button");
      b.id = "themeBtn";
      b.type = "button";
      pill(b);
      b.onclick = () => set(cur === "default" ? "classic" : "default");
    }
    (host || fixedHost()).appendChild(b);
    relabel();
    return b;
  }

  window.adrTheme = {
    get: () => cur,
    set: set,
    toggle: () => set(cur === "default" ? "classic" : "default"),
    mountSwitch: mountSwitch,
    fixedHost: fixedHost,
    pill: pill,
    relabel: relabel,
    onChange: (fn) => window.addEventListener("adr-theme-change", () => { try { fn(cur); } catch (_) {} }),
  };
})();
