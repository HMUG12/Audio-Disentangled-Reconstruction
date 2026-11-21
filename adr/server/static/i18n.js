/* i18n.js — 批次54: 控制台页面中/英切换运行时 (easy/pro/call/index 共用)。
 *
 * 用法 (页面主 script 之前):
 *   <script src="/i18n.js"></script>
 *   I.init("easy", { onSwitch: () => renderVoices() });
 *
 * 页面约定:
 *   data-i18n="easy.xxx"       → 覆盖 textContent (纯文本叶子)
 *   data-i18n-html="easy.xxx"  → 覆盖 innerHTML (字典含 <b> 等富文本)
 *   data-i18n-ph="easy.xxx"    → 覆盖 placeholder
 *   data-i18n-title="easy.xxx" → 覆盖 title 属性
 *   JS 内: I.t("easy.xxx", {n: 3})  — 字典串支持 {var} 占位
 *
 * 字典 (I18N_DICT, 构建时合并): { key: ["中文", "English"], ... }
 *   — key 按页面命名空间 (easy./pro./call./common.)，缺 key 时回落中文串, 再缺回落 key 本身。
 *
 * 语言判定: localStorage("adr_lang") > 浏览器语言 (zh* → 中文, 其余 → English)。
 * 切换按钮自动注入 header 右端, 样式与页面 "返回" 按钮一致; 切换后
 * 先 apply() 静态节点, 再跑页面 onSwitch 重渲染动态内容, 最后再 apply()
 * 补扫动态生成的 data-i18n 节点。 */
"use strict";
/* 跨页公共键 (common.*): 主题切换按钮文案 — 与 theme.js LABELS 同键同文案 */
const I18N_COMMON = {
  "common.theme_default": ["主题：默认", "Theme: Default"],
  "common.theme_classic": ["主题：经典", "Theme: Classic"],
  "common.theme_title": ["切换配色主题（默认 / 经典）", "Switch color theme (Default / Classic)"],
};
const I = (() => {
  let lang = "zh", page = "", dict = {};
  const changeCbs = [];

  function store() { try { return localStorage.getItem("adr_lang"); } catch { return null; } }
  function detect() {
    const s = store();
    if (s === "zh" || s === "en") return s;
    return (navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en";
  }
  function fmt(s, vars) {
    if (!vars) return s;
    return s.replace(/\{(\w+)\}/g, (m, k) => (vars[k] !== undefined ? String(vars[k]) : m));
  }
  function t(key, vars) {
    const e = dict[key];
    const s = !e ? (typeof e === "undefined" ? key : "") : (lang === "en" ? e[1] : e[0]);
    return fmt(s || (e ? e[0] : key), vars);
  }
  function apply(root) {
    const r = root || document;
    r.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.getAttribute("data-i18n")); });
    r.querySelectorAll("[data-i18n-html]").forEach(el => { el.innerHTML = t(el.getAttribute("data-i18n-html")); });
    r.querySelectorAll("[data-i18n-ph]").forEach(el => { el.placeholder = t(el.getAttribute("data-i18n-ph")); });
    r.querySelectorAll("[data-i18n-title]").forEach(el => { el.title = t(el.getAttribute("data-i18n-title")); });
    // 输入框默认值: 仅在用户未改动过时跟随语言 (用户输入不丢失)
    r.querySelectorAll("[data-i18n-value]").forEach(el => {
      if (el.dataset.i18nDefault === undefined) el.dataset.i18nDefault = el.value;
      if (el.value === el.dataset.i18nDefault) {
        el.value = t(el.getAttribute("data-i18n-value"));
        el.dataset.i18nDefault = el.value;
      }
    });
    document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
    const tt = dict[page + ".title"];
    if (tt) document.title = lang === "en" ? tt[1] : tt[0];
  }
  function setLang(l) {
    lang = l;
    try { localStorage.setItem("adr_lang", l); } catch {}
    const b = document.getElementById("langBtn");
    if (b) b.textContent = l === "zh" ? "EN" : "中文";
    if (window.adrTheme) window.adrTheme.relabel();
    apply();
    changeCbs.forEach(f => { try { f(); } catch {} });
    // app-shell.js 的顶栏/文档抽屉是 JS 生成并缓存的, data-i18n 覆盖不到;
    // 用文档级事件通知外壳重建 (无外壳页面无监听, 零副作用)。
    try { document.dispatchEvent(new CustomEvent("adr:lang", { detail: { lang: l } })); } catch {}
    apply(); // 补扫: onSwitch 重建的动态节点
  }
  function init(pg, opts) {
    page = pg;
    dict = Object.assign({}, I18N_COMMON, typeof I18N_DICT === "object" ? I18N_DICT : {});
    lang = detect();
    // P0: 语言切换只保留在「全局设置」页, 控制台顶栏不再注入切换按钮
    apply();
    if (opts && opts.onSwitch) changeCbs.push(opts.onSwitch);
  }
  return { init, t, apply, setLang, get lang() { return lang; } };
})();

/* 批次106 (审计 R5-FE-01): shell_origin 查询参数归一 — 只放行 tauri 壳地址与
 * 本机回环地址 (dev 壳走 http://127.0.0.1:<devport>), 其余 (javascript:/data:/
 * 外部站点/畸形串) 一律返回 "", 由调用方回落默认分支, 杜绝脚本执行与开放重定向。 */
function adrSafeShellOrigin(raw) {
  if (!raw) return "";
  let u;
  try { u = new URL(String(raw)); } catch (_) { return ""; }
  const proto = u.protocol, host = (u.hostname || "").toLowerCase();
  if (proto === "tauri:") return host === "localhost" ? "tauri://localhost" : "";
  if (proto !== "http:" && proto !== "https:") return "";
  if (host === "tauri.localhost") return u.origin;
  if (host === "localhost" || host === "127.0.0.1" || host === "[::1]") return u.origin;
  return u.origin === location.origin ? u.origin : "";
}

/* 批次110 (审计 R5-FE-02): 后端错误体归一 — FastAPI 422 的 detail 是
 * errors() 数组, 直接 String() 会得到 "[object Object]"。优先取字符串型
 * message (app.py 422 处理器补的), 其次字符串型 detail, 对象/数组才序列化,
 * 都没有就回落调用方给的兜底串 (默认 "HTTP <status>")。 */
function adrErrMsg(body, status, fallback) {
  const fb = fallback || ("HTTP " + (status || 0));
  const str = (v) => (typeof v === "string" && v.trim() ? v.trim() : "");
  if (typeof body === "string") return str(body) || fb;
  if (!body || typeof body !== "object") return fb;
  const m = str(body.message) || str(body.detail) || str(body.error && body.error.message);
  if (m) return m;
  const d = body.detail !== undefined ? body.detail : body.error;
  // FastAPI 422: detail = [{loc:[...,"text"], msg:"field required"}, ...]
  if (Array.isArray(d)) {
    const parts = d.map(e => {
      if (typeof e === "string") return e.trim();
      if (!e || typeof e !== "object") return "";
      const loc = Array.isArray(e.loc) ? e.loc.filter(x => x !== "body" && x !== "query").join(".") : "";
      const msg = typeof e.msg === "string" ? e.msg : "";
      return loc && msg ? loc + ": " + msg : msg;
    }).filter(Boolean);
    if (parts.length) return parts.join("；");
  }
  if (d && typeof d === "object") { try { return JSON.stringify(d); } catch (_) { return fb; } }
  return fb;
}

/* 批次113 (审计 R5-FE-06): 三页共用的 HTML 转义 — 模板字符串拼 innerHTML 时
 * 用这一套 (& < > " '); easy 原先只转 "<", 属性/引号型内容仍可注入。
 * 纯文本场景优先 textContent, 不必走这里。 */
function adrEsc(s) {
  return String(s === undefined || s === null ? "" : s).replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
