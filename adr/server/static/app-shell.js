/* app-shell.js — 控制台/入口页共用外壳: 顶栏、全局导航、文档抽屉。
 *
 * 设计约束:
 *  - 只增强 DOM, 不改各页既有 id / fetch / 事件 (功能零回归)。
 *  - 文档接口同源 (/api/adr/v1/docs), 符合各页 CSP connect-src 'self'。
 *  - 桌面壳页 (launcher/prewarm/loading) 无引擎同源后端, 故 docsEnabled=false。
 *  - 需在页面自身脚本之后引入; body 上标 data-adr-page="easy|pro|call|index"。
 */
"use strict";
(function () {
  var body = document.body;
  if (!body) return;
  var page = body.getAttribute("data-adr-page") || "";
  var docsEnabled = body.getAttribute("data-adr-docs") !== "off" &&
    ["index", "easy", "pro", "call", "settings"].indexOf(page) >= 0;

  var ICON = {
    mark: '<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14v-4M8 17V7M12 20V4M16 17V7M20 14v-4"/></svg>',
    docs: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    easy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v3M12 18v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M3 12h3M18 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/><circle cx="12" cy="12" r="3.5"/></svg>',
    pro: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19V9M10 19V5M16 19v-6M22 19H2"/></svg>',
    call: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M8 12h8M12 8v8"/><rect x="3" y="3" width="18" height="18" rx="4"/></svg>',
    book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v18H6.5A2.5 2.5 0 0 1 4 18.5z"/><path d="M8 7h8M8 11h6"/></svg>'
  };

  var NAV = [
    { id: "easy", href: "/easy", label: ["新手台", "Easy"] },
    { id: "pro", href: "/pro", label: ["专业台", "Pro"] },
    { id: "call", href: "/call", label: ["调用台", "API"] },
    { id: "settings", href: "/settings", label: ["设置", "Settings"] }
  ];

  /* 语言与 i18n.js 同源 (localStorage adr_lang); i18n.js 尚未装载时回落中文。 */
  function lang() {
    try {
      if (typeof I !== "undefined" && I && I.lang) return I.lang;
    } catch (e) { /* noop */ }
    try { return localStorage.getItem("adr_lang") === "en" ? "en" : "zh"; } catch (e) { return "zh"; }
  }
  function tr(zh, en) { return lang() === "en" ? en : zh; }

  /* 页面切换修复: 顶栏链接保留桌面壳参数 (desktop/shell_origin), 否则从
     /easy 切到 /pro 后「返回启动器」消失; api_key 已转存 sessionStorage,
     不再放进 URL (避免历史记录/截图外泄)。 */
  function pageHref(path) {
    var qs = "";
    try {
      var p = new URLSearchParams(location.search);
      p.delete("api_key");
      qs = p.toString();
    } catch (e) { qs = ""; }
    return path + (qs ? "?" + qs : "");
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function safeUrl(u) {
    u = String(u || "").trim();
    if (/^(https?:|mailto:|\/|#)/i.test(u)) return u;
    return "#";
  }

  /* 与各页 adrKey() 同源: sessionStorage 里的 adr_api_key; 首页没有落 key 的
     逻辑, 故这里补一次从 ?api_key= 读取并转存。文档接口启用鉴权时需要它。 */
  function authHeaders() {
    var h = { Accept: "application/json" };
    var k = "";
    try {
      k = sessionStorage.getItem("adr_api_key") || "";
      if (!k) {
        k = new URLSearchParams(location.search).get("api_key") || "";
        if (k) sessionStorage.setItem("adr_api_key", k);
      }
    } catch (e) { k = ""; }
    if (k) h["X-API-Key"] = k;
    return h;
  }

  /* ── 顶栏 ──────────────────────────────────────────────────────────── */
  var EMOJI_RE = /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}\u{2B00}-\u{2BFF}]/gu;
  var NO_STRIP = { PRE: 1, CODE: 1, TEXTAREA: 1, INPUT: 1, SCRIPT: 1, STYLE: 1 };

  function stripEmoji(root) {
    if (!root) return;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        for (var p = n.parentNode; p && p !== root; p = p.parentNode) {
          if (NO_STRIP[p.nodeName] || (p.classList && p.classList.contains("adr-doc"))) {
            return NodeFilter.FILTER_REJECT;
          }
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var nodes = [], n;
    while ((n = walker.nextNode())) nodes.push(n);
    nodes.forEach(function (t) {
      var v = t.nodeValue.replace(EMOJI_RE, "");
      if (v === t.nodeValue) return;
      t.nodeValue = v.replace(/\s{2,}/g, " ").replace(/^\s+|\s+$/g, "");
    });
  }

  function buildTopnav(into) {
    var nav = into || document.createElement("nav");
    nav.className = "adr-topnav";
    nav.setAttribute("aria-label", tr("控制台导航", "Console navigation"));
    nav.innerHTML = NAV.map(function (x) {
      return '<a href="' + esc(pageHref(x.href)) + '"' +
        (x.id === page ? ' class="on" aria-current="page"' : "") + ">" + esc(tr(x.label[0], x.label[1])) + "</a>";
    }).join("");
    return nav;
  }

  function enhanceHeader() {
    var header = document.querySelector("header");
    if (!header) return null;
    header.classList.add("adr-topbar");

    var brand = header.querySelector(".brand, .logo");
    if (brand) {
      brand.classList.add("adr-brand");
      stripEmoji(brand);
      if (!brand.querySelector(".adr-mark")) {
        var markHtml = '<span class="adr-mark" aria-hidden="true">' + ICON.mark + "</span>";
        // 带 data-i18n-html 的品牌节点会在切语言时被整段 innerHTML 重写,
        // 图标放进去会被擦掉 → 作为兄弟节点插在它前面 (布局仍是 flex 一行)。
        if (brand.hasAttribute("data-i18n-html")) {
          brand.insertAdjacentHTML("beforebegin", markHtml);
        } else {
          brand.insertAdjacentHTML("afterbegin", markHtml);
        }
      }
    } else {
      brand = document.createElement("a");
      brand.className = "adr-brand";
      brand.href = "/";
      brand.innerHTML = '<span class="adr-mark" aria-hidden="true">' + ICON.mark + "</span><b>ADR</b>";
      header.insertBefore(brand, header.firstChild);
    }

    // 页面原有 <nav> 是页内页签 (专业台) — 挪到顶栏下方, 避免与全局导航打架
    var tabnav = header.querySelector("nav");
    if (tabnav) {
      var bar = document.createElement("div");
      bar.className = "adr-subnav";
      bar.appendChild(tabnav);
      header.parentNode.insertBefore(bar, header.nextSibling);
      header.insertBefore(buildTopnav(), brand.nextSibling);
    } else {
      header.insertBefore(buildTopnav(), brand.nextSibling);
    }
    return header;
  }

  /* ── Markdown (轻量, 先转义再渲染, 无 HTML 注入面) ─────────────────── */
  function inline(s) {
    var codes = [];
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, function (_, c) {
      codes.push(c);
      return "\u0000" + (codes.length - 1) + "\u0000";
    });
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    // 图片: CSP img-src 'self' 不放外链图, 降级为可点开的链接 (徽章不占空间)
    s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, function (_, alt, url) {
      return '<a class="adr-doc-img" href="' + esc(safeUrl(url)) +
        '" target="_blank" rel="noopener noreferrer">' + (alt || "图片") + "</a>";
    });
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_, txt, url) {
      return '<a href="' + esc(safeUrl(url)) + '" target="_blank" rel="noopener noreferrer">' + txt + "</a>";
    });
    s = s.replace(/\u0000(\d+)\u0000/g, function (_, i) {
      return "<code>" + codes[+i] + "</code>";
    });
    return s;
  }

  function mdToHtml(md) {
    var lines = String(md || "").replace(/\r\n/g, "\n").split("\n");
    var out = [], list = null, i = 0;
    function closeList() { if (list) { out.push("</" + list + ">"); list = null; } }
    while (i < lines.length) {
      var ln = lines[i];
      if (/^```/.test(ln.trim())) {
        closeList();
        var buf = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i].trim())) buf.push(lines[i++]);
        i++;
        out.push("<pre><code>" + esc(buf.join("\n")) + "</code></pre>");
        continue;
      }
      if (!ln.trim()) { closeList(); i++; continue; }
      // 纯图片/徽章行 (README 顶部 shields 之类): 不渲染, 避免一屏碎片
      if (/^\s*\[?!\[/.test(ln)) { closeList(); i++; continue; }
      var h = ln.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        closeList();
        var lv = h[1].length;
        out.push("<h" + lv + ">" + inline(h[2]) + "</h" + lv + ">");
        i++;
        continue;
      }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(ln)) { closeList(); out.push("<hr>"); i++; continue; }
      if (/^\s*>/.test(ln)) {
        closeList();
        var quote = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
        out.push("<blockquote>" + inline(quote.join(" ")) + "</blockquote>");
        continue;
      }
      if (/^\s*\|.*\|\s*$/.test(ln) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || "")) {
        closeList();
        var cells = function (row) {
          return row.trim().replace(/^\||\|$/g, "").split("|").map(function (c) { return c.trim(); });
        };
        var head = cells(ln);
        i += 2;
        var rows = [];
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
        out.push("<table><thead><tr>" + head.map(function (c) { return "<th>" + inline(c) + "</th>"; }).join("") +
          "</tr></thead><tbody>" + rows.map(function (r) {
            return "<tr>" + r.map(function (c) { return "<td>" + inline(c) + "</td>"; }).join("") + "</tr>";
          }).join("") + "</tbody></table>");
        continue;
      }
      var li = ln.match(/^\s*([-*+]|\d+\.)\s+(.*)$/);
      if (li) {
        var want = /\d/.test(li[1]) ? "ol" : "ul";
        if (list !== want) { closeList(); list = want; out.push("<" + list + ">"); }
        out.push("<li>" + inline(li[2]) + "</li>");
        i++;
        continue;
      }
      closeList();
      out.push("<p>" + inline(ln) + "</p>");
      i++;
    }
    closeList();
    return out.join("\n");
  }

  /* ── 文档抽屉 ──────────────────────────────────────────────────────── */
  var API = "/api/adr/v1/docs";
  var docs = {
    open: false, index: null, slug: "", built: false, busy: false,
    updates: {}, query: "", results: null, pending: ""
  };

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  function buildDocs() {
    if (docs.built) return;
    docs.built = true;
    var root = el("div", "adr-docs");
    root.id = "adrDocs";
    root.innerHTML =
      '<div class="adr-docs-scrim" data-close></div>' +
      '<aside class="adr-docs-panel" role="dialog" aria-modal="true" aria-label="' +
      esc(tr("文档中心", "Documentation")) + '">' +
      '<div class="adr-docs-head"><b>' + esc(tr("文档中心", "Documentation")) + '</b>' +
      '<span class="adr-docs-badge" id="adrDocsBadge">' + esc(tr("内置", "Built-in")) + "</span>" +
      '<span class="adr-docs-spacer"></span>' +
      '<button type="button" id="adrDocsCheck">' + esc(tr("检查更新", "Check updates")) + "</button>" +
      '<button type="button" class="adr-docs-x" data-close aria-label="' +
      esc(tr("关闭", "Close")) + '">✕</button></div>' +
      '<div class="adr-docs-body"><nav class="adr-docs-list" id="adrDocsList">' +
      '<div class="adr-docs-search"><input id="adrDocsQ" type="search" autocomplete="off" ' +
      'placeholder="' + esc(tr("搜索文档（支持模糊匹配）", "Search docs (fuzzy match)")) + '" ' +
      'aria-label="' + esc(tr("搜索文档", "Search documentation")) + '">' +
      '<button type="button" id="adrDocsGo" aria-label="' + esc(tr("搜索", "Search")) + '">' +
      esc(tr("搜索", "Go")) + "</button></div>" +
      '<div id="adrDocsGroups"></div></nav>' +
      '<div class="adr-docs-main" id="adrDocsMain"><div class="adr-docs-loading">' +
      esc(tr("正在载入文档…", "Loading documentation…")) + "</div></div></div>" +
      '<div class="adr-docs-foot"><span id="adrDocsState">' +
      esc(tr("离线可用 · 联网可拉取官方最新版", "Works offline · fetch the latest from upstream when online")) +
      '</span><span class="adr-docs-spacer"></span><span>ADR Studio</span></div>' +
      "</aside>";
    document.body.appendChild(root);

    root.addEventListener("click", function (e) {
      if (e.target.closest("[data-close]")) closeDocs();
      var item = e.target.closest("[data-slug]");
      if (item) loadDoc(item.getAttribute("data-slug"));
    });
    root.querySelector("#adrDocsCheck").addEventListener("click", checkUpdates);
    var q = root.querySelector("#adrDocsQ");
    root.querySelector("#adrDocsGo").addEventListener("click", function () { runSearch(q.value); });
    q.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); runSearch(q.value); }
      if (e.key === "Escape" && q.value) { e.stopPropagation(); q.value = ""; runSearch(""); }
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && docs.open) closeDocs();
    });
  }

  function badgeFallback(d) {
    return d && d.fallback ? '<span class="adr-docs-hit">' + esc(tr("中文原文", "Chinese only")) + "</span>" : "";
  }

  function renderList() {
    var list = document.getElementById("adrDocsGroups");
    if (!list) return;
    if (docs.results) {
      if (!docs.results.length) {
        list.innerHTML = '<div class="adr-docs-empty">' + esc(tr("没有匹配的文档", "No matching documents")) + "</div>";
        return;
      }
      list.innerHTML = '<div class="adr-docs-group">' +
        esc(tr("搜索结果", "Search results")) + " · " + docs.results.length + "</div>" +
        docs.results.map(function (d) {
          return '<button type="button" data-slug="' + esc(d.slug) + '"' +
            (d.slug === docs.slug ? ' class="on"' : "") + "><b>" + esc(d.title) + badgeFallback(d) +
            "</b><span>" + esc(d.snippet || d.summary || "") + "</span></button>";
        }).join("");
      return;
    }
    if (!docs.index) return;
    var groups = {};
    docs.index.forEach(function (d) {
      (groups[d.group] = groups[d.group] || []).push(d);
    });
    list.innerHTML = Object.keys(groups).map(function (g) {
      return '<div class="adr-docs-group">' + esc(g) + "</div>" + groups[g].map(function (d) {
        return '<button type="button" data-slug="' + esc(d.slug) + '"' +
          (d.slug === docs.slug ? ' class="on"' : "") + "><b>" + esc(d.title) + badgeFallback(d) + "</b>" +
          "<span>" + esc(d.summary || (tr("更新于 ", "Updated ") + (d.updated || "—"))) + "</span></button>";
      }).join("");
    }).join("");
  }

  function markActive() {
    var list = document.getElementById("adrDocsGroups");
    if (!list) return;
    Array.prototype.forEach.call(list.querySelectorAll("[data-slug]"), function (b) {
      b.classList.toggle("on", b.getAttribute("data-slug") === docs.slug);
    });
  }

  /* 无结果兜底: 直接给「问题处理 / 使用指南」两个入口, 不让用户卡在空列表。 */
  function renderNoResult(q) {
    var main = document.getElementById("adrDocsMain");
    if (!main) return;
    main.innerHTML = '<div class="adr-docs-empty">' +
      esc(tr("没有找到与「" + q + "」相关的文档。", "No documents match “" + q + "”.")) +
      '<div class="adr-docs-suggest">' +
      '<button type="button" data-slug="troubleshooting">' + esc(tr("问题处理", "Troubleshooting")) + "</button>" +
      '<button type="button" data-slug="guide">' + esc(tr("使用指南", "User guide")) + "</button>" +
      "</div></div>";
  }

  function runSearch(raw) {
    var q = String(raw == null ? "" : raw).trim();
    docs.query = q;
    if (!q) {
      docs.results = null;
      renderList();
      return;
    }
    var groups = document.getElementById("adrDocsGroups");
    if (groups) groups.innerHTML = '<div class="adr-docs-loading">' + esc(tr("搜索中…", "Searching…")) + "</div>";
    fetch(API + "/search?q=" + encodeURIComponent(q) + "&lang=" + encodeURIComponent(lang()),
      { headers: authHeaders() })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error((j && j.detail) || ("HTTP " + r.status)); return j; }); })
      .then(function (res) {
        docs.results = (res && res.results) || [];
        renderList();
        if (!docs.results.length) {
          renderNoResult(q);
        } else {
          loadDoc(docs.results[0].slug);
        }
      })
      .catch(function (err) {
        docs.results = [];
        renderList();
        var main = document.getElementById("adrDocsMain");
        if (main) main.innerHTML = '<div class="adr-doc-err">' + esc(tr("搜索失败：", "Search failed: ") + err.message) + "</div>";
      });
  }

  function openDocs(wantSlug) {
    buildDocs();
    docs.open = true;
    if (wantSlug) docs.pending = wantSlug;
    var root = document.getElementById("adrDocs");
    root.classList.add("open");
    document.documentElement.style.overflow = "hidden";
    if (docs.index) {
      var target = docs.pending || docs.slug || "quickstart";
      docs.pending = "";
      loadDoc(target);
      return;
    }
    fetch(API + "?lang=" + encodeURIComponent(lang()), { headers: authHeaders() })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (data) {
        docs.index = (data && data.docs) || [];
        if (!docs.index.length) {
          document.getElementById("adrDocsMain").innerHTML =
            '<div class="adr-docs-empty">' +
            esc(tr("安装包内未找到文档。请确认 docs/ 目录完整。",
              "No bundled documentation found. Check that the docs/ folder is present.")) +
            "</div>";
          return;
        }
        renderList();
        var wanted = docs.pending && docs.index.filter(function (d) { return d.slug === docs.pending; })[0];
        docs.pending = "";
        var first = wanted ||
          docs.index.filter(function (d) { return d.slug === "quickstart"; })[0] || docs.index[0];
        loadDoc(first.slug);
      })
      .catch(function (err) {
        document.getElementById("adrDocsMain").innerHTML =
          '<div class="adr-doc-err">' + esc(tr("文档接口不可用：", "Documentation API unavailable: ") + err.message) +
          "<br>" + esc(tr("控制台页面需由 ADR 服务端提供（同源 /api/adr/v1/docs）。",
            "Console pages must be served by the ADR server (same-origin /api/adr/v1/docs).")) + "</div>";
      });
  }

  function closeDocs() {
    docs.open = false;
    var root = document.getElementById("adrDocs");
    if (root) root.classList.remove("open");
    document.documentElement.style.overflow = "";
  }

  function loadDoc(slug, source) {
    docs.slug = slug;
    markActive();
    var main = document.getElementById("adrDocsMain");
    if (!main) return;
    main.innerHTML = '<div class="adr-docs-loading">' + esc(tr("正在载入…", "Loading…")) + "</div>";
    var qs = "lang=" + encodeURIComponent(lang());
    if (source === "online") qs += "&source=online";
    var url = API + "/" + encodeURIComponent(slug) + "?" + qs;
    fetch(url, { headers: authHeaders() })
      .then(function (r) {
        return r.json().then(function (j) {
          if (!r.ok) throw new Error((j && j.detail) || ("HTTP " + r.status));
          return j;
        });
      })
      .then(function (doc) {
        var badges = '<span class="tag ' + (doc.source === "online" ? "online" : "") + '">' +
          (doc.source === "online" ? tr("在线最新", "Latest online") : tr("本地内置", "Bundled locally")) + "</span>";
        if (docs.updates[slug]) badges += '<span class="tag update">' + esc(tr("有新版本", "Update available")) + "</span>";
        if (doc.fallback) badges += '<span class="tag">' + esc(tr("中文原文（英文版待补）", "Chinese original (English pending)")) + "</span>";
        badges += "<span>" + esc(doc.updated || "") + "</span>";
        if (doc.source !== "online") {
          badges += '<span class="adr-docs-spacer" style="flex:1"></span>' +
            '<button type="button" id="adrDocsOnline" style="height:26px;padding:0 10px;border-radius:7px;' +
            'border:1px solid var(--line,#dcebfb);background:none;color:var(--sub,#5b6b82);' +
            'font-family:inherit;font-size:11.5px;cursor:pointer">' + esc(tr("在线版", "Online")) + '</button>';
        }
        main.innerHTML = '<div class="adr-docs-meta">' + badges + "</div>" +
          '<article class="adr-doc">' + mdToHtml(doc.markdown) + "</article>";
        var btn = document.getElementById("adrDocsOnline");
        if (btn) btn.addEventListener("click", function () { loadDoc(slug, "online"); });
        main.scrollTop = 0;
        var panel = document.querySelector(".adr-docs-panel");
        if (panel) panel.classList.add("reading");
      })
      .catch(function (err) {
        if (source === "online") {
          var state = document.getElementById("adrDocsState");
          if (state) state.textContent = tr("在线版不可用：", "Online version unavailable: ") +
            err.message + tr(" · 已切回本地内置", " · fell back to bundled copy");
          loadDoc(slug, "local");
          return;
        }
        main.innerHTML = '<div class="adr-doc-err">' +
          esc(tr("读取《" + slug + "》失败：", "Failed to open “" + slug + "”: ") + err.message) + "</div>";
      });
  }

  function checkUpdates() {
    var btn = document.getElementById("adrDocsCheck");
    var state = document.getElementById("adrDocsState");
    if (!btn || docs.busy) return;
    docs.busy = true;
    btn.disabled = true;
    btn.textContent = tr("检查中…", "Checking…");
    if (state) state.textContent = tr("正在比对官方仓库…", "Comparing with the upstream repository…");
    fetch(API + "/check", { headers: authHeaders() })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error((j && j.detail) || ("HTTP " + r.status)); return j; }); })
      .then(function (res) {
        if (res.online === false) {
          if (state) {
            state.textContent = tr("离线：无法访问官方仓库", "Offline: cannot reach the upstream repository") +
              (res.error ? "（" + res.error + "）" : "") +
              tr(" · 继续使用内置文档", " · continuing with bundled docs");
          }
          return;
        }
        docs.updates = {};
        (res.updates || []).forEach(function (s) { docs.updates[s] = true; });
        var badge = document.getElementById("adrDocsBadge");
        if (badge) badge.textContent = res.online ? tr("已联网", "Online") : tr("内置", "Built-in");
        if (state) {
          if (res.local === 0) {
            // 本机一份文档都没有 (安装包漏拷 docs/) — 不能报"已是最新"
            state.textContent = tr("本机安装包未包含文档，无法比对（请重新安装完整版本）",
              "No bundled docs on this install — cannot compare (reinstall the full package)");
          } else {
            state.textContent = res.updates && res.updates.length
              ? tr("发现 " + res.updates.length + " 篇有更新，可切换「在线版」查看",
                res.updates.length + " document(s) have updates — switch to “Online” to read")
              : tr("已是最新（官方仓库无变化）", "Up to date (no upstream changes)");
          }
        }
        if (docs.slug) loadDoc(docs.slug);
      })
      .catch(function (err) {
        if (state) state.textContent = tr("无法访问官方仓库：", "Cannot reach upstream: ") + err.message;
      })
      .then(function () {
        docs.busy = false;
        btn.disabled = false;
        btn.textContent = tr("检查更新", "Check updates");
      });
  }

  /* 各页共享: 带 ?api_key= 打开时统一转存 sessionStorage, 并从地址栏摘掉
     (避免进入历史记录/截图/粘贴外发, 同时保证页面切换后业务接口仍有 key)。 */
  function persistKey() {
    try {
      var params = new URLSearchParams(location.search);
      var k = params.get("api_key") || "";
      if (k && !sessionStorage.getItem("adr_api_key")) sessionStorage.setItem("adr_api_key", k);
      if (k) {
        params.delete("api_key");
        var qs = params.toString();
        history.replaceState(null, "", location.pathname + (qs ? "?" + qs : "") + location.hash);
      }
    } catch (e) { /* file:// / 隐私模式: 忽略 */ }
  }

  /* 语言切换: 顶栏与文档抽屉是 JS 生成/缓存的, i18n.js 的 data-i18n 覆盖不到,
     这里整体重建 (在 i18n.js 派发 adr:lang 事件后执行)。 */
  function onLangChange() {
    var nav = document.querySelector(".adr-topnav");
    if (nav) buildTopnav(nav);
    var db = document.querySelector(".adr-docbtn");
    if (db) {
      db.title = tr("打开内置文档中心", "Open built-in documentation");
      var label = db.querySelector("span");
      if (label) label.textContent = tr("文档", "Docs");
    }
    var root = document.getElementById("adrDocs");
    if (root && root.parentNode) root.parentNode.removeChild(root);
    docs.built = false;
    docs.index = null;
    docs.results = null;
    docs.updates = {};
    docs.query = "";
    docs.slug = "";
    if (docs.open) openDocs();
  }

  /* ── 启动 ─────────────────────────────────────────────────────────── */
  // 无 favicon 时浏览器会 404 请求 /favicon.ico, 用内联 SVG 补一个 (零额外请求)
  var FAVICON = "data:image/svg+xml," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">' +
    '<rect width="32" height="32" rx="8" fill="#2b6cb0"/>' +
    '<path d="M16 6l7 20h-4l-1.4-4.2h-3.2L13 26H9zm0 6.4l-1.4 5.6h2.8z" fill="#fff"/></svg>');

  function ensureFavicon() {
    if (document.querySelector('link[rel="icon"]')) return;
    var l = document.createElement("link");
    l.rel = "icon";
    l.href = FAVICON;
    document.head.appendChild(l);
  }

  /* P0: 首启 / 全局设置写入的服务端 ui.theme / ui.lang 作为首次默认值。
     只在本机还没有本地偏好时应用, 之后用户在各页手动切换的值优先。 */
  function applyServerUiDefaults() {
    var hasLang = false, hasTheme = false;
    try {
      hasLang = !!localStorage.getItem("adr_lang");
      hasTheme = !!localStorage.getItem("adr.theme");
    } catch (e) { return; }
    if (hasLang && hasTheme) return;
    var key = "";
    try { key = sessionStorage.getItem("adr_api_key") || ""; } catch (e) {}
    fetch("/api/adr/v1/settings", { headers: key ? { "X-API-Key": key } : {} })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !j.ui) return;
        if (!hasTheme && window.adrTheme
            && (j.ui.theme === "default" || j.ui.theme === "classic")) {
          adrTheme.set(j.ui.theme);
        }
        if (!hasLang && window.I && (j.ui.lang === "zh" || j.ui.lang === "en")
            && j.ui.lang !== I.lang) {
          I.setLang(j.ui.lang);
        }
      })
      .catch(function () {});
  }

  function boot() {
    applyServerUiDefaults();
    persistKey();
    ensureFavicon();
    if (page === "index") {
      var host = document.querySelector(".hub-top");
      // 页面模板里已有一个空的 .adr-topnav (index.html), 复用它而不是再插一个
      var existing = (host && host.querySelector(".adr-topnav")) || document.querySelector(".adr-topnav");
      var nav = buildTopnav(existing);
      if (!existing && host) host.appendChild(nav);
      var mark = document.querySelector(".hub-brand .adr-mark");
      if (mark) mark.innerHTML = ICON.mark;
    } else if (["easy", "pro", "call", "settings"].indexOf(page) >= 0) {
      var header = enhanceHeader();
      // 正文标题/按钮里的 emoji 一并清掉 (跳过 pre/code/textarea/.adr-doc)
      stripEmoji(document.querySelector("main") || document.body);
      if (header && docsEnabled) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "adr-docbtn";
        btn.title = tr("打开内置文档中心", "Open built-in documentation");
        btn.innerHTML = ICON.docs + "<span>" + esc(tr("文档", "Docs")) + "</span>";
        btn.addEventListener("click", function () { openDocs(); });
        header.appendChild(btn);
      }
    }
    // 事件委托: 文档入口可能被 i18n 的 data-i18n-html 重建, 逐节点绑会失效
    document.addEventListener("click", function (e) {
      var opener = e.target.closest && e.target.closest("[data-adr-docs-open]");
      if (!opener) return;
      e.preventDefault();
      openDocs(opener.getAttribute("data-doc-slug") || "");
    });
    if (docsEnabled && location.hash === "#docs") openDocs();
    document.addEventListener("adr:lang", onLangChange);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
