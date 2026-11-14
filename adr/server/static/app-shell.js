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
    ["index", "easy", "pro", "call"].indexOf(page) >= 0;

  var ICON = {
    mark: '<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14v-4M8 17V7M12 20V4M16 17V7M20 14v-4"/></svg>',
    docs: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    easy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v3M12 18v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M3 12h3M18 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/><circle cx="12" cy="12" r="3.5"/></svg>',
    pro: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19V9M10 19V5M16 19v-6M22 19H2"/></svg>',
    call: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M8 12h8M12 8v8"/><rect x="3" y="3" width="18" height="18" rx="4"/></svg>',
    book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v18H6.5A2.5 2.5 0 0 1 4 18.5z"/><path d="M8 7h8M8 11h6"/></svg>'
  };

  var NAV = [
    { id: "easy", href: "/easy", label: "新手台" },
    { id: "pro", href: "/pro", label: "专业台" },
    { id: "call", href: "/call", label: "调用台" }
  ];

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

  function buildTopnav() {
    var nav = document.createElement("nav");
    nav.className = "adr-topnav";
    nav.setAttribute("aria-label", "控制台导航");
    nav.innerHTML = NAV.map(function (x) {
      return '<a href="' + x.href + '"' + (x.id === page ? ' class="on" aria-current="page"' : "") + ">" + x.label + "</a>";
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
        brand.insertAdjacentHTML("afterbegin", '<span class="adr-mark" aria-hidden="true">' + ICON.mark + "</span>");
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
  var docs = { open: false, index: null, slug: "", built: false, busy: false, updates: {} };

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
      '<aside class="adr-docs-panel" role="dialog" aria-modal="true" aria-label="文档中心">' +
      '<div class="adr-docs-head"><b>文档中心</b><span class="adr-docs-badge" id="adrDocsBadge">内置</span>' +
      '<span class="adr-docs-spacer"></span>' +
      '<button type="button" id="adrDocsCheck">检查更新</button>' +
      '<button type="button" class="adr-docs-x" data-close aria-label="关闭">✕</button></div>' +
      '<div class="adr-docs-body"><nav class="adr-docs-list" id="adrDocsList"></nav>' +
      '<div class="adr-docs-main" id="adrDocsMain"><div class="adr-docs-loading">正在载入文档…</div></div></div>' +
      '<div class="adr-docs-foot"><span id="adrDocsState">离线可用 · 联网可拉取官方最新版</span>' +
      '<span class="adr-docs-spacer"></span><span>ADR Studio</span></div>' +
      "</aside>";
    document.body.appendChild(root);

    root.addEventListener("click", function (e) {
      if (e.target.closest("[data-close]")) closeDocs();
      var item = e.target.closest("[data-slug]");
      if (item) loadDoc(item.getAttribute("data-slug"));
    });
    root.querySelector("#adrDocsCheck").addEventListener("click", checkUpdates);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && docs.open) closeDocs();
    });
  }

  function renderList() {
    var list = document.getElementById("adrDocsList");
    if (!list || !docs.index) return;
    var groups = {};
    docs.index.forEach(function (d) {
      (groups[d.group] = groups[d.group] || []).push(d);
    });
    list.innerHTML = Object.keys(groups).map(function (g) {
      return '<div class="adr-docs-group">' + esc(g) + "</div>" + groups[g].map(function (d) {
        return '<button type="button" data-slug="' + esc(d.slug) + '"' +
          (d.slug === docs.slug ? ' class="on"' : "") + "><b>" + esc(d.title) + "</b>" +
          "<span>" + esc(d.summary || ("更新于 " + (d.updated || "—"))) + "</span></button>";
      }).join("");
    }).join("");
  }

  function markActive() {
    var list = document.getElementById("adrDocsList");
    if (!list) return;
    Array.prototype.forEach.call(list.querySelectorAll("[data-slug]"), function (b) {
      b.classList.toggle("on", b.getAttribute("data-slug") === docs.slug);
    });
  }

  function openDocs() {
    buildDocs();
    docs.open = true;
    var root = document.getElementById("adrDocs");
    root.classList.add("open");
    document.documentElement.style.overflow = "hidden";
    if (!docs.index) {
      fetch(API, { headers: authHeaders() })
        .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
        .then(function (data) {
          docs.index = (data && data.docs) || [];
          if (!docs.index.length) {
            document.getElementById("adrDocsMain").innerHTML =
              '<div class="adr-docs-empty">安装包内未找到文档。请确认 docs/ 目录完整。</div>';
            return;
          }
          renderList();
          var first = docs.index.filter(function (d) { return d.slug === "quickstart"; })[0] || docs.index[0];
          loadDoc(first.slug);
        })
        .catch(function (err) {
          document.getElementById("adrDocsMain").innerHTML =
            '<div class="adr-doc-err">文档接口不可用：' + esc(err.message) +
            "<br>控制台页面需由 ADR 服务端提供（同源 /api/adr/v1/docs）。</div>";
        });
    }
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
    main.innerHTML = '<div class="adr-docs-loading">正在载入…</div>';
    var url = API + "/" + encodeURIComponent(slug) + (source === "online" ? "?source=online" : "");
    fetch(url, { headers: authHeaders() })
      .then(function (r) {
        return r.json().then(function (j) {
          if (!r.ok) throw new Error((j && j.detail) || ("HTTP " + r.status));
          return j;
        });
      })
      .then(function (doc) {
        var badges = '<span class="tag ' + (doc.source === "online" ? "online" : "") + '">' +
          (doc.source === "online" ? "在线最新" : "本地内置") + "</span>";
        if (docs.updates[slug]) badges += '<span class="tag update">有新版本</span>';
        badges += "<span>" + esc(doc.updated || "") + "</span>";
        if (doc.source !== "online") {
          badges += '<span class="adr-docs-spacer" style="flex:1"></span>' +
            '<button type="button" id="adrDocsOnline" style="height:26px;padding:0 10px;border-radius:7px;' +
            'border:1px solid var(--line,#dcebfb);background:none;color:var(--sub,#5b6b82);' +
            'font-family:inherit;font-size:11.5px;cursor:pointer">在线版</button>';
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
          if (state) state.textContent = "在线版不可用：" + err.message + " · 已切回本地内置";
          loadDoc(slug, "local");
          return;
        }
        main.innerHTML = '<div class="adr-doc-err">读取《' + esc(slug) + "》失败：" + esc(err.message) + "</div>";
      });
  }

  function checkUpdates() {
    var btn = document.getElementById("adrDocsCheck");
    var state = document.getElementById("adrDocsState");
    if (!btn || docs.busy) return;
    docs.busy = true;
    btn.disabled = true;
    btn.textContent = "检查中…";
    if (state) state.textContent = "正在比对官方仓库…";
    fetch(API + "/check", { headers: authHeaders() })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error((j && j.detail) || ("HTTP " + r.status)); return j; }); })
      .then(function (res) {
        if (res.online === false) {
          if (state) {
            state.textContent = "离线：无法访问官方仓库" +
              (res.error ? "（" + res.error + "）" : "") + " · 继续使用内置文档";
          }
          return;
        }
        docs.updates = {};
        (res.updates || []).forEach(function (s) { docs.updates[s] = true; });
        var badge = document.getElementById("adrDocsBadge");
        if (badge) badge.textContent = res.online ? "已联网" : "内置";
        if (state) {
          state.textContent = res.updates && res.updates.length
            ? "发现 " + res.updates.length + " 篇有更新，可切换「在线版」查看"
            : "已是最新（官方仓库无变化）";
        }
        if (docs.slug) loadDoc(docs.slug);
      })
      .catch(function (err) {
        if (state) state.textContent = "无法访问官方仓库：" + err.message;
      })
      .then(function () {
        docs.busy = false;
        btn.disabled = false;
        btn.textContent = "检查更新";
      });
  }

  /* ── 启动 ─────────────────────────────────────────────────────────── */
  function boot() {
    if (page === "index") {
      var host = document.querySelector(".hub-top");
      if (host) {
        var nav = buildTopnav();
        host.appendChild(nav);
      }
      var mark = document.querySelector(".hub-brand .adr-mark");
      if (mark) mark.innerHTML = ICON.mark;
    } else if (["easy", "pro", "call"].indexOf(page) >= 0) {
      var header = enhanceHeader();
      // 正文标题/按钮里的 emoji 一并清掉 (跳过 pre/code/textarea/.adr-doc)
      stripEmoji(document.querySelector("main") || document.body);
      if (header && docsEnabled) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "adr-docbtn";
        btn.title = "打开内置文档中心";
        btn.innerHTML = ICON.docs + "<span>文档</span>";
        btn.addEventListener("click", openDocs);
        header.appendChild(btn);
      }
    }
    Array.prototype.forEach.call(document.querySelectorAll("[data-adr-docs-open]"), function (n) {
      n.addEventListener("click", function (e) { e.preventDefault(); openDocs(); });
    });
    if (docsEnabled && location.hash === "#docs") openDocs();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
