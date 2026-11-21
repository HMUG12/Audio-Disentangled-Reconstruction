/* sfx.js — 内置界面音效 (4 个 wav, 无依赖)。
 *
 * 开关: localStorage["adr_sound"] !== "0" (默认开), 由启动器设置页写入。
 * 用法: ADR_SFX.play("enter" | "ok" | "err" | "notify")
 * 说明: 音量固定 0.35, 浏览器自动播放策略下首次点击前的提示音会被静默拒绝,
 *       因此这里统一 catch, 不影响交互。
 */
(function () {
  var FILES = {
    enter: "assets/sfx/enter.wav",
    ok: "assets/sfx/ok.wav",
    err: "assets/sfx/err.wav",
    notify: "assets/sfx/notify.wav",
  };
  var pool = {};

  function enabled() {
    try { return localStorage.getItem("adr_sound") !== "0"; } catch (_) { return true; }
  }

  function play(name) {
    if (!enabled()) return;
    var src = FILES[name];
    if (!src) return;
    try {
      var a = pool[name] || (pool[name] = new Audio(src));
      a.volume = 0.35;
      a.currentTime = 0;
      var p = a.play();
      if (p && p.catch) p.catch(function () {});
    } catch (_) { /* 非桌面环境或无音频设备 */ }
  }

  function set(on) {
    try { localStorage.setItem("adr_sound", on ? "1" : "0"); } catch (_) {}
  }

  window.ADR_SFX = { play: play, set: set, enabled: enabled };
})();
