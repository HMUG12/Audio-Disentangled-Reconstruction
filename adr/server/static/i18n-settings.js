/* i18n-settings.js — 全局设置独立页 (/settings) 中/英字典 (由 /i18n-settings.js 提供)。
 * 格式: { key: ["中文", "English"], ... }; key 统一以 settings. 开头。 */
"use strict";
const I18N_DICT = {
  "settings.title": ["ADR Studio · 设置", "ADR Studio · Settings"],

  "settings.logo": [
    "ADR 全局设置<small>合成默认值 · 声音与模型 · 界面</small>",
    "ADR Settings<small>Synthesis defaults · Voice &amp; model · Interface</small>"
  ],
  "settings.back": ["← 返回启动器", "← Back to launcher"],
  "settings.lead": [
    "这些值只是控制台页面的默认参数，不会改变外部 API 调用方的行为。改完点保存。",
    "These values are console defaults only — they never change what external API callers get. Click Save when you are done."
  ],

  "settings.sec_synth": ["合成默认值", "Synthesis defaults"],
  "settings.sec_synth_desc": [
    "新打开的合成页面会默认带上这些参数。",
    "Newly opened synthesis pages start with these values."
  ],
  "settings.top_k": ["候选数量（top_k）", "Top-k candidates"],
  "settings.top_k_hint": [
    "每次采样保留的候选词数，越大越灵活也越慢。范围 1–100，默认 15。",
    "Candidates kept at each sampling step. Higher is more varied but slower. Range 1–100, default 15."
  ],
  "settings.top_p": ["核采样阈值（top_p）", "Nucleus sampling (top_p)"],
  "settings.top_p_hint": [
    "累计概率阈值，越小越保守。范围 0.05–1.0，默认 1.0。",
    "Cumulative probability cutoff; lower is more conservative. Range 0.05–1.0, default 1.0."
  ],
  "settings.temperature": ["随机度（temperature）", "Randomness (temperature)"],
  "settings.temperature_hint": [
    "越高起伏越大，越低越平稳。范围 0.05–2.0，默认 1.0。",
    "Higher adds variation, lower keeps delivery steady. Range 0.05–2.0, default 1.0."
  ],
  "settings.speed_factor": ["语速倍率", "Speed multiplier"],
  "settings.speed_factor_hint": [
    "1.0 为原速，大于 1 更快，小于 1 更慢。范围 0.5–2.0，默认 1.0。",
    "1.0 is the original speed; above 1 is faster, below 1 is slower. Range 0.5–2.0, default 1.0."
  ],
  "settings.seed": ["随机种子", "Random seed"],
  "settings.seed_hint": [
    "-1 表示每次随机；固定为 0–2147483647 可复现结果。默认 -1。",
    "-1 randomizes every run; pin it to 0–2147483647 for reproducible results. Default -1."
  ],
  "settings.repetition_penalty": ["重复惩罚", "Repetition penalty"],
  "settings.repetition_penalty_hint": [
    "略高于 1 可减少重复读音，过高会显得生硬。范围 1.0–2.0，默认 1.35。",
    "Slightly above 1 curbs repeated sounds; too high sounds stiff. Range 1.0–2.0, default 1.35."
  ],
  "settings.media_type": ["默认导出格式", "Default export format"],
  "settings.media_type_hint": [
    "wav 无损；mp3 / ogg 体积更小，适合分享。",
    "wav is lossless; mp3 / ogg are smaller and easier to share."
  ],
  "settings.opt_wav": ["WAV（无损）", "WAV (lossless)"],
  "settings.opt_mp3": ["MP3（体积小）", "MP3 (smaller)"],
  "settings.opt_ogg": ["OGG（体积小）", "OGG (smaller)"],
  "settings.streaming_mode": ["流式返回", "Streaming response"],
  "settings.streaming_mode_hint": [
    "关闭时一次性返回完整音频；开启或自动由引擎与调用方决定。",
    "Off returns the whole clip at once. On or Auto is resolved by the engine and the caller."
  ],
  "settings.opt_stream_off": ["关闭", "Off"],
  "settings.opt_stream_on": ["开启", "On"],
  "settings.opt_stream_auto": ["自动", "Auto"],

  "settings.sec_voice": ["声音与模型", "Voice & model"],
  "settings.sec_voice_desc": [
    "默认档案与语言只作用于控制台，外部 API 仍按请求参数走。",
    "The default profile and language apply to the console only; external API calls still follow their own request parameters."
  ],
  "settings.default_profile": ["默认音色档案", "Default voice profile"],
  "settings.default_profile_hint": [
    "新建合成时默认选中的档案。留空表示跟随引擎当前的默认音色。",
    "Profile pre-selected for new synthesis. Leave empty to follow the engine's current default voice."
  ],
  "settings.profile_none": ["跟随引擎默认", "Follow engine default"],
  "settings.profile_missing": ["（当前值，不在档案列表中）", "(current value, not in the profile list)"],
  "settings.voice_language": ["默认合成语言", "Default synthesis language"],
  "settings.voice_language_hint": [
    "影响音色匹配与读音规则。",
    "Affects voice matching and pronunciation rules."
  ],
  "settings.lang_zh": ["中文", "Chinese"],
  "settings.lang_en": ["English", "English"],

  "settings.sec_ui": ["界面", "Interface"],
  "settings.sec_ui_desc": [
    "主题和语言直接复用本机浏览器里的控制台设置，在这里改动会立刻生效。",
    "Theme and language reuse the console settings stored in this browser; changes here take effect immediately."
  ],
  "settings.ui_theme": ["主题", "Theme"],
  "settings.ui_theme_hint": [
    "读取 localStorage 的 adr.theme，与各控制台页共用。切换后本机所有页面一起变。",
    "Reads localStorage adr.theme and shares it with every console page. Switching updates all pages on this machine."
  ],
  "settings.ui_theme_switch": ["切换主题", "Switch theme"],
  "settings.theme_default": ["默认（浅蓝）", "Default (light blue)"],
  "settings.theme_classic": ["经典（原配色）", "Classic (original palette)"],
  "settings.ui_lang": ["语言", "Language"],
  "settings.ui_lang_hint": [
    "读取 localStorage 的 adr_lang，与各控制台页共用。切换后本页与其它页面立即生效。",
    "Reads localStorage adr_lang and shares it with every console page. Switching applies here and elsewhere right away."
  ],
  "settings.ui_lang_switch": ["切换语言", "Switch language"],
  "settings.ui_preview": ["去新手台看看效果", "Preview in the beginner console"],

  "settings.sec_adv": ["高级", "Advanced"],
  "settings.sec_adv_desc": [
    "这些开关影响下次打开控制台时的行为；恢复默认会把它们一并重置。",
    "These switches affect how the console behaves next time. Restoring defaults resets them too."
  ],
  "settings.motion": ["界面动效", "Interface motion"],
  "settings.motion_hint": [
    "关闭后减少过渡动画，远程桌面或低配机器更流畅。",
    "Turn off to reduce transition animations — smoother on remote desktops and low-end machines."
  ],
  "settings.onboarding_done": ["新手引导已完成", "Onboarding completed"],
  "settings.onboarding_done_hint": [
    "关闭后，下次打开新手台会重新走一遍引导。平时保持开启即可。",
    "Turn this off to replay the beginner walkthrough next time. Leave it on normally."
  ],
  "settings.on": ["已开启", "On"],
  "settings.off": ["已关闭", "Off"],

  "settings.sec_about": ["关于", "About"],
  "settings.sec_about_desc": [
    "版本信息取自服务的状态接口，取不到时显示“未知”。",
    "Version details come from the service status endpoint; when unavailable they show as unknown."
  ],
  "settings.version": ["API 版本", "API version"],
  "settings.service": ["服务", "Service"],
  "settings.meta_schema": ["设置格式版本", "Settings schema version"],
  "settings.meta_updated": ["最近保存", "Last saved"],
  "settings.version_unknown": ["未知", "Unknown"],
  "settings.updated_never": ["从未保存（当前为默认值）", "Never saved (defaults in use)"],

  "settings.save": ["保存", "Save"],
  "settings.saving": ["保存中…", "Saving…"],
  "settings.reset": ["恢复默认", "Restore defaults"],
  "settings.confirm_reset": [
    "恢复默认会覆盖当前全部设置，且无法撤销。确定继续？",
    "Restoring defaults overwrites every setting and cannot be undone. Continue?"
  ],
  "settings.save_ok": [
    "已保存。新的默认值会在下次打开控制台时生效。",
    "Saved. The new defaults apply the next time you open a console page."
  ],
  "settings.reset_ok": ["已恢复默认设置。", "Defaults restored."],

  "settings.err_401": [
    "需要 API Key，请在启动器中重新进入。",
    "An API key is required. Please re-enter from the launcher."
  ],
  "settings.err_fetch": [
    "连不上本地服务，请确认服务仍在运行。",
    "Cannot reach the local service. Make sure it is still running."
  ],
  "settings.err_http": ["请求失败（HTTP {s}）。", "Request failed (HTTP {s})."],
  "settings.err_empty": ["请填写所有数值字段。", "Fill in every numeric field."],
  "settings.err_int": ["候选数量与随机种子必须是整数。", "Top-k and seed must be whole numbers."],
  "settings.err_number": ["数值格式不正确。", "That number is not valid."]
};
