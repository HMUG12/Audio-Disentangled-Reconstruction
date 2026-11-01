; ADR Studio installer script (Inno Setup 6)
; Batch 44: NSIS has a hard ~2GB data limit (makensis #12345 error on our
; 3.89GB runtime), switched to Inno Setup which supports large single-file
; setups. Provides: license agreement page + Simplified Chinese UI + LZMA2.
;
; Build order (from repo root):
;   1. powershell -File scripts\build_runtime.ps1 [-AllowCuda]   (runtime payload)
;   2. npx -y @tauri-apps/cli@2 build --no-bundle   (in desktop/src-tauri)
;   3. ISCC.exe scripts\installer\installer.iss          (CPU 包)
;      ISCC.exe /DGPU_BUILD scripts\installer\installer.iss  (GPU 包, cu130 runtime)
; Output: desktop/src-tauri/target/release/bundle/inno/
;   ADR-Studio-<ver>-x64-setup.exe        (CPU 包)
;   ADR-Studio-<ver>-x64-gpu-setup.exe    (GPU 包)
;
; 双包共用同一 AppId: GPU 包装在 CPU 包上 = 原地升级 runtime, 不产生双份安装。

#define MyAppName "ADR Studio"
#define MyAppVersion "1.3.1"
#define MyAppPublisher "未知之致"
#define MyAppExe "adr-desktop.exe"
#define ReleaseDir "..\..\desktop\src-tauri\target\release"
; runtime 载荷目录按变体隔离 (build_runtime.ps1 产出 runtime-cpu / runtime-gpu),
; GPU 包构建由 build_installer.ps1 传 /DRUNTIME_DIR_NAME="runtime-gpu"
#ifndef RUNTIME_DIR_NAME
#define RUNTIME_DIR_NAME "runtime-cpu"
#endif
; 批次71 (复盘§二缺陷2): 构建侧可用 /DRUNTIME_DIR_ABS=<绝对路径> 覆盖 —
; 相对路径 ..\..\build\ 在 ISCC 内不归一化, 每条文件路径多占字符, 构建根
; 较深时实测 12 个文件超 MAX_PATH 260 (最长 269) → 编译报"找不到指定的路径"。
; build_installer.ps1 现默认经 Resolve-Path 传绝对路径, 并在 tauri 构建前
; 加 >250 预检 (源侧 + 装机侧)。
#ifdef RUNTIME_DIR_ABS
#define RuntimeDir RUNTIME_DIR_ABS
#else
#define RuntimeDir "..\..\build\" + RUNTIME_DIR_NAME
#endif

; GPU 变体: ISCC /DGPU_BUILD → 文件名带 -gpu 后缀 (runtime 内容由
; build_runtime.ps1 -AllowCuda 产出, 本脚本只负责命名区分)
#ifdef GPU_BUILD
#define VariantSuffix "-gpu"
#else
#define VariantSuffix ""
#endif

[Setup]
AppId={{6E1F3C7A-59B2-4A8D-9C43-D2F0A1B65E42}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Per-user install (no UAC), mirrors previous NSIS currentUser mode
DefaultDirName={localappdata}\Programs\ADR Studio
DisableProgramGroupPage=yes
LicenseFile=..\..\desktop\src-tauri\resources\license.rtf
OutputDir={#ReleaseDir}\bundle\inno
OutputBaseFilename=ADR-Studio-{#MyAppVersion}-x64{#VariantSuffix}-setup
SetupIconFile=..\..\desktop\src-tauri\icons\icon.ico
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExe}
; Large payload: lzma2 non-solid compresses blocks in parallel, single
; setup.exe can exceed 2GB safely with Inno
Compression=lzma2/max
SolidCompression=no
; 批次72 (卸载体验): 安装器与卸载器启动时检测该命名互斥体 — 壳
; (lib.rs create_app_mutex) 与服务 (adr/server/__main__.py _hold_app_mutex)
; 两端任一在跑即弹窗提示关闭, 治"卸载时不检查程序是否关闭"。
; 三端名字逐字一致 (tests/test_audit_packaging.py 防漂移)。
AppMutex=ADR-Studio-Running

[Languages]
Name: "chinesesimplified"; MessagesFile: "ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#ReleaseDir}\adr-desktop.exe"; DestDir: "{app}"; Flags: ignoreversion
; 批次60: 无 UI 服务版入口 (env 驱动, 与容器版同形态)
Source: "server.bat"; DestDir: "{app}"; Flags: ignoreversion
; Ship the AGPL-3.0 license text alongside the program (AGPL section 4)
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RuntimeDir}\*"; DestDir: "{app}\runtime"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; \
    Tasks: desktopicon

[UninstallDelete]
; 批次72 (卸载体验): 运行期生成的**可再生成**产物无条件清 (Inno 默认只删
; 安装清单文件, 这些目录会残留 = "卸载不干净"主因之一)。用户资产
; (声音档案/训练权重) 不在此段 — 走 [Code] 询问页, 默认保留。
; TTS 结果缓存 (settings.tts_cache_dir = REPO_ROOT/data/tts_cache)
Type: filesandordirs; Name: "{app}\runtime\data\tts_cache"
; GSV 训练中间产物 (gsv_finetune.py: 切片/asr/list/bind_ref, 可再生成)
Type: filesandordirs; Name: "{app}\runtime\third_party\gpt_sovits\output"
; GSV 训练日志 (tensorboard/配置, 排障用, 卸载后无意义)
Type: filesandordirs; Name: "{app}\runtime\third_party\gpt_sovits\logs"
; GSV resume 合并临时文件 (gsv_finetune.py TEMP/resume_init_*.pth)
Type: filesandordirs; Name: "{app}\runtime\third_party\gpt_sovits\TEMP"
; 用户数据被用户选择删除后 (见 [Code] CurUninstallStepChanged), 这些目录
; 会变成空壳 — 补 dirifempty 收尾, 否则 {app}\runtime 整体残留 (dirifempty
; 仅在目录为空时删除, 用户选择保留时无副作用)。
Type: dirifempty; Name: "{app}\runtime\data"
Type: dirifempty; Name: "{app}\runtime\third_party\gpt_sovits\SoVITS_weights_v2"
Type: dirifempty; Name: "{app}\runtime\third_party\gpt_sovits\GPT_weights_v2"
Type: dirifempty; Name: "{app}\runtime\third_party\rvc\assets\weights"
Type: dirifempty; Name: "{app}\runtime\third_party\rvc\logs"

[Run]
; 批次71: VC++ 2015-2022 x64 静默安装 (PBS Python 的 CRT 依赖, 缺它精简
; 系统装完 runtime 起不来; 已装则微软安装器秒过)。shellexec: 该 exe 清单
; 要求提权, 走 CreateProcess 会直接失败, ShellExecute 才能触发 UAC。
; 注意 redist 是机器级安装 — per-user 安装器这里会弹一次 UAC 属预期。
Filename: "{app}\runtime\redist\vc_redist.x64.exe"; \
    Parameters: "/install /quiet /norestart"; \
    WorkingDir: "{app}\runtime\redist"; \
    Flags: shellexec runhidden waituntilterminated; Check: VCRedistNeedsInstall
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; \
    Flags: nowait postinstall skipifsilent

[Code]
// 批次71: VC++ 2015-2022 x64 是否已装 (注册表 Runtimes\x64 键存在即装过;
// redist 机器级, 本安装器 per-user 只读 HKLM 无权限问题; 64 位安装模式
// 下无 WOW6432Node 重定向)
function VCRedistNeedsInstall(): Boolean;
begin
  Result := not RegKeyExists(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64');
end;

// WebView2 runtime check (Tauri 2 needs it; Win10/11 usually ships it).
// Mirrors what the old NSIS bootstrap did, minus the silent installer.
function InitializeSetup(): Boolean;
var
  R: Integer;
begin
  Result := True;
  if not (RegKeyExists(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}') or
          RegKeyExists(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}') or
          RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}')) then
  begin
    R := MsgBox('未检测到 WebView2 运行时（Windows 10/11 一般自带）。' #13#10
                '是否打开微软官网下载安装后再继续？', mbConfirmation, MB_YESNO);
    if R = IDYES then
      ShellExec('open', 'https://developer.microsoft.com/microsoft-edge/webview2/',
                '', '', SW_SHOWNORMAL, ewNoWait, R);
    Result := False;
  end;
end;

// ---------------------------------------------------------------------------
// 批次72 (卸载体验): 卸载询问页 — 用户资产默认保留
// ---------------------------------------------------------------------------
// 背景: Inno 默认只删安装清单里的文件, 运行期生成的用户数据留在 {app} 内
// → 既"卸不干净"也"重装得备份"。这里只问**真正的用户数据**:
//   - 声音档案 (data/voices: ref.wav + meta.json, 手录音 + 手工填的词)
//   - 训练权重 (GSV SoVITS/GPT 微调产物) 与 RVC 转换模型/检索索引
// 可再生成产物 (tts_cache / output / logs / TEMP) 已在 [UninstallDelete]
// 无条件清理, 不在此询问。静默卸载 (/SILENT 等) 一律按默认保留, 不弹窗。

// 目录存在且非空 ('.' '..' 不算内容)
function DirHasContent(const Dir: String): Boolean;
var
  FindRec: TFindRec;
begin
  Result := False;
  if not DirExists(Dir) then
    Exit;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
          Result := True;
      until Result or (not FindNext(FindRec));
    finally
      FindClose(FindRec);
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  VoicesDir, VitsDir, GptDir, RvcWeights, RvcLogs: String;
  HasVoices, HasWeights: Boolean;
  Msg: String;
begin
  if CurUninstallStep <> usUninstall then
    Exit;
  // 静默卸载: 无人可答 → 按默认 (保留) 走, 绝不擅自删用户数据
  if UninstallSilent then
    Exit;

  VoicesDir  := ExpandConstant('{app}\runtime\data\voices');
  VitsDir    := ExpandConstant('{app}\runtime\third_party\gpt_sovits\SoVITS_weights_v2');
  GptDir     := ExpandConstant('{app}\runtime\third_party\gpt_sovits\GPT_weights_v2');
  RvcWeights := ExpandConstant('{app}\runtime\third_party\rvc\assets\weights');
  RvcLogs    := ExpandConstant('{app}\runtime\third_party\rvc\logs');

  HasVoices  := DirHasContent(VoicesDir);
  HasWeights := DirHasContent(VitsDir) or DirHasContent(GptDir) or
                DirHasContent(RvcWeights) or DirHasContent(RvcLogs);
  if not (HasVoices or HasWeights) then
    Exit;

  Msg := '检测到你的用户数据：' #13#10#13#10;
  if HasVoices then
    Msg := Msg + '  · 声音档案（参考音频、档案配置）' #13#10;
  if HasWeights then
    Msg := Msg + '  · 训练权重 / RVC 转换模型' #13#10;
  Msg := Msg + #13#10 +
         '是否在卸载时一并删除？' #13#10 +
         '默认保留（推荐）：重装后可直接继续使用。' #13#10 +
         '选择「是」将永久删除，无法恢复。';

  if MsgBox(Msg, mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
  begin
    if HasVoices then
      DelTree(VoicesDir, True, True, True);
    if HasWeights then
    begin
      DelTree(VitsDir, True, True, True);
      DelTree(GptDir, True, True, True);
      DelTree(RvcWeights, True, True, True);
      DelTree(RvcLogs, True, True, True);
    end;
  end;
end;
