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
#define RuntimeDir "..\..\build\" + RUNTIME_DIR_NAME

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

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; \
    Flags: nowait postinstall skipifsilent

[Code]
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
