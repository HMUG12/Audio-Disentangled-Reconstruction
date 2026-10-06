#Requires -Version 5.1
# =====================================================================
# Convert license.source.txt (UTF-8) -> license.rtf for NSIS license
# page. RTF is ASCII-only by spec: every non-ASCII char is emitted as
# \uN? (signed 16-bit decimal), \ { } are escaped. Re-run this script
# after editing the source txt (e.g. when the final OSS license lands).
# =====================================================================
$ErrorActionPreference = "Stop"
$src = Join-Path $PSScriptRoot "..\desktop\src-tauri\resources\license.source.txt"
$dst = Join-Path $PSScriptRoot "..\desktop\src-tauri\resources\license.rtf"

$text = [IO.File]::ReadAllText($src, [Text.Encoding]::UTF8) -replace "`r`n", "`n"

$sb = New-Object Text.StringBuilder
foreach ($ch in $text.ToCharArray()) {
    $code = [int]$ch
    if ($ch -eq "\")        { [void]$sb.Append("\\") }
    elseif ($ch -eq "{")    { [void]$sb.Append("\{") }
    elseif ($ch -eq "}")    { [void]$sb.Append("\}") }
    elseif ($ch -eq "`n")   { [void]$sb.Append("\par`r`n") }
    elseif ($code -lt 128)  { [void]$sb.Append($ch) }
    else {
        $signed = if ($code -gt 32767) { $code - 65536 } else { $code }
        [void]$sb.Append(("\u{0}?" -f $signed))
    }
}

# \fs22 = 11pt; YaHei covers CJK on every modern Windows
$rtf = "{\rtf1\ansi\deff0{\fonttbl{\f0\fnil\fcharset134 'Microsoft YaHei';}}`r`n" +
       "\f0\fs22`r`n" + $sb.ToString() + "`r`n}"
[IO.File]::WriteAllText($dst, $rtf, [Text.Encoding]::ASCII)
Write-Host ("license.rtf written: {0} chars source -> {1} bytes rtf" -f $text.Length, (Get-Item $dst).Length)
