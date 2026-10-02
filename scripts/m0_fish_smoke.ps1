# M0 smoke test: Fish Speech inference (text -> speech)
# Run from repo root: powershell -ExecutionPolicy Bypass -File scripts/m0_fish_smoke.ps1
$ErrorActionPreference = "Stop"

$FishDir = Join-Path $PSScriptRoot "..\third_party\fish-speech"
$Python = Join-Path $FishDir ".venv\Scripts\python.exe"
$Checkpoint = Join-Path $FishDir "checkpoints\openaudio-s1-mini"
$OutputDir = Join-Path $PSScriptRoot "..\output"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Chinese TTS text passed via temp file to avoid console encoding issues
$TextFile = Join-Path $PSScriptRoot "m0_smoke_text.txt"
$Text = [IO.File]::ReadAllText($TextFile, [Text.Encoding]::UTF8)

Push-Location $FishDir
try {
    & $Python -m fish_speech.models.text2semantic.inference `
        --text $Text `
        --checkpoint-path $Checkpoint `
        --half `
        --output-dir $OutputDir
} finally {
    Pop-Location
}

Write-Host "Output dir: $OutputDir"
