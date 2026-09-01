$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ParentCheckpoint = Join-Path $ProjectRoot "runs\lumicycle_v2_bdd100k\checkpoints\step_00004500.pt"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv. Run scripts\setup.ps1 first."
}
if (-not (Test-Path -LiteralPath $ParentCheckpoint)) {
    throw "Missing protected V2 parent checkpoint: $ParentCheckpoint"
}

& $Python -m daynight.train `
    --config configs\lumicycle_v2_1.yaml `
    --init-from $ParentCheckpoint `
    --max-hours 8
