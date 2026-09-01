param(
    [ValidateRange(1, 24)]
    [double]$Hours = 8
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "[1/3] Running LumiRender preflight..." -ForegroundColor Cyan
& $python -m daynight.preflight_lumirender
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/3] Running the isolated 500-step pilot..." -ForegroundColor Cyan
& $python -m daynight.train_lumirender `
    --config configs/lumirender.yaml `
    --max-hours $Hours `
    --resume auto `
    --pilot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$remaining = [Math]::Max(0.25, $Hours - $stopwatch.Elapsed.TotalHours)
Write-Host ("[3/3] Pilot passed. Starting the main resumable run for up to {0:N2} hours..." -f $remaining) -ForegroundColor Green
& $python -m daynight.train_lumirender `
    --config configs/lumirender.yaml `
    --max-hours $remaining `
    --resume auto
exit $LASTEXITCODE
