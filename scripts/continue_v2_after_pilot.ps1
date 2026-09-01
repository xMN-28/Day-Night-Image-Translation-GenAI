param(
    [double]$Hours = 5
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectDirectory ".venv\Scripts\python.exe"
$pilotDirectory = Join-Path $projectDirectory "runs\pilots\lumicycle_v2_bdd100k"
$pilotPointer = Join-Path $pilotDirectory "checkpoints\latest.json"
$deadline = [DateTime]::UtcNow.AddHours($Hours)

Set-Location -LiteralPath $projectDirectory
Write-Output "Waiting for the V2 pilot to finish before continuing its complete state."

while ([DateTime]::UtcNow -lt $deadline) {
    $pilotStep = 0
    if (Test-Path -LiteralPath $pilotPointer) {
        $pilotStep = [int]((Get-Content -LiteralPath $pilotPointer -Raw | ConvertFrom-Json).step)
    }
    if ($pilotStep -ge 2000) {
        break
    }

    $pilotActive = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match "daynight\.train" -and
        $_.CommandLine -match "lumicycle_v2\.yaml" -and
        $_.CommandLine -match "--pilot"
    }
    if (-not $pilotActive -and $pilotStep -gt 0) {
        Write-Warning "Pilot stopped early at checkpoint step $pilotStep; continuing from it."
        break
    }
    Start-Sleep -Seconds 15
}

if (-not (Test-Path -LiteralPath $pilotPointer)) {
    throw "The V2 pilot did not produce a resumable checkpoint."
}

$pointer = Get-Content -LiteralPath $pilotPointer -Raw | ConvertFrom-Json
$pilotCheckpoint = Join-Path (Split-Path -Parent $pilotPointer) $pointer.filename
$remainingHours = ($deadline - [DateTime]::UtcNow).TotalHours
if ($remainingHours -le 0) {
    Write-Output "The five-hour training window ended during the pilot."
    exit 0
}

Write-Output "Continuing V2 from pilot step $($pointer.step) for up to $([math]::Round($remainingHours, 2)) more hours."
& $pythonExecutable -m daynight.train `
    --config configs/lumicycle_v2.yaml `
    --resume $pilotCheckpoint `
    --max-hours $remainingHours

if ($LASTEXITCODE -ne 0) {
    throw "V2 continuation exited with code $LASTEXITCODE."
}
