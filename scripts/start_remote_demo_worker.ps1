param(
    [string]$Server = "https://lumicycle-remote-demo.vercel.app",
    [double]$PollSeconds = 1.0
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$workerSecret = [Environment]::GetEnvironmentVariable(
    "LUMICYCLE_WORKER_SECRET",
    "User"
)

if ([string]::IsNullOrWhiteSpace($workerSecret)) {
    throw "The worker secret is not installed for this Windows user. Re-run the remote demo setup."
}

$env:LUMICYCLE_REMOTE_URL = $Server
$env:LUMICYCLE_WORKER_SECRET = $workerSecret

Push-Location -LiteralPath $projectRoot
try {
    Write-Host "LumiCycle remote worker" -ForegroundColor Cyan
    Write-Host "Website: $Server"
    Write-Host "Keep this window open and keep the PC awake during the demonstration."
    python -m daynight.remote_worker --poll-seconds $PollSeconds
}
finally {
    Pop-Location
}
