param(
    [string]$CacheRoot = "E:\LumiCycleCache",
    [int]$Edge = 512,
    [string]$Output = "outputs/turbo_reference_val"
)
$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace
$referencePython = Join-Path $CacheRoot "turbo-env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $referencePython)) {
    throw "Run scripts/setup_turbo_reference_isolated.ps1 first."
}
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = "$CacheRoot\turbo-env\Lib\site-packages;$workspace\.venv\Lib\site-packages"
    & $referencePython -m daynight.turbo_pilot --cache-root $CacheRoot --edge $Edge --output $Output
    if ($LASTEXITCODE -ne 0) { throw "External reference pilot failed; see traceback and retained download progress." }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
