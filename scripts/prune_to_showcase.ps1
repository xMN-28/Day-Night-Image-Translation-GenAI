param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$taskCacheRoot = (Resolve-Path -LiteralPath 'E:\LumiCycleCache').Path
$taskWeightsRoot = Join-Path $taskProjectRoot 'checkpoints\showcase'
$taskManifest = Get-Content -LiteralPath (Join-Path $taskWeightsRoot 'manifest.json') -Raw | ConvertFrom-Json
$taskEquivalence = @(Get-Content -LiteralPath (Join-Path $taskWeightsRoot 'inference_equivalence.json') -Raw | ConvertFrom-Json)
if ($taskManifest.models.Count -ne 10 -or $taskEquivalence.Count -ne 9) { throw 'Incomplete showcase verification' }
foreach ($taskRecord in $taskManifest.models) {
    $taskWeight = (Resolve-Path -LiteralPath $taskRecord.output).Path
    if (-not $taskWeight.StartsWith($taskWeightsRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected weight path' }
    if ((Get-FileHash -LiteralPath $taskWeight -Algorithm SHA256).Hash.ToLower() -ne $taskRecord.sha256) { throw 'Weight checksum mismatch' }
}
$taskLocalCommit = (& git -C $taskProjectRoot rev-parse HEAD).Trim()
$taskRemoteLine = & git -C $taskProjectRoot ls-remote origin refs/heads/main
if ($LASTEXITCODE -ne 0 -or ($taskRemoteLine -split '\s+')[0] -ne $taskLocalCommit) { throw 'Push source backup before pruning' }
$taskTargets = @()
foreach ($taskName in @('runs','data','datasets','tmp','.pytest_cache','.ruff_cache','__pycache__')) {
    $taskTargets += Join-Path $taskProjectRoot $taskName
}
foreach ($taskName in @('experiments','huggingface','official_models','kagglehub','torchvision',
    'teacher_targets','experiment_parents','turbo-env','img2img-turbo','klein-runtime','pip-cache','klein_huggingface')) {
    $taskTargets += Join-Path $taskCacheRoot $taskName
}
$taskTargets += Join-Path $taskProjectRoot 'remote-demo\.next'
foreach ($taskName in @('tools','torch_cache','diagnostics')) {
    $taskTargets += Join-Path $taskProjectRoot ('artifacts\' + $taskName)
}
Get-ChildItem -LiteralPath (Join-Path $taskProjectRoot 'outputs') -Directory | Where-Object {
    $_.Name -notin @('demo_showcase','night_to_day_showcase')
} | ForEach-Object { $taskTargets += $_.FullName }
$taskBefore = @{}
foreach ($taskDrive in @('D','E')) { $taskBefore[$taskDrive] = (Get-PSDrive $taskDrive).Free }
$taskReport = @()
foreach ($taskTarget in $taskTargets) {
    if (-not (Test-Path -LiteralPath $taskTarget)) { continue }
    $taskResolved = (Resolve-Path -LiteralPath $taskTarget).Path
    if ($taskResolved -in @($taskProjectRoot,$taskCacheRoot) -or
        (-not $taskResolved.StartsWith($taskProjectRoot+'\',[StringComparison]::OrdinalIgnoreCase) -and
         -not $taskResolved.StartsWith($taskCacheRoot+'\',[StringComparison]::OrdinalIgnoreCase))) {
        throw "Unsafe cleanup target: $taskResolved"
    }
    $taskInfo = Get-Item -LiteralPath $taskResolved -Force
    if ($taskInfo.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing reparse-point root: $taskResolved" }
    $taskBytes = (Get-ChildItem -LiteralPath $taskResolved -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    $taskAction = if ($Apply) { 'Removing ' } else { 'Audit ' }
    Write-Output ($taskAction + $taskResolved)
    if ($Apply) { Remove-Item -LiteralPath $taskResolved -Recurse -Force -ErrorAction Stop }
    $taskReport += [pscustomobject]@{Path=$taskResolved;LogicalBytes=$taskBytes;Removed=[bool]$Apply}
}
$taskFreed = @{}
foreach ($taskDrive in @('D','E')) { $taskFreed[$taskDrive] = [math]::Round(((Get-PSDrive $taskDrive).Free-$taskBefore[$taskDrive])/1GB,3) }
$taskResult = @{Applied=[bool]$Apply;SourceCommit=$taskLocalCommit;FreedGiB=$taskFreed;Targets=$taskReport}
$taskResult | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskWeightsRoot 'cleanup_report.json') -Encoding utf8
$taskFreed | ConvertTo-Json
