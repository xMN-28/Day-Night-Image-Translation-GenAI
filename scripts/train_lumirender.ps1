param(
    [double]$Hours = 8,
    [switch]$Pilot
)

$ErrorActionPreference = "Stop"
$arguments = @(
    "-m", "daynight.train_lumirender",
    "--config", "configs/lumirender.yaml",
    "--max-hours", $Hours,
    "--resume", "auto"
)
if ($Pilot) { $arguments += "--pilot" }
& .\.venv\Scripts\python.exe @arguments
