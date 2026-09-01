param(
    [int]$Port = 7862,
    [switch]$Start
)

$ErrorActionPreference = "Stop"
$arguments = @("-m", "daynight.training_control", "--port", $Port)
if ($Start) { $arguments += "--start" }
& .\.venv\Scripts\python.exe @arguments
