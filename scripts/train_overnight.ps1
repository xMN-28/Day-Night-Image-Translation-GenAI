param(
    [ValidateSet("cyclegan", "lumicycle")]
    [string]$Model = "lumicycle",
    [double]$Hours = 8
)

$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe -m daynight.train --config "configs/$Model.yaml" --max-hours $Hours --resume auto

