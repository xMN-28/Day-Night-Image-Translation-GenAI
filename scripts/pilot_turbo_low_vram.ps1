param(
    [int]$Steps = 20
)

$ErrorActionPreference = "Stop"
Write-Host "Turbo is an externally pretrained benchmark, not LumiRender."
Write-Host "First verifying that the optional reference can load within the 11.5 GB safety ceiling."
& .\.venv\Scripts\python.exe -m daynight.turbo_pilot --steps $Steps --max-vram-gb 11.5
