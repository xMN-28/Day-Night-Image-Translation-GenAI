$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "external" | Out-Null
if (-not (Test-Path "external\img2img-turbo")) {
    git clone --depth 1 https://github.com/GaParmar/img2img-turbo.git "external\img2img-turbo"
}
& .\.venv\Scripts\python.exe -m pip install -e ".[turbo]"
Write-Host "Turbo reference installed. Its official weights download on first use."

