$ErrorActionPreference = "Stop"

& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\python.exe -m daynight.train --config configs\lumicycle.yaml --overfit --max-hours 1 --resume auto
Write-Host "Preflight passed. Inspect runs\overfit\lumicycle_bdd100k\tensorboard before starting the pilot."

