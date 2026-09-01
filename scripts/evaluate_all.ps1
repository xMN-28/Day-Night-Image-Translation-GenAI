$ErrorActionPreference = "Stop"

& .\.venv\Scripts\python.exe -m daynight.evaluate --checkpoint runs\cyclegan_bdd100k\checkpoints --output outputs\evaluation\cyclegan --detector
& .\.venv\Scripts\python.exe -m daynight.evaluate --checkpoint runs\lumicycle_bdd100k\checkpoints --output outputs\evaluation\lumicycle --detector
Write-Host "Evaluation complete. Compare the metrics.json files under outputs\evaluation."

