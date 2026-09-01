param(
    [string]$Checkpoint = "runs/lumirender_physics_bdd100k/checkpoints"
)

$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe -m daynight.evaluate `
    --checkpoint $Checkpoint `
    --output outputs/evaluation/lumirender `
    --image-size 512 `
    --detector
& .\.venv\Scripts\python.exe -m daynight.evaluate_lumirender --checkpoint $Checkpoint
