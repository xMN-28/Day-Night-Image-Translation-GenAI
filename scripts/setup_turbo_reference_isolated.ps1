param([string]$CacheRoot = "E:\LumiCycleCache")
$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace
$referenceCommit = "86f54146590ffb4543c8cf85b5a36657da670924"
$modelRevision = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
$source = Join-Path $CacheRoot "img2img-turbo"
$environment = Join-Path $CacheRoot "turbo-env"
New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null
if (-not (Test-Path -LiteralPath $source)) {
    git clone --no-checkout --depth 1 https://github.com/GaParmar/img2img-turbo.git $source
    if ($LASTEXITCODE -ne 0) { throw "Reference clone failed" }
    git -C $source fetch --depth 1 origin $referenceCommit
    if ($LASTEXITCODE -ne 0) { throw "Pinned reference fetch failed" }
    git -C $source checkout --detach $referenceCommit
}
if ((git -C $source rev-parse HEAD).Trim() -ne $referenceCommit) {
    throw "Existing reference checkout differs from the pinned commit. Inspect it; no reset performed."
}
if (-not (Test-Path -LiteralPath (Join-Path $environment "Scripts\python.exe"))) {
    & .\.venv\Scripts\python.exe -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw "Isolated environment creation failed" }
}
# Reuse existing CUDA torch without copying/downloading it. Isolated dependencies
# take precedence; the main environment is never a pip installation target.
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = "$environment\Lib\site-packages;$workspace\.venv\Lib\site-packages"
    & "$environment\Scripts\python.exe" -m pip install --no-deps --cache-dir "$CacheRoot\pip-cache" -r configs/turbo_reference_requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Isolated reference dependencies failed" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
$modelFiles = @(
    "README.md", "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors"
)
& .\.venv\Scripts\hf.exe download stabilityai/sd-turbo @modelFiles --revision $modelRevision --cache-dir "$CacheRoot\huggingface\hub"
if ($LASTEXITCODE -ne 0) { throw "Pinned base-model download failed; rerun to resume" }
& .\.venv\Scripts\hf.exe cache verify stabilityai/sd-turbo --revision $modelRevision --cache-dir "$CacheRoot\huggingface\hub"
if ($LASTEXITCODE -ne 0) { throw "Base-model checksum verification failed" }
Write-Host "Isolated reference ready. Run scripts/run_turbo_reference_pilot.ps1. Local model unchanged."
