# Showcase-only local installation

The local project was pruned on 24 October 2026 after the college project ended.

## Start a demonstration

1. Keep this home PC online and awake.
2. Double-click `start_remote_worker.bat`.
3. Open https://lumicycle-remote-demo.vercel.app/ on any internet-connected device.
4. Choose a direction/model, upload an example and translate.

For the local offline day-to-night interface, run `python app.py` using `.venv`.
`scripts/launch_demo.ps1` launches it from the project root. The remote worker's
Windows-user credential remains installed; it is never committed to Git.

## Retained locally

- `checkpoints/showcase/`: exact inference tensors for all nine remote model
  options plus the sky matte. `manifest.json` records hashes, original hashes,
  steps and export verification. These files are NOT stored on GitHub.
- `.venv/`: the working CUDA/Python environment.
- `outputs/demo_showcase/`: fourteen daytime inputs and genuine saved outputs.
- `outputs/night_to_day_showcase/`: ten curated experimental inverse examples.
- Existing final diagrams/PDFs, local study notes, college ZIPs in Downloads and
  presentation packs under `E:/LumiCycleCache/share_packages/`.
- Python/web source, requirements, configuration, tests, Git history, Vercel
  project link and ignored local web credentials.

## Removed

Training datasets and manifests, teacher targets, external reference downloads,
intermediate checkpoints, discriminators/optimizer states, research environment
caches, bulk experimental outputs and rebuildable frontend dependencies/builds.
The full training source remains in Git, but exact training resumption is no
longer available. Rerunning training requires new data/downloads and setup.

Inference exports are deliberately small and have the SAME selected weights and
global step labels; no training, quantization or architecture change occurred.
Their serialized file hashes differ from the original full training checkpoints.
Make a separate external backup of `checkpoints/showcase/` before deleting or
moving this project: a Git clone alone cannot recover these locally trained weights.

`docs/` is retained locally and ignored/untracked in the current GitHub tree.
Older Git commits still contain the previous documents; history was not rewritten.
Required attribution remains in `CREDITS.md` and `LICENSE`.
