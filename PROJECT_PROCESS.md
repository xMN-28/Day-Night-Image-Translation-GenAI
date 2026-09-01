# How We Built LumiCycle

This is the plain-language record of the complete project: what we built, why each part exists, what actually happened during training, and how another student can reproduce it.

## 1. The goal

LumiCycle translates road-scene photographs in both directions:

- **Day → Night:** change lighting, sky, shadows, and artificial lights while preserving roads, vehicles, buildings, and signs.
- **Night → Day:** brighten the scene naturally without changing its geometry or inventing unrelated objects.

The training data is **unpaired**. A daytime image does not need a matching photograph of the same place at night. This makes the task practical, but also difficult: the model must learn the target appearance without being shown an exact before/after answer.

## 2. What we started from

The college supplied the paper *Unpaired Day-to-Night Image Translation Using Deep Generative Model*. We used its CycleGAN/CycleGANAS direction as the starting point, then designed a more rigorous local experiment.

CycleGAN itself was not invented by this team. Neither were attention, PatchNCE, DINOv2, or PatchGAN. Our contribution is the way these methods are combined in LumiCycle, the leakage-resistant data pipeline, the training safeguards, evaluation, offline demo, and the comparison with a locally trained baseline.

## 3. Hardware and software used

The main run was completed on:

- NVIDIA RTX 4070 Super with 12 GB VRAM
- 32 GB system RAM
- Windows
- Python 3.12.6
- PyTorch 2.12.1 with CUDA 13.0
- torchvision 0.27.1
- Gradio for the offline web demo
- TensorBoard and the included text monitor for live training progress

This GPU is enough for batch-size-one training at 256 px and fast 512–768 px inference. It is not realistic to promise research-lab state-of-the-art quality from one consumer GPU, but it is enough to produce a convincing college project and run controlled improvements.

## 4. Dataset preparation

We used BDD100K driving images and their `timeofday` labels. The downloaded archive was about 5.5 GB. The preparation tool recursively found the images and labels, then accepted only samples marked exactly `daytime` or `night`.

The available labeled pool contained approximately:

- 42,058 daytime candidates
- 31,957 nighttime candidates

From this pool the tool made balanced splits:

| Split | Day | Night | Purpose |
|---|---:|---:|---|
| Train | 5,000 | 5,000 | Update model weights |
| Validation | 500 | 500 | Select checkpoints and tune decisions |
| Test | 1,000 | 1,000 | Final evaluation only |

Perceptual hashes group visually duplicated or nearly duplicated images before splitting. This reduces the chance that almost the same scene appears in training and testing. Corrupt files are rejected. Very blurred or underexposed files are reported for review instead of silently changing the untouched test set.

BDD100K is not stored in this repository. Anyone reproducing the work must accept its license and download it separately.

## 5. Models

### CycleGAN baseline

The local baseline has:

- Two 9-block ResNet generators, one for each direction
- Two 70×70 PatchGAN discriminators
- Least-squares GAN loss
- Cycle-consistency loss
- Identity loss
- Replay buffers for generated images

This is the fair locally trained reference. It should not be confused with CycleGAN-Turbo, which is an optional externally pretrained reference.

### LumiCycle

LumiCycle keeps the bidirectional CycleGAN foundation and adds:

- Channel and spatial attention in the generator bottleneck
- Two-scale spectral-normalized PatchGAN discriminators
- PatchNCE contrastive loss to retain local content
- Frozen DINOv2-S features to preserve scene meaning
- Sobel edge loss to preserve lanes, cars, signs, and building boundaries
- Exponential-moving-average generator weights for steadier validation and inference
- Differentiable discriminator augmentation to reduce overfitting

The first run used these loss weights:

| Loss | Weight | What it encourages |
|---|---:|---|
| Adversarial GAN | 1.0 | Look like the target domain |
| Cycle | 10.0 | Reconstruct the original scene |
| Identity | 5.0 | Avoid unnecessary changes |
| PatchNCE | 1.0 | Keep corresponding local content |
| DINO semantic | 0.5 | Keep overall scene meaning |
| Edge | 2.0 | Keep structural boundaries |

## 6. Safety checks before the main run

We did not begin with a blind overnight run. The pipeline was checked in stages:

1. Unit and shape tests checked model outputs, losses, gradients, configuration, data splits, and checkpoint behavior.
2. A 32-image overfit test ran for 1,000 steps and completed in about 10 minutes 34 seconds.
3. A separate 2,000-step pilot completed in about 19 minutes 19 seconds and confirmed VRAM stability and throughput.
4. The full run then trained for 40,000 optimizer steps in about 5 hours 40 minutes.

Training used BF16 mixed precision, batch size 1, gradient accumulation 2, Adam, gradient clipping, replay buffers, and TTUR learning rates (`G=2e-4`, `D=1e-4`). Atomic checkpoints allow a stopped overnight run to resume safely.

## 7. What happened in the first 40,000 steps

The best validation score was reached at **step 13,000**, with a score of approximately **0.2306**. The score later worsened and ended around **0.2814** at step 40,000. The demo therefore correctly loads the best 13k checkpoint, not simply the last checkpoint.

This is a normal machine-learning result: more steps do not always produce a better model. At 13k, discriminator accuracy was about 0.989, which indicates that the discriminator had become too confident. Later, the semantic loss worsened even while cycle and edge reconstruction improved. In plain language, the system became better at reconstructing structure but was no longer becoming better at the intended appearance/meaning balance.

The original plateau recovery also halved learning rates more than once. By step 40k the scheduler had reached zero, so continuing that same optimizer state would not produce useful learning. The phase-two configuration fixes this by allowing one plateau reduction and initializing a fresh optimizer from the best model weights.

## 8. Current result

The screenshot below is a real local inference from the selected step-13,000 LumiCycle checkpoint. The road layout, vehicles, and buildings remain recognizable while the illumination is changed substantially.

![LumiCycle demo result](docs/assets/lumicycle-demo-result.png)

One measured 512 px test inference took about **0.326 seconds** on the RTX 4070 Super. Results vary by image. Difficult cases include severe glare, extremely dark inputs, unusual weather, and scenes unlike BDD100K.

## 9. Exact reproduction steps

### Install

Open PowerShell in the repository:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
python -m daynight.doctor
```

### Prepare BDD100K

Download BDD100K images and labels after accepting its terms, extract them, then run:

```powershell
python -m daynight.prepare_data --bdd-root "D:\path\to\bdd100k"
```

The resulting manifests and split data are written under `data/`, which Git intentionally ignores.

### Run checks

```powershell
python -m pytest -q
python -m daynight.train --config configs/lumicycle.yaml --overfit --resume none
python -m daynight.train --config configs/lumicycle.yaml --pilot --resume none
```

### Train and resume

Start an eight-hour session:

```powershell
python -m daynight.train --config configs/lumicycle.yaml --max-hours 8 --resume auto
```

If the computer is stopped, run the same command the next night. `--resume auto` restores the latest complete checkpoint. The best validation checkpoint is tracked separately in `runs/lumicycle_bdd100k/checkpoints/best.json`.

### Watch understandable live progress

In a second PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
python monitor.py
```

Open `http://127.0.0.1:7861`. TensorBoard remains available for detailed charts:

```powershell
tensorboard --logdir runs --port 6006
```

To monitor the isolated phase-two pilot, set its run directory before launching:

```powershell
$env:LUMICYCLE_RUN_ROOT = "runs/pilots/lumicycle_phase2_bdd100k"
python monitor.py
```

### Test the model

```powershell
python app.py
```

Open `http://127.0.0.1:7860`, upload an image, select a direction, and press **Translate**. The app automatically prefers the best checkpoint and can work offline once weights are cached.

### Evaluate

```powershell
python -m daynight.evaluate --checkpoint runs/lumicycle_bdd100k/checkpoints/step_00013000.pt
```

Final reporting should include FID/KID, DINO feature distance, edge consistency, detector consistency, fixed qualitative grids, peak VRAM, and inference latency. The test split must never be used to select a checkpoint.

## 10. Phase-two improvement plan

The next run should begin from the 13k model weights with fresh optimizers. It should not resume the completed 40k optimizer state. The targeted changes are:

1. Limit automatic plateau learning-rate reduction to one occurrence.
2. Lower the discriminator learning rate so it cannot dominate as quickly.
3. Lower identity weight slightly, allowing a stronger lighting change.
4. Keep or slightly strengthen semantic and edge protection so vehicles and geometry remain stable.
5. Add target-domain color/luminance statistics to both training and validation, because the first validation score measured structure more strongly than nighttime realism.
6. Run a short isolated pilot, compare fixed images and validation metrics with the 13k checkpoint, and continue only if it wins.
7. Try 384 px refinement only after the 256 px phase-two experiment is demonstrably better and VRAM-safe.

A useful experiment changes one controlled group of settings and records the result. Randomly extending training or changing many unrelated components at once makes it impossible to explain why the result changed.

## 11. What is and is not on GitHub

Included:

- Source code, configs, tests, scripts, documentation, and the supplied result screenshot

Excluded:

- BDD100K data and labels
- Python virtual environment and caches
- Training logs and generated artifacts
- `.pt`, `.pth`, `.ckpt`, and `.safetensors` model files
- The supplied research-paper PDF, which should be obtained through its legitimate source

The best checkpoint is roughly 538 MB, so it is intentionally kept outside ordinary Git history. A future public model release can use a clearly labeled GitHub Release or model host after licensing and attribution are reviewed.

## 12. Limitations and responsible presentation

- LumiCycle is a strong student implementation, not a claim of a new state of the art.
- Unpaired translation can hallucinate lights, change small details, or fail on out-of-distribution scenes.
- Outputs should not be presented as evidence of real events or used for safety-critical driving decisions.
- CycleGAN-Turbo, if shown, must always be labeled externally pretrained.
- Every reported result should identify its checkpoint, data split, resolution, and whether it is locally trained or externally pretrained.

That honest framing still leaves a substantial original contribution: a complete reproducible system, a thoughtful architecture/loss combination, clean experimental splits, real failure analysis, and a working offline showcase.
