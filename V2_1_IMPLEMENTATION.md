# LumiCycle V2.1 — Detail-Preserving Fine-Tuning

## Purpose

V2 improved global darkness and sky conversion, but fine structures such as tree
branches could blur or merge. V2.1 targets that failure without discarding the useful
lighting behavior already learned by V2.

## What changed

- A two-level Laplacian-pyramid refiner predicts only high-frequency corrections.
- Its final layers are initialized to zero. Before training, V2.1 and its V2 parent
  therefore produce the same image within floating-point tolerance.
- Small spectral-normalized critics inspect fixed Haar LH, HL, and HH bands rather
  than judging only RGB appearance.
- A normalized wavelet loss preserves edges despite large brightness changes.
- A spatial self-similarity loss preserves local relationships between branches,
  wires, signs, vehicle outlines, and surrounding pixels.
- A residual penalty prevents the detail branch from needlessly repainting the
  protected V2 output.
- Thirty percent of training samples come from the upper quartile of an automatic
  high-detail score. The score emphasizes the upper part of each scene, where tree
  canopies, wires, poles, and skyline boundaries commonly appear.

These components are inspired by established frequency-domain adversarial training
and spatial-correlation preservation ideas, but their integration, staged migration,
sampling, monitoring, and validation are implemented locally for LumiCycle.

## Safe migration from V2

The required parent is:

```text
runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt
```

Use `--init-from`, not `--resume`, for the first V2.1 run. The loader migrates the old
generator state under the new `base` module, copies its EMA weights, restores the V2
spatial critics, leaves new modules freshly initialized, and starts new optimizers.
The parent file SHA-256 and source step are saved in `training_state`.

## Training schedule

| Stage | V2.1 steps | Crop | Updated parameters |
|---|---:|---:|---|
| Detail warm-up | 0–999 | 256 px | Laplacian refiners and Haar critics only |
| Detail fine-tune | 1,000–5,999 | 384 px | Refiners, V2 residual/decoder, spatial critics, Haar critics |

The V2 stem and downsampling encoder remain frozen throughout. All run counters are
V2.1-local: V2.1 step 0 still descends from V2 step 4,500.

## Commands

First run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m daynight.train --config configs/lumicycle_v2_1.yaml `
  --init-from runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt `
  --max-hours 8
```

Resume after any clean stop:

```powershell
python -m daynight.train --config configs/lumicycle_v2_1.yaml --resume auto --max-hours 8
```

Live text monitor:

```powershell
$env:LUMICYCLE_RUN_ROOT = "runs/lumicycle_v2_1_bdd100k"
python monitor.py
```

## What to watch

- `G/wavelet`: normalized fine-structure mismatch; its trend should fall.
- `G/self_similarity`: local structural mismatch; lower is better.
- `G/residual`: how far the refiner moves from V2. A sudden large rise is unsafe.
- `D/frequency_accuracy`: persistent values near 100% mean the detail critic is too
  strong; values near chance mean it is no longer providing much guidance.
- `translations/detail_input_coarse_refined`: the most direct visual check that
  branches improve while lighting stays consistent.

## Acceptance gate

V2.1 should be added to the showcase selector only after a fixed-image comparison
confirms all of the following:

1. Tree branches, wires, poles, and rooflines are cleaner than V2.
2. V2's convincing darkness and full-sky conversion are retained.
3. Cars, signs, buildings, and road geometry are not repainted or shifted.
4. Validation wavelet/self-similarity scores improve without a meaningful regression
   in cycle, edge, color, or illumination scores.
5. The chosen result is repeatable from an EMA checkpoint and fits the 12 GB GPU.

V2 remains the default app model until this gate passes.
