# Night-target perceptual-feature pilot

21 October 2026. Status: pilot and matched control completed; no meaningful VGG win.

## Motivation and attribution

Pixel/footprint/sky objectives produce measurable progress but still leave smeared
lamps and artificial sky glow. A frozen feature extractor can compare local contrast
and texture over neighborhoods rather than only individual RGB values. This experiment
is inspired by [Johnson, Alahi and Fei-Fei, ECCV 2016](https://cs.stanford.edu/people/jcjohns/eccv16/),
not presented as a new algorithm or an exact reproduction of that paper.

The crucial distinction from an overly strong source-preservation loss is that
features are compared to the **nighttime pseudo-target**, not the daytime source.
The pseudo-target comes from the explicitly external CycleGAN-Turbo teacher and can
still hallucinate lights or alter text. Matching it is not proof of physical truth,
unchanged annotations, object retention or state-of-the-art quality.

## Implemented loss

`daynight/perceptual_targets.py` loads the official torchvision VGG16 ImageNet-1K
weights and retains feature layers through `relu3_3`. Loss is the average L1 distance
at `relu1_2`, `relu2_2`, and `relu3_3`. These are low/mid-level feature comparisons,
not an object detector or a whole-scene semantic understanding guarantee.

- Parameters are frozen; target features have no gradient.
- Gradients flow through predicted-image features back to the local generator.
- RGB [-1,1] becomes [0,1], then uses ImageNet mean/std input standardization.
- Aligned full spatial extents are retained: no 224-pixel classification crop.
- No InstanceNorm/LayerNorm is added to these feature outputs to erase contrast.
- The generator's existing normalization is **not removed**.
- Initial perceptual weight is 0.5. This is an experimental choice, not an optimum.
- The VGG classifier is never moved to GPU; it is not needed by this loss.

Official model metadata/preprocessing reference:
[torchvision VGG16 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html).
The installed project version was checked locally as torchvision **0.27.1+cu130**;
the online stable documentation may show a newer version. The model URL was obtained
from the installed `VGG16_Weights.IMAGENET1K_V1` enumeration.

## Offline setup, storage and integrity

```powershell
.\.venv\Scripts\python.exe -m daynight.perceptual_targets --prepare
```

Official URL: `https://download.pytorch.org/models/vgg16-397923af.pth`.
Location: `E:\LumiCycleCache\torchvision\vgg16-397923af.pth`.
Observed full SHA-256:
`397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`.
The download checks the official filename hash prefix; the full digest is recorded
in `provenance.json` and verified again when loading. Weight loading explicitly uses
`weights_only=True`. Training refuses a missing or changed local cache; it does not
silently download weights. No images are uploaded. Pretrained weights are external,
not a team invention, and are not redistributed with the project.

## State continuation and probe

The parent is the completed sky-target branch at global step 29,850:
`E:\LumiCycleCache\experiments\turbo_sky_target_weight2\step_01000.pt`.
The local RGB generator, EMA, Adam and RNG states are retained with
`--continue-state`; the backbone is not replaced by VGG or a diffusion model.
New frozen feature-model initialization happens before RNG restoration.

A real one-update BF16/batch-4/512-by-288 forward/backward completed at global step
29,851. Reported allocated VRAM was about 3.56 GiB; raw feature loss was 0.33195.
The saved update is resumed rather than repeated. All 65 unit tests pass, including
prediction-gradient, detached-target and frozen-feature tests. This is not proof of
image quality.

The actual new-objective restart audit also passed: two uninterrupted updates versus
one update plus save/restart plus one update, with a repeated uninterrupted control.
Maximum floating-state difference was **0.0**, over 260 floating tensors. RNG and
counters matched; final audit global step was 29,853. Original probe weights were
unchanged. Evidence:
`E:\LumiCycleCache\experiments\resume_audit_vgg_target\result.json`.
This is a short same-stack check, not a universal long-run guarantee.

## Running pilot and planned control

Run location: `E:\LumiCycleCache\experiments\turbo_vgg_target_pilot`.
The pilot is capped at 1,000 updates / global step 30,850 before visual review:

```powershell
.\.venv\Scripts\python.exe -m daynight.train_teacher_student --parent E:\LumiCycleCache\experiments\turbo_sky_target_weight2\step_01000.pt --targets E:\LumiCycleCache\teacher_targets\turbo_road2048_native512 --output E:\LumiCycleCache\experiments\turbo_vgg_target_pilot --steps 1000 --batch-size 4 --lr 0.0001 --highlight-weight 0.5 --footprint-weight 0.3 --semantic-weight 0 --sky-target-weight 2 --sky-matte runs/sky_student_regions_v2/best.pt --perceptual-target-weight 0.5 --continue-state --deterministic --keep-every 500 --max-minutes 20
```

Other losses and the dataset remain unchanged. VGG remains training-only, is not
stored repeatedly in generator checkpoints, and is not required for student inference.
The stable public inference API and demo default are unchanged.

A matched continuation **without** the VGG term is still needed to isolate the
perceptual objective from simply adding more updates. Its current status is recorded
below. Do not attribute any future improvement to VGG solely from comparison
with an earlier-step parent. Evaluate raw outputs, separate sky diagnostics, lights,
branches/wires, object/text artifacts and fixed comparison galleries before promotion.

## Completed pilot results, before matched-control comparison

The pilot finished at global step 30,850. All eight final held-out comparison panels
were visually inspected. The large sun/cloud glow is still present, some lamps remain
soft smears, and teacher geometry/text artifacts remain possible. **Not accepted or
promoted**, even though several metrics improve slightly against the earlier parent.

Same 48 validation sources, native resolution, latest EMA:

| Diagnostic | Parent at 29,850 | Perceptual pilot at 30,850 |
|---|---:|---:|
| Weighted teacher fidelity | 0.51192 | 0.51034 |
| Teacher-highlight L1 | 0.42036 | 0.41741 |
| Bright-pixel recall | 0.54366 | 0.56065 |
| Bright-pixel precision | 0.59191 | 0.59438 |
| Bright-pixel IoU | 0.40275 | 0.41270 |
| Extra-bright-pixel fraction | 0.002649 | 0.002818 |
| Separate-reference sky RGB L1 | 0.033209 | 0.033902 |
| Separate-reference excess sky luminance | 0.006776 | 0.007367 |

The sky diagnostics regress slightly, so reporting only improved bright-pixel IoU
would be misleading. These parent comparisons also include 1,000 extra updates;
they cannot isolate the feature loss's causal contribution.

Evidence: `outputs/sky_pilot/vgg_target_1000_validation/metrics.json` and `gallery.jpg`.

The matched no-VGG control completed in
`E:\LumiCycleCache\experiments\turbo_vgg_target_control`, not merely proposed.
It uses the same parent checkpoint, restored generator/EMA/optimizer/RNG state,
batch 4, sky weight 2, all other losses and 1,000-update cap. Its first fidelity loss
matched the VGG branch exactly at 0.4950892627. The command is the pilot command
above with the control output folder and without `--perceptual-target-weight 0.5`.
The final equal-update comparison is recorded in
`outputs/sky_pilot/vgg_target_matched_1000_validation/metrics.json`.

| Final metric | No VGG | VGG 0.5 |
|---|---:|---:|
| Weighted teacher fidelity ↓ | 0.5102956 | 0.5103407 |
| Bright-pixel IoU ↑ | 0.411474 | 0.412702 |
| Extra-bright fraction ↓ | 0.002836 | 0.002818 |
| Reference sky RGB L1 ↓ | 0.034206 | 0.033902 |
| Excess sky luminance ↓ | 0.007175 | 0.007367 |

Both finished at global 30,850. The tiny mixed differences do not demonstrate
a meaningful overall benefit from VGG. Sun/reflection glow and smeared lights
remain. Neither branch passed a broad visual acceptance gate; the no-VGG best
checkpoint is exposed only as an explicitly experimental student model option,
not silently promoted over the original V2 default.
