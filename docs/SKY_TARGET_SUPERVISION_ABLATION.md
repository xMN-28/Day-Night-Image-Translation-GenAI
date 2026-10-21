# Night-target sky supervision: matched continuation ablation

21 October 2026. Status: running; no image-quality acceptance claimed.

## Completed full-data control

The full-data, no-GAN/no-source-DINO control completed 4,000 updates at global step
28,850. Its starting point was the trained native-resolution generator, not the
32-image memorization branch. All eight final four-column validation panels were
visually inspected. Some light footprints improve, but the sun/cloud scene retains
an elongated white glow and several scenes retain daylight shadow cues. It is not
accepted as convincing nighttime conversion and does not replace the demo.

Same native-resolution cache, 48 validation sources:

| Diagnostic | Parent at 24,850 | Control at 28,850 |
|---|---:|---:|
| Weighted teacher fidelity | 0.57142 | 0.52571 |
| RGB L1, [-1,1] units | 0.04713 | 0.04339 |
| Teacher-highlight L1 | 0.45158 | 0.39187 |
| Bright-pixel recall | 0.54039 | 0.55447 |
| Bright-pixel precision | 0.52549 | 0.58693 |
| Bright-pixel IoU | 0.36714 | 0.40157 |
| Soft light-footprint error | 0.77378 | 0.55283 |
| Extra-bright-pixel fraction | 0.003723 | 0.002861 |

Evidence: `outputs/sky_pilot/turbo_full_supervised_4000_validation/`. The immutable
parent for subsequent branches is
`E:\LumiCycleCache\experiments\turbo_student_full_supervised_native512\step_04000.pt`.

## Hypothesis

Whole-image RGB loss averages a remaining sky/glare defect together with many
already-dark pixels. Region-normalized **night-target** supervision may correct
those defects more directly without increasing source-edge preservation everywhere.
This does not assume that the hypothesis will succeed.

The supplied other-AI formula `abs((G(x)-x)*A_sky)` would preserve daytime pixels.
We instead compare `G(x)` with the **external teacher's nighttime pseudo-target**.
The teacher's possible hallucinations and altered text remain limitations.

## Implemented objective

`masked_target_fidelity()` in `daynight/teacher_objectives.py` computes RGB L1 to
the detached night target, weighted by a detached, clamped soft sky alpha. Each
image is normalized by its alpha mass, at least 64 pixels. Tiny uncertain regions
therefore cannot dominate. No-sky images contribute zero to this additional term.
The mask cannot shrink itself to avoid the loss; it is frozen.

The mask predictor is the existing locally trained `SkyStudent` at
`runs/sky_student_regions_v2/best.pt`, frozen, with its checkpoint SHA recorded in
experiment provenance. Its segmentation-teacher origins remain attributed. The
existing guided refinement handles mixed vegetation/sky pixels. This model is
training-only in this experiment: the RGB generator architecture and inference
dependencies do not change. No sky pixels are analytically replaced during training
or raw-generator evaluation.

Unit tests verify that a bright prediction over a dark nighttime target receives a
gradient that darkens it, that pixels outside the mask receive no gradient from this
term, and that the mask and target receive no gradients. At this ablation's implementation, 63 tests passed;
this is an engineering check, not proof of visual success.

## Matched continuation design

Both branches start at exactly the same saved raw generator, EMA, Adam state and
RNG state from global step 28,850. Both retain all trained parameters; no trained
backbone or optimizer is reset. Frozen matte initialization occurs **before** RNG
restoration, so it does not alter source sampling or augmentation. Their first
fidelity losses matched exactly: 0.5031659007, an additional initialization check.
At the immutable 500-update milestones, both global counters were 29,350, parent
hashes matched, and Python, CPU-Torch and CUDA RNG states were identical. This
supports the intended matched-sampling control rather than merely assuming it.

- Control: existing objective only, sky-target weight 0.
- Ablation: the same objective plus sky-target weight 2.

Both use the same 2,048 training / 48 validation images, batch 4, learning rate 1e-4,
deterministic BF16, highlight weight 0.5, footprint weight 0.3, no source-DINO or GAN,
and 1,000 continuation updates. Validation remains plain weighted teacher fidelity,
not a redefined selection score. Milestones are retained every 500 updates.

Run directories:

```text
E:\LumiCycleCache\experiments\turbo_sky_target_control
E:\LumiCycleCache\experiments\turbo_sky_target_weight2
```

The common command is:

```powershell
.\.venv\Scripts\python.exe -m daynight.train_teacher_student --parent E:\LumiCycleCache\experiments\turbo_student_full_supervised_native512\step_04000.pt --targets E:\LumiCycleCache\teacher_targets\turbo_road2048_native512 --output E:\LumiCycleCache\experiments\turbo_sky_target_control --steps 1000 --batch-size 4 --lr 0.0001 --highlight-weight 0.5 --footprint-weight 0.3 --semantic-weight 0 --continue-state --deterministic --keep-every 500 --max-minutes 20
```

For the ablation, change the output folder to `turbo_sky_target_weight2` and add
`--sky-target-weight 2 --sky-matte runs/sky_student_regions_v2/best.pt`.
The first matched 250-update validation was 0.52411 for the control and 0.52044
for sky weighting. This is preliminary and cannot establish acceptance.

## Separate sky-region diagnostic

`evaluate_teacher_student.py --sky-reference-cache data/sky_student_fullframe_v1`
uses cached frozen segmentation-teacher alpha maps, **not** the predicted training
masks. These pseudo-labels are imperfect and not human ground truth; both predictors
share a teacher lineage. Metadata source identity and the actual cached RGB pixels
are verified against the current source, including preprocessing, before a mask is
used. The live check passed for all 48 validation records; 46 contain nontrivial sky.

It reports region-normalized RGB error to the night target and positive excess
luminance over that target (with a 0.05 tolerance in [0,1]). Genuine bright lights
present in the target are therefore not automatically treated as errors. Means include
all 48 images; no-sky cases contribute zero. This does not prove physical correctness,
sky realism or fine-branch preservation. The full fixed galleries must still be reviewed.

Do not evaluate a changing `latest.pt` and record a later file hash as if it matched
the earlier loaded weights. Use immutable milestone files or terminal runs.

## Completed matched 1,000-update result

Both processes finished at global step 29,850. Immutable `step_01000.pt` files
were compared over all 48 validation sources. All eight final panels were visually
inspected; the elongated sun/cloud glow persists. Neither branch passes acceptance.

| Diagnostic | Matched control | Sky target weight 2 |
|---|---:|---:|
| Weighted teacher fidelity | 0.51628 | 0.51192 |
| Teacher-highlight L1 | 0.39873 | 0.42036 |
| Bright-pixel recall | 0.55395 | 0.54366 |
| Bright-pixel precision | 0.58689 | 0.59191 |
| Bright-pixel IoU | 0.40314 | 0.40275 |
| Extra-bright-pixel fraction | 0.002843 | 0.002649 |
| Separate-reference sky RGB L1, [0,1] | 0.035736 | 0.033209 |
| Separate-reference excess sky luminance | 0.010388 | 0.006776 |

Sky weighting reduces excess sky luminance about 35% and sky RGB error about 7%,
but slightly worsens highlight fidelity/recall and leaves major visual artifacts.
It is retained as a measured experimental improvement, **not promoted** to the
showcase or represented as a complete solution.

Evidence: `outputs/sky_pilot/sky_target_matched_1000_validation/metrics.json` and
`gallery.jpg`. The next training-only feature-loss experiment is documented in
[NIGHT_TARGET_PERCEPTUAL_PILOT.md](NIGHT_TARGET_PERCEPTUAL_PILOT.md).
