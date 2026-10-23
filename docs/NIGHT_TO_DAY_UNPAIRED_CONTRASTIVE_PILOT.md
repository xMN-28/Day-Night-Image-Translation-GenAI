# Warm-start unpaired contrastive inverse pilot

22 October 2026. Experimental night-to-day only; public day-to-night unchanged.

## Question

Can the existing inverse adapt toward real daytime statistics without copying
the lighting defects in its external RGB pseudo-targets? Does patchwise
contrastive supervision preserve content better than an otherwise matched
adversarial-only continuation?

Motivation: [CUT official implementation and paper links](https://github.com/taesungp/contrastive-unpaired-translation).
This is our CUT-inspired objective adaptation, not a reproduction of CUT or
N2D3. It does not use the experimental photometric masks, optimal transport,
external restoration weights or FLUX outputs.

## Architecture and initialization

Both runs load the exact same EMA G_night_day at global step 16,000 from
`E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt`.
Parent SHA256: `29a8ecbc25a10fd3ee628fe4f0eca75a579755f1d512dca7cc58990fbb286a3b`.
The existing 9-block CBAM/InstanceNorm/transposed-convolution generator is
unchanged; all its trained weights are loaded. No context adapter is added.

New, training-only components:

- Three shared two-linear-layer ReLU projection heads, input widths 64/256/256,
  hidden/output width 128. Features are encoder indices 0, 2 and 5.
- Local PatchGAN plus global discriminator, base width 32, one local scale.
- Fresh Adam optimizers because this is a changed objective, not exact optimizer
  resumption of the supervised run.

Projection heads and discriminator are not needed for inference. Initial random
spectral-normalization vectors are stabilized with 20 zero-image, no-gradient
train-mode forwards before the first eval-mode discriminator call. No optimizer
updates occur during these forwards.

## Objective

`L_G = L_LSGAN + w * 0.5 * (PatchNCE(night,G(night)) + PatchNCE(day,G(day)))`.

Each level samples 128 corresponding spatial positions, projects and L2
normalizes them, and uses within-image dot-product classification at temperature
0.07. The same position is the positive; other sampled positions are negatives.
The source/key branch is stop-gradient. The translated/query path, including
its encoder, and projection heads receive gradients. The day term is identity
**contrastive** supervision, not day RGB identity loss.

Control uses w=0; candidate uses w=1. Both compute the same NCE path and consume
the same independently seeded patch RNG, so removing its gradient contribution
does not change the sampled training images or augmentations.

No teacher RGB, cycle, paired, semantic, smoothing, explicit denoising or
photometric-mask loss is present. Thus this experiment tests a different
objective, not a claimed complete solution for sensor noise or saturated glare.
The parent still historically learned from external targets; absence of an
external model in this run does not erase that provenance.

## Data and compute

- All 5,000 day and 5,000 night entries in the existing BDD TRAIN manifests.
- Independently sampled unpaired images; resize short side to 286, random 256
  crop and random horizontal flip; batch 1. Same Python/Torch seed 2035.
- Separate CUDA patch RNG 2036, BF16, G/D Adam 5e-5, betas (.5,.999), gradient
  clipping 3, generator EMA .99, 500 updates each.
- Hard peak-allocated-memory gate 11.5 GiB. Other live processes consume extra
  memory; allocated PyTorch memory is not total device usage.
- Atomic latest and numbered 250-step checkpoints contain generator, EMA,
  discriminator, projections, optimizers, Python/Torch/CUDA/patch RNG and recipe.
  This bounded script refuses an existing output directory; it does not yet
  expose a resume CLI or claim resume-equivalence verification.

## Reproduction

```powershell
.venv/Scripts/python.exe -m daynight.train_inverse_contrastive --parent E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt --output E:/LumiCycleCache/experiments/night2day_unpaired_GANcontrol_from16000_v3 --steps 500 --nce-weight 0
.venv/Scripts/python.exe -m daynight.train_inverse_contrastive --parent E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt --output E:/LumiCycleCache/experiments/night2day_unpaired_NCE_from16000_v3 --steps 500 --nce-weight 1
```

Use new output directories on reruns. Numbering continues to 16,500 rather than
discarding the parent's training history. Each run's recipe records parent and
code hashes. Fixed development comparisons, not random training crops, decide
visual acceptance. Test and reserved lockbox remain untouched.

## Engineering corrections before the matched runs

The initial v1 attempt failed before any update: selected feature level 2 has
256 channels, not 128. A generator/projection compatibility regression test was
added. The v2 control was interrupted early after detecting an abnormally large
first eval-mode adversarial loss from uninitialized spectral-normalization
vectors. Both matched v3 runs include the stabilization above. Earlier artifacts
were preserved, not used as controls or silently overwritten.

## Executed outcome — neither candidate accepted

Both v3 processes exited 0 after 500 updates, global step 16,500. Logged loop
times were 112.31 seconds for control and 112.22 seconds for NCE, with overlapping
GPU execution; these are not isolated throughput benchmarks. Peak allocated
memory was 1.187 GiB per process. Only the existing public worker pair remained
live after completion. No inverse training process was left running.

Checkpoints are `adaptation_00500.pt` in the two v3 directories above. Latest
files, optimizer states and numbered 250 checkpoints are retained. Checkpoint
loading and actual inference succeeded through the existing private inverse
loader without changing production registration.

Rendered comparison directory:
`outputs/night_to_day_experimental/20261022_unpaired_contrastive500`.
All six comparison sheets were opened, covering all 24 fixed development
sources. Columns are night source, clearly labeled external reference, retained
16k parent, GAN-only, and GAN+NCE. External references are display/diagnostic
comparators only and were not training inputs or targets for these two runs.

Additional native 512-pixel outputs opened: GAN+NCE indices 105 and 358, and
GAN-only index 0. These confirm rather than hide the failures visible in sheets.

| Variant | Direct visual observations | Decision |
|---|---|---|
| GAN-only | Strong repeated grid/striping texture, dark colored spots near lights, scene distortion and smeared structures across the suite | Reject |
| GAN+NCE | Better scene retention than GAN-only, but grain, cyan/green tint, persistent lamp glare and headlight-shaped road illumination; some upper scenes remain night-like | Reject |
| Retained parent | More conservative and generally cleaner, still known sky holes, glare and road pools | Keep as parent only; not quality-approved |

Specific NCE failures: 105's upper scene stays dark around structures rather
than becoming coherent daylight; 358 retains/amplifies the headlight-lit road
and has conspicuous colored noise in sky/foliage; 142/214 remain confused around
overhead structures; 8 retains an unsupported dark patch; 12 keeps patchy sky
and illuminated foliage; 283 has a night-like upper scene. These recurring
defects disqualify the candidate under the existing whole-image gate. No blind
team-rating percentage or new formal pass count is claimed.

External-reference RGB L1 (0–1) was .07505 parent, .21778 GAN-only, .12130 NCE.
This is **not** a daylight acceptance metric. NCE's final training loss .2358
likewise does not establish geometry preservation or realism.

### What this changes in our reasoning

Contrastive content preservation demonstrably matters relative to this
adversarial-only control. However, this formulation also preserves night-specific
structure/appearance and does not suppress noise. Removing RGB teacher anchors
does not itself fix the problem. Do not extrapolate that longer training will
necessarily repair it, or claim that all contrastive approaches are disproven
by a 500-update pilot.

The next distinct test must address which features should be preserved and
which illumination effects should change, with an explicit noise check. A
photometric-group contrastive adaptation remains a hypothesis; its imperfect
maps must not be used as hard object-removal masks. Better real paired
supervision remains a separate option subject to source/license approval.

### Verification

144 Python tests passed, including matching actual generator feature channels,
normalized projection sampling, corresponding-positive behavior and detached
key gradients. Relevant Ruff checks passed. These validate implementation
properties, not daylight realism. Test/lockbox untouched; public day-to-night
weights, routing, sky repair and worker unchanged. No checkpoint promoted.
