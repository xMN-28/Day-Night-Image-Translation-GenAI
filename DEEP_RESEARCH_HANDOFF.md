# Deep-Research Handoff: Day-to-Night Image Translation Project

## 1. What we are trying to build

We are a college team building a single-image day-to-night translator. A user uploads an
ordinary daytime photograph and the model should produce a convincing nighttime version while
preserving the scene: buildings, vehicles, trees, branches, wires, signs, road geometry, people,
and camera viewpoint must not move or be replaced.

The desired transformation is more than reducing exposure. It should handle:

- a genuinely nighttime sky rather than a gray/dim daytime sky;
- spatially varying illumination instead of uniform darkening;
- plausible illuminated windows, streetlights, traffic lights, and vehicle lamps;
- bloom, glare, wet-road reflections, and sensible shadow changes where appropriate;
- fine branches, wires, rooflines, text, and object boundaries;
- both road scenes and, eventually, general outdoor photographs;
- reproducible inference on one RTX 4070 Super with 12 GB VRAM;
- a local Gradio upload interface with sub-two-second 512 px inference.

The output is allowed to be one plausible night rather than the unknowable true future night.
However, it must not hallucinate large objects or disguise failure as a simple brightness change.

## 2. Hardware and verified runtime

- Windows PC
- NVIDIA RTX 4070 Super, 12 GB VRAM
- Approximately 32 GB system RAM
- Python 3.12.6
- Current verified runtime: PyTorch 2.3.1+cu121, torchvision 0.18.1+cu121
- Gradio 5.50.0
- BF16 mixed precision works on the GPU

Some older project documentation describes a planned PyTorch 2.12.1/CUDA 13 environment, but
the versions above are the versions actually reported by the active environment. Research and
reproduction advice must distinguish planned pins from the verified runtime.

The present models use far less than 12 GB. LumiRender peaked around 3.1 GB of PyTorch-allocated
memory and roughly 4 GB total GPU memory at 384 px. Memory is therefore not the primary problem;
the objective, supervision, and inductive bias are.

## 3. Starting paper and scope

The college supplied *Unpaired Day-to-Night Image Translation Using Deep Generative Model*, based
on CycleGAN/CycleGANAS ideas. We began with unpaired translation, but we are not required to stay
faithful to that paper. The professor expects a defensible architecture and original experimental
work rather than a superficial clone.

Originality should be claimed only for our integration, training strategy, evaluation, failure
analysis, and any genuinely new modules. CycleGAN, UNIT, VGG, DINOv2, PatchNCE, PatchGAN,
Gaussian illumination, and diffusion were invented elsewhere and must be attributed.

## 4. Data that is already prepared

### BDD100K

We downloaded BDD100K images and time-of-day labels. We retained strict `daytime` and `night`
labels and created balanced, duplicate-grouped splits:

| Split | Day | Night | Use |
|---|---:|---:|---|
| Train | 5,000 | 5,000 | Weight updates |
| Validation | 500 | 500 | Checkpoint selection |
| Test | 1,000 | 1,000 | Final evaluation only |

Perceptual duplicate grouping was performed before splitting. Corrupt files were rejected and
quality issues were flagged. The test set must not be used for tuning.

### Additional prepared data

- 2,416 Dark Zurich coarse/aligned day-night pairs with optical-flow confidence masks.
- 645 AMOS/timelapse day-night pairs with confidence masks.
- 400 ACDC night images used as unpaired target-domain examples.
- 5,000 cached depth/semantic pseudo-targets for factorization experiments.
- A 36-image difficult qualitative suite covering branches/wires, sky, glare, wet roads,
  vehicles, ordinary roads, landscapes, buildings, and indoor-window views.

Important discovery: many optical-flow confidence masks are effectively empty. Measured examples
from Dark Zurich had mean confidence around `0.0002–0.0006`; some were exactly zero. Textureless
sky is especially likely to be rejected by flow, which means the region most in need of a strong
day-to-night change received little or no paired pixel supervision. Some masks are healthy (for
example means around 0.08–0.12), so the dataset is not wholly unusable, but mask coverage must be
audited and redesigned before relying on paired losses.

The datasets and checkpoints are intentionally excluded from Git because of size and licensing.

## 5. Training and engineering infrastructure already built

- Atomic checkpointing of model, EMA, optimizers, schedulers, RNG state, step, and training state.
- Automatic best/latest checkpoint pointers.
- BF16, gradient clipping, gradient accumulation, replay buffers, and differentiable augmentation.
- TensorBoard plus understandable text monitoring.
- A Gradio training controller at port 7862 with **Start / Resume** and **Save & Stop**.
- Stop finishes the current optimizer step, saves it, and resume continues the same global step.
- A Windows status-file race was fixed and regression-tested.
- NaN detection, VRAM limit enforcement, validation, plateau handling, and failure diagnostics.
- A Gradio inference app at port 7860 with image upload, direction/model selection, before/after
  slider, maximum edge, metadata, and LumiRender seed/intensity/wetness controls.
- 26 local tests currently pass.

## 6. Model history and actual results

### 6.1 Local CycleGAN baseline implementation

The repository contains a standard baseline implementation:

- two nine-block ResNet generators;
- two 70×70 PatchGAN discriminators;
- LSGAN, cycle consistency, identity loss, and replay buffers.

A complete retained main-run CycleGAN checkpoint is not currently present, so do not invent
baseline results. There are only overfit/pilot artifacts. Any new comparison must either train the
baseline properly or clearly state that it is unavailable.

### 6.2 LumiCycle V1

Architecture:

- bidirectional nine-block ResNet generators, base width 64;
- channel/spatial attention;
- two-scale spectral-normalized PatchGAN discriminators;
- cycle loss 10, identity 5, PatchNCE 1, DINO semantic 0.5, Sobel edge 2, GAN 1;
- EMA inference, BF16, batch 1, gradient accumulation 2;
- generator LR `2e-4`, discriminator LR `1e-4`.

Training:

- trained to 40,000 optimizer steps;
- best validation checkpoint: step 13,000 (`best.json`);
- step-13k validation score was approximately 0.2306;
- final score around step 40k was worse, approximately 0.2814;
- discriminator accuracy around step 13k was about 0.989, indicating saturation;
- full run took roughly 5 hours 40 minutes on the local GPU.

Observed behavior:

- produces a clearly darker/night-like result;
- preserves broad road geometry reasonably well;
- can create convincing artificial-light appearance;
- sometimes leaves parts of the sky incompletely converted;
- often introduces red/cyan color fringing, halos, or branch artifacts;
- more training after 13k did not mean better quality.

Retained checkpoints include V1 13k and 40k.

### 6.3 Phase-two and LumiCycle V2

V2 was deliberately warm-started from useful V1 generator weights rather than starting over.
It reset saturated discriminators and targeted incomplete sky/global illumination.

Changes:

- global whole-frame discriminator in addition to local/multi-scale critics;
- upper-scene/regional illumination loss to pressure sky and global exposure;
- GAN 1, cycle 10, identity 2, PatchNCE 1, DINO semantic 0.75, edge 2.5,
  color statistics 0.5, illumination 2;
- generator LR `5e-5`, discriminator LR `2.5e-5`;
- discriminator updated once per two generator updates;
- fresh optimizers and critics; generator initialization transferred from V1.

Retained V2 checkpoints:

- best pointer: V2 step 4,500;
- final: V2 step 12,000.

Observed behavior:

- this is the strongest useful family so far;
- substantially more convincing nighttime lighting than LumiRender;
- better full-frame darkness and sky conversion than V1 in many examples;
- can still struggle around branches, wires, and other high-frequency detail;
- may retain or introduce some color casts and halos.

The next model should continue from the selected V2 generator weights. It should not discard this
training investment. New modules should be zero/identity initialized so their first output matches
the parent checkpoint.

Checkpoint-name ambiguity to resolve experimentally: the user recently referred to a good
“V2 5.5k” output. In the filesystem, V2's selected checkpoint is 4.5k, whereas 5.5k belongs to
V2.1. Before new training, run a blinded fixed-input comparison and record the exact SHA/path of
the parent instead of relying on conversational labels.

### 6.4 LumiCycle V2.1 — failed detail ablation

V2.1 began from V2 step 4,500 and attempted to fix branches without losing lighting:

- two-level Laplacian-pyramid high-frequency refiner;
- zero-initialized residual path;
- Haar LH/HL/HH frequency discriminators;
- normalized wavelet loss;
- spatial self-similarity loss;
- detail-biased sampling;
- 256 px refiner warm-up, then 384 px fine-tuning;
- very low LR for the V2 base.

Retained checkpoints include best 5,500 and final 6,000.

Observed result: it failed. The structure/frequency objectives overpowered the lighting objective,
and outputs drifted back toward daytime or weak twilight. It is retained only as a failed ablation
and should not be the default architecture unless a new experiment demonstrates otherwise.

### 6.5 LumiRender — failed physics-guided architecture

LumiRender was a completely new one-way day-to-night model trained from scratch, not a V2
continuation. The design tried to make image formation explicit:

- sRGB-to-linear conversion;
- scene factorizer predicting reflectance, day illumination, depth, normals, roughness, wetness,
  and soft masks for sky/road/vehicle/glass/building/emitter;
- eight stochastic anisotropic 2D Gaussian lights;
- ambient and horizon fields;
- approximate diffuse/specular terms;
- screen-space road/glass reflections and vertical light streaks;
- Gaussian bloom;
- camera exposure, white balance, tone map, vignette, and Poisson-Gaussian-like noise;
- correction network high-pass filtered and bounded to ±0.03;
- DINO semantic, paired photometric/perceptual, adversarial, reconstruction, factorization,
  geometry, emitter, physics-prior, and residual losses.

Training schedule:

1. Factorization/reconstruction to step 5,000 at 256 px.
2. Physics-synthesis pretraining to step 15,000 at 256 px.
3. Correspondence training to step 23,000 at 384 px.
4. Real-night refinement after step 23,000 at 384 px.
5. Manually stopped and safely saved at global step 34,004.

Engineering incidents:

- at the 5k transition, GAN/semantic backward initially produced NaN gradients at black pixels
  because of `sqrt(0)` and fractional-power tone-curve derivatives;
- the math was stabilized with epsilons and BF16 regression tests;
- exact stop/resume and monitoring worked afterward.

Observed result: LumiRender failed the actual task. Its output largely preserves geometry but acts
like a global brightness/desaturation transform. It does not convincingly replace the sky, create
spatially varying nighttime illumination, or use its learned Gaussian lights/reflections strongly.
It must not be presented as an improvement.

Why it failed:

- optical-flow supervision largely excludes textureless sky;
- a patch discriminator can accept global darkening as a cheap target-domain cue;
- emitter validity penalizes implausible lights, making weak/dim lights the easiest solution;
- reconstruction, reflectance, semantic, and structure constraints strongly reward unchanged
  content, while no equally strong objective demands a transformed sky;
- the bounded high-pass correction cannot change coarse illumination or replace the sky;
- explicit renderer ranges are restrictive and encourage a gray solution;
- the factorizer was frozen in the last stage, locking in earlier mistakes;
- “night intensity” currently multiplies radiance, so increasing it can brighten the image—its
  control semantics are wrong.

LumiRender 34,004 should be retained as an honest negative result. Do not spend time training it
at 640 px; that would make the same failure sharper rather than solve the objective.

## 7. Current qualitative ranking

Based on fixed collages and interactive tests:

1. A selected V2-family checkpoint is currently the practical parent for future work.
2. V1 13k/40k often looks more convincingly night than LumiRender but has color/halo artifacts.
3. V2.1 did not reliably retain nighttime transformation.
4. LumiRender preserves geometry but is not a credible night conversion.

No ranking should be treated as final until exact checkpoints are compared blind on the same fixed
suite. We have not yet completed a sufficiently rigorous FID/KID/structure/human-preference table.

## 8. Linked external repository that motivated this handoff

Repository: [solesensei/day2night](https://github.com/solesensei/day2night)

What is known from its public README:

- it is a bachelor-level image-to-image translation research project;
- it experiments with UNIT and CycleGAN;
- it uses BDD100K and Nexar/NEXET data;
- it is an older stack (Python 3.6, Ubuntu 18.04, PyTorch 0.4.1-era Docker images);
- it reports object-detector results and shows ablations involving VGG16 and normalization;
- the repository has hundreds of commits but is not a modern drop-in dependency.

The supplied screenshot crosses two independent choices:

- rows: no VGG16 perceptual component versus VGG16;
- columns: no network normalization, instance normalization, and layer normalization.

Visually, the VGG16 + no-normalization example looks promising: it produces strong local lights,
road reflections, and a complete sky/illumination change while retaining the scene.

Do not assume “no normalization” means feeding unnormalized tensors into VGG16. It more likely
refers to normalization layers in the generator/encoder. A pretrained VGG feature extractor still
normally requires its expected channel scaling/mean/std preprocessing. Inspect the implementation
and thesis before copying this setting.

Potential lesson worth testing: InstanceNorm removes per-instance channel mean and variance, which
may discard absolute illumination information central to day/night translation. A generator with
no normalization, carefully initialized residual scaling, weight/spectral normalization, or a
less destructive alternative may preserve content while allowing a stronger lighting change.

## 9. What the next deep research must do

Screen at least 100 relevant GitHub repositories and associated primary papers. Do not return a
list of fashionable names. Build a matrix containing:

- repository/paper and year;
- license and whether weights/data can legally be used;
- paired, unpaired, synthetic-paired, or diffusion-supervised training;
- exact architecture and normalization strategy;
- how sky and global illumination are transformed;
- how local emitters, bloom, shadows, and reflections are produced;
- how geometry/object preservation is enforced;
- whether code and pretrained weights actually run;
- training resolution, GPU memory, and estimated RTX 4070 Super feasibility;
- whether V2 weights can initialize it without destroying learned progress;
- evidence quality: cherry-picked images versus fixed quantitative evaluation;
- failure modes and maintenance status.

Search families should include, but not be limited to:

- UNIT/MUNIT/DRIT/CycleGAN/CUT and normalization ablations;
- perceptual VGG objectives and correct preprocessing;
- contrastive unpaired translation;
- illumination decomposition and intrinsic-image methods;
- sky-aware translation/relighting;
- local illumination and light-source synthesis;
- nighttime neural ISP and RAW/sRGB exposure modeling;
- diffusion image-to-image methods, ControlNet/adapters, one-step distillation, and low-VRAM
  students—but with clear disclosure of external pretrained capability;
- structure-preserving style transfer, correspondence, and identity-preserving residual adapters;
- histogram/wavelet/frequency methods only when they do not suppress the primary lighting change;
- normalization-free, weight-normalized, adaptive normalization, and direction-specific designs.

## 10. Questions the research output must answer

1. Why does the linked VGG16/no-normalization configuration appear stronger, and which exact code
   paths implement VGG features and network normalization?
2. Which normalization layers in V2 should be removed, retained, or replaced? Explain effects on
   absolute luminance, contrast, color, stability, and warm-start compatibility.
3. Can normalization be changed without discarding V2 weights? If not, propose identity-preserving
   adapters around existing blocks rather than a full reset.
4. What explicit supervision forces a daytime sky to become nighttime without changing branches?
5. How should low-confidence/textureless sky be supervised when optical flow is invalid?
6. How can local lights and reflections be encouraged without hallucinating lamps everywhere?
7. Should the next model remain unpaired, use coarse aligned pairs, create synthetic pairs, or use
   a frozen external teacher? State originality/attribution implications.
8. Would a VGG perceptual loss help, and at which layers/directions/weights? How should VGG inputs
   be normalized while leaving generator normalization independently configurable?
9. What is the smallest controlled architecture change likely to beat V2 on this hardware?
10. What three alternative architectures should be piloted, and what result would kill each pilot
    early rather than wasting another 34k steps?

## 11. Constraints for the proposed next model

- Start from the exact selected V2 generator and EMA weights whenever tensor shapes permit.
- Reset critics/optimizers when their domain or architecture changes.
- Zero/identity initialize new modules so step zero reproduces the parent output.
- Do not silently use the test split for checkpoint selection.
- Do not claim externally pretrained diffusion capability as our own architecture.
- Fit 12 GB VRAM with a safety ceiling around 11 GB.
- Begin with 256–384 px pilots; increase resolution only after the transformation itself works.
- Generate a fixed visual grid every 250–500 steps, including sky/branches/open landscape cases.
- Stop a pilot immediately if it merely darkens the image, weakens the sky transformation, moves
  objects, adds chromatic fringes, or loses against V2.
- Keep V2 unchanged as the fallback and default demo model until a blind gate is passed.

## 12. Recommended experimental protocol

Before a long run, select the exact V2 parent through a blind fixed-suite comparison. Record the
checkpoint path and SHA-256. Then run small, isolated ablations that change one idea at a time:

1. **Normalization/VGG pilot:** preserve V2 weights where possible; test the linked repository's
   normalization lesson and a correctly preprocessed frozen VGG perceptual loss.
2. **Sky-supervision pilot:** add a sky mask/critic and sky-specific target-distribution loss that
   does not depend on optical-flow confidence.
3. **Local-light pilot:** add a sparse residual illumination/emitter branch with positive examples
   and an anti-hallucination constraint.

Use 1,000–2,000 steps per pilot, fixed seeds, the same training subset, and identical evaluation
images. Compare against the unmodified V2 output at every checkpoint. Only combine modules after
each earns its place independently.

Required measurements:

- night-classifier confidence and luminance distributions;
- sky-region day/night classification and color/exposure statistics;
- DINO/LPIPS/edge/object-detector consistency;
- FID/KID on an untouched evaluation set, with limitations acknowledged;
- local-light precision/recall or semantic plausibility;
- human ratings for night realism, unchanged geometry, sky, lights/reflections, and artifacts;
- latency and peak VRAM.

## 13. Acceptance criteria for replacing V2

The next model is promoted only if it:

- is preferred over V2 on at least 70% of a fixed blind suite;
- converts the entire sky convincingly without erasing branches or wires;
- creates spatially varying night illumination rather than uniform darkening;
- preserves vehicles, buildings, people, signs, and road geometry;
- has no meaningful regression in edge, semantic, or detector consistency;
- works on road scenes and improves at least a held-out general-outdoor subset;
- stays below 12 GB VRAM and below two seconds for typical 512 px inference;
- produces repeatable results from a documented checkpoint;
- is presented with honest attribution and failed ablations.

## 14. Requested format from the external deep-research system

Return:

1. a ranked evidence matrix of 100+ repositories/papers;
2. a deep analysis of at least 25–30 strongest primary sources;
3. an audit of `solesensei/day2night`, especially VGG16 and normalization code;
4. three concrete warm-start-compatible candidate architectures;
5. exact modules, tensor insertion points, losses, weights, optimizers, schedules, and VRAM estimate;
6. a 2k-step pilot plan and early-kill criteria for each candidate;
7. a data/sky-mask/confidence strategy that fixes the measured supervision failure;
8. an evaluation and ablation table template;
9. risks, licensing/attribution requirements, and claims we may or may not make;
10. a final recommendation optimized for an RTX 4070 Super, not an unlimited research cluster.

The output must be critical. It should not accept the premise that a complicated architecture is
better. LumiRender demonstrated that plausible-sounding physics modules can collapse to trivial
darkening when the supervision and escape routes are wrong.
