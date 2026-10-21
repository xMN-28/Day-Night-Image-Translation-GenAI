# Official reference and potential teacher evaluation

Date: 21 October 2026. Not a claim of an accepted local-model improvement.

## Purpose and attribution

The sky/matte and resize-decoder work improved specific defects, but the small paired
lighting pilots did not establish convincing, spatially coherent night lighting. We
therefore evaluate the official pretrained CycleGAN-Turbo as a **separately labeled
reference** before deciding whether a training-only distillation experiment is justified.
Reference outputs alone do not satisfy the goal of improving the locally trained model.

Primary sources:

- [Official img2img-turbo repository](https://github.com/GaParmar/img2img-turbo).
- [One-Step Image Translation with Text-to-Image Models, arXiv:2403.12036](https://arxiv.org/abs/2403.12036).
- [Pinned SD-Turbo model card](https://huggingface.co/stabilityai/sd-turbo/blob/b261bac6fd2cf515557d5d0707481eafa0485ec2/README.md).

This model uses externally pretrained diffusion knowledge, trained adapters, and VAE
skip connections. None is represented as a team invention or a continuation of V2's
training. If a student is later distilled, its teacher dependence must remain explicit.
The repository code has an MIT license; model-weight terms are separate and must be
checked against the upstream model card/license. Do not assume MIT code licensing
automatically licenses every pretrained weight for every purpose.

## Storage and isolation

The user explicitly selected **E:**, not the external H: drive. Existing project files,
datasets, and trained models remain in place. No cleanup/deletion was performed.

```text
E:\LumiCycleCache\img2img-turbo\          pinned official source checkout
E:\LumiCycleCache\turbo-env\              isolated older reference dependencies
E:\LumiCycleCache\huggingface\hub\        selected SD-Turbo snapshot
E:\LumiCycleCache\official_models\        official day2night checkpoint and range manifest
E:\LumiCycleCache\experiments\            new controlled experiment artifacts
```

Source commit: `86f54146590ffb4543c8cf85b5a36657da670924`.

Base model: `stabilityai/sd-turbo`, revision
`b261bac6fd2cf515557d5d0707481eafa0485ec2`.

The Hugging Face CLI workflow was used to select/download and checksum-verify the
13 files required for inference (approximately 5.2 GB). Its notice about nine missing
remote files concerns intentionally unselected alternative artifacts, not failed
verification of the downloaded files. This is not a claim that the entire repository
was downloaded or verified.

The upstream day-to-night adapter/weights URL is:
`https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl` (1,685,868,769 bytes).
It is slow on a single connection. `daynight/download_reference.py` uses four verified
HTTP ranges, retaining the already received sequential prefix. It checks URL, ETag,
range headers, and byte counts, and records progress independently of preallocated
file length. It writes one partial checkpoint rather than keeping duplicate full
copies. Tests cover received-prefix retention and resumption from a range manifest.

## Compatibility changes and what remains upstream

Pinned isolated packages are in `configs/turbo_reference_requirements.txt`:
Diffusers 0.25.1, PEFT 0.11.1, Transformers 4.44.2, Hub 0.25.2, Tokenizers 0.19.1,
and Accelerate 0.33.0. Existing CUDA PyTorch 2.12.1 is reused rather than downloaded
again. Main-environment Transformers 5.16.1 / Hub 1.29.0 remain unchanged.

`daynight/turbo_pilot.py` requires the pinned, unmodified official source checkout.
Its isolated-process loader redirects `from_pretrained` calls to the pinned **local**
snapshot and loads large modules in FP16. The official checkpoint is loaded with
explicit `map_location="cpu"` and `weights_only=True`; arbitrary pickle execution
is not enabled. The upstream generator forward, scheduler, trained LoRA adapters,
and trained VAE skip connections remain upstream code/weights. The source checkout
itself is not patched.

Only the official checkpoint download contacts the authors' server. After files are
cached, Hub/Transformers loading is offline and local. Input photographs are processed
on this PC; they are not uploaded to the authors or Hugging Face.

## Comparison protocol

Use the fixed BDD validation indices `0,71,142,213,284,355,426,499`, not new test data.
Preprocessing preserves aspect ratio, corrects EXIF, and reflection-pads to multiples
of eight for the VAE. Default maximum edge is 512. This differs from a square-resize
reference protocol and must be disclosed rather than called the exact paper benchmark.
VAE sampling is seeded independently per image (`2029 + index`).

Outputs and provenance are saved in `outputs/turbo_reference_val/`, including source
paths, source/base revisions, official-checkpoint SHA-256, precision, seed, resolution,
runtime scope, and peak PyTorch allocated VRAM. A single inference per input is **not**
a statistically robust warm latency benchmark. Peak allocated VRAM excludes other
processes and driver/context overhead.

Review night appearance, sky completeness, branches/wires, geometry, vehicles, facade
highlights, shadows, and glare. A stronger reference is not automatically an acceptable
teacher: hallucinated or moved objects must not become unquestioned student targets.
No local checkpoint, Gradio default, or remote Vercel deployment is replaced by this
pilot. The main app's older Turbo dropdown adapter is a separate integration path.

## Reproduction

```powershell
.\scripts\setup_turbo_reference_isolated.ps1 -CacheRoot E:\LumiCycleCache
.\scripts\run_turbo_reference_pilot.ps1 -CacheRoot E:\LumiCycleCache -Edge 512
```

Both scripts keep the main environment outside pip's installation target. The run
script explicitly puts isolated dependencies ahead of reused main dependencies.
This setup depends on the main environment remaining available; it is not a standalone
portable reference-model bundle. Rerunning a partial reference download resumes it.

## Additional paired-critic control

While the reference downloaded, a final 1,000-update control changed only the source
of real discriminator images on paired webcam updates: corrected paired-night targets
instead of unrelated BDD night images. It reused the resize parent plus last-three-block
adaptation settings. Checkpoint:

```text
E:\LumiCycleCache\experiments\paired_critic_consistent\latest.pt
SHA256 a94bf8269c8f03bea41ac5e076b0f233bbdcce7a33163d9da467e49e51c1e95d
```

Over all 61 held-out camera inputs, photometric-pyramid error was `0.13628` versus the
parent's `0.12797`; static RGB L1 was `0.14307` versus `0.13589`. All eight fixed BDD
comparisons were visually inspected. Artificial highlights and shadow cues remain,
so this control is not promoted. It suggests the critic-domain mismatch alone does
not explain the failure of these short pilots; it does not prove the mismatch is
irrelevant in every training design.

Evidence is in `outputs/sky_pilot/paired_critic_consistent_metrics/` and
`outputs/sky_pilot/paired_critic_consistent_val/`.

## Completed reference run

The official checkpoint downloaded successfully. SHA-256:
`7c13bdf38548ecf14fa3566a784e46cd63b97b4c1189173c3405fc9a977f1a46`.
All eight fixed validation comparisons were generated and visually inspected.
The reference produces substantially more coherent night skies and localized street
lighting than the retained local generators. It is not flawless: some lamps are
invented, and lettering on a stop sign is altered. Its output is a **pseudo-target**,
not the actual nighttime scene or a reliable annotation-preserving ground truth.

The first 512-by-288 reference forward took approximately 0.87 seconds; subsequent
individual forwards took 0.13–0.14 seconds. Peak PyTorch allocation was about 3.18 GiB.
These figures exclude model loading, CPU preprocessing, other processes and driver
overhead; they are not a formal latency benchmark or a 512-square measurement.

## Warm-started local student: first pilot

`daynight/train_teacher_student.py` trains the existing local resize-decoder generator
against cached external-teacher outputs. This is **knowledge distillation**, not a
diffusion model trained from scratch. The diffusion teacher is absent during student
inference, but its contribution to the training targets must always be disclosed.

The starting checkpoint is `runs/resize_decoder_pilot/best.pt`, global step 11,100.
Its trained encoder, nine residual blocks with CBAM, and resize-convolution decoder
are retained; all generator parameters are fine-tuned in this pilot. We do not replace
the local generator with the reference diffusion backbone. Night-to-day is unchanged.

Training uses 128 distinct training scenes and 48 held-out validation scenes, rendered
by the fixed official teacher at maximum edge 512 then resized to 384 by 216.
The initial cache is compressed NPZ with byte-valued source and target arrays.
A separate repeat-generation audit checked all 176 raw teacher outputs for finiteness
and reproduced all cached target bytes exactly. Seeds and source hashes are recorded.

The objective is RGB L1 (weight 10), two-scale average-pooled L1 (weight 2),
horizontal/vertical RGB-gradient L1 (weight 1), and frozen DINO source-content cosine
loss (weight 0.25). Adam uses learning rate 1e-5 and betas (0.5, 0.999); BF16,
gradient clipping at norm 3, and EMA decay 0.99 are enabled. There is no GAN critic
or cycle objective in this distillation stage. The DINO model is training-only.

After 6,000 adaptation updates, the best EMA was at update 5,750 / global step 16,850.
Its held-out weighted teacher-fidelity loss was 0.64379, versus 0.89590 at update 250.
This is a loss in the above weighted units, **not** a realism score, FID or percentage.
Best/latest checkpoints and optimizer/RNG state are in
`E:\LumiCycleCache\experiments\turbo_distilled_v1`.
Checkpoints intentionally contain the day-to-night generator only; they load through
`SkyPipeline`/`load_base`, not the legacy full-CycleGAN resume interface.

Visual outcome: body colors are less cyan/red, but the raw generator still leaves
bright sky holes and does not reproduce the reference's convincing local lighting.
Applying the existing, separately labeled learned matte plus analytic sky compositor
removes most bright sky holes; this is not evidence that the generator learned those
changes. Skies remain flat and some daylight shadows persist. The pilot is therefore
**not promoted**, and the project goal is not considered achieved.

Evidence: `outputs/sky_pilot/turbo_student_final_hybrid_val768/` and
`outputs/sky_pilot/turbo_student_validation/`. Against old V2 4.5k, sky bright-pixel
fraction falls from 0.16875 to 0.00462 over 46 sky-containing validation images after
compositing. The retained sky hybrid previously reached 0.00189, so this is not a
claim of improvement over that hybrid. Darkness alone cannot establish realism.

## Expanded training-target pilot

To test the small-target-set bottleneck, the next cache contains 2,048 distinct
training scenes plus the same separate 48 validation selections. Targets are JPEG
quality 95 with no chroma subsampling; the original source files are referenced,
not duplicated. JPEG is lossy, unlike the first NPZ cache. The manifest records all
source paths, hashes and teacher seeds. Location:
`E:\LumiCycleCache\teacher_targets\turbo_road2048`.

The second pilot starts from the first pilot's **best** EMA, global step 16,850,
with batch size 4, learning rate 1e-5, and a fresh optimizer for the changed dataset.
It is a warm-started new experiment, not an exact optimizer resume of the old run.
Within one experiment, automatic resume restores its optimizer, EMA and RNG states.
Training status and previews are written to
`E:\LumiCycleCache\experiments\turbo_distilled_v2`.
No demo default is changed merely because teacher-fidelity loss decreases.

Reproduce after the isolated reference setup:

```powershell
E:\LumiCycleCache\turbo-env\Scripts\python.exe -m daynight.cache_turbo_targets --cache-root E:\LumiCycleCache --output E:\LumiCycleCache\teacher_targets\turbo_road2048 --train-images 2048 --val-images 48 --storage jpeg-targets
.\.venv\Scripts\python.exe -m daynight.train_teacher_student --parent E:\LumiCycleCache\experiments\turbo_distilled_v1\best.pt --targets E:\LumiCycleCache\teacher_targets\turbo_road2048 --output E:\LumiCycleCache\experiments\turbo_distilled_v2 --steps 2000 --batch-size 4 --lr 0.00001 --max-minutes 40
```

## Split and external-pretraining caveat

No project test images are used to generate these targets or select checkpoints.
Train/validation source paths are disjoint. Nevertheless, **the external teacher's
pretraining may overlap our custom BDD splits**: its exact training-image identity
overlap has not been audited. Do not describe the teacher/student comparison as a
fully unseen benchmark or claim test independence from all external pretraining.
The held-out custom validation set controls this local fine-tuning stage only.
Formal generalization claims require an independently sourced evaluation set and
an overlap audit. Teacher fidelity also rewards reproducing teacher mistakes;
manual object/text/light-placement inspection remains necessary.

## Optional highlight / conditional-critic control

The expanded-data previews still show smeared lamp-like patches rather than the
teacher's compact light sources. A controlled follow-up adds two optional objectives
without resetting or replacing the trained generator:

- `--highlight-weight`: target-only soft brightness weights above RGB maximum 0.55
  in [0,1], normalized independently per image. This gives small bright regions more
  influence than plain spatially averaged L1. It is **not** semantic emitter detection:
  white signs/cars also receive weight, and teacher hallucinations can be reinforced.
- `--adversarial-weight`: a newly initialized, spectral-normalized 70-pixel PatchGAN
  critic with 32 base channels and six input channels (source RGB + candidate RGB).
  Real pairs use source + cached teacher pseudo-target, never unrelated night photos.
  LSGAN trains the critic, Adam uses 1e-4 with betas (0.5, 0.999), and gradients are
  clipped at 3. The critic and optimizer are saved for within-run resume. It is absent
  at inference. Generator updates freeze both critic parameters and spectral-norm
  power iteration; input gradients still flow to the generator.

The source-conditioned critic is motivated by
[pix2pix, Isola et al., CVPR 2017](https://phillipi.github.io/pix2pix/), not presented
as an original algorithm. Whether it helps this small distilled generator is an
experiment, not an established result. Both options default to zero, preserving
the plain-distillation control and its backward-compatible resume behavior.
Full-generator and conditional-critic finite-gradient GPU checks pass.

The combined control uses highlight weight 0.5 and adversarial weight 0.25,
generator learning rate 2e-5, batch 4, and 2,000 adaptation updates, warm-started
from expanded-data global step 18,850. Output directory:
`E:\LumiCycleCache\experiments\turbo_distilled_lights`.
Because both objectives change together, this control cannot isolate which causes
any improvement or regression; individual ablations would be needed for that claim.

`daynight/evaluate_teacher_student.py` reports RGB/weighted fidelity, bright-region
L1, teacher-bright-pixel recall and extra-bright-pixel fraction on the 48 held-out
teacher targets. These describe **agreement with a fallible teacher**, not physical
correctness or human realism. Review its four-column gallery alongside the metrics.
The trainer still selects `best.pt` by plain weighted teacher fidelity; always review
`latest.pt` and intermediate previews as well when evaluating lighting objectives.

## Expanded-data control results

The plain-distillation expanded-data run completed 2,000 updates / global step
18,850. Best validation fidelity fell from the first student's 0.64884 to 0.57355
when **both** are evaluated against the same JPEG cache. RGB L1 fell from 0.05340
to 0.04720 in [-1,1] units. However, teacher-highlight L1 worsened slightly from
0.71636 to 0.71967, and thresholded teacher-bright-pixel recall fell from 0.23660
to 0.21234. Brightness thresholds refer to RGB maximum, not semantic light sources.

All eight four-column validation comparisons were inspected. Body-color agreement
improves modestly, but the sun/cloud-glow case retains a bright smeared sky patch,
streetlights are mostly missing, and the traffic scene keeps daylight shadow cues.
This control is **not accepted as convincing nighttime translation**. The numerical
improvement does not satisfy the visual goal.

Evidence: `outputs/sky_pilot/turbo_student_expanded_validation/metrics.json` and
`gallery.jpg`. Training peak PyTorch allocation was about 3.00 GiB; 2,000 updates
took about 5.8 minutes plus initialization/final validation. These are pilot-scale
full-frame 384-by-216 updates, not the runtime of a large diffusion training run.

## Combined lighting-control results

The lighting control completed 2,000 updates / global step 20,850. Its latest EMA
achieved these same-cache validation diagnostics against the expanded-data parent:

| Diagnostic | Parent | Lighting control |
|---|---:|---:|
| Weighted teacher fidelity, lower better | 0.57355 | 0.57366 |
| RGB L1, lower better | 0.04720 | 0.04731 |
| Teacher-highlight L1, lower better | 0.71967 | 0.46605 |
| Teacher-bright-pixel recall, higher means more target-bright pixels matched | 0.21234 | 0.52863 |
| Extra-bright-pixel fraction, lower better | 0.000714 | 0.004324 |

All eight final validation panels were visually inspected. The control learns more
bright regions, but several become oversized white/cyan blobs instead of compact
streetlights. The sun/cloud scene retains an unnatural glow; daytime shadow cues
remain in the traffic scene. Extra-bright pixels increase about sixfold. This is
**not accepted**: higher bright-pixel recall is not proof of better nighttime realism.

Checkpoint: `E:\LumiCycleCache\experiments\turbo_distilled_lights\latest.pt`.
Evidence: `outputs/sky_pilot/turbo_student_lights_validation/metrics.json` and
`gallery.jpg`. Peak allocated VRAM was about 3.02 GiB; the 2,000-update loop took
about six minutes. The original V2, retained sky hybrid, and demo default are unchanged.
All 51 unit tests pass after adding the optional objectives and validation diagnostics.

The next experiment should address light **footprint and localization**, not simply
increase highlight weight or train this configuration indefinitely. Any further
control must penalize excessive bright area as well as missed lights, retain a
plain-distillation comparison, and inspect sky boundaries and daylight shadows.
Native teacher-resolution supervision is another testable hypothesis: reducing a
512-wide teacher output to 384-wide targets discards some small-light detail. Neither
hypothesis has been established as a solution by the experiments above. No new
architecture or teacher-dependent result should be presented as successful before
visually passing the same held-out scene suite.

Follow-up implementation, compact-light results, native-resolution curriculum and
the confirmed/repaired deterministic-resume configuration bug are recorded in
[COMPACT_LIGHTS_AND_NATIVE_RESOLUTION.md](COMPACT_LIGHTS_AND_NATIVE_RESOLUTION.md).
