# Changes and experiment handover

## Remote inverse preview and repository refresh — 23 October 2026

- Restored Night → Day to the remote direction selector as experimental, using
  local EMA inverse `night2day_CARLA12_aux10_from16000_v1/adaptation_00500.pt`
  at global step 16,500. This is not a quality-gate promotion: noise, retained
  glare and lighting artifacts persist. No DarkDriving training is involved.
- Added a direction-specific inference adapter; day-to-night checkpoints and
  integrated sky repair remain unchanged. Wrong model/direction pairs are rejected.
- Refreshed README with real validation comparisons, an experimental inverse
  collage, architecture preview, current run instructions and separate provenance.
- Removed 14.87 GiB of unused downloaded FLUX reference weights on E:. Kept
  configuration/license/provenance, all locally trained weights, data and results.
  Reference weights can be downloaded again; deletion is not an architecture change.
- Exact checkpoint duplicate audit found zero byte-identical duplicate files.
  No unique trained checkpoint was deleted. Lossless NTFS compression of `runs/`
  is used to reduce D: allocation without changing checkpoint bytes or paths.


## Documentation synchronization — 22 October 2026

This entry records the latest documentation work only. **No generator or matte
weights, training objectives, inference code, demo defaults, or deployment were
changed by this update.** No training was started and nothing was pushed or deployed.

### Documents created or updated

- Created [FINAL_DAY_NIGHT_ARCHITECTURE.md](../FINAL_DAY_NIGHT_ARCHITECTURE.md)
  as the consolidated current day→night architecture: exact selected checkpoints,
  parameter counts, network layers, active final training losses, integrated sky
  repair, external teacher attribution, inference behavior, and unresolved flaws.
- Updated [CURRENT_MODEL_ARCHITECTURE.md](CURRENT_MODEL_ARCHITECTURE.md) to point
  to that snapshot and distinguish historical V2/raw-student descriptions from
  the current hybrid default.
- Updated [PROJECT_MASTER_GUIDE.md](PROJECT_MASTER_GUIDE.md): corrected the model
  identity and 90-second explanation; added current architecture/distillation
  teaching material, professor Q&A, current API example, whiteboard diagram,
  deployment checks, limitations, and attribution. Original V2 architecture,
  eight-loss recipe, training iteration, and optimization settings remain as
  **historical teaching material**, not the current final student recipe.
- Updated this changelog to clarify obsolete defaults and the withdrawn initial
  showcase selection.

### Current state to communicate

The default is **LumiCycle student + integrated sky repair — experimental**
(`distilled_best` remotely; `LumiCycle distilled student` locally), at 512 px
maximum edge. It combines the selected EMA RGB generator at cumulative global
step **30,600** with the frozen local sky/vegetation network at step **7,800**
and explicit source-guided compositing. Repair integration did not retrain or
replace those generator weights. Active learned inference parameters total
**12,545,231** across the two networks.

The selected final distillation control uses teacher fidelity, highlight,
bright-footprint, and sky losses; it does not execute the original V2 GAN/cycle
recipe. External CycleGAN-Turbo supplied generated training targets and must
be acknowledged. The selected branch has no VGG feature loss; the RGB generator
still uses InstanceNorm and resize-convolution rather than a VGG backbone.

Recent night→day experiments are a separate development track, not a modification
of this day→night release or a restored public feature. Current public inference
remains **day→night only**. Fine sky gaps, sun/reflections, artificial glow,
incorrect masks, and out-of-domain failures remain possible; documentation does
not establish a new quality improvement or a passed final-test acceptance gate.

The retained current showcase has **14 curated hybrid validation examples**.
The initial 20-image raw selection described in older entries is withdrawn.
Curated demonstrations are not a random test benchmark.

### Checks for this documentation update

The master guide's local Markdown links resolved and its code fences were balanced.
Documentation whitespace checks passed. These are document checks, not a fresh
training run, GPU benchmark, test-suite run, or remote end-to-end verification.

Entries below retain earlier implementation and experiment history. Older V2
default, initial 20-image pack, and canceled-goal statements describe their dated
handover, not today's defaults or the subsequent separate inverse experiments.

## Follow-up implementation — 22 October 2026

The local distilled student now invokes the trained sky/vegetation matte and
guided analytic sky compositor during inference, including through the remote
worker. Previously sky masks only supervised its training. Initial integration
still exposed cutouts around cables, so continuous local sky-color recovery and
a narrow source-derived edge band were added and tested. Raw inference remains
available via `sky_repair=False`; no generator weights/step counts were reset.
Matte model step 7,800 is separate from generator global step 30,600.

Local/web defaults now select the student-plus-repair hybrid at 512 px, with
explicit external Turbo-target attribution. The metadata identifies repair and
its matte checkpoint; the worker reports the real numeric generator step rather
than the checkpoint filename `best`. Existing raw comparisons remain historical.
Regression tests cover missing weights, no-sky identity, nonvegetation cable holes
and the manager path used by the worker. Visual smoke tests include the reported
cable image, bare branches, overhead wires, clear sky and ordinary road scenes.
This is not a universal sky fix or a learned removal of sun shadows/reflections.

> Showcase correction later in the day: the initial 20-image selection was
> withdrawn after full-size review exposed cable/sky holes and false glow. The
> current pack has **14 stronger examples** after screening 500 validation outputs
> and rerunning/reviewing each selected upload. See `SHOWCASE_SELECTION_REVIEW.md`.
> Earlier references below to a 20-image pack describe the superseded selection.

## Read this before presenting

This records the experiment history available at today's handover, including work
from earlier sessions. It does not imply every long training run occurred today.
No new training is being started by this release. The canceled autonomous goal
is not being resumed. Checkpoints/data remain local and are not GitHub assets.

The supported application scope is changing to **day → night only**. The inverse
mapping looked poor and is being withdrawn, not claimed fixed. Training-time
cycle reconstruction and historical inverse weights remain available as research
history. Deployment completion should be checked against the release commit and
live site; writing this document alone does not prove a deployment succeeded.

## What we tried, why, and what actually happened

| Experiment | Reason | Observed result / decision |
|---|---|---|
| V1 13k versus 40k | Test whether longer training helps | Later training was not reliably better; 13k remained a useful baseline |
| V2 warm start | Preserve learned conversion while improving global lighting | Best original demo parent is the actual V2 4.5k file; informal 5.5k naming was misleading |
| V2.1 detail/frequency constraints | Preserve branches and fine structures | Night conversion weakened; failed ablation, not an improvement |
| LumiRender, stopped 34,004 | Factorization, Gaussian lights, reflections, camera pipeline | Mostly global darkening; insufficient local light/sky transformation; rejected |
| Sky matte + analytic composition | Change sky without destroying branch boundaries | Helps selected sky boundaries but imperfect masks/fringes and flat sky remain; explicitly hybrid |
| Resize-convolution decoder | Suppress transposed-convolution grid artifacts | Measured grid bias fell roughly 91%; slight softening; does not solve lighting semantics |
| Timelapse pair correction | Obtain actual changing illumination supervision | Fixed UTC/local-time label mismatch; small camera-disjoint pilots still failed aggregate quality |
| Dark Zurich correspondence audit | Use coarse real day/night pairs | Reliable correspondence coverage was nearly empty; rejected naive full-frame paired loss |
| Official external Turbo reference | Establish a strong modern reference | Stronger lights/night appearance, but text/geometry changes and hallucinations; always external |
| Small teacher-student cache | Transfer night appearance into our compact RGB model | Reduces some color errors; sky holes and missing/smeared emitters remain |
| Expanded 2,048-source cache | Improve coverage beyond the first 128 images | Better target fidelity; not sufficient for convincing general lighting |
| Conditional critic + highlights | Teach local light emission | Increased bright recall but also large false glow/blobs; not accepted |
| Light-footprint penalty | Penalize expanded or invented bright regions | Lower extra-bright area, improved localization; still artificial cloud/sun glow |
| Native 512×288 targets | Avoid throwing away small lights during target downsampling | Better fine-light metrics; sun reflection/sky artifacts persist |
| 32-image overfit diagnostic | Check whether the student can represent sharp lamps | Training light IoU reached about 0.79 but validation worsened: memorization, not quality success |
| Full supervised target fitting | Separate capacity from overstrong source/GAN constraints | Improved held-out light IoU to about 0.40; still obvious visual artifacts |
| Teacher stochasticity audit | Test if noisy targets explain the smearing | Teacher variation was tiny (bright IoU ~0.997); not the main cause |
| Sky-target weight 2 vs matched 0 | Penalize bright sky against NIGHT target, not source day | Excess sky luminance fell ~35%, some highlight recall worsened; partial improvement |
| Frozen VGG target loss 0.5 | Compare local night texture/contrast beyond RGB | Almost tied with matched no-VGG continuation; no meaningful overall win |
| Zero-init detail skip | Provide a future high-resolution correction path without erasing parent weights | Implemented and unit-tested only; untrained, no quality claim |

## Latest matched results

Shared parent: `E:/LumiCycleCache/experiments/turbo_sky_target_weight2/step_01000.pt`
(global 29,850). Final matched runs each add 1,000 updates with identical data,
batch four, learning rate 1e-4, sky weight 2 and restored optimizer/RNG state.

| Diagnostic | No-VGG control | VGG target 0.5 |
|---|---:|---:|
| Final global step | 30,850 | 30,850 |
| Weighted teacher fidelity ↓ | 0.5102956 | 0.5103407 |
| Bright-pixel recall ↑ | 0.560138 | 0.560646 |
| Bright-pixel precision ↑ | 0.592456 | 0.594377 |
| Bright-pixel IoU ↑ | 0.411474 | 0.412702 |
| Extra-bright fraction ↓ | 0.002836 | 0.002818 |
| Sky RGB L1 ↓ | 0.034206 | 0.033902 |
| Excess sky luminance ↓ | 0.007175 | 0.007367 |

Control: `E:/LumiCycleCache/experiments/turbo_vgg_target_control`.
VGG: `E:/LumiCycleCache/experiments/turbo_vgg_target_pilot`.
Evidence: `outputs/sky_pilot/vgg_target_matched_1000_validation/metrics.json`
and labeled comparison panels. Best control fidelity was ~0.50819, but the table
uses final checkpoints for a fair equal-update comparison. Do not call these
numbers FID or accuracy. No blind acceptance gate or independent final test win
was established. Original V2 was not automatically replaced in the demo.

## Engineering improvements retained

- Explicit parent/checkpoint hashes, cache provenance and source hashes.
- EMA inference; atomic saves; global-step lineage through training phases.
- Optimizer/RNG continuation checks, verified same-source resolution curriculum.
- A short VGG-objective restart audit matched exactly on this hardware/software stack.
- Deterministic configuration ordering fixed after identifying a reproducibility bug.
- Teacher download integrity and isolated reference environment; no teacher source patches.
- Large caches on E:, not the external H: drive; datasets/weights/secrets excluded from Git.
- 69 tests passed at the previous architecture checkpoint; release changes need a fresh run.

## What remains unsolved

The model does not identify and remove physical solar illumination consistently.
A bright sun/reflection can become a spurious lamp/glow. Tiny structures and sky
boundaries remain fragile. Lights can smear, text can change and nighttime
illumination is not uniquely recoverable from a daytime photograph. Better visual
examples do not remove these limitations. Night → day was demonstrated and judged
unusable; it is being removed from supported interfaces rather than hidden behind
claims of bidirectional quality.

## Showcase curation policy

Select approximately 20 validation daytime inputs without obvious solar discs,
strong sun reflections or extreme flare. Inspect their **actual local-model**
outputs; do not use external teacher outputs as if they were ours. Keep original
inputs, generated outputs, paired comparisons and a source/checkpoint manifest.
Label the set curated best-case demonstrations, not random test performance.
Do not publish licensed BDD images in Git or deploy them as public web assets.
Keep the local showcase folder ignored; publish only its selection script and
documentation. The teacher provenance must remain visible for distilled models.

## Files for teammates

- `docs/CURRENT_MODEL_ARCHITECTURE.md`: current architecture plus historical failures.
- This file: experiment outcomes and current release scope.
- `docs/CURRENT_MODEL_PROBLEMS_AND_RESEARCH_QUESTIONS.md`: detailed research questions.
- `docs/PROJECT_MASTER_GUIDE.md`: foundations and project explanation.
- `docs/NIGHT_TARGET_PERCEPTUAL_PILOT.md`: VGG experiment implementation and evidence.

## Release implementation

Both Gradio and the remote frontend now show Day → Night only. Public Python
inference, the worker catalog and web job validation reject the removed mapping.
The evaluation loader now loads only the forward generator. Historical training
algorithms/checkpoints are intentionally untouched.

The optional distilled student is selectable locally and remotely with explicit
external-supervision attribution; V2 remains the default. The home worker launcher
uses the project's `.venv` rather than an arbitrary system Python.

`outputs/demo_showcase` contains 20 manually reviewed validation examples selected
from 120 generated candidates, each with input/output/comparison plus a contact
sheet and hashed checkpoint/source manifest. Use the distilled-student option at
512 px to reproduce them. The images remain ignored/local for dataset licensing.
The reproducible script is `python -m daynight.curate_demo`.

Verification before publication: 76 Python tests passed, Gradio configuration
constructed without the inverse control, and Next.js lint/production build passed.
Deployment and Git commit identifiers are reported separately after publication.

Originality is in the implementation/integration, controlled experiments,
diagnostics and adaptation—not inventing CycleGAN, VGG, CBAM or Turbo.
