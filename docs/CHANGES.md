# Changes and experiment handover — 21 October 2026

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
