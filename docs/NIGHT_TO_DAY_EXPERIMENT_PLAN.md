# Night→day experimental track — 22 October 2026

Status: active investigation, NOT a production feature or a successful model claim.
The finished day→night app, default checkpoint, website and public inverse rejection
remain unchanged. Historical inverse weights are accessed only by a separate CLI.

## Updated local-model boundary and diagnostics

The proposed semantic audit has been executed. Clean Transformers4.44.2 loading
does not solve unreliable night sky/structure masks; do not impose them as hard
truth. External FLUX.2 Klein4B editing was also tested on all24development images
plus eight conditioning controls. It improves daytime appearance but repeatedly
changes vehicles, buildings and overhead structures. It is an EXTERNAL reference,
not our trained model, not deployed, and not an accepted result. See visual review.

User clarification: the deliverable remains our own warm-started locally trained
inverse network, analogous to day-to-night inference, with any external supervision
disclosed. Do not silently replace the network with an external generator or call
editing its outputs a model improvement. No new FLUX distillation is implied
by these reference comparisons; poor correspondence rejects those targets.

Completed bounded experiment:32 already-reviewed synthetic-night/REAL-day training pairs,
existing inverse global16,000 warm start,1000updates, unchanged architecture/loss,
same24real-night validation and eight explicitly training-only reconstruction panels.
This isolates capacity versus real-night domain/target limitations. All eight
training diagnostics and24real-night outputs were opened. Training reconstruction
improves substantially (2.63645 to0.66212), but real-night results retain glare and
patchy skies. Synthetic-only global17k is rejected; retain16k parent. This supports
a supervision/domain gap, not a proof architecture is sufficient. Next audit original
coarse real pairs in native-night coordinates before any paired training, or test
a distinct local degradation-aware adaptation. Do not repeat blind synthetic steps.

Follow-up executed: native-night-coordinate alignment on six original Dark Zurich
training pairs with raw/gamma/local matching fails on most samples (melted geometry,
very low flow consistency); no dense targets approved. Two real-day local+global
adversarial continuations (weights.5/2,1000updates from16k) also completed. All24
cases viewed at matched500 and final1000. Broad scene layout is retained, but sky
holes/overhead errors and glare persist; neither candidate is promoted. Next test
target reliability in degradation regions against these saved uniform-loss controls,
not a further blind extension or external model substitution. Full details and
artifact paths are in NIGHT_TO_DAY_VISUAL_REVIEW.md. Test/lockbox untouched.

The subsequent retained-glare target-reliability ablation has also completed:
all24development and eight training mask overlays inspected before1000updates,
then all24matched outputs inspected. Lamps remain and cyan/yellow checkerboard
speckles appear in relaxed regions. Rejected, not promoted. Do not repeat or
broaden the same brightness mask as a claimed semantic fix. A separate same-weight
512/256/128input-scale audit was run/viewed on all24cases; smaller inference loses
detail and does not solve the lighting.107tests pass. Both negatives are recorded
with actual artifact paths and native-image review in the visual log.

Next distinct data option: bounded original-pair overlap mining, with actual visual
review and static-region reliability rather than assuming all coarse pairs work
or that six mostly failed samples prove none can. Keep16k local generator parent;
no external RGB editor substitution, unchanged development/lockbox/test boundaries.

This96pair overlap scan is now executed, with all top24panels opened. Several
highest-confidence day-directory references are visibly twilight; none approved
as full-image daylight targets. Four manually annotated static building/wall
regions were inspected and used for a bounded500-update auxiliary real-RGB loss
alongside the same111teacher pairs, warm-start16k. The regions fit much better,
but all24development outputs retain sky/glare defects; no promotion. See visual
review for masks, source/target hashes, exact cache/run paths and comparisons.
110tests pass. The result does not justify repeating small facade-only training:
new supervision must address whole-scene daylight/sky/artificial-light changes,
not merely local exposure or high flow confidence. Keep the conservative16kparent
and preserve the finished day-to-night deployment.

Fixed-camera follow-up also executed: all14DNIM TRAIN-camera clock sheets viewed;
nine selected cameras,36prepared photographic night/day pairs and manual upper
sky/building masks inspected. Five daytime sun/flare targets rejected,31retained.
Held-out cameras excluded; no new download or reversed old warped-cache use.
Matched1000-update continuations from16k compare auxiliary coarse RGB, pixel RGB,
and teacher-only control. Both full24development comparisons opened; recurring
sky/glare/road-pool defects remain, so none promoted. Nine representative training
outputs also show noise amplification and texture artifacts despite reduced
photometric error. Data, masks, rejections and checkpoint provenance are saved;
113tests pass. See visual log for exact paths and review scope. Next investigate
signal/noise/normalization sensitivity or another distinct evidence-backed mechanism,
not a blind extension of this small fixed-camera recipe. Goal remains incomplete.

## Objective and honest success criterion

Produce plausible daytime scenes from real night images while retaining the
observable scene. Codex must actually open the generated images, judge daylight
appearance and correspondence, write specific failure notes, iterate, and only
claim success when the final model passes that visual examination. Brightness,
a discriminator, a daytime classifier, or one attractive screenshot is insufficient.
Completely clipped lamps and black regions contain unrecoverable information:
plausible inference is possible, recovery of the true unseen scene is not guaranteed.

Provisional visual gate (fixed before improving candidates): on24 validation inputs
covering lit streets, deep darkness, headlights, colored lamps, foliage, urban
buildings and wet roads, at least20 must look convincingly daytime **as a whole**,
with no conspicuous pasted-sky blocks, night-style haze, or newly invented large
objects. At least2/3 must pass within every adequately represented difficulty
group, so easy inputs cannot conceal systematic failure. A fresh12-source lockbox
from validation is used once at candidate selection, not repeatedly tuned. The
existing project test remains untouched. These are working engineering gates,
not statistically powered or blinded research claims. Their numerical gate does
not override a clearly recurring serious failure observed during review.

Review source/output side by side at512 and inspect representative native crops:

1. Coherent daytime sky, ambient illumination and surface exposure, not simply
   higher gain, white fog or a pale blob over a dark street.
2. Same roads, cars, pedestrians, buildings, signs, wires and major silhouettes.
   Small ambiguous text is not claimed accurately reconstructed.
3. Headlight bloom and artificial-light dominance reduced plausibly; headlights
   and traffic signals need not vanish, since both can remain active in daytime.
4. No noisy shadows, checkerboard, strange desaturation or cartoon textures.
5. Honest labels: rejected / partial / convincingly daylight, with reasons per input.
   Also inspect outputs without source to test overall perception, then restore
   the source for preservation audit. This is a subjective Codex vision review,
   not a human blind study or a classifier score.

## Verified starting evidence

`python -m daynight.experiment_night_to_day --output <new-folder>` uses8 deterministic,
spaced BDD validation-night indices0,71,142,213,284,355,426,499. It loads EMA
G_night_day from existing V1 step13,000 and V2 step4,500, outputs all sources and
predictions separately, and compares a gamma.45 visibility-only negative control.
It never routes through the public inference method or changes deployed direction.
Initial actual outputs are in outputs/night_to_day_experimental/20261022_baselines.
See NIGHT_TO_DAY_VISUAL_REVIEW.md for actual observations, not assumed causes.

V1 looks less hazy than V2 on this first panel set, but does not convincingly solve
the task. Thus warm-start choice must be measured rather than assuming later V2
is better. Existing day→night distilled weights have no trained inverse and must
not be mistaken for a new night→day model. Any local continuation must preserve
original checkpoint files and explicitly record warm-start scope/global step.

## Research-informed shortlist and elimination logic

| Approach | Reason to test / useful mechanism | Risk and current priority |
|---|---|---|
| Gamma/Retinex/low-light enhancement | Exposes what visibility alone recovers; cheap control | Cannot remove night illumination or infer daylight; not the end solution |
| Existing V1/V2 inverse | Available warm-start weights, fast and local | Observed pasted sky/haze/bloom; baseline, not more blind training |
| Official CycleGAN-Turbo night→day | Already-cached base model/environment, actual official direction-specific checkpoint, pretrained daytime prior | External reference, possible hallucination/domain/pretraining overlap; first practical pilot |
| Teacher-supervised local continuation | Warm-start the better existing inverse; aligned same-input pseudo-targets give explicit daylight supervision | Teacher errors must be filtered; do NOT distill all outputs indiscriminately |
| N2D3 degradation-aware constraints | Distinguishes darkness, ordinary lit surfaces, saturated regions and colored/glare effects; avoids identical structure penalty everywhere | Physical masks are imperfect; use as an ablation, not a renamed claimed reproduction |
| N2D-LPNet / Laplacian low/high frequency separation | Change low-frequency illumination without indiscriminately copying night color/flare into details | High-frequency bands also contain glare/noise; blind high-frequency preservation repeats old problem |
| Real coarse day/night pairs (existing Dark Zurich) | Target daytime appearance not limited to a teacher's taste | Camera motion/dynamics and optical flow mismatch; audit native night coordinates before pixel losses |
| Diffusion editing + depth/edge/albedo conditioning | Strongest generative prior for lost detail; fallback if one-step reference still fails | More expensive; excessive guidance fixes dark artifacts, weak guidance moves objects; bounded pilot before downloads/training |
| HELIOS-style albedo-conditioned diffusion | Recent directly relevant night/day scene relighting approach | Author project inspected; no downloadable code/weights link observed on project page at review, so not yet runnable here |
| IC-Light | Useful relighting reference | Foreground/background conditioning not a ready-made faithful full-road solution; lower priority than direct night→day model |
| More aggressive sky compositing / full renderer from scratch | Could cosmetically improve one component | Cannot make night road/building lighting daytime; previous renderer failed; not the primary route |
| 3D Gaussian splatting | Appropriate for multi-view reconstructed scenes | No multi-view reconstruction/input; not suitable for this single-image task |

Sources inspected (claims are author-reported, not independently reproduced):

- [Official CycleGAN-Turbo code/paper repository](https://github.com/GaParmar/img2img-turbo).
  Existing pinned source commit86f54146590ffb4543c8cf85b5a36657da670924 supports
  night_to_day with its own night2day.pkl, not day2night weights used backward.
- [N2D3 paper, full method](https://arxiv.org/html/2411.14504v1): different regional
  degradation types and region-aware contrastive preservation. Inference remains
  a ResNet; physical prior is used for training guidance.
- [N2D-LPNet primary paper](https://openaccess.thecvf.com/content/CVPR2023W/UG2/papers/Fan_Learning_To_See_in_Nighttime_Driving_Scenes_With_Inter-Frequency_Priors_CVPRW_2023_paper.pdf).
- [HELIOS author project](https://hala-djeghim.github.io/HELIOS/) and
  [paper](https://arxiv.org/abs/2609.00901): albedo conditioning and solar-angle
  control. Publication results do not prove feasibility with our current data/GPU.
- [IC-Light official repository](https://github.com/lllyasviel/IC-Light).

This is a targeted mechanism-based shortlist, not a claim to have run/read hundreds
of repositories. Prefer compatible, verifiable candidates over paper-count theater.

## Execution sequence

### A — Evidence before training (in progress)

1. Freeze source IDs and record source hashes, split, checkpoints and inference
   precision. Generate old inverse/gain controls; review actual outputs.
2. Download only official night2day reference weights into E:/LumiCycleCache;
   reuse existing pinned SD-Turbo cache and isolated turbo-env. No C/D bulk cache,
   no H external disk. The new checkpoint is approximately1.69GB. Run8 matching
   source images, label all reference outputs EXTERNAL. Record weights SHA256.
3. Expand to24 validation cases by inspecting source content (not selecting only
   reference successes). Tag difficulty/visibility; reserve separate12-source
   validation lockbox before local adaptation. Frozen membership recorded in file.
4. Actual image review decides if Turbo is useful, not marketing metrics. Reject
   it as a teacher if geometry or night haze failures dominate; test next route.

### B — Make a local inverse learn the useful daylight change

If reference results warrant it, first cache a bounded128train/24val pilot with
seeded targets. Review target quality; do not train on rejected artifacts. Initialize
from best existing inverse (V1/V2 selected by actual comparisons), all local layers
preserved initially. If converting the decoder, separately record architecture
change and mapping audit; no silent checkpoint-key reinterpretation.

Proposed first training ablation: full-image teacher RGB/coarse target supervision,
small target-gradient/perceptual term, EMA, BF16, short250-update sanity run before
1000-update pilot. Do not apply the day→night highlight-target coefficient as if
daylight should preserve every night lamp. No initial GAN race or heavy source
pixel/edge equality. Source structure weights apply mainly to visible reliable
surfaces, not saturated/bloom or noise-dominated regions. Keep reliable source
geometry, not the source's night illumination. Compare against no-region-weighting
control and raw parent at identical held-out IDs/settings.

### C — Failure-driven refinements, not arbitrary loss stacking

- Sky still night but structures good: investigate low-frequency illumination and
  mask coverage; no pasted-sky-only acceptance. Sky/foreground exposure must agree.
- White fog / amplified light: estimate saturated/bloom reliability; do not enforce
  target-free source gradients there; add daytime-target color/contrast supervision.
- Lost geometry / hallucinated car: lower generative freedom, use reliable spatial
  features/coarse-pair alignment, reject bad pseudo-target rather than memorizing it.
- Noisy dark regions: confidence-aware denoise/restoration prior; do not preserve
  high-frequency noise as content. Flag fundamentally unobserved regions.
- Teacher good, student poor:32-image training-only overfit diagnostic followed by
  native-resolution/full-generator pilot, do not increase loss count reflexively.
- Both poor: stop distillation route and pilot degradation-aware direct adaptation
  or structure-conditioned diffusion, retaining all failure evidence.

Existing Dark Zurich metadata says2416coarse training pairs, but alignment is in
**day-camera coordinates** (warped night). That is not automatically an inverse
native-night supervised set. Motion/occlusion masks and actual confidence pixels
must be audited. Existing ACDC cache has400night and **zero normal references**;
do not claim paired ACDC supervision. No new dataset download is necessary just
to test the first reference. Sequence/location-disjoint paired validation is
required before trusting any coarse-pair gain.

Audit update: six actual spaced cached pairs were viewed. Severe deformation and
near-zero confidence make the existing warped cache unsuitable for direct inverse
pixel loss.2416pairs use1426unique day references, with up to70reuse. Do not feed
these into training just because the scalar manifest confidence says1.0. A separate
synthetic-pair control (existing strong day→night model applied to real training-day
images, supervised back to those exact day sources) can isolate whether direct
daylight supervision helps the existing inverse. This is synthetic-domain training,
not proof of generalization to real night; evaluate only on frozen real-night suite.

### D — Selection and optional experimental interface

Keep per-image visual notes, original model files and best-so-far outputs. Review
native crops, difficult groups, latest candidate versus previous best, and untouched
validation lockbox at selection. Retain rejected trials with reasons. Add a separate
local experimental UI only once there is something worth testing; never replace
the finished public day→night model or re-enable inverse production jobs by accident.
Continue until actual good transformations are demonstrated, not until a step count
or discriminator says success. Large external training/download expansion should
be justified by bounded pilot evidence, storage and licensing.

## Current decision after the first local continuation

Reference24and local+250/+1000all24outputs have been opened. The supervised signal
helps, but first local pilot fails robust visual acceptance. Mean24-source
teacher-fidelity improves3.39472→2.08602after250, then plateaus~2.04; training loss
is much lower. Do not keep training these47curated targets merely to grow the step
count. Next controlled experiments: expand/diversify audited teacher examples;
compare identical-weight warm starts with transposed versus adapted resize decoder,
explicitly noting that kernel conversion is approximate, not function-preserving.
Keep the trained inverse as parent, avoid strong source-RGB/gradient constraints
that preserve nighttime glare, and test problematic-light regions separately.
Teacher's own sky-hole/flare mistakes remain a ceiling; structural teacher filtering
or a better relighting/editing target must precede claiming those failures fixed.
No acceptance or goal completion is claimed by this progress note.

## Follow-up decisions after matched decoder and input controls

- Matched500-update transpose/resize trials from global14,000are completed and
  all24comparisons inspected. Resize is not selected: slightly worse teacher
  agreement and no solution to the recurring sky/glare failures.
- Eight custom reference-input controls (highlight rolloff and gamma+rolloff) were
  generated/viewed. No reliable general glare/sky correction; not deployed.
-128teacher training sources were generated and all eight contact sheets viewed;
  17rejected and111retained. A new continuation preserves the trained inverse
  global14,000, with no architecture reset. Its first1000updates improve the same
  24validation targets from~2.05to~1.83weighted teacher fidelity; all outputs viewed,
  still not accepted visually. Further bounded continuation is justified by the
  saved improving250-step trajectory, not a belief that more steps always help.
- Next independent restoration option, not yet executed: small automatic-mask
  inpainting pilot with existing cached SD-Turbo. Verify shape/API and apply exact
  pixel preservation outside a saved mask. Check that lamps/objects are not erased
  or invented; reject if only a cosmetically nicer but geometrically wrong image
  results. Label this external generative restoration, not locally trained model
  achievement. No hand-selected masks per demonstration image as an alleged model.

Relevant primary implementation guidance inspected:
[SD-Turbo official model card](https://huggingface.co/stabilityai/sd-turbo) and the
actual installed isolated diffusers StableDiffusionInpaintPipeline signature/source.
SD-Turbo uses guidance_scale0; its model card supports short1–4-step operation,
but that does not establish that inpainting will work for these night artifacts.

Latest bounded-run state:111-source continuation completed2,000additional updates,
global16,000, weighted teacher fidelity1.8128188401. All24latestoutputs inspected;
recurring sky/glare problems remain, so no promotion and no further blind extension.
The next goal turn should test targeted restoration/region-aware supervision against
this retained parent, with controlled comparison and full image review. Reference
input rolloff and the simple resize swap are already tested negatives; do not repeat
them as if untried or discard this model's trained progress.

## Further executed controls: restoration and genuine daytime targets

The next experiments above have now been executed; do not re-propose them as
untried. See `NIGHT_TO_DAY_VISUAL_REVIEW.md` for all-image observations and paths.

1. Eight-source masked restoration with cached EXTERNAL SD-Turbo at strengths.5
   and.85 preserved all outside-mask bytes but changed cars/lanes/buildings inside
   masks or left the glare. Masks cover16–47%of images. Rejected, not deployed.
2.64synthetic-night/REAL-day pairs from the existing frozen day→night hybrid were
   generated from training-day sources and all four sheets reviewed. Mixed with
   the111audited real-night/EXTERNAL-target examples, this makes175training sources.
3. Matched500-update continuations from existing inverse global16,000completed:
   teacher-only111and mixed175, both global16,500. All24comparison rows reviewed.
   Mixed provides small tone/road-glare changes but no robust sky/lighting fix;
   neither branch meets acceptance. More steps on this same small mixture are
   not justified solely by its decreasing imitation loss.

Next decision should target semantic/geometry uncertainty, not another unconstrained
inpainting pass: audit whether reliable sky/overhead-structure segmentation can be
obtained on lifted night sources and corresponding day predictions using cached
frozen teachers. Check its masks visually FIRST, especially142/214underpasses and
12/105branches. If masks are unreliable, do not add a semantic loss on them or call
it a physics fix. If usable, compare confidence-weighted region supervision against
the retained trained parent without forcing real nighttime glare to remain as
daytime texture. This is a proposed next check, not an already executed improvement.

The [N2D3 primary paper](https://arxiv.org/html/2411.14504v1) motivates separating
well-lit, low-light, strong-light and colored degradation effects rather than
uniformly preserving all source features. This is not a claim our threshold mask
reproduces their degradation disentanglement. The inspected paper page did not
provide an identified author-code/checkpoint link; a broad search is not proof
that none exists. Do not invent a ready-to-run N2D3 checkpoint.

## Latest user-directed research reset — 22 October 2026

The later controls and full visual reviews are recorded in
`NIGHT_TO_DAY_VISUAL_REVIEW.md`; historical proposed next steps above are not a
queue to execute blindly. Both real-timelapse auxiliary variants and the matched
teacher-only control completed and were rejected visually. The user specifically
requested different methodologies after seeing noisy results.

See [the research reset](NIGHT_TO_DAY_RESEARCH_RESET_2026_10_22.md) for the current
primary-source comparison, concrete noise/relighting hypotheses, actual UNIT
normalization-config finding, paired-data lead and staged warm-start proposal.
No training or production changes were made during that research pass. Do not
describe its proposed adapters as implemented improvements. Keep the trained16k
inverse parent and completed public day-to-night pipeline unchanged.
