# Night→day visual review log

## 22 October 2026 — first actual inverse baseline review

Evidence: outputs/night_to_day_experimental/20261022_baselines/evidence.json,
comparison_00.jpg and comparison_04.jpg. All8validation-source panels opened and
visually inspected by Codex. Columns: source, gamma-only control, V1 EMA13k inverse,
V2 EMA4.5k inverse. This is subjective vision review, not a classifier or blind study.
Generated separate PNGs are retained for full-resolution examination.

| Validation ID | Observed failure | Daytime plausibility |
|---|---|---|
| 0 | V1 adds blue sky right/bottom including implausible lower-right patch; building/street retain dark artificial lighting. V2 gray-green haze and pale corner sky. | Reject both |
| 71 | V1 uneven pasted white/blue sky and noisy trees above bright road; V2 fog, halos and remaining dark upper-left sky. | Reject both |
| 142 | V1 large daytime-looking sky blotch at upper-left while canopy/lights remain night; V2 brighter fog and bloom. | Reject both |
| 213 | V1 unnaturally rectangular pale sky, dark buildings, night headlights/reflections; V2 global haze and false blue tint. | Reject both |
| 284 | Bus geometry mostly retained, but artificial lamp dominates and scene remains blue-hour/night; gain control merely washes it out. | Reject both |
| 355 | V1 sky patch is brighter but buildings/lights stay night and textures noisy; V2 pale fog/flare. | Reject both |
| 426 | V1 has the strongest daylight impression of this set, but noisy texture, bright light fixtures and abrupt sky/building boundaries remain conspicuous. V2 washed-out haze. | V1 partial, not convincing; V2 reject |
| 499 | Severe white/blue sky filling, ghostly silhouettes and odd color/blur; V2 also pale purple/green haze. | Reject both |

Result: **0/8 confidently convincing full transformations**, one V1 partial.
This is not a population accuracy estimate. V1 is a better warm-start candidate
than V2 on these particular panels, not a universally better model claim.

Evidence supports symptoms, not a proven sole training cause. Working hypotheses:
source edge/cycle constraints preserve lamp halos or noise; unpaired losses permit
sky/lighting shortcuts; insufficient coupled daytime surface/sky supervision and
camera-domain mismatch. These require controlled ablations, not assertion.

Next actual comparison: official EXTERNAL Turbo night→day, same source IDs. No
local training started; finished public day→night inference unchanged.

## Expanded24-source baseline review

Evidence:20261022_baselines24/comparison_08.jpg through comparison_20.jpg,
plus the original8-source results. All16additional rows were opened.

| Additional ID | Observation / decision |
|---|---|
| 84 | Source itself near twilight; inverses change little except texture/color. Not evidence of successful deep-night conversion. |
| 318 | V1 retains night sky/hot red taillights; V2 pastes white sky behind dark buildings. Reject. |
| 441 | Giant lamp flare remains, buildings dark; V2 colored upper halos. Reject. |
| 358 | V1 noisy dark upper half and headlight-shaped road pool; V2 gray/green wash. Reject. |
| 22 | V1 deep-night sky/artificial light; V2 cyan blotch at top. Reject. |
| 171 | V1 pale sky islands/mixed light; V2 hazy whitish/purple sky, wet-looking night road. Reject. |
| 214 | V1 blue blob beside dark upper sky, shop lighting unchanged; V2 green/pale patch. Reject. |
| 8 | V1 partial daylight impression but noisy uneven sky and lamp bloom; V2 fog. V1 partial. |
| 105 | V1 pasted white hole/noisy sky/undiminished glare; V2 purple haze. Reject. |
| 12 | V1 blue corner sky/retained night canopy; V2 large pale blotch. Reject. |
| 50 | V1 partial brighter street but noisy fragmented sky/night glow; V2 washed-out. V1 partial. |
| 86 | V1 bright textured upper scene inferred from almost no visible source detail; V2 pale haze. Reject. |
| 78 | V1 partial daylight sky/trees but grainy textures/headlight road pools; V2 fog. V1 partial. |
| 202 | V1 street brighter but inconsistent sky patches/orange night lamps; V2 white lower-corner patch. Reject. |
| 283 | Sky stays dark with isolated white blob; saturated sign/glare retained. Reject. |
| 243 | V1 partial overcast impression, textured sky/dark foreground/night fixtures; V2 sky halos. Native crops needed before acceptance. |

Partial daylight cues exist, but recurring sky/exposure/glare defects rule out a
robust good-model claim. Source171/84have wet-looking surfaces; add confirmed
wet-road challenge examples rather than claiming complete coverage from thumbnails.

## Existing real-pair audit — important rejection

Actual evidence:20261022_pair_audit/audit.json and both pair panels. Six spaced
samples (0,400,800,1200,1600,2000) were visually opened. Cached warped-night frames
have severe wavy deformation, missing margins and wrong correspondences. Some
confidence maps are nearly black; pair400is exactly all-zero. Moving vehicles/trams
differ. Manifest scalar confidence1.0is NOT per-pixel confidence/usable supervision.

2416coarse pairs contain1426unique day references; one is reused70times. Sequence
and repeated-reference grouping are essential for any future split.
**Reject this warped cache for direct inverse pixel supervision.** Native frames
remain useful unpaired data. Better registration and dynamic/occlusion masks are
required for paired loss. This does not prove a sole cause of the earlier renderer
failure, or mean the original dataset is intrinsically bad.

## Official reference24-source review

Both comparison sheets from20261022_turbo_reference and all six sheets from
20261022_turbo_reference24 were opened. The EXTERNAL rightmost column is far more
convincingly daytime on many roads than the old inverse. Native142/source and
teacher were opened to check overhead structure. Its metal/roof region gains
sky-like blue coloring; do not treat that as recovered true daytime geometry.

Strong daylight impressions:71,213,284,355,426,499,318,358,214,50,86,78,202,283,243.
These are visual impressions of the reference, not a completed preservation audit
or statistical success count. Fully black regions' synthesized trees/buildings
cannot be verified against the source. Important exceptions:0and441retain giant
lamp halos;22keeps orange artificial-light dominance;12and105have sky-hole/blob
artifacts;84is near twilight and barely changes;171retains glare/glass corruption;
8has an odd lamp/pole edge. No claim that the full acceptance gate passed.

Reference source: pinned official CycleGAN-Turbo night2day. Actual file SHA256
cf8ffb9e2522f53a0faf77b655b8d07c0e271ceae446ba310cc4802e9bd47ab4.
Reference24uses seed4400+index. Later cache targets use5500+index (train) and
15500+index (val): different seeds must not be silently treated as identical files.

## Audited pseudo-target continuation: +250 and +1000

Cache: E:/LumiCycleCache/teacher_targets/night2day_pilot64_v1/manifest.json.
All64training source/target pairs were inspected on four contact sheets.17obviously
flawed targets excluded, leaving47. Selection JSON records IDs/reasons; all24real
validation cases retained, with no training on validation or lockbox.

Parent: existing **V1 inverse EMA global13,000**, NOT day→night student and NOT V2.
No random restart or architecture change. New inverse-only Adam for the changed
supervised objective. Full512×288, BF16, batch2, lr1e-4, EMA.99, clipping3,
10RGB+2mean pooled4/16 RGB+1mean horizontal/vertical target-gradient. No source
edge equality, GAN, cycle or VGG in this pilot. Parent and production unchanged.

Artifacts: E:/LumiCycleCache/experiments/night2day_inverse_v1_teacher250.
Folder name reflects its initial250-update pilot; actual saved recipe/status/step
are authoritative. Continued with exact local optimizer/EMA/RNG checkpoint loading
to1,000adaptation updates (global14,000). Snapshots250/500/750/1000retained.

| Adaptation / global step | 24-source mean teacher fidelity, lower preferred |
|---|---:|
| 0 /13,000 | 3.3947228044 |
| 250 /13,250 | 2.0860231618 |
| 500 /13,500 | 2.0416517754 |
| 750 /13,750 | 2.0484923323 |
| 1000 /14,000 | 2.0506935765 |

This is agreement with EXTERNAL pseudo-targets, not accuracy or a daylight score.
At+250and+1000 **all24actual output rows were opened**, including failures.
The new local inverse is noticeably more daylight-like than the old inverse,
especially426/50/78/243, and several dark-road cases gain a plausible daytime sky.
But blue sky blobs remain in12/142/214,355retains a lamp glow,318keeps a dark sky,
283still mixes night and day,441retains flare, and dark highways can preserve
headlight-shaped bright road pools. Sky/vegetation texture is grainy and soft.
The+1000checkpoint is **not accepted as a good robust transformation**.

Training losses commonly ~1.1near the end versus validation~2.05; after500the
validation agreement stalls/slightly worsens. Small curated training sample and
inherited decoder artifacts are plausible next targets. This is evidence for
testing more diverse audited targets / resize-decoder continuation, not proof of
a sole cause or a reason to run indefinitely on these47images.

The12-source validation lockbox has not been rendered or used. No public inverse
feature, checkpoint default, worker deployment or finished demo was changed.

## Matched decoder trial: original versus resize,500updates each

Both branches start at the same trained inverse EMA global14,000and use identical
47target selection, batch2, learning rate, loss and update count. Both fresh Adam
states are initialized for the comparison. Resize converts only two decoder
kernels (transpose/spatial flip/×.25), keeping other learned tensors exact. This is
**approximate initialization**, not functional equivalence; initial outputs change.
Dedicated experimental checkpoint topology metadata prevents loading resize kernels
as the original inverse. Production model factory/defaults are unchanged.

Evidence: outputs/night_to_day_experimental/20261022_decoder_matched500/evidence.json
and all six comparison sheets. All24rows were opened. RGB L1[0,1]to identical
external teacher targets (not daytime quality): parent0.0848818746;
transpose14500=0.0846817866; resize14500=0.0876077271. Weighted teacher fidelity
is2.0456626664for transpose and2.1131421675for resize.

Resize slightly changes/smooths texture, but does NOT solve sky blobs, preserved
night halos or mixed night/day lighting. Some bright road artifacts remain/worsen.
No convincing overall visual advantage; **not selected over the current parent**.
This is a negative experimental result, not permission to declare resize universally
inferior: conversion/recovery length and training data are limited.

## More varied source supervision:111accepted of128teacher targets

All eight128-source/target training contact sheets were opened.17strong-artifact
targets rejected;111accepted, including rain and underpass examples where otherwise
plausible. Selection JSON records reasons. Every24development-night source remains
evaluated. All24validation target PNG hashes are identical between the original64
cache and this128cache; this is not a changed validation-target advantage.

Warm-start parent is existing global14,000inverse EMA, original decoder retained.
First1000additional updates reach global15,000; no original weights overwritten.
Mean24-source weighted teacher fidelity improves2.05069→**1.8288811892**.
Unlike the47-source trial, saved250/500/750/1000checks continue improving.
Evidence: E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000.
All24output rows at+1000were opened. Road71and some ordinary streets are improved,
but318still has dark sky,283still mixes night/day,12/105/214retain sky patches,
441retains large glare, and78/86show bright residual road/light artifacts.
**Not accepted as a robust convincing model.** Bounded continuation to2000updates
is underway from its saved optimizer/EMA/RNG, not a fresh restart. Keep reviewing
actual new images; agreement improvements alone cannot satisfy the goal.

## Input-conditioning controls for reference glare failures

Evidence:20261022_reference_conditioning, eight difficult sources, all eight panels
opened. Same EXTERNAL night2day weights/seeds, compared raw input versus monotone
highlight rolloff and gamma.7+rolloff. Neither transform edits output pixels or
uses a hand-drawn mask. Rolloff compresses per-pixel peak RGB with x×.7/(.7+maxRGB).

Rolloff does not remove large lamp halos in0/22/441or sky-hole artifacts in12/105.
Balanced rolloff brightens some surfaces but introduces/retains blocky color and
glass artifacts;283can gain excessive color.142overhead structure remains wrongly
sky-colored. **Rejected as a complete/general solution**, not incorporated into
public inference or the local training targets. This isolates that simple exposure
conditioning cannot resolve the missing lighting/semantic prior.

Next distinct avenue, if the larger local continuation still fails: bounded
masked diffusion restoration for glare/ambiguous regions using cached SD-Turbo,
with exact pixel preservation outside a reproducible mask and explicit external
attribution. Its feasibility/API was inspected, but no inpainting result is yet
claimed. This is a proposal to test, not a demonstrated fix or a requirement to
redownload a new foundation model.

##111-source continuation completed: global16,000

The bounded run reached adaptation2,000(global16,000)and stopped cleanly. Snapshot
adaptation_02000.ptand all previous250-step snapshots are retained. All24latest
output rows were opened. Weighted teacher fidelity=**1.8128188401**versus1.8288811892
at the previous1,000-update point: a small further gain, not a large visual jump.
The same sky patches (12/105/214), residual light/road highlights (355/78/86)and
mixed night/day case283remain.318still lacks coherent daytime sky despite brighter
surfaces.426/50/71are more day-like than the original model, but isolated good
examples do not satisfy the fixed broad visual gate. **Goal remains incomplete.**
Do not extend this111-source run indefinitely on the strength of the loss alone.
The next experiment must address persistent errors, rather than merely add steps.
The12-source lockbox and project test remain unused. No public interface/deployment
was changed. Current code checks:90tests passed; relevant Ruff checks passed.

## Masked restoration executed and rejected

Evidence: `outputs/night_to_day_experimental/20261022_masked_restoration_v1`.
All eight five-column panels were opened: night source, local inverse global16,000,
saved automatic mask, EXTERNAL SD-Turbo masked restoration strength0.5 and0.85.
Four-channel cached SD-Turbo at pinned revision b261bac6fd2cf515557d5d0707481eafa0485ec2
is not an inpainting-trained checkpoint. Four nominal steps, guidance0, seed22000+ID.
No new model download or local training. Outside-mask uint8 difference is exactly
zero in all16outputs. Peak process-allocated VRAM is approximately2.8GiB, excluding
other processes. These engineering checks do not imply preserved objects inside masks.

| Development ID | Visual observation |
|---|---|
|0|Cars enlarge/move and ghosting appears; street lighting/geometry changes, not a clean glare removal.|
|12|The foreground parked car is altered/repositioned; stronger variant invents lane geometry.|
|105|Large sky-hole artifact remains, with new large vehicle/traffic-light changes.|
|142|Stronger variant invents a tall building/skyline inside the mask; weaker one leaves inconsistent overhead lighting.|
|355|Stronger output looks more daylight-like but rewrites vehicles and road markings; weaker output retains a conspicuous lamp.|
|441|Glare/haze remains; restored region changes cars/building detail without correcting the source light convincingly.|
|78|The bright headlight-shaped road patch survives both variants.|
|283|Cars and barriers change; brighter local regions still do not yield a faithful coherent day scene.|

Reject both variants as a general improvement. The source-threshold masks cover
15.9–47.3%of these images: they are not reliably tiny glare-only regions. Simple
source intensity confuses lit surfaces and useful objects with unwanted lighting.
Exact outside-mask preservation cannot make inside-mask hallucinations acceptable.
This negative result does not prove a dedicated, structure-conditioned inpainting
model cannot work; it rejects this cached base-model/no-geometry-control recipe.

Next distinct control: synthetic night→real-day training pairs produced using the
existing frozen day→night pipeline on training-day sources, mixed with the111audited
real-night/teacher pairs. This supplies genuine daytime target structure rather
than another teacher's sky artifacts, but synthetic-to-real night domain gap must
be tested on the unchanged24real-night development images. Keep global16,000as
the warm-start parent, and a matched teacher-only continuation as the control.

## Synthetic-night / real-day matched control completed: global16,500

Prepared64 evenly spaced training-day sources only using the frozen current
day→night hybrid. All four16-pair contact sheets were opened. All64 accepted for
this bounded domain-gap control: original daylight target geometry is retained;
synthetic input glare, darkening, and weather/sun artifacts are explicitly not
claimed to be genuine night observations. The existing111real-night pairs retain
their earlier individual audit. Daytime targets are REAL training photographs,
not outputs from the night2day teacher. No inverse training on validation-day data.

Mixed manifest: `E:/LumiCycleCache/teacher_targets/night2day_mixed_synthetic64_v1`.
175train sources total; uniform sampling implies64/175synthetic probability, not
an exactly50/50balanced sampler. Validation remains the same24real-night sources
with byte-identical EXTERNAL targets. The12-source lockbox and test remain unused.

Both branches begin at the SAME existing inverse EMA global16,000, original
transposed decoder, fresh Adam for this matched comparison, same seed2030, LR1e-4,
batch2, BF16, EMA.99, clipping3, and500updates. Parent weights are not overwritten.

| Branch | Additional updates / global | Mean24 teacher fidelity |
|---|---|---:|
|Retained parent|0 /16,000|1.8128188401|
|Teacher111-only control|500 /16,500|1.7998956591|
|Mixed111real-night +64synthetic|500 /16,500|1.8370013088|

Fidelity is teacher imitation, not accuracy or the acceptance gate. Mixed250was
1.8690722386, so it improved within its short run but did not beat the matched
teacher control on that metric. This is not proof no larger synthetic curriculum
could help; it is a negative result for this bounded recipe.

Artifacts: `E:/LumiCycleCache/experiments/night2day_control111_from16000_pilot`
and `E:/LumiCycleCache/experiments/night2day_mixed175_from16000_pilot`.
All six five-column sheets opened at
`outputs/night_to_day_experimental/20261022_synthetic_mixed_matched500`.
Columns are night source, EXTERNAL teacher, parent16k, teacher-control16.5k,
mixed16.5k. All24rows reviewed, not just attractive examples.

Visual notes across the fixed cases:

- 0/355/441: persistent conspicuous lamps, halos or haze. Mixed does not fix them.
- 12/105/213: bounded sky patches/holes remain.214/142still show incorrectly
  sky-like overhead structure; mixture does not resolve semantic geometry.
- 8: dark floating-looking patch under the bright upper lamp remains.
- 318: mixed makes some upper sky a little more visible, but a dark canopy/sky
  transition and strong taillight illumination remain; not a convincing broad fix.
- 78: mixed reduces the brightness/size of the headlight-shaped road patch, but
  does not eliminate it.86/358retain road/exposure artifacts and soft vegetation.
- 283: upper image still reads as night while the road reads as day; obvious failure.
- 171/202: retained wet/glass glare and mixed illumination remain; no clear overall win.
- 71/426/50/243: useful day-like transformation already present, but gains over the
  parent/control are small and texture remains soft; no new acceptance claimed.
- 22/284/499/84: slight tone variations, retained lamp/bright-object dominance or
  softness.84was already near daylight/dusk; it cannot establish a strong conversion.

No candidate promoted to the public demo. Neither small metric gain nor one reduced
glare patch satisfies the visual gate. Both bounded processes completed cleanly;
no inverse training remains running at this checkpoint. The old16kparent and every
250/500snapshot are retained, so useful trained progress has not been discarded.
Current full Python test suite:100passed; new experiment Ruff checks passed.

## Semantic audit and external editing references — not local model gains

The cached Mask2Former Cityscapes teacher was audited on development IDs
12,105,142,214,283,0,78,426 across original night, gamma-lifted night, local16k,
and external CycleGAN-Turbo predictions. Evidence:
`outputs/night_to_day_experimental/20261022_semantic_reliability_legacy444`.
All eight panels were opened. Transformers4.44.2 loads with zero missing,
unexpected or mismatched keys. The initial5.16.1 run produced missing Swin
layer-normalization keys and is provisional only; the script now rejects such
incompatible loads. Even the clean load mistakes black sky for buildings and
overhead structures for sky, with inconsistent branches and vegetation. Do not
use these as hard ground-truth masks or validate generated geometry using masks
predicted from that same generated image. Available ACDC night labels are eight
object-detection categories, not sky/stuff semantic ground truth.

A stronger EXTERNAL reference was tested: FLUX.2 Klein4B, pinned revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, Apache2.0. This is an externally
pretrained image editor, NOT a newly trained LumiCycle model. Its15.98GB necessary
files are cached on E:, five LFS file SHA256 hashes verified. Diffusers0.41.0,
huggingface-hub1.33.0 and safetensors0.8.0 are isolated in an E: target directory;
the main project environment and day-to-night worker were not replaced.

Actual reviewed outputs:

- `outputs/night_to_day_experimental/20261022_klein_reference4_v1`: four critical
  cases, original-night editing and local16k-output editing.
- `outputs/night_to_day_experimental/20261022_klein_reference24_v1`: all24 fixed
  development cases, both modes; all six sheets opened.
- `outputs/night_to_day_experimental/20261022_klein_conditioning8_v1`: eight cases
  comparing gamma-lifted reference and native multi-reference local+lifted inputs;
  both sheets opened. Same prompt, four steps, seed61000+ID.

Daylight appearance is dramatically stronger, but objects/geometry are rewritten.
71 loses its underpass in direct editing;142/214 lose elevated structures;358/86
change highway geometry and add roadside buildings; vehicles, text and storefronts
change repeatedly. Multi-reference conditioning helps71 but does not reliably fix
142/214/358 or invented traffic. Thus neither direct editing nor local+external
editing passes scene preservation. No local inverse weights were improved by
these inference-only experiments. No FLUX output is accepted as our own model's
result, no unfiltered FLUX distillation has been performed, and no public demo,
checkpoint or model default was changed. The user's originality requirement is
explicit: continue developing our warm-started local network, not substituting
external image generation. External training assistance, if used, must be credited.

## Next controlled local diagnostic

Run a32-pair capacity/overfit check using ONLY previously reviewed synthetic-night
inputs and their REAL training-day targets, warm-starting the existing local16k
EMA. This differs from the earlier mixed175 run: it isolates whether the network
can reproduce clean daytime geometry without external inverse target artifacts.
Save eight explicitly TRAINING-only source/target/prediction diagnostics at0/250/
500/750/1000 updates and keep the same24real-night validation cases. Training-set
reconstruction is not generalization. No test or12-source lockbox use. If training
fits but real night remains poor, target/domain coverage is implicated rather
than claiming an architecture change alone fixes it. If training itself fails,
investigate the architecture/optimization before scaling data or repeating mixed
teacher training. Results follow after the bounded process and image review.

## Capacity diagnostic completed — local inverse global17,000, NOT promoted

Run: `E:/LumiCycleCache/experiments/night2day_capacity32_realday_v1`.
Same local16kEMA parent, original architecture, Adam1e-4, batch2, BF16,
1000updates (about61seconds training, plus loading/validation). No FLUX calls,
FLUX targets, or FLUX weights are used by this training process. All32 training
targets are original daytime photographs; their synthetic-night inputs were made
by our frozen day-to-night pipeline, whose external supervision is already disclosed.
The24validation targets remain EXTERNAL CycleGAN-Turbo references for comparison,
not training. Source sampling is deterministic, evenly spaced within the audited
64synthetic pairs. Exact indices, parent/cache hashes and provenance are in recipe.json.

Eight TRAINING-only diagnostic reconstructions were opened individually at1000.
Their mean weighted reconstruction loss drops2.63645 to0.66212. They regain
coherent daylight sky, building exposure, road surfaces and broad scene geometry.
The underpass training case retains its bridge. They are still soft and retain
some synthetic light patches near cars/road, so this is not perfect memorization
or proof unlimited architecture capacity. It shows that the existing network
can learn substantially stronger daylight appearance when supervision is aligned.

All24 real-night validation outputs were opened across six sheets. They do NOT
inherit this improvement: 0/355/441 retain dominating glare;71 retains nighttime
lights;142/214 show mixed overhead illumination;12/105 retain dark/patchy skies;
78/86/358 have glowing road patches and corrupted upper regions;283 still reads
as night above a partially daylike road. Other outputs remain soft or mixed-lit.
Mean EXTERNAL-target fidelity worsens from1.81282(parent) to2.41104. This scalar
is not ground truth, but agrees with the visually rejected real-night results.

Decision: retain16k as the parent, not synthetic-only17k. Evidence supports a major
domain/supervision limitation, not a claim architecture is proven sufficient or
that simply increasing synthetic steps solves it. Do not blindly scale this recipe
or silently replace our model with external editing. Next inspect native-coordinate
real paired supervision and its alignment confidence, or a distinct local
degradation-aware adaptation with controlled supervision. Existing warped Dark
Zurich cache is already rejected; any re-alignment must start from original pairs
and pass actual correspondence review before training. Test and lockbox untouched;
day-to-night deployment unchanged. Full Python suite101passed; relevant Ruff passed.

## Native-coordinate real-pair alignment audit — rejected as dense targets

Evidence: `outputs/night_to_day_experimental/20261022_native_inverse_alignment_v1`.
Six spaced TRAINING pairs0,400,800,1200,1600,2000 were regenerated directly from
the original Dark Zurich correspondence CSV. Source/cached warp order was not
reused. Cached RAFT-small C_T_V2 estimates NIGHT-to-DAY displacement and samples
the original REAL day into native NIGHT camera coordinates at512x288. Two unit
tests verify sampling direction, valid borders and inconsistent-flow confidence.
Three matching inputs: raw night, gamma.45 night and our local inverse16k output.
No externally generated RGB targets and no model training in this audit.

All six three-row panels were actually opened, covering18 alignment attempts.
0/400: pronounced melted buildings/sidewalks and viewpoint mismatch;800: moving
tram plus severe wavy facade/road geometry;1600: original reference looks along
a different street direction and dense flow cannot recover correspondence;2000:
some broad street alignment but bent facades and edges.1200 is much closer and
could contain usable regions, yet moving pedestrians, lamp bloom, occlusion and
black sky remain problematic. One partially usable pair does not approve the cache.

Fraction of pixels with FB confidence>.5 is approximately30%/33%/24% for1200
(raw/gamma/local). Other sampled pairs have at most2.26%, often effectively zero.
FB consistency is NOT correctness, nor moving-object exclusion. No hard sky
supervision is recovered by this method. Gamma/local matching is not a reliable
general fix. Retain the original dataset, reject these dense warps for training;
this does not prove all2416pairs unusable or that stronger registration cannot help.

## Distinct real-day adversarial ablation — completed, not promoted

Instead of another pseudo-target-only extension, test real TRAINING-day appearance
supervision while keeping the existing local16k generator as warm start and its
audited111source/target correspondence anchors. Two bounded1000-update variants:
GAN weight.5 and2. LSGAN local+global32-channel critics, one local scale, real
photographs at256x144, D Adam5e-5,250-step G-adversarial ramp. Existing target loss
and G Adam1e-4 remain unchanged. Real-day256sources are deterministically spaced
in the training manifest and hashed in recipe.json. Separate D-sampling RNG avoids
changing the generator source/flip sequence. Critics are newly initialized training
components; generator architecture and existing progress are retained. Generator,
EMA, both optimizers, critics and RNG states are checkpointed atomically.

Paths: `E:/LumiCycleCache/experiments/night2day_realday_gan05_from16000_v1` and
`E:/LumiCycleCache/experiments/night2day_realday_gan2_from16000_v1`.
This uses real daytime photographs, NOT FLUX generation; external CycleGAN-Turbo
correspondence targets retain their attribution. Neither falling adversarial loss
nor a critic score can establish acceptance. Compare500-update outputs against
the existing matched teacher-only500control and inspect all24cases plus1000-update
trajectory. No candidate is promoted before actual visual review. Main day-to-night
checkpoint/default/UI/worker remains unchanged, test and lockbox untouched.

Both processes completed1000updates cleanly, each global17,000, peak process
allocated VRAM1.314GiB (not whole-system GPU usage). All24cases were opened in
both the matched500comparison and final1000trajectory: twelve sheets in total.
Matched500evidence: `outputs/night_to_day_experimental/20261022_realday_gan_matched500`.
Final1000evidence: `outputs/night_to_day_experimental/20261022_realday_gan_final1000`.

At500, mean RGB teacher L1 on[0,1] is0.074503(control),0.075120(GAN.5),
0.075296(GAN2). At1000, weighted teacher fidelity is1.802011(GAN.5),
1.827007(GAN2), versus parent16k1.812819. These are imitation scores, not daylight
quality; the1000comparison uses the original16kparent and is NOT a matched
1000-update teacher-only control. Do not interpret that small difference as proof
of adversarial superiority. The correctly matched500control is available above.

Actual observations: all three branches retain severe142/214overhead sky errors,
12/105sky holes and8dark floating patch.0/355/441still contain dominating lamps,
halos or washed-out glare;78/86/358still contain bright road pools.283retains
mixed nighttime trees/traffic and daylight pavement.71/426/50/243remain more
daylike but not a new general improvement. Some tone/contrast shifts appear,
especially with GAN2, yet neither weight resolves the recurring defects. Broad
vehicle/street layout is much steadier than external FLUX editing, but this does
not compensate for incomplete lighting conversion. Neither candidate meets the
visual gate; preserve all snapshots and keep16k as the conservative parent.

Next distinct hypothesis: the strong uniform teacher correspondence loss still
penalizes removal of teacher-retained nighttime halos and malformed sky. Test
degradation/reliability-weighted target supervision, first auditing its actual
regions rather than calling source brightness a semantic mask. Compare against
the saved uniform-target GAN control, keep warm-start weights and fixed suite,
and avoid copying nighttime gradients or scaling the same rejected GAN recipe
indefinitely. Real paired data requires better registration/data curation before
dense loss; no universal unusability claim follows from six audited pairs.
Current full suite104passed, relevant Ruff passed. No inverse training remains
running after the two bounded processes returned exit0. Public worker untouched.

## Retained-glare reliability ablation — completed and rejected

Implementation: `daynight/inverse_reliability.py`, opt-in experimental trainer
`--target-reliability retained_glare`. This is NOT semantic segmentation, a sky
matte, or the N2D3 algorithm. Source cores require maxRGB>.95 and luma>.6. A17px
max-pool and9px average support is multiplied by target brightness ramp.45→.70.
Target reliability falls only to.05, never zero. RGB/coarse/gradient target losses
are weighted with full-image denominators; adjacent-gradient confidence uses the
minimum of both endpoints. Validation retains the ORIGINAL uniform loss.
No source-RGB or source-gradient preservation penalty is added.

Before training all24development overlays and eight evenly spaced audited TRAIN
overlays were opened: `outputs/night_to_day_experimental/20261022_glare_reliability_audit_v1`
and `.../20261022_glare_reliability_train8_v1`. These show lamps AND reflective
signs, license plates, vehicle parts and storefronts. The mask does not identify
all road-light pools, broad haze or sky semantics. Validation pixels below.5weight
range0.39–7.99% (mean3.30%); below.95 range0.97–17.18% (mean6.00%). Approval was
only for a bounded soft-weighting experiment, not a claim of mask correctness.

Run: `E:/LumiCycleCache/experiments/night2day_retained_glare_gan2_from16000_v1`.
Existing inverse16kEMA, same111audited teacher training pairs, same256real training
day photographs and source sampling as uniform GAN2 control,1000updates, G1e-4,
D5e-5, BF16batch2, EMA.99, GAN2 with250-step ramp. No generator architecture reset,
no FLUX calls/weights/targets in this run, no new data downloads. Approx88seconds
training, process peak allocated1.311GiB. Actual uniform validation teacher loss
1.831668 versus uniform GAN2 control1.827007 and parent1.812819, not a quality gate.

All24matched final comparisons were opened at
`outputs/night_to_day_experimental/20261022_glare_reliability_matched1000`.
Sky holes12/105, overhead errors142/214, dark patch8, mixed night/day283 and
glare0/355/441 persist. Worse, small cyan/yellow checkerboard speckles appear
around bright lamps, plates and road patches (notably0/71/78/318/358). The native
512x288 output78 was separately opened to confirm these are actual model artifacts,
not contact-sheet resizing. Relaxed supervision leaves the transposed decoder
and low-resolution critic free to produce unwanted bright high-frequency patterns.
That mechanism is a hypothesis supported by pattern/location, not a proven causal
separation of decoder versus critic. Neither glare removal nor a general daylight
improvement was achieved. Reject this candidate and keep the16k parent.

Three tests cover all-ones equivalence (value AND gradient), bounded detached
source/target mask conditions, and nonzero finite relaxed supervision. Full suite
107passed. Do not simply broaden this mask: its false positives and failure to
supply the missing correct appearance are already visible.

## Spatial scale/context diagnostic — no inference workaround

`daynight/audit_inverse_scale.py` ran the SAME retained16kweights at512x288,
256x144 and128x72 on all24development inputs. Smaller outputs are explicitly
upsampled for display; native outputs are saved separately. No training, external
RGB generation, test or lockbox use. Six sheets actually opened at
`outputs/night_to_day_experimental/20261022_inverse_scale_audit_v1`.

256loses detail and often increases glare or leaves darker sky;128is heavily soft,
warps ambiguous object appearance and often reads more like illuminated night.
105does not acquire a coherent whole sky;142/214still mis-handle overhead context;
78/86/358retain bright road artifacts;283remains mixed-lit. Some lamp/sky patches
change extent with scale, but no consistent daylight conversion emerges. Reject
lower-resolution inference as a workaround. This does NOT prove global context
is unnecessary or that a trained multi-scale adapter cannot help; it rejects an
untrained input-resize substitution. Day-to-night production remains unchanged.

Next data-directed option: mine a bounded set of original coarse real pairs for
genuinely useful overlap, rather than treating the six mostly failed samples as
proof the entire dataset is unusable.1200showed usable-looking static regions;
any selected confidence regions still need visual correspondence, moving-object
and illumination review. Do not train all coarse pairs or trust flow confidence
as ground truth. All parent/failed snapshots and comparison evidence are retained.

## Broader real-pair mining and four-region training control — completed

`daynight/mine_inverse_pairs.py` scanned96evenly spaced unique-day-reference
TRAINING pairs from the original Dark Zurich CSV. Each pair uses raw and gamma.45
night matching with cached RAFT-small; real day is sampled in native night camera
coordinates at512x288. Rank combines forward/backward consistency and a local
flow-deformation penalty, NOT an accuracy guarantee. No new model/data download
or external RGB generation. Evidence/cache:
`E:/LumiCycleCache/teacher_targets/night2day_real_pairs_scan96_v1`.
All top24ranked panels were opened, not merely the best two. Lower72were computed
but not individually visually reviewed or approved. Two tests verify training-only
unique-day sampling and the deformation score on constant versus wavy flow.

Important data discovery: the best-ranked2401(score.9274),2354(.5810),2309and
several other day-directory references visibly depict blue-hour/twilight with
dominating lamps. A high overlap score does not make them appropriate daylight
targets. Reject these as full-day supervision. Other candidates show moving trams,
vehicles/pedestrians, warped buildings, bends in rails or narrow reliable overlap.
No full-image pair was approved. This does not assert all Dark Zurich day images
are twilight or that the entire dataset is unusable. Exact24review decisions are
saved in the cache's visual_review.json.

Four explicit training annotations were prepared with
`daynight/prepare_inverse_static_regions.py`:408left facade/vegetation edge,
548right facade,190right static wall,1739upper-right building/overhang. Rectangular
ROI intersected with confidence>=.6; actual supported fractions are6.00%,3.99%,
2.61%,.61%. All four source/reference/mask/used-pixel panels were opened BEFORE
the explicit approval command. These are manual training annotations, not automatic
semantic masks, demonstration-time edits, or complete paired ground truth. They
exclude the obvious moving traffic and cannot supervise sky or glare removal.
Cache: `E:/LumiCycleCache/teacher_targets/night2day_real_static4_v1`.
Every source, warped target and mask is hashed; training rejects unapproved or
changed bytes. Underlying RAFT correspondence remains externally pretrained and
credited; RGB targets are real photographs, NOT generated images.

Controlled continuation: same16kEMA parent, original generator, same111audited
teacher-target primary batches, fresh Adam1e-4, BF16batch2, EMA.99, clip3,
500updates. Add one independently sampled real-region pair per update with
3xconfidence-mass-normalized RGB-L1; no extra GAN. The AUXILIARY loss has zero
direct output gradient outside its static support, but shared network parameters
can still affect other pixels; the primary teacher loss continues elsewhere.
Separate saved RNG preserves the main input/flip sequence against the saved
teacher-only500control. All checkpoints preserve model, EMA, optimizer and RNG.

Run: `E:/LumiCycleCache/experiments/night2day_real_static4_from16000_v1`, completed
global16,500 in~45seconds training, process peak allocated1.736GiB. Four TRAINING
diagnostic outputs were opened. Mean annotated-region RGB-L1 in normalized[-1,1]
falls.45645→.04013; walls/facades are better supervised, but these training scenes
still contain artificial light dominance and patchy skies. This is learning on
known annotations, not generalization.

All24development comparisons were opened across six sheets:
`outputs/night_to_day_experimental/20261022_real_static4_matched500`.
0/355/441retain glare;142/214still have incorrect overhead sky patches;12/105
retain sky holes;8dark patch persists;78/86/358still have road illumination pools;
283remains mixed nighttime foliage and daylike pavement. Slight tone changes and
some facade exposure shifts do not constitute a robust daylight improvement.
Uniform teacher fidelity1.829988 versus matched control1.799896 and parent1.812819.
Reject candidate promotion; keep the16k parent and retain curated regions for
future use.110tests passed; relevant Ruff passed. Public day-to-night unchanged,
test/12-source lockbox unused, bounded inverse process returned exit0.

Conclusion from this trial is narrow: four tiny static regions cannot establish
whole-scene daylight. Do not scale their step count blindly or accept twilight
references based on overlap confidence. Further data must cover actual daylight
sky/ambient and artificial-light changes, with correspondence beyond isolated
facades. No new successful inverse model or completed goal is claimed.

## Fixed-camera real sky/building supervision — bounded controls

Opened all14 TRAIN-camera corrected-clock DNIM contact sheets under
`outputs/sky_pilot/timelapse_corrected_clock`. The three previously assigned
validation cameras00003548/00004232/00011160 were NOT opened or used in this
experiment. The existing cached day-to-night warped/pseudo-mask pairs were not
reversed and reused: old semantic loading and camera registration are not assumed
correct. No new images/weights were downloaded. Original Kaggle DNIM files stay
unchanged and are not redistributed.

Prepared36real-night/real-day pairs, four nighttime frames from each of nine
training cameras, preserving aspect ratio to512px width with height rounded to
a multiple of4. Targets are ORIGINAL real daytime photographs from the corrected
clock candidate list, not diffusion/GAN RGB or temporal composites. Explicit
manual camera-level upper-region annotations select42–70%of image height for
sky/buildings/hills; lower foreground traffic/plazas are excluded. These broad
masks are imperfect: tiny background people/vehicles, foliage movement, clouds,
weather and small camera jitter can remain. They are not semantic ground truth,
perfect registration, or a claim of recovering invisible geometry.

Training-camera exclusions:00001323severe night lens glare;00003837night banding;
00019607day/night view/scale mismatch;00010888traffic-heavy intersection without
sky;00011331near-black geometry/variable river deferred from this small pilot.
All36actual prepared source/reference/mask rows were opened across nine sheets.
Five DAY targets with direct sun or washed-out lens flare were rejected:
300010/300012/300013/300019/300027.31pairs retained, across the same nine cameras.
The rejection records/files remain saved, not silently deleted. Cache:
`E:/LumiCycleCache/teacher_targets/night2day_timelapse_static36_v1`.
The approved hashed regions.json contains31training records and five rejected
records. Repeated approval cannot rewrite the audit manifest. Two tests verify
camera-training boundaries and corrected-clock restrictions.

Controlled continuation retains local16kEMA, original generator and111audited
teacher-pair primary batches. Each update adds one independently sampled REAL
camera pair's annotated-region loss, weight3. No extra GAN, FLUX weights/calls/
targets, or architecture reset. Primary external CycleGAN-Turbo supervision still
has its earlier attribution. Fresh Adam1e-4, BF16batch2 primary/1auxiliary, EMA.99,
clip3; auxiliary RNG checkpointed independently. A newly run1000-update
teacher-only control uses the same parent and primary sampling, enabling a matched
global17kcomparison. Public day-to-night inference is never routed through these.

First variant uses confidence-normalized RGB-L1 on avgpool4/8 outputs to tolerate
minor registration errors. New tests cover its constant-offset/finite-gradient
behavior. Training diagnostic panels now adapt to actual source height rather
than clipping these512x384frames to the old288px canvas. Diagnostic reported
`mean_static_rgb_l1` is full-resolution and is NOT the coarsened training objective.

- Coarse: `E:/LumiCycleCache/experiments/night2day_timelapse31_from16000_v1`.
 1000updates completed global17k, elapsed loop~162s including interim validation
 and GPU contention with control; process peak allocated1.898GiB. Training-region
 full RGB diagnostic.56802→.20238. Nine representative training-camera outputs
 were opened at500; they regain sky/ambient color but retain strong noise, haze
 or nighttime lamps, especially850/9291/9320/9403/21510. Do not call those outputs
 clean daytime or claim all31training predictions were reviewed.
- Matched teacher-only control:
 `E:/LumiCycleCache/experiments/night2day_teacher_control1000_from16000_v2`.
 1000updates completed global17k; uniform validation teacher loss1.804707 versus
 coarse1.853844 and parent1.812819. Metric is imitation, not acceptance.
- All24matched development cases opened across six sheets at
 `outputs/night_to_day_experimental/20261022_timelapse31_matched1000`.
 12/105sky holes,142/214overhead errors,8dark patch,0/355/441glare and
 78/86/358road pools remain.202's broad light road patch is still conspicuous;
 283remains mixed night/day. Some tones change, but no general improvement.
 Coarse variant and control are not promoted;16kremains the conservative parent.

A second matched variant on the SAME31pairs uses pixel RGB rather than coarse
RGB within the approved regions, to test whether coarse-only supervision leaves
fine sensor artifacts unconstrained. This is a bounded1000-update trial, NOT a
long blind extension. Path:
`E:/LumiCycleCache/experiments/night2day_timelapse31_rgb_from16000_v1`.
Its completion and actual visual review must be recorded before any acceptance.
Full Python suite113passed before this last run; relevant Ruff passed. Test and
12-source BDD lockbox remain untouched; DNIM held-out cameras remain excluded.

### Pixel-RGB control completed and reviewed — not promoted

The second run completed1000updates, global17k, elapsed loop~96s including interim
validation, process peak allocated1.898GiB. Same31pairs, manual masks, original16k
parent, sampling/optimizer/loss weight as coarse; only auxiliary RGB-vs-coarse
objective differs. Mean TRAINING full-resolution static RGB diagnostic decreases
.56802→.19783, compared with coarse.20238. Uniform development teacher fidelity
is1.845791 versus coarse1.853844 and matched teacher-only control1.804707. These
small imitation differences do not establish visual improvement.

All24matched road-scene cases were opened at
`outputs/night_to_day_experimental/20261022_timelapse31_rgb_vs_coarse1000`.
Six sheets show source, explicitly EXTERNAL reference, teacher-only17k,
coarse17k and RGB17k. No recurring failure is convincingly resolved:12/105sky
holes,142/214overhead confusion,8dark floating patch,0/355/441glare and
78/86/358road pools remain.202still has a broad bright road patch;283still reads
nighttime above a brighter road.71/426/50/243remain relatively better cases but
not a new general gain. Vehicle/street layout broadly remains in place.

Nine representative RGB TRAINING outputs, one per retained camera, were opened
at1000:300000/300004/300008/300014/300016/300020/300024/300028/300032.
Some exposure/sky changes are learned, but severe sensor-texture amplification,
striping/patchiness, soft buildings and retained artificial-light dominance remain.
The castle case9403has invented-looking smeared texture over almost unobserved
areas; tower5381still retains a conspicuous bright patch. These are not accepted
daytime transformations. Only these nine representative final training outputs
were opened, not every one of31. All36DATA pairs/masks were separately opened
before selection, which is a different review claim.

Neither timelapse branch nor the teacher-only control is promoted. Retain16k as
the conservative parent; all checkpoints and data decisions preserved. This
bounded negative does NOT prove real paired supervision cannot help: it exposes
the limits of nine low-quality/static views with nearly black sensor inputs,
partial scene support and a strong old pseudo-target anchor. Do not blindly
increase steps or infer that a new architecture alone solves those limitations.
Next distinct diagnostic can isolate input-noise/normalization sensitivity on
the retained generator before adding further losses, or improve target coverage
with a separately justified experiment. No external RGB editor may be silently
substituted.113tests passed; relevant Ruff passed; all three bounded handles
returned exit0 and only the original public remote-worker process remains live.

## Frozen-generator noise sensitivity and precision controls — 22 October 2026

User requested a research reset because the inverse outputs were excessively
noisy. The preceding research turn was progress: primary-method/config inspection
changed the next action from another blind continuation to degradation isolation;
see `NIGHT_TO_DAY_RESEARCH_RESET_2026_10_22.md`.

Implemented `daynight/audit_inverse_noise.py`, an inference-only diagnostic on
the SAME trained16kEMA inverse. Exact24development membership and original source
hash metadata are checked against the frozen suite. No test/lockbox inputs,
training, outside RGB generator, weights downloads or public app changes.

Variants: unchanged source; mild5x5RGBbilateral input (spatial sigma1,
range sigma.04); mild5x5Gaussian input (sigma.6); bilateral OUTPUT control. The
output control distinguishes cosmetic smoothing from changed model behavior.
All sources are the same cached512x288RGB. Native PNGs and three fixed-coordinate
2xnearest-neighbor crop rows are saved; crop coordinates are NOT semantic labels.

First run terminal exit0:
`outputs/night_to_day_experimental/20261022_noise_sensitivity_v1`.
All24rows opened across six full-image sheets, plus native-resolution crop sheets
for358/105/243. Bilateral changes fine texture; Gaussian often softens objects;
neither removes0/355/441glare,142/214overhead misinterpretation,8dark floating
patch,105sky hole,78/86/358road illumination pools,202broad road patch, or283mixed
night/day.12still has upper sky/foliage artifacts.71/426/50remain relatively
daylike but not a new robust gain.284/84/318retain strong lamps/vehicleglow.
No filtering variant is accepted or deployed. Do not infer that a trained
restoration system cannot help merely because these hand-designed filters fail.

For every source, also add seeded Gaussian perturbations sigma1/255 in[0,1]sRGB
and measure output/inputRMSchange. The noise draw is synthetic and NOT an estimate
of true camera noise. Repeat unchanged-inputBF16inference is exactly equal in
these paired executions (maxdifference0), excluding repeated random generation
as an explanation for their visible texture in this setup. BF16sensitivity ratio
mean4.67489,median3.17965,range1.19642–14.14948. Sensitivity ratios are neither
daytime quality scores nor proof of a particular normalization/decoder fault.

Second run addedFP32controls, actual terminal exit0:
`outputs/night_to_day_experimental/20261022_noise_sensitivity_precision_v2`.
FP32meanRMSratio4.60040,median3.24947,max13.52727. Thus input sensitivity persists
withoutBF16. MeanBF16vsFP32baselineRGBRMSdifference.00115521,max.00190072 on[0,1].
FP32native outputs358/105/243opened: same grainy/patchy sky, large unsupported
sky hole105, retained lamps/road patterns; no precision-only cure. All24filter
comparisons were reviewed in the first run; only those three additionalFP32
native outputs were opened, not all24FP32outputs. Do not overstate review scope.

Across24: mean inputbilateralRMSchange.00499345; resulting model-outputRMSchange
.02260718; postbilateraloutputchange.00691599. This supports input sensitivity
but does not establish sensor-noise causality: filter-induced feature changes,
learned texture and underconstrained illumination may all contribute.

Next experiment should distinguish restoration robustness from daylight-domain
supervision, rather than present a blur filter or full precision as a solution.
Any adapter must retain parentweights, begin as identity, use justified training
supervision and face the unchanged whole-image visual gate. More trustworthy
paired coverage remains a separate unresolved need; DarkDriving author-hosted
download has not been started under the earlier Kaggle-only data preference.

Five added diagnostic tests pass (constant/shape,boundary preservation,seeded
noise reduction/determinism,splitmembershipguard,invalidparameters). FullPython
suite118passed and relevantRuffpassed. These are engineering checks, not visual
acceptance. No inverse checkpoint promoted; trained16kparent retained.

## Learned noise robustness: matched500-update continuation

The preceding goal turn was progress: frozen-weight controls demonstrated input
sensitivity inFP32as well asBF16and ruled out mild filtering as a full solution.
This turn tests trainable robustness separately from changing the backbone or
obtaining new paired data. It is NOT a denoiser reproduction or new architecture.

Added optional synthetic-noise training to `train_night_to_day_pilot.py`. For
the SAME111auditedtraininginputs and SAMEexternalCycleGAN-TurboRGBtargets:
loss=.5pilot(G(x),y)+.5pilot(G(x+noise),y)+2L1(G(x+noise),stopgrad(G(x))).
InputnoiseisGaussian sigma2/255in[0,1]sRGB, equivalent4/255inmodel[-1,1], clipped
tolegalrange. IndependentCUDA RNG2033 is checkpointed/restored; primary sampler
and flips retain Pythonseed2030. No pseudo-clean blur images are introduced.
This encourages perturbation robustness but cannot establish true sensor-noise
removal, and continues to inherit imperfect external-target lighting.

The matched control also evaluates two branches but sets sigma0: target weight
is unchanged and consistency loss is exactly0. Tests verify it has the same
loss and gradients as the original pilot. Both retain ALL16kEMA generatorweights,
original architecture, freshAdam1e-4(.5,.999), batch2, BF16, EMA.99, clip3,
500updates/global16,500. No extraGAN/cycle/semanticloss/newRGBgenerator. Both
actualhandles returnedexit0; elapsedtrainingloops~137scontrol/~135snoise including
interimvalidation/GPUcontention, peakprocessallocated2.220/2.225GiB.

Checkpoints preserved:

- `E:/LumiCycleCache/experiments/night2day_noise_control_from16000_v1/adaptation_00500.pt`
- `E:/LumiCycleCache/experiments/night2day_noise_robust_from16000_v1/adaptation_00500.pt`

All24parent/control/noisecomparisons opened across sixsheets:
`outputs/night_to_day_experimental/20261022_noise_training_matched500`.
Nativecontrolandnoiseoutputs358/105/243alsoopened. Broadgeometryremains similar;
some texture/tone changes, but persistent failures remain:0/355/441glare,
142/214incorrectoverheadlighting,8darkpatch,105largeartificialskyhole,
12sky/foliageartifact,78/86/358roadlightpools,202broadroadpatch,
283nightupper-scene.84/284vehiclelampsand318redglowremainstrong.
71/426/50remainrelativelydaylike without a generalbreakthrough. Native358still
grainy;105stillbroken;243stillsoft. Neither branch passes whole-image acceptance.

Freshfrozenevaluation on the SAME24seeded1/255perturbations:

| Weights | Mean FP32 output/input RMS change ratio | Mean BF16 ratio | Uniform teacher fidelity |
|---|---:|---:|---:|
| Parent16k | 4.60040 | 4.67489 | 1.812819 |
| Matchedcontrol16.5k | 4.48783 | 4.57968 | 1.808822 |
| Noisetraining16.5k | 2.58498 | 2.61213 | 1.821651 |

About42.4%lowermeanFP32sensitivityvscontrol is real progress on this synthetic
robustness diagnostic, NOT42.4%betterimagequality. Slightlyworse imitation is
not decisive either. Clean-sourceRGBL1toexternalreference is.075045/.074871/
.075410(parent/control/noise);againnotadaytimeacceptancescore.

Sensitivityevidence:
`outputs/night_to_day_experimental/20261022_noise_control500_sensitivity`
and `20261022_noise_robust500_sensitivity`. Their additional filter variants were
generated but NOT separately visually reviewed; the matchedclean-outputreview
above is what supports the visual verdict. No test/lockboxused.

Threeaddedtests coverzero-noisegradient-equivalence, independentRNGreplayandrange,
anddetachedconsistencyanchor. FullPython121tests passed; relevantRuffpassed.
Newrecipefields intentionally mean oldpilots cannot silently resume under a
changedrecipe. No claimofend-to-endresumeequivalencewas established bythese
unitchecks. Originalpublicday-to-nightworkerremains theonlylivePythonworkerpair.

Decision: retainnoisebranch as a useful robustnessablation, not defaultparentor
acceptedmodel. Do not extend blindly: the currenttargets still teach lighting
defects. A question was sent asking whether author-hostedDarkDriving may be used
onE:despitetheearlierKaggle-onlypreference; no download or license acceptance was
performed. Betterdaylightsupervision and frequency/contentseparation remain
unresolvednextdirections, withtrained16kparentpreserved.

## Kaggle restoration-data acquisition and frozen-model audit

Previous goal turn was progress: a matched robustness continuation reduced
synthetic-noise sensitivity but failed actual daylight acceptance. This turn
targets broader photographic supervision rather than repeating that pilot.
See `NIGHT_TO_DAY_RESTORATION_DATA_AUDIT.md` for source links, download commands,
licensing limitations, exact paths and visual findings.

Completed Kaggle-only downloads on E: (both exit0): LOLv1 and author-linked
LoLI-Street v1. No outside trained RGB generator downloaded/run. LOL has unknown
Kaggle dataset license and no found archive LICENSE/README: inspection-only, NOT
training. Do not assume RetinexNet code license licenses its images. LoLI-Street
has author-linked CC BY-NC4.0 metadata, but its TRAIN pairs are synthetic
Photoshop darkening, NOT real night/day captures. No DarkDriving author download
started; user source-choice question remains unanswered.

New `audit_restoration_pairs.py` matched exact TRAIN input/target names:
485LOLpairs and30000LoLIpairs. Deterministic24TRAINpairs per dataset were
rendered with the existing16kEMA inverse and all six sheets per dataset opened.
Audit paths `20261022_LOL_training_audit24` and `20261022_LoLI_training_audit24`
under `outputs/night_to_day_experimental`. These are data/training-source
diagnostics, not new BDDvalidation or accepted model outputs. Test/validation
photographs from the new datasets were not opened; zip extraction does not mean
evaluation. No checkpoint was trained on these images.

LOL shows heavy noise amplification on flat walls/ceilings and artificial
foliage/sky-like textures behind objects. Mean TRAINdiagnosticRGBL1.145761.
LoLI shows useful clean photographic road/sky/building/detail targets, but current
inverse overexposes already-daylike low-darkening inputs, smears clouds, changes
color/edges and introduces bright road patches. Mean TRAINdiagnosticRGBL1.090590.
Neither number is a generalization/daytime metric, and crossdataset ranking is
invalid. Only24samples per dataset were visually reviewed, not all pairs.

`index_loli_training.py` completed exit0 on all30000TRAINtargetfilenames:
9457unique target SHA256groups, illustrating why darkening variants cannot be
split independently. Zero groups within dHashdistance4of the3000existingBDD
val/test metadata hashes. This is a conservative overlap screen, NOT proof of
independent content/cameras; no heldoutBDDimagepixels were opened. Index is
audit-only at `E:/LumiCycleCache/teacher_targets/LoLI_train_index_20261022.json`.
No existing split or image changed; no test metrics computed.

FullPython129tests passed; relevantRuffpassed. Only public remote-worker process
pair remains live. Day-to-night app/model/deployment unchanged, inverse16kparent
preserved. Next safe independent experiment can use a grouped, hashed LoLItraining
subset with clean photographic targets and a matched continuation control,
explicitly testing synthetic-to-real transfer on the SAME24BDDdevelopmentcases.
Do not claim synthetic exposure training alone will remove nighttime illuminants,
train on unresolved-licenseLOL, or reset the backbone. Whole-image gate unchanged.

## LoLI photographic-target continuation: matched1000updates completed

Previous turn was progress: new Kaggle data acquired, actual frozen-model outputs
reviewed, and all30000LoLITRAINtargetfiles indexed to9457byte-hash groups with
the savedBDDholdout-hash screen. This turn tests that data, not another noise
augmentation tweak. Original trained16kEMA inverse remains the warm-start parent.

New `prepare_inverse_loli_regions.py` selects96distinct targetSHA groups evenly
across sorted hashes, independent of visual quality or BDDdevelopment results.
Cycles dense/moderate/light darkening preference, selecting one variant per
target group. Existing index excludes any heldout-similarity group; protected
BDDmanifest hashes checked again. No new split, heldoutpixels or LoLITest/Val
used. No new downloads. Source/reference shapes match; targets are REAL street
photos, sources SYNTHETIC darkening with persistent daytime shadows/clouds.
Full-pixel white masks mean paired support, not semantic confidence or real-night
correspondence. All96source/reference rows opened in six16-pair sheets before
explicit hashed approval. No appearance-based rejection was made.

Cache `E:/LumiCycleCache/teacher_targets/night2day_LoLI_static96_v1`:
regions.jsonSHA256 `75bc844f0ada25bb825e012a4b6711f30151eb1e0c3e6c9835566eb30ee6ac78`.
The cache/auxiliary code retains historical `static`/`real_regions` field names;
for this dataset they mean all-pixel synthetic-low/real-day pairing, NOT actual
static nighttime captures. Diagnostic source/target labels were corrected in the
trainer using per-row labels; the legacy mask heading says Static supervision
ONLY but the saved cache explicitly defines its meaning. Do not misrepresent it.

Both runs: unchanged original generator, warmstartALL16kEMAweights;111previously
auditedreal-night/EXTERNALCycleGAN-Turbotargetprimary batches, batch2; freshAdam
1e-4(.5,.999),BF16,clip3,EMA.99. One independently sampled LoLIpair batch1 each
step, ownPythonRNG2032andflips/checkpointstate. Addedloss is RGBL1normalized by
white-mask mass, weight3candidate/weight0control. No newGAN/cycle/semantic/noise
objective and no architecture replacement. The weight0control executes the same
auxiliary path but contributes zero gradient. Primary target loss unchanged.

Completedactual1000updates/global17k, both handles exit0:

- `E:/LumiCycleCache/experiments/night2day_LoLI96_control_from16000_v1/adaptation_01000.pt`
- `E:/LumiCycleCache/experiments/night2day_LoLI96_aux3_from16000_v1/adaptation_01000.pt`

Elapsedtrainingloops~267/~268seconds includinginterimvalidationandGPUcontention;
peakprocessallocated2.1365GiB. No claim these are exclusiveGPUthroughput numbers.

| Branch/step | Uniform BDDdevelopment teacher fidelity | Mean LoLI TRAINphoto RGBL1 |
|---|---:|---:|
| Parent16k | 1.812819 | .172399 |
| Control+500 | 1.820796 | .178427 |
| Auxiliary+500 | 1.781531 | .078141 |
| Control+1000 | 1.801549 | .179225 |
| Auxiliary+1000 | 1.747623 | .071747 |

Lower imitations/training errors do NOT establish convincing daytranslation.
Mean0..1RGBL1toEXTERNALBDDreference at1000is.074575control/.072319auxiliary.
Neither is a real BDD daytime ground-truth reconstruction metric.

All24parent/control/candidate comparisons opened at BOTH500and1000:
`outputs/night_to_day_experimental/20261022_LoLI96_matched500` and
`20261022_LoLI96_matched1000`. Relatively small tone/texture changes, but no robust
solution to0/355/441glare,142/214overheaderrors,8darkfloatingpatch,
105large unsupportedskyhole,12sky/foliageartifact,78/86/358roadlightpools,
202broadroadpatch,283mixednightupper-scene.84/284/318still retainconspicuous
vehicle/lightglow.71/426/50are relativelydaylike already, notnewgeneralwins.
Nativecandidate105/358/8at1000opened: defects are real, not sheet-resize illusions.

TRAINpredictions opened:400013/400040/400091at250,400010/400040/400091at1000.
They show improved exposure matching and preserved broadgeometry but softened
wires/foliage/tunneltexture and residual photometric errors. Onlythreeprediction
panels per checkpoint were opened, NOT all96. All96DATApairs were separately
reviewed before training, a different claim. Neither branch meets acceptance.

Decision: no publicpromotion and no blindextension of thissame1000-steprecipe.
Preserve both checkpoints as a synthetic-to-real supervision ablation; retain
conservative16kparent. The result does not prove cleaner data cannot help, nor
that1000steps establishes a convergence ceiling. It shows thisauxiliary objective
does not yet resolve the observed real-night failure modes.

Next distinct architecture hypothesis: test a spatial scene-wide context adapter
initialized at zero influence around the EXISTING learned backbone; preserve its
weights and verify initial functional equivalence. CBAM global channel pooling
does not by itself provide explicit long-range spatial feature exchange. Compare
against these same-data controls, not a new target mixture simultaneously. This
is a proposed test, not an implemented architectural improvement or guaranteedfix.
Do not repeat the failed inference-resize/approxdecoderconversion as that test.

FullPython131tests passed; relevantRuffpassed. Twoaddedtests cover deterministic
target-group selection, cyclicdarknesspreference, exclusion and rejection of any
heldoutrow. Existingproductionday-to-nightmodel/worker/deploymentunchanged;
no test/lockbox used. LOL remains license-gated and DarkDriving source-choice
question unanswered; neither was used fortraining. Goal not achieved.

## Spatial scene-context adapter: implemented, trained and rejected for promotion

Previous goal turn was progress: same-data LoLIcontinuations completed and all24
outputs reviewed, exposing persistent real-night defects despite better paired
exposure fit. This turn implements the separately proposed architecture test,
not a silent default change. Fullarchitecture/migration/control/visionrecord:
`NIGHT_TO_DAY_SCENE_CONTEXT_ABLATION.md`.

Preserved every11,447,541-parameterbackbonetensor; added198,784parameter8x8pooled,
128channel/fourheadspatialattentionresidualafter9ResNetblocks. Zerooutputprojector
initialization preservesfunction; actual24savedinitialCUDA/BF16outputs have0pixel
difference from original16k. CPUidentity/RNG/gradients/nonlocalconnection/loader
tests pass. Not a segmentationmodel, physicalrenderer, diffusionbackbone or new
pretrainedmodel. Only privateexperimentalpaths load this architecture.

Same111primaryexternal-targetpairs +96synthetic-low/real-dayLoLIauxiliarypairs,
weight3, parent16kEMA, fullbackbone+adaptertrainable, unchangedAdam1e-4/batches/
loss/clip/EMA.1000updates completeglobal17k, terminalexit0, loop124.468s including
validation; peakprocessallocated2.171995GiB. No new data/noise/GAN/cycle objective.
Saved at `E:/LumiCycleCache/experiments/night2day_context_LoLI96_from16000_v1`.
Matchedcontrol is priorLoLIweight3/no-contextcheckpoint, NOTteacheronlycontrol.

All24comparisons opened at500and1000 in
`outputs/night_to_day_experimental/20261022_context_LoLI96_matched500`
and `20261022_context_LoLI96_matched1000`.1000alsoincludesON/OFFsame-trained-
checkpointcontextablation. OFFretains othertrainedweights and is not retrained.
Adapter learnednonzero weights; ON/OFFmeanRGBL1difference.0191594provesoutput
influence, notquality. Uniformteacherfidelity1.733452vsno-context1.747623;
LoLITRAINphotoL1.070042vs.071747. These smallimprovements do not pass visualgate.
Severe skyholes/overheaderrors/floatingdarkpatch/glare/roadpools/mixednightupper
scene persist. NativeON105/8/358opened and confirmdefects. No newtraining-
predictionpanels reviewed in thisrun; do not claim all96predictions inspected.

Notpromoted; retain16kparentandallcheckpoints.135Python tests and relevantRuff
pass. Onlyoriginalpublicworkerpairlive;publicday-to-nightunchanged.Test/lockbox
untouched. Next distinctmethod should address illumination/degradation-aware
supervision rather than blindlyadding contextcapacity or extending thisrecipe.

## Photometric degradation-map audit — 22 October 2026

See [research verdict](NIGHT_TO_DAY_DEGRADATION_RESEARCH_VERDICT.md) for method,
primary sources, measured memberships and limitations. Opened all six sheets in
`outputs/night_to_day_experimental/20261022_photometric_regions_val24` and all
three in `20261022_photometric_regions_train12`: 24 fixed development sources
and 12 training sources. Maps often follow object edges/compression structure,
while missing broad lighting effects. Reject as hard removal/confidence masks.
Soft contrastive grouping remains an untested hypothesis, not an implemented
N2D3 training system. No training, promotion or production change in this audit.
140 tests and relevant Ruff passed; test and reserved lockbox untouched.

## Unpaired adversarial versus contrastive continuation — 22 October 2026

Executed two matched 500-update v3 pilots from the same16kEMA parent with all
5,000+5,000 BDD training entries available; no external RGB targets. See
[full recipe and verdict](NIGHT_TO_DAY_UNPAIRED_CONTRASTIVE_PILOT.md).
Both exited0 at16,500; peak allocated1.187GiB each. All24 outputs opened in six
sheets at `outputs/night_to_day_experimental/20261022_unpaired_contrastive500`.
Native NCE105/358 and GAN-only0 also opened. GAN-only develops patterned
artifacts and deformation. NCE is more faithful than that control but still
grainy/tinted, retains glare/roadpools and sometimes night-like upper scenes.
Neither passes whole-image acceptance; retain parent and all experimental
weights without promotion.144tests and relevantRuff pass. No inverse process
left live; only preexistingpublicworkerpair. Production/test/lockbox unchanged.

## Soft photometric-negative grouping — 23 October 2026

Completed500updates from16k, same unpairedNCErecipe/data/seeds; changed only
night-negative weights using audited4region memberships. No pixel erasure,
externalRGBtargets or OT. See [recipe and verdict](NIGHT_TO_DAY_GROUP_CONTRASTIVE_ABLATION.md).
Process exited0 at16,500; peak allocated1.188GiB. Opened allsix/24comparisons at
`outputs/night_to_day_experimental/20261023_group_contrastive500`, plus native
grouped105/358. Same grain/tint, retainedheadlightpools/lamps, darkupperscenes
and patchysky. Reject; no visual acceptance established.147tests/relevantRuff
pass. This is evidence against this narrow adaptation, not fullN2D3.
Publicday-to-night/test/lockbox unchanged; parent/checkpoints retained.

## Author-linked CARLA TRAIN supervision audit — 23 October 2026

See [paired-source audit](NIGHT_TO_DAY_PAIRED_SOURCE_AUDIT_2026_10_23.md).
Downloaded12TRAINpairs/50,098,253imagebytes from pinnedKaggleversion3, noheldout
pixels. Frozenparent16koutputs rendered; allthree/12panels opened at
`outputs/night_to_day_experimental/20261023_CARLA_training_audit12`.
Static geometry broadly corresponds but dynamic objects differ; no unrestricted
pixel-loss approval. Parent has skyholes/bands/speckles and residualnightlamps
even on rendered inputs, so sensornoisealone is insufficient explanation.
No new training/promotion.149tests/relevantRuffpass. Public/test/lockbox unchanged.

## CARLA upper-static paired continuation — 23 October 2026

Executedmatched500updates from16k with primary111external-target supervised
recipe, plus12reviewedCARLAupper-staticROIs atcoarseauxweight10vs0control.
All12preparedsource/reference/maskpanels opened before approval. Both exited0
at16,500. See [fullrecipe/verdict](NIGHT_TO_DAY_CARLA_STATIC_ABLATION.md).
Allsix/24fixeddevelopmentcomparisons opened in
`outputs/night_to_day_experimental/20261023_CARLA12_matched500`;nativecandidate
105/358alsoopened. Samerecurring skyholes/glare/roadpools/overheaderrors remain.
TRAINregionRGBL1.51403to.10201candidatevs.51584control; notgeneralization.
ThreecandidateTRAINpredictionpanels500000/500001/500006opened: smoother sky
fill but remaininggeometry/textureerrors; do notclaimall12predictionsreviewed.
Rejectpromotion.152tests/relevantRuffpass;production/test/lockboxunchanged.

## DarkDriving metadata and private TRAIN audit — 23 October 2026

CompletedfullKagglemirrorinventory54,304files, includingseparateLLE5,906exact
TRAINpairs; detectionday/trainimagesmissingdoesnotapplytoLLE. Noheldoutpixels.
Downloadedonly12TRAINpairs/25,132,719bytesviaKaggletoE, underrestrictiveoriginal
academicterms, notmirrorApachelabel. ArtifactsoutsideGit/publicdemo.
Allfourprivateauditsheets/12pairs opened withfrozen16kpredictions. Relevantreal
day/nightlightingchanges, broadlycorrespondingstaticgeometrybutmotion/pose/
dynamicmismatches. Parentretainspools/flare/skyartifacts. No blanketpairedGT
approval or training. See [fullscope/license/findings](NIGHT_TO_DAY_DARKDRIVING_PRIVATE_AUDIT.md).
155tests/relevantRuffpass;publicweights/routing/worker/test/lockboxunchanged.
