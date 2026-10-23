# CARLA static-region paired continuation — 23 October 2026

## Result first

Paired simulator supervision improved training-region reconstruction markedly,
but the fixed24REAL-night development images still fail whole-image daylight
acceptance. No model promoted. Keep the conservative16kparent and all ablation
checkpoints. The project goal is real-photo transfer, not a small simulator fit.

## Data preparation and visual approval

Source: the12TRAINpairs already downloaded from author-linked
`okarif/carla-parired-images`, version3, CC BY-SA3.0. See the paired-source audit
for authors, licensing, provenance and why dynamic objects are not pixel GT.
No further downloads, external generator or test/lockbox pixels were used.

Prepared512x288source/reference images with a fixed binary supervision rectangle:
all columns, rows0through175inclusive. The lower112rows are excluded. These
coordinates were reviewed against every prepared pair in three source/reference/
mask sheets BEFORE approval. Sky, lamps, buildings and upper foliage are the
intended static content. The mask is not a semantic segmentation or a claim of
perfect physical/pixel correspondence; replay nondeterminism remains possible.
The12pairs are not a new representative dataset or generalization benchmark.

Cache: `E:/LumiCycleCache/teacher_targets/night2day_CARLA_upper12_v1/regions.json`.
Approved-manifest SHA256:
`436d957e8517de77bcdfb821d20144cb6ee4b3334b0918b5abe5dd9728c095ea`.
Source/target/mask hashes are checked by preparation, approval and training.

The selected boundary is divisible by both4and8. A gradient test verifies that
the pooled auxiliary loss has exactly zero image-output gradient in excluded
rows. This does NOT guarantee the generator will leave those rows unchanged:
shared weights and receptive fields couple its predictions across the image.

## Matched experiment

Same existing16kEMA G_night_day parent, architecture unchanged. Both runs:

- Primary111audited BDD training sources and external CycleGAN-Turbo RGB targets,
  same as the earlier supervised pilot. External-target provenance is retained,
  not hidden by adding simulator data.
- Primary loss:10RGBL1 +2mean(pool4,pool16) targetL1 +mean spatial-gradientL1.
- Each step also samples one CARLA pair using independentPythonRNG2032. Auxiliary
  loss is mean of pool4/pool8 masked L1, normalized by pooled mask mass and RGB
  channel count. Horizontal flips are shared within each pair and its mask.
- Candidate auxiliary weight10; control weight0. Both compute the auxiliary path.
  The primary batches/augmentations and auxiliary sampling have the same seeds.
- Adam1e-4, betas(.5,.999), primarybatch2+auxiliarybatch1, BF16, clip3, EMA.99,
  500updates, no GAN/context/noise-consistency terms, no weight reinitialization
  in the generator. Fresh optimizers because this is a changed-objective pilot.

The legacy recipe field `real_region_loss` contains a generic RGB description;
`real_region_loss_mode: coarse`, the command and implementation are authoritative
for the actual pooled loss. Likewise `REAL_STATIC_TRAINING_ONLY` is a legacy
folder name, not an assertion that CARLA renders are real photographs. Each panel
is explicitly labeled SIMULATOR.

Experiments:

- `E:/LumiCycleCache/experiments/night2day_CARLA12_control_from16000_v1`
- `E:/LumiCycleCache/experiments/night2day_CARLA12_aux10_from16000_v1`

Both actual processes exited0 at500adaptation updates/global16,500. Latest and
numbered250/500checkpoints are retained with optimizer/EMA/RNG state. Logged
loop times93.53/93.61seconds, overlapping GPU execution; not isolated benchmarks.
Peak allocated memory1.736GiBperprocess, not total GPU usage.

## Reproduction

```powershell
.venv/Scripts/python.exe -m daynight.prepare_inverse_carla_regions --output E:/LumiCycleCache/teacher_targets/night2day_CARLA_upper12_v1
# Inspect all three TRAIN_regions sheets before this separate approval:
.venv/Scripts/python.exe -m daynight.prepare_inverse_carla_regions --output E:/LumiCycleCache/teacher_targets/night2day_CARLA_upper12_v1 --approve-reviewed
```

Existing caches cannot be overwritten. For either run use:

```powershell
.venv/Scripts/python.exe -m daynight.train_night_to_day_pilot --targets E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2 --selection E:/LumiCycleCache/teacher_targets/night2day_pilot128_v2/visual_selection_v1.json --parent E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt --output <NEW_EXPERIMENT_DIRECTORY> --steps 500 --real-regions E:/LumiCycleCache/teacher_targets/night2day_CARLA_upper12_v1/regions.json --real-region-weight 10 --real-region-loss coarse --max-minutes 10
```

Change only weight10to0for the control. Do not overwrite retained experiments.

## Measured reconstruction, not acceptance

Mean TRAIN static RGBL1 across all12regions, on model-normalized[-1,1]tensors:

| Updates | Control | Auxiliary10 |
|---:|---:|---:|
| 0 | .51403 | .51403 |
| 250 | .50787 | .13490 |
| 500 | .51584 | .10201 |

This reports RGB reconstruction for comparability, although the auxiliary
training objective is pooled/coarse. It is training fit, not held-out quality.
Uniform development external-target fidelity at500:1.81377control vs1.81927aux.
That statistic likewise measures imitation, not realism.

## Direct image review

Opened allsix matched sheets at
`outputs/night_to_day_experimental/20261023_CARLA12_matched500`, covering all24
fixed development sources. Columns: night source, clearly external reference,
parent16k, matchedcontrol500, auxiliary10. Additionally opened native512px
candidate105and358. No formal blind rating or new numeric pass count is claimed.

Recurring real-photo defects remain:105's oversized unsupported sky opening;
12's patchy sky/tree boundary;142/214overhead structure confusion;8floating
dark patch;358/86/78headlight-shaped road illumination;202bright road patch;
283night-like upper scene;0/355/441remaining giant artificial lamps/glare.
The candidate is visually close to its control rather than a convincing fix.

Opened three candidate TRAIN prediction panels at500:500000(recording00first),
500001(recording01first),500006(recording06first). Large black sky islands are
replaced by smoother daytime-like fill in these training examples, but blur,
incorrect geometry/detail and residual lamp texture remain. For example the
recording01building is poorly reconstructed despite a smoother sky. Do not claim
all12TRAINpredictions were visually inspected; their aggregate loss was measured.

## Interpretation and next constraint

The network can respond to clean paired lighting supervision; it is not simply
incapable of altering sky pixels. However,12simulator pairs with coarse static
supervision do not resolve real-camera relighting. Plausible contributors are
small data coverage, simulator/photograph domain gap and conflict with retained
primary teacher targets; this experiment does not isolate those contributors.
More iterations or stronger weights are not automatically a solution.

Before a larger simulator continuation, audit broader temporally spaced TRAIN
coverage and static correspondence; a better capacity fit alone cannot justify
promotion. Obtaining genuinely aligned real-night/day coverage remains a useful
alternative, subject to source/license constraints. Keep the original24image
gate and reserved lockbox rather than swapping to easier simulator evaluations.

152tests/relevantRuffpassed, including exact mask bounds, rejecting unreviewed
sizes and excluded-road gradient behavior. Public day-to-night generator/matte,
model selection/routing/remote worker and test/lockbox are unchanged. No inverse
training process remains live after these bounded runs.
