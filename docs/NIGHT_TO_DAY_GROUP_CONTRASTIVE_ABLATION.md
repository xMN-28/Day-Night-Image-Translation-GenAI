# Photometric-group contrastive ablation

23 October 2026. Private experimental inverse; no production changes.

## Hypothesis and limits

The ordinary contrastive pilot preserved unwanted night illumination as well as
objects. [N2D3](https://arxiv.org/html/2411.14504v1) motivates distinguishing
degradation types when constructing contrastive comparisons. Our earlier map
audit found that these photometric estimates are not reliable semantic labels.
Therefore this experiment only uses them to weight **negative feature pairs**;
it never erases image pixels or treats them as sky/object ground truth.

This is a deliberately limited soft-group adaptation, NOT a complete N2D3
reproduction. In particular, it does not implement the paper's optimal-transport
weighting, strict cross-region zero weights, or region-stratified sampling.
It cannot establish whether the complete published method would succeed here.

## Single changed variable

Use the existing 16k EMA parent, same generator, projection heads, discriminator,
5,000+5,000 BDD training manifests, augmentation seeds, 500 updates, learning
rates and other settings as the ordinary NCE v3 run. That retained run is the
control, not the failed GAN-only model. No context adapter or external RGB
targets are added. Historical external-target training of the parent is still
credited; this is not a from-scratch or external-free provenance claim.

For each source night crop:

1. Estimate four nonnegative memberships (dark, middle-brightness, highlight,
   effects) with the already audited photometric code, in FP32 outside autocast.
2. Area-resize memberships to each selected feature resolution and sample them
   at exactly the same spatial indices as the source/translated features.
3. For each negative pair i,j multiply its exponential similarity contribution
   by `0.02 + 0.98 * dot(m_i, m_j)`. The diagonal positive weight stays exactly 1.
4. Keep the day identity contrastive term uniform. No anchor reweighting or
   direct output/pixel-confidence masking is used.

Memberships are detached; they are computed from the input only. A 0.02 floor
retains some cross-group negatives instead of allowing an isolated group to
make the loss trivially zero. This floor is our explicit experiment choice,
not a parameter claimed to come from the paper. Areas mixing groups at coarse
feature scales produce soft weights rather than hard assignments.

Tests verify exact ordinary-loss AND gradient equivalence when all sampled
patches share one one-hot group; reduced cross-group negative contributions
without changing positives; finite query gradients; no gradients into keys or
memberships; and rejection of invalid membership partitions.

## Reproduction

```powershell
.venv/Scripts/python.exe -m daynight.train_inverse_contrastive --parent E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt --output E:/LumiCycleCache/experiments/night2day_unpaired_groupNCE_from16000_v1 --steps 500 --nce-weight 1 --photometric-groups
```

The recipe records both trainer and map-code hashes. Earlier v3 control recipe
retains its own original code hash; the default path remains ordinary NCE.
Map computation does not consume the image/augmentation or patch RNG streams.
Tests cover default loss equivalence, but bitwise equality of complete rerun
trajectories is not claimed across different GPU execution timing.

Checkpoint output continues global numbering from16k. All other scope,
checkpoint contents and bounded/no-resume limitations are documented in
`NIGHT_TO_DAY_UNPAIRED_CONTRASTIVE_PILOT.md`. Test/lockbox remain untouched.

## Executed outcome: reject

The process exited0 after500updates, reaching global16,500. Logged loop time
79.22seconds; peak allocated1.188GiB (not total device memory). All gradients
and losses passed finite checks. Latest and numbered250/500checkpoints remain
in the experiment directory. This is a completed pilot, not a running job.

Rendered `outputs/night_to_day_experimental/20261023_group_contrastive500`.
All six sheets were opened: every fixed24development source, external reference
(display only), parent16k, uniformNCEcontrol and groupedNCEcandidate. Additional
native512px grouped outputs105and358 were opened. No test/lockbox use.

The grouped candidate remains very similar to the failed uniform control:

- 105: predominantly night-like dark upper region, saturated lamps and noisy
  texture; no coherent daylight scene.
- 358/86/78: obvious headlight-shaped road pools survive, with colored speckling
  in sky/foliage/road. Native358confirms noise is not a contact-sheet artifact.
- 12: patchy artificial sky boundary and persistent night-lit tree.
- 142/214: overhead structures remain confused with sky/lighting.
- 8: floating dark patch near the lamp remains.
- 283: dark upper scene over brightened road remains.
- 0/355/441: strong lamp glare survives rather than becoming plausible daylight.

Whole-image acceptance is contradicted by recurring severe defects. No formal
blind-rating percentage or full per-case pass tally is claimed. No promotion.

External-targetRGBL1: parent.07505, uniform.12130, grouped.12527 (0–1).
These are imitation statistics, not realism scores, and external references
were not targets for these pilots. Lower final grouped trainingNCE(.19735)
is not comparable evidence of better quality because its negative weights
change the objective itself.

## What this rules out / does not rule out

This soft negative-grouping change alone is insufficient at the tested budget.
It does not prove that fullN2D3or all physical priors fail. Every same-position
positive still encourages source/translation feature agreement, including in
unreliable regions. Moving only negative weights does not explicitly teach
headlight removal or robust recovery from noise. That remaining limitation
matches the visual failure and should guide the next experiment rather than
blindly extending this recipe.

The retained16kparent remains the conservative starting point, not an accepted
night-to-day release. A next experiment needs reliable restoration/relighting
supervision or a separately validated illumination-invariant representation;
do not simply stack another unvalidated mask onto this candidate.

147tests passed after adding the three grouped-loss tests; relevantRuff passed.
Production models/routing/sky repair and worker were not modified. All failed
experimental checkpoints are preserved for comparison, not presented as the
project's best output.
