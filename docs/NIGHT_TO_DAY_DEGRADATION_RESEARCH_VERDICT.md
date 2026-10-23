# Night-to-day research verdict — 22 October 2026

## Bottom line

The experimental inverse still fails whole-image acceptance. Cleaner texture alone is not convincing daylight. Do not promote the latest auxiliary-data or context-adapter checkpoint. Preserve the trained 16k parent and the working day-to-night production pipeline.

This is a feed-forward ResNet/CBAM generator, not a diffusion model. There is no iterative denoising schedule. Grain can nevertheless be amplified by a feed-forward translator. Our controlled perturbation measurements establish sensitivity, not the identity of a faulty layer.

## What the experiments actually support

- Full precision did not remove the visible failure patterns. Unchanged-input repeated inference was deterministic in the diagnostic.
- Noise-consistency continuation lowered measured perturbation amplification from about 4.49 to 2.58 relative to its matched control, but the fixed 24-image review still failed sky, glare and road-lighting acceptance. That is improved robustness, not a successful translator.
- LoLI-Street auxiliary training improved synthetic-low/real-day training reconstruction. Its low-light training inputs are synthetic; this did not establish generalization to real BDD night scenes.
- Adding global scene-context attention altered predictions and slightly improved external-target imitation, but did not fix the recurring whole-image defects. Capacity alone was insufficient in this ablation.
- Existing external RGB targets contain retained glare and implausible skies. Strong pixel imitation can reward those defects. This is a plausible limitation, not a proven explanation for every artifact.

See the noise, restoration-data and scene-context records linked from `NIGHT_TO_DAY_VISUAL_REVIEW.md` for checkpoint paths and review scopes. No failed experiment is relabeled a quality-approved release.

## Methods worth separating

1. **Degradation-aware translation:** [N2D3](https://arxiv.org/html/2411.14504v1) distinguishes illumination degradation regions and uses region-aware contrastive learning. Its training principle is more directly relevant to night-to-day domain translation than generic exposure enhancement. It does not imply our hand-implemented maps are accurate labels.
2. **Illumination/restoration separation:** [Retinexformer official implementation](https://github.com/caiyuanhao1998/Retinexformer) is relevant to handling nonuniform illumination and corrupted detail. Low-light enhancement is not automatically daytime translation: headlights, night skies and colored light can survive a successful enhancement.
3. **Intensity/color decoupling:** [HVI/CIDNet official implementation](https://github.com/Fediory/HVI-CIDNet) motivates handling darkness and color instability separately. Changing color space alone cannot recover clipped objects or determine their true daytime appearance.
4. **Frequency separation:** N2D-LPNet is discussed in the research-reset document. It motivates separating scene-scale appearance from detail. Our existing pooled pixel losses are not an implementation of that architecture.

These are focused primary-source investigations, not a claimed review of hundreds of repositories. External systems are references, not silently substituted project outputs.

## Executed N2D3-inspired map audit

Reproducible command:

```powershell
.venv/Scripts/python.exe -m daynight.audit_inverse_degradations --split val --output outputs/night_to_day_experimental/20261022_photometric_regions_val24
.venv/Scripts/python.exe -m daynight.audit_inverse_degradations --split train --count 12 --output outputs/night_to_day_experimental/20261022_photometric_regions_train12
```

All six validation sheets (24 sources) and all three training sheets (12 accepted sources) were visually opened. No test or reserved lockbox images were used. No generator training was performed by this audit.

Implementation: deterministic three-means clustering of maximum RGB; Gaussian-color derivative invariants from the paper; standardized invariant strength splits part of the middle-brightness region into an effect membership. Approximate sRGB, a 0.02 denominator floor, finite 7x7 derivative support, and clamping the effect strength to [0,1] are explicit implementation choices. This is **not exact N2D3 reproduction**; optimal-transport reweighting and degradation-aware contrastive training are not implemented here.

Mean memberships, not classification accuracies:

| Audited inputs | Dark | Middle/well-lit | Highlight | Effect |
|---|---:|---:|---:|---:|
| Fixed development 24 | 71.59% | 23.53% | 3.85% | 1.03% |
| Training 12 | 75.27% | 21.07% | 2.48% | 1.18% |

No image had fewer than three occupied brightness clusters. This says nothing about semantic correctness.

### Visual verdict

The brightness grouping separates many lamp cores from dark surroundings, but also groups white cars, signs and road markings as highlights. The invariant responds strongly to object/material boundaries and sometimes block-like compression structure in darkness. The effect membership is often sparse along edges rather than covering broad glare or headlight pools. Examples 105 and 12 retain the exact ambiguity that matters for sky and branch handling; 214 does not identify the whole overhead structure as one object.

**Reject these maps as hard erasure, sky, inpainting or pixel-confidence masks.** That would repeat the earlier unreliable brightness-mask experiment. Photometric groups may still be investigated as soft contrastive sampling groups; they are not semantic truth. The audit does not invalidate the published method, whose complete training mechanism differs from this stabilized map implementation.

## Next bounded experiment — proposed, not yet trained

Change the objective before expanding the generator again:

1. Warm-start the retained 16k generator; retain every parent weight. Train-only projection heads and a day-domain discriminator may start fresh and must be documented as such.
2. Establish an unpaired adversarial + ordinary PatchNCE continuation using only training-domain images, without external RGB teacher imitation. This differs from our earlier GAN trials that retained strong teacher RGB anchors.
3. Only then compare against a matched photometric-group contrastive variant. Keep generator, data order, update budget and losses otherwise fixed. Describe it as an adaptation unless the complete published mechanism is reproduced.
4. Probe memory and finite gradients first. Run a short bounded pilot, render every fixed development case, and inspect native crops. Reject if cleaner texture comes with moved objects, lost cables, invented sky holes, retained light pools or merely brighter night scenes.
5. Preserve the existing whole-image acceptance gate and untouched lockbox. Do not call lower teacher error, training reconstruction or a paper's benchmark score proof of our output quality.

Better aligned real-night/day supervision remains a separate promising route. DarkDriving's author-hosted downloads require resolving the earlier Kaggle-only source preference and its license before proceeding. No such download is implied by this report.

## Verification and scope

140 tests passed after adding five photometric-map tests; relevant Ruff checks passed. Those checks validate code behavior, not image realism. Production day-to-night weights, sky repair, public model choices and live worker remain unchanged. No new inverse checkpoint was promoted or deployed in this map audit.
