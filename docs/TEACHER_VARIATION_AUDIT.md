# Teacher-variation audit and full-data control checkpoints

21 October 2026. The image-quality goal remains unachieved.

## Hypothesis tested: are stochastic targets causing blurred lamps?

The official CycleGAN-Turbo encoder samples its VAE posterior. A deterministic local
student might struggle if identical source images receive substantially different
lamp placements. Rather than assume this explains the remaining blobs, we measured it.

`daynight/audit_teacher_uncertainty.py` loads the pinned, cached official model in
the isolated E-drive environment. It uses eight predetermined validation sources,
four consecutive seeds per source, and two additional latent-posterior-mode forwards
per source: 48 forwards total. Raw finiteness is checked before image conversion.
The posterior-mode version is an explicitly labeled in-process **ablation**, not
the official default model. Pinned source files, teacher caches, local generator
weights and demo defaults are unchanged.

| Observed diagnostic | Value |
|---|---:|
| Mean pairwise seed RGB L1, [0,1] units | 0.00013942 |
| Mean pairwise bright-mask IoU, RGB maximum >0.7 | 0.99698 |
| Mean per-pixel RGB standard deviation | 0.00012398 |
| Worst per-pixel standard deviation across all eight inputs | 0.017795 |
| Posterior-mode repeat maximum difference, both seeds, all eight inputs | 0.0 |
| Mean mode-to-first-sampled-output RGB L1 | 0.00014179 |

On the **same eight sources**, the 1,000-update full-data student's RGB L1 to the
teacher is 0.0244074 in [0,1] units, about 175 times the measured mean seed variation.
This is an empirical scale comparison, not a formal irreducible-error bound.
The seed columns in the sun/cloud scene were visually inspected and look nearly
identical. The large student light smears are not explained by substantial random
lamp relocation in these checked targets. **Reject this as the main explanation on
this suite.** There is no justification here for regenerating thousands of targets
with posterior-mode encoding or claiming that such a change fixes the student.

Evidence: `outputs/turbo_teacher_uncertainty/metrics.json`, its per-image panels,
and `outputs/sky_pilot/turbo_full_supervised_1000_validation/metrics.json`.
Only validation images were used. This does not prove identical behavior for every
possible input or remove the external-pretraining-overlap caveat.

```powershell
E:\LumiCycleCache\turbo-env\Scripts\python.exe -m daynight.audit_teacher_uncertainty
```

## Full-data control at 1,000 and 2,000 updates

The ongoing full-data control is described in
[COMPACT_LIGHTS_AND_NATIVE_RESOLUTION.md](COMPACT_LIGHTS_AND_NATIVE_RESOLUTION.md).
The following measurements use immutable milestone checkpoints, not a `latest.pt`
being overwritten while an evaluation is in progress. All 48 validation scenes use
the same native-resolution teacher cache and preprocessing.

| Diagnostic | Frozen parent | Control 1k | Control 2k |
|---|---:|---:|---:|
| Weighted teacher fidelity | 0.57142 | 0.57639 | 0.55330 |
| Teacher-highlight L1 | 0.45158 | 0.43615 | 0.42408 |
| Teacher-bright-pixel recall | 0.54039 | 0.49132 | 0.50524 |
| Teacher-bright-pixel precision | 0.52549 | 0.56887 | 0.56888 |
| Teacher-bright-pixel IoU | 0.36714 | 0.36148 | 0.36995 |
| Soft light-footprint error | 0.77378 | 0.60240 | 0.58940 |
| Extra-bright-pixel fraction | 0.003723 | 0.002666 | 0.002740 |

The 2k four-column gallery was visually inspected across all eight fixed panels.
There is modest progress in excess-bright-area and target fidelity, but the
sun/cloud scene still has an elongated white glow and some streets retain daytime
shadow cues. **Not accepted and not promoted.** A small IoU increase is not proof of
convincing nighttime synthesis. Training remains capped at 4,000 updates for this
control; review later milestones before changing architecture or objectives again.

Evidence: `outputs/sky_pilot/turbo_full_supervised_1000_validation/` and
`outputs/sky_pilot/turbo_full_supervised_2000_validation/`.

## Checking the previously suggested VGG16 repository

The inspected [solesensei/day2night trainer](https://github.com/solesensei/day2night/blob/abb5e670ddc03ef7bd35ae7395a6e1b3aaf60c5d/day2night/UNIT/trainer.py)
loads VGG as a frozen training perceptual network; it is not itself the whole image
generator. Its UNIT `compute_vgg_loss` compares instance-normalized features. The
[generator definitions](https://github.com/solesensei/day2night/blob/abb5e670ddc03ef7bd35ae7395a6e1b3aaf60c5d/day2night/UNIT/networks.py)
still use InstanceNorm in the inspected `VAEGen` encoder/decoder. Thus the screenshot
alone does not establish that removing all normalization from our generator reproduces
that experiment. Its exact historical figure configuration has not been reconstructed.

Reference commit: `abb5e670ddc03ef7bd35ae7395a6e1b3aaf60c5d`. The inspected files carry
CC BY-NC-SA 4.0 notices. No repository code was copied into this project. This is a
focused primary-code review, not a claim to have screened hundreds of repositories.
A target-feature perceptual-loss control remains a possible next experiment if this
full-data control plateaus; no benefit is claimed before implementing and testing it.
