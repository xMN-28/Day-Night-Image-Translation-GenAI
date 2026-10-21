# Compact-light supervision and native-resolution curriculum

21 October 2026. Experimental work; convincing nighttime translation is not yet
established. These experiments do not replace the showcase model.

## Why another control was needed

The external-teacher lighting pilot reproduced more bright pixels, but some became
large white/cyan smears. Simply rewarding brightness is not sufficient. A lamp must
have plausible placement, footprint, color and surrounding illumination; the same
scene must also lose daylight sky and shadows. Teacher targets are fallible synthetic
examples, not actual paired nighttime photographs.

`daynight/teacher_objectives.py` adds an optional soft bright-footprint objective.
Map RGB maximum brightness above 0.55 in [0,1] linearly into a [0,1] footprint.
Compute squared prediction/teacher footprint error, normalized separately per image
by teacher footprint squared energy, with denominator at least 1. Expanding a bright
blob increases error rather than diluting it through a predicted-area denominator.
The normal RGB/highlight losses still supply gradients for missed lights below the
footprint threshold. This is an implemented photometric control, not a new published
emitter-detection algorithm. White cars/signs can qualify as well as lamps.

## Completed 384-wide control

The control continued the lighting checkpoint at global step 20,850 for 2,000 more
updates, ending at 22,850. Unlike a fresh-optimizer warm start, `--continue-state`
retained raw generator weights, EMA, critic, both Adam states and RNG states. Only the
footprint objective was added, weight 0.3. Existing highlight weight 0.5, adversarial
weight 0.25, semantic weight 0.25, and generator learning rate 2e-5 were retained.
Batch size 4, BF16, EMA 0.99, and gradient clipping at 3 were unchanged.

Same-cache validation over 48 scenes, latest EMA versus the parent:

| Diagnostic | Parent | Compact-light control |
|---|---:|---:|
| Weighted teacher fidelity, lower better | 0.57366 | 0.56181 |
| Teacher-highlight L1, lower better | 0.46605 | 0.46058 |
| Teacher-bright-pixel recall | 0.52863 | 0.51444 |
| Teacher-bright-pixel precision | 0.47991 | 0.52823 |
| Teacher-bright-pixel IoU | 0.34215 | 0.35669 |
| Soft footprint error, lower better | 0.87225 | 0.75452 |
| Extra-bright-pixel fraction, lower better | 0.004324 | 0.003496 |

Bright-pixel metrics use RGB maximum above 0.7. They measure agreement with the
teacher, not whether a real physical light exists. All eight final four-column panels
were inspected: footprints improve modestly, but lamps remain smeared, a sun/cloud
glow persists, and the traffic scene retains daylight shadow cues. **Not accepted.**

Artifacts:

- `E:\LumiCycleCache\experiments\turbo_distilled_compact_lights\latest.pt`.
- `outputs/sky_pilot/turbo_compact_lights_validation/metrics.json`.
- `outputs/sky_pilot/turbo_compact_lights_validation/gallery.jpg`.

## Native-512 curriculum

The earlier cache resized 512-wide teacher outputs to 384 by 216. This loses some
small-light detail and introduces a train/inference scale mismatch. The next control
keeps the teacher outputs at their native 512 by 288. It does not generate new teacher
knowledge, invent additional labeled images or change the split.

`E:\LumiCycleCache\teacher_targets\turbo_road2048_native512` contains 2,048 training
targets and 48 validation targets, JPEG quality 95/no chroma subsampling. Original
source files remain referenced on D:, not copied. The teacher, source image hashes,
split identities and seeds match the previous cache. No project test images enter
this process. External-teacher pretraining overlap with custom BDD splits remains
unaudited; see `REFERENCE_TEACHER_EVALUATION.md`.

`validate_resolution_curriculum()` verifies the old manifest hash against the parent,
identical source/split/hash/seed identities, identical teacher recipes except output
size, increasing resolution no larger than the teacher's maximum edge, and unchanged
aspect ratio. An unverified dataset change cannot silently pass as state continuation.

The native-512 run starts from compact-light global step 22,850, retaining both
optimizers and all model/EMA/critic/RNG state. The one-update probe saved global step
22,851, with both optimizer counters at 4,001 and peak allocated VRAM about 3.16 GiB.
The subsequent command resumes that exact saved update, rather than repeating it.
Global steps denote cumulative generator updates across objectives, not interchangeable
CycleGAN epochs or a scalar measure of model quality.

```powershell
.\.venv\Scripts\python.exe -m daynight.train_teacher_student --parent E:\LumiCycleCache\experiments\turbo_distilled_compact_lights\latest.pt --targets E:\LumiCycleCache\teacher_targets\turbo_road2048_native512 --previous-targets E:\LumiCycleCache\teacher_targets\turbo_road2048 --output E:\LumiCycleCache\experiments\turbo_distilled_native512 --steps 2000 --batch-size 4 --lr 0.00002 --highlight-weight 0.5 --adversarial-weight 0.25 --footprint-weight 0.3 --continue-state --deterministic --max-minutes 30
```

Run the same command to resume. `--steps` is the total updates within this stage,
not the number of additional updates on every invocation. Plain-text progress is in
the run's `status.json`; fixed previews appear every 250 updates. A `STOP` file in that
run directory requests stopping after a completed update and saving `latest.pt`.
Remove that marker before resuming. Do not terminate the Python process to request
a safe checkpoint. The original Gradio default remains unchanged.

## Restart audit: a confirmed configuration bug and fix

`daynight/audit_student_resume.py` compares two uninterrupted updates against one
update, checkpoint/save, process restart and a second update. It also optionally
repeats the uninterrupted branch to detect background nondeterminism. Copies live
in a newly created audit folder; the original checkpoint is never overwritten.

Initial strict checks failed despite identical RNG states and step counts. Enabling
deterministic algorithms alone initially did not fix them. Inspection found that the
shared `seed_everything()` helper re-enabled cuDNN benchmarking **after** the caller
disabled it. Timing-based algorithm selection could therefore differ between runs.
`configure_execution()` now applies deterministic backend settings **after** seeding;
the cuBLAS workspace environment is set before CUDA use. A regression test covers
this ordering. Main app behavior and old training entry points are not changed.

With `--deterministic`, the corrected audit passed with maximum floating-state
difference **0.0**, over 298 floating tensors, including generator, EMA, critic,
both optimizer states and RNG. Both final global counters were 20,852. The repeated
uninterrupted control also matched exactly. Evidence:
`E:\LumiCycleCache\experiments\resume_audit_lights_deterministic_fixed\result.json`.
The earlier failed audits are retained and are not represented as successful tests.

```powershell
.\.venv\Scripts\python.exe -m daynight.audit_student_resume --checkpoint E:\LumiCycleCache\experiments\turbo_distilled_lights\latest.pt --targets E:\LumiCycleCache\teacher_targets\turbo_road2048 --output E:\LumiCycleCache\experiments\my_new_resume_audit --deterministic --repeat-control
```

The output directory must not already exist. An audit of a resolution-curriculum
checkpoint additionally needs its original `--previous-targets` argument.
The tested tolerance was rtol 1e-6 / atol 1e-7; it was not relaxed to conceal drift.
This verifies a short continuation on this hardware/software stack, **not** long-run
bitwise equivalence on arbitrary machines. The native run explicitly enables this
mode. Earlier nondeterministic-mode runs retain their truthful provenance.

## Completed native-resolution control

The native-512 control finished 2,000 updates at global step **24,850**. Latest EMA
SHA-256: `e17a6a5a6ba7a509f00ae746b856810268c2d72b33b75536d18e1305ab6dfee4`.
A frozen, hash-verified copy for diagnostic branches is at
`E:\LumiCycleCache\experiment_parents\native512_step_24850.pt`; this prevents a later
resume from silently changing a branch's parent artifact.

The 48-scene comparison evaluates **both** the 384-trained parent and native-trained
candidate at 512 by 288 against the same native targets. Do not directly compare these
losses with the 384-by-216 table above.

| Native-resolution diagnostic | Parent | Candidate |
|---|---:|---:|
| Weighted teacher fidelity | 0.61243 | 0.57142 |
| Teacher-highlight L1 | 0.50676 | 0.45158 |
| Teacher-bright-pixel recall | 0.48462 | 0.54039 |
| Teacher-bright-pixel precision | 0.51264 | 0.52549 |
| Teacher-bright-pixel IoU | 0.33461 | 0.36714 |
| Soft footprint error | 0.77547 | 0.77378 |
| Extra-bright-pixel fraction | 0.003566 | 0.003723 |

Peak allocated VRAM was 3.23 GiB and the loop took about 8.6 minutes. All eight
four-column validation panels and the separate eight-scene fixed-suite gallery were
visually reviewed. The local model has cleaner body colors and more local lights than
the earlier retained sky hybrid. But elongated glows remain, the sun/cloud scene still
looks wrong, traffic images retain daylight shadows, and some sign lettering/lighting
inherits teacher artifacts. Applying the existing matte/compositor fixes many sky holes
but does not establish that the generator learned correct skies. **Not promoted.**

Evidence: `outputs/sky_pilot/turbo_native512_validation/` and
`outputs/sky_pilot/turbo_native512_fixed_val512/`. The latter explicitly separates raw
generator output from learned-matte/analytic sky compositing.

## 32-training-image capacity diagnostic

The next diagnostic selects 32 evenly spaced **training** records from the same
2,048-source native cache; validation remains the separate 48 records. Selection is
recorded in `training_selection.json`. It warm-starts from the frozen step-24,850
generator, uses a fresh Adam optimizer at 1e-4, batch 4, deterministic BF16, RGB/coarse/
target-gradient fidelity plus highlight 0.5 and footprint 0.3. Source-DINO and GAN
objectives are disabled to test target-fitting capacity without competing objectives.
This is a diagnostic branch, not a replacement training run or discarded main progress.

Output: `E:\LumiCycleCache\experiments\turbo_student_overfit32_native512`.
Any gallery evaluated with `--split train --subset-count 32` is explicitly labeled
**training memorization**, never held-out performance. A convincing memorized sample
cannot satisfy the goal by itself. Compare training and validation outcomes before
deciding whether to change optimization or architecture.

### Diagnostic outcome

After 1,000 updates (diagnostic global step 25,850), all eight training panels and
all eight separate validation panels were inspected. On the 32 training scenes,
teacher fidelity fell from 0.52143 to 0.26033, highlight L1 from 0.37955 to 0.10835,
and bright-region IoU rose from 0.43013 to 0.79133. Many memorized lamps are visibly
sharper and closer to the teacher. This establishes useful fitting capacity; it does
not establish perfect reconstruction or generalization.

On the separate 48 validation scenes, fidelity worsened from 0.57142 to 0.68650,
highlight L1 from 0.45158 to 0.47310, and bright-region IoU fell from 0.36714 to
0.31276. Some sky glows and large light blobs worsen. **The diagnostic is rejected
as a deployable upgrade.** It does not justify completing the goal using memorized
training examples, nor does it prove the current architecture has no limitations.

Evidence is in `outputs/sky_pilot/turbo_overfit32_training_diagnostic/` (TRAIN labels)
and `outputs/sky_pilot/turbo_overfit32_validation/` (VAL labels).

## Full-data objective control now running

The next experiment changes only the size of the training selection relative to
the capacity diagnostic: use all 2,048 training records, not 32. Both start from the
same immutable step-24,850 parent, with fresh Adam at 1e-4, batch 4, deterministic
BF16, the same target fidelity/highlight/footprint losses, no source-DINO loss and
no adversarial critic. **It does not warm-start from the memorized diagnostic.**
This is a new objective-control branch with intentional optimizer reset, not an
exact optimizer continuation of the older GAN/DINO objective. Generator weights
and the cumulative parent step remain retained; subsequent resumes within this
branch restore its full optimizer/EMA/RNG state.

Location: `E:\LumiCycleCache\experiments\turbo_student_full_supervised_native512`.
Planned cap: 4,000 updates (main-line global step 28,850), max 30 minutes per invocation.
Validation is every 250 updates and is persisted to `validation.jsonl` and
`last_validation.json`; milestone checkpoints are retained every 1,000 updates.
Source-code digests are captured once at process start and stored in checkpoints.

```powershell
.\.venv\Scripts\python.exe -m daynight.train_teacher_student --parent E:\LumiCycleCache\experiment_parents\native512_step_24850.pt --targets E:\LumiCycleCache\teacher_targets\turbo_road2048_native512 --output E:\LumiCycleCache\experiments\turbo_student_full_supervised_native512 --steps 4000 --batch-size 4 --lr 0.0001 --highlight-weight 0.5 --footprint-weight 0.3 --semantic-weight 0 --deterministic --keep-every 1000 --max-minutes 30
```

No success is claimed before inspecting held-out images. If this control also
produces averaged light blobs, reconsider data/conditional uncertainty and semantic
capacity rather than claiming that an extra 1,000 steps or another loss guarantees
a fix. The benchmark teacher remains external and all original project test images
remain excluded from tuning.

Mid-run milestone results and the completed teacher-sampling hypothesis check are
recorded in [TEACHER_VARIATION_AUDIT.md](TEACHER_VARIATION_AUDIT.md). This includes
negative evidence against stochastic teacher lamps as the main cause of the blobs.

The completed 4k control and the next matched sky-supervision continuation are
documented in [SKY_TARGET_SUPERVISION_ABLATION.md](SKY_TARGET_SUPERVISION_ABLATION.md).
