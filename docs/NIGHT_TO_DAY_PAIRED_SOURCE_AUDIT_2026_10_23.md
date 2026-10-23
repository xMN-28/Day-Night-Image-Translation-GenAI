# Paired supervision search and CARLA audit — 23 October 2026

## Decision summary

The current inverse remains visually unacceptable. The last group-contrastive
pilot was completed and rejected, so this pass investigated supervision rather
than another weighting change. Kaggle-only download preference was respected.
No new model was trained, promoted or substituted for the production model.

Found an author-linked CARLA pair collection; downloaded only12TRAINpairs,
50,098,253 image bytes, to E:. Opened all three comparison sheets. These are
**simulator renders**, not real-night photographic ground truth. The pair
geometry is useful for static regions, but moving objects differ in some pairs.
Do not approve full-image pixel loss on filename correspondence alone.

## Sources inspected

### Author-linked CARLA paired images

- [Kaggle dataset](https://www.kaggle.com/datasets/okarif/carla-parired-images),
  pinned version3, creator Omar ARIF, Kaggle image license CC BY-SA3.0.
- Its description links [the creators' repository](https://github.com/omar-arif/day-night-image-conversion-GAN),
  Hamza Benmendil and Omar Khalifa Arif. Their setup records a driving trajectory
  and replays it under two lighting conditions. The source-code repository's
  Apache license is separate from the Kaggle image license.
- Metadata inventory:2,400files,1,000exact day_train/night_train filename pairs.
  Other split filenames were inventoried as metadata only, never downloaded or
  opened. No claim of clean route-independent author splits is made.
- Sample:12pairs across all10available training recording prefixes, sampled
  without quality-based cherry-picking. With12samples, the current round-robin
  selection picks first frames in10sequences and last frames in the first2.
  This is a coverage audit, not a uniformly distributed temporal sample or a
  ready-to-train representative subset.
- The creators explicitly discuss nondeterminism between simulator replays.
  Their same-light replay analysis is not evidence of exact pixel registration
  for every day/night pair in our cache.

### DarkDriving Kaggle mirror: do not assume Apache rights

Public Kaggle metadata for `benxelua/darkdriving` reports version2, about30.04GB,
empty description and Apache2.0. The first file-list page contains paths under
`darkdriving_det/Det_data/day/images/val/`. We did **not** inventory the entire
mirror and cannot claim its restoration-pair coverage, pairing quality or
author authorization. No image bytes were downloaded.

The [authors' dataset license](https://github.com/DriveMindLab/DarkDriving-ICRA-2026/blob/main/LICENSE)
instead permits academic/noncommercial use with attribution and restricts
redistribution, including derivatives. A mirror label does not resolve that
conflict. Do not treat the mirror as unrestricted Apache-licensed data or put
images/outputs into public demo artifacts. The
[canonical data page](https://github.com/DriveMindLab/DarkDriving-ICRA-2026)
links author-hosted downloads; those still require resolving the user's
Kaggle-only preference. Download size and pairing quality remain unverified.

### ACDC mirror lacks required references

`khalilusmanuk/acdc-clean-dataset-for-training` metadata lists about7.49GB and
MIT. Complete4,018file-name inventory:4,006adverse-condition images plus12
detection annotation JSONs. Night entries:400train,106val,500test. No normal/day
reference-image files appear in that complete inventory. No pixels downloaded,
including no test/val pixels. Its description mentioning correspondences is
not sufficient evidence that those counterparts are actually supplied.

This does not show that canonical ACDC lacks references; only that this mirror
cannot supply the desired paired relighting supervision as inventoried.

### Other search results not silently adopted

No Alderley result appeared in the queried Kaggle dataset-search endpoint;
that is not proof no mirror exists. `kll0224/day2night` has unknown license and
an inspected listing beginning with BDD-like test filenames; it was not treated
as paired ground truth or downloaded. LOL/SICE/RAW low-light collections are
not automatically true day/night relighting datasets. Previously audited
LoLI-Street has synthetic darkened training inputs and did not fix real-night
validation in our continuation experiment.

## Executed CARLA frozen-parent audit

```powershell
$env:KAGGLEHUB_CACHE='E:/LumiCycleCache/kagglehub'
.venv/Scripts/python.exe -m daynight.audit_carla_pairs --cache E:/LumiCycleCache/teacher_targets/CARLA_audit12_20261023 --output outputs/night_to_day_experimental/20261023_CARLA_training_audit12 --parent E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt
```

Actual exit0. Complete metadata and selected-file sizes are stored in the cache
index. Audit records contain each downloaded image SHA256 and native dimensions.
The script requires exact train folder/name matches, pins the dataset version,
caps the audit at24pairs/300MiB, refuses an existing output directory and never
executes downloaded repository code. Kagglehub retrieves the selected files
only; the4.38GBdataset was not downloaded in full.

Frozen16kEMA generator, unchanged weights; BF16inference;512x288audit display.
All panels are labeled TRAIN diagnostic, not validation performance. No result
is a claimed generalization score.

### Direct visual observations (all12panels opened)

- Recording00first/last: static buildings align broadly; inverse produces large
  unsupported black sky holes, blurry boundaries and residual night lamps.
- Recording01first/last: clean rendered daytime sky versus inverse black bands,
  blue patches and speckled/flare-like texture. Road/structure contrast is wrong.
- Recording02/03: wide sky/landscape targets versus inverse horizontal dark
  blotches and invented/smeared detail. Severe failures despite simulator input.
- Recording04: reference foliage/road is coherent; inverse leaves a near-black
  sky wedge and strongly overilluminates branches.
- Recording05/07/08: buildings/road mostly line up between rendered conditions,
  but inverse changes detail and retains dark sky islands/light glow.
- Recording06: a clear car in the daytime reference has no clearly observable
  corresponding car in the night panel. Do not teach the model to invent it
  using full-image L1. Lamp/bridge/building geometry can still be reviewed as
  static correspondence separately.
- Recording09: a daytime vehicle and foliage differ in visibility/appearance;
  global alignment cannot establish dynamic-object pixel validity.

Inference: our recurring sky blotches also occur on rendered images, so sensor
noise alone is not a sufficient explanation. This does not identify a single
faulty network layer or establish real-photo transfer from CARLA.

## Next usable experiment, not yet executed

Audit a broader temporally spaced TRAIN-only CARLA subset and static-region
correspondence before training. Preserve author-held-out images unopened. Use
recording/route grouping for any local split, not random adjacent frames. Do not
use dynamic objects as exact paired targets. A warm-started static-pair
continuation could test whether clean supervised sky/lamp transformations can
be learned, with the same fixed24REAL-night development images as the transfer
gate. Simulator fit is not the requested end state and cannot justify promotion.

All existing weights remain intact. Public day-to-night, test set and reserved
lockbox are unchanged.149tests passed; relevant Ruff passed. Two new tests cover
TRAIN-only exact-name pairing/path rejection and deterministic sequence sampling.
No inverse training process was started during this audit.
