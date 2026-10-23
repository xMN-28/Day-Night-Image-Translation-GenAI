# DarkDriving mirror: complete inventory and private TRAIN audit

23 October 2026. No new training or production change in this pass.

## Why revisit this source

The CARLA static-region trial learned simulator skies but failed real BDD
transfer. This pass checked whether the previously located Kaggle DarkDriving
mirror actually supplies relevant real paired supervision. This changes the
evidence from a first-page listing to a complete inventory and12visually
inspected TRAINpairs. It does not establish a successful night-to-day model.

## Provenance, license and publication boundary

- Mirror: `benxelua/darkdriving`, pinned version2. Kaggle metadata says Apache2.0
  and supplies no description establishing uploader authorization.
- Canonical [DarkDriving authors' page](https://github.com/DriveMindLab/DarkDriving-ICRA-2026)
  reports5,906training and3,632testpairs. Their
  [LLE runner instructions](https://github.com/DriveMindLab/DarkDriving-ICRA-2026/blob/main/LLE/README.md)
  describe train/day, train/night, test/day and test/night folders. The official
  loader checks sorted basename agreement, but only warns on mismatch; our
  candidate pairing requires exact names rather than silently pairing by order.
- Apply the restrictive [original academic license](https://github.com/DriveMindLab/DarkDriving-ICRA-2026/blob/main/LICENSE),
  not the mirror's permissive label: academic/noncommercial use, attribution,
  no dataset/derivative redistribution. No assertion is made that the authors
  authorized this mirror or that its bytes match an official archive.
- The small audit is local academic investigation only. Downloaded images,
  predictions and contact sheets stay on E: outside the Git workspace and out
  of public demos. Do not publish data, derived images or trained weights from
  this source without resolving applicable rights. This is not legal clearance
  for subsequent publication or a claim of user acceptance of new public-use
  terms. No author-hosted download or external message was performed.

## Complete metadata evidence

Inventory file:
`E:/LumiCycleCache/teacher_targets/DarkDriving_mirror_inventory_20261023.json`.

-54,304listedfiles;32,072image entries across multiple overlapping collections.
-30,039,552,199listedbytes. This is listing size, not a full dataset download.
-The detection collection lacks day/images/train despite having3,339day/train
  label files. That deficiency does **not** describe the separate LLEcollection.
-The restoration LLEtrain/day and train/night each contain5,906jpgfiles;
  all5,906have exact corresponding names. LLEtest/day and test/night each
  contain3,632image entries. Counts match the canonical page, but matching
  counts/names are not proof of provenance, image alignment or license authority.
-No identical day-image filenames between restoration train and test.22of the
  23training filename prefixes also occur in test; the singletonin_5prefix is
  train-only. These are filename-derived prefixes, not verified route labels.
  Consequently do not infer route independence from the split names.
-Only filenames/byte sizes were inventoried for author-held-out data. No
  held-out pixels or performance metrics were used. BDDtest and its reserved
  12source lockbox remain untouched.

Kaggle caps pages at200entries despite requesting1,000. The first inventory
attempt stopped on a transient404. A durable page checkpoint and retry handling
were added; the next attempt also stopped once, then resumed from its saved
page160instead of discarding/restarting it. Final inventory reached272pages
and actual exit0. No404or observation timeout was misrepresented as end-of-list.
Complete and partial inventory flags are explicit; incomplete listings are
rejected by the audit selector.

## Executed private image audit

```powershell
.venv/Scripts/python.exe -m daynight.index_darkdriving_mirror --output E:/LumiCycleCache/teacher_targets/DarkDriving_mirror_inventory_20261023.json
$env:KAGGLEHUB_CACHE='E:/LumiCycleCache/kagglehub'
.venv/Scripts/python.exe -m daynight.audit_darkdriving_pairs --index E:/LumiCycleCache/teacher_targets/DarkDriving_mirror_inventory_20261023.json --output E:/LumiCycleCache/experiments/DarkDriving_private_audit12_20261023 --parent E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt
```

Use new destinations for a fresh rerun. Completed both commands with exit0.
Only12evenly spaced sorted TRAINpair names were selected; not a balanced sample
of all23prefixes or a curated easy-image suite. Selected images total25,132,719
bytes, not30GB. Individual-file downloads only, allviaKaggle. Cached originals
are2448x2048; aspect-preserving display512x428(rounded to a multiple of4).
Each downloaded file's byte count and SHA256 are recorded. No outside code was
executed; no external image editor or new pretrained model was used.

Local outputs/audit manifest:
`E:/LumiCycleCache/experiments/DarkDriving_private_audit12_20261023`.
AllfourPRIVATE_auditsheets were opened: all12night inputs, day references and
frozen16kparent predictions. This is TRAINdiagnostic review, not validation.

## Direct visual findings

The pairs are visibly real-camera roads with meaningful lighting changes, not
simulator renders or simply a darkening filter. Dominant buildings, roads,
gantries and vegetation broadly correspond. Some motion/pose/appearance
differences remain; exact dense alignment is **not** established by this review.

-1_1: same gantry/road layout, but night input has severe flare and headlight
  illumination/motion streaks. Day reference has neutral road lighting and no
  corresponding artificial glow. Parent retains the pool/flare and alters sky.
-2_2: oncoming vehicle lights dominate night; day vehicle appearance/position
  differs. Dynamic vehicle supervision needs exclusion/confidence, not blindL1.
-3_3: large building and tree structure correspond; parent keeps night-like
  foreground contrast, altered facade appearance and overly cool daylight.
-in_1samples: clear real nighttime headlight fields versus a day road with no
  pool. Parent produces broad unsupported colored/black sky patches or retains
  the pool. Motion blur makes road-marking pixel correspondence uncertain.
-in_2samples: road curve, fence and facade broadly align, but near-field blur
  and small positional differences persist. Parent has false dark sky bands,
  colored noise and bright source-light shapes.
-in_3: major building edge and distant scene correspond; parent adds sky
  striping/patches while leaving the road artificially lit.
-in_4samples: corresponding sign/building layout but changes in vegetation and
  perspective/position require checking. Parent keeps bright pools and sky
  artifacts rather than coherent daylight.
-out_1/out_4: day references explicitly turn off street/building lights and
  change sky/road illumination; parent keeps illuminated fixtures/windows and
  adds sky islands/halos.

These are useful target transformations for our actual failure modes. They do
not demonstrate transfer to BDD or prove every dataset pair usable. The current
parent remains rejected as a convincing complete inverse.

## Next gate before training

The audit manifest remains `approved_for_training:false`. Check correspondence
and reliable support, dynamic objects, duplicate groups and existing protected
BDDhashmetadata before selecting a broader TRAINsubset. Do not run blanket
full-image pixel supervision on this12pairfilename match. Prefer an explicit
registration/confidence audit and inspect warped references before training.
Night motion blur and saturated lamps are genuine information-loss cases; do
not interpret the day reference as observable source detail in those regions.

A subsequent experiment can warm-start the retained local generator using real
paired relighting, with a matched control and unchanged24REALBDDdevelopment
gate. Keep public deployment prohibited for data/derivative artifacts pending
rights clarification. Do not copy the authors' released enhancement model and
call its outputs ours. This audit has not trained or accepted any checkpoint.

155tests and relevantRuffpass. Newtests cover path traversal, exactTRAINpairing,
excluding detection/testdata and rejecting incomplete inventories. Production
generator/matte/routing/worker and all existing weights are unchanged. No
inverse training process was started; only the original publicworkerpair remains.
