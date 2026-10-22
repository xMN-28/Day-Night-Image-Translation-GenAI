# Showcase correction — 21 October 2026

## Why the first pack was withdrawn

The original 20 were accepted too readily from small contact sheets. Full-size
review showed obvious white pockets between cables, sky streaks, floating light
blobs and retained solar glow. In particular validation index 12 (the original
`01_val_0012.jpg`) is not a defensible good example. That selection was a review
failure; it was not evidence that fine-structure sky conversion had been solved.

The earlier guided sky-matte/compositing pipeline is a separate hybrid. The
current distilled student does not invoke it: it directly predicts RGB using a
downsampled ResNet bottleneck and resize-convolution decoder. Fine structures,
imperfect target supervision and the tradeoff between appearance change and
detail retention still produce unconverted or falsely illuminated pockets.
The new selection does not repair the model or validate a specific causal ablation.

## New review procedure and result

1. Generate all 500 validation-day candidates at 512 px with the same student.
2. Screen every contact sheet, rejecting obvious sun/glare, sky holes, floating
   glow and implausible bright sky columns; avoid near-duplicate scenes.
3. Inspect shortlisted input/output comparisons at full native 512×288 size.
4. Select 14 stronger cases rather than padding the folder to 20.
5. Rerun the exact JPEG upload files (not merely their pre-encoding originals).
6. Inspect all 14 freshly generated comparison panels at native size.

Retained indices, in showcase order:
37, 23, 161, 43, 393, 48, 407, 287, 397, 235, 367, 410, 102, 98.
They favor comparatively clean sky transitions, coherent nighttime road appearance
and lighting attached to cars, signs or facades, rather than isolated sky blobs.
They are **not certified artifact-free**: softness, altered text, facade lighting
and inaccurate illumination remain. This is a subjective best-case showcase,
not an unbiased held-out test result or proof of generalization.

Checkpoint: `E:/LumiCycleCache/experiments/turbo_vgg_target_control/best.pt`.
Global step 30,600; SHA-256:
`1277b23bda9c13bc4797a19f44fa164000060308b5270a2b436c9b6617ce2185`.
Use the explicitly external-target-supervised local student option, edge 512.
No architecture/weight/default deployment change and no training occurred.

The old folder was recoverably moved to
`outputs/demo_showcase_rejected_20261021_01`, not deleted. The replacement is
`outputs/demo_showcase`; its `daytime_inputs` now contains exactly 14 uploads.
Local PNGs and paired panels are freshly generated from those uploads. The
manifest records checkpoint identity and SHA-256 of each actual upload file.
All licensed image assets remain ignored/local.

`daynight.curate_demo` now refuses to reuse a nonempty showcase folder and rejects
changed checkpoint hashes. This prevents obsolete rejected files from leaking
into a replacement selection and prevents mislabeled checkpoint provenance.
