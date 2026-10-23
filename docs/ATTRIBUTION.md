# Credits, licenses and provenance

Local contributions include integration, training recipes, safeguards, sky repair,
experimental audits and deployment. Established components are not claimed as
new inventions.

- CycleGAN: Zhu et al., ICCV 2017, cycle-consistent unpaired translation.
- CUT / PatchNCE: Park et al., ECCV 2020, contrastive image translation.
- DINOv2: Oquab et al., frozen features in relevant training recipes.
- [CycleGAN-Turbo / img2img-turbo](https://github.com/GaParmar/img2img-turbo):
  Parmar et al., *One-Step Image Translation with Text-to-Image Models* (2024).
  Its external RGB pseudo-targets supervised the local day-to-night student and
  the inverse's earlier lineage. This is not fully from-scratch training and is
  distinct from using the teacher at inference.
- The supplied IEEE paper and additional research are discussed in
  [the master guide](PROJECT_MASTER_GUIDE.md) and literature reviews.

Optional depth, segmentation and flow tools are experiment-specific, not
components used by every deployed generator. FLUX was an external reference,
not the public inference model.

## Data rights

- BDD100K: obtain under its own terms. README comparisons are curated validation
  illustrations, not a dataset release or measured success rate.
- CARLA day/night pairs: synthetic simulator images, not real paired photographs.
  Source metadata lists CC BY-SA 3.0; see the source audit for attribution.
- LoLIStreet training night variants include synthetic darkening; do not label
  them real night photographs.
- DarkDriving: permissive Kaggle mirror metadata does not replace the author's
  restrictive academic/noncommercial/no-redistribution license. Private audit
  images and derived results remain outside Git and the public demo. The exposed
  16.5k inverse was not trained on DarkDriving.

See [paired-source audit](NIGHT_TO_DAY_PAIRED_SOURCE_AUDIT_2026_10_23.md),
[DarkDriving audit](NIGHT_TO_DAY_DARKDRIVING_PRIVATE_AUDIT.md), and
[teacher evaluation](REFERENCE_TEACHER_EVALUATION.md). Repository licensing does
not override third-party code, dataset or pretrained-weight terms.
