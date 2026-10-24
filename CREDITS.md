# Credits and licenses

LumiCycle integrates established image-translation components with local training,
sky repair, failure analysis and deployment. It does not claim invention of
CycleGAN (Zhu et al., ICCV 2017), CUT/PatchNCE (Park et al., ECCV 2020), attention,
residual networks, DINOv2 or distillation.

The local RGB student used externally pretrained CycleGAN-Turbo pseudo-targets.
The experimental inverse retains historical teacher-target supervision and CARLA
synthetic auxiliary training. Inference runs local feed-forward networks, not an
external image-generation API. Official teacher implementation:
[GaParmar/img2img-turbo](https://github.com/GaParmar/img2img-turbo), Parmar et al.,
*One-Step Image Translation with Text-to-Image Models* (2024).

BDD100K imagery is governed by its dataset terms. Curated comparison images are
illustrations, not a dataset release or measured success rate. CARLA paired-data
source metadata lists CC BY-SA 3.0; simulator images are not real paired photos.
DarkDriving audit data and derived results are not published here; the exposed
inverse was not trained on it. Repository licensing does not override upstream
code, data or pretrained-weight terms. Preserve required third-party notices.
