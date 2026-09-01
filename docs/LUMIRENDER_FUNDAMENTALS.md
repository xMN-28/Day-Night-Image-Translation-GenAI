# LumiRender fundamentals

LumiRender treats day-to-night conversion as an underdetermined inverse-rendering problem. A single daytime photograph does not reveal which lamps will be on, whether the road will be wet, or what exposure a nighttime camera would choose. The model therefore preserves inferred scene properties and samples only the missing nighttime conditions. A seed makes that sample reproducible.

## Image formation

The network first converts display-encoded sRGB values to approximate linear radiance. Lighting arithmetic in sRGB is incorrect because sRGB is gamma encoded. For each pixel, LumiRender approximates

```text
I_linear = R * (L_ambient + L_local) + S + Reflection
```

where `R` is reflectance/albedo, `L` is incident illumination, and `S` is a roughness-aware specular term. This decomposition is not uniquely recoverable from one RGB image, so frozen depth and segmentation models provide training-only pseudo-labels and reconstruction/regularization losses prevent arbitrary solutions.

The result passes through an explicit camera model:

```text
I_srgb = encode(tone_map(vignette(exposure * white_balance * I_linear) + sensor_noise))
```

The camera stage learns exposure, white balance, highlight compression, vignetting, shot noise, and read noise. Shot noise scales with the square root of radiance; read noise is approximately signal independent.

## What “Gaussian” means here

Three Gaussian ideas are useful, and one is intentionally excluded:

1. **Anisotropic 2D Gaussian illumination.** Each sampled lamp has a center, two scales, orientation, color and intensity. On-frame centers are anchored to predicted emitter candidates; off-frame centers may illuminate the image edge. This follows the practical idea used in Punnappurath et al.'s [CVPR 2022 day-to-night synthesis](https://openaccess.thecvf.com/content/CVPR2022/html/Punnappurath_Day-to-Night_Image_Synthesis_for_Training_Nighttime_Neural_ISPs_CVPR_2022_paper.html).
2. **Gaussian point-spread functions.** Multi-scale convolution around bright emitters approximates lens bloom and glare. It is deliberately bounded so lamps glow without erasing branches or signs.
3. **Gaussian/Laplacian frequency separation.** The learned correction is high-pass filtered and limited to ±0.03 in linear RGB. It cannot repaint the coarse illumination field.
4. **3D Gaussian splatting is excluded.** It reconstructs/render views from multi-view observations. LumiRender receives one photograph, has no camera trajectory, and is not a novel-view synthesis system.

## Geometry, materials and reflections

The factorizer predicts relative depth, surface normals, roughness, wetness and six soft semantic regions. The renderer combines a Lambert-like diffuse term with a bounded Blinn–Phong/GGX-inspired highlight term. This is an image-space approximation, not a full path tracer.

Road reflections are the vertically flipped light field, softened, stretched vertically and gated by road probability, relative depth and wetness. Glass receives a smaller roughness-gated reflection. This handles plausible light streaks without claiming to recover unseen scene geometry. Optical-flow correspondence confidence removes occluded and strongly moving regions from paired losses.

## Why a bounded correction network exists

An analytic image-space renderer misses fine effects such as small halos and local color bleeding. A shallow correction network can repair high-frequency errors, but its output is high-pass filtered and clamped. It cannot introduce a new car, move a branch, or replace the sky illumination. The acceptance evaluator checks the bound directly.

## Fundamental limitations

- Albedo and illumination are ambiguous in one photograph.
- Lights behind the camera or outside the frame are unknowable.
- Screen-space reflections cannot reproduce occluded reflected geometry.
- A Cityscapes semantic teacher is road biased; timelapse data is needed for broader scenes.
- The renderer produces a plausible night, not the one true future nighttime state.

These limitations are why LumiRender exposes seed, intensity and wetness controls and why claims are based on a fixed blind comparison suite rather than one attractive example.
