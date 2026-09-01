# Viva Questions and Answers

## Core understanding

### What problem are you solving?

We translate RGB road scenes between day and night without requiring the same scene to be photographed at both times. The model is bidirectional and is demonstrated through an upload-based application.

### Why is the dataset unpaired?

Exact pairs are difficult because traffic, pedestrians, weather, camera position, and exposure change over time. Unpaired learning only needs a collection of day images and a separate collection of night images.

### What does a GAN do?

A generator creates translated images. A discriminator learns to distinguish generated images from real target-domain images. Their competition encourages realism.

### Why are there two generators?

One maps day to night and the other maps night to day. Together they support cycle reconstruction, which constrains the underdetermined unpaired mapping.

### What is cycle consistency?

After translating day to night, translating the result back should approximately recover the original day image. The reverse direction follows the same rule.

### Why is identity loss needed?

When a generator already receives an image from its output domain, it should mostly leave it unchanged. This discourages unnecessary color and structural changes.

## LumiCycle contribution

### How is LumiCycle different from CycleGAN?

It adds attention inside the generators, two discriminator scales with spectral normalization, PatchNCE content preservation, frozen DINO semantic consistency, edge consistency, EMA inference, and stronger training/evaluation safeguards.

### Why use attention?

Day/night changes are spatially uneven. Attention can emphasize regions and feature channels associated with sky, road, headlights, signs, and deep shadows.

### What is PatchNCE?

It treats corresponding spatial features as positive pairs and other locations as negatives. This encourages the translated output to preserve the input layout.

### Why use DINOv2?

DINOv2 provides strong visual representations. We freeze it and use only its feature distance, giving a semantic preservation signal without training a new semantic network.

### Does using DINO mean the project is copied?

No. Pretrained feature losses are standard research components, like using ImageNet features for perceptual loss. We attribute DINO and do not claim it as ours. Our work is the architecture, integration, training pipeline, experiments, and application.

### Why use edge loss?

Edges represent geometry such as lanes, vehicles, poles, and signs. Adversarial realism alone can change these structures.

## Data and evaluation

### What was wrong with mixing test images during NAS?

The test set is supposed to estimate performance on unseen data. Using it to choose architecture leaks information and makes the final score optimistic.

### How did you avoid leakage?

We group exact and near-duplicate images using perceptual hashes before assigning groups to train, validation, or test. Test manifests are never loaded by training.

### Why not use only SSIM or PSNR?

Those metrics compare aligned pixels. In unpaired translation, the real target night image is a different scene, so pixel comparison is not meaningful.

### What does FID measure?

FID compares feature distributions of generated and real target images. Lower values generally indicate closer distribution matching, but FID alone does not prove object preservation.

### Why include detector retention?

A visually attractive translation may erase a pedestrian or change a vehicle. We run the same frozen detector before and after translation and match detections by class and overlap.

### Is detector retention the same as ground-truth mAP?

No. It is a consistency measure. Ground-truth mAP is stronger when compatible labels and transforms are available. We name the metric accurately and do not overclaim it.

## Training and hardware

### Is 12 GB VRAM enough?

Yes for the designed 256 px, batch-one BF16 training. We measured a complete LumiCycle generator objective including attention, both discriminator scales, PatchNCE, edge loss, and DINO at about 2.11 GB peak allocated VRAM on the target machine, leaving headroom for the remaining training state and desktop use. Larger batches or high-resolution training would require more memory.

### Why gradient accumulation?

It combines gradients from multiple microbatches before an optimizer step, approximating a larger batch without storing all images simultaneously.

### What is EMA?

Exponential moving average smooths generator weights across training steps. The averaged weights often give more stable inference than the latest raw update.

### How does overnight resume work?

The checkpoint stores every state required to continue: networks, optimizers, schedulers, replay buffers, random states, step, metrics, and configuration. Files are written atomically so interruption cannot replace the last good checkpoint with a partial file.

### Can you guarantee the loss never plateaus?

No honest GAN project can guarantee that. We prevent common causes, monitor multiple symptoms, preserve diagnostics, restore the best checkpoint, and reduce learning rates when validation stagnates.

## Limitations

### Can night-to-day recover invisible details?

No. It can create a plausible estimate using learned priors, but details never captured by the sensor are unknowable.

### Is this safe for autonomous driving?

Not as a direct control component. Generated images can hallucinate. The project is a research and demonstration pipeline and explicitly evaluates safety-related failure modes.

### Why compare with CycleGAN-Turbo?

It is a strong modern reference showing what a pretrained diffusion prior can achieve. We label it external and keep our local LumiCycle as the default model.
