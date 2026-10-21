# LumiCycle Team Video Watchlist

This list is intentionally short. Watch the four core videos in order, then read the matching
sections of `PROJECT_MASTER_GUIDE.md`. Watching alone is not enough: after each video, one teammate
should explain the idea without looking at the screen.

## Core path — approximately 30 minutes

### 1. GAN fundamentals

[What are GANs (Generative Adversarial Networks)? — IBM Technology](https://www.youtube.com/watch?v=TpMIssRdhco)

Learn these ideas:

- generator versus discriminator;
- why their training is adversarial;
- why a generated image can look realistic without matching a particular input scene.

After watching, answer: **Which network creates our nighttime image, and which network is thrown
away during inference?**

### 2. CycleGAN in under five minutes

[CycleGAN Explained in 5 Minutes! — Matchue](https://www.youtube.com/watch?v=-8hfnlxEPn4)

Learn these ideas:

- paired versus unpaired image translation;
- two generators and two discriminators;
- adversarial loss and cycle-consistency loss;
- why day → night → day should reconstruct the original scene.

This is the most important video for the project. After watching, draw the full cycle in both
directions from memory.

### 3. ResNet generator blocks

[ResNet (actually) explained in under 10 minutes — rupert ai](https://www.youtube.com/watch?v=o_3mboe1jYI)

Learn these ideas:

- residual/skip connection;
- why a block learns a correction instead of rebuilding the whole representation;
- why gradients can pass through deep stacks more easily.

Our generators contain nine residual blocks. Do not confuse these internal residual skips with
U-Net encoder-to-decoder skip connections; LumiCycle V2 does not have the latter.

### 4. CBAM attention

[Convolutional Block Attention Module (CBAM) Paper Explained — Soroush Mehraban](https://www.youtube.com/watch?v=lpFufRf9jFI)

The useful portion is only about six minutes. Learn:

- channel attention: **what feature types matter?**
- spatial attention: **where in the feature map do they matter?**
- how CBAM is inserted into a residual block.

Remember: CBAM does not explicitly know what a sky, road, or headlight is. It learns feature gates
from the training objective.

## Optional but strongly recommended — 19 minutes

[CycleGAN Paper Walkthrough — Aladdin Persson](https://www.youtube.com/watch?v=5jziBapziYE)

This covers the problem statement, cycle loss, network architecture, identity loss, limitations,
and implementation details. It is the best single follow-up before a viva. Relevant chapters:

- `02:00` problem statement;
- `04:30` training and cycle consistency;
- `07:16` losses;
- `09:20` architecture;
- `13:15` identity loss;
- `14:44` limitations;
- `16:08` implementation details.

## Optional detailed alternative — 26 minutes

[Unpaired Image-to-Image Translation Using CycleGAN — DigitalSreeni](https://www.youtube.com/watch?v=VzIO5_R9XEM)

Watch this if PatchGAN, generator structure, or InstanceNorm is still unclear. The implementation
shown is not our repository and may use another framework; study the concepts, then verify our
actual implementation in `daynight/models/networks.py`.

## What the videos do not teach about our custom V2

The videos explain established components. After watching, use `PROJECT_MASTER_GUIDE.md` to learn
the parts specific to this project:

- two PatchGAN scales plus one whole-frame global critic per domain;
- PatchNCE, frozen DINOv2, and Sobel content preservation;
- custom color-statistics and regional-illumination losses;
- why V2 reused V1 generators but reset critics and optimizers;
- why V2 step 4,500 is preferred over final step 12,000;
- why V2.1 and LumiRender failed;
- current limitations and claims the team must not make.

## Group check after watching

Without notes, every member should answer:

1. Why did we choose CycleGAN rather than ordinary supervised image translation?
2. What are the two generators and two target domains?
3. What does cycle consistency protect, and how can it also hurt the translation?
4. Why does PatchGAN output a grid instead of one score?
5. Why did V2 add a global critic?
6. What does CBAM do, and what does it *not* understand?
7. Which networks are required when the user presses Translate?
8. What did our team modify beyond standard CycleGAN?
