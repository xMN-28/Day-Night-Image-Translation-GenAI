# LumiCycle Architecture

## System flow

```mermaid
flowchart LR
    BDD[BDD100K images + labels] --> FILTER[Strict day/night filter]
    FILTER --> HASH[Corruption checks + dHash groups]
    HASH --> SPLIT[Leakage-safe train / val / test]
    SPLIT --> BASE[CycleGAN baseline]
    SPLIT --> LUMI[LumiCycle training]
    BASE --> EVAL[Common evaluation protocol]
    LUMI --> EVAL
    TURBO[External Turbo reference] -. benchmark only .-> EVAL
    LUMI --> API[translate API]
    API --> UI[Offline Gradio demo]
```

## Bidirectional model

```mermaid
flowchart TB
    D[Real day] --> GDN[G day→night]
    GDN --> FN[Fake night]
    FN --> DN[Multi-scale night discriminator]
    FN --> GND[G night→day]
    GND --> RD[Reconstructed day]

    N[Real night] --> GND
    GND --> FD[Fake day]
    FD --> DD[Multi-scale day discriminator]
    FD --> GDN
    GDN --> RN[Reconstructed night]

    D -. identity .-> GND
    N -. identity .-> GDN
```

Each generator uses reflection padding, two downsampling stages, nine residual blocks, channel/spatial attention, two upsampling stages, and a `tanh` RGB output. LumiCycle retains CycleGAN’s cycle and identity objectives while adding:

- PatchNCE between spatial features before and after translation.
- DINOv2-S cosine distance to discourage semantic drift.
- Sobel edge distance to preserve geometry.
- Multi-scale spectral PatchGANs for global lighting and local texture.

## Training state machine

```mermaid
stateDiagram-v2
    [*] --> LoadOrInitialize
    LoadOrInitialize --> TrainMicrosteps
    TrainMicrosteps --> OptimizerStep
    OptimizerStep --> Log
    Log --> Validate: interval reached
    Log --> TrainMicrosteps: continue
    Validate --> SaveBest: score improved
    Validate --> RestoreAndReduce: plateau / collapse / saturation
    SaveBest --> TrainMicrosteps
    RestoreAndReduce --> TrainMicrosteps
    TrainMicrosteps --> SaveAtomic: time limit / Ctrl+C
    SaveAtomic --> [*]
```

The test split has no incoming edge to training or checkpoint selection. It is opened only by the final evaluator.

