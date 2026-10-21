"""Optional paired pseudo-target objectives; not claims of real paired ground truth."""

import torch
import torch.nn.functional as F


def highlight_fidelity(prediction, target, threshold=0.55):
    """Normalize per image so small teacher light sources are not averaged away.

    Inputs use [-1,1]. Detached soft weights depend only on the teacher target.
    This can also emphasize white signs/cars: it is not an emitter detector.
    """
    rgb = (target.detach().float() + 1) / 2
    brightness = rgb.amax(1, keepdim=True)
    weight = ((brightness - threshold) / (1 - threshold)).clamp(0, 1)
    error = (prediction.float() - target.float()).abs().mean(1, keepdim=True)
    return ((error * weight).sum((1, 2, 3)) / weight.sum((1, 2, 3)).clamp_min(1)).mean()


def conditioned_pair(source, candidate):
    """Keep correspondence: the critic sees source RGB plus candidate RGB."""
    if source.shape != candidate.shape or source.ndim != 4 or source.shape[1] != 3:
        raise ValueError("Conditional critic expects matching BCHW RGB tensors")
    return torch.cat((source, candidate), dim=1)


def light_footprint_fidelity(prediction, target, threshold=0.55):
    """Match soft bright-region shape, charging for extra area as well as misses.

    Squared error is normalized by *target* bright-region energy per image.
    Unlike normalizing by predicted area, expanding a blob cannot dilute its cost.
    Brightness is not a semantic emitter label; signs/cars can also qualify.
    The RGB/highlight objective supplies gradients for lights below this threshold.
    """

    def footprint(image):
        brightness = ((image.float() + 1) / 2).amax(1)
        return ((brightness - threshold) / (1 - threshold)).clamp(0, 1)

    predicted = footprint(prediction)
    expected = footprint(target.detach())
    return (
        (predicted - expected).square().sum((1, 2)) / expected.square().sum((1, 2)).clamp_min(1)
    ).mean()


def masked_target_fidelity(prediction, target, mask, minimum_mass=64):
    """Region-normalized RGB agreement with the NIGHT target, not daytime source.

    Frozen soft masks cannot shrink to evade this objective. Tiny uncertain regions
    receive reduced weight. No source edge/chroma equality is imposed.
    """
    if prediction.shape != target.shape or mask.shape != prediction[:, :1].shape:
        raise ValueError("Expected matching BCHW RGB prediction/target and B1HW mask")
    weight = mask.detach().float().clamp(0, 1)
    error = (prediction.float() - target.detach().float()).abs().mean(1, keepdim=True)
    return ((error * weight).sum((1, 2, 3)) / weight.sum((1, 2, 3)).clamp_min(minimum_mass)).mean()


def least_squares_critic(real_logits, fake_logits):
    return 0.5 * (
        F.mse_loss(real_logits, torch.ones_like(real_logits)) + fake_logits.square().mean()
    )


def least_squares_generator(fake_logits):
    return F.mse_loss(fake_logits, torch.ones_like(fake_logits))
