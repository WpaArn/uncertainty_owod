import torch
import torch.nn as nn
import math
from torch.nn import functional as F

def slide_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    auto_iou: float = 0.5,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Slide Loss used for adjusting weights between easy and hard examples.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 or 1).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        auto_iou: A float value for adjusting the modulating weight.
        reduction: 'none' | 'mean' | 'sum'
                   'none': No reduction will be applied to the output.
                   'mean': The output will be averaged.
                   'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_modulation = ((1 - p_t) ** gamma)
    loss = ce_loss * focal_modulation

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if auto_iou < 0.2:
        auto_iou = 0.2

    b1 = targets <= auto_iou - 0.1
    a1 = 1.0
    b2 = (targets > (auto_iou - 0.1)) & (targets < auto_iou)
    a2 = math.exp(1.0 - auto_iou)
    b3 = targets >= auto_iou
    a3 = torch.exp(-(targets - 1.0))
    modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
    loss *= modulating_weight

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss