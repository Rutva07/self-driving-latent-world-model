"""Mask-aware winner-take-all trajectory regression with mode probabilities."""

import torch
from torch.nn import functional as F


def trajectory_loss(output, batch, confidence_weight=0.15, fde_weight=0.2):
    pred = output["trajectories"]
    target = batch["future"].unsqueeze(2)
    valid = batch["future_mask"]
    eligible = valid.any(-1) & batch["history_mask"][:, :, -1]
    if not eligible.any():
        raise ValueError("Batch has no supervised trajectory targets")
    distance = torch.linalg.vector_norm(pred - target, dim=-1)
    count = valid.sum(-1).clamp(min=1)
    ade = (distance * valid.unsqueeze(2)).sum(-1) / count.unsqueeze(-1)
    best = ade.detach().argmin(-1)
    chosen = pred.gather(2, best[..., None, None, None].expand(
        -1, -1, 1, pred.shape[3], 2)).squeeze(2)
    huber = F.smooth_l1_loss(chosen, batch["future"], reduction="none", beta=1.0).sum(-1)
    per_agent = (huber * valid).sum(-1) / count
    # Last valid labeled frame, even for prematurely disappearing tracks.
    last_index = (valid.long() * torch.arange(1, valid.shape[-1] + 1,
                  device=valid.device)).amax(-1).clamp(min=1) - 1
    fde = torch.linalg.vector_norm(chosen.gather(
        2, last_index[..., None, None].expand(-1, -1, 1, 2)).squeeze(2) -
        batch["future"].gather(
            2, last_index[..., None, None].expand(-1, -1, 1, 2)).squeeze(2), dim=-1)
    classification = F.cross_entropy(output["logits"].flatten(0, 1),
                                     best.flatten(), reduction="none").reshape_as(best)
    count_agents = eligible.sum().clamp(min=1)
    loss_regression = (per_agent * eligible).sum() / count_agents
    loss_fde = (fde * eligible).sum() / count_agents
    loss_confidence = (classification * eligible).sum() / count_agents
    total = loss_regression + fde_weight * loss_fde + confidence_weight * loss_confidence
    return total, {"regression": loss_regression.detach(), "fde": loss_fde.detach(),
                   "confidence": loss_confidence.detach(), "total": total.detach()}
