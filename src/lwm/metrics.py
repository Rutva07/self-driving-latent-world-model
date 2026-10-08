"""Open-loop displacement metrics (not official Waymo challenge metrics)."""

import torch

from lwm.physics import rollout

HORIZONS = (10, 30, 50, 80)


def constant_velocity(batch, modes=1):
    return physics_baseline(batch, mode="cv", modes=modes)


def physics_baseline(batch, mode="ctrv", modes=1):
    prediction = rollout(batch, mode=mode)
    return {"trajectories": prediction.unsqueeze(2).expand(-1, -1, modes, -1, -1),
            "logits": torch.zeros((*prediction.shape[:2], modes), device=prediction.device)}


class MetricAccumulator:
    def __init__(self, horizons=HORIZONS):
        self.horizons = tuple(horizons)
        self.sums = {}
        self.counts = {}

    def add(self, name, values, mask):
        mask = mask.to(torch.bool)
        n = int(mask.sum())
        if not n:
            return
        self.sums[name] = self.sums.get(name, 0.0) + float(values[mask].double().sum().item())
        self.counts[name] = self.counts.get(name, 0) + n

    @torch.no_grad()
    def update(self, output, batch):
        pred = output["trajectories"]
        target = batch["future"][:, :, None]
        valid = batch["future_mask"] & batch["history_mask"][:, :, -1, None]
        distance = torch.linalg.vector_norm(pred - target, dim=-1)
        chosen = output["logits"].argmax(-1)
        top1 = distance.gather(2, chosen[..., None, None].expand(
            -1, -1, 1, distance.shape[-1])).squeeze(2)
        for h in self.horizons:
            h = min(h, valid.shape[-1])
            usable = valid[..., :h]
            num = usable.sum(-1)
            eligible = num > 0
            oracle_ade = ((distance[..., :h] * usable.unsqueeze(2)).sum(-1) /
                          num.clamp(min=1).unsqueeze(-1)).amin(-1)
            top_ade = (top1[..., :h] * usable).sum(-1) / num.clamp(min=1)
            suffix = f"{h / 10:g}s"
            self.add(f"minADE_{suffix}_m", oracle_ade, eligible)
            self.add(f"top1_ADE_{suffix}_m", top_ade, eligible)
            endpoint = valid[..., h - 1]
            oracle_fde = distance[..., h - 1].amin(-1)
            top_fde = top1[..., h - 1]
            self.add(f"minFDE_{suffix}_m", oracle_fde, endpoint)
            self.add(f"top1_FDE_{suffix}_m", top_fde, endpoint)
            threshold = 2.0 if h <= 10 else 4.0 if h <= 30 else 8.0
            self.add(f"top1_hit_{suffix}_at_{threshold:g}m",
                     (top_fde <= threshold).float(), endpoint)
            self.add(f"oracle_hit_{suffix}_at_{threshold:g}m",
                     (oracle_fde <= threshold).float(), endpoint)
        return self

    def compute(self):
        return {key: self.sums[key] / self.counts[key] for key in sorted(self.sums)}
