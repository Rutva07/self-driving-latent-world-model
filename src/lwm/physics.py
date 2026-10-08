"""Observed-history-only kinematic rollout priors."""

import torch


def rollout(batch, mode="ctrv"):
    """Return [B,A,H,2] positions: constant velocity or constant-turn-rate/acceleration."""
    history = batch["history"]
    B, A, T, _ = history.shape
    H = batch["future"].shape[-2]
    pos = history[:, :, -1, :2]
    velocity = history[:, :, -1, 2:4]
    if mode == "cv":
        time = torch.arange(1, H + 1, device=history.device,
                            dtype=history.dtype).view(1, 1, H, 1) / 10
        return pos.unsqueeze(-2) + velocity.unsqueeze(-2) * time
    if mode != "ctrv":
        raise ValueError(f"Unknown physics prior: {mode}")
    seen = batch["history_mask"].clone()
    seen[..., -1] = False
    index = torch.arange(T, device=history.device)
    # Use a valid sample close to 0.5 seconds in the past, if one exists.
    difference = torch.abs(index - (T - 6))
    cost = torch.where(seen, difference, 100000)
    prev = cost.argmin(-1)
    has_prev = seen.any(-1)
    prev_values = history.gather(2, prev[..., None, None].expand(-1, -1, 1, 8)).squeeze(2)
    delta_t = ((T - 1 - prev).clamp(min=1) / 10).to(history.dtype)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    past_speed = torch.linalg.vector_norm(prev_values[..., 2:4], dim=-1)
    acceleration = ((speed - past_speed) / delta_t).clamp(-3, 3)
    yaw_now = torch.atan2(history[..., -1, 4], history[..., -1, 5])
    yaw_prev = torch.atan2(prev_values[..., 4], prev_values[..., 5])
    delta_yaw = yaw_now - yaw_prev
    delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))
    yaw_rate = (delta_yaw / delta_t).clamp(-0.4, 0.4)
    acceleration = torch.where(has_prev, acceleration, 0.)
    yaw_rate = torch.where(has_prev, yaw_rate, 0.)
    # Initial velocity direction can differ slightly from bounding-box heading.
    vel_heading = torch.atan2(velocity[..., 1], velocity[..., 0])
    tmid = (torch.arange(H, device=history.device,
                         dtype=history.dtype) + 0.5).view(1, 1, H) / 10
    speeds = (speed[..., None] + acceleration[..., None] * tmid).clamp(min=0)
    angles = vel_heading[..., None] + yaw_rate[..., None] * tmid
    step = torch.stack([speeds * torch.cos(angles),
                        speeds * torch.sin(angles)], dim=-1) / 10
    return pos[..., None, :] + torch.cumsum(step, dim=-2)
