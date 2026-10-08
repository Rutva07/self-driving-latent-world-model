#!/usr/bin/env python3
"""Visualize one ego-centric scene with all predicted modes."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lwm.data.dataset import SceneDataset
from lwm.engine import load_for_evaluation, to_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--agent", type=int, default=1)
    p.add_argument("--out", default="outputs/scene.png")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    scene = SceneDataset(args.data_dir)[args.index]
    model, device, _ = load_for_evaluation(args.checkpoint, args.device)
    with torch.no_grad():
        out = model(to_device({k: v.unsqueeze(0) for k, v in scene.items()}, device))
    a = args.agent
    if not scene["history_mask"][a, -1]:
        raise ValueError("Selected agent is not visible; choose a different --agent")
    map_xy = scene["map"][scene["map_mask"], :2].numpy()
    hist = scene["history"][a, scene["history_mask"][a], :2].numpy()
    future = scene["future"][a, scene["future_mask"][a]].numpy()
    pred = out["trajectories"][0, a].cpu().numpy()
    scores = torch.softmax(out["logits"][0, a], -1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 7))
    if len(map_xy):
        ax.scatter(map_xy[:, 0], map_xy[:, 1], s=1.7, c="0.75", label="Road samples")
    ax.plot(hist[:, 0], hist[:, 1], "k.-", lw=2, label="Agent history")
    if len(future):
        ax.plot(future[:, 0], future[:, 1], "g-", lw=2, label="Ground truth")
    for i, path in enumerate(pred):
        ax.plot(path[:, 0], path[:, 1], lw=1.4, alpha=0.7,
                label=f"Mode {i} (p={scores[i]:.2f})")
    ax.scatter([0], [0], marker="*", c="red", s=150, label="Ego at t=0")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ego-relative X (m)")
    ax.set_ylabel("Ego-relative Y (m)")
    ax.set_title(f"Scene {args.index}, agent {a}: {len(pred)} future modes")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
