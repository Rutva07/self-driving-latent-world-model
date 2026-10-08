#!/usr/bin/env python3
"""Offline end-to-end smoke test; generates local procedural data."""
import json
import tempfile
from pathlib import Path

import torch

from lwm.config import load_config
from lwm.data.dataset import SceneDataset
from lwm.data.synthetic import generate_split
from lwm.engine import load_for_evaluation, train, to_device


def main():
    with tempfile.TemporaryDirectory(prefix="lwm-smoke-") as temp:
        root = Path(temp)
        generate_split(root / "train", 12, 431, "linear", agents=8, map_points=32)
        generate_split(root / "val", 4, 953, "linear", agents=8, map_points=32)
        config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
        result = train(config, root / "train", root / "val", root / "run", "cpu")
        model, device, _ = load_for_evaluation(root / "run/best.pt", "cpu")
        sample = SceneDataset(root / "val")[0]
        with torch.no_grad():
            predictions = model(to_device({k: v.unsqueeze(0) for k, v in sample.items()}, device))
        assert predictions["trajectories"].shape == (1, 8, 3, 80, 2)
        assert torch.isfinite(predictions["trajectories"]).all()
        assert (root / "run/history.csv").exists()
        assert (root / "run/last.pt").exists()
        assert result["best_top1_ADE_8s_m"] < float("inf")
        print("SMOKE PASS: 2 training epochs, eval, checkpoint load, multimodal forecast")
        print(json.dumps({"best_top1_ADE_8s_m": result["best_top1_ADE_8s_m"],
                          "baseline_top1_ADE_8s_m": result["baseline"]["top1_ADE_8s_m"]},
                         indent=2))


if __name__ == "__main__":
    main()
