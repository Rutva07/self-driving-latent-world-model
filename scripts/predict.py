#!/usr/bin/env python3
"""Export predicted trajectories/probabilities in ego-relative meters."""
import argparse
import json

import torch

from lwm.data.dataset import SceneDataset
from lwm.engine import load_for_evaluation, to_device
from lwm.utils import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", default="outputs/prediction.json")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    sample = SceneDataset(args.data_dir)[args.index]
    model, device, _ = load_for_evaluation(args.checkpoint, args.device)
    with torch.no_grad():
        output = model(to_device({k: v.unsqueeze(0) for k, v in sample.items()}, device))
    probabilities = output["logits"][0].softmax(-1).cpu().tolist()
    paths = output["trajectories"][0].cpu().tolist()
    visible = sample["history_mask"][:, -1].tolist()
    result = {"coordinate_frame": "ego heading at t=0", "units": "meters",
              "timestep_seconds": 0.1, "forecast_agents": [
                  {"index": i, "type": int(sample["agent_type"][i]),
                   "mode_probabilities": probabilities[i], "predicted_xy": paths[i]}
                  for i, active in enumerate(visible) if active]}
    save_json(args.out, result)
    print(json.dumps({"saved": args.out, "agents": len(result["forecast_agents"])}, indent=2))


if __name__ == "__main__":
    main()
