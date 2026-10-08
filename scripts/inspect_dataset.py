#!/usr/bin/env python3
"""Inspect validation, masking, agent types and spatial distributions."""
import argparse
import json
from collections import Counter

import numpy as np

from lwm.data.dataset import SceneDataset
from lwm.data.schema import validate_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--limit", type=int, default=500)
    args = p.parse_args()
    ds = SceneDataset(args.data_dir)
    counts = Counter()
    futures = []
    neighbors = []
    maps = []
    for file in ds.files[:args.limit]:
        with np.load(file, allow_pickle=False) as z:
            scene = {k: z[k] for k in z.files}
        validate_scene(scene)
        visible = scene["history_mask"][:, -1]
        counts.update(map(int, scene["agent_type"][visible]))
        futures.append(float(scene["future_mask"].sum() / max(1, visible.sum() * 80)))
        neighbors.append(int(visible.sum()))
        maps.append(int(scene["map_mask"].sum()))
    print(json.dumps({"source": args.data_dir, "total_scenes": len(ds),
                      "inspected": min(args.limit, len(ds)),
                      "agent_type_counts": counts, "mean_visible_agents": np.mean(neighbors),
                      "mean_map_tokens": np.mean(maps),
                      "mean_future_valid_fraction": np.mean(futures)}, indent=2))


if __name__ == "__main__":
    main()
