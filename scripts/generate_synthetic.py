#!/usr/bin/env python3
"""Generate independently seeded procedural train/val/test splits."""
import argparse
from pathlib import Path

from lwm.data.synthetic import generate_split
from lwm.utils import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--train", type=int, default=640)
    p.add_argument("--val", type=int, default=128)
    p.add_argument("--test", type=int, default=128)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--difficulty", choices=["linear", "mixed"], default="mixed")
    p.add_argument("--agents", type=int, default=24)
    p.add_argument("--map-points", type=int, default=128)
    args = p.parse_args()
    if min(args.train, args.val, args.test) < 1 or args.agents < 7 or args.map_points < 1:
        p.error("All split sizes must be positive, agents >= 7, map-points >= 1")
    root = Path(args.out)
    for offset, (name, count) in enumerate((('train', args.train), ('val', args.val),
                                            ('test', args.test))):
        generate_split(root / name, count, args.seed + offset * 1000000,
                       args.difficulty, args.agents, args.map_points)
        print(f"{name}: {count} synthetic scenes")
    save_json(root / "metadata.json", {"source": "procedural synthetic (NOT Waymo)",
                                       "difficulty": args.difficulty,
                                       "seed": args.seed,
                                       "train": args.train, "val": args.val, "test": args.test})


if __name__ == "__main__":
    main()
