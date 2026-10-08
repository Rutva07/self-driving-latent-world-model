#!/usr/bin/env python3
"""Train full/no-map/no-neighbors models with identical data and seed."""
import argparse
from pathlib import Path

from lwm.config import load_config
from lwm.engine import train
from lwm.utils import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--out", default="outputs/ablations")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    summary = {}
    for name, overrides in (("full", {}), ("no_map", {"use_map": False}),
                            ("no_interactions", {"use_interactions": False})):
        cfg = load_config(args.config)
        cfg["model"].update(overrides)
        result = train(cfg, args.train_dir, args.val_dir,
                       Path(args.out) / name, device_name=args.device)
        summary[name] = {"best_top1_ADE_8s_m": result["best_top1_ADE_8s_m"]}
        save_json(Path(args.out) / "comparison.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
