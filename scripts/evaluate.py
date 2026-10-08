#!/usr/bin/env python3
import argparse
import json

from lwm.engine import load_for_evaluation, evaluate_model, make_loader
from lwm.utils import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="")
    p.add_argument("--raw-weights", action="store_true", help="ignore EMA")
    args = p.parse_args()
    model, device, _ = load_for_evaluation(args.checkpoint, args.device,
                                            use_ema=not args.raw_weights)
    loader = make_loader(args.data_dir, args.batch_size, 0)
    learned = evaluate_model(model, loader, device)
    baseline = evaluate_model(model, loader, device, baseline=True)
    kinematic = evaluate_model(model, loader, device, baseline=True, physics_mode="ctrv")
    report = {"source": args.data_dir, "num_scenes": len(loader.dataset),
              "model": learned, "constant_velocity": baseline, "kinematic": kinematic,
              "note": "Custom open-loop metrics; not official Waymo leaderboard results"}
    print(json.dumps(report, indent=2))
    if args.out:
        save_json(args.out, report)


if __name__ == "__main__":
    main()
