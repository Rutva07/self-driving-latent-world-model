#!/usr/bin/env python3
import argparse

from lwm.config import load_config
from lwm.engine import train


def main():
    p = argparse.ArgumentParser(description="Train multi-agent latent driving world model")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--output-dir", default="outputs/waymo")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--resume")
    p.add_argument("--no-map", action="store_true")
    p.add_argument("--no-interactions", action="store_true")
    args = p.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.no_map:
        config["model"]["use_map"] = False
    if args.no_interactions:
        config["model"]["use_interactions"] = False
    train(config, args.train_dir, args.val_dir, args.output_dir,
          device_name=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
