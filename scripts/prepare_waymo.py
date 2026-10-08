#!/usr/bin/env python3
"""Prepare licensed WOMD tf.Example shards into masked ego-centric NPZ scenes."""
import argparse
import glob
from pathlib import Path

from lwm.data.waymo import convert_shards
from lwm.utils import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="glob for WOMD tf.Example TFRecord shards")
    p.add_argument("--output", required=True, help="directory for converted scenes")
    p.add_argument("--max-scenarios", type=int, default=0, help="0 = all")
    p.add_argument("--max-agents", type=int, default=24)
    p.add_argument("--max-map-points", type=int, default=128)
    p.add_argument("--verify-crc", action="store_true", help="slower, strict TFRecord CRC checks")
    args = p.parse_args()
    files = [Path(x) for x in glob.glob(args.input)]
    if not files:
        p.error(f"No shards match: {args.input}")
    result = convert_shards(files, args.output, args.max_scenarios,
                            args.max_agents, args.max_map_points, args.verify_crc)
    if not result["converted"]:
        raise SystemExit("No scenarios converted. Verify that shards are tf.Example, not Scenario proto.")
    print(result)
    save_json(Path(args.output) / "conversion.json", {"format": "WOMD tf.Example",
             "source_shards": [str(x) for x in sorted(files)],
             "max_agents": args.max_agents, "max_map_points": args.max_map_points, **result})


if __name__ == "__main__":
    main()
