"""Convert Waymo WOMD tf.Example features to ego-centric motion scenes."""

import hashlib
from pathlib import Path

import numpy as np

from lwm.data.schema import empty_scene, validate_scene, HISTORY_STEPS, FUTURE_STEPS
from lwm.data.tfrecord import iter_tfrecord, read_example


def _array(features, key, rows, cols, dtype=np.float32, optional=False):
    if key not in features:
        if optional:
            return np.zeros((rows, cols), dtype)
        raise KeyError(f"Missing Waymo feature '{key}'; use WOMD tf.Example format")
    v = np.asarray(features[key], dtype=dtype)
    if v.size != rows * cols:
        raise ValueError(f"{key}: expected {rows}x{cols}, found {v.size} entries")
    return v.reshape(rows, cols)


def _xy_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def convert_example(features, max_agents=24, max_map_points=128):
    if max_agents < 1 or max_map_points < 1:
        raise ValueError("max_agents and max_map_points must be positive")
    n = len(features["state/current/x"])
    if not n:
        raise ValueError("No tracks in scenario")
    is_sdc = np.asarray(features["state/is_sdc"], dtype=np.int64)
    if is_sdc.size != n or is_sdc.sum() != 1:
        raise ValueError("Exactly one SDC is required")
    ego_id = int(np.flatnonzero(is_sdc)[0])
    current_mask = _array(features, "state/current/valid", n, 1, np.int64)[:, 0].astype(bool)
    if not current_mask[ego_id]:
        raise ValueError("SDC must be valid at current time")
    curr_x = _array(features, "state/current/x", n, 1)[:, 0]
    curr_y = _array(features, "state/current/y", n, 1)[:, 0]
    curr_yaw = _array(features, "state/current/bbox_yaw", n, 1)[:, 0]
    origin = np.array([curr_x[ego_id], curr_y[ego_id]], np.float32)
    transform = _xy_rotation(-float(curr_yaw[ego_id]))
    dist = np.hypot(curr_x - origin[0], curr_y - origin[1])
    priority = np.asarray(features.get("state/tracks_to_predict", [0] * n), dtype=np.int64)
    if priority.size != n:
        priority = np.zeros(n, np.int64)
    other = np.flatnonzero(current_mask & (np.arange(n) != ego_id))
    other = sorted(other, key=lambda a: (-int(priority[a] != 0), float(dist[a]), int(a)))
    chosen = [ego_id] + other[:max_agents - 1]
    scene = empty_scene(max_agents, max_map_points)
    types = np.asarray(features["state/type"], dtype=np.int64)
    if types.size != n:
        raise ValueError("Wrong state/type length")
    arrays = {}
    for group, steps in (("past", 10), ("current", 1), ("future", FUTURE_STEPS)):
        arrays[group] = {}
        for name in ("x", "y", "velocity_x", "velocity_y", "bbox_yaw", "length", "width", "valid"):
            arrays[group][name] = _array(features, f"state/{group}/{name}", n, steps)
    for output_a, source_a in enumerate(chosen):
        hist = {key: np.concatenate([arrays["past"][key][source_a],
                                     arrays["current"][key][source_a]])
                for key in arrays["past"]}
        valid = hist["valid"].astype(bool)
        xy = (np.stack([hist["x"], hist["y"]], axis=-1) - origin) @ transform.T
        velocity = np.stack([hist["velocity_x"], hist["velocity_y"]], -1) @ transform.T
        heading = hist["bbox_yaw"] - curr_yaw[ego_id]
        values = np.stack([xy[:, 0], xy[:, 1], velocity[:, 0], velocity[:, 1],
                           np.sin(heading), np.cos(heading), hist["length"],
                           hist["width"]], axis=-1)
        values[~valid] = 0
        scene["history"][output_a] = values
        scene["history_mask"][output_a] = valid
        scene["agent_type"][output_a] = int(np.clip(types[source_a], 0, 4))
        future = arrays["future"]
        fvalid = future["valid"][source_a].astype(bool) & valid[-1]
        fxy = (np.stack([future["x"][source_a], future["y"][source_a]], -1)
               - origin) @ transform.T
        fxy[~fvalid] = 0
        scene["future"][output_a] = fxy
        scene["future_mask"][output_a] = fvalid
    xyz = np.asarray(features.get("roadgraph_samples/xyz", []), np.float32)
    dirs = np.asarray(features.get("roadgraph_samples/dir", []), np.float32)
    kinds = np.asarray(features.get("roadgraph_samples/type", []), np.int64)
    valid_map = np.asarray(features.get("roadgraph_samples/valid", []), np.int64)
    tokens = []
    if xyz.size:
        if xyz.size % 3 or dirs.size != xyz.size:
            raise ValueError("Invalid roadgraph dimensions")
        count = xyz.size // 3
        if kinds.size != count or valid_map.size != count:
            raise ValueError("Invalid roadgraph type/valid dimensions")
        pos = (xyz.reshape(-1, 3)[:, :2] - origin) @ transform.T
        direction = dirs.reshape(-1, 3)[:, :2] @ transform.T
        idx = np.flatnonzero((valid_map > 0) & (np.linalg.norm(pos, axis=-1) < 120))
        if len(idx):
            idx = idx[np.argsort(np.sum(pos[idx] ** 2, axis=-1), kind="stable")]
            for j in idx[:max_map_points]:
                tokens.append([*pos[j], *direction[j], np.clip(kinds[j], 0, 31) / 32, 0.0])
    # Only current traffic lights are observed; never read future traffic lights.
    lights = features.get("traffic_light_state/current/valid")
    if lights:
        lvalid = np.asarray(lights, np.int64).reshape(-1)
        lx = np.asarray(features["traffic_light_state/current/x"], np.float32)
        ly = np.asarray(features["traffic_light_state/current/y"], np.float32)
        state = np.asarray(features["traffic_light_state/current/state"], np.int64)
        if len(lvalid) != len(lx) or len(lx) != len(ly) or len(ly) != len(state):
            raise ValueError("Mismatched traffic light fields")
        for k in np.flatnonzero(lvalid):
            relative = (np.array([lx[k], ly[k]], np.float32) - origin) @ transform.T
            if np.linalg.norm(relative) < 120:
                tokens.append([relative[0], relative[1], 0, 0, 20 / 32,
                               np.clip(state[k], 0, 8) / 8])
    if tokens:
        # Reserve nearby signal tokens and sample the rest of the map by distance.
        tokens = sorted(tokens, key=lambda token: token[0] ** 2 + token[1] ** 2)
        tokens = np.asarray(tokens[:max_map_points], np.float32)
        scene["map"][:len(tokens)] = tokens
        scene["map_mask"][:len(tokens)] = True
    validate_scene(scene)
    sid = features.get("scenario/id", [b"unknown"])[0]
    sid = sid.decode("utf-8", errors="replace") if isinstance(sid, bytes) else str(sid)
    return scene, sid


def convert_shards(shards, output_dir, max_scenarios=0, max_agents=24,
                   max_map_points=128, verify_crc=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count, skipped = 0, 0
    for path in sorted(map(Path, shards)):
        for payload in iter_tfrecord(path, verify_crc=verify_crc):
            try:
                scene, sid = convert_example(read_example(payload), max_agents,
                                             max_map_points)
            except (ValueError, KeyError, AssertionError) as error:
                skipped += 1
                if skipped <= 3:
                    print(f"Skipping invalid scenario in {path.name}: {error}")
                continue
            fingerprint = hashlib.sha256(f"{path.name}:{sid}".encode()).hexdigest()[:20]
            dest = output_dir / f"{fingerprint}.npz"
            np.savez_compressed(dest, **scene)
            count += 1
            if max_scenarios and count >= max_scenarios:
                return {"converted": count, "skipped": skipped}
    return {"converted": count, "skipped": skipped}
