"""Procedural driving scenes for offline smoke tests; not Waymo data."""

from pathlib import Path
import numpy as np

from lwm.data.schema import empty_scene, validate_scene, FUTURE_STEPS


def make_scene(seed, agents=24, map_points=128, difficulty="mixed"):
    rng = np.random.default_rng(seed)
    scene = empty_scene(agents, map_points)
    t = np.arange(-10, FUTURE_STEPS + 1, dtype=np.float32) / 10.0
    num = int(rng.integers(7, min(agents, 15) + 1))
    for a in range(num):
        typ = int(rng.choice([1, 2, 3], p=[0.72, 0.20, 0.08]))
        if a == 0:
            typ, x0, y0, yaw, speed, accel, turn = 1, 0.0, 0.0, 0.0, float(rng.uniform(4, 15)), 0.0, 0.0
        else:
            x0, y0 = float(rng.uniform(-25, 65)), float(rng.uniform(-22, 22))
            yaw = float(rng.choice([0, np.pi, np.pi / 2, -np.pi / 2]))
            speed = float(rng.uniform(0.4, 2.0) if typ == 2 else
                          rng.uniform(2, 7) if typ == 3 else rng.uniform(3, 17))
            accel = float(rng.uniform(-0.55, 0.55))
            turn = 0.0 if difficulty == "linear" else float(rng.choice([0, 0, 0, -0.13, 0.13]))
        if difficulty == "linear":
            accel, turn = 0.0, 0.0
        distance = speed * t + 0.5 * accel * t**2
        heading = yaw + turn * t
        xy = np.stack([x0 + distance * np.cos(yaw + turn * t / 2),
                       y0 + distance * np.sin(yaw + turn * t / 2)], -1)
        vel = (speed + accel * t)[:, None] * np.stack([np.cos(heading), np.sin(heading)], -1)
        feat = np.concatenate([xy, vel, np.sin(heading)[:, None], np.cos(heading)[:, None],
                               np.full((len(t), 1), 4.5 if typ == 1 else 0.7),
                               np.full((len(t), 1), 1.9 if typ == 1 else 0.6)], axis=1)
        scene["history"][a] = feat[:11]
        scene["history_mask"][a] = True
        scene["agent_type"][a] = typ
        scene["future"][a] = xy[11:]
        scene["future_mask"][a] = True
        if a > 0 and rng.random() < 0.18:
            scene["history_mask"][a, :int(rng.integers(1, 8))] = False
            scene["history"][a, ~scene["history_mask"][a]] = 0
        if a > 0 and rng.random() < 0.18:
            stop = int(rng.integers(20, FUTURE_STEPS))
            scene["future_mask"][a, stop:] = False
            scene["future"][a, stop:] = 0
    count = 0
    for lane_y in [-16., -12., -8., -4., 0., 4., 8., 12., 16.]:
        for x in np.linspace(-55, 110, 24):
            if count == map_points:
                break
            curve = 0.0015 * x * x * np.sign(lane_y)
            vec = np.array([1., 0.003 * x * np.sign(lane_y)], np.float32)
            vec /= np.linalg.norm(vec)
            scene["map"][count] = [x, lane_y + curve, vec[0], vec[1], 1 / 32, 0]
            scene["map_mask"][count] = True
            count += 1
        if count == map_points:
            break
    return validate_scene(scene)


def generate_split(directory, count, seed, difficulty="mixed", agents=24, map_points=128):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        scene = make_scene(seed + i, agents=agents, map_points=map_points, difficulty=difficulty)
        np.savez_compressed(directory / f"scene_{i:06d}.npz", **scene)
    return count
