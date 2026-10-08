"""Canonical scene schema. Arrays are float32 except masks/types."""

import numpy as np

HISTORY_STEPS = 11
FUTURE_STEPS = 80
AGENT_FEATURES = 8  # x,y,vx,vy,sin(yaw),cos(yaw),length,width
MAP_FEATURES = 6  # x,y,dir_x,dir_y,map_type/32,signal_state/8
MAX_AGENTS = 24
MAX_MAP_POINTS = 128


def empty_scene(agents=MAX_AGENTS, map_points=MAX_MAP_POINTS):
    return {
        "history": np.zeros((agents, HISTORY_STEPS, AGENT_FEATURES), np.float32),
        "history_mask": np.zeros((agents, HISTORY_STEPS), bool),
        "agent_type": np.zeros((agents,), np.int64),
        "future": np.zeros((agents, FUTURE_STEPS, 2), np.float32),
        "future_mask": np.zeros((agents, FUTURE_STEPS), bool),
        "map": np.zeros((map_points, MAP_FEATURES), np.float32),
        "map_mask": np.zeros((map_points,), bool),
    }


def validate_scene(scene):
    a, t, c = scene["history"].shape
    assert t == HISTORY_STEPS and c == AGENT_FEATURES, "Invalid history shape"
    assert scene["history_mask"].shape == (a, t)
    assert scene["agent_type"].shape == (a,)
    assert scene["future"].shape == (a, FUTURE_STEPS, 2)
    assert scene["future_mask"].shape == (a, FUTURE_STEPS)
    m = scene["map"].shape[0]
    assert scene["map"].shape == (m, MAP_FEATURES)
    assert scene["map_mask"].shape == (m,)
    assert scene["history_mask"][0, -1], "Agent zero must be visible ego"
    for key in ("history", "future", "map"):
        assert np.isfinite(scene[key]).all(), f"Nonfinite {key}"
    assert not scene["future_mask"][~scene["history_mask"][:, -1]].any(), (
        "Padded or currently unobserved agents cannot be prediction targets"
    )
    return scene
