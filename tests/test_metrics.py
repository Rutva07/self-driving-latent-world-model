import torch

from lwm.data.synthetic import make_scene
from lwm.metrics import MetricAccumulator, constant_velocity


def test_constant_velocity_linear_dataset():
    scene = make_scene(101, agents=8, map_points=32, difficulty="linear")
    batch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in scene.items()}
    scores = MetricAccumulator().update(constant_velocity(batch), batch).compute()
    assert scores["top1_ADE_8s_m"] < 1e-4
    assert scores["top1_hit_8s_at_8m"] == 1.0


def test_metric_mask_invariant():
    scene = make_scene(101, agents=8, map_points=32)
    batch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in scene.items()}
    pred = constant_velocity(batch)
    before = MetricAccumulator().update(pred, batch).compute()
    batch["future"] = batch["future"].clone()
    batch["future"][~batch["future_mask"]] = 1000000.0
    after = MetricAccumulator().update(pred, batch).compute()
    assert before == after
