import torch

from lwm.data.synthetic import make_scene
from lwm.losses import trajectory_loss
from lwm.models.world_model import LatentWorldModel


def batch():
    sc = make_scene(4, agents=8, map_points=32)
    return {k: torch.from_numpy(v.copy()).unsqueeze(0) for k, v in sc.items()}


def test_forward_backward():
    model = LatentWorldModel(d_model=32, heads=4, temporal_layers=1,
                             latent_layers=1, num_latents=4, num_modes=3)
    output = model(batch())
    assert output["trajectories"].shape == (1, 8, 3, 80, 2)
    assert output["latents"].shape == (1, 4, 32)
    loss, _ = trajectory_loss(output, batch())
    assert torch.isfinite(loss)
    loss.backward()
    assert model.trajectory_head.weight.grad is not None
    assert torch.isfinite(model.trajectory_head.weight.grad).all()


def test_no_map_and_single_ego():
    model = LatentWorldModel(d_model=32, heads=4, temporal_layers=1,
                             latent_layers=1, num_latents=4, num_modes=2,
                             use_map=False, use_interactions=False)
    output = model(batch())
    assert torch.isfinite(output["trajectories"]).all()
