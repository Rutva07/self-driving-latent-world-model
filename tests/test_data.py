import numpy as np

from lwm.data.dataset import rotate_scene, SceneDataset
from lwm.data.synthetic import generate_split, make_scene
from lwm.data.schema import validate_scene


def test_scene_and_splits(tmp_path):
    scene = make_scene(13, agents=8, map_points=32)
    validate_scene(scene)
    assert scene["history"].shape == (8, 11, 8)
    assert scene["future"].shape == (8, 80, 2)
    generate_split(tmp_path, 3, 9, agents=8, map_points=32)
    ds = SceneDataset(tmp_path, augment=True, validate=True)
    assert len(ds) == 3
    assert np.isfinite(ds[0]["history"].numpy()).all()


def test_rotation_respects_heading_and_distances():
    scene = make_scene(7, agents=8, map_points=32)
    rotated = rotate_scene(scene, np.pi / 2)
    original = scene["history"][0, -1]
    actual = rotated["history"][0, -1]
    np.testing.assert_allclose(actual[4:6], [1, 0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(original[2:4]),
                               np.linalg.norm(actual[2:4]), atol=1e-5)
    np.testing.assert_array_equal(scene["future_mask"], rotated["future_mask"])
