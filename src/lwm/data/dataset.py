"""Lazy NPZ scene dataset and rigid SE(2) augmentation."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from lwm.data.schema import validate_scene


def rotate_scene(scene, angle):
    out = {k: v.copy() for k, v in scene.items()}
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    h = out["history"]
    h[..., 0:2] = h[..., 0:2] @ rot.T
    h[..., 2:4] = h[..., 2:4] @ rot.T
    direction = h[..., 4:6][..., ::-1].copy() @ rot.T
    h[..., 4:6] = direction[..., ::-1]
    out["future"] = out["future"] @ rot.T
    out["map"][..., 0:2] = out["map"][..., 0:2] @ rot.T
    out["map"][..., 2:4] = out["map"][..., 2:4] @ rot.T
    return out


class SceneDataset(Dataset):
    def __init__(self, directory, augment=False, validate=False):
        self.files = sorted(Path(directory).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz scenes in {directory}")
        self.augment = augment
        self.validate = validate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index], allow_pickle=False) as z:
            scene = {key: z[key] for key in z.files}
        if self.validate:
            validate_scene(scene)
        if self.augment:
            scene = rotate_scene(scene, np.random.uniform(-np.pi, np.pi))
        return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in scene.items()}
