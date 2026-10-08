from pathlib import Path
import numpy as np
import pytest

from lwm.data.tfrecord import Example, iter_tfrecord, serialize_tfrecord, read_example
from lwm.data.waymo import convert_example


def fake_features():
    n = 3
    f = {}
    for group, steps in (("past", 10), ("current", 1), ("future", 80)):
        for key in ("x", "y", "velocity_x", "velocity_y", "bbox_yaw", "length", "width", "valid"):
            arr = np.zeros((n, steps), np.float32)
            if key == "valid":
                arr[:] = 1
            if key == "length":
                arr[:] = 4.4
            if key == "width":
                arr[:] = 1.9
            if key == "x":
                arr[:] = np.array([10, 20, 30])[:, None]
            f[f"state/{group}/{key}"] = arr.flatten().tolist()
    f["state/is_sdc"] = [1, 0, 0]
    f["state/type"] = [1, 1, 2]
    f["state/tracks_to_predict"] = [0, 0, 1]
    f["roadgraph_samples/xyz"] = [10.0, 0., 0., 20., 2., 0.]
    f["roadgraph_samples/dir"] = [1., 0., 0., 1., 0., 0.]
    f["roadgraph_samples/type"] = [1, 1]
    f["roadgraph_samples/valid"] = [1, 1]
    f["scenario/id"] = [b"test-waymo-example"]
    return f


def test_waymo_conversion_ego_and_order():
    scene, sid = convert_example(fake_features(), max_agents=3, max_map_points=8)
    assert sid == "test-waymo-example"
    np.testing.assert_allclose(scene["history"][0, -1, :2], [0, 0])
    np.testing.assert_allclose(scene["history"][1, -1, :2], [20, 0])
    assert scene["agent_type"].tolist() == [1, 2, 1]
    assert scene["map_mask"].sum() == 2


def test_tfrecord_roundtrip_crc(tmp_path):
    message = Example()
    message.features.feature["test_values"].float_list.value.extend([1., 2., 3.])
    message.features.feature["scenario/id"].bytes_list.value.append(b"sample")
    target = tmp_path / "sample.tfrecord"
    serialize_tfrecord([message], target)
    record = list(iter_tfrecord(target, verify_crc=True))
    assert len(record) == 1
    assert read_example(record[0])["test_values"] == pytest.approx([1., 2., 3.])
    broken = bytearray(target.read_bytes())
    broken[-5] ^= 1
    target.write_bytes(broken)
    with pytest.raises(ValueError, match="CRC"):
        list(iter_tfrecord(target, verify_crc=True))


def test_complete_waymo_tfrecord_conversion(tmp_path):
    from lwm.data.waymo import convert_shards
    message = Example()
    fake = fake_features()
    for key, values in fake.items():
        target = message.features.feature[key]
        if key == "scenario/id":
            target.bytes_list.value.extend(values)
        elif key in {"state/is_sdc", "state/tracks_to_predict", "roadgraph_samples/type",
                     "roadgraph_samples/valid"}:
            target.int64_list.value.extend(values)
        else:
            target.float_list.value.extend(values)
    shard = tmp_path / "mock.tfrecord"
    serialize_tfrecord([message], shard)
    result = convert_shards([shard], tmp_path / "processed", verify_crc=True)
    assert result == {"converted": 1, "skipped": 0}
    files = list((tmp_path / "processed").glob("*.npz"))
    assert len(files) == 1
    with np.load(files[0]) as record:
        assert record["history"].shape == (24, 11, 8)
        np.testing.assert_allclose(record["history"][0, -1, :2], [0, 0])
