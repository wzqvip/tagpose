import json
import math

import numpy as np
import pytest

from tagpose import synth
from tagpose.camera import (
    default_intrinsics,
    intrinsics_to_dict,
    load_cameras,
    load_intrinsics,
    make_camera_entry,
    mujoco_camera,
    quat_from_mat,
    save_intrinsics,
    upsert_camera,
)

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def test_default_intrinsics_from_fov():
    K = default_intrinsics(640, 480, 60.0)
    assert K[0, 0] == pytest.approx(320.0 / math.tan(math.radians(30.0)))
    assert K[1, 1] == pytest.approx(K[0, 0])
    assert (K[0, 2], K[1, 2]) == (320.0, 240.0)


def test_quat_roundtrip_random_rotations():
    rng = np.random.default_rng(0)
    for _ in range(200):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        w, x, y, z = quat_from_mat(Q)
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        assert np.allclose(R, Q, atol=1e-9)


def test_mujoco_camera_conventions():
    R, t = synth.pose_for([0.06, -0.12, 0.45])
    cam = mujoco_camera(R, t, K, 480)
    R_mj = R @ np.diag([1.0, -1.0, -1.0])

    assert np.allclose(cam["pos"], t, atol=1e-5)
    assert np.allclose(np.array(cam["xyaxes"]).reshape(2, 3).T, R_mj[:, :2], atol=1e-5)
    # quat -> matrix must reproduce R_mj
    w, x, y, z = cam["quat_wxyz"]
    R_q = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    assert np.allclose(R_q, R_mj, atol=1e-5)


def test_mujoco_fovy_covers_the_frame():
    cam = mujoco_camera(np.eye(3), [0, 0, 1.0], K, 480)
    half = K[1, 1] * math.tan(math.radians(cam["fovy"]) / 2)
    assert half == pytest.approx(240.0, abs=1e-3)


def test_intrinsics_roundtrip(tmp_path):
    path = tmp_path / "intr.json"
    D = np.array([-0.05, 0.01, 0.0, 0.0, 0.0])
    save_intrinsics(str(path), K, D)
    K2, D2 = load_intrinsics(str(path))
    assert np.allclose(K2, K, atol=1e-6)
    assert np.allclose(D2, D, atol=1e-8)


def test_intrinsics_plain_text(tmp_path):
    path = tmp_path / "intr.yaml"
    path.write_text("fx: 500\nfy: 500\ncx: 320\ncy: 240\n", encoding="utf-8")
    K2, D2 = load_intrinsics(str(path))
    assert D2 is None
    assert K2[0, 0] == 500 and K2[0, 2] == 320


def test_camera_entry_schema():
    R, t = synth.pose_for([0.0, 0.0, 0.5])
    entry = make_camera_entry(R, t, K, None, resolution=(640, 480), rms=0.4, captured_at="t0")
    assert entry["resolution"] == [640, 480]
    assert set(entry["intrinsics"]) >= {"fx", "fy", "cx", "cy"}
    assert "R" in entry["T_scene_cam"] and "t" in entry["T_scene_cam"]
    assert set(entry["mujoco"]) == {"pos", "quat_wxyz", "xyaxes", "fovy"}
    assert entry["rms_px"] == 0.4
    json.dumps(entry)  # must be serialisable


def test_upsert_merges_cameras(tmp_path):
    path = tmp_path / "cameras.json"
    R, t = synth.pose_for([0.0, 0.0, 0.5])
    upsert_camera(str(path), "front", make_camera_entry(R, t, K))
    upsert_camera(str(path), "side", make_camera_entry(R, t, K))
    doc = load_cameras(str(path))
    assert doc["version"] == 1
    assert doc["scene_frame"] == "board"
    assert set(doc["cameras"]) == {"front", "side"}


def test_load_cameras_missing_file(tmp_path):
    doc = load_cameras(str(tmp_path / "nope.json"))
    assert doc == {"version": 1, "scene_frame": "board", "cameras": {}}


def test_intrinsics_to_dict_has_no_none():
    data = intrinsics_to_dict(K, None)
    assert "dist" not in data
    assert all(isinstance(v, float) for v in data.values())
