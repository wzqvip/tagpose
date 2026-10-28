"""Hardware-free integration check: synth render -> detect -> PnP -> cues -> MuJoCo.

Run it after touching anything in ``pose``, ``detect`` or ``camera``::

    python -m tagpose selftest
"""

from __future__ import annotations

import math

import numpy as np

import cv2

from . import bundle, synth
from .camera import default_intrinsics, mujoco_camera
from .detect import correspondences, detect, make_detector
from .pose import guidance, relative, solve_pose

K = default_intrinsics(640, 480, 60.0)
K[0, 0] = K[1, 1] = 500.0
K[0, 2], K[1, 2] = 320.0, 240.0
SIZE = (640, 480)
TAGS = bundle.grid_layout([0, 1, 2, 3], 0.08, 0.02, cols=2)
POSITION = np.array([0.06, -0.12, 0.45])


def run(verbose: bool = True) -> int:
    """Return the number of failed checks (0 == everything is fine)."""
    failures: "list[str]" = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    R_gt, t_gt = synth.pose_for(POSITION)

    if verbose:
        print("[1] synthetic render + pose recovery")
    frame = synth.render(TAGS, R_gt, t_gt, K, SIZE)
    corners, ids = detect(make_detector(), frame)
    check("4 tags detected", ids is not None and len(ids) == 4, f"got {len(ids) if ids is not None else 0}")
    if ids is None:
        return len(failures)

    obj, img, used = correspondences(corners, ids, TAGS)
    solved = solve_pose(obj, img, K)
    check("solve returned a pose", solved is not None)
    if solved is None:
        return len(failures)

    R_est, t_est, rms = solved
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R_gt.T @ R_est) - 1) / 2))))
    pos_mm = float(np.linalg.norm(t_est - t_gt)) * 1000
    check("reprojection rms < 1.5 px", rms < 1.5, f"rms={rms:.2f}")
    check("rotation error < 1 deg", angle < 1.0, f"{angle:.2f} deg")
    check("translation error < 3 mm", pos_mm < 3.0, f"{pos_mm:.2f} mm")

    if verbose:
        print("[2] guidance directions (current screen frame)")
    Rs, ts = R_gt, t_gt

    def cues(delta_cam):
        delta_board = Rs.T @ np.asarray(delta_cam, float)
        R2 = Rs
        t2 = -Rs @ (POSITION + delta_board)
        return guidance(*relative(Rs, ts, R2, t2))[0]

    check("camera moved right -> MOVE LEFT", any(n == "MOVE LEFT" for n, _ in cues([0.05, 0, 0])))
    check("camera moved down  -> MOVE UP", any(n == "MOVE UP" for n, _ in cues([0, 0.05, 0])))
    check("camera moved back  -> CLOSER", any(n == "CLOSER" for n, _ in cues([0, 0, -0.05])))
    check("camera moved in    -> FARTHER", any(n == "FARTHER" for n, _ in cues([0, 0, 0.05])))
    check("inside tolerance  -> no cues", cues([0.002, 0, 0]) == [])

    th = math.radians(5.0)
    yaw = np.array(
        [[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]]
    )
    R2 = yaw.T @ Rs
    pan_lines = guidance(*relative(Rs, ts, R2, -R2 @ POSITION))[0]
    check("yaw right 5 deg -> PAN LEFT", any(n == "PAN LEFT" for n, _ in pan_lines), str(pan_lines))

    if verbose:
        print("[3] MuJoCo export")
    cam = mujoco_camera(R_est, t_est, K, SIZE[1])
    w, x, y, z = cam["quat_wxyz"]
    R_q = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    R_mj = R_est @ np.diag([1.0, -1.0, -1.0])
    check("quat reproduces R_mj", np.allclose(R_q, R_mj, atol=1e-6))
    check(
        "xyaxes matches R_mj",
        np.allclose(np.array(cam["xyaxes"]).reshape(2, 3).T, R_mj[:, :2], atol=1e-6),
    )
    check("cam_pos is the camera position", np.allclose(cam["pos"], t_est, atol=1e-6))
    half_h = K[1, 1] * math.tan(math.radians(cam["fovy"]) / 2)
    check("fovy covers half the image", abs(half_h - SIZE[1] / 2) < 0.01, f"{half_h:.3f}")

    if verbose:
        print()
        print("PASS" if not failures else f"FAIL: {len(failures)} check(s) -> {failures}")
    return len(failures)
