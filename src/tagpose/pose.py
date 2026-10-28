"""Pose solving, relative pose, and the "get back to the saved viewpoint" cues.

Conventions
-----------
All poses are camera-from-board (OpenCV): ``p_cam = R @ p_board + t``.
The camera frame is x right, **y down**, z forward.

Every function here is pure numpy + OpenCV and has no notion of a UI. The exact
math is written out in ``docs/guidance-math.md`` so a phone app can port it.
"""

from __future__ import annotations

import math

import numpy as np

import cv2

__all__ = [
    "solve_pose",
    "invert",
    "relative",
    "rotation_angle",
    "guidance",
]


def solve_pose(obj_pts, img_pts, K, D=None):
    """Solve board -> camera from 2D-3D correspondences.

    Returns ``(R, t, rms_px)``. Image points are undistorted first, so the
    resulting pose is the pose of the *ideal pinhole* camera -- which is what a
    renderer (MuJoCo, Unity, ...) can actually reproduce.

    ``None`` is returned when the solve fails.
    """
    obj_pts = np.asarray(obj_pts, float).reshape(-1, 3)
    img_pts = np.asarray(img_pts, float).reshape(-1, 2)
    if len(obj_pts) < 4:
        return None
    if D is not None:
        und = cv2.undistortPoints(img_pts.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    else:
        und = img_pts

    flags = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj_pts, und, K, None, flags=flags)
    if not ok:
        return None
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, und, K, None, rvec, tvec)

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - und) ** 2, axis=1))))
    if not np.isfinite(rms):
        return None
    return cv2.Rodrigues(rvec)[0], np.asarray(tvec, float).reshape(3), rms


def invert(R, t):
    """Inverse of ``p_a = R p_b + t``."""
    Rt = np.asarray(R, float).T
    return Rt, -(Rt @ np.asarray(t, float).reshape(3))


def relative(R_saved, t_saved, R_now, t_now):
    """Pose of the *current* camera expressed in the *saved* camera frame.

    ``p_saved = R_err @ p_now + t_err``; ``t_err`` is therefore where the camera
    is now, in the coordinates of the saved viewpoint.
    """
    R_err = np.asarray(R_saved, float) @ np.asarray(R_now, float).T
    t_err = np.asarray(t_saved, float).reshape(3) - R_err @ np.asarray(t_now, float).reshape(3)
    return R_err, t_err


def rotation_angle(R) -> float:
    """Magnitude of a rotation matrix, in degrees."""
    c = (np.trace(np.asarray(R, float)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def guidance(R_err, t_err, pos_tol: float = 0.015, ang_tol: float = 2.0):
    """Turn a pose error into human cues.

    Translation cues are expressed in the **current screen frame**, because that
    is what the person holding the camera actually sees:

    * ``MOVE RIGHT`` / ``MOVE LEFT``   -- x axis, screen horizontal
    * ``MOVE DOWN``  / ``MOVE UP``     -- y axis, screen vertical (y is down)
    * ``CLOSER`` / ``FARTHER``         -- z axis, towards / away from the scene

    Rotation cues are the residual yaw (``PAN``), pitch (``TILT``) and roll
    (``ROTATE``) of the current optical axis relative to the saved one.

    Returns ``(lines, pos_err_m, ang_err_deg)`` where ``lines`` is a list of
    ``(label, "3.4 cm")`` tuples. Empty when inside both tolerances.
    """
    R_err = np.asarray(R_err, float)
    t_err = np.asarray(t_err, float).reshape(3)

    pos_err = float(np.linalg.norm(t_err))
    ang_err = rotation_angle(R_err)

    correction = -R_err.T @ t_err  # how to move, in the current camera frame
    lines: "list[tuple[str, str]]" = []
    for axis, positive, negative in (
        (0, "MOVE RIGHT", "MOVE LEFT"),
        (1, "MOVE DOWN", "MOVE UP"),
        (2, "CLOSER", "FARTHER"),
    ):
        value = float(correction[axis])
        if abs(value) > pos_tol:
            lines.append((positive if value > 0 else negative, f"{abs(value) * 100:.1f} cm"))

    forward = R_err @ np.array([0.0, 0.0, 1.0])
    pan = math.degrees(math.atan2(forward[0], forward[2]))
    tilt = math.degrees(math.atan2(forward[1], forward[2]))
    right = R_err @ np.array([1.0, 0.0, 0.0])
    roll = math.degrees(math.atan2(right[1], right[0]))

    if abs(pan) > ang_tol:
        lines.append(("PAN LEFT" if pan > 0 else "PAN RIGHT", f"{abs(pan):.1f}\u00b0"))
    if abs(tilt) > ang_tol:
        lines.append(("TILT UP" if tilt > 0 else "TILT DOWN", f"{abs(tilt):.1f}\u00b0"))
    if abs(roll) > ang_tol:
        lines.append(("ROTATE CCW" if roll > 0 else "ROTATE CW", f"{abs(roll):.1f}\u00b0"))

    return lines, pos_err, ang_err
