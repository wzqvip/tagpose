"""Camera intrinsics, the ``cameras.json`` contract, and the MuJoCo export.

``cameras.json`` is plain JSON (hence YAML-loadable) with a versioned shape::

    {
      "version": 1,
      "scene_frame": "board",          # what T_scene_cam is relative to
      "cameras": {
        "front": {
          "resolution": [640, 480],
          "intrinsics": {"fx": .., "fy": .., "cx": .., "cy": .., "dist": [..]},
          "T_scene_cam": {"R": [[..]], "t": [..]},   # p_scene = R p_cam + t
          "rms_px": 0.8,
          "mujoco": {"pos": [..], "quat_wxyz": [..], "fovy": 65.2}
        }
      }
    }

Consumers (e.g. a MuJoCo scene builder) should read only ``T_scene_cam`` and
``intrinsics``; ``mujoco`` is a convenience mirror.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

__all__ = [
    "load_intrinsics",
    "save_intrinsics",
    "default_intrinsics",
    "intrinsics_to_dict",
    "quat_from_mat",
    "mujoco_camera",
    "load_cameras",
    "upsert_camera",
]


# --------------------------------------------------------------------------- intrinsics


def _parse_simple_yaml(text: str) -> dict:
    """Tiny ``key: value`` reader so we do not need PyYAML for one file."""
    out = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            out[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip()] = value
    return out


def load_intrinsics(path: str):
    """Read ``fx, fy, cx, cy [, dist]`` from JSON or ``key: value`` text.

    Returns ``(K, D)`` with ``D=None`` when no distortion is given.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    data = json.loads(text) if path.lower().endswith(".json") else _parse_simple_yaml(text)

    fx = float(data["fx"])
    fy = float(data.get("fy", fx))
    cx = float(data["cx"])
    cy = float(data["cy"])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    dist = data.get("dist", data.get("d"))
    if dist is None:
        return K, None
    D = np.asarray(dist, float).reshape(-1)
    return K, (D if D.size and np.any(D) else None)


def save_intrinsics(path: str, K, D=None) -> None:
    K = np.asarray(K, float)
    payload = {
        "fx": round(float(K[0, 0]), 6),
        "fy": round(float(K[1, 1]), 6),
        "cx": round(float(K[0, 2]), 6),
        "cy": round(float(K[1, 2]), 6),
    }
    if D is not None:
        payload["dist"] = [round(float(v), 8) for v in np.asarray(D).reshape(-1)]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def intrinsics_to_dict(K, D=None) -> dict:
    K = np.asarray(K, float)
    out = {
        "fx": round(float(K[0, 0]), 6),
        "fy": round(float(K[1, 1]), 6),
        "cx": round(float(K[0, 2]), 6),
        "cy": round(float(K[1, 2]), 6),
    }
    if D is not None:
        out["dist"] = [round(float(v), 8) for v in np.asarray(D).reshape(-1)]
    return out


def default_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Guess K from a horizontal field of view. Fine for direction cues only."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- MuJoCo


def quat_from_mat(R) -> np.ndarray:
    """Rotation matrix -> ``(w, x, y, z)`` unit quaternion (MuJoCo order)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def mujoco_camera(R, t, K, image_height: int) -> dict:
    """OpenCV camera pose -> MuJoCo ``<camera>`` fields.

    OpenCV cameras look along ``+z`` with ``+y`` down; MuJoCo cameras look along
    ``-z`` with ``+y`` up. Hence ``R_mj = R_cv @ diag(1, -1, -1)``.

    ``fovy`` is derived from ``fy`` and the image height, so the rendered view
    matches the pinhole model -- but note MuJoCo always puts the principal point
    at the image centre. If your real ``cx, cy`` are offset (after cropping, for
    example) mirror the crop on the rendered image instead of fudging ``fovy``.
    """
    R_cv = np.asarray(R, float)
    t_cv = np.asarray(t, float).reshape(3)
    R_mj = R_cv @ np.diag([1.0, -1.0, -1.0])
    fovy = math.degrees(2 * math.atan(image_height / (2.0 * float(K[1, 1]))))
    return {
        "pos": [round(float(v), 6) for v in t_cv],
        "quat_wxyz": [round(float(v), 6) for v in quat_from_mat(R_mj)],
        "xyaxes": [round(float(v), 6) for v in np.concatenate([R_mj[:, 0], R_mj[:, 1]])],
        "fovy": round(fovy, 6),
    }


# --------------------------------------------------------------------------- cameras.json


def load_cameras(path: str) -> dict:
    if not os.path.exists(path):
        return {"version": 1, "scene_frame": "board", "cameras": {}}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def upsert_camera(path: str, name: str, entry: dict, scene_frame: str = "board") -> dict:
    """Insert/replace one camera in ``cameras.json`` and write it back."""
    doc = load_cameras(path)
    doc.setdefault("version", 1)
    doc.setdefault("scene_frame", scene_frame)
    doc.setdefault("cameras", {})[name] = entry
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, path)
    return doc


def make_camera_entry(
    R,
    t,
    K,
    D=None,
    resolution=(640, 480),
    rms=None,
    captured_at: "str | None" = None,
    source: "str | None" = None,
) -> dict:
    """Build the ``cameras.json`` record for one camera."""
    width, height = int(resolution[0]), int(resolution[1])
    entry = {
        "resolution": [width, height],
        "intrinsics": intrinsics_to_dict(K, D),
        "T_scene_cam": {
            "R": np.round(np.asarray(R, float), 6).tolist(),
            "t": np.round(np.asarray(t, float).reshape(3), 6).tolist(),
        },
        "mujoco": mujoco_camera(R, t, K, height),
    }
    if rms is not None and np.isfinite(rms):
        entry["rms_px"] = round(float(rms), 3)
    if captured_at:
        entry["captured_at"] = captured_at
    if source:
        entry["source"] = source
    return entry
