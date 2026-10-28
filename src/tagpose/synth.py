"""Synthetic board renderer -- lets you test the whole pipeline without a camera.

Used by ``tagpose selftest`` and by the test suite. Given a known pose it renders
what the board would look like, so detection + PnP can be checked against ground
truth.
"""

from __future__ import annotations

import numpy as np

import cv2

from .board import marker_image
from .bundle import TagSpec, tag_corners

__all__ = ["look_at", "pose_for", "render"]


def look_at(position) -> np.ndarray:
    """Board-from-camera rotation for a camera at ``position`` looking at the origin."""
    C = np.asarray(position, float).reshape(3)
    z = -C / np.linalg.norm(C)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def pose_for(position):
    """``(R, t)`` such that ``p_cam = R @ p_board + t`` for a camera at ``position``."""
    R_board_from_cam = look_at(position)
    R = R_board_from_cam.T
    t = -R @ np.asarray(position, float).reshape(3)
    return R, t


def render(
    tags: "list[TagSpec]",
    R,
    t,
    K,
    size=(640, 480),
    supersample: int = 4,
    family: str = "tag36h11",
) -> np.ndarray:
    """Render the board as seen from the given pose. Returns a grayscale image."""
    width, height = int(size[0]), int(size[1])
    ss = max(1, int(supersample))
    K_ss = np.asarray(K, float).copy()
    K_ss[0] *= ss
    K_ss[1] *= ss

    img = np.full((height * ss, width * ss), 255, np.uint8)
    rvec = cv2.Rodrigues(np.asarray(R, float))[0]
    t = np.asarray(t, float).reshape(3)

    for tag in tags:
        obj = tag_corners(tag)
        pts, _ = cv2.projectPoints(obj, rvec, t.reshape(-1, 1), K_ss, None)
        pts = pts.reshape(-1, 2).astype(np.float32)
        if not np.all(np.isfinite(pts)):
            continue
        if pts[:, 0].max() < 0 or pts[:, 1].max() < 0:
            continue
        if pts[:, 0].min() > width * ss or pts[:, 1].min() > height * ss:
            continue

        side = 600
        marker = marker_image(tag.id, side, family)
        src = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
        matrix = cv2.getPerspectiveTransform(src, pts)
        warped = cv2.warpPerspective(marker, matrix, (width * ss, height * ss), borderValue=255)
        mask = cv2.warpPerspective(np.full((side, side), 255, np.uint8), matrix, (width * ss, height * ss))
        img[mask > 0] = warped[mask > 0]

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
