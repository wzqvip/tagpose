"""On-screen HUD for the guided-return demo.

Text is English on purpose: OpenCV's Hershey fonts cannot draw CJK, and shipping
a font file for a preview is not worth it. Console output stays in the caller's
language.
"""

from __future__ import annotations

import numpy as np

import cv2

__all__ = ["COL_OK", "COL_WARN", "COL_BAD", "COL_GHOST", "draw_quad", "draw_ghost", "draw_hud"]

FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_OK = (80, 200, 80)
COL_WARN = (0, 165, 255)
COL_BAD = (60, 60, 230)
COL_GHOST = (0, 140, 255)
COL_FG = (245, 245, 245)
COL_DIM = (150, 150, 150)


def draw_quad(image, points_2d, color, thickness: int = 2) -> None:
    pts = np.asarray(points_2d, int).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], True, color, thickness, cv2.LINE_AA)


def draw_ghost(image, obj_pts, R_saved, t_saved, K, n_tags: int, color=COL_GHOST) -> None:
    """Project the board corners with the *saved* pose -- the "align to this" wireframe."""
    rvec = cv2.Rodrigues(np.asarray(R_saved, float))[0]
    projected, _ = cv2.projectPoints(
        np.asarray(obj_pts, float), rvec, np.asarray(t_saved, float).reshape(-1, 1), K, None
    )
    projected = projected.reshape(-1, 2)
    for i in range(n_tags):
        draw_quad(image, projected[i * 4 : i * 4 + 4], color)


def draw_hud(
    image,
    n_used: int,
    n_total: int,
    rms: float,
    distance: float,
    saved_label: "str | None",
    lines,
    locked: bool,
    debug: bool = False,
) -> None:
    h, w = image.shape[:2]
    pad = 10

    cv2.rectangle(image, (0, 0), (w, 46), (28, 28, 28), -1)
    cv2.putText(image, f"tags {n_used}/{n_total}", (pad, 30), FONT, 0.62, COL_FG, 1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"rms {rms:4.1f}px" if np.isfinite(rms) else "rms  --",
        (pad + 150, 30),
        FONT,
        0.62,
        COL_OK if np.isfinite(rms) and rms < 2.0 else COL_WARN,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"dist {distance:5.3f}m" if np.isfinite(distance) else "dist   --",
        (pad + 320, 30),
        FONT,
        0.62,
        COL_FG,
        1,
        cv2.LINE_AA,
    )
    if saved_label:
        cv2.putText(image, f"saved {saved_label}", (pad + 500, 30), FONT, 0.55, COL_DIM, 1, cv2.LINE_AA)
    else:
        cv2.putText(
            image, "press 's' to save viewpoint", (pad + 500, 30), FONT, 0.55, COL_WARN, 1, cv2.LINE_AA
        )

    base = h - 18 - 26 * max(0, len(lines) - 1)
    for i, (label, value) in enumerate(lines):
        y = base + i * 26
        cv2.putText(image, label, (w // 2 - 160, y), FONT, 0.85, COL_WARN, 2, cv2.LINE_AA)
        cv2.putText(image, value, (w // 2 + 110, y), FONT, 0.85, COL_WARN, 2, cv2.LINE_AA)

    if saved_label and locked:
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), COL_OK, 6)
        cv2.putText(image, "LOCKED", (w // 2 - 90, h // 2 - 10), FONT, 1.4, COL_OK, 3, cv2.LINE_AA)

    cx, cy = w // 2, h // 2
    cv2.line(image, (cx - 14, cy), (cx + 14, cy), COL_DIM, 1, cv2.LINE_AA)
    cv2.line(image, (cx, cy - 14), (cx, cy + 14), COL_DIM, 1, cv2.LINE_AA)

    if debug:
        cv2.putText(
            image,
            "s save | c calib-f | x clear | d debug | q quit",
            (pad, 70),
            FONT,
            0.5,
            COL_DIM,
            1,
            cv2.LINE_AA,
        )
