"""Tag bundle: the board frame and where each tag sits in it.

Board frame
-----------
Origin at the centre of the whole board, x right, y up, z out of the board face
(towards the camera). A planar bundle therefore gives you "distance to the
centre of the tags" directly as ``||t||``.

A marker's ``size`` is the edge length of its **outer black square** (this is
what the detector returns corners for), in metres.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "TagSpec",
    "grid_layout",
    "row_layout",
    "tag_corners",
    "bundle_points",
    "save",
    "load",
]


@dataclass(frozen=True)
class TagSpec:
    """One tag: identity, physical size, and its pose in the board frame."""

    id: int
    size: float  # edge of the outer black square, metres
    R: tuple  # 3x3 row-major, tag -> board
    t: tuple  # 3, tag origin in board frame

    def matrix(self) -> "tuple[np.ndarray, np.ndarray]":
        return np.asarray(self.R, float).reshape(3, 3), np.asarray(self.t, float).reshape(3)


def _normalise_ids(ids) -> "list[int]":
    out = [int(i) for i in ids]
    if not out:
        raise ValueError("ids must not be empty")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate tag ids: {out}")
    return out


def grid_layout(ids, tag_size: float, gap: float, cols: int = 2) -> "list[TagSpec]":
    """Lay tags out in a grid; pitch is ``tag_size + gap``.

    ``gap`` is split evenly on both sides of each tag, so it doubles as the
    quiet zone. Keep ``gap >= tag_size / 8`` (one module) or detection rate
    drops near the edges.
    """
    ids = _normalise_ids(ids)
    cols = max(1, min(int(cols), len(ids)))
    rows = math.ceil(len(ids) / cols)
    pitch = tag_size + gap
    tags = []
    for k, tid in enumerate(ids):
        r, c = divmod(k, cols)
        x = (c - (cols - 1) / 2.0) * pitch
        y = -((r - (rows - 1) / 2.0) * pitch)  # first row on top
        tags.append(
            TagSpec(
                id=tid,
                size=float(tag_size),
                R=tuple(np.eye(3).ravel().tolist()),
                t=(float(x), float(y), 0.0),
            )
        )
    return tags


def row_layout(ids, tag_size: float, gap: float) -> "list[TagSpec]":
    """Single row of tags."""
    return grid_layout(ids, tag_size, gap, cols=len(_normalise_ids(ids)))


def tag_corners(tag: TagSpec) -> np.ndarray:
    """The tag's 4 corners in the board frame, in ``cv2.aruco`` corner order.

    Order is top-left, top-right, bottom-right, bottom-left for a tag facing
    the camera upright. Getting this order wrong is the single most common
    cause of a pose that looks "flipped" -- see ``tagpose.detect.correspondences``.
    """
    s = tag.size / 2.0
    local = np.array(
        [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
        float,
    )
    R, t = tag.matrix()
    return (R @ local.T).T + t


def bundle_points(tags: "list[TagSpec]") -> np.ndarray:
    """All corner points of all tags, stacked ``(4 * len(tags), 3)``."""
    return np.vstack([tag_corners(tag) for tag in tags])


def save(path: str, tags: "list[TagSpec]", meta: "dict | None" = None) -> None:
    payload = {
        "version": 1,
        "meta": meta or {},
        "tags": [asdict(tag) for tag in tags],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load(path: str) -> "list[TagSpec]":
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return [
        TagSpec(
            id=int(item["id"]),
            size=float(item["size"]),
            R=tuple(item["R"]),
            t=tuple(item["t"]),
        )
        for item in payload["tags"]
    ]
