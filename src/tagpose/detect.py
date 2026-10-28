"""AprilTag detection and 2D-3D correspondences."""

from __future__ import annotations

import sys

import numpy as np

import cv2

from .bundle import TagSpec, tag_corners

__all__ = ["FAMILIES", "make_detector", "detect", "correspondences"]

FAMILIES = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
    "aruco4x4": "DICT_4X4_50",
}


def make_detector(family: str = "tag36h11", subpixel: bool = True):
    """Return a ``gray -> (corners, ids)`` callable.

    Works with both the OpenCV 4.7+ ``ArucoDetector`` object API and the older
    function API. Needs ``cv2.aruco`` (``opencv-contrib-python`` on builds that
    still split it out).
    """
    if not hasattr(cv2, "aruco"):
        sys.exit(
            "this OpenCV build has no cv2.aruco module; "
            "install opencv-contrib-python (or a newer opencv-python)"
        )
    dict_name = FAMILIES.get(family.lower(), family)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        if subpixel:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)[:2]

    params = cv2.aruco.DetectorParameters_create()
    if subpixel:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)[:2]


def detect(detector, gray) -> "tuple[list[np.ndarray], np.ndarray | None]":
    """Run the detector, returning ``(corners, ids)`` with ``ids`` as ``(N,)``."""
    corners, ids = detector(gray)
    if ids is None or len(ids) == 0:
        return [], None
    return list(corners), np.asarray(ids).reshape(-1)


def correspondences(corners, ids, tags: "list[TagSpec]"):
    """Pair detected corners with 3D board points **by tag id**.

    ``cv2.aruco`` returns markers in detection order, which has nothing to do
    with the order of your tag list. Pairing by index silently produces a
    garbage pose that still "solves" -- keep this function the only place that
    is allowed to do the pairing.

    Returns ``(obj_pts, img_pts, used_ids)`` or ``(None, None, [])`` if no
    detected tag belongs to the bundle.
    """
    by_id = {tag.id: tag for tag in tags}
    obj, img, used = [], [], []
    for quad, tid in zip(corners, ids):
        spec = by_id.get(int(tid))
        if spec is None:
            continue
        obj.append(tag_corners(spec))
        img.append(np.asarray(quad, float).reshape(-1, 2))
        used.append(int(tid))
    if not obj:
        return None, None, []
    return np.vstack(obj), np.vstack(img), used
