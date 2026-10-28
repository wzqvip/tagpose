"""tagpose -- AprilTag 6-DoF camera pose: measure, save, and guide back to it.

Typical use: put a printed tag bundle in the scene, stand the camera where you
want it, save, and let the tool tell you how to get back there later. The same
pose is what a simulator needs to render from the physical camera.
"""

from __future__ import annotations

from .bundle import TagSpec, bundle_points, grid_layout, load, row_layout, save, tag_corners
from .camera import (
    default_intrinsics,
    load_intrinsics,
    make_camera_entry,
    mujoco_camera,
    quat_from_mat,
    save_intrinsics,
    upsert_camera,
)
from .detect import correspondences, detect, make_detector
from .pose import guidance, invert, relative, rotation_angle, solve_pose

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "TagSpec",
    "grid_layout",
    "row_layout",
    "tag_corners",
    "bundle_points",
    "save",
    "load",
    "make_detector",
    "detect",
    "correspondences",
    "solve_pose",
    "invert",
    "relative",
    "rotation_angle",
    "guidance",
    "load_intrinsics",
    "save_intrinsics",
    "default_intrinsics",
    "quat_from_mat",
    "mujoco_camera",
    "make_camera_entry",
    "upsert_camera",
]
