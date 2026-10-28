"""Command line interface: ``tagpose board | guide | selftest | export``."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import cv2

from . import bundle as bundle_mod
from .camera import (
    default_intrinsics,
    load_intrinsics,
    make_camera_entry,
    mujoco_camera,
    upsert_camera,
)
from .detect import correspondences, detect, make_detector
from .hud import draw_ghost, draw_hud, draw_quad
from .pose import guidance, relative, solve_pose

__all__ = ["main"]


# --------------------------------------------------------------------------- board


def cmd_board(args) -> int:
    from .board import generate

    ids = [int(v) for v in args.ids.replace(",", " ").split()]
    summary = generate(
        args.out,
        ids=ids,
        paper=args.paper,
        tag_size=args.tag_size,
        gap=args.gap,
        cols=args.cols,
        dpi=args.dpi,
        family=args.family,
    )
    width, height = summary["board_mm"]
    print(f"tags      : {summary['tags']}")
    print(f"tag size  : {summary['tag_size_mm']:.1f} mm   gap {summary['gap_mm']:.1f} mm")
    print(f"board     : {width:.0f} x {height:.0f} mm   ({args.paper.upper()})")
    if not summary["fits"]:
        print(
            f"  [warn] does not fit {args.paper.upper()}: max tag size for this paper "
            f"is {summary['max_tag_size_mm']} mm. Try --paper a3 or --cols 1."
        )
    print(f"files     : {', '.join(summary['files'])}")
    print()
    print("print at 100% (no fit-to-page), matte paper, then measure the black square")
    print("with calipers and correct the size in the .json if it is off.")
    return 0


# --------------------------------------------------------------------------- guide


def _pose_document(R, t, rms, resolution, saved_at: str) -> dict:
    return {
        "R_cam_board": np.round(R, 6).tolist(),
        "t_cam_board": np.round(t, 6).tolist(),
        "rms_px": round(float(rms), 3),
        "resolution": [int(resolution[0]), int(resolution[1])],
        "saved_at": saved_at,
    }


def cmd_guide(args) -> int:
    if args.bundle:
        tags = bundle_mod.load(args.bundle)
    else:
        ids = [int(v) for v in args.tags.replace(",", " ").split()]
        tags = bundle_mod.grid_layout(ids, args.tag_size, args.gap, cols=args.cols)
    tag_by_id = {tag.id: tag for tag in tags}
    obj_all = bundle_mod.bundle_points(tags)

    if args.intrinsics:
        K, D = load_intrinsics(args.intrinsics)
    else:
        K = default_intrinsics(args.width, args.height, args.fov_deg)
        D = None
        print(
            f"[warn] no intrinsics given, guessing f={K[0, 0]:.1f}px from "
            f"{args.fov_deg:.0f} deg HFOV. Distance will be a few percent off; "
            "direction cues are unaffected."
        )

    source = args.video if args.video else (
        int(args.camera) if str(args.camera).isdigit() else args.camera
    )
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        sys.exit(f"cannot open video source: {source}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    capture.set(cv2.CAP_PROP_EXPOSURE, -6)

    detector = make_detector(args.family)
    saved = None
    debug = False
    print("keys: s save | c calibrate focal | x clear | d debug | q quit")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            corners, ids = detect(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            n_used, rms, distance = 0, float("nan"), float("nan")
            R = t = None
            if ids is not None:
                obj, img, used = correspondences(corners, ids, tags)
                for quad, tid in zip(corners, ids):
                    if int(tid) in tag_by_id:
                        draw_quad(frame, quad, (80, 200, 80))
                if obj is not None:
                    solved = solve_pose(obj, img, K, D)
                    if solved is not None:
                        R, t, rms = solved
                        n_used = len(used)
                        distance = float(np.linalg.norm(t))

            lines, locked = [], False
            if saved is not None and R is not None:
                R_err, t_err = relative(saved[0], saved[1], R, t)
                lines, pos_err, ang_err = guidance(R_err, t_err, args.pos_tol, args.ang_tol)
                locked = pos_err <= args.pos_tol and ang_err <= args.ang_tol
                draw_ghost(frame, obj_all, saved[0], saved[1], K, len(tags))

            if not args.headless:
                draw_hud(
                    frame,
                    n_used,
                    len(tags),
                    rms,
                    distance,
                    saved[2] if saved else None,
                    lines,
                    locked,
                    debug,
                )
                cv2.imshow("tagpose", frame)

            key = -1 if args.headless else (cv2.waitKey(1) & 0xFF)
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if R is None:
                    print("[s] no tag detected, nothing to save")
                    continue
                stamp = time.strftime("%H:%M:%S")
                saved = (R.copy(), t.copy(), stamp)
                cam = mujoco_camera(R, t, K, height)
                print(f"[s] saved  dist={distance * 100:.1f} cm  rms={rms:.2f} px")
                print(f"    MuJoCo: pos={cam['pos']}")
                print(f"            quat={cam['quat_wxyz']}  (w,x,y,z)")
                print(f"            fovy={cam['fovy']}")
                if args.save_json:
                    with open(args.save_json, "w", encoding="utf-8") as fh:
                        json.dump(
                            _pose_document(R, t, rms, (width, height), time.strftime("%Y-%m-%dT%H:%M:%S")),
                            fh,
                            indent=2,
                        )
                    print(f"    pose -> {args.save_json}")
                if args.cameras_out:
                    entry = make_camera_entry(
                        R,
                        t,
                        K,
                        D,
                        resolution=(width, height),
                        rms=rms,
                        captured_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        source=str(source),
                    )
                    upsert_camera(args.cameras_out, args.name, entry)
                    print(f"    camera '{args.name}' -> {args.cameras_out}")
            elif key == ord("x"):
                saved = None
                print("[x] cleared")
            elif key == ord("d"):
                debug = not debug
            elif key == ord("c"):
                if args.known_dist is None:
                    print("[c] needs --known-dist 0.40")
                elif ids is None:
                    print("[c] no tag detected")
                else:
                    edges = []
                    for quad, tid in zip(corners, ids):
                        spec = tag_by_id.get(int(tid))
                        if spec is None:
                            continue
                        pts = np.asarray(quad, float).reshape(4, 2)
                        side = float(
                            np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)])
                        )
                        edges.append(side * args.known_dist / spec.size)
                    if edges:
                        focal = float(np.median(edges))
                        K = np.array(
                            [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]]
                        )
                        D = None
                        print(
                            f"[c] f={focal:.1f}px from a tag at {args.known_dist * 100:.0f} cm. "
                            "Coarse one-point calibration -- use a checkerboard for real work."
                        )

            if not args.headless and cv2.getWindowProperty("tagpose", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    return 0


# --------------------------------------------------------------------------- export


def cmd_export(args) -> int:
    with open(args.pose, encoding="utf-8") as fh:
        pose = json.load(fh)
    if not args.intrinsics:
        print("export needs --intrinsics so it can compute fovy", file=sys.stderr)
        return 2
    K, D = load_intrinsics(args.intrinsics)
    entry = make_camera_entry(
        np.asarray(pose["R_cam_board"], float),
        np.asarray(pose["t_cam_board"], float),
        K,
        D,
        resolution=tuple(pose.get("resolution", (640, 480))),
        rms=pose.get("rms_px"),
        captured_at=pose.get("saved_at"),
        source=args.source,
    )
    upsert_camera(args.cameras_out, args.name, entry)
    print(f"camera '{args.name}' -> {args.cameras_out}")
    print(json.dumps(entry["mujoco"], indent=2))
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tagpose",
        description="AprilTag 6-DoF camera pose: measure a viewpoint, save it, guide back to it.",
    )
    parser.add_argument("--version", action="version", version="tagpose 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("board", help="generate a printable tag board")
    b.add_argument("--ids", default="0,1,2,3")
    b.add_argument("--paper", default="a3", choices=sorted(("a4", "a3", "letter")))
    b.add_argument("--tag-size", type=float, default=None, help="tag edge in mm")
    b.add_argument("--gap", type=float, default=None, help="gap in mm (default: size/4)")
    b.add_argument("--cols", type=int, default=2)
    b.add_argument("--dpi", type=int, default=300)
    b.add_argument("--family", default="tag36h11")
    b.add_argument("--out", default="board", help="output prefix")
    b.set_defaults(func=cmd_board)

    g = sub.add_parser("guide", help="live guidance back to a saved viewpoint")
    g.add_argument("--bundle", default=None, help="board.json from `tagpose board`")
    g.add_argument("--tags", default="0,1,2,3", help="ids to use when --bundle is absent")
    g.add_argument("--tag-size", type=float, default=0.08, help="tag edge in METRES")
    g.add_argument("--gap", type=float, default=0.02, help="gap in METRES")
    g.add_argument("--cols", type=int, default=2)
    g.add_argument("--family", default="tag36h11")
    g.add_argument("--camera", default="0", help="camera index or /dev/videoN")
    g.add_argument("--video", default=None, help="use a video file instead of a camera")
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--intrinsics", default=None, help="fx/fy/cx/cy[/dist] json or key: value text")
    g.add_argument("--fov-deg", type=float, default=60.0, help="fallback horizontal FOV")
    g.add_argument("--pos-tol", type=float, default=0.015, help="position tolerance in metres")
    g.add_argument("--ang-tol", type=float, default=2.0, help="angle tolerance in degrees")
    g.add_argument("--known-dist", type=float, default=None, help="distance for one-point focal calibration")
    g.add_argument("--save-json", default=None, help="write the saved pose here")
    g.add_argument("--cameras-out", default=None, help="upsert into this cameras.json on save")
    g.add_argument("--name", default="camera", help="camera name used with --cameras-out")
    g.add_argument("--headless", action="store_true")
    g.set_defaults(func=cmd_guide)

    s = sub.add_parser("selftest", help="hardware-free end-to-end check")
    s.set_defaults(func=lambda _args: 1 if _run_selftest() else 0)

    e = sub.add_parser("export", help="turn a saved pose into a cameras.json entry")
    e.add_argument("--pose", required=True)
    e.add_argument("--intrinsics", default=None)
    e.add_argument("--cameras-out", default="cameras.json")
    e.add_argument("--name", default="camera")
    e.add_argument("--source", default=None)
    e.set_defaults(func=cmd_export)

    return parser


def _run_selftest() -> int:
    from .selftest import run

    return run()


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
