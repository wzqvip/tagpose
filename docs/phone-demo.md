# Phone demo — options and a plan

The Python CLI is the reference implementation. A phone app is a **second UI over the same
math**: input AprilTag corners → 6-DoF pose → compare with the saved pose → draw cues.

This document records what you get off the shelf, so nobody re-implements a marker tracker.

---

## 1. What is actually available

| route | AprilTag? | pose | intrinsics | persistence | effort |
|---|---|---|---|---|---|
| **Unity AR Foundation 6.5+, `ARMarkerManager`** | ✅ native, plus ArUco/QR | ✅ | ✅ from the AR session | ✅ anchors | **1–2 days** |
| **HawkEye** (Android, ARCore + AprilTag) | ✅ | ✅ 6-DoF, publishes over NetworkTables | ✅ | ✗ | hours to try |
| **apriltag-js-standalone** (WASM in browser) | ✅ | needs your own PnP | ✗ you supply it | ✗ | hours |
| **TagSLAM** (ROS) | ✅ + bundle SLAM | ✅ full trajectory | from calibration | ✅ map | days |
| **ARCore / ARKit world anchors** | ✗ (no tags) | ✅ but decimetre-level | ✅ | ✅ | hours |

Notes worth having:

- **AR Foundation's marker support is the closest thing to "just works"**: it reports a
  marker's pose, size, type and ID, and the AR session underneath already provides camera
  intrinsics and a persistent world frame. You still write the save/compare/guide UI — that
  is the 40 lines in [guidance-math.md](guidance-math.md).
- **HawkEye** is the fastest way to *look* at phone→tag pose, but its UI is aimed at FRC
  robots (NetworkTables), not at reproducing a viewpoint.
- **A plain web page** (`getUserMedia` + a WASM detector) is the lowest-friction demo: no
  store, no install, works on both platforms. The catch is intrinsics — the browser does
  not give you `f`. Two workarounds: read the sensor size and focal length from the OS, or
  do a one-point calibration (hold the tag at a measured distance, solve `f = s_px · z / s`).
  For *direction cues* a rough `f` is fine; only the reported distance is affected.
- **Hybrid is what a product would do**: AR anchor for coarse relocalisation, tag for the
  last few centimetres.

## 2. Recommended path

**M0 — done.** Python CLI: measure, save, guide, export `cameras.json`. This is what you
use for actual calibration work; it is the ground truth for any UI.

**M1 — web demo (half a day to a day).** Single HTML page, `apriltag-js-standalone` via a
CDN, `getUserMedia` preview, one-point focal calibration stored in `localStorage`, the same
`relative()` + `guidance()` from the math doc, canvas overlay for the ghost wireframe. Runs
on any phone, no install, easy to show in a meeting. This is the shortest path to
"look, it tells me to move left".

**M2 — Unity app (1–2 days).** AR Foundation + `ARMarkerManager`, one scene: detect the
bundle → save anchor + pose → on relaunch, relocalise and draw arrows. This is the version
that survives an app restart and a different lighting condition.

**M3 — hybrid.** Keep the tag as the fine reference and let ARKit/ARCore carry the coarse
world frame, so the user does not have to keep the whole board in view.

Do **not** start at M2: the bundle geometry, the tolerance tuning and the cue signs are all
cheap to iterate in Python or JS and expensive to iterate in a native build.

## 3. Porting checklist

| step | notes |
|---|---|
| board layout | reuse `board.json` verbatim — tag id, edge length, pose per tag |
| detection | phone SDK's detector; keep the **same marker family** as the printed board |
| corner order | the order must match the 3D object points; verify by overlaying |
| PnP | any solver; undistort first so the pose is for the ideal pinhole |
| relative pose | exactly §2 of the math doc |
| cues | exactly §3–§4, including the current-frame flip |
| deadband | add hysteresis before shipping, or the badge flickers |
| output | emit the same `cameras.json` so the phone and the CLI are interchangeable |

## 4. Why this repo does not ship the app yet

- The algorithm needs to be right before it gets a UI; the CLI is the place where that is
  cheap to change.
- A Unity project brings binary assets and a separate build pipeline — it belongs in its own
  repository once M1/M2 are real, not mixed into a Python tool.
- The phone is not the camera that matters for sim-to-real. The D405 is. The phone is a
  probe and a demo.

When M1 exists, link it here; when M2 exists, split it into `tagpose-app`.
