# `cameras.json` — the contract

The only thing a consumer should depend on. Everything else is an implementation detail.

```json
{
  "version": 1,
  "scene_frame": "board",
  "cameras": {
    "<name>": {
      "resolution": [640, 480],
      "intrinsics": {
        "fx": 505.3, "fy": 505.0, "cx": 319.6, "cy": 241.2,
        "dist": [-0.052, 0.012, 0.0, 0.0, 0.0]
      },
      "T_scene_cam": {
        "R": [[0.99, 0.03, -0.13], [...]],
        "t": [0.06, -0.12, 0.45]
      },
      "rms_px": 0.31,
      "captured_at": "2026-10-28T14:03:22",
      "source": "/dev/video4",
      "mujoco": {"pos": [...], "quat_wxyz": [...], "xyaxes": [...], "fovy": 65.24}
    }
  }
}
```

It is plain JSON, on purpose: it is also valid YAML, it needs no parser dependency, and it
cannot silently change meaning through indentation.

## Fields

| field | required | meaning |
|---|---|---|
| `version` | yes | schema version, currently `1` |
| `scene_frame` | yes | what `T_scene_cam` is expressed in. `"board"` means the tag board frame |
| `cameras.<name>.resolution` | yes | `[width, height]` in pixels, **of the image the pose belongs to** |
| `cameras.<name>.intrinsics` | yes | pinhole `fx, fy, cx, cy` at that resolution, `dist` optional (OpenCV order `k1 k2 p1 p2 k3…`) |
| `cameras.<name>.T_scene_cam` | yes | `p_scene = R @ p_cam + t`, OpenCV camera convention |
| `cameras.<name>.rms_px` | no | reprojection error at capture time; a health indicator, not a precision guarantee |
| `cameras.<name>.captured_at` | no | ISO-8601 local timestamp |
| `cameras.<name>.source` | no | what produced the pose (device path, video file, phone) |
| `cameras.<name>.mujoco` | no | precomputed MuJoCo fields; a convenience mirror of `T_scene_cam` + `intrinsics` |

## Conventions

**Camera frame (OpenCV):** x right, y **down**, z forward along the optical axis.

`T_scene_cam` is *camera-from-scene-inverse*: it maps a point in camera coordinates to
scene coordinates.

**MuJoCo camera frame:** x right, y **up**, z **backward** (a camera looks down `-z`).
Hence the conversion done for you in the `mujoco` block:

```
R_mj = R @ diag(1, -1, -1)
pos  = t
quat = quat(R_mj)                 # (w, x, y, z)
fovy = 2 * atan(H / (2 * fy))
```

**Resolutions and crops.** `intrinsics` must describe the exact image the pose was solved
on. If your pipeline crops and rescales before feeding a policy, do **not** bake that into
`fovy` — MuJoCo always puts the principal point at the image centre, so an off-centre crop
cannot be represented. Render the full frame from the raw intrinsics, then apply the same
crop to the rendered image. The two images then line up by construction.

## Consumer sketch

```python
import json, mujoco

doc = json.load(open("cameras.json"))
cam = doc["cameras"]["front"]

cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
model.cam_pos[cid] = cam["mujoco"]["pos"]
model.cam_quat[cid] = cam["mujoco"]["quat_wxyz"]
model.cam_fovy[cid] = cam["mujoco"]["fovy"]
```

Reading `T_scene_cam` and converting yourself is equally fine — the `mujoco` block exists so
that the common case has no chance of a convention mistake.
