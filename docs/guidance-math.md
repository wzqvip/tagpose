# Guidance math — the whole algorithm, for porting

Everything the guided-return UI does is below. It is ~40 lines of numpy
(`src/tagpose/pose.py`) and ports to C# / Kotlin / TypeScript verbatim.

## 1. Notation

A camera pose is a rotation and translation, *camera-from-board*:

```
p_cam = R · p_board + t
```

* `p_board` — a point in the tag-board frame (origin at the board centre).
* `p_cam` — the same point in camera coordinates: **x right, y down, z forward**.
* `t` is the board origin seen from the camera; `‖t‖` is the distance to the board centre.

The camera's position *in the board frame* is `C = -Rᵀ·t`.

## 2. Error between the saved and the current pose

Given `(R_s, t_s)` saved and `(R_n, t_n)` now, the current camera expressed in the saved
camera's frame is

```
R_err = R_s · R_nᵀ
t_err = t_s - R_err · t_n
```

so that `p_saved = R_err · p_now + t_err`.

Sanity check: at `p_now = 0` (the camera centre) we get `p_saved = t_err`, i.e. `t_err` is
"where the camera is now, in the saved camera's coordinates".

Scalar errors:

```
position error  = ‖t_err‖
rotation error  = acos((trace(R_err) - 1) / 2)          [radians]
```

## 3. Translation cues

We want the *correction*: the vector the camera must move by, expressed in the **current
camera frame** (so that "left" means left on the screen the user is looking at).

The correction in the saved frame is `-t_err`; converting a displacement from saved to
current uses `R_errᵀ`:

```
c = -R_errᵀ · t_err
```

Interpretation, with the OpenCV camera axes (x right, y down, z forward):

| component | condition | cue |
|---|---|---|
| `c.x` | `> 0` | `MOVE RIGHT` |
| `c.x` | `< 0` | `MOVE LEFT` |
| `c.y` | `> 0` | `MOVE DOWN` (y points down) |
| `c.y` | `< 0` | `MOVE UP` |
| `c.z` | `> 0` | `CLOSER` (move forward, towards the scene) |
| `c.z` | `< 0` | `FARTHER` |

Show an axis only when `|c[axis]| > pos_tol`. Print the magnitude in centimetres.

> Worked check. Saved camera at `C`, looking at the board; move the camera 5 cm along its
> own +x. Then `t_err = R_s·(5 cm, 0, 0)` and `c = -(5 cm, 0, 0)`, so the UI says
> `MOVE LEFT 5.0 cm`. Correct: it drifted right, it must come back left.

## 4. Rotation cues

Take the residual forward axis and right axis in the saved frame:

```
fwd   = R_err · (0, 0, 1)
right = R_err · (1, 0, 0)

pan  = atan2(fwd.x,  fwd.z)      # > 0  -> current view aims right of target -> PAN LEFT
tilt = atan2(fwd.y,  fwd.z)      # > 0  -> aims downward                  -> TILT UP
roll = atan2(right.y, right.x)   # > 0  -> camera rolled clockwise        -> ROTATE CCW
```

Each is in degrees; show it only when `|angle| > ang_tol`.

## 5. Lock condition

```
locked = ‖t_err‖ <= pos_tol  and  rotation_error <= ang_tol
```

Defaults: `pos_tol = 0.015 m`, `ang_tol = 2°`. On `locked`, draw the green border and
`LOCKED`, and suppress the cue list.

Use hysteresis in a real app (e.g. enter at 1.5 cm, leave at 2.5 cm) or the badge will
flicker.

## 6. The ghost wireframe

Project the board's 3D tag corners with the **saved** pose and the camera intrinsics:

```
uv = project(K, R_s, t_s, corner_board)
```

Draw the quad edges in orange on the live image. When the real tag corners sit on the
ghost, the camera is back at the saved viewpoint. This is more useful than the arrows for
fine alignment — people are good at overlaying shapes and bad at estimating centimetres.

## 7. Order of operations for the user

**Fix rotation first, then translation.** At short range a small pitch error looks like a
large vertical translation error, so chasing the arrows while tilted sends people in
circles.

## 8. Things that go wrong

| symptom | cause |
|---|---|
| `MOVE LEFT` when it should be `MOVE RIGHT` | you used `p_saved = R_err·p_now - t_err` (sign of `t_err`) |
| `CLOSER` / `FARTHER` swapped | you used the saved frame instead of the current frame for `c`, or your camera z points backward |
| pose flips or mirrors | tag corner order mismatched to the 3D object points, or `solvePnP` fed distorted points |
| distance is off by a constant factor | wrong tag edge length (printing scale) |
| angle jitters ±2° when nearly frontal | planar pose ambiguity — enlarge the bundle or mount one tag on a different plane |
