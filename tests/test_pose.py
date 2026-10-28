import math

import numpy as np
import pytest

from tagpose import bundle, synth
from tagpose.detect import correspondences, detect, make_detector
from tagpose.pose import guidance, invert, relative, rotation_angle, solve_pose

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
SIZE = (640, 480)
TAGS = bundle.grid_layout([0, 1, 2, 3], 0.08, 0.02, cols=2)
HOME = np.array([0.06, -0.12, 0.45])


@pytest.fixture(scope="module")
def solved():
    R_gt, t_gt = synth.pose_for(HOME)
    frame = synth.render(TAGS, R_gt, t_gt, K, SIZE)
    corners, ids = detect(make_detector(), frame)
    assert ids is not None and len(ids) == 4
    obj, img, used = correspondences(corners, ids, TAGS)
    assert len(used) == 4
    R_est, t_est, rms = solve_pose(obj, img, K)
    return R_gt, t_gt, R_est, t_est, rms


def test_reprojection_is_tight(solved):
    assert solved[4] < 1.5


def test_pose_accuracy(solved):
    R_gt, t_gt, R_est, t_est, _ = solved
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R_gt.T @ R_est) - 1) / 2))))
    assert angle < 1.0
    assert np.linalg.norm(t_est - t_gt) < 0.003


def test_distance_is_the_board_centre_distance(solved):
    _, t_gt, _, t_est, _ = solved
    assert abs(np.linalg.norm(t_est) - np.linalg.norm(t_gt)) < 0.005


def test_correspondences_pair_by_id_not_order():
    # shuffled ids must still produce a correct pose
    R_gt, t_gt = synth.pose_for(HOME)
    frame = synth.render(TAGS, R_gt, t_gt, K, SIZE)
    corners, ids = detect(make_detector(), frame)
    order = np.argsort(ids)[::-1]
    corners = [corners[i] for i in order]
    ids = ids[order]
    obj, img, _ = correspondences(corners, ids, TAGS)
    R_est, t_est, rms = solve_pose(obj, img, K)
    assert rms < 1.5
    assert np.linalg.norm(t_est - t_gt) < 0.003


def test_unknown_tags_are_ignored():
    corners = [np.zeros((1, 4, 2), float)]
    ids = np.array([99])
    obj, img, used = correspondences(corners, ids, TAGS)
    assert obj is None and img is None and used == []


def test_invert_is_inverse():
    R_gt, t_gt = synth.pose_for(HOME)
    R2, t2 = invert(R_gt, t_gt)
    assert np.allclose(R_gt @ R2, np.eye(3), atol=1e-12)
    assert np.allclose(R_gt @ t2 + t_gt, 0.0, atol=1e-12)


def test_relative_identity():
    R, t = synth.pose_for(HOME)
    R_err, t_err = relative(R, t, R, t)
    assert np.allclose(R_err, np.eye(3))
    assert np.allclose(t_err, 0.0)
    assert rotation_angle(R_err) == pytest.approx(0.0, abs=1e-9)


def _cues(delta_cam):
    """Move the camera along its own axes, keeping its orientation."""
    R, t = synth.pose_for(HOME)
    delta_board = R.T @ np.asarray(delta_cam, float)
    t2 = -R @ (HOME + delta_board)
    return guidance(*relative(R, t, R, t2))[0]


@pytest.mark.parametrize(
    "delta,expected",
    [
        ([0.05, 0, 0], "MOVE LEFT"),
        ([-0.05, 0, 0], "MOVE RIGHT"),
        ([0, 0.05, 0], "MOVE UP"),
        ([0, -0.05, 0], "MOVE DOWN"),
        ([0, 0, -0.05], "CLOSER"),
        ([0, 0, 0.05], "FARTHER"),
    ],
)
def test_translation_cues(delta, expected):
    labels = [label for label, _ in _cues(delta)]
    assert expected in labels, labels


def test_no_cues_inside_tolerance():
    assert _cues([0.002, 0.0, 0.0]) == []


def test_deadband_boundary():
    lines, pos_err, ang_err = guidance(*relative(*synth.pose_for(HOME), *synth.pose_for(HOME)), 0.015, 2.0)
    assert lines == []
    assert pos_err == pytest.approx(0.0, abs=1e-12)
    assert ang_err == pytest.approx(0.0, abs=1e-9)


def test_yaw_cue_direction_and_magnitude():
    R, t = synth.pose_for(HOME)
    th = math.radians(5.0)
    yaw = np.array(
        [[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]]
    )
    R2 = yaw.T @ R
    lines, _, ang = guidance(*relative(R, t, R2, -R2 @ HOME))
    labels = {label: value for label, value in lines}
    assert "PAN LEFT" in labels
    assert float(labels["PAN LEFT"].rstrip("\u00b0")) == pytest.approx(5.0, abs=0.3)
    assert ang == pytest.approx(5.0, abs=0.3)


def test_roll_cue_is_not_mistaken_for_pan():
    R, t = synth.pose_for(HOME)
    th = math.radians(4.0)
    roll = np.array(
        [[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]]
    )
    R2 = roll.T @ R
    lines, _, _ = guidance(*relative(R, t, R2, -R2 @ HOME))
    labels = [label for label, _ in lines]
    assert any(label.startswith("ROTATE") for label in labels), labels
    assert not any(label.startswith("PAN") for label in labels), labels
