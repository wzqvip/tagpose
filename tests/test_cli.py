import json

import numpy as np
import pytest

from tagpose import bundle
from tagpose.cli import main
from tagpose.selftest import run


def test_selftest_passes(capsys):
    assert run(verbose=False) == 0


def test_board_command_writes_three_files(tmp_path):
    prefix = tmp_path / "board"
    assert main(["board", "--paper", "a4", "--out", str(prefix)]) == 0

    for suffix in (".svg", ".json", ".png"):
        assert (tmp_path / f"board{suffix}").exists()

    tags = bundle.load(str(tmp_path / "board.json"))
    assert [t.id for t in tags] == [0, 1, 2, 3]
    assert tags[0].size == 0.05  # a4 default

    svg = (tmp_path / "board.svg").read_text(encoding="utf-8")
    assert "mm" in svg and "scale bar" in svg


def test_board_does_not_fit_is_reported(tmp_path, capsys):
    assert main(["board", "--paper", "a4", "--tag-size", "80", "--out", str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    assert "does not fit" in out
    assert "max tag size" in out


def test_export_command(tmp_path):
    pose = {
        "R_cam_board": np.eye(3).tolist(),
        "t_cam_board": [0.0, 0.0, 0.5],
        "rms_px": 0.3,
        "resolution": [640, 480],
        "saved_at": "2026-01-01T00:00:00",
    }
    pose_path = tmp_path / "pose.json"
    pose_path.write_text(json.dumps(pose), encoding="utf-8")

    intr_path = tmp_path / "intr.json"
    intr_path.write_text(
        json.dumps({"fx": 500, "fy": 500, "cx": 320, "cy": 240}), encoding="utf-8"
    )

    cameras = tmp_path / "cameras.json"
    code = main(
        [
            "export",
            "--pose",
            str(pose_path),
            "--intrinsics",
            str(intr_path),
            "--cameras-out",
            str(cameras),
            "--name",
            "front",
        ]
    )
    assert code == 0

    doc = json.loads(cameras.read_text(encoding="utf-8"))
    assert doc["cameras"]["front"]["mujoco"]["pos"] == [0.0, 0.0, 0.5]
    assert doc["cameras"]["front"]["mujoco"]["fovy"] == pytest.approx(51.2820, abs=1e-3)


def test_export_requires_intrinsics(tmp_path):
    pose_path = tmp_path / "pose.json"
    pose_path.write_text(
        json.dumps({"R_cam_board": np.eye(3).tolist(), "t_cam_board": [0, 0, 1]}),
        encoding="utf-8",
    )
    assert main(["export", "--pose", str(pose_path)]) == 2


def test_guide_rejects_bad_source():
    with pytest.raises(SystemExit):
        main(["guide", "--video", "definitely-not-a-file.avi", "--headless"])
