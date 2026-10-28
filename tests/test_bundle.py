import numpy as np
import pytest

from tagpose import bundle
from tagpose.bundle import TagSpec, grid_layout, row_layout, tag_corners


def test_grid_layout_centres_on_origin():
    tags = grid_layout([0, 1, 2, 3], 0.08, 0.02, cols=2)
    pts = bundle.bundle_points(tags)
    assert pts.shape == (16, 3)
    # the board frame origin sits at the centre of the bundle
    assert np.allclose(pts.mean(axis=0), 0.0, atol=1e-12)


def test_grid_layout_pitch_and_first_row_on_top():
    tags = grid_layout([0, 1, 2, 3], 0.08, 0.02, cols=2)
    centres = {tag.id: (tag.t[0], tag.t[1]) for tag in tags}
    assert centres[0] == pytest.approx((-0.05, 0.05))
    assert centres[3] == pytest.approx((0.05, -0.05))
    # pitch is 0.10, so the two columns are 0.10 apart
    assert centres[1][0] - centres[0][0] == pytest.approx(0.10)


def test_row_layout_matches_ids():
    tags = row_layout([7, 8, 9], 0.05, 0.0125)
    assert [t.id for t in tags] == [7, 8, 9]
    assert all(t.t[1] == 0.0 for t in tags)


def test_tag_corners_span_and_order():
    tag = TagSpec(id=0, size=0.08, R=np.eye(3).ravel().tolist(), t=[0.0, 0.0, 0.0])
    c = tag_corners(tag)
    assert c[:, 0].max() - c[:, 0].min() == pytest.approx(0.08)
    assert c[:, 1].max() - c[:, 1].min() == pytest.approx(0.08)
    assert np.allclose(c[:, 2], 0.0)
    # top-left, top-right, bottom-right, bottom-left (y is up in the board frame)
    assert c[0, 0] < 0 and c[0, 1] > 0
    assert c[1, 0] > 0 and c[1, 1] > 0
    assert c[2, 0] > 0 and c[2, 1] < 0
    assert c[3, 0] < 0 and c[3, 1] < 0


def test_tag_corners_respect_tag_transform():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg about z
    tag = TagSpec(id=1, size=0.04, R=R.ravel().tolist(), t=[0.1, 0.2, 0.3])
    c = tag_corners(tag)
    assert np.allclose(c.mean(axis=0), [0.1, 0.2, 0.3], atol=1e-12)


def test_rejects_empty_and_duplicate_ids():
    with pytest.raises(ValueError):
        grid_layout([], 0.08, 0.02)
    with pytest.raises(ValueError):
        grid_layout([1, 1], 0.08, 0.02)


def test_save_load_roundtrip(tmp_path):
    tags = grid_layout([0, 1, 2, 3], 0.08, 0.02, cols=2)
    path = tmp_path / "board.json"
    bundle.save(str(path), tags, meta={"family": "tag36h11"})
    again = bundle.load(str(path))
    assert [t.id for t in again] == [t.id for t in tags]
    assert all(a.size == b.size for a, b in zip(tags, again))
    assert np.allclose(np.vstack([np.asarray(t.t) for t in again]),
                       np.vstack([np.asarray(t.t) for t in tags]))
