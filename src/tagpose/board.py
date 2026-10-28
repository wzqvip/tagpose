"""Printable board generation (SVG with exact millimetre dimensions)."""

from __future__ import annotations

import base64
import math

import numpy as np

import cv2

from . import bundle
from .bundle import TagSpec

__all__ = ["PAPERS", "PAPER_TAG", "marker_image", "board_svg", "board_png", "generate"]

# printable area in mm, with a 10 mm margin already removed
PAPERS = {
    "a4": (190.0, 277.0),
    "a3": (277.0, 400.0),
    "letter": (196.0, 259.0),
}
# sensible default tag edge per paper
PAPER_TAG = {"a4": 50.0, "a3": 80.0, "letter": 48.0}

_MARGIN_MM = 15.0
_STRIP_MM = 22.0


def _dictionary(family: str = "tag36h11"):
    from .detect import FAMILIES

    name = FAMILIES.get(family.lower(), family)
    if not hasattr(cv2.aruco, "getPredefinedDictionary"):
        raise RuntimeError("cv2.aruco is unavailable; install opencv-contrib-python")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def marker_image(tag_id: int, side_px: int = 600, family: str = "tag36h11") -> np.ndarray:
    """Grayscale image of one marker, black border included, no white quiet zone."""
    dictionary = _dictionary(family)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, int(tag_id), int(side_px))
    return cv2.aruco.drawMarker(dictionary, int(tag_id), int(side_px))


def _extent_mm(tags: "list[TagSpec]"):
    half_x = max(abs(t.t[0]) for t in tags) + max(t.size for t in tags) / 2.0
    half_y = max(abs(t.t[1]) for t in tags) + max(t.size for t in tags) / 2.0
    width = 2 * (half_x * 1000 + _MARGIN_MM)
    height = 2 * (half_y * 1000 + _MARGIN_MM) + _STRIP_MM
    return width, height, half_x * 1000, half_y * 1000


def board_svg(tags: "list[TagSpec]", family: str = "tag36h11") -> str:
    """SVG at true size: print at 100%, never "fit to page"."""
    width, height, half_x, half_y = _extent_mm(tags)
    cx, cy = width / 2, _MARGIN_MM + half_y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width:.2f}mm" height="{height:.2f}mm" '
        f'viewBox="0 0 {width:.2f} {height:.2f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="0.5" y="0.5" width="{width - 1:.2f}" height="{height - 1:.2f}" '
        'fill="none" stroke="#999999" stroke-width="0.3"/>',
    ]

    for tag in tags:
        side = tag.size * 1000
        x = cx + tag.t[0] * 1000 - side / 2
        y = cy - tag.t[1] * 1000 - side / 2
        png = cv2.imencode(".png", marker_image(tag.id, 600, family))[1].tobytes()
        b64 = base64.b64encode(png).decode("ascii")
        parts.append(
            f'<image x="{x:.3f}" y="{y:.3f}" width="{side:.3f}" height="{side:.3f}" '
            f'xlink:href="data:image/png;base64,{b64}" '
            f'href="data:image/png;base64,{b64}"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{y - 2.5:.2f}" font-family="monospace" font-size="4" '
            f'fill="#444444">id {tag.id} - {side:.0f} mm</text>'
        )

    bar_y = height - _STRIP_MM / 2
    bar_x = cx - 50
    parts.append(
        f'<line x1="{bar_x}" y1="{bar_y}" x2="{bar_x + 100}" y2="{bar_y}" '
        'stroke="#000000" stroke-width="0.6"/>'
    )
    for i in range(11):
        tick = 3 if i % 5 == 0 else 1.8
        parts.append(
            f'<line x1="{bar_x + i * 10}" y1="{bar_y}" x2="{bar_x + i * 10}" '
            f'y2="{bar_y - tick}" stroke="#000000" stroke-width="0.4"/>'
        )
    parts.append(
        f'<text x="{bar_x}" y="{bar_y + 5}" font-family="monospace" font-size="3.5" '
        'fill="#333333">scale bar = 100 mm - print at 100% (no fit-to-page)</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def board_png(tags: "list[TagSpec]", dpi: int = 300, family: str = "tag36h11") -> np.ndarray:
    """Raster preview of the same layout (not for measurement)."""
    px_per_mm = dpi / 25.4
    width, height, half_x, half_y = _extent_mm(tags)
    width_px = int(round((2 * half_x + 2 * _MARGIN_MM) * px_per_mm))
    height_px = int(round((2 * half_y + 2 * _MARGIN_MM) * px_per_mm))
    canvas = np.full((height_px, width_px), 255, np.uint8)
    cx, cy = width_px / 2, height_px / 2

    for tag in tags:
        side = int(round(tag.size * 1000 * px_per_mm))
        img = marker_image(tag.id, max(8, side), family)
        x = int(round(cx + tag.t[0] * 1000 * px_per_mm - side / 2))
        y = int(round(cy - tag.t[1] * 1000 * px_per_mm - side / 2))
        canvas[y : y + side, x : x + side] = img
        cv2.putText(
            canvas,
            f"id {tag.id}",
            (x, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            60,
            1,
            cv2.LINE_AA,
        )
    return canvas


def generate(
    out_prefix: str,
    ids=(0, 1, 2, 3),
    paper: str = "a3",
    tag_size: "float | None" = None,
    gap: "float | None" = None,
    cols: int = 2,
    dpi: int = 300,
    family: str = "tag36h11",
) -> dict:
    """Write ``<out>.svg``, ``<out>.png`` and ``<out>.json``; return a summary."""
    paper = paper.lower()
    tag_size_mm = float(tag_size) if tag_size else PAPER_TAG.get(paper, 80.0)
    gap_mm = float(gap) if gap is not None else tag_size_mm / 4.0

    tags = bundle.grid_layout(ids, tag_size_mm / 1000.0, gap_mm / 1000.0, cols=cols)
    width, height, _, _ = _extent_mm(tags)

    with open(f"{out_prefix}.svg", "w", encoding="utf-8") as fh:
        fh.write(board_svg(tags, family))
    cv2.imwrite(f"{out_prefix}.png", board_png(tags, dpi, family))
    bundle.save(
        f"{out_prefix}.json",
        tags,
        meta={"family": family, "tag_size_mm": tag_size_mm, "gap_mm": gap_mm},
    )

    printable_w, printable_h = PAPERS.get(paper, (190.0, 277.0))
    fits = width <= printable_w and height <= printable_h
    summary = {
        "tags": [t.id for t in tags],
        "tag_size_mm": tag_size_mm,
        "gap_mm": gap_mm,
        "board_mm": (round(width, 1), round(height, 1)),
        "paper": paper,
        "fits": fits,
        "files": [f"{out_prefix}.svg", f"{out_prefix}.png", f"{out_prefix}.json"],
    }
    if not fits:
        n = len(tags)
        ncols = min(cols, n)
        nrows = math.ceil(n / ncols)
        kx = ncols + (ncols - 1) * 0.25
        ky = nrows + (nrows - 1) * 0.25
        summary["max_tag_size_mm"] = math.floor(
            min(
                (printable_w - 2 * _MARGIN_MM) / kx,
                (printable_h - 2 * _MARGIN_MM - _STRIP_MM) / ky,
            )
        )
    return summary
