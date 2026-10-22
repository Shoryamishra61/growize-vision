#!/usr/bin/env python
"""Generate a printable ArUco reference and a physical-size check ruler.

Print at 100% / "actual size", then MEASURE the outer black square with a
ruler or caliper.  Enter that measured value in Growize Vision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from growize_vision.markers import generate_marker  # noqa: E402


def make_board(
    marker_id: int,
    dictionary: str,
    side_mm: float,
    dpi: int,
    output: Path,
) -> Path:
    marker = generate_marker(
        marker_id=marker_id,
        dictionary_name=dictionary,
        side_mm=side_mm,
        dpi=dpi,
        quiet_zone_mm=12.0,
    )
    marker_px = marker.shape[0]
    # Add a footer with a 100 mm verification line, generated at the same DPI.
    px_per_mm = dpi / 25.4
    width = max(marker_px, int(round(120.0 * px_per_mm)))
    board = np.full((marker_px + int(round(42 * px_per_mm)), width, 3), 255, np.uint8)
    mx = (width - marker_px) // 2
    board[:marker_px, mx : mx + marker_px] = marker
    footer_y = marker_px
    footer_h = board.shape[0] - footer_y
    line_y = footer_y + int(footer_h * 0.42)
    x0 = int(round(10.0 * px_per_mm))
    x1 = x0 + int(round(100.0 * px_per_mm))
    cv2.line(board, (x0, line_y), (x1, line_y), (0, 0, 0), 2)
    for tick in (0, 50, 100):
        x = x0 + int(round(tick * px_per_mm))
        cv2.line(board, (x, line_y - 10), (x, line_y + 10), (0, 0, 0), 2)
    cv2.putText(
        board,
        f"VERIFY LINE = 100 mm. Outer black square = {side_mm:.1f} mm nominal.",
        (x0, footer_y + int(footer_h * 0.82)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.38, 0.48 * px_per_mm / 11.8),
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), board)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--side-mm", type=float, default=50.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "aruco_marker_0_50mm.png")
    args = parser.parse_args()
    output = make_board(
        args.id, args.dictionary, args.side_mm, args.dpi, args.out
    )
    print(f"Wrote {output}")
    print("Print at 100% / actual size, then measure the outer black square.")
    print("Do NOT assume the nominal size is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
