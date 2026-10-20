"""Converting positions between the frame that places the tiles and the pixels of each tile."""

from __future__ import annotations

from typing import Any

import numpy as np

from alhena._errors import MembershipError

ORIGINS_FRAME: str = "tile origins"
"""Frame name recorded for positions placed by tile origins given by hand or from a table."""


def _as_tile_scale(tile_scale: Any) -> float:
    """Return the pixels of the frame of the tile origins one tile pixel spans."""
    try:
        scale = float(tile_scale)
    except (TypeError, ValueError) as error:
        raise MembershipError(f"tile_scale must be a number, got {tile_scale!r}.") from error
    if not np.isfinite(scale) or scale <= 0.0:
        raise MembershipError(f"tile_scale must be finite and positive, got {tile_scale}.")
    return scale


def _frame_mpp(mpp: float | None, scale: float) -> float | None:
    """Return the microns one pixel of the frame spans, None where the tiles' were not declared."""
    return None if mpp is None else float(mpp) / scale


def _tile_pixels(offsets: np.ndarray, scale: float) -> np.ndarray:
    """Carry offsets from a tile's origin, in pixels of its frame, into the pixels of the tile."""
    if scale == 1.0:
        return offsets
    return (offsets + 0.5) / scale - 0.5


def _window_grid(window: np.ndarray, scale: float) -> tuple[int, int]:
    """Return the width and height, in tile pixels, of a window in the frame of the origins."""
    x0, y0, x1, y1 = (float(value) for value in window)
    return int(np.rint((x1 - x0) / scale)), int(np.rint((y1 - y0) / scale))


def _window_centers(windows: np.ndarray) -> np.ndarray:
    """Return the center of each x0, y0, x1, y1 window, halfway from its first pixel to its last."""
    bounds = np.asarray(windows, dtype=np.float64).reshape(-1, 4)
    return (bounds[:, :2] + bounds[:, 2:] - 1.0) / 2.0


def _containing_tiles(bounds: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return the row of the tile region holding each position, -1 outside every region.

    A region runs from the first pixel of its bounds to its last, pixel centers at integer
    coordinates. Where regions overlap, the one whose center lies nearest wins, ties to the first.
    """
    windows = np.asarray(bounds, dtype=np.float64).reshape(-1, 4)
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    found = np.full(points.shape[0], -1, dtype=np.int64)
    nearest = np.full(points.shape[0], np.inf)
    order = np.argsort(points[:, 0], kind="stable")
    by_x = points[order, 0]
    centers = _window_centers(windows)
    for row, (x0, y0, x1, y1) in enumerate(windows.tolist()):
        start, stop = np.searchsorted(by_x, (x0 - 0.5, x1 - 0.5), side="left")
        held = order[start:stop]
        held = held[(points[held, 1] >= y0 - 0.5) & (points[held, 1] < y1 - 0.5)]
        if held.size == 0:
            continue
        distance = np.hypot(points[held, 0] - centers[row, 0], points[held, 1] - centers[row, 1])
        closer = distance < nearest[held]
        found[held[closer]] = row
        nearest[held[closer]] = distance[closer]
    return found


def _windows_inside(
    windows: np.ndarray, extent: tuple[int, int], scale: float = 1.0
) -> np.ndarray | None:
    """Return the height and width of each window's grid its image fills, or None where all are."""
    # Tile pixel c starts at frame pixel x0 + scale * c, so it shows the image before the edge.
    width, height = extent
    bounds = np.asarray(windows, dtype=np.float64).reshape(-1, 4)
    # Each window's width and height in tile pixels, as _window_grid gives them one at a time.
    grids = np.rint((bounds[:, 2:] - bounds[:, :2]) / scale).astype(np.int64)
    columns = np.clip(np.ceil((width - bounds[:, 0]) / scale), 0, grids[:, 0])
    rows = np.clip(np.ceil((height - bounds[:, 1]) / scale), 0, grids[:, 1])
    inside = np.column_stack([rows, columns]).astype(np.int64)
    if bool((inside == grids[:, ::-1]).all()):
        return None
    return inside
