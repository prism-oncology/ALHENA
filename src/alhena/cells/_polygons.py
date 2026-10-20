"""Membership built from polygons."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from alhena import _progress
from alhena._constants import REFERENCE_MPP
from alhena._errors import MembershipError
from alhena._frames import _as_tile_scale, _frame_mpp, _tile_pixels, _window_centers, _window_grid
from alhena._geometry import _checked_mpp, _positive
from alhena._rings import _ring_boxes, _ring_centroids, _stacked
from alhena.cells._canvas import _as_cell_ids, _TokenRows
from alhena.cells._table import _ATTR_TILE_BOUNDS, _build_membership, _CellPositions, _segment_rows

POLYGONS_FRAME: str = "polygons"
"""Name of the frame of positions read from boundaries whose windows are not a tile source's."""

# Most cells, along each axis, of the grid the nucleus boxes are filed on.
_GRID_CELLS_PER_AXIS = 1 << 20
# A box meeting more cells than this is read by every window rather than filed.
_WIDE_BOX_CELLS = 64
# A tile finer than its frame centers its first pixels up to half a frame pixel before its window.
_EDGE_REACH = 0.5
# Distance under which a pixel center counts as a vertex of the ring.
_VERTEX_TOLERANCE = 1e-12


def membership_from_polygons(
    polygons: Sequence[np.ndarray],
    tile_bounds: np.ndarray,
    *,
    cell_ids: Sequence[Any] | np.ndarray | None = None,
    mpp: float | None = None,
    target_mpp: float = REFERENCE_MPP,
    tile_scale: float = 1.0,
    progress: bool | None = None,
) -> pd.DataFrame:
    """Build a membership table by rasterizing nucleus polygons into each tile.

    Polygons and tile windows share one frame, so a nucleus in several tiles is found in each; where
    polygons overlap, the later one owns the pixel. A nucleus is placed at the area centroid of its
    boundary. The table records `mpp` and the grid of every window, so a call refuses it beside
    tiles that reach the canvas otherwise.

    Args:
        polygons: One `(m, 2)` array of x and y vertices per nucleus, in the shared frame.
        tile_bounds: An `(n_tiles, 4)` array of x0, y0, x1, y1 windows in the shared frame, upper
            bound excluded, by tile index.
        cell_ids: Identifier of each polygon. Defaults to positions.
        mpp: Microns per pixel of the tiles, as declared for the images. Without it, each window is
            stretched onto the whole canvas.
        target_mpp: Microns per pixel the model reads at.
        tile_scale: Pixels of the shared frame one tile pixel spans.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        A table with `MEMBERSHIP_COLUMNS`, recording the resolution, the tile centers and the
        nucleus centroids, and the microns per pixel of the frame where `mpp` is given.

    Raises:
        AlhenaValueError: If a resolution is not positive or exceeds `MAX_MICROSCOPE_MPP`.
        MembershipError: If a polygon has fewer than three vertices, or a tile window is empty.
    """
    target_mpp = _positive(target_mpp, "target_mpp")
    if mpp is not None:
        mpp = _checked_mpp(mpp, warn=True, target_mpp=target_mpp)
    return _polygons_membership(
        polygons,
        tile_bounds,
        cell_ids=cell_ids,
        mpp=mpp,
        target_mpp=target_mpp,
        progress=progress,
        tile_scale=tile_scale,
    )


def _polygons_membership(
    polygons: Sequence[np.ndarray],
    tile_bounds: np.ndarray,
    *,
    mpp: float | None,
    progress: bool | None,
    cell_ids: Sequence[Any] | np.ndarray | None = None,
    target_mpp: float = REFERENCE_MPP,
    tile_scale: float = 1.0,
    inside: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the `membership_from_polygons` table at a resolution checked already."""
    shapes = list(polygons)
    identifiers = _as_cell_ids(cell_ids, len(shapes), require_unique=True)
    scale = _as_tile_scale(tile_scale)
    bounds = np.asarray(tile_bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 4:
        raise MembershipError(f"Expected tile_bounds of shape (n_tiles, 4), got {bounds.shape}.")
    if bounds.shape[0] and not np.isfinite(bounds).all():
        raise MembershipError("Tile bounds must be finite.")

    vertices = _checked_rings(shapes)
    nearby = _BoxIndex(_ring_boxes(vertices), bounds)
    rows = _TokenRows(mpp, target_mpp, inside)
    grids: list[tuple[int, int, int]] = []

    n_tiles = bounds.shape[0]
    with _progress.track(
        range(n_tiles), n_tiles, "Mapping nuclei to image patch tokens", "tile", progress
    ) as items:
        for tile in items:
            width, height = _window_grid(bounds[tile], scale)
            if width <= 0 or height <= 0:
                raise MembershipError(
                    f"Tile {tile} has an empty window {tuple(bounds[tile].tolist())}; bounds are "
                    "x0, y0, x1, y1 with the upper bound excluded."
                )
            grids.append((int(tile), height, width))
            overlapping = nearby.meeting(tile)
            if overlapping.size == 0:
                continue
            canvas = _draw_rings(vertices, overlapping, bounds[tile], scale)
            rows.add(int(tile), canvas, identifiers[overlapping])

    frame = _build_membership(
        **rows.columns(identifiers.dtype),
        supplied=identifiers,
        source="polygons",
        approximate=False,
        fitted_at=(mpp, target_mpp),
        grids=grids,
        tile_centers=_window_centers(bounds),
        cell_centers=_CellPositions(identifiers, _ring_centroids(vertices)),
        positions_frame=POLYGONS_FRAME,
        positions_mpp=_frame_mpp(mpp, scale),
    )
    # The windows the nuclei were drawn on, so that a table given other tiles is reported.
    frame.attrs[_ATTR_TILE_BOUNDS] = bounds
    return frame


def _checked_rings(polygons: Sequence[Any]) -> list[np.ndarray]:
    """Return every polygon as float vertices, refusing one that is no ring of finite vertices."""
    rings: list[np.ndarray] = []
    for position, polygon in enumerate(polygons):
        try:
            points = np.asarray(polygon, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise MembershipError(f"Polygon {position} is not an array of vertices.") from error
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            raise MembershipError(
                f"Polygon {position} has shape {points.shape}; a nucleus boundary is an (m, 2) "
                "array of at least three vertices."
            )
        if not np.isfinite(points).all():
            raise MembershipError(f"Polygon {position} holds a non-finite vertex.")
        rings.append(points)
    return rings


def _draw_rings(
    rings: Sequence[np.ndarray], rows: np.ndarray, window: np.ndarray, scale: float
) -> np.ndarray:
    """Draw the rings named by rows on the pixels of one tile window, coded from one in that order.

    A pixel belongs to a ring when its center lies inside the ring or on it, the rule the masks the
    model was trained on were drawn with. A later ring owns a pixel it shares with an earlier one.
    """
    width, height = _window_grid(window, scale)
    canvas = np.zeros((height, width), dtype=np.int32)
    offset = np.asarray(window[:2], dtype=np.float64)
    local = [_tile_pixels(rings[row] - offset, scale) for row in rows]
    ring, pixel_rows, pixel_columns = _pixels_within_rings(local, (height, width))
    # Codes rise with the order of the rings, so the largest over a pixel is the latest ring's.
    np.maximum.at(canvas, (pixel_rows, pixel_columns), (ring + 1).astype(np.int32))
    return canvas


def _pixels_within_rings(
    rings: Sequence[np.ndarray], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ring, row and column of each pixel whose center lies inside a ring or on it."""
    # Crossings each side of a center (O'Rourke, chapter 7): odd inside, unequal parity on it.
    empty = np.zeros(0, dtype=np.int64)
    if not len(rings):
        return empty, empty, empty
    height, width = shape
    stacked, starts, lengths = _stacked(rings)
    x, y = stacked[:, 0], stacked[:, 1]
    ring = np.repeat(np.arange(len(rings), dtype=np.int64), lengths)
    # The edge ending at each vertex starts at the vertex before it in its ring.
    before = np.arange(-1, stacked.shape[0] - 1, dtype=np.int64)
    before[starts] = starts + lengths - 1

    # Each row of a ring's box is one run of counts, a sentinel past its end bringing it to zero.
    boxes = _ring_boxes(rings)
    top = np.maximum(np.ceil(boxes[:, 1]), 0.0)
    bottom = np.minimum(np.floor(boxes[:, 3]), height - 1.0)
    left = np.maximum(np.ceil(boxes[:, 0]), 0.0)
    right = np.minimum(np.floor(boxes[:, 2]), width - 1.0)
    n_columns = np.maximum(right - left + 1.0, 0.0).astype(np.int64)
    run = n_columns + 1
    size = np.maximum(bottom - top + 1.0, 0.0).astype(np.int64) * run
    size[n_columns == 0] = 0
    offset = np.zeros(len(rings), dtype=np.int64)
    offset[1:] = np.cumsum(size)[:-1]
    total = int(size.sum())
    if total == 0:
        return empty, empty, empty

    low, high = np.minimum(y, y[before]), np.maximum(y, y[before])
    drawn = size[ring] > 0
    # An edge with one end alone above a center crosses its right ray; alone below, its left one.
    right_rows = _edge_rows(np.ceil(low), np.ceil(high) - 1.0, top[ring], bottom[ring], drawn)
    left_rows = _edge_rows(np.floor(low) + 1.0, np.floor(high), top[ring], bottom[ring], drawn)

    plus: list[np.ndarray] = []
    minus: list[np.ndarray] = []
    for (edge, row), rightward in ((right_rows, True), (left_rows, False)):
        owner = ring[edge]
        base = offset[owner] + (row - top[owner]).astype(np.int64) * run[owner]
        switch = _switch_columns(stacked, edge, before[edge], row, left[owner], rightward)
        column = np.clip(switch - left[owner], 0, n_columns[owner]).astype(np.int64)
        if rightward:
            # The centers left of the crossing see it to their right.
            plus.append(base)
            minus.append(base + column)
        else:
            # The centers right of it see it to their left.
            plus.append(base + column)
            minus.append(base + n_columns[owner])
    counts = [
        np.cumsum(np.bincount(up, minlength=total) - np.bincount(down, minlength=total))
        for up, down in zip(plus, minus, strict=True)
    ]
    keep = ((counts[0] & 1) | (counts[1] & 1)).astype(bool)

    # A center on a vertex lies on the ring, whatever the crossings count.
    vertex_x, vertex_y = np.rint(x), np.rint(y)
    on = (
        (np.abs(x - vertex_x) < _VERTEX_TOLERANCE)
        & (np.abs(y - vertex_y) < _VERTEX_TOLERANCE)
        & drawn
        & (vertex_x >= left[ring])
        & (vertex_x <= right[ring])
        & (vertex_y >= top[ring])
        & (vertex_y <= bottom[ring])
    )
    owner = ring[on]
    keep[
        offset[owner]
        + (vertex_y[on] - top[owner]).astype(np.int64) * run[owner]
        + (vertex_x[on] - left[owner]).astype(np.int64)
    ] = True

    kept = np.flatnonzero(keep)
    owner = np.searchsorted(offset, kept, side="right") - 1
    position = kept - offset[owner]
    pixel_rows = top[owner].astype(np.int64) + position // run[owner]
    pixel_columns = left[owner].astype(np.int64) + position % run[owner]
    return owner, pixel_rows, pixel_columns


def _edge_rows(
    first: np.ndarray, last: np.ndarray, top: np.ndarray, bottom: np.ndarray, drawn: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return each pair of an edge and a row of centers it crosses, within its ring's box."""
    first = np.maximum(first, top)
    last = np.minimum(last, bottom)
    counts = np.where(drawn, np.maximum(last - first + 1.0, 0.0), 0.0).astype(np.int64)
    edge = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    # The position of each row among those of its edge.
    step = np.arange(edge.shape[0], dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    return edge, np.repeat(first, counts) + step


def _switch_columns(
    vertices: np.ndarray,
    edge: np.ndarray,
    before: np.ndarray,
    row: np.ndarray,
    left: np.ndarray,
    rightward: bool,
) -> np.ndarray:
    """Return the first column of centers that do not see an edge on the side tested."""
    # The test decides the two columns beside a crossing; the others lie a pixel or more away.
    estimate = left + _crossing(vertices, edge, before, left, row)
    near = np.floor(estimate)
    beside = _crossing(vertices, edge, before, near, row)
    after = _crossing(vertices, edge, before, near + 1.0, row)
    if rightward:
        return np.where(beside <= 0.0, near, np.where(after <= 0.0, near + 1.0, near + 2.0))
    return np.where(beside < 0.0, near, np.where(after < 0.0, near + 1.0, near + 2.0))


def _crossing(
    vertices: np.ndarray, edge: np.ndarray, before: np.ndarray, column: np.ndarray, row: np.ndarray
) -> np.ndarray:
    """Return where an edge crosses the row of a center, along the row and from the center."""
    x0 = vertices[edge, 0] - column
    y0 = vertices[edge, 1] - row
    x1 = vertices[before, 0] - column
    y1 = vertices[before, 1] - row
    return (x0 * y1 - x1 * y0) / (y1 - y0)


def _meeting(boxes: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Return which ring boxes may hold a pixel center of a window, its upper bounds excluded."""
    x0, y0, x1, y1 = window
    return (
        (boxes[:, 0] < x1)
        & (boxes[:, 2] >= x0 - _EDGE_REACH)
        & (boxes[:, 1] < y1)
        & (boxes[:, 3] >= y0 - _EDGE_REACH)
    )


class _BoxIndex:
    """Nucleus boxes filed on a grid, so each tile window reads only the boxes near it."""

    def __init__(self, boxes: np.ndarray, windows: np.ndarray) -> None:
        """File the boxes of the nuclei, bounds included, against windows, upper bound excluded."""
        self._boxes = boxes
        self._windows = windows
        self._keys = np.empty(0, dtype=np.int64)
        self._members = np.empty(0, dtype=np.int64)
        self._wide = np.empty(0, dtype=np.int64)
        self._reads = np.zeros(windows.shape[0], dtype=bool)
        if boxes.shape[0] == 0:
            return
        low = boxes[:, :2].min(axis=0)
        high = boxes[:, 2:].max(axis=0)
        # About the side of a window, at least that of a box, within the cells allowed per axis.
        sides = [np.median(np.concatenate([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]))]
        if windows.shape[0]:
            sides.append(
                np.median(
                    np.concatenate([windows[:, 2] - windows[:, 0], windows[:, 3] - windows[:, 1]])
                )
            )
        side = max(float(max(sides)), float((high - low).max()) / _GRID_CELLS_PER_AXIS)
        self._origin = low
        self._side = side if side > 0.0 else 1.0

        first_column = self._cells(boxes[:, 0], 0).astype(np.int64)
        last_column = self._cells(boxes[:, 2], 0).astype(np.int64)
        first_row = self._cells(boxes[:, 1], 1).astype(np.int64)
        last_row = self._cells(boxes[:, 3], 1).astype(np.int64)
        columns = int(last_column.max()) + 1
        rows = int(last_row.max()) + 1
        across = last_column - first_column + 1
        count = across * (last_row - first_row + 1)
        wide = count > _WIDE_BOX_CELLS
        self._wide = np.flatnonzero(wide)
        filed = np.flatnonzero(~wide)
        count = count[filed]
        owner = np.repeat(filed, count)
        # The position of each filing among those of its box, which runs across then down.
        step = np.arange(owner.shape[0], dtype=np.int64)
        step -= np.repeat(np.cumsum(count) - count, count)
        per_row = np.repeat(across[filed], count)
        column = np.repeat(first_column[filed], count) + step % per_row
        row = np.repeat(first_row[filed], count) + step // per_row
        keys = row * columns + column
        order = np.argsort(keys, kind="stable")
        self._keys = keys[order]
        self._members = owner[order]
        self._columns = columns

        # The cells each window may read a box from, clamped to the grid; past it, none.
        first_x = self._cells(windows[:, 0] - _EDGE_REACH, 0)
        last_x = self._cells(windows[:, 2], 0)
        first_y = self._cells(windows[:, 1] - _EDGE_REACH, 1)
        last_y = self._cells(windows[:, 3], 1)
        self._reads = (last_x >= 0) & (first_x < columns) & (last_y >= 0) & (first_y < rows)
        self._first_column = np.clip(first_x, 0, columns - 1).astype(np.int64)
        self._last_column = np.clip(last_x, 0, columns - 1).astype(np.int64)
        self._first_row = np.clip(first_y, 0, rows - 1).astype(np.int64)
        self._last_row = np.clip(last_y, 0, rows - 1).astype(np.int64)

    def _cells(self, values: np.ndarray, axis: int) -> np.ndarray:
        """Return the cell each coordinate falls in along one axis, never decreasing with it."""
        return np.floor((values - self._origin[axis]) / self._side)

    def meeting(self, window: int) -> np.ndarray:
        """Return the boxes that may hold a pixel center of one window, ascending."""
        found = self._wide
        if self._reads[window]:
            rows = self._columns * np.arange(
                self._first_row[window], self._last_row[window] + 1, dtype=np.int64
            )
            start = np.searchsorted(self._keys, rows + self._first_column[window], side="left")
            stop = np.searchsorted(self._keys, rows + self._last_column[window], side="right")
            filed = self._members[_segment_rows(start, stop - start)]
            found = np.concatenate([filed, self._wide]) if self._wide.size else filed
        candidates = np.unique(found)
        return candidates[_meeting(self._boxes[candidates], self._windows[window])]
