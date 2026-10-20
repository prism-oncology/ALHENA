"""Membership built from centroids."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from alhena import _progress
from alhena._constants import INPUT_SIZE, REFERENCE_MPP
from alhena._errors import PACKAGE_PREFIXES, GeometryWarning, MembershipError
from alhena._frames import ORIGINS_FRAME, _frame_mpp
from alhena._geometry import _checked_mpp, _positive
from alhena.cells._canvas import (
    _POSITION_SLACK,
    _as_cell_ids,
    _as_origins,
    _as_tile_indices,
    _frame_position,
    _tile_centers,
    _TokenRows,
)
from alhena.cells._table import _build_membership, _CellPositions, _factorize_cells


def membership_from_centroids(
    centroids: np.ndarray,
    tile_indices: Sequence[int] | np.ndarray,
    radius_px: float | Sequence[float] | np.ndarray,
    *,
    cell_ids: Sequence[Any] | np.ndarray | None = None,
    tile_shape: tuple[int, int] = (INPUT_SIZE, INPUT_SIZE),
    mpp: float | None = None,
    target_mpp: float = REFERENCE_MPP,
    tile_origins: np.ndarray | None = None,
    tile_scale: float = 1.0,
    progress: bool | None = None,
) -> pd.DataFrame:
    """Build a membership table by rasterizing a disk around each nucleus centroid.

    The pooling weights follow a circle of the radius given rather than the segmented shape, so the
    table is flagged approximate and a warning is raised; where disks overlap, the later centroid
    owns the pixel. Centroids are read in the pixels of their own tile. Given the tile origins, the
    table records the tile centers and one position per cell, the mean of its rows.

    Args:
        centroids: An `(n, 2)` array of x and y in the pixels of the tile of each centroid, before
            any resampling.
        tile_indices: Index of the tile each centroid belongs to.
        radius_px: Nucleus radius in tile pixels, one value or one per centroid.
        cell_ids: Identifier of each centroid, repeated where a cell is given on several tiles.
            Defaults to positions.
        tile_shape: Height and width of the tiles the centroids were measured on.
        mpp: Microns per pixel of the tiles, as declared for the images. Without it, the disks are
            stretched onto the whole canvas.
        target_mpp: Microns per pixel the model reads at.
        tile_origins: The x, y origin of every tile in a shared frame, `(n_tiles, 2)`, by tile
            index. None records no position.
        tile_scale: Pixels of that frame one tile pixel spans.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        A table with `MEMBERSHIP_COLUMNS`, flagged approximate and recording the resolution, and
        where origins were given the tile centers, the cell positions and the microns per pixel of
        their frame.

    Raises:
        AlhenaValueError: If a resolution is not positive or exceeds `MAX_MICROSCOPE_MPP`.
        MembershipError: If the inputs disagree, a cell repeats on a tile, or a radius is invalid.
    """
    target_mpp = _positive(target_mpp, "target_mpp")
    if mpp is not None:
        mpp = _checked_mpp(mpp, warn=True, target_mpp=target_mpp)
    return _centroids_membership(
        centroids,
        tile_indices,
        radius_px,
        cell_ids=cell_ids,
        tile_shape=tile_shape,
        mpp=mpp,
        target_mpp=target_mpp,
        progress=progress,
        tile_origins=tile_origins,
        tile_scale=tile_scale,
    )


def _centroids_membership(
    centroids: np.ndarray,
    tile_indices: Sequence[int] | np.ndarray,
    radius_px: float | Sequence[float] | np.ndarray,
    *,
    mpp: float | None,
    target_mpp: float,
    progress: bool | None,
    cell_ids: Sequence[Any] | np.ndarray | None = None,
    tile_shape: tuple[int, int] = (INPUT_SIZE, INPUT_SIZE),
    tile_origins: np.ndarray | None = None,
    tile_scale: float = 1.0,
    inside: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the `membership_from_centroids` table at a resolution checked already."""
    points = np.asarray(centroids, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise MembershipError(f"Expected centroids of shape (n, 2), got {points.shape}.")
    if not np.isfinite(points).all():
        raise MembershipError(
            "Centroids must be finite; a nucleus with no position cannot be placed on a tile."
        )

    n_cells = points.shape[0]
    tiles = _as_tile_indices(tile_indices, n_cells, "centroids")
    radii = _as_radii(radius_px, n_cells)
    identifiers = _as_cell_ids(cell_ids, n_cells)
    height, width = int(tile_shape[0]), int(tile_shape[1])
    if height <= 0 or width <= 0:
        raise MembershipError(f"tile_shape must be positive, got {tile_shape}.")
    _reject_repeated_cells(tiles, identifiers)
    placement = _as_origins(tile_origins, tile_scale, tiles)
    tile_centers = cell_centers = frame_mpp = None
    if placement is not None:
        origins, scale = placement
        tile_centers = _tile_centers(origins, (height, width), scale)
        cell_centers = _centroid_positions(points, tiles, identifiers, origins, scale)
        frame_mpp = _frame_mpp(mpp, scale)

    rows = _TokenRows(mpp, target_mpp, inside)
    groups = _group_by_tile(tiles)
    n_groups = int(np.unique(tiles).size)
    with _progress.track(
        groups, n_groups, "Mapping nuclei to image patch tokens", "tile", progress
    ) as items:
        for tile, members in items:
            canvas = np.zeros((height, width), dtype=np.int32)
            for code, row in enumerate(members, start=1):
                _paint_disk(canvas, points[row, 0], points[row, 1], float(radii[row]), code)
            rows.add(int(tile), canvas, identifiers[members])

    frame = _build_membership(
        **rows.columns(identifiers.dtype),
        supplied=identifiers,
        source="centroids",
        approximate=True,
        fitted_at=(mpp, target_mpp),
        grids=[(int(tile), height, width) for tile in np.unique(tiles).tolist()],
        tile_centers=tile_centers,
        cell_centers=cell_centers,
        positions_frame=None if placement is None else ORIGINS_FRAME,
        positions_mpp=frame_mpp,
    )
    warnings.warn(
        "Nucleus shapes were approximated as circles around their centroids, so cell pooling "
        "uses approximate boundaries. If segmentation masks or polygons are available, use "
        "them for more faithful pooling.",
        GeometryWarning,
        skip_file_prefixes=PACKAGE_PREFIXES,
    )
    return frame


def _as_radii(radius_px: Any, n_cells: int) -> np.ndarray:
    """Return the radius of each centroid, refusing radii that are not finite positive numbers."""
    try:
        radii = np.asarray(radius_px, dtype=np.float64)
    except (TypeError, ValueError) as error:
        given = repr(radius_px) if isinstance(radius_px, str) else type(radius_px).__name__
        raise MembershipError(
            f"radius_px must be one number or one per centroid, in tile pixels, got {given}."
        ) from error
    if radii.ndim > 1 or (radii.ndim == 1 and radii.shape[0] != n_cells):
        raise MembershipError(
            f"radius_px has shape {radii.shape} for {n_cells} centroids; give one radius, or one "
            "per centroid."
        )
    if radii.size and not np.isfinite(radii).all():
        raise MembershipError(f"radius_px must be finite, got {radii[~np.isfinite(radii)][0]}.")
    if radii.size and radii.min() <= 0:
        raise MembershipError("radius_px must be positive; a zero radius covers no token.")
    return np.broadcast_to(radii, (n_cells,))


def _reject_repeated_cells(tiles: np.ndarray, identifiers: np.ndarray) -> None:
    keys = pd.MultiIndex.from_arrays([tiles, identifiers])
    if keys.duplicated().any():
        raise MembershipError(
            "A cell is given twice on one tile. Merge its entries before building the membership."
        )


def _group_by_tile(tiles: np.ndarray):
    """Yield each tile index once, with the input rows belonging to it in input order."""
    order = np.argsort(tiles, kind="stable")
    if order.size == 0:
        return
    sorted_tiles = tiles[order]
    boundary = np.empty(order.shape[0], dtype=bool)
    boundary[0] = True
    boundary[1:] = sorted_tiles[1:] != sorted_tiles[:-1]
    starts = np.nonzero(boundary)[0]
    for start, stop in zip(starts, np.append(starts[1:], order.shape[0]), strict=True):
        yield int(sorted_tiles[start]), order[start:stop]


def _paint_disk(canvas: np.ndarray, x: float, y: float, radius: float, value: int) -> None:
    height, width = canvas.shape
    left = max(int(np.floor(x - radius)), 0)
    right = min(int(np.ceil(x + radius)) + 1, width)
    top = max(int(np.floor(y - radius)), 0)
    bottom = min(int(np.ceil(y + radius)) + 1, height)
    if left >= right or top >= bottom:
        return
    rows = np.arange(top, bottom, dtype=np.float64)[:, None]
    columns = np.arange(left, right, dtype=np.float64)[None, :]
    inside = (columns - x) ** 2 + (rows - y) ** 2 <= radius * radius
    window = canvas[top:bottom, left:right]
    window[inside] = value


def _centroid_positions(
    points: np.ndarray,
    tiles: np.ndarray,
    identifiers: np.ndarray,
    origins: np.ndarray,
    scale: float,
) -> _CellPositions:
    """Return the position of each cell, the mean of its rows, refusing rows a tile pixel apart."""
    if points.shape[0] == 0:
        return _CellPositions(identifiers[:0], np.zeros((0, 2)))
    placed = np.column_stack(
        [
            _frame_position(origins[tiles, 0], points[:, 0], scale),
            _frame_position(origins[tiles, 1], points[:, 1], scale),
        ]
    )
    codes, values = _factorize_cells(identifiers)
    counts = np.bincount(codes, minlength=values.shape[0])
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    sorted_points = placed[order]
    # The rows of a cell are one run of this order, so any two lie less than the longest run apart.
    for gap in range(1, int(counts.max())):
        same = sorted_codes[gap:] == sorted_codes[:-gap]
        distance = np.hypot(*(sorted_points[gap:] - sorted_points[:-gap]).T)
        apart = same & (distance > scale * (1.0 + _POSITION_SLACK))
        if apart.any():
            row = int(np.flatnonzero(apart)[0])
            cell = values.tolist()[int(sorted_codes[row])]
            first, second = sorted((int(tiles[order[row]]), int(tiles[order[row + gap]])))
            raise MembershipError(
                f"Cell {cell!r} is given on tiles {first} and {second} at positions "
                f"{distance[row]:.2f} pixels apart in the frame of the tile origins, where two "
                f"views of one nucleus differ by one tile pixel at most, {scale:g} pixels of that "
                "frame. Check the identifiers, the origins and their scale."
            )
    x = np.bincount(codes, weights=placed[:, 0], minlength=values.shape[0]) / counts
    y = np.bincount(codes, weights=placed[:, 1], minlength=values.shape[0]) / counts
    return _CellPositions(values, np.column_stack([x, y]))
