"""The nuclei of one tile drawn on its pixels or its tokens, from any form the cell path reads."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, TypeAlias

import numpy as np
import pandas as pd

from alhena._constants import TOKEN_GRID
from alhena._errors import (
    PACKAGE_PREFIXES,
    AlhenaValueError,
    GeometryError,
    GeometryWarning,
    MembershipError,
)
from alhena._frames import _as_tile_scale, _window_grid
from alhena._rings import _ring_boxes
from alhena.cells._canvas import _as_cell_ids, _as_tile_indices
from alhena.cells._centroids import _as_radii, _paint_disk
from alhena.cells._inputs import _CENTROIDS, _LABELS, _POLYGONS, _read_nuclei, _tile_windows
from alhena.cells._polygons import _checked_rings, _draw_rings, _meeting
from alhena.cells._table import _columns_of, validate_membership
from alhena.io._formats import CELL_ID_STRIDE
from alhena.types import MEMBERSHIP_COLUMNS, AnnotationResult

# Red, green, blue and opacity on the unit scale, as the overlay tints a cell.
_RGBA: TypeAlias = "tuple[float, float, float, float]"

_TINT_INPUT_ADVICE: Final[str] = (
    "Pass the nuclei as an alhena.io reader returns them, a label map covering the tile pixel for "
    f"pixel, or a membership table with columns {MEMBERSHIP_COLUMNS}."
)


def _tile_nuclei(
    nuclei: Any, tile_index: int, shape: tuple[int, int]
) -> tuple[np.ndarray, list[Any] | None]:
    """Draw the nuclei of one tile on its pixels, as the cell path draws them before pooling.

    Returns the code of the nucleus each pixel shows, zero for background, and the cell each code
    names: code k names the k-th identifier, or, where None is returned, each code is itself the
    identifier, as in a label map.
    """
    kind, payload, options = _read_nuclei(nuclei)
    if kind == _LABELS:
        return _label_map_of_tile(payload, options, tile_index, shape), None
    if kind == _POLYGONS:
        return _polygons_of_tile(payload, options, tile_index, shape)
    if kind == _CENTROIDS:
        return _centroids_of_tile(payload, options, tile_index, shape)
    raise AlhenaValueError(
        "A membership table is drawn on the token grid, not on the pixels of a tile. Pass the "
        "nuclei the table was built from to draw them pixel for pixel."
    )


def _label_map_of_tile(
    maps: Sequence[np.ndarray], options: Mapping[str, Any], tile_index: int, shape: tuple[int, int]
) -> np.ndarray:
    """Return the label map of one tile, or an empty one where no map covers it."""
    indices = options.get("tile_indices")
    if indices is None:
        _check_tile_index(tile_index, len(maps))
        chosen = np.asarray(maps[tile_index])
    else:
        named = np.asarray(indices).reshape(-1)
        if named.shape[0] != len(maps):
            raise AlhenaValueError(
                f"tile_indices names {named.shape[0]} tiles for {len(maps)} label maps; one tile "
                "per map is required."
            )
        held = np.flatnonzero(named == tile_index)
        if held.size == 0:
            # A label image gives no map to a tile over no nucleus.
            return np.zeros(shape, dtype=np.int64)
        chosen = np.asarray(maps[int(held[0])])
    if chosen.shape != shape:
        raise GeometryError(
            f"The label map of tile {tile_index} is {chosen.shape[1]} by {chosen.shape[0]} and the "
            f"tile drawn is {shape[1]} by {shape[0]}. Draw the tile the map was segmented on."
        )
    return chosen


def _polygons_of_tile(
    polygons: Sequence[Any], options: Mapping[str, Any], tile_index: int, shape: tuple[int, int]
) -> tuple[np.ndarray, list[Any]]:
    """Draw the outlines meeting one tile window, in their order, as the membership draws them."""
    windows = _tile_windows(options.get("tile_bounds"))
    _check_tile_index(tile_index, windows.shape[0])
    window = windows[tile_index]
    scale = _as_tile_scale(options.get("tile_scale", 1.0))
    width, height = _window_grid(window, scale)
    if (height, width) != shape:
        raise GeometryError(
            f"Tile {tile_index} spans {width} by {height} pixels in the windows the polygons were "
            f"read against, and the tile drawn is {shape[1]} by {shape[0]}. Draw the tile of that "
            "window, read from the same tile source."
        )
    rings = _checked_rings(polygons)
    identifiers = _as_cell_ids(options.get("cell_ids"), len(rings), require_unique=True)
    rows = np.flatnonzero(_meeting(_ring_boxes(rings), window))
    return _draw_rings(rings, rows, window, scale), identifiers[rows].tolist()


def _centroids_of_tile(
    points: Any, options: Mapping[str, Any], tile_index: int, shape: tuple[int, int]
) -> tuple[np.ndarray, list[Any]]:
    """Paint the disk of every centroid of one tile, in their order, as the membership paints it."""
    centers = np.asarray(points, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise MembershipError(f"Expected centroids of shape (n, 2), got {centers.shape}.")
    for name in ("tile_indices", "radius_px"):
        if options.get(name) is None:
            raise AlhenaValueError(
                f"Centroids are drawn with their {name}, which these nuclei do not carry. Pass "
                "the mapping read_centroids returns."
            )
    n_cells = centers.shape[0]
    tiles = _as_tile_indices(options["tile_indices"], n_cells, "centroids")
    radii = _as_radii(options["radius_px"], n_cells)
    identifiers = _as_cell_ids(options.get("cell_ids"), n_cells)
    measured = options.get("tile_shape")
    if measured is not None and tuple(int(value) for value in measured) != shape:
        raise GeometryError(
            f"The centroids were measured on tiles of {int(measured[1])} by {int(measured[0])} "
            f"pixels, and the tile drawn is {shape[1]} by {shape[0]}. Draw a tile they were "
            "measured on."
        )
    rows = np.flatnonzero(tiles == tile_index)
    canvas = np.zeros(shape, dtype=np.int32)
    for code, row in enumerate(rows, start=1):
        _paint_disk(canvas, centers[row, 0], centers[row, 1], float(radii[row]), code)
    return canvas, identifiers[rows].tolist()


def _check_tile_index(tile_index: int, n_tiles: int) -> None:
    """Refuse a tile index outside the tiles the nuclei were read against."""
    if not 0 <= tile_index < n_tiles:
        covered = f"tiles 0 to {n_tiles - 1}" if n_tiles else "no tile"
        raise AlhenaValueError(
            f"Tile {tile_index} was asked for, and these nuclei cover {covered}. Pass the "
            "tile_index of the tile drawn among the tiles the nuclei were read against."
        )


def _cell_labels(
    label_map: Any, shape: tuple[int, int], result: AnnotationResult, tile_index: int | None
) -> np.ndarray:
    """Return a label map naming cells as the annotations do, numbering instances by their tile."""
    labels = _checked_label_map(label_map, shape)
    if not _numbered_by_tile(result):
        return labels
    nuclei = labels[labels > 0]
    if nuclei.size == 0 or int(nuclei.max()) >= CELL_ID_STRIDE:
        return labels
    if tile_index is None:
        warnings.warn(
            f"This label map holds identifiers below {CELL_ID_STRIDE}, which the annotations "
            "give to the nuclei of tile 0 alone, so it is drawn as the map of tile 0. Pass "
            "tile_index= to draw it as the map of another tile.",
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
        return labels
    return np.where(labels > 0, labels.astype(np.int64) + int(tile_index) * CELL_ID_STRIDE, 0)


def _numbered_by_tile(result: AnnotationResult) -> bool:
    """Return whether the result names its cells as `read_label_maps` numbers them tile by tile."""
    cells = getattr(result, "cells", None)
    if cells is None:
        return False
    identifiers = np.asarray(cells.cell_id)
    tiles = np.asarray(cells.tile_index)
    if (
        identifiers.size == 0
        or identifiers.shape != tiles.shape
        or not np.issubdtype(identifiers.dtype, np.integer)
    ):
        return False
    # Tile 0 keeps its instance numbers, so only cells of later tiles show the numbering.
    return bool((identifiers // CELL_ID_STRIDE == tiles).all() and (tiles > 0).any())


def _checked_label_map(label_map: Any, shape: tuple[int, int]) -> np.ndarray:
    """Return a label map of the tile's size, refusing one that holds no cell identifiers."""
    labels = np.asarray(label_map)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise GeometryError(
            f"Expected an (H, W) integer label map, got shape {labels.shape} of dtype "
            f"{labels.dtype}. {_TINT_INPUT_ADVICE}"
        )
    if labels.shape != shape:
        raise GeometryError(
            f"The label map is {labels.shape[1]} by {labels.shape[0]} and the tile is "
            f"{shape[1]} by {shape[0]}. A label map names the cell of every tile pixel."
        )
    if labels.size and int(labels.min()) < 0:
        raise GeometryError(
            "A label map holds cell identifiers and zero for background, so it carries no "
            "negative value."
        )
    return labels


def _pixel_tint(labels: np.ndarray, rgba: Callable[[Any], _RGBA]) -> np.ndarray:
    """Tint every nucleus over its own pixels, zero being background."""
    values, inverse = np.unique(labels, return_inverse=True)
    table = np.zeros((values.size, 4), dtype=np.float32)
    for position, value in enumerate(values.tolist()):
        if value != 0:
            table[position] = rgba(value)
    return table[inverse.reshape(labels.shape)]


def _token_tint(frame: pd.DataFrame, tile_index: int, rgba: Callable[[Any], _RGBA]) -> np.ndarray:
    """Tint the token grid of one tile, one class per token."""
    if tuple(frame.columns) != MEMBERSHIP_COLUMNS:
        raise MembershipError(
            f"A membership table has columns {MEMBERSHIP_COLUMNS}, got {tuple(frame.columns)}. "
            f"{_TINT_INPUT_ADVICE}"
        )
    # One tile's rows are a valid argument, so only the structural checks apply, not the partition.
    structural = _columns_of(frame)
    tiles = structural["tile_index"].to_numpy()
    validate_membership(structural, int(tiles.max()) + 1 if tiles.size else 0)

    rows = structural[structural["tile_index"] == int(tile_index)]
    if not len(rows):
        raise MembershipError(
            f"The membership table holds no row of tile {tile_index}, so there is nothing to "
            "draw. Pass the tile_index the nuclei were built under."
        )
    # Every cell of the tile is resolved, so the counts cover the tile, not only the tokens painted.
    for key in pd.unique(rows["cell_id"].to_numpy()).tolist():
        rgba(key)

    # Two stable sorts: by token, falling nucleus pixels, rising identifier, whatever the order.
    ordered = rows.sort_values("cell_id", kind="stable").sort_values(
        ["token_index", "nucleus_px"], ascending=[True, False], kind="stable"
    )
    owned = ordered.drop_duplicates("token_index", keep="first")

    grid = np.zeros((TOKEN_GRID, TOKEN_GRID, 4), dtype=np.float32)
    tokens = owned["token_index"].to_numpy().tolist()
    for token, key in zip(tokens, owned["cell_id"].tolist(), strict=True):
        grid[token // TOKEN_GRID, token % TOKEN_GRID] = rgba(key)
    return grid
