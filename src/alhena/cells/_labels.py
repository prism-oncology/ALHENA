"""Membership built from label maps."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from alhena import _progress
from alhena._constants import REFERENCE_MPP
from alhena._errors import PACKAGE_PREFIXES, GeometryError, GeometryWarning
from alhena._frames import ORIGINS_FRAME, _frame_mpp
from alhena._geometry import _checked_mpp, _positive
from alhena.cells._canvas import (
    _as_label_mask,
    _as_origins,
    _as_tile_indices,
    _frame_position,
    _tile_centers,
    _TokenRows,
)
from alhena.cells._table import _build_membership, _CellPositions, _concatenate


def membership_from_labels(
    label_maps: np.ndarray | Sequence[np.ndarray],
    tile_indices: Sequence[int] | np.ndarray,
    *,
    mpp: float | None = None,
    target_mpp: float = REFERENCE_MPP,
    tile_origins: np.ndarray | None = None,
    tile_scale: float = 1.0,
    progress: bool | None = None,
) -> pd.DataFrame:
    """Build a membership table from integer label maps, one per tile.

    Label values are cell identifiers, unique across the tiles. Given the tile origins, the table
    also records the tile centers and one position per nucleus, from the map holding most of its
    pixels before the crop, the lowest tile index among equals. The table records `mpp` and the size
    of every map, so a call refuses it beside tiles that reach the canvas otherwise.

    Args:
        label_maps: One `(H, W)` integer array per tile, zero for background, of any size.
        tile_indices: Index of the tile each label map belongs to.
        mpp: Microns per pixel of the tiles, as declared for the images. Without it, each map is
            stretched onto the whole canvas.
        target_mpp: Microns per pixel the model reads at.
        tile_origins: The x, y origin of every tile in a shared frame, `(n_tiles, 2)`, by tile
            index. None records no position.
        tile_scale: Pixels of that frame one tile pixel spans.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        A table with `MEMBERSHIP_COLUMNS`. Its attributes hold the cells that reached no token and
        the resolution, and where origins were given the tile centers, the nucleus positions and the
        microns per pixel of their frame.

    Raises:
        AlhenaValueError: If a resolution is not positive or exceeds `MAX_MICROSCOPE_MPP`.
        GeometryError: If a label map is not a 2-D array of non-negative integers.
        MembershipError: If the tile indices or the origins do not match the maps.
    """
    target_mpp = _positive(target_mpp, "target_mpp")
    if mpp is not None:
        mpp = _checked_mpp(mpp, warn=True, target_mpp=target_mpp)
    return _labels_membership(
        label_maps,
        tile_indices,
        mpp=mpp,
        target_mpp=target_mpp,
        progress=progress,
        tile_origins=tile_origins,
        tile_scale=tile_scale,
    )


def _labels_membership(
    label_maps: np.ndarray | Sequence[np.ndarray],
    tile_indices: Sequence[int] | np.ndarray,
    *,
    mpp: float | None,
    target_mpp: float,
    progress: bool | None,
    tile_origins: np.ndarray | None = None,
    tile_scale: float = 1.0,
    inside: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the `membership_from_labels` table at a checked resolution, clearing past `inside`."""
    maps = label_maps if isinstance(label_maps, Sequence) else list(label_maps)
    tiles = _as_tile_indices(tile_indices, len(maps), "label_maps")
    placement = _as_origins(tile_origins, tile_scale, tiles)
    rows = _TokenRows(mpp, target_mpp, inside)
    supplied: list[np.ndarray] = []
    grids: list[tuple[int, int, int]] = []
    views = _LabelViews()

    pairs = zip(maps, tiles, strict=True)
    with _progress.track(
        pairs, len(maps), "Mapping nuclei to image patch tokens", "tile", progress
    ) as items:
        for label_map, tile in items:
            source = _as_label_mask(label_map)
            grids.append((int(tile), int(source.shape[0]), int(source.shape[1])))
            if placement is None:
                present = np.unique(source)
                supplied.append(present[present > 0].astype(np.int64, copy=False))
            else:
                supplied.append(views.add(int(tile), source))
            rows.add(int(tile), source)

    tile_centers = cell_centers = frame_mpp = None
    if placement is not None:
        origins, scale = placement
        tile_centers = views.tile_centers(origins, scale)
        cell_centers = views.positions(origins, scale)
        frame_mpp = _frame_mpp(mpp, scale)
    return _build_membership(
        **rows.columns(np.int64),
        supplied=_concatenate(supplied, np.int64),
        source="labels",
        approximate=False,
        fitted_at=(mpp, target_mpp),
        grids=grids,
        tile_centers=tile_centers,
        cell_centers=cell_centers,
        positions_frame=None if placement is None else ORIGINS_FRAME,
        positions_mpp=frame_mpp,
    )


def _two_label_maps(tile: int, first: Sequence[int], second: Sequence[int]) -> GeometryError:
    """Say that one tile was given label maps of two sizes, each as a height and a width."""
    return GeometryError(
        f"Tile {tile} is given two label maps, of {int(first[1])} by {int(first[0])} and "
        f"{int(second[1])} by {int(second[0])} pixels. A map shares its tile's pixel grid, so the "
        "maps of one tile have one size."
    )


class _LabelViews:
    """Where the nuclei of each map lie, gathered as the maps are read, so none is read twice."""

    def __init__(self) -> None:
        self._tiles: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._counts: list[np.ndarray] = []
        self._sum_x: list[np.ndarray] = []
        self._sum_y: list[np.ndarray] = []
        # The first and last column, then row, each label covers in the map.
        self._boxes: list[np.ndarray] = []
        self._shapes: dict[int, tuple[int, int]] = {}

    def add(self, tile: int, source: np.ndarray) -> np.ndarray:
        """Add the map of one tile and return the labels it holds, ascending."""
        shape = (int(source.shape[0]), int(source.shape[1]))
        known = self._shapes.setdefault(tile, shape)
        if known != shape:
            raise _two_label_maps(tile, known, shape)
        rows, columns = np.nonzero(source)
        values = source[rows, columns]
        # Stable, so the pixels of each label keep the row order nonzero gives them.
        order = np.argsort(values, kind="stable")
        values, rows, columns = values[order], rows[order], columns[order]
        opens = np.ones(values.size, dtype=bool)
        opens[1:] = values[1:] != values[:-1]
        starts = np.flatnonzero(opens)
        counts = np.diff(np.append(starts, values.size))
        labels = values[starts].astype(np.int64, copy=False)
        boxes = np.empty((labels.size, 4), dtype=np.int64)
        boxes[:, 0] = np.minimum.reduceat(columns, starts)
        boxes[:, 1] = rows[starts]
        boxes[:, 2] = np.maximum.reduceat(columns, starts)
        boxes[:, 3] = rows[starts + counts - 1]
        self._tiles.append(np.full(labels.shape, tile, dtype=np.int64))
        self._labels.append(labels)
        self._counts.append(counts.astype(np.int64, copy=False))
        self._sum_x.append(np.add.reduceat(columns, starts).astype(np.float64))
        self._sum_y.append(np.add.reduceat(rows, starts).astype(np.float64))
        self._boxes.append(boxes)
        return labels

    def tile_centers(self, origins: np.ndarray, scale: float) -> np.ndarray:
        """Return the center of every tile given a map, NaN for a tile given none."""
        centers = np.full((origins.shape[0], 2), np.nan, dtype=np.float64)
        for tile, shape in self._shapes.items():
            centers[tile] = _tile_centers(origins[tile : tile + 1], shape, scale)[0]
        return centers

    def positions(self, origins: np.ndarray, scale: float) -> _CellPositions:
        """Return each nucleus's position, from its largest view; warn of views that do not meet."""
        labels = _concatenate(self._labels, np.int64)
        tiles = _concatenate(self._tiles, np.int64)
        counts = _concatenate(self._counts, np.int64)
        # Within each label, the view holding the most pixels comes first, then the lowest tile.
        order = np.lexsort((tiles, -counts, labels))
        first = np.ones(order.shape[0], dtype=bool)
        first[1:] = labels[order][1:] != labels[order][:-1]
        chosen = order[first]
        x = _concatenate(self._sum_x, np.float64)[chosen] / counts[chosen]
        y = _concatenate(self._sum_y, np.float64)[chosen] / counts[chosen]
        placed = np.column_stack(
            [
                _frame_position(origins[tiles[chosen], 0], x, scale),
                _frame_position(origins[tiles[chosen], 1], y, scale),
            ]
        )
        if not first.all():
            # The view of its label holding most of it, beside each view.
            main = order[np.maximum.accumulate(np.where(first, np.arange(order.shape[0]), 0))]
            sizes = np.zeros((max(self._shapes) + 1, 2), dtype=np.float64)
            for tile, shape in self._shapes.items():
                sizes[tile] = shape
            boxes = np.concatenate(self._boxes)
            _warn_apart(labels, tiles, boxes, sizes[tiles], order, main, origins, scale)
        return _CellPositions(labels[chosen], placed)


def _warn_apart(
    labels: np.ndarray,
    tiles: np.ndarray,
    boxes: np.ndarray,
    sizes: np.ndarray,
    views: np.ndarray,
    main: np.ndarray,
    origins: np.ndarray,
    scale: float,
) -> None:
    """Warn of a label whose view in some map does not meet the view holding most of it."""
    # The frame pixels of a view and of its map; tile pixel c starts at x0 + scale * c.
    start = origins[tiles] + scale * boxes[:, :2] - 0.5
    stop = origins[tiles] + scale * (boxes[:, 2:] + 1.0) - 0.5
    edge = origins[tiles] - 0.5
    end = origins[tiles] + scale * sizes[:, ::-1] - 0.5
    # Views of one nucleus meet across any space between their maps, within a tile pixel.
    reach = np.maximum(np.maximum(edge[views] - end[main], edge[main] - end[views]), 0.0) + scale
    meet = (start[views] <= stop[main] + reach) & (start[main] <= stop[views] + reach)
    apart = ~meet.all(axis=1)
    if not apart.any():
        return
    named = np.unique(labels[views[apart]])
    example = int(np.flatnonzero(apart & (labels[views] == named[0]))[0])
    first, second = sorted((int(tiles[main[example]]), int(tiles[views[example]])))
    where = f"two maps of tile {first}" if first == second else f"tiles {first} and {second}"
    others = f", as do {named.size - 1} other identifiers" if named.size > 1 else ""
    warnings.warn(
        f"Identifier {int(named[0])} names nuclei in {where} whose pixels do not meet in the "
        f"frame of the tile origins{others}: each identifier is read as one cell, from one tile. "
        "If instance numbers restart in each tile, make them unique first: io.read_label_maps "
        "renumbers them.",
        GeometryWarning,
        skip_file_prefixes=PACKAGE_PREFIXES,
    )
