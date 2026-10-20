"""Nuclei read as centroids from a table."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alhena import _progress
from alhena._errors import PACKAGE_PREFIXES, AlhenaValueError, GeometryWarning
from alhena._frames import _tile_pixels
from alhena.io._carry import _apply, _carrier, _extent, global_cell_ids
from alhena.io._files import (
    _choice,
    _column_names,
    _coordinates,
    _identifier_columns,
    _named,
    _pick_column,
    _table,
    _text_identifiers,
)
from alhena.io._formats import CELL_ID_STRIDE, CENTROID_COLUMNS, CENTROID_FRAMES, COORDINATE_UNITS
from alhena.io._sources import TileSource, as_tile_source, tile_bounds

_VERTEX_TABLE_ADVICE: str = (
    "A table of boundary vertices is read by read_polygons, or by read_nuclei with kind='polygons'."
)

_ORIGINS_FRAME_ADVICE: str = (
    "A table holding one row per nucleus, in the frame the tile origins are written in, is read "
    "with frame='origins'."
)


def read_centroids(
    source: str | Path | pd.DataFrame,
    *,
    tiles: TileSource | np.ndarray | Sequence[np.ndarray] | None = None,
    tile_indices: Sequence[int] | np.ndarray | None = None,
    radius_px: float | None = None,
    columns: Mapping[str, str] | None = None,
    stride: int = CELL_ID_STRIDE,
    frame: str = "tile",
    units: str = "px",
    transform: np.ndarray | Sequence[Sequence[float]] | str | Path | None = None,
    progress: bool | None = None,
) -> dict[str, Any]:
    """Read nucleus centroids from a table.

    By default each row names its tile and gives its position in that tile's pixels. With
    `frame="origins"`, each row is one nucleus in the frame of the tile origins, given to every tile
    holding it, and a centroid no tile holds is left out with a warning. A centroid has no extent: a
    disk of the radius given stands in for the nucleus.

    Args:
        source: A DataFrame, or the path of a `.csv`, `.tsv`, `.txt`, `.parquet` or `.pq` file.
        tiles: The tiles the centroids were measured on: a tile source, or tiles in memory as
            `as_tile_source` takes them. Required with `frame="origins"` and with `units="um"`.
        tile_indices: Tile of each centroid, where the table has no such column; refused with
            `frame="origins"`.
        radius_px: Nucleus radius in tile pixels, where the table has no such column.
        columns: Columns holding `x`, `y`, `tile_index`, `cell_id` and `radius_px`, where named
            otherwise.
        stride: Number of identifiers in the block each tile owns, where a table read with
            `frame="tile"` names no cells.
        frame: One of `CENTROID_FRAMES`.
        units: One of `COORDINATE_UNITS`.
        transform: A 2 by 3 or 3 by 3 affine matrix carrying the coordinates into the frame of the
            tile origins, or a `.csv` or `.npy` file of one; only with `frame="origins"`.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        A mapping holding `centroids`, `tile_indices`, `radius_px`, `cell_ids`, and `tile_shape`
        where the tiles were given. With `frame="origins"`, it holds one entry per centroid and tile
        holding it, and the windows and frame of the tiles, `tile_bounds` and `tile_frame`.

    Raises:
        AlhenaValueError: If the table is unreadable, names a cell twice, or fits no tile.
        GeometryError: If tiles given in memory are not uint8 RGB.
    """
    tiles = None if tiles is None else as_tile_source(tiles)
    frame = _choice(frame, CENTROID_FRAMES, "frame")
    units = _choice(units, COORDINATE_UNITS, "units")
    chosen = _column_names(columns, CENTROID_COLUMNS, "a table of centroids")
    in_origins = frame == CENTROID_FRAMES[1]
    if in_origins:
        _check_origins_frame(tiles, tile_indices, chosen, stride)
    elif transform is not None:
        raise AlhenaValueError(
            "transform carries coordinates into the frame of the tile origins, and frame='tile' "
            "reads each row in the pixels of its tile. Pass frame='origins' for a table of one "
            "row per nucleus."
        )
    carrier = _carrier(units, transform, tiles, origins=in_origins)

    table = _table(source, id_columns=_identifier_columns(chosen, CENTROID_COLUMNS))
    named = _named(source)
    x_column = _pick_column(
        table, chosen.get("x"), CENTROID_COLUMNS["x"], "x centroid", advice=_VERTEX_TABLE_ADVICE
    )
    y_column = _pick_column(
        table, chosen.get("y"), CENTROID_COLUMNS["y"], "y centroid", advice=_VERTEX_TABLE_ADVICE
    )
    points = _coordinates(table, (x_column, y_column), named, "centroid")
    n_cells = points.shape[0]
    if n_cells == 0:
        raise AlhenaValueError(f"No centroid in {named}.")
    if carrier is not None:
        points = _apply(carrier, points)
    radii = _centroid_radii(table, chosen, radius_px, n_cells, named)
    if in_origins:
        assert tiles is not None
        identifiers = _row_identifiers(table, chosen, named)
        return _centroids_on_tiles(points, radii, identifiers, tiles, named, progress)

    if tile_indices is None:
        tile_column = _pick_column(
            table,
            chosen.get("tile_index"),
            CENTROID_COLUMNS["tile_index"],
            "tile index",
            advice=_ORIGINS_FRAME_ADVICE,
        )
        indices = _whole_indices(table[tile_column], "row", named, tile_column)
    else:
        indices = _whole_indices(np.asarray(tile_indices).reshape(-1), "position", "tile_indices")
        if indices.shape[0] != n_cells:
            raise AlhenaValueError(
                f"{indices.shape[0]} tile indices were given for {n_cells} centroids; one index "
                "per centroid is required."
            )
    if int(indices.min()) < 0:
        raise AlhenaValueError("tile_index must not be negative.")
    if tiles is not None and int(indices.max()) >= len(tiles):
        raise AlhenaValueError(
            f"tile_index reaches {int(indices.max())}, outside the {len(tiles)} tiles given."
        )

    identifiers = _centroid_identifiers(table, chosen, indices, stride, named)
    payload: dict[str, Any] = {
        "centroids": points,
        "tile_indices": indices,
        "radius_px": radii,
        "cell_ids": identifiers,
    }
    if tiles is not None and tiles.tile_shape is not None:
        _check_centroid_frame(points, tiles.tile_shape, named)
        payload["tile_shape"] = tiles.tile_shape
    return payload


def _centroid_radii(
    frame: pd.DataFrame,
    chosen: Mapping[str, str],
    radius_px: float | None,
    n_cells: int,
    named: str,
) -> np.ndarray:
    """Return the disk radius of every centroid, from the table or from the argument."""
    if radius_px is None:
        column = _pick_column(
            frame,
            chosen.get("radius_px"),
            CENTROID_COLUMNS["radius_px"],
            "nucleus radius",
            required=False,
        )
        if column is None:
            raise AlhenaValueError(
                f"No nucleus radius in {named}, and a centroid has no extent of its own. Pass "
                "radius_px, or name the column carrying it through columns=."
            )
        written = frame[column]
        radii = pd.to_numeric(written, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        unreadable = ~(np.isfinite(radii) & (radii > 0.0))
        if unreadable.any():
            raise _unreadable(
                unreadable, written, "row", named, column, "radius", "a number above zero"
            )
        return radii
    try:
        radius = float(radius_px)
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(
            f"radius_px is a number of tile pixels, and was given {radius_px!r}."
        ) from error
    if not np.isfinite(radius) or radius <= 0.0:
        raise AlhenaValueError(f"radius_px must be finite and above zero, got {radius_px!r}.")
    return np.full(n_cells, radius, dtype=np.float64)


def _whole_indices(values: Any, unit: str, named: str, column: Any = None) -> np.ndarray:
    """Return tile indices as int64, refusing one that is missing or not a whole number."""
    written = values if isinstance(values, pd.Series) else pd.Series(values)
    numbers = pd.to_numeric(written, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    unreadable = ~np.isfinite(numbers) | (numbers != np.floor(numbers))
    if unreadable.any():
        raise _unreadable(unreadable, written, unit, named, column, "tile index", "a whole number")
    return numbers.astype(np.int64)


def _unreadable(
    flagged: np.ndarray,
    written: pd.Series,
    unit: str,
    named: str,
    column: Any,
    what: str,
    expected: str,
) -> AlhenaValueError:
    """Return the refusal of the values a column or an argument gives that cannot be read."""
    row = int(np.flatnonzero(flagged)[0])
    count = int(flagged.sum())
    one = count == 1
    value = written.iloc[row]
    shown = "missing" if pd.isna(value) else repr(str(value))
    held = f"whose {column!r} is {shown}" if column is not None else f"which is {shown}"
    return AlhenaValueError(
        f"{count} {unit if one else unit + 's'} of {named} {'gives' if one else 'give'} no {what} "
        f"that can be read, first {unit} {row}, {held}, where {expected} is expected."
    )


def _centroid_identifiers(
    frame: pd.DataFrame,
    chosen: Mapping[str, str],
    indices: np.ndarray,
    stride: int,
    named: str,
) -> np.ndarray:
    """Return the cell identifier of every centroid, minting one where the table names none."""
    identifiers = _named_cells(frame, chosen, named)
    if identifiers is not None:
        return identifiers
    ordinal = pd.Series(indices).groupby(indices).cumcount().to_numpy()
    return global_cell_ids(indices, ordinal, stride=stride)


def _named_cells(table: pd.DataFrame, chosen: Mapping[str, str], named: str) -> np.ndarray | None:
    """Return the cells the identifier column of a table names, or None where it has none."""
    column = _pick_column(
        table, chosen.get("cell_id"), CENTROID_COLUMNS["cell_id"], "cell identifier", required=False
    )
    if column is None:
        return None
    identifiers = table[column].to_numpy()
    if pd.isna(identifiers).any():
        raise AlhenaValueError(
            f"Column {column!r} holds a missing value; every nucleus must be named."
        )
    return _text_identifiers(identifiers, f"Column {column!r} of {named}")


def _check_centroid_frame(points: np.ndarray, tile_shape: tuple[int, int], named: str) -> None:
    """Refuse centroids outside their tile, whose pixels span -0.5 up to its sides less 0.5."""
    height, width = tile_shape
    inside = (
        (points[:, 0] >= -0.5)
        & (points[:, 0] < width - 0.5)
        & (points[:, 1] >= -0.5)
        & (points[:, 1] < height - 0.5)
    )
    if bool(inside.all()):
        return
    outside = int((~inside).sum())
    raise AlhenaValueError(
        f"{outside} of {points.shape[0]} centroids of {named} fall outside the {width} by {height} "
        f"pixels of the tile they name, which span -0.5 up to {width - 0.5:g} across and "
        f"{height - 0.5:g} down; positions in the frame of a whole slide would. "
        f"{_ORIGINS_FRAME_ADVICE}"
    )


def _check_origins_frame(
    tiles: TileSource | None,
    tile_indices: Any,
    chosen: Mapping[str, str],
    stride: int,
) -> None:
    """Refuse what a table in the frame of the tile origins cannot be read with."""
    if tiles is None:
        raise AlhenaValueError(
            "frame='origins' places each centroid on the tiles holding it through the origins of "
            "the tiles, and no tiles were given. Pass tiles=."
        )
    if tiles.coords is None:
        raise AlhenaValueError(
            "frame='origins' places each centroid on the tiles holding it through the origins of "
            "the tiles, and these tiles carry none. Read them from a file carrying their origins, "
            "or give them coords=, for instance from read_tile_origins."
        )
    given = [
        name
        for name, present in (
            ("tile_indices", tile_indices is not None),
            ("columns['tile_index']", "tile_index" in chosen),
            ("stride", stride != CELL_ID_STRIDE),
        )
        if present
    ]
    if given:
        raise AlhenaValueError(
            f"{' and '.join(given)} describe a table naming the tile of each row, and "
            "frame='origins' reads one naming none. Drop them, or read the table with frame='tile'."
        )


def _row_identifiers(table: pd.DataFrame, chosen: Mapping[str, str], named: str) -> np.ndarray:
    """Return the cell each row describes, numbering the rows where the table names no cells."""
    identifiers = _named_cells(table, chosen, named)
    if identifiers is None:
        return np.arange(len(table), dtype=np.int64)
    repeated = pd.Index(identifiers).duplicated()
    if repeated.any():
        cell = identifiers.tolist()[int(np.flatnonzero(repeated)[0])]
        raise AlhenaValueError(
            f"Cell {cell!r} is named on more than one row of {named}. With frame='origins', each "
            "row is one nucleus: name each cell once."
        )
    return identifiers


def _centroids_on_tiles(
    points: np.ndarray,
    radii: np.ndarray,
    identifiers: np.ndarray,
    tiles: TileSource,
    named: str,
    progress: bool | None,
) -> dict[str, Any]:
    """Give each centroid in the frame of the tile origins to every tile holding its pixel."""
    bounds = tile_bounds(tiles)
    shape = _shared_tile_shape(tiles, bounds, named)
    rows, held = _tiles_holding(points, bounds, progress)
    placed = np.zeros(points.shape[0], dtype=bool)
    placed[rows] = True
    missed = int((~placed).sum())
    if missed == points.shape[0]:
        raise AlhenaValueError(
            f"No tile holds any of the {missed} centroids of {named}: they span "
            f"{_extent(points)} in the frame of the tile origins, and the tiles "
            f"{_extent(bounds.reshape(-1, 2))}. Check units= and transform=, and that these are "
            "the tiles the table was written against."
        )
    if missed:
        one = missed == 1
        warnings.warn(
            f"{missed} of the {points.shape[0]} centroids of {named} "
            f"{'lies' if one else 'lie'} on no tile and {'is' if one else 'are'} left out: the "
            f"tiles span {_extent(bounds.reshape(-1, 2))} in the frame of their origins. "
            f"{'It is' if one else 'They are'} not among the nuclei a run reports as reaching no "
            "token.",
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
    scale = float(tiles.coords_scale)
    return {
        "centroids": _tile_pixels(points[rows] - bounds[held, :2], scale),
        "tile_indices": held,
        "radius_px": radii[rows],
        "cell_ids": identifiers[rows],
        "tile_shape": shape,
        "tile_bounds": bounds,
        "tile_frame": tiles.coords_frame,
    }


def _shared_tile_shape(tiles: TileSource, bounds: np.ndarray, named: str) -> tuple[int, int]:
    """Return the height and width every tile shares, which is the canvas disks are painted on."""
    if tiles.tile_shape is not None:
        return tiles.tile_shape
    scale = float(tiles.coords_scale)
    widths = np.rint((bounds[:, 2] - bounds[:, 0]) / scale).astype(np.int64)
    heights = np.rint((bounds[:, 3] - bounds[:, 1]) / scale).astype(np.int64)
    if widths.min() != widths.max() or heights.min() != heights.max():
        raise AlhenaValueError(
            f"The centroids of {named} are painted as disks on one canvas per tile, of one "
            f"size, and these tiles run from {int(widths.min())} to {int(widths.max())} pixels "
            f"across and from {int(heights.min())} to {int(heights.max())} down. Read tiles of "
            "one size."
        )
    return int(heights[0]), int(widths[0])


def _tiles_holding(
    points: np.ndarray, bounds: np.ndarray, progress: bool | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return each pair of a point and a tile holding it, ordered by point and then by tile."""
    low = bounds[:, :2] - 0.5
    high = bounds[:, 2:] - 0.5
    order = np.argsort(points[:, 0], kind="stable")
    along = points[order, 0]
    found_rows: list[np.ndarray] = []
    found_tiles: list[np.ndarray] = []
    n_tiles = bounds.shape[0]
    with _progress.track(
        range(n_tiles), n_tiles, "Placing centroids on tiles", "tile", progress
    ) as windows:
        for tile in windows:
            start = int(np.searchsorted(along, low[tile, 0], side="left"))
            stop = int(np.searchsorted(along, high[tile, 0], side="left"))
            if start == stop:
                continue
            candidates = order[start:stop]
            down = points[candidates, 1]
            inside = candidates[(down >= low[tile, 1]) & (down < high[tile, 1])]
            if inside.size:
                found_rows.append(inside)
                found_tiles.append(np.full(inside.shape, tile, dtype=np.int64))
    if not found_rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rows = np.concatenate(found_rows)
    held = np.concatenate(found_tiles)
    ordered = np.lexsort((held, rows))
    return rows[ordered].astype(np.int64, copy=False), held[ordered]
