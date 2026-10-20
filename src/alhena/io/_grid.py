"""Where the tiles of one image are cut: grids, selected origins and the image edges."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from alhena._errors import PACKAGE_PREFIXES, AlhenaValueError, GeometryWarning
from alhena.io._files import _column_names, _coordinates, _load_npy, _pick_column, _read_table
from alhena.io._formats import HDF5_SUFFIXES, ORIGIN_COLUMNS, TABLE_SUFFIXES
from alhena.io._hdf5 import _read_h5_origins
from alhena.io._sources import TileSource, _first_off_pixel, _is_real


class _Selection(NamedTuple):
    """The origins of the tiles to cut from one image, as an array or a file gives them."""

    # Level-zero x and y of each tile, (n_tiles, 2) int64.
    origins: np.ndarray
    # Side of a tile in level-zero pixels, as a file of origins records it, or None.
    side: float | None = None
    # Where the side is recorded, as a message names it.
    recorded: str = ""


def _selection(
    coords: np.ndarray | pd.DataFrame | str | Path,
    columns: Mapping[str, str] | None,
    dataset: str,
) -> _Selection:
    """Read the origins of the tiles to cut, from an array, a DataFrame or a file."""
    if isinstance(coords, pd.DataFrame):
        return _Selection(_origins_in_table(coords, columns, "coords"))
    if not isinstance(coords, str | Path):
        if columns is not None:
            raise AlhenaValueError(
                "columns names the columns of a table of origins, and coords was given as an array."
            )
        return _Selection(_as_origins_to_cut(coords, "coords"))
    path = Path(coords)
    suffix = path.suffix.lower()
    if not path.is_file():
        raise AlhenaValueError(f"No tile origins at {path}.")
    if columns is not None and suffix not in TABLE_SUFFIXES:
        raise AlhenaValueError(
            f"columns names the columns of a table of origins, and {path.name} is not one of "
            f"{list(TABLE_SUFFIXES)}."
        )
    if suffix == ".npy":
        return _Selection(_as_origins_to_cut(np.asarray(_load_npy(path)), path.name))
    if suffix in HDF5_SUFFIXES:
        values, record = _read_h5_origins(path, dataset)
        origins = _as_origins_to_cut(values, f"{path.name}, {dataset}")
        if record is None:
            return _Selection(origins)
        return _Selection(origins, record.frame_side, f"{path.name} records in {record.where}")
    if suffix in TABLE_SUFFIXES:
        table = _read_table(path, "tile origins")
        return _Selection(_origins_in_table(table, columns, path.name))
    raise AlhenaValueError(
        f"Origins of the tiles to cut are read from a .npy array, from a table, one of "
        f"{list(TABLE_SUFFIXES)}, or from an HDF5 file, one of {list(HDF5_SUFFIXES)}, and "
        f"{path.name} has suffix {path.suffix!r}."
    )


def _origins_in_table(
    table: pd.DataFrame, columns: Mapping[str, str] | None, named: str
) -> np.ndarray:
    """Return the origins of the tiles to cut that a table holds, read by column name."""
    fields_read = {"x": ORIGIN_COLUMNS["x"], "y": ORIGIN_COLUMNS["y"]}
    chosen = _column_names(columns, fields_read, "a table of origins to cut")
    x_column = _pick_column(table, chosen.get("x"), ORIGIN_COLUMNS["x"], "x origin")
    y_column = _pick_column(table, chosen.get("y"), ORIGIN_COLUMNS["y"], "y origin")
    return _as_origins_to_cut(_coordinates(table, (x_column, y_column), named, "origin"), named)


def _as_origins_to_cut(values: Any, described: str) -> np.ndarray:
    """Return origins of tiles to cut as whole pixels, refusing any that cannot select one tile."""
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 2:
        raise AlhenaValueError(
            f"{described} has shape {array.shape}, where (n_tiles, 2) holding the x and y origin "
            "of every tile to cut is expected."
        )
    if array.shape[0] == 0:
        raise AlhenaValueError(f"{described} holds no origin, so it selects no tile to cut.")
    if not _is_real(array):
        raise AlhenaValueError(
            f"{described} has dtype {array.dtype}, where numbers of pixels are expected."
        )
    row = _first_off_pixel(array)
    if row is not None:
        raise AlhenaValueError(
            f"{described} places tile {row} at x {float(array[row, 0])}, y {float(array[row, 1])}, "
            "which is not a whole number of pixels. Give the top-left pixel of each tile."
        )
    origins = array.astype(np.int64)
    repeated = pd.DataFrame(origins).duplicated().to_numpy()
    if repeated.any():
        row = int(np.flatnonzero(repeated)[0])
        raise AlhenaValueError(
            f"{described} gives the origin x {origins[row, 0]}, y {origins[row, 1]} more than "
            f"once, again in row {row}. Give each origin once."
        )
    return origins


def _check_recorded_side(selection: _Selection, source: TileSource) -> None:
    """Warn where a file of origins records a tile side the tiles cut do not span within a pixel."""
    if selection.side is None or source.tile_shape is None:
        return
    side = source.tile_shape[1]
    downsample = float(getattr(source, "downsample", 1.0))
    spans = side * downsample
    if abs(selection.side - spans) < 1.0:
        return
    level = int(getattr(source, "level", 0))
    cut = f"{side} pixels at level {level}" if level else f"{side} pixels"
    suggested = max(round(selection.side / downsample), 1)
    warnings.warn(
        f"{selection.recorded} that the tiles it places span {selection.side:g} pixels of level "
        f"zero, and the tiles cut here, {cut}, span {spans:g}. Pass tile_px={suggested} to cut "
        "the tiles it describes.",
        GeometryWarning,
        skip_file_prefixes=PACKAGE_PREFIXES,
    )


def _grid_steps(tile_px: Any, stride_px: Any, selecting: bool) -> tuple[int, int]:
    """Return the side of the tiles cut from one image and the step between them, in pixels."""
    side = _whole_pixels(tile_px, "tile_px")
    if stride_px is None:
        return side, side
    if selecting:
        raise AlhenaValueError(
            "stride_px spaces the tiles of a grid, and coords selects the tiles to cut instead. "
            "Pass one or the other."
        )
    return side, _whole_pixels(stride_px, "stride_px")


def _whole_pixels(value: Any, name: str) -> int:
    """Return a count of pixels as an int, refusing one that is not a whole number above zero."""
    number = None
    if isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool):
        number = float(value)
    if number is None or not math.isfinite(number) or number != math.floor(number) or number < 1:
        raise AlhenaValueError(f"{name} is a whole number of pixels, at least one, got {value!r}.")
    return int(number)


def _axis_origins(extent: int, tile_px: int, stride: int, edge: str) -> np.ndarray:
    """Return the origins of a grid along one axis of an image, under an edge rule."""
    if edge == "drop":
        return np.arange(0, max(extent - tile_px + 1, 0), stride, dtype=np.int64)
    count = -(-max(extent - tile_px, 0) // stride) + 1
    origins = np.arange(count, dtype=np.int64) * stride
    origins = origins[origins < extent]
    if edge == "shift":
        if extent < tile_px:
            return np.zeros(0, dtype=np.int64)
        origins = np.minimum(origins, extent - tile_px)
    return origins


def _tile_grid(width: int, height: int, tile_px: int, stride: int, edge: str) -> np.ndarray:
    """Return the x, y origin of every tile of a grid over an image, row by row."""
    columns = _axis_origins(width, tile_px, stride, edge)
    rows = _axis_origins(height, tile_px, stride, edge)
    if columns.size == 0 or rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    x, y = np.meshgrid(columns, rows)
    return np.column_stack([x.reshape(-1), y.reshape(-1)]).astype(np.int64)


def _no_tile_fits(
    tile_px: int, extent: tuple[int, int], where: str, edge: str, remedy: str = ""
) -> AlhenaValueError:
    """Return the refusal of a grid in which the edge rule leaves no tile."""
    width, height = extent
    if edge == "drop":
        rule = "and edge='drop' leaves out every tile the edge cuts"
    else:
        rule = "so edge='shift' has no tile to move inside it"
    remedies = ["lower tile_px", *([remedy] if remedy else [])]
    return AlhenaValueError(
        f"A {tile_px} by {tile_px} tile does not fit in the {width} by {height} pixels of "
        f"{where}, {rule}. Pass edge='pad' to read it padded white, or "
        f"{', or '.join(remedies)}."
    )


def _region_window(region: Any) -> tuple[float, float, float, float]:
    """Read a region as x0, y0, x1, y1, refusing one that is not four bounds enclosing an area."""
    try:
        bounds = np.asarray(region, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(
            f"region is x0, y0, x1, y1 in level-zero pixels, and was given {region!r}."
        ) from error
    if bounds.shape[0] != 4 or not np.isfinite(bounds).all():
        raise AlhenaValueError(
            f"region is four finite bounds x0, y0, x1, y1 in level-zero pixels, and was given "
            f"{region!r}."
        )
    x0, y0, x1, y1 = (float(value) for value in bounds)
    if x1 <= x0 or y1 <= y0:
        raise AlhenaValueError(
            f"region {(x0, y0, x1, y1)} encloses no area: x1 and y1 must exceed x0 and y0, upper "
            "bounds excluded."
        )
    return x0, y0, x1, y1


def _inside_region(
    origins: np.ndarray, side: float, region: tuple[float, float, float, float], where: str
) -> np.ndarray:
    """Return which tiles lie wholly inside a region, refusing a region holding none."""
    x0, y0, x1, y1 = region
    inside = (
        (origins[:, 0] >= x0)
        & (origins[:, 1] >= y0)
        & (origins[:, 0] + side <= x1)
        & (origins[:, 1] + side <= y1)
    )
    if not inside.any():
        raise AlhenaValueError(
            f"No tile of {where} lies wholly inside region {region}, so none is kept. Widen the "
            "region, and check that it is written in level-zero pixels."
        )
    return inside


def _placed_selection(
    origins: np.ndarray,
    downsample: float,
    extent: tuple[int, int],
    tile_px: int,
    edge: str,
    where: str,
) -> np.ndarray:
    """Return the pixel of the level read each selected tile starts at, under the edge rule."""
    corners = np.floor(origins / downsample).astype(np.int64)
    size = np.asarray(extent, dtype=np.int64)
    outside = (origins < 0).any(axis=1) | (corners >= size).any(axis=1)
    if outside.any():
        row = int(np.flatnonzero(outside)[0])
        placed = f"Origin {row}, at x {origins[row, 0]}, y {origins[row, 1]}"
        if downsample == 1.0:
            found = f"{placed}, lies outside the {extent[0]} by {extent[1]} pixels of {where}"
        else:
            # The origin is in level-zero pixels, and the level is measured in its own.
            found = (
                f"{placed} in level-zero pixels, falls in pixel x {corners[row, 0]}, y "
                f"{corners[row, 1]} of {where}, outside its {extent[0]} by {extent[1]} pixels"
            )
        raise AlhenaValueError(f"{found}. Give origins inside the image.")
    if downsample != 1.0:
        _refuse_one_tile_twice(
            corners, origins, f"fall in one pixel of {where}, so they select one tile"
        )
    cut = (corners + tile_px > size).any(axis=1)
    if not cut.any() or edge == "pad":
        return corners
    if edge == "drop":
        count = int(cut.sum())
        if count == corners.shape[0]:
            raise AlhenaValueError(
                f"Every tile the {count} origins select extends past the edge of {where}, and "
                "edge='drop' leaves out every tile the edge cuts. Pass edge='pad' to read them "
                "padded white, or edge='shift' to move them inside the image."
            )
        one = count == 1
        warnings.warn(
            f"{count} of the {corners.shape[0]} tiles the origins select "
            f"{'extends' if one else 'extend'} past the edge of {where} and "
            f"{'is' if one else 'are'} left out under edge='drop'. Pass edge='pad' to read "
            f"{'it' if one else 'them'} padded white, or edge='shift' to move "
            f"{'it' if one else 'them'} inside the image.",
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
        return corners[~cut]
    if (size < tile_px).any():
        raise _no_tile_fits(tile_px, extent, where, edge)
    shifted = np.minimum(corners, size - tile_px)
    _refuse_one_tile_twice(
        shifted,
        origins,
        "select one tile once edge='shift' moves the tiles the edge cuts inside the image; select "
        "it once, or pass edge='pad' to read the cut tiles where they lie",
    )
    return shifted


def _refuse_one_tile_twice(corners: np.ndarray, origins: np.ndarray, reason: str) -> None:
    """Refuse two selected origins that come to start their tiles at one pixel."""
    later = pd.DataFrame(corners).duplicated().to_numpy()
    if not later.any():
        return
    second = int(np.flatnonzero(later)[0])
    first = int(np.flatnonzero((corners == corners[second]).all(axis=1))[0])
    raise AlhenaValueError(
        f"The origins x {origins[first, 0]}, y {origins[first, 1]} and x {origins[second, 0]}, "
        f"y {origins[second, 1]} {reason}."
    )


def _origin_names(origins: np.ndarray) -> tuple[str, ...]:
    """Return the name of each tile cut from one image, after its level-zero origin."""
    return tuple(f"x{int(x)}_y{int(y)}" for x, y in origins)


def _padded(window: np.ndarray, height: int, width: int, fill: int) -> np.ndarray:
    """Return a window cut short by the edge of its image, completed to full size with `fill`."""
    if window.shape[:2] == (height, width):
        return window
    padded = np.full((height, width, *window.shape[2:]), fill, dtype=window.dtype)
    padded[: window.shape[0], : window.shape[1]] = window
    return padded
