"""Checks that tiles and nuclei agree before any tile is embedded."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alhena._constants import INPUT_SIZE, REFERENCE_MPP
from alhena._errors import AlhenaValueError, GeometryError, MembershipError
from alhena._frames import _as_tile_scale, _window_grid
from alhena._geometry import _target_scale
from alhena.cells._canvas import _POSITION_SLACK
from alhena.cells._centroids import _as_radii
from alhena.cells._inputs import (
    _BUILDER_OPTIONS,
    _CENTROIDS,
    _LABELS,
    _POLYGONS,
    _TABLE,
    _check_indices,
    _Placement,
    _tile_names,
    _tile_windows,
    _windows_at_origins,
)
from alhena.cells._labels import _two_label_maps
from alhena.cells._table import (
    _ATTR_GRIDS,
    _ATTR_MPP,
    _ATTR_TARGET_MPP,
    _ATTR_TILE_BOUNDS,
    _columns_of,
    validate_membership,
)
from alhena.io import TileSource, tile_bounds
from alhena.io._labels import _LabelWindows
from alhena.types import TileInput


def _check_tile_alignment(
    kind: str,
    payload: Any,
    options: Mapping[str, Any],
    n_tiles: int,
    tiles: TileInput = None,
    placement: _Placement | None = None,
) -> None:
    """Check the nuclei and the tiles describe the same tiles, before any forward pass."""
    if kind == _TABLE:
        validate_membership(payload, n_tiles)
        _check_table_windows(payload, tiles, n_tiles)
        return
    if kind == _LABELS:
        _check_read_tiles(options, n_tiles, placement)
        indices = options.get("tile_indices")
        if indices is None:
            if len(payload) != n_tiles:
                raise AlhenaValueError(
                    f"{len(payload)} label maps were given for {n_tiles} tiles. Pass one map per "
                    "tile, or name the tile of each map with 'tile_indices'."
                )
            return
        _check_indices(indices, n_tiles, expected=len(payload))
        return
    if kind == _CENTROIDS:
        for required in ("tile_indices", "radius_px"):
            if required not in options:
                raise AlhenaValueError(
                    f"Centroids need {required!r}: a centroid carries no tile and no extent of "
                    f"its own. The centroid builder takes {list(_BUILDER_OPTIONS[_CENTROIDS])}."
                )
        _check_read_tiles(options, n_tiles, placement)
        _check_indices(options["tile_indices"], n_tiles)
        # Radii count against the centroids; indices of another count, the builder refuses.
        if hasattr(payload, "__len__") and np.size(options["tile_indices"]) == len(payload):
            _as_radii(options["radius_px"], len(payload))
        return
    windows = _tile_windows(options.get("tile_bounds"))
    if windows.shape[0] != n_tiles:
        raise AlhenaValueError(
            f"{windows.shape[0]} tile windows were given for {n_tiles} tiles; one window per "
            "tile is required, and the tile index is the row index."
        )
    _check_windows_frame(options.get("tile_frame"), windows, placement)


def _check_read_tiles(
    options: Mapping[str, Any], n_tiles: int, placement: _Placement | None
) -> None:
    """Refuse nuclei a reader placed against other tiles than these, by count or by window."""
    recorded = options.get("tile_bounds")
    if recorded is None:
        return
    windows = np.asarray(recorded, dtype=np.float64)
    if windows.ndim != 2 or windows.shape[1] != 4:
        raise AlhenaValueError(
            f"tile_bounds has shape {windows.shape}, where the (n_tiles, 4) windows of the tiles "
            "the nuclei were read against are expected."
        )
    count = windows.shape[0]
    if count != n_tiles:
        raise AlhenaValueError(
            f"These nuclei were read against {count} {'tile' if count == 1 else 'tiles'}, and "
            f"{n_tiles} {'tile was' if n_tiles == 1 else 'tiles were'} given. Read the nuclei "
            "against the tiles they are given to, passing that source as tiles=."
        )
    _check_windows_frame(options.get("tile_frame"), windows, placement)


def _check_windows_frame(named: Any, windows: np.ndarray, placement: _Placement | None) -> None:
    """Refuse windows of the frame of these tiles that do not start at their origins."""
    if named is None or placement is None or str(named) != placement.frame:
        return
    if _windows_at_origins(windows, placement):
        return
    raise AlhenaValueError(
        f"These nuclei were read against tile windows of the frame {placement.frame!r}, which "
        "these tiles name as well, and the windows do not start at the origins of these tiles. "
        "Read the nuclei against the tiles they are given to, passing that source as tiles=."
    )


def _check_table_windows(frame: Any, tiles: TileInput, n_tiles: int) -> None:
    """Refuse a membership table built against other windows than those of the tiles given."""
    recorded = getattr(frame, "attrs", {}).get(_ATTR_TILE_BOUNDS)
    if recorded is None or not isinstance(tiles, TileSource) or tiles.coords is None:
        return
    windows = np.asarray(recorded, dtype=np.float64)
    if windows.ndim != 2 or windows.shape[1] != 4:
        return
    count = windows.shape[0]
    if count != n_tiles:
        raise MembershipError(
            f"This membership table was built against {count} tile "
            f"{'window' if count == 1 else 'windows'}, and {n_tiles} "
            f"{'tile was' if n_tiles == 1 else 'tiles were'} given. Build the table against the "
            "windows of these tiles, io.tile_bounds(tiles)."
        )
    if tiles.tile_shape is None:
        return
    own = np.asarray(tile_bounds(tiles), dtype=np.float64)
    if np.allclose(windows, own, rtol=_POSITION_SLACK, atol=_POSITION_SLACK):
        return
    raise MembershipError(
        "This membership table was built against other windows than those of these tiles: its "
        f"first window is {windows[0].tolist()} and the first tile spans {own[0].tolist()}. "
        "Build the table against the windows of these tiles, io.tile_bounds(tiles)."
    )


def _reached(shape: Sequence[int], mpp: float | None, target_mpp: float) -> tuple[int, int]:
    """Return the height and width a tile or a mask reaches before it is fitted to the canvas."""
    if mpp is None:
        return INPUT_SIZE, INPUT_SIZE
    scale = _target_scale(mpp, target_mpp)
    height, width = int(shape[0]), int(shape[1])
    if scale == 1.0:
        return height, width
    return round(height * scale), round(width * scale)


def _fit_refusal(
    name: str,
    shape: Sequence[int],
    recorded: float | None,
    target_mpp: float,
    declared: float | None,
    fitted: tuple[int, int],
    image: tuple[int, int],
) -> MembershipError:
    """Say how a membership table and a tile reach the canvas at two sizes, and what to do."""
    size = f"{int(shape[1])} by {int(shape[0])} pixels"
    masks = f"{fitted[1]} by {fitted[0]} pixels"
    images = f"{image[1]} by {image[0]} pixels"
    lost = "The masks would then point at other tokens than the nuclei they outline."
    if recorded is None:
        return MembershipError(
            f"This membership table was built with no resolution, so it stretches the masks of "
            f"tile {name}, {size}, onto the whole {INPUT_SIZE} pixel canvas, and the tiles are "
            f"declared at {declared:g} micron per pixel, which brings the image of that tile onto "
            f"{images} before it is fitted to the canvas. {lost} Build the table with "
            f"mpp={declared:g}."
        )
    built = f"{recorded:g} micron per pixel"
    toward_default = target_mpp == REFERENCE_MPP
    target = f"the default target_mpp of {REFERENCE_MPP}"
    if not toward_default:
        built += f" toward {target_mpp:g}, where the encoder reads {REFERENCE_MPP}"
    if declared is None:
        advice = f"declare mpp={recorded:g} for the tiles"
        if not toward_default:
            advice += f", and build the table with {target}"
        return MembershipError(
            f"This membership table was built at {built}, which brings the masks of tile {name}, "
            f"drawn on {size}, onto {masks} before they are fitted to the canvas, and the tiles "
            "are declared at no resolution, so each is resized onto the whole "
            f"{INPUT_SIZE} pixel canvas as it is. {lost} Declare one resolution for both: "
            f"{advice}."
        )
    if toward_default:
        advice = (
            f"Declare one resolution for both: build the table with mpp={declared:g}, or declare "
            f"mpp={recorded:g} for the tiles."
        )
    elif recorded == declared:
        advice = f"Build the table with {target}."
    else:
        advice = f"Build the table with mpp={declared:g} and {target}."
    return MembershipError(
        f"This membership table was built at {built}, and the tiles are declared at {declared:g}: "
        f"the masks of tile {name}, drawn on {size}, reach {masks} before they are fitted to the "
        f"canvas, and its image {images}. {lost} {advice}"
    )


@dataclass(frozen=True)
class _CanvasFit:
    """The check a table built with no resolution leaves to the tiles known only once read."""

    declared: float
    pending: np.ndarray

    def check(self, tiles: TileInput, index: int, shape: Sequence[int]) -> None:
        """Refuse a tile, once read, unless the resolution declared brings it onto the canvas."""
        if not self.pending[index]:
            return
        canvas = (INPUT_SIZE, INPUT_SIZE)
        image = _reached(shape, self.declared, REFERENCE_MPP)
        if image != canvas:
            name = _tile_names(tiles, np.array([index]))[0]
            raise _fit_refusal(name, shape, None, REFERENCE_MPP, self.declared, canvas, image)


def _recorded_grids(frame: pd.DataFrame) -> np.ndarray:
    """Return the rows of tile index, height and width a builder drew the nuclei on."""
    rows = frame.attrs.get(_ATTR_GRIDS)
    if rows is None:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def _check_table_fit(
    tiles: TileInput, frame: pd.DataFrame, declared: float | None, shapes: np.ndarray
) -> _CanvasFit | None:
    """Refuse a membership table whose masks reach the canvas otherwise than its tiles do."""
    if _ATTR_MPP not in frame.attrs:
        return None
    recorded = frame.attrs[_ATTR_MPP]
    if recorded is None and declared is None:
        return None
    named = np.unique(_columns_of(frame)["tile_index"].to_numpy(dtype=np.int64))
    canvas = (INPUT_SIZE, INPUT_SIZE)
    if recorded is None:
        assert declared is not None
        known = named[shapes[named, 0] >= 0]
        for height, width in np.unique(shapes[known], axis=0).tolist():
            image = _reached((height, width), declared, REFERENCE_MPP)
            if image != canvas:
                tile = known[(shapes[known] == (height, width)).all(axis=1)][:1]
                name = _tile_names(tiles, tile)[0]
                raise _fit_refusal(
                    name, (height, width), None, REFERENCE_MPP, declared, canvas, image
                )
        waiting = named[shapes[named, 0] < 0]
        if waiting.size == 0:
            return None
        pending = np.zeros(shapes.shape[0], dtype=bool)
        pending[waiting] = True
        return _CanvasFit(float(declared), pending)
    target_mpp = float(frame.attrs.get(_ATTR_TARGET_MPP, REFERENCE_MPP))
    rows = _recorded_grids(frame)
    rows = rows[np.isin(rows[:, 0], named)]
    for height, width in np.unique(rows[:, 1:], axis=0).tolist():
        fitted = _reached((height, width), float(recorded), target_mpp)
        image = _reached((height, width), declared, REFERENCE_MPP)
        if fitted != image:
            tile = rows[(rows[:, 1] == height) & (rows[:, 2] == width), 0][:1]
            name = _tile_names(tiles, tile)[0]
            raise _fit_refusal(
                name, (height, width), float(recorded), target_mpp, declared, fitted, image
            )
    return None


def _table_grids(frame: pd.DataFrame, n_tiles: int) -> np.ndarray | None:
    """Return the grid a table drew the nuclei of each tile on, -1 for none, None without mpp."""
    if frame.attrs.get(_ATTR_MPP) is None:
        return None
    grids = np.full((n_tiles, 2), -1, dtype=np.int64)
    for tile, height, width in _recorded_grids(frame).tolist():
        if not 0 <= tile < n_tiles:
            continue
        if grids[tile, 0] >= 0 and (int(grids[tile, 0]), int(grids[tile, 1])) != (height, width):
            raise GeometryError(
                f"The membership table drew the nuclei of tile {tile} on two grids, "
                f"{grids[tile, 1]} by {grids[tile, 0]} and {width} by {height} pixels. A mask "
                "shares the pixel grid of its tile, so the masks of one tile have one size."
            )
        grids[tile] = (height, width)
    return grids


def _nucleus_grids(
    kind: str,
    payload: Any,
    options: dict[str, Any],
    n_tiles: int,
    mpp: float | None,
    placed: bool,
    shapes: np.ndarray,
) -> np.ndarray | None:
    """Return the pixel grid the nuclei of each tile are drawn on, -1 where there is none."""
    if kind == _TABLE:
        return None if mpp is None else _table_grids(payload, n_tiles)
    grids = np.full((n_tiles, 2), -1, dtype=np.int64)
    if kind == _LABELS:
        if mpp is None and not placed:
            return None
        indices = options.get("tile_indices")
        tiles = np.arange(len(payload)) if indices is None else np.asarray(indices).reshape(-1)
        if not np.issubdtype(tiles.dtype, np.integer):
            # The builder refuses indices that are not integers, and says so.
            return None
        if isinstance(payload, _LabelWindows):
            # Windows of a label image share the tile shape: reading them would only repeat it.
            map_shapes = [payload.map_shape] * len(payload)
        else:
            map_shapes = [(int(np.shape(entry)[0]), int(np.shape(entry)[1])) for entry in payload]
        for tile, shape in zip(tiles.tolist(), map_shapes, strict=True):
            if grids[tile, 0] >= 0 and tuple(grids[tile].tolist()) != shape:
                raise _two_label_maps(tile, grids[tile].tolist(), shape)
            grids[tile] = shape
        return grids
    if kind == _CENTROIDS:
        # Centroids given no tile_shape take the size of the tiles they lie on, else the known.
        if "tile_shape" not in options:
            named = np.unique(np.asarray(options["tile_indices"]).reshape(-1)).astype(np.int64)
            held = shapes[named][shapes[named, 0] >= 0]
            sizes = np.unique(held if held.shape[0] else shapes[shapes[:, 0] >= 0], axis=0)
            if held.shape[0] and sizes.shape[0] > 1:
                raise GeometryError(
                    "Centroids are painted on one tile_shape for every tile, and the tiles they "
                    f"lie on are of several sizes, such as {sizes[0, 1]} by {sizes[0, 0]} and "
                    f"{sizes[1, 1]} by {sizes[1, 0]} pixels. Read tiles of one size."
                )
            if sizes.shape[0] == 1:
                options["tile_shape"] = (int(sizes[0, 0]), int(sizes[0, 1]))
        try:
            height, width = (int(value) for value in options.get("tile_shape", (INPUT_SIZE,) * 2))
        except (TypeError, ValueError):
            return None
        tiles = np.unique(np.asarray(options["tile_indices"]).reshape(-1))
        grids[tiles.astype(np.int64)] = (height, width)
        return grids
    if kind == _POLYGONS:
        if mpp is None:
            return None
        bounds = _tile_windows(options.get("tile_bounds"))
        scale = _as_tile_scale(options.get("tile_scale", 1.0))
        if not np.isfinite(bounds).all():
            return None
        for tile in range(n_tiles):
            width, height = _window_grid(bounds[tile], scale)
            grids[tile] = (height, width)
        return grids
    return None


def _check_nucleus_grids(
    tiles: TileInput, kind: str, grids: np.ndarray, shapes: np.ndarray
) -> np.ndarray | None:
    """Refuse nuclei drawn on another grid than their tile; return the grids left for later."""
    drawn = grids[:, 0] >= 0
    known = shapes[:, 0] >= 0
    differs = drawn & known & (grids != shapes).any(axis=1)
    if differs.any():
        tile = int(np.flatnonzero(differs)[0])
        name = _tile_names(tiles, np.array([tile]))[0]
        raise _grid_mismatch(kind, name, grids[tile], shapes[tile])
    pending = drawn & ~known
    if not pending.any():
        return None
    later = np.full_like(grids, -1)
    later[pending] = grids[pending]
    return later


def _grid_mismatch(kind: str, name: str, grid: np.ndarray, shape: Sequence[int]) -> GeometryError:
    """Say which nuclei are drawn on another grid than their tile, and what that breaks."""
    drawn = f"{int(grid[1])} by {int(grid[0])}"
    tile = f"{int(shape[1])} by {int(shape[0])}"
    if kind == _CENTROIDS:
        return GeometryError(
            f"Tile {name} is {tile} pixels, and centroids are painted on {drawn} for every tile. "
            "Give the size of the tiles as tile_shape, or read tiles of one size."
        )
    if kind == _POLYGONS:
        return GeometryError(
            f"Tile {name} is {tile} pixels and its window spans {drawn} of them at the scale of "
            "the tile. Give the windows of these tiles, and the tile_scale of their origins."
        )
    if kind == _TABLE:
        return GeometryError(
            f"Tile {name} is {tile} pixels and the membership table drew its nuclei on {drawn}. "
            "Build the table against these tiles: their windows and tile_scale for boundaries, "
            "their size for label maps and centroids."
        )
    return GeometryError(
        f"Tile {name} is {tile} pixels and its label map {drawn}. A label map shares the pixel "
        "grid of its tile: resample the map to its tile."
    )
