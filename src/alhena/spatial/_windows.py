"""The windows of tiles, read from an array or a shapes element."""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from alhena._errors import AlhenaValueError
from alhena.io._carry import _apply
from alhena.spatial._elements import _SAME_TRANSFORMATION, _carrier, _Frame, _frame, _read, _shapes

PATCHES_KEY: Final[str] = "embeddings_patches"
"""Shapes element a patch pipeline writes its tile windows to."""

_BOUNDS_COLUMNS: Final[tuple[str, ...]] = ("minx", "miny", "maxx", "maxy")

# Column a patch pipeline writes the full window of a tile to, beside a geometry it may clip.
_WINDOWS_COLUMN: Final[str] = "bboxes"

# Share of its bounds a window geometry may fall short by: rounding, never a clipped pixel.
_BOX_SHORTFALL: Final[float] = 1e-6


def patch_bounds(
    sdata: Any,
    *,
    patches: str | np.ndarray = PATCHES_KEY,
    coordinate_system: str | None = None,
    image_key: str | None = None,
) -> np.ndarray:
    """Return the window of every tile, in the frame the nuclei are read against.

    Windows given as an array are returned as they are. The windows of a shapes element are read in
    the frame `cells_from_shapes` reads, and refused where that frame rotates, mirrors or shears
    them. A `bboxes` column keeping each full window beside a geometry clipped to the tissue is read
    instead; without one, every geometry must fill its bounds, and a window clipped along the axes
    reads as a smaller tile.

    Args:
        sdata: The object holding the tile windows.
        patches: Shapes element holding one box per tile, or an `(n_tiles, 4)` array of x0, y0, x1,
            y1 windows. The tile index is the row index.
        coordinate_system: Coordinate system the windows are read in, or carried through into the
            pixels of `image_key`.
        image_key: Image the tiles are read from, in whose level-zero pixels the windows are read.

    Returns:
        An `(n_tiles, 4)` float64 array of x0, y0, x1, y1 windows.

    Raises:
        AlhenaValueError: If the windows are missing or malformed, or cannot be read as boxes.
        MissingExtraError: If the spatial extra is not installed.
    """
    frame = _frame(sdata, coordinate_system, image_key)
    if not isinstance(patches, str):
        return _array_windows(patches)
    return _element_windows(sdata, _shapes(sdata, patches), patches, frame)


def _array_windows(patches: np.ndarray) -> np.ndarray:
    """Return tile windows given as an array, refusing one that describes no tile."""
    windows = np.asarray(patches, dtype=np.float64)
    if windows.ndim != 2 or windows.shape[1] != 4:
        raise AlhenaValueError(
            f"Tile windows have shape {windows.shape}, where an (n_tiles, 4) array of x0, y0, "
            "x1, y1 is expected."
        )
    if windows.shape[0] == 0:
        raise AlhenaValueError(
            "These tile windows hold no row. Pass the windows of the tiles the embeddings were "
            "computed over."
        )
    return windows


def _element_windows(sdata: Any, element: Any, name: str, frame: _Frame) -> np.ndarray:
    """Return the windows of a shapes element in the frame named, from `bboxes` where it is."""
    if len(element.geometry) == 0:
        raise AlhenaValueError(f"The element {name!r} holds no tile window.")
    carrier = _carrier(sdata, element, name, frame)
    if carrier is not None:
        _check_aligned(carrier, name, frame)

    full = _column_windows(element, name)
    if full is not None:
        if carrier is not None:
            # A carrier that passed the check scales and shifts, so each corner stays a corner.
            return np.column_stack([_apply(carrier, full[:, :2]), _apply(carrier, full[:, 2:])])
        if not frame.stored:
            raise AlhenaValueError(
                f"The windows of {name!r} are held in its {_WINDOWS_COLUMN!r} column, and this "
                "object declares no coordinate system whose transformation would carry them into "
                f"{frame.described}. Read them from a SpatialData object, or as they are stored."
            )
        return full

    read = _read(element, frame, carrier)
    windows = _bounds(read, name)
    _check_boxes(read.geometry, windows, name)
    return windows


def _bounds(element: Any, name: str) -> np.ndarray:
    """Return the x0, y0, x1, y1 bounds of every geometry of a shapes element."""
    bounds = element.bounds
    missing = [column for column in _BOUNDS_COLUMNS if column not in bounds.columns]
    if missing:
        raise AlhenaValueError(
            f"The element {name!r} yields no {missing} bounds, so its geometries do not describe "
            "tile windows."
        )
    return bounds.loc[:, list(_BOUNDS_COLUMNS)].to_numpy(dtype=np.float64)


def _column_windows(element: Any, name: str) -> np.ndarray | None:
    """Return the full windows a `bboxes` column holds, each around its geometry, or None."""
    columns = getattr(element, "columns", None)
    if columns is None or _WINDOWS_COLUMN not in columns:
        return None
    entries = list(element[_WINDOWS_COLUMN])
    try:
        windows = np.asarray(
            [np.asarray(entry, dtype=np.float64) for entry in entries], dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(
            f"The {_WINDOWS_COLUMN!r} column of {name!r} holds something other than one x0, y0, "
            "x1, y1 window per row."
        ) from error
    if windows.shape != (len(entries), 4) or not np.isfinite(windows).all():
        raise AlhenaValueError(
            f"The {_WINDOWS_COLUMN!r} column of {name!r} holds values of shape {windows.shape}, "
            "where one finite x0, y0, x1, y1 window per row is expected."
        )
    bounds = _bounds(element, name)
    slack = _SAME_TRANSFORMATION * np.maximum(1.0, np.abs(windows))
    outside = (
        (bounds[:, :2] < windows[:, :2] - slack[:, :2])
        | (bounds[:, 2:] > windows[:, 2:] + slack[:, 2:])
    ).any(axis=1)
    if outside.any():
        raise AlhenaValueError(
            f"{int(outside.sum())} of the {len(entries)} geometries of {name!r} reach outside the "
            f"window its {_WINDOWS_COLUMN!r} column gives them. Drop the column to read the "
            "windows from the geometries, or pass the windows as an array."
        )
    return windows


def _check_boxes(geometry: Any, windows: np.ndarray, name: str) -> None:
    """Refuse window geometries that do not fill their bounds, as one clipped to the tissue."""
    areas = np.asarray(geometry.area, dtype=np.float64)
    boxes = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    short = areas < boxes * (1.0 - _BOX_SHORTFALL)
    if short.any():
        raise AlhenaValueError(
            f"{int(short.sum())} of the {windows.shape[0]} windows of {name!r} do not fill the box "
            "they are bounded by, as a window clipped to the tissue does not. Keep each full "
            f"window in a {_WINDOWS_COLUMN!r} column of the element, as x0, y0, x1, y1, or pass "
            "the windows as an array."
        )


def _check_aligned(carrier: np.ndarray, name: str, frame: _Frame) -> None:
    """Refuse tile windows the frame they are read in rotates, mirrors or shears."""
    linear = carrier[:2, :2]
    diagonal = max(abs(float(linear[0, 0])), abs(float(linear[1, 1])))
    if (
        linear[0, 0] > 0.0
        and linear[1, 1] > 0.0
        and abs(float(linear[0, 1])) <= _SAME_TRANSFORMATION * diagonal
        and abs(float(linear[1, 0])) <= _SAME_TRANSFORMATION * diagonal
    ):
        return
    raise AlhenaValueError(
        f"The windows of {name!r} are carried into {frame.described} by a transformation that "
        "rotates, mirrors or shears them. Pass image_key naming the image the windows were cut "
        "from, in whose pixels they are boxes."
    )
