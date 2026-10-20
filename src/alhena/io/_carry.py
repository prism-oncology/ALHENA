"""Cell identifiers, coordinate units and affine transforms the nucleus readers share."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alhena._errors import AlhenaValueError
from alhena.io._formats import CELL_ID_STRIDE, COORDINATE_UNITS, TRANSFORM_SUFFIXES
from alhena.io._sources import TileSource

# Largest departure from 0, 0, 1 of a last row read as affine, for matrices written in decimals.
_AFFINE_ROW_TOLERANCE: float = 1e-9


def global_cell_ids(
    tile_index: int | Sequence[int] | np.ndarray,
    instance: int | Sequence[int] | np.ndarray,
    stride: int = CELL_ID_STRIDE,
) -> np.ndarray:
    """Make per-tile instance numbers unique across the tiles, as `tile_index * stride + instance`.

    The rule depends on the tile index and the instance number alone. A cell lying in two tiles gets
    two identifiers, so the rule applies only to instances numbered within each tile.

    Args:
        tile_index: Index of the tile each instance was numbered in.
        instance: Instance number within that tile, zero or above.
        stride: Number of identifiers in the block each tile owns.

    Returns:
        An int64 array of identifiers, broadcast over the two inputs.

    Raises:
        AlhenaValueError: If an argument is out of range, or an instance does not fit its block.
    """
    if stride < 1:
        raise AlhenaValueError(f"stride must be at least one, got {stride}.")
    tiles = np.asarray(tile_index, dtype=np.int64)
    instances = np.asarray(instance, dtype=np.int64)
    if tiles.size and int(tiles.min()) < 0:
        raise AlhenaValueError("tile_index must not be negative.")
    if instances.size and int(instances.min()) < 0:
        raise AlhenaValueError("instance must not be negative.")
    if instances.size and int(instances.max()) >= stride:
        raise AlhenaValueError(
            f"Instance {int(instances.max())} does not fit in a block of {stride} identifiers. "
            "Raise stride."
        )
    return tiles * np.int64(stride) + instances


def _pixel_size(tiles: TileSource | None, origins: bool) -> float:
    """Return the microns one pixel spans in the frame of a tile, or of the tile origins."""
    frame = "frame of the tile origins" if origins else "frame of a tile"
    if tiles is None:
        raise AlhenaValueError(
            f"units='um' reads microns, which are divided by the microns one pixel of the {frame} "
            "spans, and no tiles were given to take it from. Pass tiles=."
        )
    if tiles.mpp is None:
        raise AlhenaValueError(
            f"units='um' reads microns, which are divided by the microns one pixel of the {frame} "
            "spans, and these tiles declare no resolution. Declare mpp= when reading them."
        )
    return float(tiles.mpp) / float(tiles.coords_scale) if origins else float(tiles.mpp)


def _carrier(
    units: str, transform: Any, tiles: TileSource | None, origins: bool
) -> np.ndarray | None:
    """Return the matrix taking coordinates into frame pixels, or None if they are in them."""
    if transform is not None:
        if units != COORDINATE_UNITS[0]:
            raise AlhenaValueError(
                "units='um' was given beside a transform. Put the unit scale into the matrix and "
                "leave units at 'px'."
            )
        return _affine(transform)
    if units == COORDINATE_UNITS[0]:
        return None
    scale = 1.0 / _pixel_size(tiles, origins)
    return np.diag([scale, scale, 1.0])


def _affine(transform: Any) -> np.ndarray:
    """Return a transform as a 3 by 3 affine matrix, refusing a projective or collapsing one."""
    if isinstance(transform, str | Path):
        file = Path(transform)
        suffix = file.suffix.lower()
        if suffix not in TRANSFORM_SUFFIXES:
            raise AlhenaValueError(
                f"A transform is read from {list(TRANSFORM_SUFFIXES)} files, and {file.name} has "
                f"suffix {file.suffix!r}."
            )
        if not file.is_file():
            raise AlhenaValueError(f"No transform at {file}.")
        try:
            if suffix == ".npy":
                values = np.load(file, allow_pickle=False)
            else:
                values = np.loadtxt(file, delimiter=",", ndmin=2)
        except (ValueError, OSError) as error:
            raise AlhenaValueError(
                f"{file.name} holds no matrix of numbers that can be read: {error}"
            ) from error
        described = file.name
    else:
        values = transform
        described = "The transform"
    try:
        matrix = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(
            f"{described} is not a matrix of numbers. A transform is a 2 by 3 or 3 by 3 affine "
            "matrix."
        ) from error
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    elif matrix.shape != (3, 3):
        raise AlhenaValueError(
            f"{described} has shape {matrix.shape}, where a 2 by 3 or 3 by 3 affine matrix is "
            "expected."
        )
    if not np.isfinite(matrix).all():
        raise AlhenaValueError(f"{described} holds a value that is not finite.")
    if np.abs(matrix[2] - (0.0, 0.0, 1.0)).max() > _AFFINE_ROW_TOLERANCE:
        raise AlhenaValueError(
            f"{described} ends in the row {matrix[2].tolist()}, where an affine matrix ends in "
            "0, 0, 1: it is projective."
        )
    if not _invertible(matrix):
        raise AlhenaValueError(
            f"{described} collapses the plane onto a line, so it cannot carry nuclei into the "
            "frame of the tile origins."
        )
    matrix[2] = (0.0, 0.0, 1.0)
    return matrix


def _invertible(matrix: np.ndarray) -> bool:
    """Return whether an affine matrix is finite and carries the plane onto the whole plane."""
    return bool(np.isfinite(matrix).all()) and float(np.linalg.det(matrix[:2, :2])) != 0.0


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return (n, 2) points carried by a 3 by 3 affine matrix."""
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def _extent(points: np.ndarray) -> str:
    """Describe the box a set of positions spans, as a message names it."""
    low = points.min(axis=0)
    high = points.max(axis=0)
    return f"x {low[0]:.1f} to {high[0]:.1f} and y {low[1]:.1f} to {high[1]:.1f}"
