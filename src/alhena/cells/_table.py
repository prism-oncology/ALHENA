"""The membership table assigning tokens to cells, and its checks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from alhena._constants import INPUT_SIZE, NUM_TOKENS
from alhena._errors import MembershipError
from alhena.types import MEMBERSHIP_COLUMNS

_ATTR_SOURCE = "alhena_membership_source"
_ATTR_APPROXIMATE = "alhena_membership_approximate"
_ATTR_MISSING = "alhena_missing_cells"
_ATTR_INPUT_CELLS = "alhena_n_input_cells"
_ATTR_TILE_CENTERS = "alhena_tile_centers"
_ATTR_CELL_CENTERS = "alhena_cell_centers"
_ATTR_FRAME = "alhena_centroids_frame"
_ATTR_FRAME_MPP = "alhena_centroids_mpp"
# The resolution the masks were fitted at, None where they were stretched onto the canvas.
_ATTR_MPP = "alhena_membership_mpp"
_ATTR_TARGET_MPP = "alhena_membership_target_mpp"
# The pixel grid of each tile's nuclei: rows of tile index, height and width.
_ATTR_GRIDS = "alhena_tile_grids"
# The windows a table was built against, so that a table given other tiles is told apart.
_ATTR_TILE_BOUNDS = "alhena_tile_bounds"


def validate_membership(frame: pd.DataFrame, n_tiles: int) -> None:
    """Check that a membership table can be pooled against a set of tiles.

    Args:
        frame: A table with `MEMBERSHIP_COLUMNS`.
        n_tiles: Number of tiles the table indexes into.

    Raises:
        MembershipError: If a column, dtype, range or key is wrong, or the recorded cells disagree.
    """
    if not isinstance(frame, pd.DataFrame):
        raise MembershipError(f"Expected a membership table, got {type(frame).__name__}.")
    if tuple(frame.columns) != MEMBERSHIP_COLUMNS:
        raise MembershipError(
            f"A membership table has columns {MEMBERSHIP_COLUMNS}, got {tuple(frame.columns)}."
        )
    if n_tiles < 0:
        raise MembershipError(f"n_tiles must not be negative, got {n_tiles}.")

    table = _columns_of(frame)
    for column in ("tile_index", "token_index", "nucleus_px"):
        if not pd.api.types.is_integer_dtype(table[column]):
            raise MembershipError(
                f"Column {column} must be an integer column, got {table[column].dtype}."
            )
    if table["cell_id"].isna().any():
        raise MembershipError("Column cell_id holds missing values; every row names one cell.")

    if len(table):
        tile = table["tile_index"].to_numpy()
        token = table["token_index"].to_numpy()
        pixels = table["nucleus_px"].to_numpy()
        if int(tile.min()) < 0 or int(tile.max()) >= n_tiles:
            raise MembershipError(
                f"tile_index runs from {int(tile.min())} to {int(tile.max())}, outside the "
                f"{n_tiles} tiles given."
            )
        if int(token.min()) < 0 or int(token.max()) >= NUM_TOKENS:
            raise MembershipError(
                f"token_index runs from {int(token.min())} to {int(token.max())}, outside the "
                f"{NUM_TOKENS} tokens of the {INPUT_SIZE} pixel canvas."
            )
        if int(pixels.min()) <= 0:
            raise MembershipError(
                "nucleus_px must be positive; a row records the nucleus pixels that fall in "
                "one token."
            )
        repeated = table.duplicated(subset=["tile_index", "cell_id", "token_index"])
        if bool(repeated.any()):
            raise MembershipError(
                f"{int(repeated.sum())} rows repeat a (tile_index, cell_id, token_index) key. "
                "Pixel counts must be aggregated first, or a token contributes twice."
            )
    _validate_partition(frame)


def missing_cells(frame: pd.DataFrame) -> np.ndarray:
    """Return the cells of the input that reached no token, sorted.

    A nucleus outside the center crop, or lost to nearest-neighbor downsampling, has no token to
    pool.

    Args:
        frame: A table one of the membership builders returned.

    Returns:
        The identifiers of the cells with no token, sorted.

    Raises:
        MembershipError: If the table records no input set, not having been built by a builder.
    """
    if _ATTR_MISSING not in frame.attrs:
        raise MembershipError(
            "This table records no set of input cells, so the cells that reached no token "
            "cannot be recovered from it. Build it with membership_from_labels, "
            "membership_from_centroids or membership_from_polygons."
        )
    return np.asarray(frame.attrs[_ATTR_MISSING])


def membership_source(frame: pd.DataFrame) -> str:
    """Return the kind of nucleus input a membership table was built from.

    Args:
        frame: A membership table.

    Returns:
        `"labels"`, `"centroids"` or `"polygons"`, or `"unknown"` for a table built by hand.
    """
    return str(frame.attrs.get(_ATTR_SOURCE, "unknown"))


def membership_is_approximate(frame: pd.DataFrame) -> bool:
    """Report whether the pixel counts of a membership table approximate the nucleus shape.

    Args:
        frame: A membership table a builder made.

    Returns:
        True for disks drawn around centroids, False for label maps and outlines.

    Raises:
        MembershipError: If the table records no provenance, not having been built by a builder.
    """
    if _ATTR_APPROXIMATE not in frame.attrs:
        raise MembershipError(
            "This table records no provenance, so whether its pixel counts approximate the "
            "nucleus shape is unknown. Build it with one of the membership builders."
        )
    return bool(frame.attrs[_ATTR_APPROXIMATE])


def _membership_provenance(frame: pd.DataFrame) -> tuple[str, bool | None, int | None]:
    """Return where a membership table came from, whether it approximates, and what it dropped."""
    source = membership_source(frame)
    try:
        approximate: bool | None = membership_is_approximate(frame)
    except MembershipError:
        # A table assembled by hand records no provenance: unknown, which is not the same as no.
        approximate = None
    try:
        without_tokens: int | None = int(missing_cells(frame).size)
    except MembershipError:
        without_tokens = None
    return source, approximate, without_tokens


def _with_nuclei_lost(note: str, frame: pd.DataFrame, without_tokens: int | None) -> str:
    """Add to a crop note how many of the nuclei given reached no token, whatever lost them."""
    given = frame.attrs.get(_ATTR_INPUT_CELLS)
    if without_tokens is None or given is None:
        return note
    if without_tokens == 0:
        return f"{note} Each of the {int(given)} nuclei given reached a token."
    verb = "has" if without_tokens == 1 else "have"
    return (
        f"{note} {without_tokens} of the {int(given)} nuclei given reached no token and {verb} "
        "no row in the result."
    )


def _segment_rows(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return the positions of consecutive runs of rows, each given by its start and length."""
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.zeros(counts.shape[0], dtype=np.int64)
    offsets[1:] = np.cumsum(counts)[:-1]
    shift = np.repeat(np.asarray(starts, dtype=np.int64) - offsets, counts)
    return shift + np.arange(total, dtype=np.int64)


def _concatenate(parts: Sequence[np.ndarray], dtype) -> np.ndarray:
    if not len(parts):
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts)


def _factorize_cells(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map cell identifiers to integer codes ordered by identifier."""
    try:
        codes, uniques = pd.factorize(values, sort=True)
    except TypeError as error:
        raise MembershipError(
            "Cell identifiers must be comparable with one another, so that ties resolve in one "
            "order. Name every cell by an integer, or every cell by text."
        ) from error
    return np.asarray(codes, dtype=np.int64), np.asarray(uniques)


def _found_in(values: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Return which of some cell identifiers a pool holds, as `==` compares them."""
    held = np.asarray(values)
    candidates = np.asarray(pool)
    # numpy.isin compares Python objects pair by pair, so those go through a hash table.
    if held.dtype != object and candidates.dtype != object:
        return np.isin(held, candidates)
    if held.size == 0:
        return np.zeros(held.shape, dtype=bool)
    # An index of objects holds text as it is given, rather than converted to a string dtype.
    return np.asarray(pd.Index(held, dtype=object).isin(candidates), dtype=bool)


def _columns_of(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the columns of a table without its attributes, which pandas would deep-copy."""
    return pd.DataFrame(frame, copy=False)


class _CellPositions(Mapping):
    """The position of each cell by identifier, never changed, so copies of a table share it."""

    def __init__(self, identifiers: Any, points: Any) -> None:
        self._index = pd.Index(np.asarray(identifiers))
        self._points = np.array(points, dtype=np.float64).reshape(-1, 2)
        self._points.flags.writeable = False

    def __getitem__(self, identifier: Any) -> list[float]:
        return self._points[self._index.get_loc(identifier)].tolist()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._index.tolist())

    def __len__(self) -> int:
        return len(self._index)

    def __deepcopy__(self, memo: dict) -> _CellPositions:
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} cells)"

    def points_of(self, identifiers: Any) -> np.ndarray:
        """Return the position of each cell named, NaN where none is recorded."""
        rows = self._index.get_indexer(np.asarray(identifiers))
        points = np.full((rows.shape[0], 2), np.nan, dtype=np.float64)
        points[rows >= 0] = self._points[rows[rows >= 0]]
        return points


def _build_membership(
    tiles: np.ndarray,
    cells: np.ndarray,
    tokens: np.ndarray,
    pixels: np.ndarray,
    supplied: np.ndarray,
    source: str,
    approximate: bool,
    fitted_at: tuple[float | None, float],
    grids: Sequence[tuple[int, int, int]],
    tile_centers: np.ndarray | None,
    cell_centers: _CellPositions | None,
    positions_frame: str | None,
    positions_mpp: float | None,
) -> pd.DataFrame:
    """Assemble a membership table, recording the missing cells, the geometry and positions."""
    frame = pd.DataFrame(
        {
            "tile_index": np.asarray(tiles, dtype=np.int32),
            "cell_id": cells,
            "token_index": np.asarray(tokens, dtype=np.int32),
            "nucleus_px": np.asarray(pixels, dtype=np.int32),
        }
    )
    given = pd.unique(np.asarray(supplied))
    covered = pd.unique(frame["cell_id"].to_numpy())
    unexpected = covered[~_found_in(covered, given)]
    if unexpected.size:
        raise MembershipError(
            f"{unexpected.size} cells reached a token without being given as input, so the "
            "table cannot partition the cells it was built from."
        )
    missing = given[~_found_in(given, covered)]

    frame.attrs[_ATTR_SOURCE] = source
    frame.attrs[_ATTR_APPROXIMATE] = bool(approximate)
    # Sorted as pooling orders cells, integers before text where both name cells.
    frame.attrs[_ATTR_MISSING] = _factorize_cells(missing)[1]
    frame.attrs[_ATTR_INPUT_CELLS] = int(given.size)
    mpp, target_mpp = fitted_at
    frame.attrs[_ATTR_MPP] = None if mpp is None else float(mpp)
    frame.attrs[_ATTR_TARGET_MPP] = float(target_mpp)
    frame.attrs[_ATTR_GRIDS] = np.unique(np.asarray(grids, dtype=np.int64).reshape(-1, 3), axis=0)
    if tile_centers is not None:
        frame.attrs[_ATTR_TILE_CENTERS] = np.asarray(tile_centers, dtype=np.float64)
    if cell_centers is not None:
        frame.attrs[_ATTR_CELL_CENTERS] = cell_centers
    if positions_frame is not None:
        frame.attrs[_ATTR_FRAME] = str(positions_frame)
    if positions_mpp is not None:
        frame.attrs[_ATTR_FRAME_MPP] = float(positions_mpp)
    return frame


def _validate_partition(frame: pd.DataFrame) -> None:
    """Check a builder table still partitions the cells it was built from."""
    if _ATTR_MISSING not in frame.attrs or _ATTR_INPUT_CELLS not in frame.attrs:
        return
    absent = np.asarray(frame.attrs[_ATTR_MISSING])
    covered = pd.unique(_columns_of(frame)["cell_id"].to_numpy())
    if covered.size and absent.size and _found_in(covered, absent).any():
        raise MembershipError(
            "A cell is recorded both as reaching a token and as reaching none; membership and "
            "the missing set must not overlap."
        )
    expected = int(frame.attrs[_ATTR_INPUT_CELLS])
    if covered.size + absent.size != expected:
        raise MembershipError(
            f"{covered.size} cells hold tokens and {absent.size} hold none, against "
            f"{expected} cells given. Membership and the missing set must partition the input."
        )
