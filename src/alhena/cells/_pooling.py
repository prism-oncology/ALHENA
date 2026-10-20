"""Pooling token embeddings into cell embeddings."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from alhena._constants import INPUT_SIZE, NUM_TOKENS, POOL_EPS
from alhena._errors import GeometryError, MembershipError
from alhena.cells._table import (
    _ATTR_CELL_CENTERS,
    _ATTR_FRAME,
    _ATTR_FRAME_MPP,
    _ATTR_TILE_CENTERS,
    _CellPositions,
    _columns_of,
    _factorize_cells,
    _membership_provenance,
    _segment_rows,
    validate_membership,
)
from alhena.types import CellEmbeddings

GROUP_COLUMNS: tuple[str, ...] = ("tile_index", "cell_id", "alpha")
"""Columns of the table of (tile, cell) groups `pool_tokens` returns, in order."""


def pool_tokens(
    tokens: np.ndarray | torch.Tensor,
    membership: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Pool tile tokens into one vector per (tile, cell) group.

    A token weighs its nucleus pixel count over `alpha + 1e-6`, `alpha` being the nucleus pixels of
    the cell in that tile, so the weights sum to `alpha / (alpha + 1e-6)`. The weights of a tile
    form one matrix, so the result does not depend on the order of the rows.

    Args:
        tokens: An `(n_tiles, 256, dim)` array or tensor of post-LayerNorm patch tokens, without the
            class and register tokens.
        membership: A table with `MEMBERSHIP_COLUMNS`.

    Returns:
        The pooled vectors, `(n_groups, dim)` float32, and a table with `GROUP_COLUMNS` ordered by
        cell identifier then tile index.

    Raises:
        GeometryError: If the tokens do not carry the token grid of the canvas.
        MembershipError: If the membership table is malformed.
    """
    array = _as_token_array(tokens)
    validate_membership(membership, n_tiles=array.shape[0])
    rows = _pooling_rows(membership)
    return rows.pooled(array, np.arange(len(rows.groups))), rows.groups


def _weighted_sums(
    tokens: np.ndarray,
    tile: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    token: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Sum the weighted token rows of each run of rows, one matrix product per tile."""
    pooled = np.zeros((starts.shape[0], tokens.shape[2]), dtype=np.float32)
    by_tile = np.argsort(tile, kind="stable")
    for runs in np.split(by_tile, np.flatnonzero(np.diff(tile[by_tile])) + 1):
        if not runs.size:
            continue
        source = tokens[tile[runs[0]]]
        rows = _segment_rows(starts[runs], lengths[runs])
        weights = np.zeros((runs.size, tokens.shape[1]), dtype=np.float32)
        weights[np.repeat(np.arange(runs.size), lengths[runs]), token[rows]] = weight[rows]
        # A zero weight times a token that is not finite spoils a sum, so those are summed apart.
        with np.errstate(invalid="ignore"):
            sums = weights @ source
            for position in np.flatnonzero(~np.isfinite(sums).all(axis=1)).tolist():
                run = runs[position]
                own = slice(starts[run], starts[run] + lengths[run])
                sums[position] = weight[own] @ source[token[own]]
        pooled[runs] = sums
    return pooled


@dataclass(frozen=True)
class _PoolingRows:
    """Rows sorted by cell, tile and token, one run per group of a cell and a tile."""

    token: np.ndarray
    weight: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    groups: pd.DataFrame

    def pooled(self, tokens: np.ndarray, chosen: np.ndarray) -> np.ndarray:
        """Return the pooled vector of each group chosen, given by its row in `groups`."""
        tiles = self.groups["tile_index"].to_numpy(dtype=np.int64)[chosen]
        return _weighted_sums(
            tokens, tiles, self.starts[chosen], self.lengths[chosen], self.token, self.weight
        )


def _pooling_rows(membership: pd.DataFrame) -> _PoolingRows:
    """Order the rows of a membership table for pooling and weigh each of them."""
    table = _columns_of(membership)
    tile = table["tile_index"].to_numpy(dtype=np.int64)
    token = table["token_index"].to_numpy(dtype=np.int64)
    pixels = table["nucleus_px"].to_numpy().astype(np.float32, copy=False)
    codes, values = _factorize_cells(table["cell_id"].to_numpy())

    order = np.lexsort((token, tile, codes))
    sorted_codes = codes[order]
    sorted_tile = tile[order]
    boundary = np.ones(order.shape[0], dtype=bool)
    boundary[1:] = (sorted_codes[1:] != sorted_codes[:-1]) | (sorted_tile[1:] != sorted_tile[:-1])
    starts = np.nonzero(boundary)[0]
    lengths = np.diff(np.append(starts, order.shape[0]))

    sorted_pixels = pixels[order]
    alpha = np.add.reduceat(sorted_pixels, starts).astype(np.float32, copy=False)
    weights = sorted_pixels / (np.repeat(alpha, lengths) + np.float32(POOL_EPS))

    groups = pd.DataFrame(
        {
            "tile_index": sorted_tile[starts].astype(np.int32, copy=False),
            "cell_id": values[sorted_codes[starts]],
            "alpha": alpha,
        }
    )
    return _PoolingRows(
        token=token[order],
        weight=weights,
        starts=starts,
        lengths=lengths,
        groups=groups,
    )


def resolve_main_tile(
    groups: pd.DataFrame,
    tile_centers: np.ndarray | None = None,
    cell_centers: Mapping[Any, Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the tile each cell is read from when tiles overlap.

    A cell is read from the tile holding most of its nucleus, then from the tile whose center its
    centroid is nearest, then from the lowest tile index, then the lowest row. Without a shared
    frame, a cell found in several tiles is refused.

    Args:
        groups: A table with `GROUP_COLUMNS`, one row per (tile, cell) group.
        tile_centers: An `(n_tiles, 2)` array of tile centers in the shared frame.
        cell_centers: Nucleus centroid of each cell in the shared frame, by identifier.

    Returns:
        The selected row of `groups` for each cell, and how many tiles held it, by cell identifier.

    Raises:
        MembershipError: If a cell is found in several tiles and no center places it.
    """
    if tuple(groups.columns) != GROUP_COLUMNS:
        raise MembershipError(
            f"A group table has columns {GROUP_COLUMNS}, got {tuple(groups.columns)}."
        )
    n_rows = len(groups)
    if n_rows == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)

    codes, values = _factorize_cells(groups["cell_id"].to_numpy())
    tile = groups["tile_index"].to_numpy(dtype=np.int64)
    alpha = groups["alpha"].to_numpy(dtype=np.float64)
    counts = np.bincount(codes, minlength=values.shape[0])

    if int(counts.max()) > 1:
        dist2 = _center_distances(codes, values, tile, tile_centers, cell_centers, counts)
    else:
        dist2 = np.zeros(n_rows, dtype=np.float64)

    row_index = np.arange(n_rows, dtype=np.int64)
    order = np.lexsort((row_index, tile, dist2, -alpha, codes))
    sorted_codes = codes[order]
    first = np.empty(n_rows, dtype=bool)
    first[0] = True
    first[1:] = sorted_codes[1:] != sorted_codes[:-1]
    selected = order[first]
    return selected.astype(np.int64, copy=False), counts[sorted_codes[first]].astype(np.int32)


def pool_cells(
    tokens: np.ndarray | torch.Tensor,
    membership: pd.DataFrame,
    tile_centers: np.ndarray | None = None,
    cell_centers: Mapping[Any, Sequence[float]] | None = None,
    project: Callable[[np.ndarray], np.ndarray] | None = None,
) -> CellEmbeddings:
    """Pool tile tokens into one embedding per cell, resolving cells found in several tiles.

    The result holds one row per cell that reached a token, by identifier, with the provenance the
    membership table records. The encoder, the cell space and the geometry are not visible from
    tokens and are left None; `AlhenaEncoder.encode_cells` records them.

    Args:
        tokens: An `(n_tiles, 256, dim)` array or tensor of post-LayerNorm patch tokens.
        membership: A table with `MEMBERSHIP_COLUMNS`.
        tile_centers: An `(n_tiles, 2)` array of tile centers in a shared frame. Defaults to what
            the table recorded.
        cell_centers: Nucleus centroids in that frame, by identifier. Defaults to what the table
            recorded, with the name and scale of its frame; centroids given here carry neither.
        project: The image-side cell head, applied to the pooled pre-projection cell embeddings.

    Returns:
        One embedding per cell, with the nucleus pixels supporting it, the tile it was read from,
        how many tiles held it, and whether it may be used.

    Raises:
        GeometryError: If the tokens do not carry the token grid of the canvas.
        MembershipError: If the table is malformed, or a cell in several tiles has no center.
    """
    array = _as_token_array(tokens)
    validate_membership(membership, n_tiles=array.shape[0])
    rows = _pooling_rows(membership)
    groups = rows.groups
    if tile_centers is None:
        tile_centers = membership.attrs.get(_ATTR_TILE_CENTERS)
    frame = frame_mpp = None
    if cell_centers is None:
        cell_centers = membership.attrs.get(_ATTR_CELL_CENTERS)
        frame = membership.attrs.get(_ATTR_FRAME)
        frame_mpp = membership.attrs.get(_ATTR_FRAME_MPP)

    selected, candidates = resolve_main_tile(groups, tile_centers, cell_centers)
    # Each cell is pooled from the tile it is read from alone, as the cell path pools it.
    backbone = rows.pooled(array, selected)
    alpha = groups["alpha"].to_numpy(dtype=np.float32)[selected]
    valid = _usable(backbone, alpha)

    cell_space = None
    if project is not None:
        cell_space = np.asarray(project(backbone))
        _check_projection(cell_space, backbone.shape[0])

    identifiers = groups["cell_id"].to_numpy()[selected]
    source, approximate, without_tokens = _membership_provenance(membership)
    centroids = _cell_centroids(identifiers, cell_centers)
    return CellEmbeddings(
        cell_id=identifiers,
        backbone=backbone,
        cell_space=cell_space,
        alpha=alpha,
        tile_index=groups["tile_index"].to_numpy(dtype=np.int32)[selected],
        n_candidate_tiles=candidates,
        valid=valid,
        membership_source=source,
        membership_approximate=approximate,
        n_cells_without_tokens=without_tokens,
        centroids=centroids,
        centroids_frame=None if centroids is None or frame is None else str(frame),
        centroids_mpp=None if centroids is None or frame_mpp is None else float(frame_mpp),
    )


def _usable(backbone: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return which pooled cells may be annotated: finite, of nonzero length, pooled from pixels."""
    return np.isfinite(backbone).all(axis=1) & (np.linalg.norm(backbone, axis=1) > 0) & (alpha > 0)


def _check_projection(cell_space: np.ndarray, n_cells: int) -> None:
    """Refuse a cell-head output that is not one nonzero-width row per cell."""
    if cell_space.ndim != 2 or cell_space.shape[0] != n_cells:
        raise GeometryError(
            f"The cell head returned shape {cell_space.shape} for {n_cells} cells; one row per "
            "cell is required."
        )
    if cell_space.shape[1] == 0:
        raise GeometryError(
            "The cell head returned zero-width embeddings, which would retrieve nothing."
        )


def _cell_centroids(
    identifiers: np.ndarray, cell_centers: Mapping[Any, Sequence[float]] | None
) -> np.ndarray | None:
    """Return the centroid of each cell in the shared frame, NaN where none was recorded."""
    if cell_centers is None:
        return None
    return _positions_of(cell_centers, identifiers)


def _positions_of(
    cell_centers: Mapping[Any, Sequence[float]], identifiers: np.ndarray
) -> np.ndarray:
    """Return the position each cell is recorded at, NaN where none is."""
    if isinstance(cell_centers, _CellPositions):
        return cell_centers.points_of(identifiers)
    points = np.full((identifiers.shape[0], 2), np.nan, dtype=np.float64)
    for position, identifier in enumerate(identifiers):
        point = cell_centers.get(identifier)
        if point is not None:
            points[position] = np.asarray(point, dtype=np.float64)
    return points


def _as_token_array(tokens: np.ndarray | torch.Tensor) -> np.ndarray:
    array = tokens.detach().cpu().numpy() if hasattr(tokens, "detach") else tokens
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3:
        raise GeometryError(
            f"Expected tokens of shape (n_tiles, {NUM_TOKENS}, dim), got {array.shape}."
        )
    if array.shape[1] != NUM_TOKENS:
        raise GeometryError(
            f"Tokens carry {array.shape[1]} positions per tile against the {NUM_TOKENS} of the "
            f"{INPUT_SIZE} pixel canvas. Drop the class and register tokens, and pin the input "
            "size, before pooling."
        )
    if array.shape[2] == 0:
        raise GeometryError("Tokens have width zero and carry nothing to pool.")
    return array


def _center_distances(
    codes: np.ndarray,
    values: np.ndarray,
    tile: np.ndarray,
    tile_centers: np.ndarray | None,
    cell_centers: Mapping[Any, Sequence[float]] | None,
    counts: np.ndarray,
) -> np.ndarray:
    """Return how far the nucleus of each row lies from its tile's center, for shared cells."""
    if tile_centers is None or cell_centers is None:
        raise MembershipError(
            f"A cell was found in {int(counts.max())} tiles, and choosing one needs tile centers "
            "and nucleus centroids in one shared coordinate frame. Read the tiles from a source "
            "carrying their origins, pass tile_origins to the builder, or pass tile_centers and "
            "cell_centers; if instance numbers restart in each tile, make them unique first: "
            "io.read_label_maps renumbers them."
        )
    centers = np.asarray(tile_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise MembershipError(f"Expected tile_centers of shape (n_tiles, 2), got {centers.shape}.")
    if int(tile.max()) >= centers.shape[0]:
        raise MembershipError(
            f"tile_centers holds {centers.shape[0]} rows but tile index {int(tile.max())} is "
            "used; one center per tile is required."
        )
    # Only the cells found in several tiles are placed, by the centroids of their nuclei.
    shared = np.flatnonzero(counts > 1)
    nuclei = _positions_of(cell_centers, values[shared])
    missing = np.flatnonzero(np.isnan(nuclei).any(axis=1))
    if missing.size:
        first = int(shared[missing[0]])
        raise MembershipError(
            f"cell_centers has no centroid for cell {values[first : first + 1].tolist()[0]!r}, "
            f"which was found in {int(counts[first])} tiles."
        )
    placed = np.full(values.shape[0], -1, dtype=np.int64)
    placed[shared] = np.arange(shared.shape[0])
    rows = np.flatnonzero(placed[codes] >= 0)
    delta = nuclei[placed[codes[rows]]] - centers[tile[rows]]
    distances = np.zeros(codes.shape[0], dtype=np.float64)
    distances[rows] = (delta * delta).sum(axis=1)
    return distances
