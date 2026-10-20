"""Encoding cells, from reading the inputs to pooling their tokens."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from alhena import _progress
from alhena._constants import REFERENCE_MPP
from alhena._errors import PACKAGE_PREFIXES, GeometryWarning
from alhena._frames import _frame_mpp
from alhena._geometry import (
    _checked_mpp,
    _crop_note,
    _merge_records,
    _prepare,
    _resolved_mpp,
    _warn_resolution,
)
from alhena.cells._checks import (
    _CanvasFit,
    _check_nucleus_grids,
    _check_table_fit,
    _check_tile_alignment,
    _grid_mismatch,
    _nucleus_grids,
)
from alhena.cells._inputs import (
    _TABLE,
    _check_identifiers,
    _is_prepared,
    _known_tile_shapes,
    _median_nucleus_area,
    _membership_from_nuclei,
    _pixels_inside,
    _Placement,
    _read_nuclei,
    _table_on_tile_windows,
    _tile_geometry,
    _tile_names,
    _tile_placement,
)
from alhena.cells._pooling import (
    _cell_centroids,
    _check_projection,
    _pooling_rows,
    _usable,
    _weighted_sums,
    resolve_main_tile,
)
from alhena.cells._table import (
    _ATTR_CELL_CENTERS,
    _ATTR_FRAME,
    _ATTR_FRAME_MPP,
    _ATTR_TILE_CENTERS,
    _membership_provenance,
    _segment_rows,
    _with_nuclei_lost,
    validate_membership,
)
from alhena.io import tile_bounds
from alhena.preprocess import describe_geometry
from alhena.types import (
    CellEmbeddings,
    GeometryReport,
    NucleiInput,
    ResampleInfo,
    TileEmbeddings,
    TileInput,
)


@dataclass(frozen=True)
class _CellInputs:
    """Tiles and nuclei read and checked against one another, from shapes alone."""

    n_tiles: int
    prepared: bool
    mpp: float | None
    kind: str
    payload: Any
    options: dict[str, Any]
    report: GeometryReport
    placement: _Placement | None
    # The grids left to check as tiles are read, -1 elsewhere, and the fit a table leaves them.
    pending_grids: np.ndarray | None
    pending_fit: _CanvasFit | None


@dataclass(frozen=True)
class _CellPlan:
    """Everything the cell path settles before its forward pass, cells in identifier order."""

    tiles: Any
    prepared: bool
    mpp: float | None
    report: GeometryReport
    placement: _Placement | None
    # The tiles the pass reads, ascending.
    read: np.ndarray
    cell_id: np.ndarray
    tile_index: np.ndarray
    alpha: np.ndarray
    n_candidate_tiles: np.ndarray
    # The pooling rows of cell i run from row_start[i] for row_count[i], in token and weight.
    row_start: np.ndarray
    row_count: np.ndarray
    token: np.ndarray
    weight: np.ndarray
    membership_source: str
    membership_approximate: bool | None
    n_cells_without_tokens: int | None
    centroids: np.ndarray | None
    centroids_frame: str | None
    centroids_mpp: float | None
    kind: str
    pending_grids: np.ndarray | None
    pending_fit: _CanvasFit | None


def _inspect_cells(
    tiles: TileInput,
    nuclei: NucleiInput,
    mpp: float | None,
    defer_crop: bool = False,
    cell_id: Sequence[Any] | np.ndarray | None = None,
    progress: bool | None = None,
) -> _CellInputs:
    """Read the tiles and the nuclei, check them against each other, and warn about the geometry."""
    if mpp is not None:
        mpp = _checked_mpp(mpp, warn=True)
    mpp = _resolved_mpp(tiles, mpp)
    kind, payload, options = _read_nuclei(nuclei)
    n_tiles, tile_size, first_shape = _tile_geometry(tiles)
    prepared = _is_prepared(tiles, mpp)
    placement = _tile_placement(tiles)
    _check_tile_alignment(kind, payload, options, n_tiles, tiles, placement)
    _check_identifiers(kind, payload, options, cell_id)
    pending = fit = None
    if not prepared:
        shapes = _known_tile_shapes(tiles, n_tiles, first_shape)
        grids = _nucleus_grids(kind, payload, options, n_tiles, mpp, placement is not None, shapes)
        if grids is not None:
            pending = _check_nucleus_grids(tiles, kind, grids, shapes)
        if kind == _TABLE:
            # After the grids, so a table fitted at a resolution is compared on its tiles' grids.
            fit = _check_table_fit(tiles, payload, mpp, shapes)
    report = describe_geometry(
        mpp=mpp,
        tile_size_px=tile_size,
        median_nucleus_area_px=_median_nucleus_area(kind, payload, options, progress),
        target_mpp=REFERENCE_MPP,
    )
    deferred = _crop_note(report) if defer_crop else None
    for note in report.warnings:
        if note != deferred:
            warnings.warn(note, GeometryWarning, skip_file_prefixes=PACKAGE_PREFIXES)
    return _CellInputs(
        n_tiles=n_tiles,
        prepared=prepared,
        mpp=mpp,
        kind=kind,
        payload=payload,
        options=options,
        report=report,
        placement=placement,
        pending_grids=pending,
        pending_fit=fit,
    )


def _plan_cells(
    tiles: TileInput,
    nuclei: NucleiInput,
    mpp: float | None,
    *,
    cell_id: Sequence[Any] | np.ndarray | None = None,
    all_tiles: bool = False,
    progress: bool | None = None,
) -> _CellPlan:
    """Settle every cell and every tile the pass will read, before any tile is embedded."""
    inputs = _inspect_cells(tiles, nuclei, mpp, defer_crop=True, cell_id=cell_id, progress=progress)
    frame = _membership_from_nuclei(
        inputs.kind,
        inputs.payload,
        inputs.options,
        inputs.mpp,
        REFERENCE_MPP,
        cell_id,
        progress=progress,
        placement=inputs.placement,
        inside=_pixels_inside(tiles),
    )
    if inputs.kind != _TABLE:
        validate_membership(frame, inputs.n_tiles)
    if not inputs.prepared:
        _warn_resolution(inputs.mpp, REFERENCE_MPP)

    source, approximate, without_tokens = _membership_provenance(frame)
    crop_note = _crop_note(inputs.report)
    if crop_note is not None:
        warnings.warn(
            _with_nuclei_lost(crop_note, frame, without_tokens),
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
    cell_centers = frame.attrs.get(_ATTR_CELL_CENTERS)
    rows = _pooling_rows(frame)
    selected, candidates = resolve_main_tile(
        rows.groups, frame.attrs.get(_ATTR_TILE_CENTERS), cell_centers
    )
    identifiers = rows.groups["cell_id"].to_numpy()[selected]
    tile_index = rows.groups["tile_index"].to_numpy(dtype=np.int32)[selected]
    alpha = rows.groups["alpha"].to_numpy(dtype=np.float32)[selected]
    row_count = rows.lengths[selected]
    chosen = _segment_rows(rows.starts[selected], row_count)
    token = rows.token[chosen]
    weight = rows.weight[chosen]

    row_start = np.zeros(row_count.shape[0], dtype=np.int64)
    if row_count.size:
        row_start[1:] = np.cumsum(row_count)[:-1]
    if all_tiles:
        read = np.arange(inputs.n_tiles, dtype=np.int64)
    else:
        read = np.unique(tile_index).astype(np.int64)
    centroids = _cell_centroids(identifiers, cell_centers)
    named = frame.attrs.get(_ATTR_FRAME)
    placement = inputs.placement
    if inputs.kind == _TABLE and placement is not None and _table_on_tile_windows(frame, placement):
        # Placed against the windows of these tiles, so in the frame the tiles name.
        named = placement.frame
    frame_mpp = frame.attrs.get(_ATTR_FRAME_MPP)
    return _CellPlan(
        tiles=tiles,
        prepared=inputs.prepared,
        mpp=inputs.mpp,
        report=inputs.report,
        placement=inputs.placement,
        read=read,
        cell_id=identifiers,
        tile_index=tile_index,
        alpha=alpha,
        n_candidate_tiles=candidates,
        row_start=row_start,
        row_count=row_count,
        token=token,
        weight=weight,
        membership_source=source,
        membership_approximate=approximate,
        n_cells_without_tokens=without_tokens,
        centroids=centroids,
        centroids_frame=None if centroids is None or named is None else str(named),
        centroids_mpp=None if centroids is None or frame_mpp is None else float(frame_mpp),
        kind=inputs.kind,
        pending_grids=inputs.pending_grids,
        pending_fit=inputs.pending_fit,
    )


def _encode_plan(
    encoder: Any,
    plan: _CellPlan,
    *,
    batch_size: int,
    progress: bool | None,
) -> CellEmbeddings:
    """Embed the tiles a plan reads, one batch at a time, pooling the cells of each batch."""
    n_cells = int(plan.cell_id.shape[0])
    embed_dim = int(encoder.info.embed_dim)
    # Projecting no cell first refuses an encoder without a cell head before the forward pass.
    probe = np.asarray(encoder.project_cell(np.zeros((0, embed_dim), dtype=np.float32)))
    _check_projection(probe, 0)
    backbone = np.zeros((n_cells, embed_dim), dtype=np.float32)
    cell_space = np.zeros((n_cells, int(probe.shape[1])), dtype=np.float32)
    tile_rows = np.zeros((plan.read.shape[0], embed_dim), dtype=np.float32)

    # Cells by tile: the cells of one batch, a run of the tiles read, are a run of this order.
    by_tile = np.argsort(plan.tile_index, kind="stable")
    tile_of_cell = plan.tile_index[by_tile]

    records: list[ResampleInfo] = []
    n_read = int(plan.read.shape[0])
    # The height and width of each tile read, as the pass reads it.
    sizes = np.zeros((n_read, 2), dtype=np.float64)
    with _progress.bar(n_read, "Encoding tiles", "tile", progress) as bar:
        for start in range(0, n_read, batch_size):
            stop = min(start + batch_size, n_read)
            record, shapes = _encode_batch(
                encoder,
                plan,
                slice(start, stop),
                by_tile,
                tile_of_cell,
                backbone,
                cell_space,
                tile_rows,
            )
            if record is not None:
                records.append(record)
            if shapes is not None:
                sizes[start:stop] = shapes
            bar.update(stop - start)

    placement = plan.placement
    tiles = TileEmbeddings(
        embeddings=tile_rows,
        tile_index=plan.read.copy(),
        tile_ids=_tile_names(plan.tiles, plan.read),
        geometry=_merge_records(records),
        encoder_id=str(encoder.info.encoder_id),
        # They cover what the cells need, not a sampling of the image, and say so.
        purpose="cell_context",
        bounds=_source_windows(plan, sizes),
        bounds_frame=None if placement is None else placement.frame,
        bounds_mpp=None if placement is None else _frame_mpp(plan.mpp, placement.scale),
        image_shape=getattr(plan.tiles, "image_shape", None),
        backbone_digest=encoder.info.backbone_digest,
    )
    return CellEmbeddings(
        cell_id=plan.cell_id,
        backbone=backbone,
        cell_space=cell_space,
        alpha=plan.alpha,
        tile_index=plan.tile_index,
        n_candidate_tiles=plan.n_candidate_tiles,
        valid=_usable(backbone, plan.alpha),
        encoder_id=str(encoder.info.encoder_id),
        cell_space_id=str(encoder.info.cell_space_id),
        geometry=plan.report,
        membership_source=plan.membership_source,
        membership_approximate=plan.membership_approximate,
        n_cells_without_tokens=plan.n_cells_without_tokens,
        centroids=plan.centroids,
        centroids_frame=plan.centroids_frame,
        centroids_mpp=plan.centroids_mpp,
        tiles=tiles,
        backbone_digest=encoder.info.backbone_digest,
    )


def _encode_batch(
    encoder: Any,
    plan: _CellPlan,
    window: slice,
    by_tile: np.ndarray,
    tile_of_cell: np.ndarray,
    backbone: np.ndarray,
    cell_space: np.ndarray,
    tile_rows: np.ndarray,
) -> tuple[ResampleInfo | None, np.ndarray | None]:
    """Embed a batch of the tiles read and pool its cells; return its record and its tile sizes."""
    indices = plan.read[window]
    batch, record, shapes = _prepare_tiles_at(plan, indices)
    pooled_tiles, tokens = encoder.encode_with_tokens(
        batch, batch_size=int(indices.shape[0]), progress=False
    )
    tile_rows[window] = pooled_tiles

    first = np.searchsorted(tile_of_cell, indices[0], side="left")
    last = np.searchsorted(tile_of_cell, indices[-1], side="right")
    members = by_tile[first:last]
    if members.size == 0:
        return record, shapes
    local = np.searchsorted(indices, plan.tile_index[members])
    pooled = _weighted_sums(
        tokens, local, plan.row_start[members], plan.row_count[members], plan.token, plan.weight
    )
    backbone[members] = pooled
    projected = np.asarray(encoder.project_cell(pooled))
    _check_projection(projected, members.shape[0])
    cell_space[members] = projected
    return record, shapes


def _prepare_tiles_at(
    plan: _CellPlan, indices: np.ndarray
) -> tuple[torch.Tensor, ResampleInfo | None, np.ndarray | None]:
    """Read and prepare the tiles at some indices, checking unknown shapes; return their sizes."""
    tiles = plan.tiles
    if isinstance(tiles, torch.Tensor):
        selected = tiles.index_select(0, torch.as_tensor(indices, dtype=torch.long)).detach()
        if plan.prepared:
            return selected.to(dtype=torch.float32), None, None
        batch, record = _prepare(selected.cpu().numpy(), plan.mpp, REFERENCE_MPP, progress=False)
        return batch, record, None
    read = [tiles[int(index)] for index in indices]
    shapes = [tuple(np.shape(tile)[:2]) for tile in read]
    if plan.pending_grids is not None or plan.pending_fit is not None:
        for index, shape in zip(indices.tolist(), shapes, strict=True):
            if plan.pending_grids is not None:
                grid = plan.pending_grids[index]
                if grid[0] >= 0 and tuple(grid.tolist()) != shape:
                    name = _tile_names(tiles, np.array([index]))[0]
                    raise _grid_mismatch(plan.kind, name, grid, shape)
            if plan.pending_fit is not None:
                plan.pending_fit.check(tiles, index, shape)
    batch, record = _prepare(read, plan.mpp, REFERENCE_MPP, progress=False)
    return batch, record, np.asarray(shapes, dtype=np.float64)


def _source_windows(plan: _CellPlan, sizes: np.ndarray) -> np.ndarray | None:
    """Return the window of each tile read, from its size as read, or None without origins."""
    if plan.placement is None:
        return None
    shapes = np.full((len(plan.tiles), 2), np.nan, dtype=np.float64)
    shapes[plan.read] = sizes
    return tile_bounds(plan.tiles, tile_shapes=shapes)[plan.read]
