"""The encode command."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from alhena._defaults import _batch_size
from alhena._errors import PACKAGE_PREFIXES, AlhenaValueError, GeometryWarning
from alhena._layout import CELLS_SUFFIX, TILE_EMBEDDING_KEY
from alhena.cli._reading import _NUCLEI_READING, _open_tiles, _read_nuclei, _require_suffix
from alhena.cli._shared import (
    _NPY_SUFFIX,
    _UNDECLARED_MPP,
    CELLS_SUFFIXES,
    EMBEDDING_SUFFIXES,
    _field,
    _file_size,
    _geometry_record,
    _given,
    _no_token_note,
    _note,
    _print_resolution,
    _progress_of,
    _reported_warnings,
    _write_embeddings,
    geometry_path,
)

if TYPE_CHECKING:
    from alhena.types import TileEmbeddings

# Options that describe the cells of nuclei, which encode takes only beside --nuclei.
_CELL_OPTIONS: tuple[tuple[str, str], ...] = (
    *_NUCLEI_READING,
    ("--all-tiles", "all_tiles"),
    ("--tile-embeddings", "tile_embeddings"),
)


def _encode(args: argparse.Namespace) -> int:
    """Embed tiles, or the cells of their nuclei, and write the embeddings out."""
    from alhena.encoder._inference import _resolve_device

    # Before any read, so a batch size or a device out of range costs no reading.
    _batch_size(args.batch_size)
    if args.device is not None:
        _resolve_device(args.device)
    if args.nuclei is not None:
        return _encode_cells(args)
    given = _given(args, _CELL_OPTIONS)
    if given:
        one = len(given) == 1
        raise AlhenaValueError(
            f"{', '.join(given)} {'describes' if one else 'describe'} the cells of a set of "
            "nuclei, and no --nuclei was given. Pass --nuclei to write cells, or drop "
            f"{'that option' if one else 'those options'} to write tile embeddings."
        )
    return _encode_tiles(args)


def _encode_tiles(args: argparse.Namespace) -> int:
    """Embed tiles with the backbone and write the embeddings out."""
    from alhena.api import load_encoder
    from alhena.preprocess import UNDECLARED_RESOLUTION_WARNING

    progress = _progress_of(args)
    suffix = _require_suffix(args.out, EMBEDDING_SUFFIXES, "Tile embeddings are written")
    with _reported_warnings(collapse=(UNDECLARED_RESOLUTION_WARNING,)), _open_tiles(args) as source:
        if source.mpp is None:
            warnings.warn(_UNDECLARED_MPP, GeometryWarning, skip_file_prefixes=PACKAGE_PREFIXES)
        encoder = load_encoder(
            backbone=args.backbone,
            bundle=args.bundle,
            device=args.device,
            cache_dir=args.cache_dir,
            progress=progress,
        )
        # Read one batch at a time, so a source larger than memory is embedded, not loaded.
        tiles = encoder.encode_tiles(source, batch_size=args.batch_size, progress=progress)
    plan = tiles.geometry
    assert plan is not None
    record = _geometry_record(plan, encoder.info.encoder_id, tiles.embeddings.shape)
    _write_embeddings(tiles, args.out, record)

    _field("encoder id", encoder.info.encoder_id)
    _print_resolution(source)
    _print_fit(plan.pad_fraction, plan.crop_fraction, source.mpp is not None)
    _field(
        "embeddings",
        f"{tiles.embeddings.shape[0]} "
        f"{'tile' if tiles.embeddings.shape[0] == 1 else 'tiles'} "
        f"x {tiles.embeddings.shape[1]} dimensions",
    )
    _print_windows(tiles)
    _field("written", f"{args.out} ({_file_size(args.out)})")
    _note_tile_format(args.out, suffix)
    return 0


def _print_windows(tiles: TileEmbeddings) -> None:
    """Report where the tiles sit, and say so plainly where they sit nowhere."""
    if tiles.bounds is None:
        _field("tile bounds", "unavailable (tile origins not provided)")
        return
    _field("tile bounds", f"in the frame of {tiles.bounds_frame}")


def _print_fit(pad_fraction: float | None, crop_fraction: float | None, declared: bool) -> None:
    """Report the largest pad and crop of a tile, `unknown` where no resolution was declared."""
    known = declared and pad_fraction is not None and crop_fraction is not None
    _field("white pad", f"{pad_fraction:.1%} of the canvas" if known else "unknown")
    _field("cropped away", f"{crop_fraction:.1%} of the field" if known else "unknown")
    if known:
        _note("Both percentages are the maximum observed across tiles.")


def _note_tile_format(out: Path, suffix: str) -> None:
    """Say what a file of tile embeddings carries beside the array, by its format."""
    if suffix == _NPY_SUFFIX:
        _note(
            f"A .npy file stores only the embedding matrix; use .h5 or {CELLS_SUFFIX} "
            "to retain tile IDs."
        )
        _note(f"Geometry is written beside it, in {geometry_path(out).name}.")
    elif suffix == CELLS_SUFFIX:
        _note(
            f'One row per tile, with embeddings in AnnData .obsm["{TILE_EMBEDDING_KEY}"], '
            "geometry, and tile bounds when available."
        )
        _note("It is read back with alhena.io.read_tile_embeddings.")
    else:
        _note("Geometry is stored as attributes of the HDF5 file.")


def _encode_cells(args: argparse.Namespace) -> int:
    """Embed the cells of a set of nuclei in one pass over the tiles, and write them out."""
    from alhena.api import load_encoder
    from alhena.cells._path import _encode_plan, _plan_cells
    from alhena.preprocess import UNDECLARED_RESOLUTION_WARNING

    progress = _progress_of(args)
    _require_suffix(args.out, CELLS_SUFFIXES, "A table of cells is written")
    tile_out: Path | None = args.tile_embeddings
    tile_suffix = None
    if tile_out is not None:
        tile_suffix = _require_suffix(tile_out, EMBEDDING_SUFFIXES, "Tile embeddings are written")
        if tile_suffix == _NPY_SUFFIX and not args.all_tiles:
            raise AlhenaValueError(
                f"{tile_out.name} is a .npy array, which carries no tile names, and the tiles no "
                f"cell is read from are skipped. Write .h5, .hdf5 or {CELLS_SUFFIX}, which carry "
                "the names, or pass --all-tiles."
            )
    with _reported_warnings(collapse=(UNDECLARED_RESOLUTION_WARNING,)), _open_tiles(args) as source:
        if source.mpp is None:
            warnings.warn(_UNDECLARED_MPP, GeometryWarning, skip_file_prefixes=PACKAGE_PREFIXES)
        nuclei = _read_nuclei(args, source, progress)
        # Built before the encoder is loaded, so nuclei that do not fit cost no load of the weights.
        plan = _plan_cells(source, nuclei, source.mpp, all_tiles=args.all_tiles, progress=progress)
        encoder = load_encoder(
            backbone=args.backbone,
            bundle=args.bundle,
            device=args.device,
            cache_dir=args.cache_dir,
            progress=progress,
        )
        cells = _encode_plan(encoder, plan, batch_size=args.batch_size, progress=progress)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cells.write_h5ad(args.out)
    tiles = cells.tiles
    read = 0 if tiles is None else len(tiles)
    written_tiles = tile_out is not None and tiles is not None and read > 0
    if written_tiles:
        assert tiles is not None and tile_out is not None and tiles.geometry is not None
        record = _geometry_record(tiles.geometry, encoder.info.encoder_id, tiles.embeddings.shape)
        _write_embeddings(tiles, tile_out, record)

    _field("encoder id", encoder.info.encoder_id)
    _field("cell space id", encoder.info.cell_space_id)
    _print_resolution(source)
    _field("tiles read", f"{read} of {len(source)}")
    if not args.all_tiles and read < len(source):
        _note("Tiles no cell is read from were skipped. Pass --all-tiles to read every tile.")
    usable = int(np.count_nonzero(cells.valid))
    _field("cells", f"{len(cells)} total; {usable} usable for annotation")
    if cells.n_cells_without_tokens:
        _note(_no_token_note(cells.n_cells_without_tokens))
    if cells.membership_approximate:
        _note("Nucleus shapes are approximated, so the pooling weights are approximate.")
    if cells.centroids is None:
        _field("positions", "unavailable (tile origins not provided)")
    elif cells.centroids_frame is None:
        _field("positions", "present, but coordinate frame is unspecified")
    else:
        _field("positions", f"in the frame of {cells.centroids_frame}")
    _field("written", f"{args.out} ({_file_size(args.out)})")
    if tile_out is not None:
        if not written_tiles:
            _field("tile embeddings", "none written, since no tile was read")
        else:
            assert tiles is not None and tile_suffix is not None
            _field("tile embeddings", f"{tile_out} ({_file_size(tile_out)})")
            _print_fit(tiles.pad_fraction, tiles.crop_fraction, source.mpp is not None)
            _note_tile_format(tile_out, tile_suffix)
    return 0
