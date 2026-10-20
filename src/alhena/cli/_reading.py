"""Reading the tiles and the nuclei a command names."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from alhena._errors import AlhenaError, AlhenaValueError
from alhena.cli._shared import _NPY_SUFFIX
from alhena.io import (
    H5_BARCODE_DATASET,
    H5_COORDS_DATASET,
    H5_IMAGE_DATASET,
    IMAGE_SUFFIXES,
    SLIDE_SUFFIXES,
)
from alhena.io._labels import _is_label_image
from alhena.io._nuclei import _nuclei_kind

if TYPE_CHECKING:
    from alhena.io import TileSource

NUCLEI_ID_SCOPES: tuple[str, ...] = ("per_tile", "cohort")
"""Readings of label values: unique within a tile and then renumbered, or unique in the cohort."""

# Datasets --h5-datasets renames, by default name, each mirroring an argument of alhena.io.
_H5_DATASET_ARGUMENTS: dict[str, str] = {
    H5_IMAGE_DATASET: "image_dataset",
    H5_BARCODE_DATASET: "barcode_dataset",
    H5_COORDS_DATASET: "coords_dataset",
}

# Options reading --nuclei: flag, attribute, the io argument it mirrors, the kinds it applies to.
_NUCLEI_OPTIONS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("--radius-px", "radius_px", "radius_px", ("centroids",)),
    ("--nuclei-frame", "nuclei_frame", "frame", ("centroids",)),
    ("--nuclei-units", "nuclei_units", "units", ("centroids", "polygons")),
    ("--nuclei-transform", "nuclei_transform", "transform", ("centroids", "polygons")),
    ("--nuclei-columns", "nuclei_columns", "columns", ("centroids", "polygons")),
    ("--nuclei-id-property", "nuclei_id_property", "id_property", ("polygons",)),
    ("--nuclei-geometry-property", "nuclei_geometry_property", "geometry_property", ("polygons",)),
    ("--nuclei-multipart", "nuclei_multipart", "multipart", ("polygons",)),
)

# Every option describing how the nuclei are read, by the name the command line spells it.
_NUCLEI_READING: tuple[tuple[str, str], ...] = (
    ("--nuclei-kind", "nuclei_kind"),
    ("--nuclei-ids", "nuclei_ids"),
    *((flag, attribute) for flag, attribute, _, _ in _NUCLEI_OPTIONS),
)


def _id_scope(args: argparse.Namespace) -> str:
    """Return how label values name cells, within their tile by default."""
    return NUCLEI_ID_SCOPES[0] if args.nuclei_ids is None else str(args.nuclei_ids)


def _require_suffix(path: Path, allowed: tuple[str, ...], written: str) -> str:
    """Return the suffix of a path, refusing a format that is not written."""
    suffix = path.suffix.lower()
    if suffix not in allowed:
        raise AlhenaValueError(
            f"{written} as {' or '.join(allowed)}, and {path.name} names "
            f"{suffix if suffix else 'no suffix'}."
        )
    return suffix


# Closing of the readers' refusal for columns they cannot find, reworded to name the option.
_PYTHON_COLUMNS: str = "Name it through columns=."


@contextmanager
def _columns_named_by(flag: str, example: str) -> Iterator[None]:
    """Reword the column refusals a block raises to name the option `flag`, not `columns=`."""
    try:
        yield
    except AlhenaValueError as error:
        text = str(error)
        worded = text.replace(_PYTHON_COLUMNS, f"Name it through {flag}, as {example}.")
        worded = worded.replace("columns['tile_index']", f"{flag} tile_index=NAME")
        worded = re.sub(r"(?<![\w.-])columns=", flag, worded)
        worded = re.sub(r"(?<![\w.'-])columns names", f"{flag} names", worded)
        if worded == text:
            raise
        raise AlhenaValueError(worded) from error


def _tile_source(
    path: Path,
    mpp: float | None = None,
    tile_px: int | None = None,
    stride_px: int | None = None,
    coords_scale: float | None = None,
    coords: Path | None = None,
    edge: str | None = None,
    datasets: Sequence[str] | None = None,
    coords_columns: Sequence[str] | None = None,
    region: Sequence[float] | None = None,
) -> TileSource:
    """Open the tiles a path holds, without reading them, as the tile options describe them."""
    from alhena.io import (
        HDF5_SUFFIXES,
        read_image,
        read_slide,
        read_tile_origins,
        read_tiles,
    )

    if not path.exists():
        raise AlhenaValueError(f"No tiles at {path}.")
    named = _dataset_names(datasets)
    columns = (
        _named_pairs(coords_columns, "--coords-columns", "FIELD=COLUMN", "x=top_left_x")
        if coords_columns is not None
        else None
    )
    if columns is not None and coords is None:
        raise AlhenaValueError(
            "--coords-columns names the columns of the table --coords is, and --coords names no "
            "table."
        )
    # A directory holds images whatever its name ends in.
    suffix = "" if path.is_dir() else path.suffix.lower()
    if _cut_from_one_image(path):
        if coords_scale is not None:
            raise AlhenaValueError(
                f"{path.name} places its own tiles: their origins are recorded in the pixels of "
                "its full-resolution level, and their scale is the downsample of the level they "
                "are read at. Drop --coords-scale."
            )
        options: dict[str, Any] = {"stride_px": stride_px, "mpp": mpp}
        if edge is not None:
            options["edge"] = edge
        if coords is not None:
            options["coords"] = coords
        if columns is not None:
            options["columns"] = columns
        if region is not None:
            options["region"] = region
        if named:
            options.update(_origins_dataset(named, path, coords))
        # A grid in which no tile fits is refused, so none comes back empty.
        with _columns_named_by("--coords-columns", "x=NAME y=NAME"):
            if suffix in SLIDE_SUFFIXES:
                return read_slide(path, tile_px=tile_px, **options)
            return read_image(path, tile_px=tile_px, **options)
    cutting = [
        flag
        for flag, value in (
            ("--tile-px", tile_px),
            ("--stride-px", stride_px),
            ("--edge", edge),
            ("--region", region),
        )
        if value is not None
    ]
    if cutting:
        raise AlhenaValueError(
            f"{' and '.join(cutting)} {'describes' if len(cutting) == 1 else 'describe'} how an "
            f"image is cut into tiles, and the tiles at {path.name} arrive cut: they are read as "
            f"they are stored. Drop {'it' if len(cutting) == 1 else 'them'}, or name the image "
            "they were cut from as the tiles to read."
        )
    if coords is not None and suffix in HDF5_SUFFIXES:
        raise AlhenaValueError(
            f"{path.name} is the sole record of its tiles, so --coords is not given beside it. "
            f"Their origins are read from its {named.get('coords_dataset', H5_COORDS_DATASET)!r} "
            "dataset: write them there, or drop --coords."
        )
    if named and suffix not in HDF5_SUFFIXES:
        raise AlhenaValueError(
            f"--h5-datasets names datasets of an HDF5 file, and neither {path.name} nor --coords "
            "is one."
        )

    if coords is None:
        source = read_tiles(path, mpp=mpp, coords_scale=coords_scale, **named)
    else:
        # The rows name the tiles: opened once for their names, then with the origins found.
        with _columns_named_by("--coords-columns", "x=NAME y=NAME"):
            origins = read_tile_origins(coords, tiles=read_tiles(path), columns=columns)
        source = read_tiles(path, mpp=mpp, coords=origins, coords_scale=coords_scale)
    if len(source) == 0:
        raise AlhenaValueError(f"{path} holds no tile.")
    return source


@contextmanager
def _open_tiles(args: argparse.Namespace) -> Iterator[TileSource]:
    """Open the tiles a command names, and close them once the command has read them."""
    source = _tiles_of(args)
    try:
        yield source
    finally:
        source.close()


def _tiles_of(args: argparse.Namespace) -> TileSource:
    """Open the tiles a command names, with every option describing them."""
    return _tile_source(
        args.tiles,
        args.mpp,
        args.tile_px,
        args.stride_px,
        args.coords_scale,
        args.coords,
        edge=args.edge,
        datasets=args.h5_datasets,
        coords_columns=args.coords_columns,
        region=args.region,
    )


def _cut_from_one_image(path: Path) -> bool:
    """Return whether a path holds one image to cut: a slide, an image file or a 3-D `.npy`."""
    if path.is_dir():
        return False
    suffix = path.suffix.lower()
    if suffix in SLIDE_SUFFIXES or suffix in IMAGE_SUFFIXES:
        return True
    if suffix == _NPY_SUFFIX and path.is_file():
        try:
            return _load_array(path).ndim == 3
        except AlhenaError:
            return False
    return False


def _load_array(path: Path) -> np.ndarray:
    """Return the array a `.npy` file holds, mapped from disk rather than read into memory."""
    from alhena.io._files import _load_npy

    return _load_npy(path)


def _dataset_names(pairs: Sequence[str] | None) -> dict[str, str]:
    """Return the dataset arguments --h5-datasets gives, from pairs written DATASET=NAME."""
    named = _named_pairs(pairs or (), "--h5-datasets", "DATASET=NAME", "img=imgs")
    unknown = [key for key in named if key not in _H5_DATASET_ARGUMENTS]
    if unknown:
        raise AlhenaValueError(
            f"--h5-datasets renames {', '.join(_H5_DATASET_ARGUMENTS)}, the datasets an HDF5 "
            f"file holds by default, and was given {unknown[0]!r}."
        )
    return {_H5_DATASET_ARGUMENTS[key]: name for key, name in named.items()}


def _origins_dataset(named: dict[str, str], path: Path, coords: Path | None) -> dict[str, str]:
    """Return the dataset of origins in the file --coords names, as --h5-datasets names it."""
    from alhena.io import HDF5_SUFFIXES

    if coords is None or coords.suffix.lower() not in HDF5_SUFFIXES:
        raise AlhenaValueError(
            f"--h5-datasets names datasets of an HDF5 file, and {path.name} is cut into tiles "
            "with no HDF5 file of origins beside it: --coords names none."
        )
    others = sorted(set(named) - {"coords_dataset"})
    if others:
        renamed = [key for key, argument in _H5_DATASET_ARGUMENTS.items() if argument in others]
        raise AlhenaValueError(
            f"--h5-datasets renames {', '.join(renamed)}, which a file of tiles holds, and "
            f"{coords.name} holds the origins of the tiles to cut from {path.name}, under "
            f"{H5_COORDS_DATASET} or the name given to it."
        )
    return named


def _read_nuclei(
    args: argparse.Namespace, tiles: TileSource, progress: bool | None = None
) -> dict[str, Any]:
    """Read the nuclei --nuclei names, in the form its suffix or --nuclei-kind gives."""
    from alhena import io

    path: Path = args.nuclei
    if not path.exists():
        raise AlhenaValueError(f"No nuclei at {path}.")
    kind = _nuclei_kind(path) if args.nuclei_kind is None else str(args.nuclei_kind)
    misplaced = [
        flag
        for flag, attribute, _, kinds in _NUCLEI_OPTIONS
        if getattr(args, attribute) is not None and kind not in kinds
    ]
    if args.nuclei_ids is not None and kind != io.NUCLEI_KINDS[0]:
        misplaced.insert(0, "--nuclei-ids")
    if misplaced:
        raise AlhenaValueError(
            f"{' and '.join(misplaced)} {'reads' if len(misplaced) == 1 else 'read'} nuclei of "
            f"another form than {path.name}, which is read as {kind.replace('_', ' ')}. Drop "
            f"{'it' if len(misplaced) == 1 else 'them'}, or name the form of the file with "
            "--nuclei-kind."
        )

    if kind == io.NUCLEI_KINDS[0]:
        if _is_label_image(path):
            if args.nuclei_ids == NUCLEI_ID_SCOPES[0]:
                raise AlhenaValueError(
                    f"{path.name} is the label image of the image the tiles are cut from, which "
                    "names each nucleus once across that image, so its values are not read as "
                    f"numbered within each tile. Drop --nuclei-ids {NUCLEI_ID_SCOPES[0]}."
                )
            return io.read_label_image(path, tiles=tiles, progress=progress)
        renumber = _id_scope(args) != NUCLEI_ID_SCOPES[1]
        return io.read_label_maps(path, tiles=tiles, renumber=renumber)

    options: dict[str, Any] = {
        argument: getattr(args, attribute)
        for _, attribute, argument, kinds in _NUCLEI_OPTIONS
        if getattr(args, attribute) is not None and kind in kinds
    }
    if "columns" in options:
        options["columns"] = _named_pairs(
            options["columns"], "--nuclei-columns", "FIELD=COLUMN", "x=x_centroid"
        )
    with _columns_named_by("--nuclei-columns", "FIELD=COLUMN"):
        if kind == io.NUCLEI_KINDS[1]:
            return io.read_centroids(path, tiles=tiles, progress=progress, **options)
        return io.read_polygons(path, tiles=tiles, progress=progress, **options)


def _named_pairs(pairs: Sequence[str], flag: str, form: str, example: str) -> dict[str, str]:
    """Return what an option taking KEY=VALUE pairs names, refusing a malformed or repeated key."""
    named: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = str(pair).partition("=")
        if not separator or not key or not value:
            raise AlhenaValueError(
                f"{flag} takes pairs written {form}, such as {example}, and was given {pair!r}."
            )
        if key in named:
            raise AlhenaValueError(f"{flag} names {key!r} twice.")
        named[key] = value
    return named
