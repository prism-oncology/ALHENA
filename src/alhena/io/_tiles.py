"""Read tiles cut already, and their origins from a table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from alhena._errors import AlhenaValueError
from alhena.io._files import _column_names, _coordinates, _load_npy, _named, _pick_column, _table
from alhena.io._formats import (
    H5_BARCODE_DATASET,
    H5_COORDS_DATASET,
    H5_IMAGE_DATASET,
    HDF5_SUFFIXES,
    IMAGE_SUFFIXES,
    ORIGIN_COLUMNS,
    SLIDE_SUFFIXES,
    UNREAD_SLIDE_FORMATS,
    _unread_format_message,
)
from alhena.io._hdf5 import HDF5TileSource, _dataset_to_read, _h5_top_level_names, _renamed_datasets
from alhena.io._sources import (
    ArrayTileSource,
    DirectoryTileSource,
    TileSource,
    _first_off_pixel,
    as_tile_source,
)


def read_tiles(
    source: np.ndarray | Sequence[np.ndarray] | str | Path,
    *,
    mpp: float | None = None,
    tile_ids: Sequence[str] | None = None,
    coords: np.ndarray | None = None,
    pattern: str | None = None,
    coords_scale: float | None = None,
    image_dataset: str = H5_IMAGE_DATASET,
    barcode_dataset: str | None = H5_BARCODE_DATASET,
    coords_dataset: str | None = H5_COORDS_DATASET,
) -> TileSource:
    """Read tiles from an array, a `.npy` file, a directory of images or an HDF5 file.

    An HDF5 file holds the tiles in `img`, and may hold their names in `barcode` and their origins
    in `coords`. A single image or slide file is refused: `read_image` and `read_slide` cut it into
    tiles.

    Args:
        source: A `(n_tiles, height, width, 3)` uint8 array, a list of such tiles, or the path of a
            `.npy` file, a directory of images or an HDF5 file.
        mpp: Microns per pixel of the tiles, carried by the source for later calls.
        tile_ids: Name of each tile of an array or a directory. Defaults to positions, or to the
            file stems of a directory.
        coords: Top-left pixel of each tile in a shared frame, `(n_tiles, 2)`, x then y, in whole
            pixels.
        pattern: Glob selecting the files of a directory.
        coords_scale: Frame pixels per tile pixel, for tiles stored at another size than they were
            cut at. Defaults to what an HDF5 file records, else 1.
        image_dataset: HDF5 dataset holding the tiles.
        barcode_dataset: HDF5 dataset holding the tile names, or None to name them by position.
        coords_dataset: HDF5 dataset holding the tile origins, or None to read none.

    Returns:
        A tile source that reads its tiles on demand.

    Raises:
        AlhenaValueError: If the source holds no readable tile, or an argument does not fit them.
        GeometryError: If the tiles are not uint8 RGB.
    """
    renamed = _renamed_datasets(image_dataset, barcode_dataset, coords_dataset)
    named = f"{' and '.join(renamed)} {'names' if len(renamed) == 1 else 'name'} datasets"
    if isinstance(source, str | Path):
        path = Path(source)
        suffix = "" if path.is_dir() else path.suffix.lower()
        if renamed and suffix not in HDF5_SUFFIXES:
            raise AlhenaValueError(
                f"{named} of an HDF5 file, and {path.name} is not "
                f"one of the HDF5 suffixes {list(HDF5_SUFFIXES)}."
            )
        if path.is_dir():
            return DirectoryTileSource(
                path,
                pattern=pattern,
                tile_ids=tile_ids,
                coords=coords,
                mpp=mpp,
                coords_scale=coords_scale,
            )
        if suffix in UNREAD_SLIDE_FORMATS:
            raise AlhenaValueError(_unread_format_message(path))
        if suffix in SLIDE_SUFFIXES or suffix in IMAGE_SUFFIXES:
            raise AlhenaValueError(
                f"{path.name} is a single image file, not a set of tiles. Cut it into tiles with "
                "read_slide for a whole slide, or read_image for any other image."
            )
        if suffix == ".npy" and path.is_file():
            if pattern is not None:
                raise AlhenaValueError(
                    f"pattern selects files of a directory, and {path.name} is an array."
                )
            stacked = _load_npy(path)
            if stacked.ndim == 3:
                raise AlhenaValueError(
                    f"{path.name} holds one image of shape {stacked.shape}, not a stack of tiles "
                    "of shape (n_tiles, height, width, 3). Cut it into tiles with read_image."
                )
            return ArrayTileSource(
                stacked, tile_ids=tile_ids, coords=coords, mpp=mpp, coords_scale=coords_scale
            )
        if suffix in HDF5_SUFFIXES:
            if tile_ids is not None or coords is not None:
                raise AlhenaValueError(
                    f"{path.name} is the sole record of its tiles, so tile_ids and coords are "
                    "refused beside it. Write them into its "
                    f"{barcode_dataset or H5_BARCODE_DATASET!r} and "
                    f"{coords_dataset or H5_COORDS_DATASET!r} datasets instead."
                )
            held = _h5_top_level_names(path)
            return HDF5TileSource(
                path,
                image_dataset=image_dataset,
                coords_dataset=_dataset_to_read(coords_dataset, H5_COORDS_DATASET, held),
                barcode_dataset=_dataset_to_read(barcode_dataset, H5_BARCODE_DATASET, held),
                mpp=mpp,
                coords_scale=coords_scale,
            )
        if not path.exists():
            raise AlhenaValueError(f"No tiles at {path}.")
        raise AlhenaValueError(
            f"{path.name} has suffix {path.suffix!r}. read_tiles reads a directory of images, a "
            f".npy array of tiles, or a file with one of the HDF5 suffixes {list(HDF5_SUFFIXES)}."
        )
    if pattern is not None:
        raise AlhenaValueError("pattern selects files of a directory and tiles were given.")
    if renamed:
        raise AlhenaValueError(f"{named} of an HDF5 file, and tiles were given.")
    return ArrayTileSource(
        source, tile_ids=tile_ids, coords=coords, mpp=mpp, coords_scale=coords_scale
    )


def read_tile_origins(
    source: str | Path | pd.DataFrame,
    *,
    tiles: TileSource | np.ndarray | Sequence[np.ndarray],
    columns: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Read the origin of every tile from a table naming the tiles.

    Rows are matched to tiles by name, compared as text. The origins are what `read_tiles` takes as
    `coords`.

    Args:
        source: A DataFrame, or the path of a `.csv`, `.tsv`, `.txt`, `.parquet` or `.pq` file.
        tiles: The tiles the origins belong to: a tile source, or tiles in memory, named by
            position.
        columns: Columns holding `tile_id`, `x` and `y`, where named otherwise.

    Returns:
        An `(n_tiles, 2)` int64 array, x then y, in the order of `tiles.tile_ids`.

    Raises:
        AlhenaValueError: If the table cannot be read, or does not give each tile one origin.
        GeometryError: If tiles given in memory are not uint8 RGB.
    """
    tiles = as_tile_source(tiles)
    frame = _table(source, "tile origins", as_text=True)
    named = _named(source)
    chosen = _column_names(columns, ORIGIN_COLUMNS, "a table of tile origins")
    name_column = _pick_column(frame, chosen.get("tile_id"), ORIGIN_COLUMNS["tile_id"], "tile name")
    x_column = _pick_column(frame, chosen.get("x"), ORIGIN_COLUMNS["x"], "x origin")
    y_column = _pick_column(frame, chosen.get("y"), ORIGIN_COLUMNS["y"], "y origin")
    if frame.shape[0] == 0:
        raise AlhenaValueError(f"No tile origin in {named}.")

    unnamed = int(frame[name_column].isna().sum())
    if unnamed:
        raise AlhenaValueError(
            f"{unnamed} of the {frame.shape[0]} rows of {named} name no tile in "
            f"{name_column!r}. Every row names the tile whose origin it gives."
        )
    names = pd.Index([str(value) for value in frame[name_column].tolist()])
    repeated = names[names.duplicated()]
    if len(repeated):
        raise AlhenaValueError(
            f"Tile {repeated[0]!r} is named on more than one row of {named}. Name each tile once."
        )
    rows = pd.Index(tiles.tile_ids).get_indexer(names)
    unknown = names[rows < 0]
    if len(unknown):
        naming = ""
        if isinstance(tiles, DirectoryTileSource):
            naming = " A tile of a directory is named by the stem of its file."
        raise AlhenaValueError(
            f"{len(unknown)} of the {len(names)} rows of {named} name no tile of this "
            f"source, first {unknown[0]!r}. Its tiles are named {list(tiles.tile_ids[:8])}"
            f"{' and more' if len(tiles) > 8 else ''}.{naming}"
        )
    covered = np.zeros(len(tiles), dtype=bool)
    covered[rows] = True
    if not covered.all():
        absent = [tiles.tile_ids[index] for index in np.flatnonzero(~covered)]
        raise AlhenaValueError(
            f"{len(absent)} of the {len(tiles)} tiles have no row in {named}, first "
            f"{absent[0]!r}. Give each tile a row, or read the tiles without origins."
        )

    values = _coordinates(frame, (x_column, y_column), named, "origin")
    row = _first_off_pixel(values)
    if row is not None:
        raise AlhenaValueError(
            f"Row {row} of {named}, the origin of tile {names[row]!r}, x {float(values[row, 0])}, "
            f"y {float(values[row, 1])}, is not a whole number of pixels. Give the tile's top-left "
            "pixel."
        )
    origins = np.zeros((len(tiles), 2), dtype=np.int64)
    origins[rows] = values.astype(np.int64)
    return origins
