"""Tiles read from an HDF5 file, with their names and origins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from alhena._errors import AlhenaValueError, GeometryError
from alhena.io._formats import (
    H5_BARCODE_DATASET,
    H5_COORDS_DATASET,
    H5_IMAGE_DATASET,
    H5_SCALE_RECORDS,
)
from alhena.io._sources import TileSource, _as_coords_scale, _first_off_pixel, _is_real


class HDF5TileSource(TileSource):
    """Tiles held in one HDF5 file, read when they are indexed.

    `read_tiles` builds it. A dataset an argument names must exist, and None reads the tiles without
    names or origins. A file may record the side of its tiles under one of `H5_SCALE_RECORDS`, on
    its datasets as named; two records, or a record and `coords_scale`, must agree within half a
    pixel.

    Args:
        path: Path of the HDF5 file.
        image_dataset: Dataset holding the tiles.
        barcode_dataset: Dataset holding the tile names, or None to name them by position.
        coords_dataset: Dataset holding the tile origins, or None to read none.
        mpp: Microns per pixel of the tiles, carried by the source for later calls.
        coords_scale: Frame pixels per tile pixel. Defaults to what the file records, else 1.

    Raises:
        AlhenaValueError: If the file or a dataset named is missing, or the records disagree.
        GeometryError: If the tiles are not an `(n_tiles, height, width, 3)` uint8 array.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        image_dataset: str = H5_IMAGE_DATASET,
        barcode_dataset: str | None = H5_BARCODE_DATASET,
        coords_dataset: str | None = H5_COORDS_DATASET,
        mpp: float | None = None,
        coords_scale: float | None = None,
    ) -> None:
        file = Path(path)
        if not file.is_file():
            raise AlhenaValueError(f"No tile file at {file}.")
        self._path = file
        self._image_dataset = image_dataset
        self._handle: Any = None

        handle = self._open()
        try:
            if image_dataset not in handle:
                raise AlhenaValueError(
                    f"{file} holds no dataset {image_dataset!r}, only {sorted(handle)}. Name the "
                    "dataset holding the tiles with image_dataset=."
                )
            images = handle[image_dataset]
            if images.ndim != 4 or int(images.shape[3]) != 3:
                raise GeometryError(
                    f"{file} holds {image_dataset!r} with shape {tuple(images.shape)}, where "
                    "(n_tiles, height, width, 3) is expected."
                )
            if images.dtype != np.uint8:
                raise GeometryError(
                    f"{file} holds {image_dataset!r} with dtype {images.dtype}, where uint8 is "
                    "expected. Tiles are bytes, one per channel."
                )
            n_tiles = int(images.shape[0])
            if n_tiles == 0:
                raise AlhenaValueError(f"{file} holds no tile.")
            names = None
            if barcode_dataset is not None:
                names = _read_h5_barcodes(handle, barcode_dataset, n_tiles, file)
            origins = None
            frame = None
            shape = (int(images.shape[1]), int(images.shape[2]))
            if coords_dataset is not None:
                origins = _read_h5_coords(handle, coords_dataset, n_tiles, file)
                held = {H5_COORDS_DATASET: coords_dataset, H5_IMAGE_DATASET: image_dataset}
                coords_scale = _h5_coords_scale(handle, held, shape, coords_scale, file)
                frame = f"{file.name}, {coords_dataset}"
            # Inside the guard, since TileSource refuses repeated names and a non-positive mpp.
            super().__init__(
                n_tiles,
                tile_ids=names,
                coords=origins,
                mpp=mpp,
                tile_shape=shape,
                coords_scale=coords_scale,
                coords_frame=frame,
            )
        except Exception:
            self.close()
            raise

    @property
    def path(self) -> Path:
        """Path of the file the tiles are read from."""
        return self._path

    def _open(self) -> Any:
        if self._handle is None:
            import h5py

            self._handle = h5py.File(self._path, "r")
        return self._handle

    def _read(self, index: int) -> np.ndarray:
        return np.asarray(self._open()[self._image_dataset][index])

    def close(self) -> None:
        """Close the file. Reading from the source again reopens it."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _h5_top_level_names(path: Path) -> set[str]:
    """Return the top-level names of an HDF5 file, or none if the file is missing."""
    if not path.is_file():
        return set()
    import h5py

    with h5py.File(path, "r") as handle:
        return set(handle)


def _read_h5_barcodes(handle: Any, dataset: str, n_tiles: int, path: Path) -> tuple[str, ...]:
    """Read the tile names of an HDF5 tile file, which are variable-length utf-8."""
    if dataset not in handle:
        raise AlhenaValueError(
            f"{path} holds no dataset {dataset!r}, only {sorted(handle)}. Pass "
            "barcode_dataset=None to read the tiles without their names."
        )
    values = np.asarray(handle[dataset][()])
    stored_shape = values.shape
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.shape[0] != n_tiles:
        raise AlhenaValueError(
            f"{path} holds {dataset!r} with shape {stored_shape}, where ({n_tiles}, 1) is expected."
        )
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes | np.bytes_) else str(value)
        for value in values
    )


def _read_h5_coords(handle: Any, dataset: str, n_tiles: int, path: Path) -> np.ndarray:
    """Read the tile origins of an HDF5 tile file."""
    if dataset not in handle:
        raise AlhenaValueError(
            f"{path} holds no dataset {dataset!r}, only {sorted(handle)}. Pass "
            "coords_dataset=None to read the tiles without their origins."
        )
    values = np.asarray(handle[dataset][()])
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] != n_tiles:
        raise AlhenaValueError(
            f"{path} holds {dataset!r} with shape {values.shape}, where ({n_tiles}, 2) holding "
            "the x and y origin of every tile is expected."
        )
    if not _is_real(values):
        raise AlhenaValueError(
            f"{path} holds {dataset!r} with dtype {values.dtype}, where numbers of pixels are "
            "expected."
        )
    row = _first_off_pixel(values)
    if row is not None:
        raise AlhenaValueError(
            f"{path} holds {dataset!r} whose row {row} places a tile at x {values[row, 0]:g}, y "
            f"{values[row, 1]:g}, which is not a whole number of pixels. Rewrite the dataset, or "
            "read the tiles with coords_dataset=None."
        )
    return values.astype(np.int64)


class _ScaleRecord(NamedTuple):
    """One record of the scale of its origins an HDF5 file holds, as a convention writes it."""

    # The dataset the record sits on, as the file names it.
    dataset: str
    frame_attribute: str
    # The side of one tile in the pixels of the frame of the origins.
    frame_side: float
    tile_attribute: str
    # The side of one tile in its own pixels, or None.
    tile_side: float | None

    @property
    def where(self) -> str:
        """Where the side in the frame is recorded, as a message names it."""
        return f"{self.frame_attribute!r} on {self.dataset!r}"


def _scale_records(handle: Any, names: Mapping[str, str | None], path: Path) -> list[_ScaleRecord]:
    """Return each record of the scale of its origins an HDF5 file holds."""
    records: list[_ScaleRecord] = []
    for default, frame_attribute, tile_attribute in H5_SCALE_RECORDS:
        name = names.get(default, default)
        if name is None or name not in handle:
            continue
        attributes = handle[name].attrs
        if frame_attribute not in attributes:
            continue
        frame_side = _positive_attribute(attributes, frame_attribute, name, path)
        tile_side = None
        if tile_attribute in attributes:
            tile_side = _positive_attribute(attributes, tile_attribute, name, path)
        records.append(_ScaleRecord(name, frame_attribute, frame_side, tile_attribute, tile_side))
    return records


def _one_recorded_side(
    records: Sequence[_ScaleRecord], path: Path, advice: str
) -> _ScaleRecord | None:
    """Return the record of the tile side, refusing two records more than half a pixel apart."""
    if not records:
        return None
    first = records[0]
    for other in records[1:]:
        if abs(other.frame_side - first.frame_side) > 0.5:
            raise AlhenaValueError(
                f"{path} records the side of its tiles twice, {first.frame_side:g} in "
                f"{first.where} and {other.frame_side:g} in {other.where}. {advice}"
            )
    return first


def _h5_coords_scale(
    handle: Any,
    names: Mapping[str, str | None],
    tile_shape: tuple[int, int],
    given: float | None,
    path: Path,
) -> float | None:
    """Return the scale of the origins of an HDF5 file: the one it records, else the one given."""
    if given is not None:
        given = _as_coords_scale(given, has_coords=True)
    records = _scale_records(handle, names, path)
    if not records:
        return given
    advice = "Correct the attributes, or read the tiles with coords_dataset=None."
    height, width = tile_shape
    if height != width:
        raise AlhenaValueError(
            f"{path} records the side of its tiles in {records[0].where}, and its tiles are "
            f"{width} by {height} pixels, so one side does not describe them. {advice}"
        )
    for record in records:
        if record.tile_side is not None and record.tile_side != width:
            raise AlhenaValueError(
                f"{path} records tiles {record.tile_side:g} pixels across in "
                f"{record.tile_attribute!r} on {record.dataset!r}, and holds tiles {width} pixels "
                f"across, so the record describes other tiles. {advice}"
            )
    record = _one_recorded_side(records, path, advice)
    assert record is not None
    recorded = record.frame_side / width
    if given is not None and abs(given * width - record.frame_side) > 0.5:
        raise AlhenaValueError(
            f"{path} records in {record.where} that its tiles, {width} pixels across, span "
            f"{record.frame_side:g} pixels of the frame of its origins, a coords_scale of "
            f"{recorded:.6g}, and coords_scale={given:g} was passed. Pass none to read the scale "
            "from the file."
        )
    return recorded


def _positive_attribute(attributes: Any, key: str, name: str, path: Path) -> float:
    """Return an attribute recording a number of pixels, refusing anything but a positive number."""
    value = attributes[key]
    array = np.asarray(value)
    number = None
    if array.size == 1 and np.issubdtype(array.dtype, np.number):
        number = float(array.reshape(-1)[0])
    if number is None or not np.isfinite(number) or number <= 0.0:
        raise AlhenaValueError(
            f"{path} records {key!r} = {value!r} on {name!r}, where a positive number of pixels "
            "is expected."
        )
    return number


def _renamed_datasets(
    image_dataset: str, barcode_dataset: str | None, coords_dataset: str | None
) -> list[str]:
    """Return the dataset arguments of `read_tiles` given otherwise than by default."""
    return [
        name
        for name, given, default in (
            ("image_dataset", image_dataset, H5_IMAGE_DATASET),
            ("barcode_dataset", barcode_dataset, H5_BARCODE_DATASET),
            ("coords_dataset", coords_dataset, H5_COORDS_DATASET),
        )
        if given != default
    ]


def _dataset_to_read(name: str | None, default: str, held: set[str]) -> str | None:
    """Return the dataset to read: the default one if the file holds it, any other always."""
    if name is None or (name == default and name not in held):
        return None
    return name


def _read_h5_origins(path: Path, dataset: str) -> tuple[np.ndarray, _ScaleRecord | None]:
    """Return the origins an HDF5 file of origins holds, and the record of their level-zero side."""
    import h5py

    with h5py.File(path, "r") as handle:
        if dataset not in handle or not isinstance(handle[dataset], h5py.Dataset):
            raise AlhenaValueError(
                f"{path} holds no dataset {dataset!r}, only {sorted(handle)}. Name the dataset "
                "holding the origins with coords_dataset=."
            )
        values = np.asarray(handle[dataset][()])
        records = _scale_records(handle, {H5_COORDS_DATASET: dataset}, path)
    advice = "Correct the attributes, or pass the origins the file holds as an array."
    return values, _one_recorded_side(records, path, advice)
