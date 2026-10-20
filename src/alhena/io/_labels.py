"""Nuclei read as label maps, one per tile, or as one label image."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from alhena import _progress
from alhena._errors import AlhenaValueError
from alhena.io._carry import global_cell_ids
from alhena.io._files import _list_files, _load_npy
from alhena.io._formats import CELL_ID_STRIDE, H5_LABEL_DATASET, HDF5_SUFFIXES, LABEL_SUFFIXES
from alhena.io._grid import _padded
from alhena.io._sources import TileSource, as_tile_source, tile_bounds


def read_label_maps(
    source: np.ndarray | Sequence[np.ndarray] | str | Path,
    *,
    tiles: TileSource | np.ndarray | Sequence[np.ndarray] | None = None,
    dataset: str = H5_LABEL_DATASET,
    renumber: bool = True,
    stride: int = CELL_ID_STRIDE,
) -> dict[str, Any]:
    """Read one integer label map per tile.

    Values numbered within each tile are renumbered by `global_cell_ids`; `renumber=False` keeps
    identifiers already unique across the tiles. One label image over the whole image is read by
    `read_label_image` instead.

    Args:
        source: An `(n_tiles, height, width)` integer array, a list of `(height, width)` maps, or
            the path of a directory of label files, an HDF5 file or a memory-mapped `.npy` stack.
        tiles: The tiles the maps belong to. Required for maps stored one per file or HDF5
            dataset, matched to them by name; a stack holds one map per tile, in their order.
        dataset: Dataset of an HDF5 file stacking every map. Where the file holds no such dataset,
            its two-dimensional datasets are read one per tile, in sorted order.
        renumber: Make the label values unique across the tiles.
        stride: Number of identifiers in the block each tile owns, when renumbering.

    Returns:
        A mapping holding `label_maps` and `tile_indices`. Where the tiles carry their positions,
        it also holds their windows and frame, `tile_bounds` and `tile_frame`.

    Raises:
        AlhenaValueError: If a map is unreadable or not integer, or the maps do not fit the tiles.
        GeometryError: If tiles given in memory are not uint8 RGB.
    """
    tiles = None if tiles is None else as_tile_source(tiles)
    names: list[str] | None = None
    if isinstance(source, str | Path):
        path = Path(source)
        stack = path.name
        if path.is_dir():
            files = _list_files(path, None, LABEL_SUFFIXES, "label map")
            names = [file.stem for file in files]
            maps = [_read_label_file(file) for file in files]
        elif path.suffix.lower() in HDF5_SUFFIXES:
            names, maps = _read_h5_label_maps(path, dataset)
            if names is None:
                stack = f"Dataset {dataset!r} of {path.name}"
        elif path.suffix.lower() == ".npy":
            # Mapped from disk: without renumbering, a stack larger than memory stays there.
            stacked = _load_npy(path)
            if stacked.ndim != 3:
                raise AlhenaValueError(
                    f"{path.name} holds an array of shape {tuple(stacked.shape)}, where "
                    "(n_tiles, height, width) is expected. An array of two dimensions is a label "
                    "image, which read_label_image cuts along the tiles."
                )
            maps = _label_maps_in_memory(stacked)
        else:
            raise AlhenaValueError(
                f"{path.name} is neither a directory of label maps, an HDF5 file, nor a stacked "
                ".npy array. A single label file is a label image, which read_label_image cuts "
                "along the tiles."
            )
    else:
        maps = _label_maps_in_memory(source)
        stack = f"The {'array' if isinstance(source, np.ndarray) else 'sequence'} of label maps"

    indices = _align_to_tiles(names, tiles, len(maps), stack)
    if renumber:
        maps = [
            _renumber_labels(entry, int(index), stride)
            for entry, index in zip(maps, indices, strict=True)
        ]
    return {"label_maps": maps, "tile_indices": indices, **_read_against(tiles)}


def read_label_image(
    source: np.ndarray | str | Path,
    *,
    tiles: TileSource | None = None,
    progress: bool | None = None,
) -> dict[str, Any]:
    """Cut the label image of one large image into a label map for each tile cut from it.

    The identifiers are kept as they are, so a nucleus two overlapping tiles hold is one cell.
    Pixels a padded tile holds past the edge of the image are background, and a tile over no
    nucleus gets no map. Each map is read from the label image when it is used, so a label image
    larger than memory stays where it is held: keep an HDF5 file holding it open until then.

    Args:
        source: A `(height, width)` integer array, an array read by slicing (memmap, zarr, HDF5),
            or the path of a memory-mapped `.npy` or a `.png`, `.tif` or `.tiff` file.
        tiles: The tiles `read_slide` or `read_image` cut, at level zero, from the image the label
            image covers.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        A mapping holding `label_maps`, a sequence of `(height, width)` maps read when indexed,
        `tile_indices`, and the windows and frame of the tiles, `tile_bounds` and `tile_frame`.

    Raises:
        AlhenaValueError: If the label image does not fit the tiles, or no tile holds a nucleus.
    """
    tiles = None if tiles is None else as_tile_source(tiles)
    if tiles is None:
        raise AlhenaValueError(
            "A label image is cut along the tiles cut from the image it labels, and no tiles "
            "were given. Pass tiles=, as read_slide or read_image returns them."
        )
    if tiles.image_shape is None or tiles.coords is None or tiles.tile_shape is None:
        raise AlhenaValueError(
            "These tiles were not cut from one image by read_slide or read_image, so no window "
            "of a label image lies under them. Tiles that arrive already cut take one label map "
            "each, which read_label_maps reads."
        )
    scale = float(tiles.coords_scale)
    if scale != 1.0:
        raise AlhenaValueError(
            f"Each pixel of these tiles spans {scale:g} pixels of level zero of "
            f"{tiles.coords_frame}, the grid of its label image. Read the tiles at level zero, or "
            "give the nuclei as polygons or centroids."
        )
    labels, described = _open_label_image(source)
    height, width = tiles.image_shape
    found = (int(labels.shape[0]), int(labels.shape[1]))
    if found != (height, width):
        raise AlhenaValueError(
            f"{described} is {found[1]} by {found[0]} pixels, and the image the tiles were cut "
            f"from, {tiles.coords_frame}, is {width} by {height}. Give the label image of that "
            "image, at its full resolution."
        )
    origins = np.asarray(tiles.coords, dtype=np.int64)
    every = _LabelWindows(labels, origins, tiles.tile_shape, described)
    indices: list[int] = []
    n_tiles = len(tiles)
    with _progress.track(
        range(n_tiles), n_tiles, "Cutting the label image into tiles", "tile", progress
    ) as positions:
        for index in positions:
            window = every[index]
            if window.size and int(window.min()) < 0:
                raise AlhenaValueError(
                    f"{described} holds a negative value under tile {tiles.tile_ids[index]!r}. A "
                    "label image holds zero for background and a positive identifier for each "
                    "nucleus."
                )
            if window.any():
                indices.append(index)
    if not indices:
        raise AlhenaValueError(
            f"No tile holds a nucleus of {described}: every pixel under the {n_tiles} tiles is "
            "background. Check that it is the label image of the image the tiles were cut from, "
            "and that the tiles lie over its nuclei."
        )
    held = np.asarray(indices, dtype=np.int64)
    maps = _LabelWindows(labels, origins[held], tiles.tile_shape, described)
    return {"label_maps": maps, "tile_indices": held, **_read_against(tiles)}


def label_map_from_masks(
    masks: np.ndarray | Sequence[np.ndarray],
    cell_ids: Sequence[int] | np.ndarray,
    shape: tuple[int, int] | None = None,
    progress: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Paint one mask per nucleus into one label map, resolving the pixels masks share.

    A shared pixel goes to the smaller nucleus, and between equal areas to the lower identifier. The
    map keeps the identifiers given, so read it with `renumber=False`.

    Args:
        masks: One mask per nucleus, true or nonzero where it covers the pixel: an
            `(n_nuclei, height, width)` array or a list of `(height, width)` arrays.
        cell_ids: Positive identifier of each nucleus, unique across the tiles.
        shape: Height and width of the map. Defaults to that of the masks; required without any.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        The int64 label map, zero for background, and the sorted identifiers left with no pixel.

    Raises:
        AlhenaValueError: If an identifier is not positive or unique, or a mask's shape differs.
    """
    if isinstance(masks, np.ndarray) and masks.ndim != 3:
        raise AlhenaValueError(
            f"A stacked array of masks has shape (n_nuclei, height, width), got {masks.shape}. "
            "Pass one mask per nucleus."
        )
    try:
        ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    except (OverflowError, TypeError, ValueError) as error:
        raise AlhenaValueError(
            "Cell identifiers are whole numbers a 64-bit integer holds, and the ones given are not."
        ) from error
    n_masks = len(masks)
    if ids.shape[0] != n_masks:
        raise AlhenaValueError(
            f"{ids.shape[0]} identifiers were given for {n_masks} masks; one identifier per "
            "nucleus is required."
        )
    if ids.size and int(ids.min()) < 1:
        raise AlhenaValueError(
            f"Cell identifiers are positive, and {int(ids.min())} was given. Zero marks "
            "background in a label map."
        )
    if np.unique(ids).shape[0] != ids.shape[0]:
        raise AlhenaValueError(
            "Cell identifiers repeat. An identifier names one nucleus across all the tiles read "
            "together, not one nucleus within a tile."
        )
    if shape is None:
        if n_masks == 0:
            raise AlhenaValueError(
                "No mask was given to take the shape of the label map from. Pass shape=."
            )
        shape = np.shape(masks[0])[:2]
    if len(shape) != 2:
        raise AlhenaValueError(f"shape is the height and width of the map, got {shape!r}.")
    size = (int(shape[0]), int(shape[1]))

    # Each mask is scanned once, for the pixels it covers, and painted from those alone.
    covered: list[np.ndarray] = []
    with _progress.track(
        range(n_masks), n_masks, "Reading nucleus masks", "mask", progress
    ) as positions:
        for position in positions:
            array = np.asarray(masks[position])
            if array.shape != size:
                raise AlhenaValueError(
                    f"Mask {position} has shape {array.shape}, where the label map is {size}."
                )
            covered.append(np.flatnonzero(array > 0))
    areas = np.asarray([pixels.size for pixels in covered], dtype=np.int64)

    label_map = np.zeros(size, dtype=np.int64)
    painted = label_map.reshape(-1)
    # By decreasing area, then identifier, so the smallest nucleus is painted last and wins.
    for position in np.lexsort((-ids, -areas)):
        painted[covered[position]] = ids[position]

    present = np.unique(label_map)
    # setdiff1d keeps the input order when told the values are unique, so they are sorted here.
    lost = np.sort(np.setdiff1d(ids, present[present > 0], assume_unique=True))
    return label_map, lost


def _read_label_file(path: Path) -> np.ndarray:
    """Read one label file as a two-dimensional integer array."""
    if not path.is_file():
        raise AlhenaValueError(f"No label map at {path}.")
    if path.suffix.lower() == ".npy":
        values = _load_npy(path, mapped=False)
    else:
        buffer = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        if buffer.size == 0:
            raise AlhenaValueError(f"{path} is empty.")
        # IMREAD_COLOR would narrow sixteen-bit identifiers to bytes over three channels.
        values = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if values is None:
            raise AlhenaValueError(f"{path} could not be decoded as a label map.")
    return _as_label_map(values, str(path))


def _as_label_map(values: np.ndarray, described: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise AlhenaValueError(
            f"{described} has shape {array.shape}, where a (height, width) map is expected: "
            "identifiers written as colors cannot be read back."
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise AlhenaValueError(
            f"{described} has dtype {array.dtype}, where an integer map is expected."
        )
    return array


def _label_maps_in_memory(source: Any) -> list[np.ndarray]:
    """Read label maps already held in memory, which name no tile."""
    if isinstance(source, np.ndarray):
        if source.ndim != 3:
            raise AlhenaValueError(
                "A stacked array of label maps has shape (n_tiles, height, width), got "
                f"{source.shape}."
            )
        entries = list(source)
    elif isinstance(source, Sequence) and not isinstance(source, str | bytes):
        entries = list(source)
    else:
        raise AlhenaValueError(
            f"Label maps of type {type(source).__name__} are not read. Pass an array, a sequence "
            "of maps, a directory or an HDF5 file."
        )
    return [_as_label_map(entry, f"label map {position}") for position, entry in enumerate(entries)]


def _read_h5_label_maps(path: Path, dataset: str) -> tuple[list[str] | None, list[np.ndarray]]:
    """Read label maps from an HDF5 file: named after their datasets, or unnamed from a stack."""
    import h5py

    with h5py.File(path, "r") as handle:
        if dataset in handle:
            values = np.asarray(handle[dataset][()])
            if values.ndim != 3:
                raise AlhenaValueError(
                    f"{path} holds {dataset!r} with shape {values.shape}, where (n_tiles, height, "
                    "width) is expected."
                )
            maps = [
                _as_label_map(values[index], f"{dataset}[{index}]")
                for index in range(values.shape[0])
            ]
            return None, maps
        names = [
            name
            for name in sorted(handle)
            if isinstance(handle[name], h5py.Dataset) and handle[name].ndim == 2
        ]
        if not names:
            raise AlhenaValueError(
                f"{path} holds no dataset {dataset!r} and no two-dimensional dataset to read one "
                f"map per tile from. Its top-level datasets are {sorted(handle)}."
            )
        maps = [_as_label_map(np.asarray(handle[name][()]), f"{path}:{name}") for name in names]
    return names, maps


def _align_to_tiles(
    names: Sequence[str] | None, tiles: TileSource | None, n_maps: int, stack: str
) -> np.ndarray:
    """Return the tile index of each map: by name where the maps are named, else by position."""
    if names is None:
        if tiles is not None and n_maps != len(tiles):
            raise AlhenaValueError(
                f"{stack} holds {n_maps} {'map' if n_maps == 1 else 'maps'}, and "
                f"{len(tiles)} {'tile was' if len(tiles) == 1 else 'tiles were'} given. A stack "
                "of label maps holds one map per tile, in the order of the tiles."
            )
        return np.arange(n_maps, dtype=np.int64)
    if tiles is None:
        raise AlhenaValueError(
            f"{stack} names each label map after its tile, and no tiles were given to match the "
            "names to. Pass tiles=, the tiles the maps belong to."
        )
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise AlhenaValueError(
            f"{stack} holds more than one label map named {repeated[0]!r}. A tile takes one map, "
            "named after it."
        )
    positions = {name: index for index, name in enumerate(tiles.tile_ids)}
    indices = []
    for name in names:
        if name not in positions:
            raise AlhenaValueError(
                f"Nucleus map {name!r} names no tile of this source. Its tiles are named "
                f"{list(tiles.tile_ids[:8])}{' and more' if len(tiles) > 8 else ''}."
            )
        indices.append(positions[name])
    return np.asarray(indices, dtype=np.int64)


def _read_against(tiles: TileSource | None) -> dict[str, Any]:
    """Return the windows and frame of the tiles nuclei were read against, where they are known."""
    if tiles is None or tiles.coords is None or tiles.tile_shape is None:
        return {}
    return {"tile_bounds": tile_bounds(tiles), "tile_frame": tiles.coords_frame}


def _renumber_labels(values: np.ndarray, tile_index: int, stride: int) -> np.ndarray:
    """Move the positive labels of one tile into the identifier block that tile owns."""
    array = np.asarray(values)
    if array.size and int(array.min()) < 0:
        raise AlhenaValueError(
            "A label map holds a negative value. Maps hold zero for background and positive "
            "identifiers for nuclei."
        )
    nuclei = array > 0
    renumbered = np.zeros(array.shape, dtype=np.int64)
    if nuclei.any():
        highest = int(array.max())
        if highest >= stride:
            raise AlhenaValueError(
                f"The label map of tile {tile_index} holds the identifier {highest}, which does "
                f"not fit in a block of {stride} identifiers. Pass renumber=False if the "
                "identifiers are unique across the tiles already, or raise stride."
            )
        renumbered[nuclei] = global_cell_ids(tile_index, array[nuclei], stride=stride)
    return renumbered


def _is_label_image(source: Any) -> bool:
    """Return whether label maps are given as one label image: a single file or a 2-D array."""
    if isinstance(source, str | Path):
        path = Path(source)
        if path.is_dir() or path.suffix.lower() not in LABEL_SUFFIXES:
            return False
        if path.suffix.lower() == ".npy" and path.is_file():
            return _load_npy(path).ndim == 2
        return True
    if isinstance(source, Sequence):
        return False
    return len(getattr(source, "shape", ())) == 2


def _open_label_image(source: Any) -> tuple[Any, str]:
    """Return a label image as an array read by slicing, and how messages name it."""
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.is_file():
            raise AlhenaValueError(f"No label image at {path}.")
        suffix = path.suffix.lower()
        if suffix not in LABEL_SUFFIXES:
            raise AlhenaValueError(
                f"A label image is read from {list(LABEL_SUFFIXES)} files, and {path.name} has "
                f"suffix {path.suffix!r}."
            )
        described = path.name
        labels = _load_npy(path) if suffix == ".npy" else _read_label_file(path)
    elif hasattr(source, "shape") and hasattr(source, "dtype"):
        described = "The label image"
        labels = source
    else:
        raise AlhenaValueError(
            f"A label image of type {type(source).__name__} is not read. Pass a (height, width) "
            "integer array, an array read by slicing such as a memmap or a zarr array, or the "
            f"path of a {', '.join(LABEL_SUFFIXES)} file."
        )
    shape = tuple(int(value) for value in labels.shape)
    if len(shape) == 3 and shape[2] == 1:
        shape = shape[:2]
    if len(shape) != 2:
        stacked = (
            " A stack of one label map per tile is read by read_label_maps."
            if len(shape) == 3
            else ""
        )
        raise AlhenaValueError(
            f"{described} has shape {tuple(labels.shape)}, where a (height, width) label image of "
            f"the whole image is expected.{stacked}"
        )
    if not _holds_integers(labels):
        raise AlhenaValueError(
            f"{described} has dtype {labels.dtype}, where an integer map is expected."
        )
    return labels, described


def _holds_integers(values: Any) -> bool:
    """Report whether an array, or an array read by slicing, holds integers."""
    try:
        return bool(np.issubdtype(np.dtype(values.dtype), np.integer))
    except TypeError:
        return False


class _LabelWindows(Sequence):
    """The windows of a label image under some tiles, each read when it is indexed."""

    def __init__(
        self, labels: Any, origins: np.ndarray, shape: tuple[int, int], described: str
    ) -> None:
        self._labels = labels
        self._origins = origins
        self.map_shape = (int(shape[0]), int(shape[1]))
        self._described = described

    def __len__(self) -> int:
        return int(self._origins.shape[0])

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(f"Map {index} is outside the {len(self)} maps read.")
        x, y = (int(value) for value in self._origins[position])
        try:
            return _label_window(self._labels, x, y, self.map_shape)
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            raise AlhenaValueError(
                f"{self._described} could not be read under the tile at x {x}, y {y}: {error}. "
                "Its maps are read as they are used, so keep an HDF5 file holding it open until "
                "then."
            ) from error

    def __repr__(self) -> str:
        height, width = self.map_shape
        return f"<{len(self)} label maps of {width} by {height} pixels, read when indexed>"


def _label_window(labels: Any, x: int, y: int, shape: tuple[int, int]) -> np.ndarray:
    """Return the part of a label image under one tile, background where it passes the edge."""
    height, width = shape
    window = np.asarray(labels[y : y + height, x : x + width])
    if window.ndim == 3:
        window = window[:, :, 0]
    return _padded(window, height, width, 0)
