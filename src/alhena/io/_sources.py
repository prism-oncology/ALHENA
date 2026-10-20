"""Tile sources: what every tile reader returns, for tiles in memory or in image files."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from alhena._defaults import DEFAULT_BATCH_SIZE, _batch_size
from alhena._errors import AlhenaValueError, GeometryError
from alhena._frames import ORIGINS_FRAME
from alhena._geometry import _checked_mpp, _resolved_mpp
from alhena.io._files import _list_files, _read_image_file
from alhena.io._formats import IMAGE_SUFFIXES

# Values of `mpp_source`; a property of the slide is named by its own key.
_DECLARED_MPP_SOURCE: str = "mpp argument"
_OME_MPP_SOURCE: str = "OME PhysicalSizeX"
_TAGS_MPP_SOURCE: str = "TIFF resolution tags"


class TileSource(Sequence):
    """A sequence of uint8 RGB tiles read on demand.

    Indexing returns one tile, and `batches` reads several at a time, so a source larger than memory
    is embedded one batch at a time.

    Attributes:
        tile_ids: Name of each tile, in the order the source returns them.
        coords: Top-left pixel of each tile in the frame the tiles were cut from, `(n_tiles, 2)`, x
            then y, or None.
        coords_scale: Pixels of that frame one tile pixel spans.
        coords_frame: Name of the frame `coords` is written in, or None.
        mpp: Microns per pixel of the tiles, or None where none was declared.
        mpp_source: Where `mpp` was read from, `"mpp argument"` where the caller declared it, or
            None.
        tile_shape: Height and width shared by every tile, or None where the source declares none,
            as for tiles of several sizes or a directory of files.
        image_shape: Height and width of the image the tiles were cut from, in the pixels of the
            frame of `coords`, or None for tiles cut already.
    """

    def __init__(
        self,
        n_tiles: int,
        *,
        tile_ids: Sequence[str] | None = None,
        coords: np.ndarray | None = None,
        mpp: float | None = None,
        tile_shape: tuple[int, int] | None = None,
        coords_scale: float | None = None,
        coords_frame: str | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> None:
        """Record what a subclass knows about its tiles before any of them is read.

        Args:
            n_tiles: How many tiles the source holds.
            tile_ids: Name of each tile. Defaults to positions.
            coords: Origin of each tile, `(n_tiles, 2)`, x then y, or None.
            mpp: Microns per pixel of the tiles, or None.
            tile_shape: Height and width shared by every tile, or None.
            coords_scale: Pixels of the frame of the origins one tile pixel spans. Defaults to 1;
                refused without origins.
            coords_frame: Name of the frame of the origins. Defaults to `ORIGINS_FRAME`.
            image_shape: Height and width of the image the tiles were cut from, or None.

        Raises:
            AlhenaValueError: If `mpp` is out of range, or the names, origins or scale do not fit.
        """
        if n_tiles < 0:
            raise AlhenaValueError(
                f"A tile source holds no negative count of tiles, got {n_tiles}."
            )
        checked = None if mpp is None else _checked_mpp(mpp, warn=True)
        self._n_tiles = int(n_tiles)
        self.tile_ids = _as_tile_ids(tile_ids, self._n_tiles)
        self.coords = _as_coords(coords, self.tile_ids)
        self.coords_scale = _as_coords_scale(coords_scale, self.coords is not None)
        self.coords_frame = None
        if self.coords is not None:
            self.coords_frame = ORIGINS_FRAME if coords_frame is None else str(coords_frame)
        self.mpp = checked
        self.mpp_source = None if checked is None else _DECLARED_MPP_SOURCE
        self.tile_shape = None if tile_shape is None else (int(tile_shape[0]), int(tile_shape[1]))
        self.image_shape = (
            None if image_shape is None else (int(image_shape[0]), int(image_shape[1]))
        )
        # The height and width of each tile read so far, -1 for a tile not read yet.
        self._sizes = np.full((self._n_tiles, 2), -1, dtype=np.int64)

    @abstractmethod
    def _read(self, index: int) -> np.ndarray:
        """Return tile `index` as the source holds it."""

    def _pixels_inside(self) -> np.ndarray | None:
        """Return the height and width of each tile the image fills, or None if all are whole."""
        return None

    def __len__(self) -> int:
        return self._n_tiles

    def __getitem__(self, index: int) -> np.ndarray:
        """Return one tile as a `(height, width, 3)` uint8 RGB array; a slice is refused."""
        if isinstance(index, slice):
            raise AlhenaValueError(
                "A tile source is indexed one tile at a time. Use batches(batch_size) to read "
                "several."
            )
        position = int(index)
        if position < 0:
            position += self._n_tiles
        if not 0 <= position < self._n_tiles:
            raise IndexError(f"Tile {index} is outside the {self._n_tiles} tiles of this source.")
        tile = _check_tile(self._read(position), self.tile_ids[position])
        self._sizes[position] = tile.shape[:2]
        return tile

    def __iter__(self) -> Iterator[np.ndarray]:
        for position in range(self._n_tiles):
            yield self[position]

    def batches(self, batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[list[np.ndarray]]:
        """Yield tiles in batches, reading each batch only when it is asked for.

        Args:
            batch_size: Tiles per batch.

        Yields:
            Lists of `(height, width, 3)` uint8 RGB tiles, which need not share a size.

        Raises:
            AlhenaValueError: If the batch size is below one.
        """
        batch_size = _batch_size(batch_size)
        for start in range(0, self._n_tiles, batch_size):
            stop = min(start + batch_size, self._n_tiles)
            yield [self[position] for position in range(start, stop)]

    def _tile_sizes(self) -> np.ndarray:
        """Return the height and width of every tile, reading only the tiles not read yet."""
        for position in np.flatnonzero(self._sizes[:, 0] < 0).tolist():
            self._sizes[position] = np.shape(self[position])[:2]
        return self._sizes.copy()

    def close(self) -> None:
        """Release any file handle the source holds. Reading from it again reopens the file."""

    def __enter__(self) -> TileSource:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_tiles={self._n_tiles}, tile_shape={self.tile_shape}, "
            f"mpp={self.mpp})"
        )


class ArrayTileSource(TileSource):
    """Tiles already held in memory, as a stacked array or as one array per tile.

    `read_tiles` and `as_tile_source` build it.

    Args:
        tiles: A `(n_tiles, height, width, 3)` uint8 array, or a list of such tiles.
        tile_ids: Name of each tile. Defaults to positions.
        coords: Top-left pixel of each tile in a shared frame, `(n_tiles, 2)`, x then y.
        mpp: Microns per pixel of the tiles, carried by the source for later calls.
        coords_scale: Frame pixels per tile pixel, for tiles stored at another size. Defaults to 1.

    Raises:
        AlhenaValueError: If no tile is given, or the tiles, names or origins are unusable.
        GeometryError: If the tiles are not uint8 RGB.
    """

    def __init__(
        self,
        tiles: np.ndarray | Sequence[np.ndarray],
        *,
        tile_ids: Sequence[str] | None = None,
        coords: np.ndarray | None = None,
        mpp: float | None = None,
        coords_scale: float | None = None,
    ) -> None:
        if isinstance(tiles, np.ndarray):
            if tiles.ndim != 4:
                raise GeometryError(
                    "A stacked array of tiles has shape (n_tiles, height, width, 3), got "
                    f"{tiles.shape}. Pass one array per tile to mix tile sizes."
                )
            entries = [np.asarray(tile) for tile in tiles]
        elif isinstance(tiles, Sequence) and not isinstance(tiles, str | bytes):
            entries = [np.asarray(tile) for tile in tiles]
        else:
            raise AlhenaValueError(
                f"Tiles of type {type(tiles).__name__} are not read. Pass a (n_tiles, height, "
                "width, 3) uint8 array or a sequence of (height, width, 3) uint8 tiles."
            )
        if not entries:
            raise AlhenaValueError("No tiles were given.")
        for position, entry in enumerate(entries):
            _check_tile(entry, str(position))
        shapes = {entry.shape[:2] for entry in entries}
        shape = next(iter(shapes)) if len(shapes) == 1 else None
        super().__init__(
            len(entries),
            tile_ids=tile_ids,
            coords=coords,
            mpp=mpp,
            tile_shape=shape,
            coords_scale=coords_scale,
        )
        self._tiles = entries
        self._sizes[:] = [entry.shape[:2] for entry in entries]

    def _read(self, index: int) -> np.ndarray:
        return self._tiles[index]


class DirectoryTileSource(TileSource):
    """Tiles held one image file per tile, read when they are indexed.

    `read_tiles` builds it. Files are taken in sorted order, hidden files never, and each tile is
    named by the stem of its file.

    Args:
        directory: Directory holding the image files.
        pattern: Glob selecting the files.
        suffixes: Suffixes a file must carry to be read.
        tile_ids: Name of each tile. Defaults to the file stems.
        coords: Top-left pixel of each tile in a shared frame, `(n_tiles, 2)`, x then y.
        mpp: Microns per pixel of the tiles, carried by the source for later calls.
        coords_scale: Frame pixels per tile pixel, for tiles stored at another size. Defaults to 1.

    Raises:
        AlhenaValueError: If the directory holds no image file, or names or origins do not fit.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        pattern: str | None = None,
        suffixes: Sequence[str] = IMAGE_SUFFIXES,
        tile_ids: Sequence[str] | None = None,
        coords: np.ndarray | None = None,
        mpp: float | None = None,
        coords_scale: float | None = None,
    ) -> None:
        self._paths = _list_files(Path(directory), pattern, suffixes, "tile image")
        names = tuple(path.stem for path in self._paths) if tile_ids is None else tile_ids
        super().__init__(
            len(self._paths), tile_ids=names, coords=coords, mpp=mpp, coords_scale=coords_scale
        )

    @property
    def paths(self) -> tuple[Path, ...]:
        """Path of each tile, in the order the source returns them."""
        return tuple(self._paths)

    def _read(self, index: int) -> np.ndarray:
        return _read_image_file(self._paths[index])


def as_tile_source(
    tiles: TileSource | np.ndarray | Sequence[np.ndarray],
    *,
    mpp: float | None = None,
) -> TileSource:
    """Return tiles as a tile source.

    A tile source is returned as it is, and tiles in memory are wrapped in one. A file is read by
    `read_tiles`, `read_slide` or `read_image` instead.

    Args:
        tiles: A tile source, a `(n_tiles, height, width, 3)` uint8 array, or a list of
            `(height, width, 3)` uint8 tiles, which may differ in size.
        mpp: Microns per pixel of the tiles. A source's own resolution is checked against it, not
            replaced.

    Returns:
        The source itself, or one holding the tiles given.

    Raises:
        AlhenaValueError: If the tiles are missing or of an unknown kind, or `mpp` contradicts them.
        GeometryError: If the tiles are not uint8 RGB.
    """
    if isinstance(tiles, TileSource):
        # A declared resolution is checked against the source's own, which it does not replace.
        _resolved_mpp(tiles, None if mpp is None else _checked_mpp(mpp, warn=True))
        return tiles
    if isinstance(tiles, np.ndarray) or (
        isinstance(tiles, Sequence) and not isinstance(tiles, str | bytes)
    ):
        return ArrayTileSource(tiles, mpp=mpp)
    raise AlhenaValueError(
        f"Tiles of type {type(tiles).__name__} are not read as a tile source. Pass a tile "
        "source, a (n_tiles, height, width, 3) uint8 array or a sequence of (height, width, 3) "
        "uint8 tiles, or read a file with read_tiles, read_slide or read_image."
    )


def tile_bounds(source: TileSource, tile_shapes: np.ndarray | None = None) -> np.ndarray:
    """Return the window of every tile, in the frame its coordinates are written in.

    A tile `w` pixels wide ends `coords_scale * w` pixels after its origin. A source declaring no
    shared tile size, such as a directory, is read for the sizes of the tiles it has not read yet,
    and keeps them; `tile_shapes` gives them instead.

    Args:
        source: A tile source declaring coordinates.
        tile_shapes: Height and width of every tile, `(n_tiles, 2)`, for a caller that has read them
            already.

    Returns:
        An `(n_tiles, 4)` float64 array of x0, y0, x1, y1, the upper bound excluded.

    Raises:
        AlhenaValueError: If the source declares no coordinates, or the sizes do not match it.
    """
    if source.coords is None:
        raise AlhenaValueError(
            "This tile source declares no coordinates, so its tiles have no windows. Read tiles "
            "from a file carrying their origins, or pass coords= when building the source."
        )
    if tile_shapes is not None:
        sizes = np.asarray(tile_shapes, dtype=np.float64).reshape(-1, 2)
        if sizes.shape[0] != len(source):
            raise AlhenaValueError(
                f"{sizes.shape[0]} tile sizes were given for the {len(source)} tiles of this "
                "source. Give one height and one width per tile, in the order the source "
                "returns them."
            )
    if len(source) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    origins = np.asarray(source.coords, dtype=np.float64)
    if tile_shapes is not None:
        heights, widths = sizes[:, 0], sizes[:, 1]
    elif source.tile_shape is not None:
        heights = np.full(len(source), float(source.tile_shape[0]))
        widths = np.full(len(source), float(source.tile_shape[1]))
    else:
        sizes = source._tile_sizes().astype(np.float64)
        heights, widths = sizes[:, 0], sizes[:, 1]
    scale = float(source.coords_scale)
    return np.column_stack(
        [
            origins[:, 0],
            origins[:, 1],
            origins[:, 0] + scale * widths,
            origins[:, 1] + scale * heights,
        ]
    )


def _check_tile(tile: np.ndarray, described: str) -> np.ndarray:
    """Check that a tile is the uint8 RGB array every reader promises."""
    array = np.asarray(tile)
    if array.ndim != 3 or array.shape[2] != 3:
        raise GeometryError(
            f"Tile {described} has shape {array.shape}, where a (height, width, 3) RGB tile is "
            "expected."
        )
    if array.dtype != np.uint8:
        raise GeometryError(
            f"Tile {described} has dtype {array.dtype}, where uint8 is expected. Tiles are bytes, "
            "one per channel."
        )
    return array


def _as_tile_ids(tile_ids: Sequence[str] | None, n_tiles: int) -> tuple[str, ...]:
    if tile_ids is None:
        return tuple(str(position) for position in range(n_tiles))
    names = tuple(str(value) for value in tile_ids)
    if len(names) != n_tiles:
        raise AlhenaValueError(
            f"{len(names)} tile names were given for {n_tiles} tiles; one name per tile is "
            "required."
        )
    if len(set(names)) != len(names):
        raise AlhenaValueError("tile_ids repeats a name; one name identifies one tile.")
    return names


def _as_coords(coords: np.ndarray | None, tile_ids: Sequence[str]) -> np.ndarray | None:
    """Return the origins of the tiles in whole pixels, refusing one lying between two pixels."""
    if coords is None:
        return None
    values = np.asarray(coords)
    n_tiles = len(tile_ids)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] != n_tiles:
        raise AlhenaValueError(
            f"coords has shape {values.shape}, where ({n_tiles}, 2) holding the x and y origin of "
            "every tile is expected."
        )
    if not _is_real(values):
        raise AlhenaValueError(f"coords must be numeric, got {values.dtype}.")
    row = _first_off_pixel(values)
    if row is not None:
        raise AlhenaValueError(
            f"coords places tile {tile_ids[row]!r} at x {float(values[row, 0])}, y "
            f"{float(values[row, 1])}, "
            "which is not a whole number of pixels. Give the top-left pixel of each tile."
        )
    return values.astype(np.int64)


def _is_real(values: np.ndarray) -> bool:
    """Report whether an array holds integers or floating-point numbers."""
    return bool(np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating))


def _first_off_pixel(origins: np.ndarray) -> int | None:
    """Return the first row whose origin is not a whole, finite number of pixels, or None."""
    if np.issubdtype(origins.dtype, np.integer):
        return None
    off = ~(np.isfinite(origins) & (origins == np.floor(origins))).all(axis=1)
    return int(np.flatnonzero(off)[0]) if off.any() else None


def _as_coords_scale(coords_scale: float | None, has_coords: bool) -> float:
    """Return the frame pixels one tile pixel spans, one where none was given."""
    if coords_scale is None:
        return 1.0
    if not has_coords:
        raise AlhenaValueError(
            "coords_scale scales the origins of the tiles, and these tiles carry no origins. Read "
            "the tiles with their origins, or leave coords_scale out."
        )
    try:
        scale = float(coords_scale)
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(f"coords_scale must be a number, got {coords_scale!r}.") from error
    if not np.isfinite(scale) or scale <= 0.0:
        raise AlhenaValueError(f"coords_scale must be finite and positive, got {coords_scale}.")
    return scale
