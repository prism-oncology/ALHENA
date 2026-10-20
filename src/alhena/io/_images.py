"""Tiles cut from an image file read whole."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alhena._errors import AlhenaValueError, GeometryError
from alhena._frames import _windows_inside
from alhena._geometry import _PAD_VALUE, _checked_mpp, _filling_side
from alhena.io._files import _choice, _load_npy, _read_image_file
from alhena.io._formats import (
    EDGE_RULES,
    H5_COORDS_DATASET,
    IMAGE_FRAME,
    IMAGE_SUFFIXES,
    SLIDE_SUFFIXES,
    UNREAD_SLIDE_FORMATS,
    _unread_format_message,
)
from alhena.io._grid import (
    _as_origins_to_cut,
    _check_recorded_side,
    _grid_steps,
    _inside_region,
    _no_tile_fits,
    _origin_names,
    _padded,
    _placed_selection,
    _region_window,
    _selection,
    _tile_grid,
)
from alhena.io._sources import _DECLARED_MPP_SOURCE, TileSource


class ImageTileSource(TileSource):
    """Tiles cut from one image held whole, read one window at a time.

    `read_image` builds it. Origins are in the pixels of the image, and `coords_frame` names the
    file and its level zero, or is `IMAGE_FRAME` for an array.

    Attributes:
        edge: The rule the tiles the edge of the image cuts were read under.
    """

    def __init__(
        self,
        image: np.ndarray | str | Path,
        *,
        tile_px: int | None = None,
        stride_px: int | None = None,
        mpp: float | None = None,
        edge: str = "shift",
        coords: np.ndarray | None = None,
        coords_frame: str | None = None,
        region: Sequence[float] | None = None,
    ) -> None:
        """Hold one image and settle the tiles to cut from it.

        Args:
            image: A `(height, width, 3)` uint8 RGB array, an array read by slicing such as a memory
                map, or the path of a `.npy` file or an image file.
            tile_px: Side of each tile, in pixels of the image. Defaults to the side that fills the
                canvas at `mpp`; one of the two is required.
            stride_px: Step between tile origins. Defaults to `tile_px`; refused beside `coords`.
            mpp: Microns per pixel of the image.
            edge: One of `EDGE_RULES`: what becomes of the tiles the edge of the image cuts.
            coords: Origins of the tiles to cut instead of the grid, `(n_tiles, 2)`, x then y, in
                pixels of the image.
            coords_frame: Name of the frame of the origins. Defaults to the file and its level zero,
                or `IMAGE_FRAME` for an array.
            region: x0, y0, x1, y1 in pixels of the image, upper bounds excluded: only the tiles
                lying wholly inside it are kept.

        Raises:
            AlhenaValueError: If the image cannot be read, or an argument does not fit the image.
            GeometryError: If the image is not a `(height, width, 3)` uint8 array.
        """
        rule = _choice(edge, EDGE_RULES, "edge")
        if tile_px is None and mpp is None:
            raise AlhenaValueError(
                "The side of the tiles cannot be set: neither mpp nor tile_px was given. Declare "
                "mpp= to cut tiles that fill the canvas, or give tile_px=."
            )
        selected = None if coords is None else _as_origins_to_cut(coords, "coords")
        kept = None if region is None else _region_window(region)
        self._path: Path | None = None
        if isinstance(image, str | Path):
            self._path = _image_file(Path(image))
        checked = None if mpp is None else _checked_mpp(mpp, warn=True)
        side, stride = _grid_steps(
            _filling_side(checked) if tile_px is None else tile_px,
            stride_px,
            selecting=selected is not None,
        )
        pixels = image if self._path is None else _open_image(self._path)
        where = "the image" if self._path is None else self._path.name
        height, width = _image_extent(pixels, "The image" if self._path is None else where)
        if selected is None:
            corners = _tile_grid(width, height, side, stride, rule)
            if corners.shape[0] == 0:
                raise _no_tile_fits(side, (width, height), where, rule)
        else:
            corners = _placed_selection(selected, 1.0, (width, height), side, rule, where)
        if kept is not None:
            corners = corners[_inside_region(corners, side, kept, where)]
        # The resolution was checked and warned about above: the base class gets none.
        super().__init__(
            corners.shape[0],
            tile_ids=_origin_names(corners),
            coords=corners,
            tile_shape=(side, side),
            coords_frame=_image_frame(self._path, coords_frame),
            image_shape=(height, width),
        )
        self.mpp = checked
        self.mpp_source = None if checked is None else _DECLARED_MPP_SOURCE
        self.edge = rule
        self._tile_px = side
        self._pixels: Any = pixels

    @property
    def path(self) -> Path | None:
        """Path of the image file the tiles are cut from, or None for an array."""
        return self._path

    def _image(self) -> Any:
        if self._pixels is None:
            assert self._path is not None
            self._pixels = _open_image(self._path)
        return self._pixels

    def _read(self, index: int) -> np.ndarray:
        x, y = (int(value) for value in self.coords[index])
        side = self._tile_px
        # A copy, so a tile never shares memory with the image, however the image is held.
        window = np.array(self._image()[y : y + side, x : x + side])
        return _padded(window, side, side, _PAD_VALUE)

    def _pixels_inside(self) -> np.ndarray | None:
        if self.coords is None or self.image_shape is None:
            return None
        height, width = self.image_shape
        origins = self.coords.astype(np.int64)
        return _windows_inside(np.hstack([origins, origins + self._tile_px]), (width, height))

    def close(self) -> None:
        """Release the pixels of an image file, read again on demand; an array is left alone."""
        if self._path is not None:
            self._pixels = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_tiles={len(self)}, tile_shape={self.tile_shape}, "
            f"image_shape={self.image_shape}, mpp={self.mpp}, edge={self.edge!r})"
        )


def read_image(
    image: np.ndarray | str | Path,
    *,
    mpp: float | None = None,
    tile_px: int | None = None,
    stride_px: int | None = None,
    edge: str = "shift",
    coords: np.ndarray | pd.DataFrame | str | Path | None = None,
    columns: Mapping[str, str] | None = None,
    coords_dataset: str = H5_COORDS_DATASET,
    region: Sequence[float] | None = None,
) -> ImageTileSource:
    """Cut one large image into tiles read one window at a time: a grid, or the tiles selected.

    The image is cut as `read_slide` cuts level zero of a slide: a TIFF file gives the same tiles
    with both under the same `edge`, which defaults to `"shift"` here and `"pad"` there. No
    resolution is read from the file: `mpp` declares it, and without it `tile_px` is required.

    Args:
        image: A `(height, width, 3)` uint8 RGB array, an array read by slicing such as a memmap, or
            the path of a `.npy` file or an image file, one of `IMAGE_SUFFIXES`.
        mpp: Microns per pixel of the image.
        tile_px: Side of each tile, in pixels of the image. Defaults to the side that fills the
            canvas at `mpp`.
        stride_px: Step between tile origins. Defaults to `tile_px`; refused beside `coords`.
        edge: One of `EDGE_RULES`: what becomes of the tiles the edge of the image cuts.
        coords: Origins of the tiles to cut instead of the grid, in pixels of the image, as
            `read_slide` takes them.
        columns: Columns of a table of origins holding `x` and `y`, where named otherwise.
        coords_dataset: Dataset of an HDF5 file of origins holding them.
        region: x0, y0, x1, y1 in pixels of the image, upper bounds excluded: only the tiles lying
            wholly inside it are kept, from the grid or from `coords`.

    Returns:
        A tile source carrying the resolution declared, the origins of its tiles and the shape of
        the image.

    Raises:
        AlhenaValueError: If the image cannot be read, or an argument does not fit the image.
        GeometryError: If the image is not a `(height, width, 3)` uint8 array.
    """
    selection = None if coords is None else _selection(coords, columns, coords_dataset)
    source = ImageTileSource(
        image,
        tile_px=tile_px,
        stride_px=stride_px,
        mpp=mpp,
        edge=edge,
        coords=None if selection is None else selection.origins,
        region=region,
    )
    if selection is not None:
        _check_recorded_side(selection, source)
    return source


def _image_file(path: Path) -> Path:
    """Return the path of an image `read_image` reads, refusing a file it does not."""
    suffix = path.suffix.lower()
    if suffix in UNREAD_SLIDE_FORMATS:
        raise AlhenaValueError(_unread_format_message(path))
    if suffix in SLIDE_SUFFIXES and suffix not in IMAGE_SUFFIXES:
        raise AlhenaValueError(f"{path.name} is a whole slide, read by read_slide.")
    if suffix != ".npy" and suffix not in IMAGE_SUFFIXES:
        raise AlhenaValueError(
            f"An image is read from a .npy array or from an image file, one of "
            f"{list(IMAGE_SUFFIXES)}, and {path.name} has suffix {path.suffix!r}."
        )
    if not path.is_file():
        raise AlhenaValueError(f"No image at {path}.")
    return path


def _open_image(path: Path) -> np.ndarray:
    """Return the pixels of an image file: a `.npy` array mapped from disk, or a file decoded."""
    return _load_npy(path) if path.suffix.lower() == ".npy" else _read_image_file(path)


def _image_frame(path: Path | None, given: str | None) -> str:
    """Return the name of the frame an image places its tiles in."""
    if given is not None:
        return str(given)
    return IMAGE_FRAME if path is None else f"{path.name}, level 0"


def _image_extent(image: Any, where: str) -> tuple[int, int]:
    """Return the height and width of an image, refusing anything but a uint8 RGB array."""
    shape = getattr(image, "shape", None)
    if shape is None or not hasattr(image, "dtype"):
        raise AlhenaValueError(
            f"An image of type {type(image).__name__} is not read. Pass a (height, width, 3) "
            "uint8 array, an array read by slicing such as a memmap, or the path of an image file."
        )
    shape = tuple(int(value) for value in shape)
    if len(shape) != 3 or shape[2] != 3:
        stacked = " A stack of tiles already cut is read by read_tiles." if len(shape) == 4 else ""
        raise GeometryError(
            f"{where} has shape {shape}, where a (height, width, 3) RGB image is expected.{stacked}"
        )
    try:
        byte = np.dtype(image.dtype) == np.uint8
    except TypeError:
        byte = False
    if not byte:
        raise GeometryError(
            f"{where} has dtype {image.dtype}, where uint8 is expected. Tiles are bytes, one per "
            "channel."
        )
    return shape[0], shape[1]
