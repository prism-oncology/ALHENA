"""Tiles cut from a whole slide or a TIFF file, read one region at a time, and areas to show."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alhena._constants import MAX_MICROSCOPE_MPP
from alhena._errors import AlhenaValueError, MissingExtraError
from alhena._frames import _windows_inside
from alhena._geometry import _PAD_VALUE, _filling_side
from alhena.io._files import _choice, _on_white
from alhena.io._formats import (
    EDGE_RULES,
    H5_COORDS_DATASET,
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
from alhena.io._images import _image_extent, _open_image
from alhena.io._resolution import _resolve_slide_mpp
from alhena.io._sources import TileSource

# The first four bytes of a TIFF file: the byte order, then 42, or 43 for a BigTIFF.
_TIFF_HEADERS: tuple[bytes, ...] = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


class SlideTileSource(TileSource):
    """Tiles cut from a whole slide, read one region at a time: a grid, or the tiles selected.

    `read_slide` builds it. Origins are in level-zero pixels whatever the level read, and
    `coords_scale` is the downsample of that level. A selected origin read at a coarser level is
    recorded at the first level-zero pixel of the level pixel it falls in.

    Attributes:
        level: Pyramid level the regions are read from.
        downsample: Pixels of level zero one pixel of that level covers.
        edge: The rule the tiles the edge of the level cuts were read under.
        mpp_source: Where the resolution of level zero was read from, such as `"mpp argument"`,
            `"OME PhysicalSizeX"`, `"aperio.MPP"` or `"TIFF resolution tags"`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        tile_px: int | None = None,
        stride_px: int | None = None,
        level: int = 0,
        mpp: float | None = None,
        edge: str = "pad",
        coords: np.ndarray | None = None,
        region: Sequence[float] | None = None,
    ) -> None:
        """Open a slide and settle the tiles to read from one of its levels.

        Args:
            path: Path of the slide.
            tile_px: Side of each tile, in pixels of the level read. Defaults to the side that
                fills the canvas at the resolution of that level.
            stride_px: Step between tile origins, in those pixels. Defaults to `tile_px`; refused
                beside `coords`.
            level: Pyramid level the tiles are read from.
            mpp: Microns per pixel of level zero, taking precedence over the slide's own.
            edge: One of `EDGE_RULES`: what becomes of the tiles the edge of the level cuts.
            coords: Level-zero origins of the tiles to cut instead of the grid, `(n_tiles, 2)`,
                x then y.
            region: x0, y0, x1, y1 in level-zero pixels, upper bounds excluded: only the tiles
                lying wholly inside it are kept.

        Raises:
            AlhenaValueError: If the slide cannot be read, the level is invalid, or no tile fits.
            GeometryError: If `mpp` is not given and the slide declares no usable resolution.
            MissingExtraError: If tiffslide is not installed.
        """
        file = Path(path)
        if not file.is_file():
            raise AlhenaValueError(f"No slide at {file}.")
        _check_slide_format(file)
        rule = _choice(edge, EDGE_RULES, "edge")
        if level < 0:
            raise AlhenaValueError(f"level must not be negative, got {level}.")
        selected = None if coords is None else _as_origins_to_cut(coords, "coords")
        kept = None if region is None else _region_window(region)

        self._path = file
        self.level = int(level)
        self.edge = rule
        self._slide: Any = None
        slide = self._open()
        try:
            dimensions, downsamples = _slide_layout(slide, file)
            if self.level >= len(dimensions):
                raise AlhenaValueError(
                    f"{file} has {len(dimensions)} pyramid levels and level {level} was asked for."
                )
            width, height = (int(value) for value in dimensions[self.level])
            self.downsample = float(downsamples[self.level])
            if not np.isfinite(self.downsample) or self.downsample < 1.0:
                raise AlhenaValueError(
                    f"{file.name} reports a downsample of {self.downsample} for level {level}, "
                    "where a finite number of at least one is expected."
                )
            base_mpp, source = _resolve_slide_mpp(slide, file, mpp)
            level_mpp = base_mpp * self.downsample
            if level_mpp > MAX_MICROSCOPE_MPP:
                raise AlhenaValueError(
                    f"Level {level} of {file.name} is at {level_mpp:.4g} micron per pixel, "
                    f"coarser than the {MAX_MICROSCOPE_MPP:g} microns of a microscope image. Read "
                    "a finer level."
                )
            # Cut once the level's resolution is known, so the default side fills the canvas.
            side, stride = _grid_steps(
                _filling_side(level_mpp) if tile_px is None else tile_px,
                stride_px,
                selecting=selected is not None,
            )
            self._tile_px = side
            where = f"level {self.level} of {file.name}"
            if selected is None:
                corners = _tile_grid(width, height, side, stride, rule)
                if corners.shape[0] == 0:
                    raise _no_tile_fits(side, (width, height), where, rule, "read a finer level")
            else:
                corners = _placed_selection(
                    selected, self.downsample, (width, height), side, rule, where
                )
            origins = _level_zero_origins(corners, self.downsample)
            if kept is not None:
                inside = _inside_region(origins, side * self.downsample, kept, where)
                corners, origins = corners[inside], origins[inside]
            # The resolution was checked and warned about already: the base class gets none.
            super().__init__(
                origins.shape[0],
                tile_ids=_origin_names(origins),
                coords=origins,
                tile_shape=(side, side),
                coords_scale=self.downsample,
                coords_frame=f"{file.name}, level 0",
                image_shape=(int(dimensions[0][1]), int(dimensions[0][0])),
            )
            self.mpp = level_mpp
            self.mpp_source = source
            self._corners = corners
            self._extent = (width, height)
        except Exception:
            self.close()
            raise

    @property
    def path(self) -> Path:
        """Path of the slide the tiles are read from."""
        return self._path

    def _open(self) -> Any:
        if self._slide is None:
            self._slide = _open_slide(self._path)
        return self._slide

    def _read(self, index: int) -> np.ndarray:
        # read_region takes level-zero pixels at every level, and fills past the edge with black.
        x, y = self.coords[index]
        region = self._open().read_region(
            (int(x), int(y)), self.level, (self._tile_px, self._tile_px)
        )
        return _white_past_edge(_region_to_rgb(region), self._corners[index], self._extent)

    def _pixels_inside(self) -> np.ndarray | None:
        windows = np.hstack([self._corners, self._corners + self._tile_px])
        return _windows_inside(windows, self._extent)

    def close(self) -> None:
        """Close the slide. Reading from the source again reopens it."""
        if self._slide is not None:
            self._slide.close()
            self._slide = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_tiles={len(self)}, tile_shape={self.tile_shape}, "
            f"level={self.level}, mpp={self.mpp}, mpp_source={self.mpp_source!r})"
        )


def read_slide(
    path: str | Path,
    *,
    tile_px: int | None = None,
    stride_px: int | None = None,
    level: int = 0,
    mpp: float | None = None,
    edge: str = "pad",
    coords: np.ndarray | pd.DataFrame | str | Path | None = None,
    columns: Mapping[str, str] | None = None,
    coords_dataset: str = H5_COORDS_DATASET,
    region: Sequence[float] | None = None,
) -> SlideTileSource:
    """Cut a whole slide into tiles read one region at a time: a grid, or the tiles selected.

    The slide is any TIFF file tiffslide opens, whatever its pyramid. Without `mpp`, the
    resolution of level zero is read from the slide's metadata. The grid is cut in the pixels of
    the level read, and the origins of its tiles are recorded in level-zero pixels.

    Args:
        path: Path of the slide.
        tile_px: Side of each tile, in pixels of the level read. Defaults to the side that fills
            the canvas at the resolution of that level.
        stride_px: Step between tile origins, in those pixels. Defaults to `tile_px`; refused
            beside `coords`.
        level: Pyramid level the tiles are read from.
        mpp: Microns per pixel of level zero, taking precedence over the resolution the slide
            declares.
        edge: One of `EDGE_RULES`: what becomes of the tiles the edge of the level cuts.
        coords: Level-zero origins of the tiles to cut instead of the grid, x then y: an array, a
            DataFrame of `x` and `y`, or the path of a table, a `.npy` file or an HDF5 file.
        columns: Columns of a table of origins holding `x` and `y`, where named otherwise.
        coords_dataset: Dataset of an HDF5 file of origins holding them.
        region: x0, y0, x1, y1 in level-zero pixels, upper bounds excluded: only the tiles lying
            wholly inside it are kept, from the grid or from `coords`.

    Returns:
        A tile source carrying the resolution of the level it reads, and where it was read from.

    Raises:
        AlhenaValueError: If the slide cannot be read, an argument is out of range, or no tile fits.
        GeometryError: If `mpp` is not given and the slide declares no usable resolution.
        MissingExtraError: If tiffslide is not installed.
    """
    selection = None if coords is None else _selection(coords, columns, coords_dataset)
    source = SlideTileSource(
        path,
        tile_px=tile_px,
        stride_px=stride_px,
        level=level,
        mpp=mpp,
        edge=edge,
        coords=None if selection is None else selection.origins,
        region=region,
    )
    if selection is not None:
        _check_recorded_side(selection, source)
    return source


def read_region(
    image: str | Path | np.ndarray,
    *,
    window: Sequence[float] | np.ndarray | None = None,
    downsample: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read one rectangle of a slide or an image for display, as coarse as a downsample allows.

    A slide is read from its coarsest level within the downsample, and an image is decimated by the
    largest whole step within it. The extent places the result in level-zero pixels, pixel centers
    at integer coordinates, as tile bounds and cell centroids are written, so both land on their
    tissue when drawn over it. A window reaching past the edge is cut there.

    Args:
        image: Path of a slide, any TIFF file tiffslide opens; or an image as `read_image` takes it,
            a `(height, width, 3)` uint8 array or the path of a `.npy` file or another image file.
        window: x0, y0, x1, y1 in level-zero pixels, upper bounds excluded, or an `(n, 4)` array of
            such windows, such as `TileEmbeddings.bounds`, whose union is read. All of it when None.
        downsample: Level-zero pixels one pixel of the result may span at most.

    Returns:
        The `(height, width, 3)` uint8 image, and its left, right, bottom and top edges in
        level-zero pixels, the extent `matplotlib.pyplot.imshow` takes.

    Raises:
        AlhenaValueError: If the image cannot be read, or the window or `downsample` is invalid.
        GeometryError: If an array is not a `(height, width, 3)` uint8 image.
        MissingExtraError: If tiffslide is not installed when a slide is read.
    """
    limit = float(downsample)
    if not np.isfinite(limit) or limit < 1.0:
        raise AlhenaValueError(
            f"downsample is the level-zero pixels one pixel of the image spans, at least one, and "
            f"was given {downsample!r}."
        )
    if not isinstance(image, str | Path):
        return _image_region(image, None, window, limit)
    file = Path(image)
    if not file.is_file():
        raise AlhenaValueError(f"No file at {file}.")
    suffix = file.suffix.lower()
    if suffix == ".npy" or (suffix in IMAGE_SUFFIXES and suffix not in SLIDE_SUFFIXES):
        return _image_region(_open_image(file), file.name, window, limit)
    _check_slide_format(file)
    slide = _open_slide(file)
    try:
        dimensions, downsamples = _slide_layout(slide, file)
        x0, y0, x1, y1 = _display_window(window, dimensions[0], file.name)
        # The coarsest level whose pixels span no more than the downsample asked for.
        level = max(
            (index for index, factor in enumerate(downsamples) if factor <= limit),
            key=lambda index: downsamples[index],
            default=0,
        )
        factor = float(downsamples[level])
        # To the nearest pixel of the level, whose downsample need not divide the window.
        size = (max(round((x1 - x0) / factor), 1), max(round((y1 - y0) / factor), 1))
        image = _region_to_rgb(slide.read_region((x0, y0), level, size))
    finally:
        slide.close()
    left, top = x0 - 0.5, y0 - 0.5
    return image, (left, left + size[0] * factor, top + size[1] * factor, top)


def _image_region(
    pixels: Any, name: str | None, window: Any, limit: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return a rectangle of an image held whole, decimated by the whole step within a limit."""
    height, width = _image_extent(pixels, "The image" if name is None else name)
    x0, y0, x1, y1 = _display_window(window, (width, height), name or "the image")
    step = int(np.floor(limit))
    picture = np.ascontiguousarray(
        pixels[y0 : int(np.ceil(y1)) : step, x0 : int(np.ceil(x1)) : step]
    )
    left, top = x0 - 0.5, y0 - 0.5
    return picture, (left, left + picture.shape[1] * step, top + picture.shape[0] * step, top)


def _display_window(
    window: Any, extent: tuple[int, int], name: str
) -> tuple[int, int, float, float]:
    """Return the window to read, the union of those given, cut at the edge of the slide."""
    width, height = (int(value) for value in extent)
    if window is None:
        return 0, 0, float(width), float(height)
    windows = np.asarray(window, dtype=np.float64)
    if windows.ndim == 1:
        windows = windows.reshape(1, -1)
    if windows.ndim != 2 or windows.shape[1] != 4 or windows.shape[0] == 0:
        raise AlhenaValueError(
            f"window has shape {np.shape(window)}, where x0, y0, x1, y1, or an (n, 4) array of "
            "them, is expected."
        )
    if not np.isfinite(windows).all():
        raise AlhenaValueError("window holds a bound that is not a finite number of pixels.")
    if bool((windows[:, 2] <= windows[:, 0]).any() or (windows[:, 3] <= windows[:, 1]).any()):
        raise AlhenaValueError(
            "window holds a rectangle whose x1 or y1 does not exceed its x0 or y0. Bounds are "
            "x0, y0, x1, y1, upper bounds excluded."
        )
    x0 = max(int(np.floor(windows[:, 0].min())), 0)
    y0 = max(int(np.floor(windows[:, 1].min())), 0)
    x1 = min(float(windows[:, 2].max()), float(width))
    y1 = min(float(windows[:, 3].max()), float(height))
    if x1 <= x0 or y1 <= y0:
        raise AlhenaValueError(
            f"The window lies outside {name}, whose level zero is {width} by {height} pixels."
        )
    return x0, y0, x1, y1


def _tiffslide() -> Any:
    """Return tiffslide, imported on use so that reading tiles never needs it."""
    try:
        import tiffslide
    except ImportError as error:
        raise MissingExtraError(
            'Reading whole slides needs tiffslide. Install it with pip install "alhena[wsi]".'
        ) from error
    return tiffslide


def _open_slide(path: Path) -> Any:
    """Open a slide with tiffslide, refusing a file it cannot open."""
    reader = _tiffslide()
    try:
        return reader.TiffSlide(str(path))
    except Exception as error:
        # Whatever the reader raises on a file it cannot open is a property of the file.
        raise AlhenaValueError(_unreadable_slide_message(path, error)) from error


def _white_past_edge(tile: np.ndarray, corner: np.ndarray, extent: tuple[int, int]) -> np.ndarray:
    """Return a tile with every pixel past the edge of its image white, as the canvas pads."""
    height, width = tile.shape[:2]
    inside_x = max(extent[0] - int(corner[0]), 0)
    inside_y = max(extent[1] - int(corner[1]), 0)
    return _padded(tile[:inside_y, :inside_x], height, width, _PAD_VALUE)


def _level_zero_origins(origins: np.ndarray, downsample: float) -> np.ndarray:
    """Return the first level-zero pixel the slide reader carries to each origin of a level."""
    level0 = np.floor(origins * downsample).astype(np.int64)
    # The reader divides and truncates; repeating it keeps rounding from reading one pixel short.
    short = (level0 / downsample).astype(np.int64) < origins
    while short.any():
        level0[short] += 1
        short = (level0 / downsample).astype(np.int64) < origins
    return level0


def _check_slide_format(path: Path) -> None:
    """Refuse a file the slide reader cannot open: a format it does not read, or no TIFF at all."""
    if path.suffix.lower() in UNREAD_SLIDE_FORMATS:
        raise AlhenaValueError(_unread_format_message(path))
    with path.open("rb") as handle:
        header = handle.read(4)
    if header not in _TIFF_HEADERS:
        raise AlhenaValueError(
            f"{path.name} does not open with a TIFF header, so the slide reader cannot read it. "
            "Export the slide to a pyramidal TIFF."
        )


def _slide_layout(slide: Any, path: Path) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    """Return the size and downsample of every level, refusing a slide the reader cannot lay out."""
    try:
        dimensions = tuple(tuple(level) for level in slide.level_dimensions)
        downsamples = tuple(slide.level_downsamples)
    except Exception as error:
        # Whatever the reader raises on a file it cannot lay out is a property of the file.
        raise AlhenaValueError(_unreadable_slide_message(path, error)) from error
    return dimensions, downsamples


def _unreadable_slide_message(path: Path, error: Exception) -> str:
    """Return the refusal of a TIFF file the slide reader cannot open or lay out."""
    found = str(error).strip() or "with no message"
    planar = (
        " The file stores its color one plane per channel rather than one pixel at a time: "
        "rewriting it with the channels interleaved reads it."
        if "SYX" in found
        else ""
    )
    return (
        f"{path.name} cannot be read as a slide. tiffslide raised {type(error).__name__} on it: "
        f"{found}. A stack of planes is read once the plane to embed is exported as an RGB TIFF."
        f"{planar}"
    )


def _region_to_rgb(region: Any) -> np.ndarray:
    """Return a slide region as an RGB array, with any transparent pixel laid on white."""
    if getattr(region, "mode", None) == "RGB":
        return np.array(region)
    if hasattr(region, "convert"):
        # Through RGBA, so that an image mode carrying transparency keeps it.
        region = region.convert("RGBA")
    array = np.asarray(region)
    if array.ndim != 3 or array.shape[2] != 4:
        return np.ascontiguousarray(array)
    if array.dtype != np.uint8:
        # Samples wider than a byte are not blended: the check every tile goes through refuses them.
        return np.ascontiguousarray(array[:, :, :3])
    return _on_white(array)
