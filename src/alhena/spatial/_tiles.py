"""Tiles read from an image of a SpatialData object, and tile embeddings written back."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

from alhena._errors import AlhenaValueError, GeometryError
from alhena._layout import TILE_INDEX_COLUMN, TILES_PROVENANCE_KEY
from alhena.io import ImageTileSource, TileSource
from alhena.spatial._elements import REGION_KEY, _frame, _image_element_frame, _require
from alhena.spatial._windows import patch_bounds
from alhena.types import TileEmbeddings

# Kinds of element a SpatialData object holds, whose names it keeps unique across kinds.
_ELEMENT_KINDS: Final[dict[str, str]] = {
    "images": "image",
    "labels": "labels element",
    "points": "points element",
    "shapes": "shapes element",
    "tables": "table",
}

# Channels of an H&E image, decimals a window side is compared on, and the finest scale's name.
_RGB_CHANNELS: Final[int] = 3
_WINDOW_DIGITS: Final[int] = 6
_LEVEL_ZERO: Final[str] = "scale0"

TILE_SHAPES_KEY: Final[str] = "alhena_tiles"
"""Name the rectangles of a tile-embedding result are stored under by default."""
TILE_TABLE_KEY: Final[str] = "alhena_tile_embeddings"
"""Name the table of tile embeddings is stored under by default."""

DEFAULT_COORDINATE_SYSTEM: Final[str] = "global"
"""Coordinate system rectangles are placed in where no image places them."""


class _ImagePlane:
    """An image element read lazily as the `(height, width, 3)` windows a tile source cuts."""

    def __init__(self, data: Any, shape: tuple[int, int]) -> None:
        self._data = data
        self.shape = (shape[0], shape[1], _RGB_CHANNELS)
        self.dtype = np.dtype(np.uint8)

    def __getitem__(self, window: tuple[slice, slice]) -> np.ndarray:
        rows, columns = window
        pixels = self._data[:_RGB_CHANNELS, rows, columns]
        values = np.asarray(getattr(pixels, "values", pixels))
        return np.moveaxis(values, 0, -1)


def tile_source(
    sdata: Any,
    *,
    image_key: str | None = None,
    tile_px: int | None = None,
    stride_px: int | None = None,
    mpp: float | None = None,
    edge: str = "shift",
    patches: str | np.ndarray | None = None,
    coordinate_system: str | None = None,
) -> TileSource:
    """Return the tiles of an image of a SpatialData object as a tile source.

    The tiles are a grid cut as `alhena.io.read_image` cuts an image, or the windows `patches`
    selects, in the level-zero pixels of the image. The pixels of a window are read when its tile is
    read.

    Args:
        sdata: The object holding the image.
        image_key: Image the tiles are cut from, needed only when the object holds several.
        tile_px: Side of each tile, in level-zero pixels. Defaults to the side that fills the canvas
            at `mpp`, or with `patches` to the side of its windows, which it must then equal.
        stride_px: Step between tile origins. Defaults to `tile_px`; refused beside `patches`.
        mpp: Microns per pixel of the image at level zero, carried by the tiles.
        edge: One of `alhena.io.EDGE_RULES`: what becomes of the tiles the edge of the image cuts.
        patches: Shapes element holding one box per tile, or an `(n_tiles, 4)` array of x0, y0, x1,
            y1 windows, squares of one side, read as `patch_bounds` reads them.
        coordinate_system: Coordinate system the windows of `patches` are carried through into the
            pixels of the image.

    Returns:
        A tile source over the image, naming its tiles by their origin, in the frame named after
        the image, such as `image 'he', level 0`.

    Raises:
        AlhenaValueError: If the image is missing or ambiguous, or the windows or sizes do not fit.
        GeometryError: If the image is not three channels of bytes.
        MissingExtraError: If the spatial extra is not installed.
    """
    name, element = _one_image(sdata, image_key)
    plane = _image_plane(element, name)
    coords = None
    if patches is not None:
        if stride_px is not None:
            raise AlhenaValueError(
                "stride_px spaces the tiles of a grid, and patches selects the windows to cut "
                "instead. Pass one or the other."
            )
        coords, side = _windows_to_cut(sdata, patches, coordinate_system, name)
        if tile_px is not None and tile_px != side:
            raise AlhenaValueError(
                f"tile_px is {tile_px!r}, and the windows of patches are {side} pixels across. "
                "Leave tile_px out: the windows set the side of the tiles."
            )
        tile_px = side
    # The windows are in the level-zero pixels of the image, so the frame is named after it.
    return ImageTileSource(
        plane,
        tile_px=tile_px,
        stride_px=stride_px,
        mpp=mpp,
        edge=edge,
        coords=coords,
        coords_frame=_image_element_frame(name),
    )


def _one_image(sdata: Any, image_key: str | None) -> tuple[str, Any]:
    """Return the image the tiles are cut from, and the name it is held under."""
    images = getattr(sdata, "images", {})
    if image_key is not None:
        return str(image_key), _frame(sdata, None, image_key).image
    held = sorted(str(key) for key in images)
    if len(held) != 1:
        holds = f"holds the images {held}" if held else "holds no image"
        raise AlhenaValueError(
            f"This object {holds}, so which one the tiles are cut from is not settled. Name it "
            "with image_key=."
        )
    return held[0], images[held[0]]


def _image_plane(element: Any, name: str) -> _ImagePlane:
    """Return level zero of an image element as the pixels a tile source cuts."""
    data = _level_zero(element, name)
    shape = tuple(int(value) for value in getattr(data, "shape", ()))
    if len(shape) != 3:
        raise GeometryError(
            f"The image {name!r} has shape {shape}, where an image stored as (c, y, x) is expected."
        )
    if shape[0] != _RGB_CHANNELS:
        raise GeometryError(
            f"The image {name!r} holds {shape[0]} channels, where the three of an H&E image are "
            "expected. Tiles are read as RGB bytes."
        )
    if np.dtype(data.dtype) != np.uint8:
        raise GeometryError(
            f"The image {name!r} has dtype {data.dtype}, where uint8 is expected. Tiles are "
            "bytes, one per channel."
        )
    return _ImagePlane(data, (shape[1], shape[2]))


def _level_zero(element: Any, name: str) -> Any:
    """Return the array an image element holds, its finest scale for a multiscale image."""
    if hasattr(element, "dtype"):
        return element
    scales = [str(key) for key in getattr(element, "keys", dict)()]
    if not scales:
        raise GeometryError(
            f"The image {name!r} holds no array to cut tiles from, and no scale holding one."
        )
    finest = _LEVEL_ZERO if _LEVEL_ZERO in scales else scales[0]
    held = [value for value in element[finest].values() if hasattr(value, "dtype")]
    if len(held) != 1:
        raise GeometryError(
            f"The scale {finest!r} of the image {name!r} holds {len(held)} arrays, where one "
            "image is expected."
        )
    return held[0]


def _windows_to_cut(
    sdata: Any,
    patches: str | np.ndarray,
    coordinate_system: str | None,
    image_key: str,
) -> tuple[np.ndarray, int]:
    """Return the origins of the windows to cut, and the side they share."""
    windows = patch_bounds(
        sdata, patches=patches, coordinate_system=coordinate_system, image_key=image_key
    )
    sides = np.unique(
        np.round(
            np.column_stack([windows[:, 2] - windows[:, 0], windows[:, 3] - windows[:, 1]]),
            _WINDOW_DIGITS,
        )
    )
    if sides.shape[0] != 1:
        raise AlhenaValueError(
            f"These windows run {sorted(float(side) for side in sides)} pixels across, where a "
            "tile source cuts tiles of one side. Cut a grid by leaving patches out, or select "
            "windows of one side."
        )
    side = float(sides[0])
    if side != np.floor(side):
        raise AlhenaValueError(
            f"These windows run {side} pixels across, and tiles are cut at whole pixels of the "
            "image. Give windows a whole number of pixels across."
        )
    origins = np.round(windows[:, :2], _WINDOW_DIGITS)
    between = ~(origins == np.floor(origins)).all(axis=1)
    if between.any():
        row = int(np.flatnonzero(between)[0])
        raise AlhenaValueError(
            f"Window {row} starts at x {float(windows[row, 0])}, y {float(windows[row, 1])}, "
            "between two "
            "pixels, and tiles are cut at whole pixels of the image. Give windows starting at "
            "whole pixels."
        )
    return origins.astype(np.int64), int(side)


def tiles_to_spatialdata(
    sdata: Any | None,
    tiles: TileEmbeddings,
    shapes_key: str | None = None,
    table_key: str | None = None,
    image_key: str | None = None,
    overwrite: bool = False,
) -> Any:
    """Write tile embeddings back as one rectangle and one table row per tile.

    The rectangles are the regions of the source image the embeddings stand for, and the table is
    the one `TileEmbeddings.write_h5ad` writes, tied to them. With `sdata` None, both go into an
    object of their own. The rectangles take the transformations of the image the tiles were cut
    from, else the identity in `DEFAULT_COORDINATE_SYSTEM`.

    Args:
        sdata: Object the rectangles and the table are added to, or None to build one.
        tiles: The tile embeddings, carrying the spatial bounds of their tiles.
        shapes_key: Name the rectangles are stored under. Defaults to `TILE_SHAPES_KEY`.
        table_key: Name the table is stored under. Defaults to `TILE_TABLE_KEY`.
        image_key: Image of `sdata` the tiles were cut from, whose transformations the rectangles
            take. Defaults to the image of `sdata` their bounds are written in.
        overwrite: Replace elements already stored under those names.

    Returns:
        The object, with the rectangles and the table in it.

    Raises:
        AlhenaValueError: If the tiles lack bounds, a name is taken, or `image_key` is wrong.
        MissingExtraError: If the spatial extra is not installed.
    """
    if not isinstance(tiles, TileEmbeddings):
        raise AlhenaValueError(
            "tiles_to_spatialdata takes the object first and the tile embeddings second, "
            "tiles_to_spatialdata(sdata, tiles), and was given an object of type "
            f"{type(tiles).__name__} as the tiles. Pass None as the object to build one."
        )
    shapes_key = TILE_SHAPES_KEY if shapes_key is None else str(shapes_key)
    table_key = TILE_TABLE_KEY if table_key is None else str(table_key)
    n_tiles = len(tiles)
    if n_tiles == 0:
        raise AlhenaValueError(
            "There are no tiles to write: a table with no row annotates nothing."
        )
    if tiles.bounds is None:
        raise AlhenaValueError(
            "These tile embeddings carry no spatial bounds, so there is no rectangle to write. "
            "Read the tiles with a reader that carries their origins, or with "
            "alhena.spatial.tile_source, and embed them with AlhenaEncoder.encode_tiles."
        )

    if shapes_key.lower() == table_key.lower():
        raise AlhenaValueError(
            f"shapes_key and table_key both name {shapes_key!r}, and an object names each "
            "element once, whatever the case of its letters. Name the rectangles and the table "
            "otherwise."
        )
    _check_names(sdata, {shapes_key: "shapes", table_key: "tables"})
    taken = [
        name
        for name, group in ((shapes_key, "shapes"), (table_key, "tables"))
        if name in getattr(sdata, group, {})
    ]
    if taken and not overwrite:
        raise AlhenaValueError(
            f"{taken} already name elements of this object. Pass overwrite=True to replace "
            "them, or name the rectangles and the table otherwise."
        )
    cut_from = _bounds_image(sdata, tiles.bounds_frame)
    if image_key is None:
        image_key = cut_from
    elif cut_from is not None and str(image_key) != cut_from:
        raise AlhenaValueError(
            f"The bounds of these tiles are written in the pixels of the image {cut_from!r}, and "
            f"image_key names {image_key!r}. Leave image_key out, so the rectangles take the "
            "transformations of the image the tiles were cut from."
        )

    # After the checks, so a caller's mistake is refused before a library is imported.
    models = _require("spatialdata.models")
    geopandas = _require("geopandas")
    geometry = _require("shapely.geometry")
    if sdata is None:
        sdata = _require("spatialdata").SpatialData()

    table = tiles.to_anndata()
    instances = np.asarray(table.obs[TILE_INDEX_COLUMN])
    windows = np.asarray(tiles.bounds, dtype=np.float64)
    boxes = geopandas.GeoDataFrame(
        {TILE_INDEX_COLUMN: instances},
        geometry=[geometry.box(*window) for window in windows.tolist()],
        index=pd.Index(instances),
    )
    sdata.shapes[shapes_key] = models.ShapesModel.parse(
        boxes, transformations=_placed_as(sdata, image_key)
    )

    table.obs[REGION_KEY] = pd.Categorical([shapes_key] * n_tiles)
    table.uns[TILES_PROVENANCE_KEY].update(
        {
            "region": shapes_key,
            "region_key": REGION_KEY,
            "instance_key": TILE_INDEX_COLUMN,
        }
    )
    sdata.tables[table_key] = models.TableModel.parse(
        table, region=shapes_key, region_key=REGION_KEY, instance_key=TILE_INDEX_COLUMN
    )
    return sdata


def _check_names(sdata: Any, wanted: dict[str, str]) -> None:
    """Refuse a name SpatialData would refuse once the first element is stored."""
    for name, group in wanted.items():
        argument = "shapes_key" if group == "shapes" else "table_key"
        if (
            name in ("", ".", "..")
            or name.startswith("__")
            or not all(character.isalnum() or character in "_-." for character in name)
        ):
            raise AlhenaValueError(
                f"{argument} is {name!r}, and an element is named by letters, digits, underscores, "
                f"dots and hyphens, not starting with two underscores. Pass {argument}= another "
                "name."
            )
        for kind, element in _ELEMENT_KINDS.items():
            for held in getattr(sdata, kind, {}):
                if str(held).lower() == name.lower() and (kind != group or str(held) != name):
                    raise AlhenaValueError(
                        f"{name!r} is taken by the {element} {str(held)!r} of this object, which "
                        "names each element once, whatever its kind or the case of its letters. "
                        f"Pass {argument}= another name."
                    )


def _bounds_image(sdata: Any, frame: str | None) -> str | None:
    """Return the image of an object whose level-zero pixels a frame names, or None."""
    if frame is None:
        return None
    images = (str(key) for key in getattr(sdata, "images", {}))
    return next((key for key in images if _image_element_frame(key) == frame), None)


def _placed_as(sdata: Any, image_key: str | None) -> dict[str, Any]:
    """Return the transformations the rectangles take: the image's, or the default identity."""
    transformations = _require("spatialdata.transformations")
    if image_key is None:
        return {DEFAULT_COORDINATE_SYSTEM: transformations.Identity()}
    image = _frame(sdata, None, image_key).image
    placed = transformations.get_transformation(image, get_all=True)
    return {str(system): placement for system, placement in placed.items()}
