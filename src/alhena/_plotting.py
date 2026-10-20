"""Helpers the figures of `alhena.viz` share."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from alhena._errors import AlhenaValueError, GeometryError, MissingExtraError
from alhena._layout import TILE_BOUNDS_COLUMNS
from alhena.io import TileSource
from alhena.types import TileEmbeddings

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Opacity kept at the confidence floor when shading by confidence, so an uncertain annotation shows.
_MIN_SHADE: Final[float] = 0.2

# Categories the qualitative palette of tile regions colors, those of tab20.
_MOST_CATEGORIES: Final[int] = 20


def _require(module: str) -> ModuleType:
    """Import one matplotlib module, or name the extra that installs it."""
    try:
        return import_module(module)
    except ImportError as error:
        raise MissingExtraError(
            "Drawing needs matplotlib, which ALHENA does not install. Install it with "
            'pip install "alhena[viz]".'
        ) from error


def _axes(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    """Return the axes to draw on, creating a figure only when none was given."""
    plt = _require("matplotlib.pyplot")
    if ax is not None:
        return ax
    _, created = plt.subplots(figsize=figsize)
    return created


def _as_rgb(value: Any) -> tuple[float, float, float]:
    """Read a palette entry as unit RGB: integer triples as bytes, anything else via matplotlib."""
    byte_triple = (
        isinstance(value, tuple | list)
        and len(value) == 3
        and all(isinstance(channel, int | np.integer) for channel in value)
    )
    if byte_triple:
        return (
            float(value[0]) / 255.0,
            float(value[1]) / 255.0,
            float(value[2]) / 255.0,
        )
    red, green, blue = _require("matplotlib.colors").to_rgb(value)
    return (float(red), float(green), float(blue))


def _missing_note(missing: Sequence[str]) -> str:
    """Explain missing reference classes without implying their absence from tissue."""
    if not missing:
        return ""
    singular = len(missing) == 1
    target = "this class" if singular else "these classes"
    absence = "it is" if singular else "they are"
    return (
        f"The reference contains no cells of\n{', '.join(missing)}.\n"
        f"ALHENA cannot predict {target}.\n"
        f"This does not mean {absence} absent\nfrom the tissue."
    )


def _check_unit_interval(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 < number <= 1.0:
        raise AlhenaValueError(f"{name} must lie in (0, 1], got {value!r}.")
    return number


def _shade(confidence: float, floor: float, alpha: float) -> float:
    """Scale opacity by confidence, from the confidence floor of the axis up to mass one."""
    if not np.isfinite(confidence):
        return alpha
    span = max(1.0 - floor, float(np.finfo(np.float32).eps))
    scaled = float(np.clip((confidence - floor) / span, 0.0, 1.0))
    return alpha * (_MIN_SHADE + (1.0 - _MIN_SHADE) * scaled)


def _as_display_tile(tile: np.ndarray, what: str = "tile") -> np.ndarray:
    """Return a tile, or an image, in a layout matplotlib draws, rejecting a normalized tensor."""
    if isinstance(tile, TileSource):
        # Converting a source to an array would read every one of its tiles first.
        raise AlhenaValueError(
            f"A tile source reads its tiles on demand and is no single {what}. Pass one tile, "
            "source[i], or an image alhena.io.read_region reads."
        )
    image = np.asarray(tile)
    if image.ndim != 3 or image.shape[2] != 3:
        hint = ""
        if image.ndim == 3 and image.shape[0] == 3:
            hint = f" A {what} is drawn channel-last; transpose a (3, H, W) tensor first."
        raise GeometryError(
            f"Expected an (H, W, 3) RGB {what} to draw on, got shape {image.shape}.{hint}"
        )
    if image.dtype == np.uint8:
        return image
    values = np.asarray(image, dtype=np.float32)
    lowest = float(values.min()) if values.size else 0.0
    highest = float(values.max()) if values.size else 1.0
    if lowest < 0.0 or highest > 1.0:
        raise GeometryError(
            f"A {what} is drawn either as uint8 bytes or as floats in [0, 1], and this one runs "
            f"from {lowest:.3g} to {highest:.3g}. A normalized tensor is not an image; draw the "
            f"{what} as it was read, before normalization."
        )
    return values


def _source_tile(source: TileSource, tile_index: int | None) -> np.ndarray:
    """Return the tile of a source an overlay draws, refusing to choose one among several."""
    if tile_index is None:
        if len(source) != 1:
            raise AlhenaValueError(
                f"The tile source holds {len(source)} tiles. Name the one to draw with tile_index, "
                "its index among the tiles the nuclei were read against."
            )
        return source[0]
    position = int(tile_index)
    if not 0 <= position < len(source):
        raise AlhenaValueError(
            f"tile_index {position} is outside the {len(source)} tiles of the source."
        )
    return source[position]


def _is_table(value: Any) -> bool:
    """Return whether a value is an AnnData table, which `obs` and `uns` mark."""
    return hasattr(value, "obs") and hasattr(value, "uns")


def _table_column(table: Any, key: str, colors: Mapping[Any, Any] | None) -> tuple[Any, Any]:
    """Return a column of a table of tiles, and the colors scanpy stores beside it unless given."""
    if not _is_table(table):
        raise AlhenaValueError(
            f"values names the column {key!r}, and only a table of tiles has columns. Pass the "
            "table as tiles, or one value per tile as values."
        )
    if key not in table.obs.columns:
        raise AlhenaValueError(
            f"The table of tiles holds no column {key!r}. Its columns are "
            f"{list(table.obs.columns)}."
        )
    column = table.obs[key]
    stored = table.uns.get(f"{key}_colors")
    if colors is None and stored is not None and isinstance(column.dtype, pd.CategoricalDtype):
        categories = list(column.cat.categories)
        # A palette that no longer fits the categories is left for the qualitative one.
        if len(stored) == len(categories):
            colors = dict(zip(categories, list(stored), strict=True))
    return column, colors


def _region_bounds(tiles: Any) -> np.ndarray:
    """Return the bounds of tiles, of tile embeddings or of their table, refusing no bounds."""
    held = tiles.bounds if isinstance(tiles, TileEmbeddings) else tiles
    if _is_table(tiles):
        absent = [name for name in TILE_BOUNDS_COLUMNS if name not in tiles.obs.columns]
        if absent:
            raise AlhenaValueError(
                f"This table lacks {absent}, the bounds a table of tile embeddings carries "
                "when its tiles were read with their origins."
            )
        held = tiles.obs[list(TILE_BOUNDS_COLUMNS)].to_numpy(dtype=np.float64)
    if held is None:
        raise AlhenaValueError(
            "These tile embeddings carry no spatial bounds, so their regions cannot be drawn. "
            "Read the tiles with their origins, as the readers of alhena.io place them."
        )
    try:
        bounds = np.asarray(held, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(
            "Tile regions are drawn from tile embeddings, their table or an (n_tiles, 4) array of "
            f"bounds, and an object of type {type(tiles).__name__} was given."
        ) from error
    if bounds.ndim != 2 or bounds.shape[1] != 4 or not np.isfinite(bounds).all():
        raise AlhenaValueError(
            f"Tile bounds have shape {bounds.shape}, where an (n_tiles, 4) array of finite x0, y0, "
            "x1, y1 bounds is expected."
        )
    return bounds


def _image_extent(extent: Sequence[float]) -> tuple[float, float, float, float]:
    """Return the left, right, bottom and top edges of an image, refusing anything else."""
    edges = np.asarray(extent, dtype=np.float64).reshape(-1)
    if edges.shape[0] != 4 or not np.isfinite(edges).all():
        raise AlhenaValueError(
            f"extent holds {edges.shape[0]} values, where the finite left, right, bottom and top "
            "edges of the image, as alhena.io.read_region returns them, are expected."
        )
    left, right, bottom, top = (float(value) for value in edges)
    return left, right, bottom, top


def _drawn_as_categories(values: pd.Series) -> bool:
    """Return whether values are drawn as categories rather than on a color scale."""
    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(dtype):
        return True
    return not pd.api.types.is_numeric_dtype(dtype)


def _category_palette(categories: Sequence[Any], colors: Mapping[Any, Any] | None) -> dict:
    """Return the color of each category: the one given, else the qualitative palette's."""
    if colors is not None:
        missing = [value for value in categories if value not in colors]
        if missing:
            raise AlhenaValueError(
                f"No color is given for the categories {missing}. Give colors= a color for every "
                "category drawn."
            )
        return {value: _as_rgb(colors[value]) for value in categories}
    if len(categories) > _MOST_CATEGORIES:
        raise AlhenaValueError(
            f"{len(categories)} categories are more than the {_MOST_CATEGORIES} colors of the "
            "qualitative palette. Give colors= a color for every category."
        )
    # Up to ten categories take the ten strong colors of tab10, more the twenty of tab20.
    named = "tab10" if len(categories) <= 10 else "tab20"
    chosen = _require("matplotlib").colormaps[named].colors
    return {
        value: (float(chosen[at][0]), float(chosen[at][1]), float(chosen[at][2]))
        for at, value in enumerate(categories)
    }
