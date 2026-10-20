"""Tiles and nuclei in the form the cell path reads them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from alhena import _progress
from alhena._constants import INPUT_SIZE
from alhena._errors import AlhenaValueError, GeometryError
from alhena._rings import _ring_areas
from alhena.cells._canvas import _POSITION_SLACK
from alhena.cells._centroids import _centroids_membership
from alhena.cells._labels import _labels_membership
from alhena.cells._polygons import _checked_rings, _polygons_membership
from alhena.cells._table import _ATTR_FRAME, _ATTR_TILE_BOUNDS, _columns_of
from alhena.io import TileSource
from alhena.io._labels import _LabelWindows
from alhena.types import MEMBERSHIP_COLUMNS, NucleiInput, TileInput

# Kinds of cell identifier a table of cells names its rows by, as pandas infers them.
_TABLE_ID_KINDS: tuple[str, ...] = ("integer", "string")

# Words a message names each kind of identifier by, keyed by what pandas infers of them.
_ID_KIND_WORDS: dict[str, str] = {
    "integer": "integers",
    "string": "text",
    "bytes": "bytes",
    "floating": "decimal numbers",
    "mixed-integer-float": "integers and decimal numbers",
    "boolean": "true and false values",
    "empty": "no identifier",
}

_LABELS = "labels"
_CENTROIDS = "centroids"
_POLYGONS = "polygons"
_TABLE = "table"

# Key of a nuclei mapping to its builder, chosen by the key written, never by the array.
_NUCLEI_KEYS: dict[str, str] = {
    "label_maps": _LABELS,
    "centroids": _CENTROIDS,
    "polygons": _POLYGONS,
}

# Arguments of each builder beside the nuclei; the tile origins come from the tile source.
_BUILDER_OPTIONS: dict[str, tuple[str, ...]] = {
    _LABELS: ("tile_indices",),
    _CENTROIDS: ("tile_indices", "radius_px", "cell_ids", "tile_shape"),
    _POLYGONS: ("tile_bounds", "cell_ids", "tile_scale"),
}

# Fields a reader records beside the nuclei for the checks, accepted though no builder takes them.
_CARRIED_FIELDS: dict[str, tuple[str, ...]] = {
    _LABELS: ("tile_bounds", "tile_frame"),
    _CENTROIDS: ("tile_bounds", "tile_frame"),
    _POLYGONS: ("tile_frame",),
}

_NUCLEI_ADVICE = (
    "Nuclei are given as integer label maps, one per tile; as a mapping naming 'label_maps', "
    "'centroids' or 'polygons' together with the arguments of that builder; or as a membership "
    f"table with columns {MEMBERSHIP_COLUMNS}."
)


def _read_nuclei(nuclei: NucleiInput) -> tuple[str, Any, dict[str, Any]]:
    """Return which builder the nuclei select, the nuclei themselves and the builder's options."""
    if isinstance(nuclei, pd.DataFrame):
        return _TABLE, nuclei, {}
    if isinstance(nuclei, Mapping):
        named = [key for key in _NUCLEI_KEYS if key in nuclei]
        if len(named) != 1:
            raise AlhenaValueError(
                f"A mapping of nuclei names exactly one of {list(_NUCLEI_KEYS)}, got "
                f"{sorted(nuclei)}."
            )
        key = named[0]
        kind = _NUCLEI_KEYS[key]
        options = {name: value for name, value in nuclei.items() if name != key}
        accepted = (*_BUILDER_OPTIONS[kind], *_CARRIED_FIELDS.get(kind, ()))
        unexpected = sorted(set(options) - set(accepted))
        if unexpected:
            named = "is not an argument" if len(unexpected) == 1 else "are not arguments"
            raise AlhenaValueError(
                f"{', '.join(map(repr, unexpected))} {named} of the {kind} builder, which takes "
                f"{list(_BUILDER_OPTIONS[kind])}. The resolution is declared once, as the mpp of "
                "the call."
            )
        payload = _as_label_maps(nuclei[key]) if kind == _LABELS else nuclei[key]
        return kind, payload, options
    return _LABELS, _as_label_maps(nuclei), {}


def _as_label_maps(nuclei: Any) -> Sequence[np.ndarray]:
    """Read nuclei given as arrays, which are label maps and never positions."""
    if isinstance(nuclei, _LabelWindows):
        # Windows of a label image, checked as it was opened and read one at a time.
        return nuclei
    if isinstance(nuclei, np.ndarray):
        if nuclei.ndim != 3:
            raise AlhenaValueError(
                f"An array of label maps has shape (n_tiles, height, width), got {nuclei.shape}. "
                + _NUCLEI_ADVICE
            )
        maps = list(nuclei)
    elif isinstance(nuclei, Sequence) and not isinstance(nuclei, (str, bytes)):
        try:
            maps = [np.asarray(entry) for entry in nuclei]
        except (TypeError, ValueError) as error:
            raise AlhenaValueError(
                "One of the label maps is not a rectangular array. " + _NUCLEI_ADVICE
            ) from error
    else:
        raise AlhenaValueError(
            f"Nuclei of type {type(nuclei).__name__} are not read. " + _NUCLEI_ADVICE
        )
    for position, entry in enumerate(maps):
        if entry.ndim != 2 or not np.issubdtype(entry.dtype, np.integer):
            raise AlhenaValueError(
                f"Label map {position} has shape {entry.shape} and dtype {entry.dtype}, where an "
                "integer map of one identifier per nucleus is expected. " + _NUCLEI_ADVICE
            )
    return maps


def _check_indices(values: Any, n_tiles: int, expected: int | None = None) -> None:
    """Check that tile indices are one per nucleus and point at tiles that were given."""
    indices = np.asarray(values)
    if indices.ndim != 1:
        raise AlhenaValueError(
            f"tile_indices is a one-dimensional array of tile indices, got shape {indices.shape}."
        )
    if expected is not None and indices.shape[0] != expected:
        raise AlhenaValueError(
            f"{indices.shape[0]} tile indices were given for {expected} nuclei; one index per "
            "nucleus is required."
        )
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= n_tiles):
        raise AlhenaValueError(
            f"tile_indices runs from {int(indices.min())} to {int(indices.max())}, outside the "
            f"{n_tiles} tiles given."
        )


def _identifier_kind(values: Any) -> str:
    """Return the kind of a set of cell identifiers, in the words a message names it by."""
    inferred = pd.api.types.infer_dtype(np.asarray(values).reshape(-1), skipna=False)
    if inferred in _ID_KIND_WORDS:
        return _ID_KIND_WORDS[inferred]
    if inferred.startswith("mixed"):
        return "values of several kinds"
    return f"{inferred} values"


def _plain_identifier(value: Any) -> bool:
    """Report whether one identifier is an integer or text, the kinds a table names rows by."""
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def _check_identifier_kind(values: Any, described: str) -> None:
    """Refuse cell identifiers neither all integers nor all text, before any tile is embedded."""
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        return
    if pd.api.types.infer_dtype(array, skipna=False) in _TABLE_ID_KINDS:
        return
    items = array.tolist()
    example = next((item for item in items if not _plain_identifier(item)), items[0])
    kind = _identifier_kind(array)
    advice = "Name every cell by an integer, or every cell by text."
    if kind == _ID_KIND_WORDS["bytes"]:
        advice = "Decode them to text first, as the readers of alhena.io do."
    raise AlhenaValueError(
        f"{described} are {kind}, such as {example!r}, where a table of cells holds integers or "
        f"text. {advice}"
    )


def _check_identifiers(kind: str, payload: Any, options: Mapping[str, Any], cell_id: Any) -> None:
    """Refuse the identifiers the nuclei are named by where no table of cells can hold them."""
    if kind == _LABELS:
        return
    if kind == _TABLE:
        _check_identifier_kind(_columns_of(payload)["cell_id"], "The cells of the membership table")
        return
    named = cell_id if cell_id is not None else options.get("cell_ids")
    if named is not None:
        _check_identifier_kind(named, "The cell identifiers")


def _tile_windows(bounds: Any) -> np.ndarray:
    """Read the tile windows polygons are placed against."""
    advice = (
        "Polygons need 'tile_bounds': an (n_tiles, 4) array holding the x0, y0, x1, y1 window of "
        "every tile, in the frame the polygons are written in."
    )
    if bounds is None:
        raise AlhenaValueError(advice)
    try:
        windows = np.asarray(bounds, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(advice) from error
    if windows.ndim != 2 or windows.shape[1] != 4:
        raise AlhenaValueError(f"tile_bounds has shape {windows.shape}. " + advice)
    return windows


def _median_nucleus_area(
    kind: str, payload: Any, options: Mapping[str, Any], progress: bool | None = None
) -> float | None:
    """Return the median nucleus area in tile pixels, or None where it is not recorded."""
    if kind == _LABELS:
        areas: list[np.ndarray] = []
        with _progress.track(payload, len(payload), "Measuring nuclei", "tile", progress) as maps:
            for entry in maps:
                values = entry[entry > 0]
                if values.size:
                    areas.append(np.unique(values, return_counts=True)[1])
        if not areas:
            return None
        return float(np.median(np.concatenate(areas)))
    if kind == _CENTROIDS:
        radii = np.asarray(options["radius_px"], dtype=np.float64).reshape(-1)
        if radii.size == 0:
            return None
        return float(np.pi * np.median(radii * radii))
    if kind == _POLYGONS:
        areas = _polygon_areas(payload)
        return float(np.median(areas)) if areas.size else None
    # A membership table holds canvas pixels, from which the area on the tiles is not recoverable.
    return None


def _polygon_areas(polygons: Any) -> np.ndarray:
    """Return the area each nucleus boundary encloses, in the pixels of the shared frame."""
    return np.abs(_ring_areas(_checked_rings(polygons)))


def _membership_from_nuclei(
    kind: str,
    payload: Any,
    options: Mapping[str, Any],
    mpp: float | None,
    target_mpp: float,
    cell_id: Sequence[Any] | np.ndarray | None,
    *,
    progress: bool | None,
    placement: _Placement | None,
    inside: np.ndarray | None,
) -> pd.DataFrame:
    """Run the builder the nuclei select, at a checked resolution, placing nuclei by origin."""
    if kind == _TABLE:
        if cell_id is not None:
            raise AlhenaValueError(
                "A membership table names its own cells in its cell_id column, so cell_id has "
                "nothing to name here."
            )
        return payload

    carried = _CARRIED_FIELDS.get(kind, ())
    arguments = {name: value for name, value in options.items() if name not in carried}
    origins: dict[str, Any] = {}
    if placement is not None:
        origins = {"tile_origins": placement.origins, "tile_scale": placement.scale}
    if kind == _LABELS:
        if cell_id is not None:
            raise AlhenaValueError(
                "Label values are the cell identifiers, so cell_id has nothing to name here. "
                "Renumber the label maps across the tiles instead."
            )
        indices = arguments.pop("tile_indices", None)
        if indices is None:
            indices = np.arange(len(payload), dtype=np.int64)
        frame = _labels_membership(
            payload,
            indices,
            mpp=mpp,
            target_mpp=target_mpp,
            progress=progress,
            inside=inside,
            **origins,
        )
        return _named_frame(frame, placement)

    if cell_id is not None:
        if "cell_ids" in arguments:
            raise AlhenaValueError(
                "The nuclei were named twice, through cell_id and through 'cell_ids'. Pass one "
                "of them."
            )
        arguments["cell_ids"] = cell_id

    if kind == _CENTROIDS:
        frame = _centroids_membership(
            np.asarray(payload),
            arguments.pop("tile_indices"),
            arguments.pop("radius_px"),
            mpp=mpp,
            target_mpp=target_mpp,
            progress=progress,
            inside=inside,
            **arguments,
            **origins,
        )
        return _named_frame(frame, placement)
    windows = arguments.pop("tile_bounds")
    frame = _polygons_membership(
        payload,
        windows,
        mpp=mpp,
        target_mpp=target_mpp,
        progress=progress,
        inside=inside,
        **arguments,
    )
    if placement is not None and _windows_at_origins(windows, placement):
        return _named_frame(frame, placement)
    return frame


@dataclass(frozen=True)
class _Placement:
    """Where a tile source puts its tiles: origins, frame pixels per tile pixel, frame name."""

    origins: np.ndarray
    scale: float
    frame: str


def _windows_at_origins(windows: Any, placement: _Placement) -> bool:
    """Report whether polygon windows start at the origins of their tiles, in their frame."""
    starts = np.asarray(windows, dtype=np.float64)[:, :2]
    if starts.shape != placement.origins.shape:
        return False
    return bool(np.allclose(starts, placement.origins, rtol=_POSITION_SLACK, atol=_POSITION_SLACK))


def _table_on_tile_windows(frame: pd.DataFrame, placement: _Placement) -> bool:
    """Report whether a membership table was built against the windows of these tiles."""
    windows = frame.attrs.get(_ATTR_TILE_BOUNDS)
    return windows is not None and _windows_at_origins(windows, placement)


def _tile_placement(tiles: TileInput) -> _Placement | None:
    """Return where a tile source places its tiles, or None where the tiles carry no origins."""
    if not isinstance(tiles, TileSource) or tiles.coords is None:
        return None
    return _Placement(
        origins=np.asarray(tiles.coords, dtype=np.float64),
        scale=float(tiles.coords_scale),
        frame=str(tiles.coords_frame),
    )


def _pixels_inside(tiles: TileInput) -> np.ndarray | None:
    """Return the height and width of each tile its image fills, or None where it fills all."""
    if not isinstance(tiles, TileSource):
        return None
    return tiles._pixels_inside()


def _named_frame(frame: pd.DataFrame, placement: _Placement | None) -> pd.DataFrame:
    """Record on a table built against a tile source the name the source gives its frame."""
    if placement is not None:
        frame.attrs[_ATTR_FRAME] = placement.frame
    return frame


def _tile_geometry(tiles: TileInput) -> tuple[int, int, tuple[int, int] | None]:
    """Return the tile count, the side of a tile, and the shape of the first raw tile or None."""
    first_shape: tuple[int, int] | None = None
    if isinstance(tiles, torch.Tensor):
        if tiles.ndim != 4:
            raise AlhenaValueError(
                f"A batch of tiles is four-dimensional, got shape {tuple(tiles.shape)}."
            )
        # A prepared batch is channel-first on the canvas; a raw one is channel-last bytes.
        if torch.is_floating_point(tiles):
            size = int(tiles.shape[-1])
        else:
            size = int(tiles.shape[1])
            first_shape = (int(tiles.shape[1]), int(tiles.shape[2]))
        count = int(tiles.shape[0])
    elif isinstance(tiles, np.ndarray):
        if tiles.ndim != 4:
            raise AlhenaValueError(
                "A stacked array of tiles has shape (n_tiles, height, width, 3), got "
                f"{tiles.shape}. Pass one array per tile to mix tile sizes."
            )
        count, size = int(tiles.shape[0]), int(tiles.shape[1])
        first_shape = (int(tiles.shape[1]), int(tiles.shape[2]))
    elif isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)):
        if not len(tiles):
            raise AlhenaValueError("No tiles were given.")
        first = np.asarray(tiles[0])
        if first.ndim != 3:
            raise AlhenaValueError(
                f"Every tile is a (height, width, 3) RGB array, got shape {first.shape}."
            )
        count, size = len(tiles), int(first.shape[0])
        first_shape = (int(first.shape[0]), int(first.shape[1]))
    else:
        raise AlhenaValueError(
            f"Tiles of type {type(tiles).__name__} are not read. Pass a (n_tiles, height, width, "
            "3) uint8 array, a sequence of such tiles, or a prepared batch."
        )
    if count == 0:
        raise AlhenaValueError("No tiles were given.")
    return count, size, first_shape


def _known_tile_shapes(
    tiles: TileInput, n_tiles: int, first_shape: tuple[int, int] | None
) -> np.ndarray:
    """Return the height and width of each raw tile known without reading it, -1 where not."""
    shapes = np.full((n_tiles, 2), -1, dtype=np.int64)
    declared = getattr(tiles, "tile_shape", None)
    if isinstance(tiles, (torch.Tensor, np.ndarray)):
        shapes[:] = (int(tiles.shape[1]), int(tiles.shape[2]))
    elif isinstance(tiles, (list, tuple)):
        for position, tile in enumerate(tiles):
            shape = np.shape(tile)
            if len(shape) >= 2:
                shapes[position] = (int(shape[0]), int(shape[1]))
    elif declared is not None:
        shapes[:] = (int(declared[0]), int(declared[1]))
    elif first_shape is not None:
        shapes[0] = first_shape
    return shapes


def _is_prepared(tiles: TileInput, mpp: float | None) -> bool:
    """Report whether the tiles arrived prepared, refusing a prepared batch that cannot be read."""
    if not isinstance(tiles, torch.Tensor) or not torch.is_floating_point(tiles):
        return False
    if mpp is not None:
        raise GeometryError(
            "These tiles are already prepared, so there is nothing left for mpp to resample. "
            "Pass raw uint8 tiles with their mpp, or prepared tiles without it."
        )
    if tuple(tiles.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
        raise GeometryError(
            f"Prepared tiles must have shape (N, 3, {INPUT_SIZE}, {INPUT_SIZE}), got "
            f"{tuple(tiles.shape)}. Use prepare_tiles to reach it."
        )
    return True


def _tile_names(tiles: TileInput, indices: np.ndarray) -> tuple[str, ...]:
    """Return the name of each tile read: the source's own name where it has one."""
    names = getattr(tiles, "tile_ids", None)
    if names is not None and len(names) == _tile_count(tiles):
        return tuple(str(names[int(index)]) for index in indices)
    return tuple(str(int(index)) for index in indices)


def _tile_count(tiles: TileInput) -> int:
    return int(tiles.shape[0]) if isinstance(tiles, (torch.Tensor, np.ndarray)) else len(tiles)
