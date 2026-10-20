"""Nuclei read as polygons, from GeoJSON, GeoParquet or a table of vertices."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alhena import _progress
from alhena._errors import PACKAGE_PREFIXES, AlhenaValueError, GeometryWarning, MissingExtraError
from alhena._rings import _ring_areas
from alhena.io._carry import _apply, _carrier, _extent
from alhena.io._files import (
    _choice,
    _column_names,
    _coordinates,
    _identifier_columns,
    _is_geodataframe,
    _is_geoparquet,
    _named,
    _pick_column,
    _table,
    _text_identifiers,
)
from alhena.io._formats import (
    BOUNDARY_TABLE_SUFFIXES,
    COORDINATE_UNITS,
    MULTIPART_RULES,
    PART_SEPARATOR,
    VERTEX_COLUMNS,
)
from alhena.io._sources import TileSource, as_tile_source, tile_bounds


def read_polygons(
    source: str | Path | pd.DataFrame,
    *,
    tiles: TileSource | None = None,
    id_property: str = "cell_id",
    geometry_property: str | None = None,
    columns: Mapping[str, str] | None = None,
    units: str = "px",
    transform: np.ndarray | Sequence[Sequence[float]] | str | Path | None = None,
    multipart: str = "refuse",
    progress: bool | None = None,
) -> dict[str, Any]:
    """Read nucleus boundaries from GeoJSON, a table of vertices or GeoParquet.

    Boundaries are read in the frame of the tile origins, so a nucleus in several tiles is found in
    each. A file of `BOUNDARY_TABLE_SUFFIXES` is a table of vertices, one row per vertex in outline
    order, or GeoParquet where the file declares itself one; any other file is GeoJSON: a feature
    collection, a list of features, one feature or one geometry. A boundary of several polygons is
    read under `multipart`.

    Args:
        source: A GeoDataFrame, a DataFrame of vertices, or the path of a GeoJSON file, a table of
            vertices or a GeoParquet file.
        tiles: The tiles the boundaries are placed against, whose windows are added to the result.
            Required with `units="um"`.
        id_property: Property of a GeoJSON feature naming the cell; where absent, the feature `id`,
            then its position.
        geometry_property: Member of a GeoJSON feature holding the nucleus outline, read instead of
            its geometry: beside the geometry, else among the properties.
        columns: Columns holding `cell_id`, `x` and `y` of a table of vertices, or `cell_id` of
            GeoParquet, where named otherwise.
        units: One of `COORDINATE_UNITS`.
        transform: A 2 by 3 or 3 by 3 affine matrix carrying the coordinates into the frame of the
            tile origins, or the path of a `.csv` or `.npy` file holding one.
        multipart: One of `MULTIPART_RULES`.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        A mapping holding `polygons` and `cell_ids`, and where the tiles were given `tile_bounds`,
        `tile_scale` and `tile_frame`, with which other tiles of that frame are refused.

    Raises:
        AlhenaValueError: If the source is unreadable, names a cell twice, or meets no tile.
        MissingExtraError: If a GeoParquet file is given and geopandas is not installed.
    """
    tiles = None if tiles is None else as_tile_source(tiles)
    units = _choice(units, COORDINATE_UNITS, "units")
    multipart = _choice(multipart, MULTIPART_RULES, "multipart")
    form = _boundary_form(source)
    named = _named(source)
    if form == "geojson":
        if columns is not None:
            raise AlhenaValueError(
                f"columns names the columns of a table, and {named} is read as GeoJSON, whose "
                "features are named by id_property."
            )
    else:
        misplaced = [
            name
            for name, given in (
                ("geometry_property", geometry_property is not None),
                ("id_property", id_property != "cell_id"),
            )
            if given
        ]
        if misplaced:
            raise AlhenaValueError(
                f"{' and '.join(misplaced)} name members of a GeoJSON feature, and {named} is "
                "read as a table. The columns of a table are named through columns=."
            )
    carrier = _carrier(units, transform, tiles, origins=True)

    area: Callable[[Any], float] = _part_area
    if form == "geojson":
        boundaries, identifiers = _geojson_boundaries(
            Path(source), id_property, geometry_property, progress
        )
    elif form == "geoparquet":
        held = isinstance(source, pd.DataFrame)
        chosen = _column_names(
            columns,
            {"cell_id": VERTEX_COLUMNS["cell_id"]},
            "a GeoDataFrame" if held else "a GeoParquet file",
        )
        geometries = source if held else _read_geoparquet(Path(source))
        boundaries, identifiers = _geoparquet_boundaries(geometries, chosen, named, progress)
        area = _geometry_area
    else:
        chosen = _column_names(columns, VERTEX_COLUMNS, "a table of boundary vertices")
        table = _table(
            source, "nucleus boundaries", id_columns=_identifier_columns(chosen, VERTEX_COLUMNS)
        )
        boundaries, identifiers = _vertex_boundaries(table, chosen, named, progress)

    kept, identifiers = _select_parts(
        boundaries, identifiers, multipart, named, area, rings=form == "vertices"
    )
    if form == "geoparquet":
        kept = _read_geometry_parts(kept)
    repeated = pd.Index(identifiers).duplicated()
    if repeated.any():
        cell = identifiers.tolist()[int(np.flatnonzero(repeated)[0])]
        raise AlhenaValueError(
            f"Two boundaries of {named} name the same cell, {cell!r}. Name each cell once."
        )
    polygons = [outline for outline, _ in kept]
    _warn_holes(kept, named)
    if carrier is not None:
        polygons = _carry_outlines(polygons, carrier)

    payload: dict[str, Any] = {"polygons": polygons, "cell_ids": identifiers}
    if tiles is not None:
        bounds = tile_bounds(tiles)
        _check_boundaries_meet(
            polygons,
            bounds,
            named,
            "in the frame of their origins, in which boundaries are read",
            "Check units= and transform=, and that these are the tiles the boundaries were "
            "segmented on.",
        )
        payload["tile_bounds"] = bounds
        payload["tile_scale"] = float(tiles.coords_scale)
        # So that these nuclei are refused beside other tiles of the same frame.
        payload["tile_frame"] = tiles.coords_frame
    return payload


def _read_geojson(path: Path) -> list[Any]:
    """Return the features of a GeoJSON file."""
    if not path.is_file():
        raise AlhenaValueError(f"No polygon file at {path}.")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AlhenaValueError(f"{path} is not readable as GeoJSON: {error}.") from error

    if isinstance(document, list):
        features = document
    elif isinstance(document, Mapping):
        kind = str(document.get("type", ""))
        if kind == "FeatureCollection":
            features = list(document.get("features") or [])
        elif kind in {"Feature", "Polygon", "MultiPolygon"}:
            features = [document]
        else:
            raise AlhenaValueError(
                f"{path} holds an object of type {kind!r}. Nuclei are read from a "
                "FeatureCollection, a list of features, a single feature, or a single geometry."
            )
    else:
        raise AlhenaValueError(f"{path} holds no feature.")
    if not features:
        raise AlhenaValueError(f"{path} holds no nucleus boundary.")
    return features


def _boundary_form(source: Any) -> str:
    """Return which form boundaries are read as: GeoJSON, vertices or GeoParquet."""
    if isinstance(source, pd.DataFrame):
        return "geoparquet" if _is_geodataframe(source) else "vertices"
    if not isinstance(source, str | os.PathLike):
        raise AlhenaValueError(
            f"Boundaries of type {type(source).__name__} are not read. Pass a GeoDataFrame, a "
            "DataFrame of vertices, or the path of a GeoJSON file, a table of vertices or a "
            "GeoParquet file."
        )
    file = Path(source)
    if file.suffix.lower() not in BOUNDARY_TABLE_SUFFIXES:
        return "geojson"
    return "geoparquet" if _is_geoparquet(file) else "vertices"


# A polygon as the readers hand it on: its outer ring, without the closing vertex, and holes.
_Polygon = tuple[np.ndarray, list[np.ndarray]]


def _geojson_boundaries(
    file: Path, id_property: str, geometry_property: str | None, progress: bool | None
) -> tuple[list[list[_Polygon]], np.ndarray]:
    """Return the polygons of every feature of a GeoJSON file, and the cell each feature names."""
    features = _read_geojson(file)
    named = [_feature_id(feature, id_property) for feature in features]
    known = [value for value in named if value is not None]
    if known and len(known) != len(named):
        raise AlhenaValueError(
            f"{len(known)} of {len(named)} boundaries of {file.name} name a cell. Name all of "
            "them or none."
        )
    identifiers = np.asarray(known) if known else np.arange(len(features))

    boundaries: list[list[_Polygon]] = []
    lacking: list[int] = []
    with _progress.track(
        features, len(features), "Reading nucleus boundaries", "boundary", progress
    ) as rows:
        for position, feature in enumerate(rows):
            if not isinstance(feature, Mapping):
                raise AlhenaValueError(f"Boundary {position} is not a GeoJSON object.")
            geometry = _feature_geometry(feature, geometry_property)
            if geometry is None:
                lacking.append(position)
                boundaries.append([])
                continue
            boundaries.append(_geojson_polygons(geometry, position))
    if lacking:
        one = len(lacking) == 1
        if geometry_property is None:
            lacks = f"{'carries' if one else 'carry'} a null geometry"
        else:
            lacks = (
                f"{'holds' if one else 'hold'} no {geometry_property!r}, beside the geometry or "
                "among the properties"
            )
        raise AlhenaValueError(
            f"{len(lacking)} of the {len(features)} features of {file.name} {lacks}, the first "
            f"feature {lacking[0]}. Give every feature an outline."
        )
    return boundaries, identifiers


def _feature_geometry(feature: Mapping[str, Any], geometry_property: str | None) -> Any:
    """Return the outline of a feature: the member named, by its geometry or in its properties."""
    if geometry_property is None:
        return feature.get("geometry", feature)
    if feature.get(geometry_property) is not None:
        return feature[geometry_property]
    properties = feature.get("properties")
    if isinstance(properties, Mapping):
        return properties.get(geometry_property)
    return None


def _geojson_polygons(geometry: Any, position: int) -> list[_Polygon]:
    """Return the polygons one GeoJSON geometry holds."""
    if not isinstance(geometry, Mapping):
        raise AlhenaValueError(f"Boundary {position} carries no geometry.")
    kind = str(geometry.get("type", ""))
    coordinates = geometry.get("coordinates")
    if kind == "Polygon":
        polygons = [coordinates]
    elif kind == "MultiPolygon":
        polygons = list(coordinates or [])
    else:
        raise _not_polygonal(f"Boundary {position}", kind)
    if not polygons:
        raise AlhenaValueError(f"Boundary {position} holds no polygon.")
    described = f"Boundary {position}"
    read: list[_Polygon] = []
    for rings in polygons:
        # A GeoJSON polygon is an array of rings, which JSON reads as a list.
        if not isinstance(rings, list) or not rings:
            raise AlhenaValueError(f"{described} holds no ring.")
        read.append(
            (_outline_ring(rings[0], described), [_ring(ring, described) for ring in rings[1:]])
        )
    return read


def _ring(values: Any, described: str) -> np.ndarray:
    """Return one ring of a polygon as an (m, 2) array of finite x and y."""
    try:
        points = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(
            f"{described} holds a ring that is not an array of vertices."
        ) from error
    if points.ndim != 2 or points.shape[1] < 2:
        raise AlhenaValueError(
            f"{described} has shape {points.shape}, where an (m, 2) ring of vertices is expected."
        )
    points = points[:, :2]
    if not np.isfinite(points).all():
        raise AlhenaValueError(f"{described} holds a non-finite vertex.")
    return points


def _outline_ring(values: Any, described: str) -> np.ndarray:
    """Return the outer ring of a polygon without its closing vertex, refusing fewer than three."""
    points = _ring(values, described)
    if points.shape[0] > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if points.shape[0] < 3:
        raise AlhenaValueError(
            f"{described} has {points.shape[0]} distinct vertices; a nucleus boundary needs at "
            "least three."
        )
    return points


def _geopandas() -> Any:
    """Return geopandas, imported on use so that reading anything else never needs it."""
    try:
        import geopandas
    except ImportError as error:
        raise MissingExtraError(
            'Reading GeoParquet needs geopandas. Install it with pip install "alhena[spatial]", '
            "or pip install geopandas."
        ) from error
    return geopandas


def _read_geoparquet(file: Path) -> Any:
    """Return the GeoDataFrame a GeoParquet file holds."""
    geopandas = _geopandas()
    try:
        return geopandas.read_parquet(file)
    except (OSError, ValueError, TypeError) as error:
        raise AlhenaValueError(f"{file} could not be read as GeoParquet: {error}") from error


def _geoparquet_boundaries(
    frame: Any, chosen: Mapping[str, str], named: str, progress: bool | None
) -> tuple[list[list[_GeometryPart]], np.ndarray]:
    """Return the polygons of every row of a GeoDataFrame, and the cell each row names."""
    try:
        geometries = frame.geometry
    except AttributeError as error:
        raise AlhenaValueError(
            f"No active geometry column in {named}. Name the column holding the outlines with "
            "set_geometry()."
        ) from error
    if len(frame) == 0:
        raise AlhenaValueError(f"No nucleus boundary in {named}.")
    column = _pick_column(
        frame, chosen.get("cell_id"), VERTEX_COLUMNS["cell_id"], "cell identifier", required=False
    )
    identifiers = np.asarray(frame.index.to_numpy() if column is None else frame[column].to_numpy())
    if pd.isna(identifiers).any():
        raise AlhenaValueError(
            f"The cells of {named} are named with a missing value; every nucleus must be named."
        )
    described = f"The index of {named}" if column is None else f"Column {column!r} of {named}"
    identifiers = _text_identifiers(identifiers, described)
    return _geometry_parts(geometries, len(frame), progress), identifiers


# A polygon of a geometry column: its boundary as messages name it, the polygon, its read form.
_GeometryPart = tuple[str, Any, _Polygon | None]


def _geometry_parts(
    geometries: Any, count: int, progress: bool | None
) -> list[list[_GeometryPart]]:
    """Return the polygons of every geometry, each read once no rule can drop it."""
    boundaries: list[list[_GeometryPart]] = []
    with _progress.track(
        geometries, count, "Reading nucleus boundaries", "boundary", progress
    ) as rows:
        for position, geometry in enumerate(rows):
            described = f"Boundary {position}"
            polygons = _polygons_of(geometry, described)
            # No rule drops the polygon of a boundary of one, so it is read as the bar passes it.
            read = _shapely_polygon(polygons[0], described) if len(polygons) == 1 else None
            boundaries.append([(described, polygon, read) for polygon in polygons])
    return boundaries


def _read_geometry_parts(parts: Sequence[_GeometryPart]) -> list[_Polygon]:
    """Return the outline and holes of each polygon kept, reading those not read yet."""
    return [
        _shapely_polygon(polygon, described) if read is None else read
        for described, polygon, read in parts
    ]


def _geometry_area(part: _GeometryPart) -> float:
    """Return the area a polygon of a geometry column encloses, its holes taken out."""
    return float(part[1].area)


def _polygons_of(geometry: Any, described: str) -> list[Any]:
    """Return the polygons one geometry holds, read through the interface shapely gives it."""
    if geometry is None or getattr(geometry, "is_empty", False):
        raise AlhenaValueError(f"{described} is empty.")
    kind = getattr(geometry, "geom_type", None)
    if kind == "Polygon":
        polygons = [geometry]
    elif kind == "MultiPolygon":
        polygons = list(geometry.geoms)
    else:
        raise _not_polygonal(described, kind)
    if not polygons:
        raise AlhenaValueError(f"{described} holds no polygon.")
    return polygons


def _not_polygonal(described: str, kind: Any) -> AlhenaValueError:
    """Return the refusal of a geometry that is neither a Polygon nor a MultiPolygon."""
    return AlhenaValueError(
        f"{described} has geometry type {kind!r}. Nuclei are read from Polygon and MultiPolygon "
        "geometries; positions with a radius are read by alhena.cells.membership_from_centroids, "
        "which reports its own approximation."
    )


def _shapely_polygon(polygon: Any, described: str) -> _Polygon:
    """Return the outline and the holes of one polygon of a geometry column."""
    return (
        _outline_ring(np.asarray(polygon.exterior.coords), described),
        [_ring(np.asarray(hole.coords), described) for hole in polygon.interiors],
    )


def _vertex_boundaries(
    table: pd.DataFrame, chosen: Mapping[str, str], named: str, progress: bool | None
) -> tuple[list[list[_Polygon]], np.ndarray]:
    """Return the polygons of each cell of a table of vertices, one per ring, and the cell names."""
    id_column = _pick_column(
        table,
        chosen.get("cell_id"),
        VERTEX_COLUMNS["cell_id"],
        "cell identifier",
        advice="A table of vertices groups its rows into outlines by the cell each row names.",
    )
    x_column = _pick_column(table, chosen.get("x"), VERTEX_COLUMNS["x"], "x vertex")
    y_column = _pick_column(table, chosen.get("y"), VERTEX_COLUMNS["y"], "y vertex")
    if len(table) == 0:
        raise AlhenaValueError(f"No vertex in {named}.")
    points = _coordinates(table, (x_column, y_column), named, "vertex")
    cells = table[id_column].to_numpy()
    if pd.isna(cells).any():
        raise AlhenaValueError(
            f"Column {id_column!r} of {named} holds a missing value; every vertex names the "
            "cell it outlines."
        )
    codes, uniques = pd.factorize(cells)
    # Decoded once factorized, so an identifier repeated on every vertex of its cell is read once.
    identifiers = _text_identifiers(np.asarray(uniques), f"Column {id_column!r} of {named}")
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    if starts.shape[0] != identifiers.shape[0]:
        runs = codes[starts]
        again = int(np.flatnonzero(pd.Index(runs).duplicated())[0])
        cell = identifiers.tolist()[int(runs[again])]
        raise AlhenaValueError(
            f"The vertices of cell {cell!r} are not consecutive rows of {named}: they resume "
            f"at row {int(starts[again])}, after rows of another cell. Write the vertices of each "
            "cell as one run of rows, in outline order."
        )
    stops = np.r_[starts[1:], codes.shape[0]]
    several = _rings_before_the_end(points, starts, stops)
    names = identifiers.tolist()
    boundaries: list[list[_Polygon]] = []
    with _progress.track(
        range(starts.shape[0]),
        starts.shape[0],
        "Reading nucleus boundaries",
        "boundary",
        progress,
    ) as runs_read:
        for run in runs_read:
            described = f"Cell {names[run]!r}"
            vertices = points[starts[run] : stops[run]]
            rings = _closed_rings(vertices) if several[run] else [vertices]
            boundaries.append([(_outline_ring(ring, described), []) for ring in rings])
    return boundaries, identifiers


def _rings_before_the_end(points: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Return which runs of vertices close a ring before their last vertex, holding several."""
    lengths = stops - starts
    firsts = np.repeat(starts, lengths)
    offsets = np.arange(points.shape[0]) - firsts
    # A vertex repeating the first of its run, three or more vertices on, closes a ring there.
    closing = (offsets >= 3) & (points == points[firsts]).all(axis=1)
    closing[stops - 1] = False
    return np.logical_or.reduceat(closing, starts)


def _closed_rings(points: np.ndarray) -> list[np.ndarray]:
    """Return the rings of a run of vertices, each closed by a vertex repeating its first."""
    rings: list[np.ndarray] = []
    start = 0
    while start < points.shape[0]:
        repeats = np.flatnonzero((points[start + 3 :] == points[start]).all(axis=1))
        stop = points.shape[0] if repeats.size == 0 else start + 4 + int(repeats[0])
        rings.append(points[start:stop])
        start = stop
    return rings


def _select_parts(
    boundaries: Sequence[Sequence[Any]],
    identifiers: np.ndarray,
    multipart: str,
    described: str,
    area: Callable[[Any], float],
    rings: bool = False,
) -> tuple[list[Any], np.ndarray]:
    """Return the polygons read as nuclei under the multipart rule, and the name of each."""
    kept: list[Any] = []
    chosen: list[list[int]] = []
    several: list[int] = []
    for position, polygons in enumerate(boundaries):
        count = len(polygons)
        if count == 1 or multipart == MULTIPART_RULES[1]:
            parts = list(range(count))
        elif multipart == MULTIPART_RULES[2]:
            parts = [int(np.argmax([area(polygon) for polygon in polygons]))]
        else:
            parts = []
        if count > 1:
            several.append(position)
        chosen.append(parts)
        kept.extend(polygons[part] for part in parts)
    one = len(several) == 1
    if several and multipart == MULTIPART_RULES[0]:
        first = several[0]
        count = len(boundaries[first])
        held = (
            f"is outlined by {count} closed rings"
            if rings
            else f"is a MultiPolygon of {count} parts"
        )
        raise AlhenaValueError(
            f"{len(several)} of the {len(boundaries)} boundaries of {described} "
            f"{'holds' if one else 'hold'} several polygons: cell {identifiers.tolist()[first]!r} "
            f"{held}. Pass multipart='split' to read each polygon as a nucleus, or "
            "multipart='largest' to keep the largest."
        )
    if several and multipart == MULTIPART_RULES[2]:
        warnings.warn(
            f"{len(several)} of the boundaries of {described} {'is' if one else 'are'} "
            f"multi-part, and {'it was' if one else 'each was'} reduced to its polygon enclosing "
            "the largest area. The pixels of the polygons dropped support no token, and the cell "
            "is pooled over the polygon kept.",
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
    return kept, _part_identifiers(identifiers, chosen, multipart)


def _part_identifiers(
    identifiers: np.ndarray, chosen: Sequence[Sequence[int]], multipart: str
) -> np.ndarray:
    """Return the identifier of each polygon kept; under `"split"`, every identifier is text."""
    values = np.asarray(identifiers)
    if multipart != MULTIPART_RULES[1]:
        return values
    names: list[str] = []
    for identifier, parts in zip(values.tolist(), chosen, strict=True):
        if len(parts) == 1:
            names.append(str(identifier))
        else:
            names.extend(f"{identifier}{PART_SEPARATOR}{part}" for part in parts)
    return np.asarray(names, dtype=str)


def _part_area(polygon: _Polygon) -> float:
    """Return the area a polygon encloses, its holes taken out."""
    outline, holes = polygon
    areas = np.abs(_ring_areas([outline, *holes]))
    return float(areas[0] - areas[1:].sum())


def _warn_holes(polygons: Sequence[_Polygon], named: str) -> None:
    """Warn that the holes of the polygons read are filled when they are rasterized."""
    holed = sum(1 for _, holes in polygons if holes)
    if holed:
        warnings.warn(
            f"{holed} of the boundaries of {named} {'encloses' if holed == 1 else 'enclose'} "
            "a hole, which is filled when the boundary is rasterized. Those cells pool over the "
            "interior of their outline.",
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )


def _carry_outlines(outlines: list[np.ndarray], carrier: np.ndarray) -> list[np.ndarray]:
    """Return outlines with every vertex carried by an affine matrix, in one product."""
    lengths = [outline.shape[0] for outline in outlines]
    carried = _apply(carrier, np.concatenate(outlines, axis=0))
    return np.split(carried, np.cumsum(lengths)[:-1])


def _check_boundaries_meet(
    polygons: list[np.ndarray], bounds: np.ndarray, named: str, frame: str, advice: str
) -> None:
    """Refuse boundaries that meet no tile window, the sign that the two lie in other frames."""
    stacked = np.concatenate(polygons, axis=0)
    low = stacked.min(axis=0)
    high = stacked.max(axis=0)
    # A window holds its lower bound and not its upper one, as the rasterizer reads it.
    if (
        low[0] < bounds[:, 2].max()
        and high[0] >= bounds[:, 0].min()
        and low[1] < bounds[:, 3].max()
        and high[1] >= bounds[:, 1].min()
    ):
        return
    raise AlhenaValueError(
        f"The boundaries of {named} span {_extent(stacked)} and meet no tile window: the tiles "
        f"span {_extent(bounds.reshape(-1, 2))} {frame}. {advice}"
    )


def _feature_id(feature: Any, id_property: str) -> Any:
    """Return the cell a feature names, or None where it names none."""
    if not isinstance(feature, Mapping):
        return None
    properties = feature.get("properties")
    if isinstance(properties, Mapping) and properties.get(id_property) is not None:
        return properties[id_property]
    return feature.get("id")
