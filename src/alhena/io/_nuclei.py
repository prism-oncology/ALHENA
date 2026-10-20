"""Read nuclei with the reader their source selects."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alhena._errors import AlhenaValueError
from alhena.io._centroids import read_centroids
from alhena.io._files import _is_geodataframe, _is_geoparquet
from alhena.io._formats import (
    GEOJSON_SUFFIXES,
    HDF5_SUFFIXES,
    LABEL_SUFFIXES,
    NUCLEI_KINDS,
    TABLE_SUFFIXES,
)
from alhena.io._labels import _is_label_image, read_label_image, read_label_maps
from alhena.io._polygons import read_polygons
from alhena.io._sources import TileSource


def read_nuclei(
    source: Any,
    *,
    kind: str | None = None,
    tiles: TileSource | np.ndarray | Sequence[np.ndarray] | None = None,
    progress: bool | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Read nuclei into the mapping `CellTyper.annotate` reads, with the reader the source selects.

    The reader follows from what the source is, never from its values, unless `kind` names it. A
    table of boundary vertices is read with `kind="polygons"`.

    Args:
        source: Label maps, a DataFrame, or the path of a directory, an HDF5 file, a label image, a
            table, a GeoJSON file or a GeoParquet file.
        kind: One of `NUCLEI_KINDS`, naming the reader.
        tiles: The tiles these nuclei were segmented on, a tile source or tiles in memory.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.
        **options: Passed on to the reader selected.

    Returns:
        A mapping holding the nuclei under `label_maps`, `centroids` or `polygons`, with the
        arguments their builder needs.

    Raises:
        AlhenaValueError: If the kind or an option is unknown, or the source cannot be read.
        GeometryError: If tiles given in memory are not uint8 RGB.
        MissingExtraError: If a GeoParquet file is given and geopandas is not installed.
    """
    selected = _nuclei_kind(source) if kind is None else kind
    readers: dict[str, Callable[..., dict[str, Any]]] = {
        "label_maps": read_label_maps,
        "centroids": read_centroids,
        "polygons": read_polygons,
    }
    if selected not in readers:
        raise AlhenaValueError(
            f"Unknown kind {selected!r}. Nuclei are read as one of {NUCLEI_KINDS}."
        )
    reader = readers[selected]
    if selected == NUCLEI_KINDS[0] and _is_label_image(source):
        reader = read_label_image
    accepted = [name for name in inspect.signature(reader).parameters if name != "tiles"][1:]
    unknown = sorted(set(options) - set(accepted))
    if unknown:
        offered = [name for name in accepted if name != "progress"]
        raise AlhenaValueError(
            f"{unknown} are not options of the reader {reader.__name__}, which reads these "
            f"nuclei and takes {offered}."
        )
    if "progress" in accepted:
        options["progress"] = progress
    return reader(source, tiles=tiles, **options)


def _nuclei_kind(source: Any) -> str:
    """Return which reader a nucleus source selects, from what the source is, never its values."""
    if isinstance(source, pd.DataFrame):
        return "polygons" if _is_geodataframe(source) else "centroids"
    if not isinstance(source, str | Path):
        return "label_maps"
    path = Path(source)
    if path.is_dir():
        return "label_maps"
    suffix = path.suffix.lower()
    if suffix in HDF5_SUFFIXES or suffix in LABEL_SUFFIXES:
        return "label_maps"
    if suffix in TABLE_SUFFIXES:
        return "polygons" if _is_geoparquet(path) else "centroids"
    if suffix in GEOJSON_SUFFIXES:
        return "polygons"
    raise AlhenaValueError(
        f"{path.name} has suffix {path.suffix!r}, which names no kind of nucleus input. Pass "
        "kind= to name the reader."
    )
