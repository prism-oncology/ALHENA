"""Cells and their annotations written into a table of a SpatialData object."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from alhena._errors import AlhenaValueError
from alhena._layout import CELL_ID_COLUMN, CELLS_PROVENANCE_KEY, SPATIAL_KEY, _annotation_name
from alhena.io import add_annotation
from alhena.spatial._elements import CELL_TABLE_KEY, INSTANCE_KEY, REGION_KEY, _element, _require
from alhena.types import AnnotationResult, CellEmbeddings

if TYPE_CHECKING:
    from anndata import AnnData

_INTEGER: Final[str] = "integer"
_STRING: Final[str] = "string"


def cells_to_spatialdata(
    sdata: Any,
    cells: CellEmbeddings | AnnotationResult,
    region: str,
    table_key: str | None = None,
    region_key: str = REGION_KEY,
    instance_key: str = INSTANCE_KEY,
    name: str | None = None,
    overwrite: bool = False,
) -> AnnData:
    """Write cells, or their annotations, into the table of cells of the element of their nuclei.

    The first call creates the table `write_h5ad` writes, tied to the element by `region_key` and
    `instance_key`; a later call with annotations of the same cells adds them, as
    `alhena.io.add_annotation` does. The cell identifiers must be instances of the element. The
    positions are those the cells record, else the boundary centroids, with no resolution recorded.

    Args:
        sdata: The object the table is stored in.
        cells: The cells `encode_cells` returns, or the annotations `CellTyper.annotate` returns,
            which carry them.
        region: Element the table annotates, whose instances the cell identifiers name.
        table_key: Name the table of cells is stored under, `CELL_TABLE_KEY` when None.
        region_key: Column naming the annotated element.
        instance_key: Column holding the instance identifiers.
        name: Name the fields of an annotation carry. Defaults to the axis of the annotations.
        overwrite: Replace the annotation the table holds under `name`, and that one alone; for
            cells, replace the table they would be written to.

    Returns:
        The table of cells, stored in the object, with the annotation in it where one was given.

    Raises:
        AlhenaValueError: If the identifiers do not match the element, or a stored table conflicts.
        MissingExtraError: If the spatial extra is not installed.
    """
    annotated = isinstance(cells, AnnotationResult)
    if not annotated and not isinstance(cells, CellEmbeddings):
        raise AlhenaValueError(
            "cells_to_spatialdata takes the object first, the cells or their annotations second "
            "and the element they annotate third, cells_to_spatialdata(sdata, cells, region), and "
            f"was given an object of type {type(cells).__name__} as the cells."
        )
    if not annotated and name is not None:
        raise AlhenaValueError(
            "name names an annotation, and cells carry none. Pass the annotations, or leave name "
            "out."
        )
    models = _require("spatialdata.models")
    chosen = _annotation_name(name, cells.axis) if annotated else None
    key = CELL_TABLE_KEY if table_key is None else str(table_key)

    identifiers = np.asarray(cells.cell_id)
    n_cells = identifiers.shape[0]
    if n_cells == 0:
        raise AlhenaValueError(
            "There are no cells to write: a table with no row annotates nothing."
        )
    element = _element(sdata, region)
    instances = _element_instances(element)
    positions = _check_instances(identifiers, instances, region)

    stored = getattr(sdata, "tables", {})
    if key in stored and (annotated or not overwrite):
        table = stored[key]
        _check_tied(table, key, (str(region), str(region_key), str(instance_key)))
        if not annotated:
            raise AlhenaValueError(
                f"A table of cells is stored under {key!r} already. Pass overwrite=True to replace "
                "it, which drops the annotations it holds, or add annotations of its cells with "
                "their AnnotationResult."
            )
        return add_annotation(table, cells, chosen, overwrite=overwrite)

    table = cells.to_anndata(chosen) if annotated else cells.to_anndata()
    taken = [region_key] if region_key in table.obs.columns else []
    if instance_key != CELL_ID_COLUMN and instance_key in table.obs.columns:
        taken.append(instance_key)
    if taken:
        raise AlhenaValueError(
            f"{taken} already name columns of the table of cells. Name the region and instance "
            "columns otherwise."
        )

    table.obs[region_key] = pd.Categorical([str(region)] * n_cells)
    if instance_key != CELL_ID_COLUMN:
        table.obs[instance_key] = identifiers
    record = table.uns[CELLS_PROVENANCE_KEY]
    if SPATIAL_KEY not in table.obsm:
        centroids = _centroids(element, positions)
        if centroids is not None:
            table.obsm[SPATIAL_KEY] = centroids
            record["centroids_frame"] = f"element {str(region)!r}"
            # The coordinate system of the element carries no pixel size the cells could state.
            record.pop("centroids_mpp", None)
    record.update(
        {"region": str(region), "region_key": str(region_key), "instance_key": str(instance_key)}
    )

    table = models.TableModel.parse(
        table, region=str(region), region_key=region_key, instance_key=instance_key
    )
    sdata.tables[key] = table
    return table


# Provenance entries naming the element a table annotates and the columns tying it there.
_TIED_BY: Final[tuple[str, ...]] = ("region", "region_key", "instance_key")


def _check_tied(table: Any, table_key: str, asked: tuple[str, str, str]) -> None:
    """Refuse a stored table that is not a table of cells tied to the element as asked."""
    record = getattr(table, "uns", {}).get(CELLS_PROVENANCE_KEY)
    if not isinstance(record, Mapping):
        raise AlhenaValueError(
            f"A table is already stored under {table_key!r}, and it is not a table of cells "
            "ALHENA wrote. Name another table with table_key=, or remove that one first."
        )
    if record.get(_TIED_BY[0]) is None:
        raise AlhenaValueError(
            f"The table of cells stored under {table_key!r} records no element it annotates, so "
            "cells_to_spatialdata did not store it. Add the annotations to it with "
            "alhena.io.add_annotation, or name another table with table_key=."
        )
    tied = tuple(str(record.get(key)) for key in _TIED_BY)
    if tied != asked:
        raise AlhenaValueError(
            f"The table of cells stored under {table_key!r} annotates {tied[0]!r} through the "
            f"columns {tied[1]!r} and {tied[2]!r}, and this call names {asked[0]!r} through "
            f"{asked[1]!r} and {asked[2]!r}. Add the annotations of its cells with the element "
            "and the columns it is tied by, or store other cells under another table_key."
        )


def _element_instances(element: Any) -> np.ndarray:
    """Return the instances of an element, which are what a table annotating it must name."""
    spatialdata = _require("spatialdata")
    return np.asarray(spatialdata.get_element_instances(element))


def _identifier_kind(values: np.ndarray, described: str) -> str:
    """Return whether identifiers are integers or strings, refusing anything else."""
    inferred = pd.api.types.infer_dtype(values, skipna=False)
    if inferred == _INTEGER:
        return _INTEGER
    if inferred == _STRING:
        return _STRING
    raise AlhenaValueError(
        f"{described} are {inferred}, where a table annotates an element by integer or string "
        "identifiers. Name the cells with the index values of the element they were segmented "
        "from."
    )


def _check_instances(identifiers: np.ndarray, instances: np.ndarray, region: str) -> np.ndarray:
    """Return the row of the element each cell names, refusing identifiers it does not hold."""
    kind = _identifier_kind(identifiers, "The cell identifiers")
    element_kind = _identifier_kind(instances, f"The instances of {region!r}")
    if kind != element_kind:
        raise AlhenaValueError(
            f"The cells are named by {kind} identifiers and {region!r} holds {element_kind} "
            "instances. Name the cells with the index values of the element."
        )

    known = pd.Index(instances)
    if not known.is_unique:
        raise AlhenaValueError(
            f"The instances of {region!r} repeat a value, so a cell identifier names more than "
            "one instance of it."
        )
    wanted = pd.Index(identifiers)
    if not wanted.is_unique:
        repeated = int(wanted.duplicated().sum())
        raise AlhenaValueError(
            f"{repeated} cell identifiers repeat. An identifier names one cell across the whole "
            "cohort, never one cell per tile."
        )

    positions = known.get_indexer(wanted)
    absent = np.flatnonzero(positions < 0)
    if absent.size:
        examples = [str(value) for value in wanted[absent[:3]]]
        raise AlhenaValueError(
            f"{absent.size} of {wanted.size} cell identifiers name no instance of {region!r}, for "
            f"example {examples}. Annotate the element the nuclei were read from."
        )
    return positions.astype(np.intp, copy=False)


def _centroids(element: Any, positions: np.ndarray) -> np.ndarray | None:
    """Return the centroid of each annotated boundary, or None for an element holding none."""
    geometry = getattr(element, "geometry", None)
    if geometry is None or not hasattr(element, "bounds"):
        return None
    centroids = geometry.iloc[positions].centroid
    return np.column_stack(
        [centroids.x.to_numpy(dtype=np.float64), centroids.y.to_numpy(dtype=np.float64)]
    )
