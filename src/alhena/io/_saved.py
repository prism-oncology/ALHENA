"""Read back the embeddings and annotations written earlier, and add annotations to cells."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alhena._errors import AlhenaValueError
from alhena._layout import (
    BACKBONE_KEY,
    CANDIDATE_TILES_COLUMN,
    CELL_ID_COLUMN,
    CELL_SPACE_KEY,
    CELLS_PROVENANCE_KEY,
    CLASS_ENTROPY_COLUMN,
    CONFIDENCE_COLUMN,
    EMBEDDING_VALID_COLUMN,
    LABEL_COLUMN,
    NEIGHBOR_ENTROPY_COLUMN,
    NUCLEUS_PX_COLUMN,
    SPATIAL_KEY,
    TILE_BOUNDS_COLUMNS,
    TILE_EMBEDDING_KEY,
    TILE_ID_COLUMN,
    TILE_INDEX_COLUMN,
    TILES_PROVENANCE_KEY,
    VALID_COLUMN,
    _anndata,
    _annotation_key,
    _annotation_name,
    _annotation_records,
    _check_annotated_cells,
    _plain,
    _quiet_index_notice,
    _write_annotation,
)
from alhena.types import (
    AnnotationResult,
    CellEmbeddings,
    GeometryReport,
    ResampleInfo,
    TileEmbeddings,
)

if TYPE_CHECKING:
    from anndata import AnnData

_CELL_COLUMNS: tuple[str, ...] = (
    CELL_ID_COLUMN,
    TILE_INDEX_COLUMN,
    NUCLEUS_PX_COLUMN,
    CANDIDATE_TILES_COLUMN,
    EMBEDDING_VALID_COLUMN,
)

_TILE_COLUMNS: tuple[str, ...] = (TILE_ID_COLUMN, TILE_INDEX_COLUMN)

# Entries of an annotation's record that lay out the table, and are no provenance of the annotation.
_LAYOUT_KEYS: tuple[str, ...] = ("axis", "class_order", "mass_key")


def read_cells(source: str | Path | AnnData) -> CellEmbeddings:
    """Read cells back from a table ALHENA wrote, to annotate them again without the encoder.

    A table of annotations, or the table `alhena.spatial.cells_to_spatialdata` stores, reads back
    the same way.

    Args:
        source: Path of an `.h5ad` file, or an AnnData table in memory.

    Returns:
        The cells in the order of the table, with the provenance they were written with; `tiles` is
        None.

    Raises:
        AlhenaValueError: If the file is missing, or the table lacks a field the layout carries.
    """
    table = _read_table(source, "cells")
    record = table.uns.get(CELLS_PROVENANCE_KEY)
    if not isinstance(record, Mapping):
        raise AlhenaValueError(
            f"This table carries no {CELLS_PROVENANCE_KEY!r} entry in uns, so it is not a table "
            "of cells written by ALHENA. Write cells with CellEmbeddings.write_h5ad, or "
            "annotations with AnnotationResult.write_h5ad."
        )
    absent = [name for name in _CELL_COLUMNS if name not in table.obs.columns]
    if BACKBONE_KEY not in table.obsm:
        absent.append(BACKBONE_KEY)
    if absent:
        raise AlhenaValueError(
            f"This table lacks {absent}, which every table of cells carries. It may have been "
            "edited since it was written."
        )

    obs = table.obs
    return CellEmbeddings(
        cell_id=_stored_identifiers(obs[CELL_ID_COLUMN], record.get("cell_id_dtype")),
        backbone=np.asarray(table.obsm[BACKBONE_KEY], dtype=np.float32),
        cell_space=_stored_array(table.obsm, CELL_SPACE_KEY, np.float32),
        alpha=obs[NUCLEUS_PX_COLUMN].to_numpy(dtype=np.float32),
        tile_index=obs[TILE_INDEX_COLUMN].to_numpy(),
        n_candidate_tiles=obs[CANDIDATE_TILES_COLUMN].to_numpy(),
        valid=obs[EMBEDDING_VALID_COLUMN].to_numpy(dtype=bool),
        encoder_id=_stored_text(record.get("encoder_id")),
        cell_space_id=_stored_text(record.get("cell_space_id")),
        geometry=_stored_geometry(record.get("geometry")),
        membership_source=_stored_text(record.get("membership_source")),
        membership_approximate=_stored_flag(record.get("membership_approximate")),
        n_cells_without_tokens=_stored_count(record.get("n_cells_without_tokens")),
        centroids=_stored_array(table.obsm, SPATIAL_KEY, np.float64),
        centroids_frame=_stored_text(record.get("centroids_frame")),
        centroids_mpp=_stored_number(record.get("centroids_mpp")),
        backbone_digest=_stored_text(record.get("backbone_digest")),
    )


def read_annotations(source: str | Path | AnnData, *, name: str | None = None) -> AnnotationResult:
    """Read one annotation back from a table ALHENA wrote, with the cells it was made from.

    A table `AnnotationResult.write_h5ad` wrote, `add_annotation` added to or
    `alhena.spatial.cells_to_spatialdata` stored reads back the same way. The similarities of the
    retrieved neighbors are not written, so they read back zero columns wide.

    Args:
        source: Path of an `.h5ad` file, or an AnnData table in memory.
        name: Name the annotation was stored under, its axis unless another was given. May be left
            out where the table holds one annotation.

    Returns:
        The annotation, in the order of the table, with its cells attached as `read_cells` reads
        them.

    Raises:
        AlhenaValueError: If the table is not one ALHENA wrote, or `name` picks no annotation.
    """
    table = _read_table(source, "annotations")
    cells = read_cells(table)
    _, records = _annotation_records(table)
    held = list(records)
    if not held:
        raise AlhenaValueError(
            "This table holds cells and no annotation. Annotate the cells read_cells reads out of "
            "it, or read a table AnnotationResult.write_h5ad wrote."
        )
    if name is None:
        if len(held) > 1:
            raise AlhenaValueError(
                f"This table holds the annotations {held}. Name the one to read with name=."
            )
        chosen = held[0]
    elif name in records:
        chosen = name
    else:
        raise AlhenaValueError(f"This table holds no annotation named {name!r}, only {held}.")

    record = records[chosen]
    if not isinstance(record, Mapping):
        raise AlhenaValueError(
            f"The record of the annotation {chosen!r} is not a mapping. The table may have been "
            "edited since it was written."
        )
    columns = {
        stem: _annotation_key(stem, chosen)
        for stem in (
            LABEL_COLUMN,
            CONFIDENCE_COLUMN,
            VALID_COLUMN,
            CLASS_ENTROPY_COLUMN,
            NEIGHBOR_ENTROPY_COLUMN,
        )
    }
    absent = [key for key in columns.values() if key not in table.obs.columns]
    absent += [key for key in _LAYOUT_KEYS if key not in record]
    mass_key = str(record.get("mass_key", ""))
    if mass_key and mass_key not in table.obsm:
        absent.append(mass_key)
    if absent:
        raise AlhenaValueError(
            f"This table lacks {absent}, which the annotation {chosen!r} was written with. It may "
            "have been edited since it was written."
        )

    obs = table.obs
    labels = obs[columns[LABEL_COLUMN]].astype(object)
    return AnnotationResult(
        cell_id=cells.cell_id,
        labels=np.asarray(labels.where(labels.notna(), "").map(str).tolist(), dtype=str),
        confidence=obs[columns[CONFIDENCE_COLUMN]].to_numpy(dtype=np.float32),
        mass=np.asarray(table.obsm[mass_key], dtype=np.float32),
        class_order=tuple(str(value) for value in np.asarray(record["class_order"]).tolist()),
        class_entropy=obs[columns[CLASS_ENTROPY_COLUMN]].to_numpy(dtype=np.float32),
        neighbor_entropy=obs[columns[NEIGHBOR_ENTROPY_COLUMN]].to_numpy(dtype=np.float32),
        neighbor_similarity=np.empty((len(cells), 0), dtype=np.float32),
        valid=obs[columns[VALID_COLUMN]].to_numpy(dtype=bool),
        axis=str(record["axis"]),
        meta=_plain({key: value for key, value in record.items() if key not in _LAYOUT_KEYS}),
        cells=cells,
    )


def _as_annotations(value: Any, name: str | None) -> AnnotationResult:
    """Return annotations given as a result, or read from a table of cells ALHENA wrote."""
    if isinstance(value, AnnotationResult):
        if name is not None:
            raise AlhenaValueError(
                "name picks one annotation of a table of cells, and annotations were given. "
                "Leave name out."
            )
        return value
    if hasattr(value, "obs") and hasattr(value, "uns"):
        return read_annotations(value, name=name)
    raise AlhenaValueError(
        "Annotations are read from an AnnotationResult or from the table of cells they were "
        f"written to, and an object of type {type(value).__name__} was given."
    )


def add_annotation(
    table: AnnData,
    result: AnnotationResult,
    name: str | None = None,
    overwrite: bool = False,
) -> AnnData:
    """Add an annotation to a table of the cells it was made from, beside those it holds.

    The table is changed in place. Each annotation adds fields named after it, so annotations on
    several axes, or against several references, share one table.

    Args:
        table: A table of cells ALHENA wrote, held in memory, with the cell-space embedding.
        result: The annotations, made from the cells of the table, in its order.
        name: Name the fields of the annotation carry. Defaults to the axis of the annotations.
        overwrite: Replace the annotation stored under `name`, and that one alone.

    Returns:
        The table, with the annotation added.

    Raises:
        AlhenaValueError: If the annotations are of other cells, or the name is unusable or taken.
    """
    if isinstance(table, str | Path):
        raise AlhenaValueError(
            "add_annotation adds to a table held in memory. Read the file with "
            "anndata.read_h5ad, add the annotation to the table, and write the table back with "
            "its write_h5ad."
        )
    if not isinstance(result, AnnotationResult):
        raise AlhenaValueError(
            "add_annotation takes the table first and the annotations second, "
            "add_annotation(table, result), and was given an object of type "
            f"{type(result).__name__} as the annotations. The annotations are the result "
            "CellTyper.annotate returns."
        )
    chosen = _annotation_name(name, result.axis)
    _check_annotated_cells(
        result,
        read_cells(table),
        held="the table",
        advice="Annotate the cells read_cells reads out of the table, in its order, and write "
        "the annotations of other cells to a table of their own.",
    )
    _write_annotation(table, result, chosen, overwrite)
    return table


def read_tile_embeddings(source: str | Path | AnnData) -> TileEmbeddings:
    """Read tile embeddings back from a table ALHENA wrote.

    A table written for tiles sharing no frame reads back without bounds.

    Args:
        source: Path of an `.h5ad` file, or an AnnData table in memory.

    Returns:
        The tiles in the order of the table, with the bounds and the provenance they were written
        with.

    Raises:
        AlhenaValueError: If the file is missing, or the table lacks a field the layout carries.
    """
    table = _read_table(source, "tile embeddings")
    record = table.uns.get(TILES_PROVENANCE_KEY)
    if not isinstance(record, Mapping):
        raise AlhenaValueError(
            f"This table carries no {TILES_PROVENANCE_KEY!r} entry in uns, so it is not a table "
            "of tile embeddings written by ALHENA. Write tiles with "
            "TileEmbeddings.write_h5ad."
        )
    absent = [name for name in _TILE_COLUMNS if name not in table.obs.columns]
    if TILE_EMBEDDING_KEY not in table.obsm:
        absent.append(TILE_EMBEDDING_KEY)
    if "purpose" not in record:
        absent.append("purpose")
    if absent:
        raise AlhenaValueError(
            f"This table lacks {absent}, which every table of tile embeddings carries. It may "
            "have been edited since it was written."
        )

    obs = table.obs
    held = [name for name in TILE_BOUNDS_COLUMNS if name in obs.columns]
    if held and len(held) != len(TILE_BOUNDS_COLUMNS):
        raise AlhenaValueError(
            f"This table holds {held} and not all the tile bounds {list(TILE_BOUNDS_COLUMNS)}. "
            "It may have been edited since it was written."
        )
    bounds = (
        np.column_stack([obs[name].to_numpy(dtype=np.float64) for name in TILE_BOUNDS_COLUMNS])
        if held
        else None
    )
    return TileEmbeddings(
        embeddings=np.asarray(table.obsm[TILE_EMBEDDING_KEY], dtype=np.float32),
        tile_index=obs[TILE_INDEX_COLUMN].to_numpy(),
        tile_ids=tuple(str(name) for name in obs[TILE_ID_COLUMN].tolist()),
        geometry=_stored_resample(record.get("geometry")),
        encoder_id=_stored_text(record.get("encoder_id")),
        purpose=str(record["purpose"]),
        bounds=bounds,
        bounds_frame=_stored_text(record.get("bounds_frame")),
        bounds_mpp=_stored_number(record.get("bounds_mpp")),
        image_shape=_stored_shape(record.get("image_shape")),
        backbone_digest=_stored_text(record.get("backbone_digest")),
    )


def _read_table(source: str | Path | AnnData, described: str) -> AnnData:
    """Return the table a path names, or the table given, refusing what is no AnnData table."""
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.is_file():
            raise AlhenaValueError(f"No table of {described} at {path}.")
        anndata = _anndata()
        with _quiet_index_notice(anndata):
            table = anndata.read_h5ad(path)
    else:
        table = source
    for attribute in ("obs", "obsm", "uns"):
        if not hasattr(table, attribute):
            raise AlhenaValueError(
                f"A {type(table).__name__} carries no {attribute}, where an AnnData table is "
                "expected."
            )
    return table


def _stored_identifiers(column: pd.Series, dtype_text: Any) -> np.ndarray:
    """Return the identifiers of a table in the dtype they were written from."""
    values = np.asarray(column)
    if dtype_text is None:
        return values
    dtype = np.dtype(str(dtype_text))
    if dtype.kind in "iu":
        return values.astype(dtype)
    text = [str(value) for value in values.tolist()]
    return np.asarray(text, dtype=dtype if dtype.kind == "U" else object)


def _stored_array(arrays: Any, key: str, dtype: type) -> np.ndarray | None:
    """Return an array of `obsm` in a given dtype, or None where the table holds none."""
    return np.asarray(arrays[key], dtype=dtype) if key in arrays else None


def _stored_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _stored_flag(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _stored_count(value: Any) -> int | None:
    return None if value is None else int(value)


def _stored_number(value: Any) -> float | None:
    return None if value is None else float(value)


def _stored_geometry(record: Any) -> GeometryReport | None:
    """Return the geometry report a table recorded, or None where it recorded none."""
    if not isinstance(record, Mapping):
        return None
    try:
        return GeometryReport(
            mpp=_stored_number(record.get("mpp")),
            tile_size_px=int(record["tile_size_px"]),
            field_of_view_um=float(record["field_of_view_um"]),
            token_footprint_um=float(record["token_footprint_um"]),
            reference_mpp=float(record["reference_mpp"]),
            mpp_ratio=_stored_number(record.get("mpp_ratio")),
            median_nucleus_tokens=_stored_number(record.get("median_nucleus_tokens")),
            pad_fraction=float(record["pad_fraction"]),
            crop_fraction=float(record["crop_fraction"]),
            warnings=tuple(
                str(note) for note in np.asarray(record["warnings"]).reshape(-1).tolist()
            ),
        )
    except KeyError as error:
        raise AlhenaValueError(
            f"The geometry this table records lacks {error.args[0]!r}. It may have been edited "
            "since it was written."
        ) from error


def _stored_resample(record: Any) -> ResampleInfo | None:
    """Return the preprocessing record a table of tiles wrote, or None where it wrote none."""
    if not isinstance(record, Mapping):
        return None
    try:
        return ResampleInfo(
            source_mpp=_stored_number(record.get("source_mpp")),
            target_mpp=float(record["target_mpp"]),
            scale=float(record["scale"]),
            source_size=_stored_size(record["source_size"]),
            resampled_size=_stored_size(record["resampled_size"]),
            pad_fraction=float(record["pad_fraction"]),
            crop_fraction=float(record["crop_fraction"]),
            padded=bool(record["padded"]),
            cropped=bool(record["cropped"]),
            mpp_declared=bool(record["mpp_declared"]),
        )
    except KeyError as error:
        raise AlhenaValueError(
            f"The geometry this table records lacks {error.args[0]!r}. It may have been edited "
            "since it was written."
        ) from error


def _stored_size(value: Any) -> tuple[int, int]:
    """Return a height and a width a table recorded, refusing anything else."""
    values = np.asarray(value).reshape(-1).tolist()
    if len(values) != 2:
        raise AlhenaValueError(
            f"This table records {values} where a height and a width are expected. It may have "
            "been edited since it was written."
        )
    return (int(values[0]), int(values[1]))


def _stored_shape(value: Any) -> tuple[int, int] | None:
    """Return the image shape a table recorded, or None where it recorded none."""
    return None if value is None else _stored_size(value)
