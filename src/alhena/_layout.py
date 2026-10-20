"""The AnnData layout of tile and cell embeddings, and the tables written in it."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alhena._constants import CLASS_COLORS, UNKNOWN_LABEL
from alhena._errors import AlhenaValueError
from alhena._version import __version__

if TYPE_CHECKING:
    from anndata import AnnData

    from alhena.types import AnnotationResult, CellEmbeddings, TileEmbeddings

CELLS_SUFFIX: str = ".h5ad"
"""Suffix of a table of cells, and of annotations written together with their cells."""

CELL_ID_COLUMN: str = "cell_id"
"""Column of a table of cells holding the identifiers, in the type they were given in."""

TILE_INDEX_COLUMN: str = "alhena_tile_index"
"""Column of the tile index: each tile's position among those given, or each cell's tile."""
NUCLEUS_PX_COLUMN: str = "alhena_nucleus_px"
"""Column holding the nucleus pixels supporting each cell in the tile it was read from."""
CANDIDATE_TILES_COLUMN: str = "alhena_n_candidate_tiles"
"""Column holding how many tiles held each nucleus."""
EMBEDDING_VALID_COLUMN: str = "alhena_embedding_valid"
"""Column holding whether each cell embedding may be used."""

LABEL_COLUMN: str = "alhena_celltype"
"""Stem of the column of an annotation's predicted cell types, suffixed with its name."""
CONFIDENCE_COLUMN: str = "alhena_confidence"
"""Stem of the column of the mass the winning class received."""
VALID_COLUMN: str = "alhena_valid"
"""Stem of the column of whether an annotation was assigned."""
CLASS_ENTROPY_COLUMN: str = "alhena_class_entropy"
"""Stem of the column of the class entropy."""
NEIGHBOR_ENTROPY_COLUMN: str = "alhena_neighbor_entropy"
"""Stem of the column of the neighbor entropy."""

BACKBONE_KEY: str = "X_alhena_backbone"
"""Array of the pre-projection cell embeddings."""
CELL_SPACE_KEY: str = "X_alhena_cell"
"""Array of the embeddings in the shared cell space."""
SPATIAL_KEY: str = "spatial"
"""Array of positions: the nucleus centroids, or the centers of the tile bounds."""
MASS_KEY: str = "alhena_mass"
"""Stem of the array of an annotation's class mass, in class order."""

CELLS_PROVENANCE_KEY: str = "alhena_cells"
"""Entry of `uns` holding the provenance of the cells."""
PROVENANCE_KEY: str = "alhena"
"""Entry of `uns` holding the provenance of the annotations."""
ANNOTATIONS_KEY: str = "annotations"
"""Entry under `PROVENANCE_KEY` holding one record per annotation, keyed by its name."""

TILES_SUFFIX: str = CELLS_SUFFIX
"""Suffix of a table of tile embeddings, the one a table of cells is written to."""

TILE_ID_COLUMN: str = "tile_id"
"""Column of a table of tiles holding the name of each tile, which its index holds too."""

TILE_BOUNDS_COLUMNS: tuple[str, ...] = (
    "alhena_tile_x0",
    "alhena_tile_y0",
    "alhena_tile_x1",
    "alhena_tile_y1",
)
"""Columns of the bounds x0, y0, x1, y1 of each tile in its frame, upper bounds excluded."""

TILE_EMBEDDING_KEY: str = "X_alhena_tile"
"""Array of a table of tiles holding the tile embeddings."""

TILES_PROVENANCE_KEY: str = "alhena_tiles"
"""Entry of `uns` holding the provenance of tile embeddings."""

# A slash separates the keys of the formats a table is written to, so no key may hold one.
_KEY_SEPARATOR: str = "/"

# Suffix of the uns entry scanpy reads the colors of a categorical column from.
_COLORS_SUFFIX: str = "_colors"


def _anndata() -> Any:
    """Return anndata, imported on use so that importing the package never needs it."""
    import anndata

    return anndata


@contextmanager
def _quiet_index_notice(anndata: Any) -> Iterator[None]:
    """Silence the notice some anndata releases give on turning an empty index into strings."""
    category = getattr(anndata, "ImplicitModificationWarning", Warning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Transforming to str index", category=category)
        yield


def _cell_table(cells: CellEmbeddings) -> AnnData:
    """Lay cells out in the one table layout ALHENA writes, with no annotation yet."""
    identifiers = _table_identifiers(cells.cell_id)
    n_cells = int(identifiers.shape[0])
    columns: dict[str, Any] = {
        CELL_ID_COLUMN: identifiers,
        TILE_INDEX_COLUMN: _per_cell(cells.tile_index, n_cells, "tile indices"),
        NUCLEUS_PX_COLUMN: _per_cell(cells.alpha, n_cells, "nucleus pixel counts", np.float32),
        CANDIDATE_TILES_COLUMN: _per_cell(cells.n_candidate_tiles, n_cells, "candidate counts"),
        EMBEDDING_VALID_COLUMN: _per_cell(cells.valid, n_cells, "validity flags", bool),
    }
    arrays = {BACKBONE_KEY: _per_row(cells.backbone, n_cells, "pre-projection cell embeddings")}
    if cells.cell_space is not None:
        arrays[CELL_SPACE_KEY] = _per_row(cells.cell_space, n_cells, "cell-space embeddings")
    if cells.centroids is not None:
        arrays[SPATIAL_KEY] = _per_row(cells.centroids, n_cells, "centroids", np.float64)

    observations = pd.DataFrame(
        columns, index=pd.Index([str(value) for value in identifiers.tolist()], dtype=object)
    )
    anndata = _anndata()
    with _quiet_index_notice(anndata):
        table = anndata.AnnData(X=np.empty((n_cells, 0), dtype=np.float32), obs=observations)
    for key, values in arrays.items():
        table.obsm[key] = values
    table.uns[CELLS_PROVENANCE_KEY] = _cells_record(cells, identifiers.dtype)
    return table


def _annotation_table(result: AnnotationResult, cells: CellEmbeddings, name: str | None) -> AnnData:
    """Lay an annotation out in a table of the cells it was made from."""
    chosen = _annotation_name(name, result.axis)
    _check_annotated_cells(
        result,
        cells,
        held="the embeddings",
        advice="Annotate the embeddings that are written, so that every row of the table "
        "describes one cell.",
    )
    table = _cell_table(cells)
    _write_annotation(table, result, chosen, overwrite=False)
    return table


def _annotation_name(name: str | None, axis: str) -> str:
    """Return the name an annotation is stored under: the one given, or else its axis."""
    chosen = axis if name is None else name
    if not isinstance(chosen, str):
        raise AlhenaValueError(
            f"An annotation is named by text, and was given {chosen!r}, of type "
            f"{type(chosen).__name__}."
        )
    if not chosen.strip():
        raise AlhenaValueError(
            "The annotation name is empty. Name the annotation with name=, which is its axis "
            "where none is given."
        )
    if _KEY_SEPARATOR in chosen:
        raise AlhenaValueError(
            f"The annotation name {chosen!r} holds a slash, and the formats a table is written to "
            "refuse a slash in a key. Name the annotation otherwise with name=, which is its axis "
            "where none is given."
        )
    return chosen


def _annotation_key(stem: str, name: str) -> str:
    """Return the key one field of an annotation is stored under: its stem, then the name."""
    return f"{stem}_{name}"


def _check_annotated_cells(
    result: AnnotationResult, cells: CellEmbeddings, held: str, advice: str
) -> None:
    """Refuse annotations made from other cells than those a table holds or is built from."""
    if cells.cell_space is None:
        raise AlhenaValueError(
            "These cells carry no cell-space projection, so the table would hold annotations with "
            "no retrieval embedding beside them. Pool them with the cell head of the encoder."
        )
    called = np.asarray(result.cell_id)
    identifiers = np.asarray(cells.cell_id)
    if not _same_identifiers(called, identifiers):
        raise AlhenaValueError(
            f"The annotations name {called.size} cells and {held} {identifiers.size}, and the two "
            f"do not agree cell for cell. {advice}"
        )
    space = _annotated_space(result)
    held_space = cells.cell_space_id
    if space is not None and held_space is not None and space != str(held_space):
        raise AlhenaValueError(
            f"The annotations were made in cell space {space!r}, and these cells lie in "
            f"{str(held_space)!r}, so the annotations were made from other cells. {advice}"
        )


def _same_identifiers(first: np.ndarray, second: np.ndarray) -> bool:
    """Return whether two arrays name the same cells in the same order, whatever their dtypes."""
    if first.ndim != 1 or first.shape != second.shape:
        return False
    if first.size == 0:
        return True
    kinds = {pd.api.types.infer_dtype(values, skipna=False) for values in (first, second)}
    return len(kinds) == 1 and bool(pd.Index(first).equals(pd.Index(second)))


def _annotated_space(result: AnnotationResult) -> str | None:
    """Return the cell space an annotation was made in, or None where it records none."""
    space = (result.meta or {}).get("cell_space_id")
    if space is None and result.cells is not None:
        space = result.cells.cell_space_id
    return None if space is None else str(space)


def _write_annotation(table: AnnData, result: AnnotationResult, name: str, overwrite: bool) -> None:
    """Write one annotation's fields and provenance into a table of its cells, all or nothing."""
    n_cells = int(table.n_obs)
    columns: dict[str, Any] = {
        _annotation_key(LABEL_COLUMN, name): pd.Categorical(
            _per_cell(result.labels, n_cells, "labels")
        ),
        _annotation_key(CONFIDENCE_COLUMN, name): _per_cell(
            result.confidence, n_cells, "confidences", np.float32
        ),
        _annotation_key(VALID_COLUMN, name): _per_cell(
            result.valid, n_cells, "annotation flags", bool
        ),
        _annotation_key(CLASS_ENTROPY_COLUMN, name): _per_cell(
            result.class_entropy, n_cells, "class entropies", np.float32
        ),
        _annotation_key(NEIGHBOR_ENTROPY_COLUMN, name): _per_cell(
            result.neighbor_entropy, n_cells, "neighbor entropies", np.float32
        ),
    }
    mass_key = _annotation_key(MASS_KEY, name)
    mass = _per_row(result.mass, n_cells, "class mass")

    provenance, records = _annotation_records(table)
    if name in records:
        if not overwrite:
            raise AlhenaValueError(
                f"This table holds an annotation named {name!r} already. Pass overwrite=True to "
                "replace it, and it alone, or name this annotation otherwise with name=."
            )
    else:
        taken = [key for key in columns if key in table.obs.columns]
        if mass_key in table.obsm:
            taken.append(mass_key)
        if taken:
            raise AlhenaValueError(
                f"{taken} already name fields of this table, and no annotation of it is named "
                f"{name!r}, so writing one of that name would replace them. Name this annotation "
                "otherwise with name=."
            )

    for key, values in columns.items():
        table.obs[key] = values
    table.obsm[mass_key] = mass
    records[name] = _annotation_record(result, mass_key)
    table.uns[PROVENANCE_KEY] = {**provenance, ANNOTATIONS_KEY: records}
    colors_key = _annotation_key(LABEL_COLUMN, name) + _COLORS_SUFFIX
    colors = _palette_colors(table.obs[_annotation_key(LABEL_COLUMN, name)].cat.categories)
    if colors is None:
        table.uns.pop(colors_key, None)
    else:
        table.uns[colors_key] = colors


def _palette_colors(classes: Sequence[Any]) -> list[str] | None:
    """Return the color of each class in the default palette, or None where one has none."""
    colors = []
    for name in (str(value) for value in classes):
        # The empty string marks a cell without an annotation, drawn as the unknown class.
        rgb = CLASS_COLORS.get(UNKNOWN_LABEL if name == "" else name)
        if rgb is None:
            return None
        colors.append("#{:02x}{:02x}{:02x}".format(*rgb))
    return colors


def _annotation_records(table: AnnData) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return copies of a table's annotation provenance entry and of its records by name."""
    provenance = table.uns.get(PROVENANCE_KEY)
    if provenance is None:
        return {}, {}
    records = provenance.get(ANNOTATIONS_KEY) if isinstance(provenance, Mapping) else None
    if not isinstance(provenance, Mapping) or (
        records is not None and not isinstance(records, Mapping)
    ):
        raise AlhenaValueError(
            f"This table holds {PROVENANCE_KEY!r} in uns with no mapping of annotations under "
            f"{ANNOTATIONS_KEY!r}. It may have been edited since it was written."
        )
    return dict(provenance), dict(records or {})


def _tile_table(tiles: TileEmbeddings) -> AnnData:
    """Lay tile embeddings out in the one table layout ALHENA writes for them."""
    n_tiles = len(tiles)
    names = _table_tile_ids(tiles.tile_ids, n_tiles)
    columns: dict[str, Any] = {
        TILE_ID_COLUMN: names,
        TILE_INDEX_COLUMN: _per_cell(tiles.tile_index, n_tiles, "tile indices", unit="tile"),
    }
    arrays = {
        TILE_EMBEDDING_KEY: _per_row(tiles.embeddings, n_tiles, "tile embeddings", unit="tile")
    }
    if tiles.bounds is not None:
        windows = _per_row(tiles.bounds, n_tiles, "tile bounds", np.float64, unit="tile")
        if windows.shape[1] != len(TILE_BOUNDS_COLUMNS):
            raise AlhenaValueError(
                f"The tile bounds are {windows.shape[1]} columns wide, where the bounds of a tile "
                f"are {list(TILE_BOUNDS_COLUMNS)}, four columns holding x0, y0, x1, y1."
            )
        for position, column in enumerate(TILE_BOUNDS_COLUMNS):
            columns[column] = windows[:, position]
        arrays[SPATIAL_KEY] = np.asarray(tiles.centers, dtype=np.float64)

    observations = pd.DataFrame(columns, index=pd.Index(names.tolist(), dtype=object))
    anndata = _anndata()
    with _quiet_index_notice(anndata):
        table = anndata.AnnData(X=np.empty((n_tiles, 0), dtype=np.float32), obs=observations)
    for key, values in arrays.items():
        table.obsm[key] = values
    table.uns[TILES_PROVENANCE_KEY] = _tiles_record(tiles)
    return table


def _table_tile_ids(tile_ids: Sequence[str], n_tiles: int) -> np.ndarray:
    """Return tile names a table can carry: one per tile, unique, as text."""
    names = np.asarray([str(name) for name in tile_ids], dtype=object)
    if names.shape[0] != n_tiles:
        raise AlhenaValueError(
            f"{names.shape[0]} tile names were given for {n_tiles} embeddings, where one name "
            "per tile is expected."
        )
    if n_tiles and not pd.Index(names).is_unique:
        raise AlhenaValueError(
            "Tile names repeat, so two embeddings would describe one row of the table. Name the "
            "tiles by their origin, as the readers of alhena.io do, or pass unique tile_ids."
        )
    return names


def _h5ad_path(path: str | Path) -> Path:
    """Return the path a table is written to, refusing another format."""
    target = Path(path)
    if target.suffix.lower() != CELLS_SUFFIX:
        raise AlhenaValueError(
            f"A table is written as {CELLS_SUFFIX}, and {target.name} names "
            f"{target.suffix if target.suffix else 'no suffix'}."
        )
    return target


def _table_identifiers(values: np.ndarray) -> np.ndarray:
    """Return cell identifiers a table can carry: one per cell, unique, integers or strings."""
    identifiers = np.asarray(values)
    if identifiers.ndim != 1:
        raise AlhenaValueError(
            f"Cell identifiers have shape {identifiers.shape}, where one per cell is expected."
        )
    if identifiers.size == 0:
        return identifiers
    kind = pd.api.types.infer_dtype(identifiers, skipna=False)
    if kind not in ("integer", "string"):
        raise AlhenaValueError(
            f"The cell identifiers are {kind}, where a table names its rows by integer or string "
            "identifiers."
        )
    if not pd.Index(identifiers).is_unique:
        raise AlhenaValueError(
            "Cell identifiers repeat. They must be unique across all the tiles read together, not "
            "only within one tile."
        )
    if kind == "integer" and identifiers.dtype == object:
        # Integers held as objects are written as integers, and so read back as integers.
        return identifiers.astype(np.int64)
    return identifiers


def _per_cell(
    values: Any,
    n_cells: int,
    described: str,
    dtype: type | None = None,
    unit: str = "cell",
) -> np.ndarray:
    """Return one value per row as a column, refusing an array of another length."""
    column = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
    if column.ndim != 1 or column.shape[0] != n_cells:
        raise AlhenaValueError(
            f"The {described} have shape {column.shape}, where one per {unit}, {n_cells}, is "
            "expected."
        )
    return column


def _per_row(
    values: Any,
    n_cells: int,
    described: str,
    dtype: type = np.float32,
    unit: str = "cell",
) -> np.ndarray:
    """Return one row per observation as an array, refusing another length or a width of zero."""
    array = np.ascontiguousarray(values, dtype=dtype)
    if array.ndim != 2 or array.shape[0] != n_cells:
        raise AlhenaValueError(
            f"The {described} have shape {array.shape}, where one row per {unit}, {n_cells} "
            "rows, is expected."
        )
    if array.shape[1] == 0:
        raise AlhenaValueError(
            f"The {described} are zero columns wide, so the table would hold no embedding of them. "
            f"Pass them as the encoder returns them, one row per {unit}."
        )
    return array


def _cell_columns(cells: CellEmbeddings, cell_id: np.ndarray) -> dict[str, np.ndarray]:
    """Return what the cells record of each cell apart from its embeddings, as table columns."""
    recorded = np.asarray(cells.cell_id)
    if recorded.shape != cell_id.shape or not np.array_equal(recorded, cell_id):
        raise AlhenaValueError(
            f"The annotations name {cell_id.shape[0]} cells and the attached embeddings "
            f"{recorded.shape[0]}, and the two do not agree cell for cell. Attach the "
            "CellEmbeddings the annotations were made from, as CellTyper.annotate does."
        )
    columns: dict[str, np.ndarray] = {}
    if cells.centroids is not None:
        positions = np.asarray(cells.centroids, dtype=np.float64)
        columns["x"] = positions[:, 0]
        columns["y"] = positions[:, 1]
    tile_index = np.asarray(cells.tile_index, dtype=np.int64)
    columns["tile_index"] = tile_index
    if cells.tiles is not None:
        # The tiles hold only those read, so a cell's tile is found among them by its index.
        read = pd.Index(np.asarray(cells.tiles.tile_index, dtype=np.int64))
        position = read.get_indexer(tile_index)
        if (position < 0).any():
            raise AlhenaValueError(
                f"A cell was read from tile {int(tile_index[position < 0][0])}, and the tiles "
                "attached to the cells hold no embedding of it, so the tile cannot be named. "
                "Attach the tiles of the pass the cells were read in."
            )
        columns["tile_id"] = np.asarray(cells.tiles.tile_ids, dtype=object)[position]
    columns["nucleus_px"] = np.asarray(cells.alpha, dtype=np.float32)
    columns["n_candidate_tiles"] = np.asarray(cells.n_candidate_tiles, dtype=np.int64)
    columns["embedding_valid"] = np.asarray(cells.valid, dtype=bool)
    return columns


def _cells_record(cells: CellEmbeddings, id_dtype: np.dtype) -> dict[str, Any]:
    """Return the provenance of cells in the types a table stores."""
    geometry = cells.geometry
    return _plain(
        {
            "encoder_id": cells.encoder_id,
            "backbone_digest": cells.backbone_digest,
            "cell_space_id": cells.cell_space_id,
            "geometry": None
            if geometry is None
            else {item.name: getattr(geometry, item.name) for item in fields(geometry)},
            "membership_source": cells.membership_source,
            "membership_approximate": cells.membership_approximate,
            "n_cells_without_tokens": cells.n_cells_without_tokens,
            "centroids_frame": cells.centroids_frame,
            "centroids_mpp": cells.centroids_mpp,
            "cell_id_dtype": id_dtype.str,
            "alhena_version": __version__,
        }
    )


def _tiles_record(tiles: TileEmbeddings) -> dict[str, Any]:
    """Return the provenance of tile embeddings in the types a table stores."""
    geometry = tiles.geometry
    return _plain(
        {
            "encoder_id": tiles.encoder_id,
            "backbone_digest": tiles.backbone_digest,
            "purpose": tiles.purpose,
            "geometry": None
            if geometry is None
            else {item.name: getattr(geometry, item.name) for item in fields(geometry)},
            "bounds_frame": tiles.bounds_frame,
            "bounds_mpp": tiles.bounds_mpp,
            "image_shape": None if tiles.image_shape is None else list(tiles.image_shape),
            "embedding_key": TILE_EMBEDDING_KEY,
            "alhena_version": __version__,
        }
    )


def _annotation_record(result: AnnotationResult, mass_key: str) -> dict[str, Any]:
    """Return the provenance of one annotation in the types a table stores."""
    record: dict[str, Any] = dict(result.meta or {})
    record.update(
        {
            "axis": result.axis,
            "class_order": list(result.class_order),
            "mass_key": mass_key,
            "alhena_version": __version__,
        }
    )
    return _plain(record)


def _plain(value: Any) -> Any:
    """Return a value in the types a table stores, which hold no null."""
    if isinstance(value, Mapping):
        return {str(key): _plain(entry) for key, entry in value.items() if entry is not None}
    if value is None:
        return ""
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(entry) for entry in value.tolist()]
    if isinstance(value, list | tuple | set | frozenset):
        return [_plain(entry) for entry in value]
    return str(value)
