"""Build a reference shard from a user's own single-cell embeddings and labels.

The embeddings come from the standalone atlas script, run in its own environment, and the shard is
self-checked before it is returned.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import numpy as np
import pandas as pd
from safetensors.numpy import save_file

from alhena import _progress
from alhena._constants import (
    CELL_SPACE_DIM,
    CLASS_AXES,
    DEFAULT_AXIS,
    DEFAULT_K,
    DEFAULT_TAU,
    SCCONCEPT_DIM,
    UNKNOWN_LABEL,
    _check_class_names,
    label_order,
)
from alhena._constants import SHARD_SUFFIX as SHARD_SUFFIX
from alhena._defaults import _batch_size
from alhena._errors import PACKAGE_PREFIXES, AlhenaValueError, EnvelopeWarning, ReferenceError
from alhena._shards import EMBEDDINGS_TENSOR as EMBEDDINGS_TENSOR
from alhena._shards import shard_metadata as shard_metadata
from alhena.encoder import AlhenaEncoder
from alhena.types import ReferenceInfo, SelfCheckReport

if TYPE_CHECKING:
    from alhena.reference import Reference

LabelSource: TypeAlias = "str | Path | Sequence[str] | np.ndarray | pd.DataFrame"
"""Labels given as an array, a table, or the path of a table."""

ReferenceLike: TypeAlias = "Reference | ReferenceInfo | str | Path"
"""A loaded reference, its `ReferenceInfo`, or the path of its shard."""

DEFAULT_DATASET: Final[str] = "cls_emb"
"""Dataset holding the embeddings in the HDF5 file the embedding step writes."""

UNLABELED_CODE: Final[int] = -1
"""Class code of a row carrying no label on an axis."""

MAX_CLASSES: Final[int] = 127
"""Largest class code an int8 array can hold, so the widest taxonomy a shard carries."""

DEFAULT_UNLABELED: Final[tuple[str, ...]] = (UNKNOWN_LABEL, "")
"""Label values read as no label. The unknown class is always among them."""

DEFAULT_PROJECTION_BATCH: Final[int] = 8192
"""Rows projected in one pass."""

USER_BUILT_KEY: Final[str] = "user_built"
"""Metadata key marking a shard built from a user's own atlas."""

USER_BUILT_SOURCE: Final[str] = "user-built"
"""Prefix of `ReferenceInfo.source` marking a user-built shard once loaded."""

ROW_ORDER: Final[str] = "source_atlas"
"""Value of the metadata field recording that rows keep the order of the source atlas."""

_TRUE: Final[str] = "true"
_FALSE: Final[str] = "false"

_OPERATOR_NOTE: Final[str] = (
    f"The operator k={DEFAULT_K}, tau={DEFAULT_TAU} was selected on a pool of atlases, on a "
    "five-class and a three-class axis, and is not tuned on this taxonomy."
)


@dataclass(frozen=True)
class ShardBuildReport:
    """What a reference build wrote, and how the reference reads at the build's `k` and `tau`.

    Attributes:
        path: Path of the shard.
        name: Name the shard carries.
        axis: Axis the self-check was run on.
        axes: Every axis the shard codes, in the order the labels declared them.
        class_orders: Classes of each axis, in the order fixing their codes. The unknown class,
            stored after them and coding no row, is left out.
        n_rows: Rows written: every row of the source atlas.
        n_labeled: Rows carrying a class, per axis.
        n_invalid_embeddings: Rows whose projection is unusable, so they carry no class on any axis.
        class_support: Cells per class, per axis.
        cell_space_id: Identity of the cell space the reference was projected into.
        user_built: Always True for a shard written here.
        info: Name, cell space and support of the shard as the loader reads it.
        self_check: Leave-one-out retrieval of the reference against itself.
    """

    path: Path
    name: str
    axis: str
    axes: tuple[str, ...]
    class_orders: dict[str, tuple[str, ...]]
    n_rows: int
    n_labeled: dict[str, int]
    n_invalid_embeddings: int
    class_support: dict[str, dict[str, int]]
    cell_space_id: str
    user_built: bool
    info: ReferenceInfo
    self_check: SelfCheckReport


# --- Inputs -------------------------------------------------------------------------------


def read_scconcept_embeddings(
    path: str | Path,
    *,
    dataset: str = DEFAULT_DATASET,
) -> np.ndarray:
    """Read single-cell embeddings from the HDF5 file the embedding step writes.

    Args:
        path: Path of the HDF5 file.
        dataset: Dataset holding the embeddings.

    Returns:
        An `(n_cells, 512)` float32 array in the row order of the source atlas.

    Raises:
        AlhenaValueError: If the file or dataset is missing, or the array is not `(n_cells, 512)`.
    """
    import h5py

    file = Path(path)
    if not file.is_file():
        raise AlhenaValueError(f"No single-cell embedding file at {file}.")
    with h5py.File(file, "r") as handle:
        if dataset not in handle:
            raise AlhenaValueError(
                f"{file} holds no dataset {dataset!r}. Its top-level datasets are {sorted(handle)}."
            )
        values = np.asarray(handle[dataset][()], dtype=np.float32)
    _validate_embedding_matrix(values, f"The embeddings in {file}")
    return values


def _validate_embedding_matrix(values: np.ndarray, described: str) -> None:
    """Check that an embedding matrix has one row per cell and the width the head reads."""
    if values.ndim != 2:
        raise AlhenaValueError(
            f"{described} have shape {values.shape}, not (n_cells, {SCCONCEPT_DIM})."
        )
    if values.shape[0] == 0:
        raise AlhenaValueError(f"{described} hold no cell.")
    if values.shape[1] != SCCONCEPT_DIM:
        advice = (
            "They hold one cell per column: pass their transpose."
            if values.shape[0] == SCCONCEPT_DIM
            else "Pass the embeddings the embedding step writes, one row per cell."
        )
        raise AlhenaValueError(
            f"{described} have shape {values.shape}, {values.shape[1]} columns wide where the "
            f"reference head reads {SCCONCEPT_DIM}. {advice}"
        )


def read_labels(
    source: LabelSource,
    *,
    n_rows: int,
    column: str | None = None,
) -> np.ndarray:
    """Read one label per cell, matched to the embeddings by row position, never by identifier.

    Args:
        source: An array of labels, a table, or the path of a `.csv`, `.tsv` or `.parquet` file.
        n_rows: Number of cells the labels must cover.
        column: Column holding the labels, needed when a table holds several.

    Returns:
        An `(n_rows,)` array of stripped label strings, empty where the value is null.

    Raises:
        AlhenaValueError: If the file, the column or the row count does not match.
    """
    if isinstance(source, str | Path):
        table: pd.DataFrame | None = _read_table(Path(source))
        described = str(source)
    elif isinstance(source, pd.DataFrame):
        table = source
        described = "the label table"
    else:
        table = None
        described = "the labels"
        if column is not None:
            raise AlhenaValueError(
                f"A column named {column!r} was asked for, but {described} are an array with "
                "no columns."
            )

    values: Any = source if table is None else _select_column(table, column, described)
    labels = _as_label_strings(values)
    if labels.shape[0] != n_rows:
        raise AlhenaValueError(
            f"{described} carry {labels.shape[0]} labels against {n_rows} cells. Labels are "
            "matched to cells by position, in the row order of the source atlas."
        )
    return labels


def _read_table(path: Path) -> pd.DataFrame:
    """Read a label table from its path."""
    if not path.is_file():
        raise AlhenaValueError(f"No label table at {path}.")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise AlhenaValueError(
        f"Labels are read from .csv, .tsv or .parquet files, and {path.name} has suffix "
        f"{path.suffix!r}. Pass an array instead for any other format."
    )


def _select_column(frame: pd.DataFrame, column: str | None, described: str) -> np.ndarray:
    """Return the label column of a table."""
    names = [str(name) for name in frame.columns]
    if column is None:
        if len(names) != 1:
            raise AlhenaValueError(
                f"{described} holds columns {names}. Name the one carrying the labels through "
                "label_column."
            )
        column = names[0]
    if column not in names:
        raise AlhenaValueError(f"{described} holds no column {column!r}. It holds {names}.")
    return frame.iloc[:, names.index(column)].to_numpy()


def _as_label_strings(values: Any) -> np.ndarray:
    """Return labels as stripped strings, with every null value read as no label."""
    array = np.asarray(values, dtype=object).reshape(-1)
    if array.size == 0:
        return np.empty(0, dtype="<U1")
    null = np.asarray(pd.isna(array), dtype=bool)
    return np.array(
        ["" if missing else str(value).strip() for value, missing in zip(array, null, strict=True)]
    )


# --- Class axes ---------------------------------------------------------------------------


def encode_labels(
    labels: LabelSource,
    class_order: Sequence[str],
    unlabeled: Sequence[str] = DEFAULT_UNLABELED,
) -> tuple[np.ndarray, dict[str, int]]:
    """Code labels against a class order, marking rows carrying no class.

    Args:
        labels: One label per cell.
        class_order: Class names, in the order fixing their codes. The unknown class codes no row.
        unlabeled: Label values read as no label.

    Returns:
        An int8 array of class codes, `-1` where a row carries no class, and the cells per class.

    Raises:
        AlhenaValueError: If the class order is empty, repeated or too long, or a label is unknown.
    """
    names = tuple(str(name).strip() for name in class_order)
    if not names:
        raise AlhenaValueError("The class order is empty, so no label can be coded.")
    if len(set(names)) != len(names):
        raise AlhenaValueError(f"The class order repeats a name: {names}.")
    if len(names) > MAX_CLASSES:
        raise AlhenaValueError(
            f"The class order holds {len(names)} classes, above the {MAX_CLASSES} a shard codes "
            "in an int8 array."
        )

    positions = {name: index for index, name in enumerate(names) if name != UNKNOWN_LABEL}
    skip = {UNKNOWN_LABEL, ""} | {str(value).strip() for value in unlabeled}

    values = _as_label_strings(labels)
    codes = np.full(values.shape[0], UNLABELED_CODE, dtype=np.int8)
    if values.size:
        unique, inverse = np.unique(values, return_inverse=True)
        lookup = np.full(unique.shape[0], UNLABELED_CODE, dtype=np.int8)
        unmatched: list[str] = []
        for index, name in enumerate(unique):
            if name in skip:
                continue
            position = positions.get(str(name))
            if position is None:
                unmatched.append(str(name))
            else:
                lookup[index] = position
        if unmatched:
            shown = ", ".join(repr(name) for name in unmatched[:8])
            raise AlhenaValueError(
                f"{len(unmatched)} label value(s) match no class of the axis: {shown}. Add them "
                "to class_order to score them, or to unlabeled to leave those rows out of the "
                "retrieval pool."
            )
        codes = lookup[inverse.reshape(-1)]

    support = {name: int((codes == position).sum()) for name, position in positions.items()}
    return codes, support


def _validate_axis_name(axis: str) -> None:
    """Check that an axis name can be the name of a class-code tensor in the shard."""
    if not axis.strip():
        raise AlhenaValueError("An axis name is empty.")
    if axis == EMBEDDINGS_TENSOR:
        raise AlhenaValueError(
            f"An axis cannot be called {EMBEDDINGS_TENSOR!r}, the name of the embedding matrix "
            "in the shard."
        )


def _resolve_class_order(axis: str, given: Sequence[str] | None) -> tuple[str, ...]:
    """Return the stored class order of an axis, the unknown class last."""
    declared = CLASS_AXES.get(axis)
    if given is None:
        if declared is None:
            raise AlhenaValueError(
                f"Axis {axis!r} is not one of the declared axes {sorted(CLASS_AXES)}. Pass "
                "class_order to declare the classes of a custom taxonomy."
            )
        return label_order(axis)

    order = tuple(str(name).strip() for name in given)
    if not order:
        raise AlhenaValueError(f"The class order given for axis {axis!r} is empty.")
    _check_class_names(order)
    if len(set(order)) != len(order):
        raise AlhenaValueError(f"The class order given for axis {axis!r} repeats a name.")
    if UNKNOWN_LABEL in order:
        raise AlhenaValueError(
            f"The class order given for axis {axis!r} holds {UNKNOWN_LABEL!r}. Leave it out: a "
            f"row with no class is coded {UNLABELED_CODE} and leaves the retrieval pool at load."
        )
    if len(order) >= MAX_CLASSES:
        raise AlhenaValueError(
            f"The class order given for axis {axis!r} holds {len(order)} classes. A shard codes "
            f"at most {MAX_CLASSES} per axis and the build appends the unknown class, so a "
            f"custom taxonomy names at most {MAX_CLASSES - 1}."
        )
    if declared is not None and order != tuple(sorted(declared)):
        raise AlhenaValueError(
            f"Axis {axis!r} is a declared axis, its class order fixed at {label_order(axis)}. "
            "Give a custom taxonomy a name of its own."
        )
    return (*order, UNKNOWN_LABEL)


def _warn_untested_class_count(axis: str, k: int, n_classes: int) -> None:
    """Report a custom taxonomy, which the operator has not been scored on."""
    warnings.warn(
        f"Axis {axis!r} carries {n_classes} classes, so k over classes is {k / n_classes:.2f} "
        f"at k={k}. {_OPERATOR_NOTE}",
        EnvelopeWarning,
        skip_file_prefixes=PACKAGE_PREFIXES,
    )


# --- Projection ---------------------------------------------------------------------------


def _batch_bounds(count: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, count)) for start in range(0, count, batch_size)]


def _project_reference(
    encoder: AlhenaEncoder,
    embeddings: np.ndarray,
    batch_size: int,
    progress: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project rows into the cell space at half precision, and mark those finite and nonzero."""
    n_rows = embeddings.shape[0]
    stored = np.empty((n_rows, CELL_SPACE_DIM), dtype=np.float16)
    usable = np.zeros(n_rows, dtype=bool)

    bounds = _batch_bounds(n_rows, batch_size)
    with _progress.bar(
        n_rows, "Projecting reference cells into the cell space", "cell", progress, scaled=True
    ) as bar:
        for start, stop in bounds:
            block = np.asarray(encoder.project_reference(embeddings[start:stop]), dtype=np.float32)
            if block.ndim != 2 or block.shape[0] != stop - start:
                raise ReferenceError(
                    f"Projecting rows {start} to {stop} returned shape {block.shape}, not one "
                    "cell-space vector per cell."
                )
            # Checked here, not at load: a projection of zero width reads as an embedding.
            if block.shape[1] != CELL_SPACE_DIM:
                raise ReferenceError(
                    f"The reference head returned {block.shape[1]}-dimensional vectors where the "
                    f"cell space is {CELL_SPACE_DIM}-dimensional."
                )
            half = block.astype(np.float16)
            recovered = half.astype(np.float32)
            finite_in = np.isfinite(block).all(axis=1)
            finite_out = np.isfinite(recovered).all(axis=1)
            if bool((finite_in & ~finite_out).any()):
                raise ReferenceError(
                    f"{int((finite_in & ~finite_out).sum())} projected rows exceed the range a "
                    "shard stores at half precision. Pass the single-cell embeddings as the "
                    "embedding step writes them, unnormalized."
                )
            stored[start:stop] = half
            # Judged on the stored values, which are the ones the loader upcasts and normalizes.
            usable[start:stop] = finite_out & (np.linalg.norm(recovered, axis=1) > 0.0)
            bar.update(stop - start)

    return stored, usable


# --- Shard ---------------------------------------------------------------------------------


def is_user_built(reference: ReferenceLike) -> bool:
    """Report whether a reference was built from a user's own atlas.

    The mark is `USER_BUILT_KEY` in the metadata of a shard, and `USER_BUILT_SOURCE` in
    `ReferenceInfo.source`.

    Args:
        reference: A loaded reference, a `ReferenceInfo`, or the path of a shard.

    Returns:
        True when the reference was built from a user's own atlas.

    Raises:
        AlhenaValueError: If the argument is none of those three.
    """
    if isinstance(reference, str | Path):
        return shard_metadata(reference).get(USER_BUILT_KEY, _FALSE).strip().lower() == _TRUE
    if isinstance(reference, ReferenceInfo):
        marker = str(reference.source).strip()
        return marker == USER_BUILT_SOURCE or marker.startswith(f"{USER_BUILT_SOURCE}:")
    info = getattr(reference, "info", None)
    if isinstance(info, ReferenceInfo):
        return is_user_built(info)
    raise AlhenaValueError(
        "Pass a loaded reference, a ReferenceInfo, or the path of a shard, got "
        f"{type(reference).__name__}."
    )


def _shard_metadata_fields(
    name: str,
    encoder: AlhenaEncoder,
    class_orders: Mapping[str, Sequence[str]],
    support: Mapping[str, Mapping[str, int]],
    n_labeled: Mapping[str, int],
    n_rows: int,
    source: str,
    license: str,
) -> dict[str, str]:
    """Assemble the metadata a shard carries. Every value is a string."""
    axes = list(class_orders)
    return {
        "shard": name,
        "name": name,
        "axes": json.dumps(axes),
        "class_order": json.dumps({axis: list(class_orders[axis]) for axis in axes}),
        "support": json.dumps({axis: dict(support[axis]) for axis in axes}),
        "n_scored": json.dumps({axis: int(n_labeled[axis]) for axis in axes}),
        "n_cells": str(int(n_rows)),
        "cell_space_id": encoder.info.cell_space_id,
        "encoder_id": encoder.info.encoder_id,
        "scconcept_model": encoder.info.scconcept_model,
        "scconcept_version": encoder.info.scconcept_version,
        "row_order": ROW_ORDER,
        "source": source,
        "license": license,
        USER_BUILT_KEY: _TRUE,
    }


def _user_built_source(source: str) -> str:
    """Return the source field, marked so the loaded reference carries the build mark."""
    described = str(source).strip()
    return f"{USER_BUILT_SOURCE}: {described}" if described else USER_BUILT_SOURCE


# --- Build --------------------------------------------------------------------------------


def _axis_argument(
    value: Any,
    axes: Sequence[str],
    described: str,
) -> dict[str, Any]:
    """Spread an argument given once, or given per axis, over the axes being built."""
    if isinstance(value, Mapping):
        unknown = sorted(set(map(str, value)) - set(axes))
        if unknown:
            raise AlhenaValueError(
                f"{described} names axes {unknown} that the labels do not cover. The labels "
                f"cover {list(axes)}."
            )
        return {axis: value.get(axis) for axis in axes}
    if value is not None and len(axes) > 1:
        raise AlhenaValueError(
            f"{described} was given once for the {len(axes)} axes {list(axes)}. Give it per "
            "axis, keyed by axis name."
        )
    return dict.fromkeys(axes, value)


def _resolve_output(output: str | Path, name: str | None, overwrite: bool) -> tuple[Path, str]:
    """Return the shard path and the name it carries."""
    path = Path(output)
    if path.is_dir():
        raise AlhenaValueError(
            f"{path} is a directory. Pass the path of the shard to write, ending in {SHARD_SUFFIX}."
        )
    if path.suffix != SHARD_SUFFIX:
        raise AlhenaValueError(
            f"A reference shard is a {SHARD_SUFFIX} file, and {path.name} ends in {path.suffix!r}."
        )
    if path.exists() and not overwrite:
        raise AlhenaValueError(
            f"{path} already holds a shard. Pass overwrite=True to replace it, or name another "
            "file."
        )
    resolved = str(name).strip() if name is not None else path.stem
    if not resolved:
        raise AlhenaValueError("The shard name is empty.")
    return path, resolved


def build_reference(
    embeddings: str | Path | np.ndarray,
    labels: LabelSource | Mapping[str, LabelSource],
    encoder: AlhenaEncoder,
    output: str | Path,
    *,
    name: str | None = None,
    axis: str = DEFAULT_AXIS,
    class_order: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    label_column: str | Mapping[str, str] | None = None,
    dataset: str = DEFAULT_DATASET,
    unlabeled: Sequence[str] = DEFAULT_UNLABELED,
    source: str = "",
    license: str = "",
    k: int = DEFAULT_K,
    tau: float = DEFAULT_TAU,
    batch_size: int = DEFAULT_PROJECTION_BATCH,
    self_check_cells: int | None = None,
    seed: int = 0,
    overwrite: bool = False,
    progress: bool | None = None,
) -> ShardBuildReport:
    """Build a reference shard from single-cell embeddings and a user's own labels.

    The embeddings are projected by the reference head and written at half precision in atlas
    order, with one class-code array per axis, `-1` where a row has no class. The shard is read
    back and retrieved against itself for the report. A custom taxonomy raises `EnvelopeWarning`,
    since `k` and `tau` were not selected on it. The shard is marked `user_built`.

    Args:
        embeddings: Path of the HDF5 file holding `cls_emb`, or an `(n_cells, 512)` array, in the
            row order of the source atlas.
        labels: One label per cell, as an array, a table or the path of a table; or a mapping from
            axis name to any of those.
        encoder: The encoder whose reference head defines the cell space.
        output: Path of the shard to write.
        name: Name the shard carries. Defaults to the file stem.
        axis: Axis the self-check runs on, one of the axes coded.
        class_order: Class names of a custom taxonomy in the order fixing their codes, or a mapping
            from axis name to such an order. Omitted for a declared axis.
        label_column: Column holding the labels when a table has several columns, or a mapping from
            axis name to column.
        dataset: Dataset holding the embeddings in the HDF5 file.
        unlabeled: Label values read as no label.
        source: Where the atlas comes from, recorded in the shard.
        license: Terms the atlas is distributed under, recorded in the shard.
        k: Neighbors the self-check votes over.
        tau: Softmax temperature the self-check votes at.
        batch_size: Rows projected in one pass.
        self_check_cells: Cells the self-check samples. Its own default when omitted.
        seed: Seed of the self-check sample.
        overwrite: Replace a shard already at `output`.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        The path written, the shard's name, cell space and support, and its self-check.

    Raises:
        AlhenaValueError: If an input is malformed, a label matches no class, or `output` exists.
        ReferenceError: If the projection cannot be stored, or if the shard does not load back.
        BundleError: If the encoder carries no reference head.
    """
    if not isinstance(encoder, AlhenaEncoder):
        raise AlhenaValueError(
            f"encoder must be an AlhenaEncoder carrying the reference head, got "
            f"{type(encoder).__name__}."
        )
    if int(k) < 1:
        raise AlhenaValueError(f"k must be at least one, got {k}.")
    if float(tau) <= 0.0:
        raise AlhenaValueError(f"tau must be positive, got {tau}.")
    batch_size = _batch_size(batch_size)

    path, shard_name = _resolve_output(output, name, overwrite)

    if isinstance(embeddings, str | Path):
        _progress.step(f"Reading the single-cell embeddings from {embeddings}", progress)
        matrix = read_scconcept_embeddings(embeddings, dataset=dataset)
    else:
        matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
        _validate_embedding_matrix(matrix, "The embeddings given")
    n_rows = int(matrix.shape[0])

    label_sources = (
        {str(key): value for key, value in labels.items()}
        if isinstance(labels, Mapping)
        else {str(axis): labels}
    )
    if not label_sources:
        raise AlhenaValueError("No labels were given, so the shard would code no axis.")
    axes = tuple(label_sources)
    primary = _primary_axis(axis, axes)
    columns = _axis_argument(label_column, axes, "label_column")
    orders_given = _axis_argument(class_order, axes, "class_order")

    class_orders: dict[str, tuple[str, ...]] = {}
    codes: dict[str, np.ndarray] = {}
    support: dict[str, dict[str, int]] = {}
    for name_of_axis in axes:
        _validate_axis_name(name_of_axis)
        order = _resolve_class_order(name_of_axis, orders_given[name_of_axis])
        values = read_labels(
            label_sources[name_of_axis], n_rows=n_rows, column=columns[name_of_axis]
        )
        axis_codes, axis_support = encode_labels(values, order, unlabeled=unlabeled)
        class_orders[name_of_axis] = order
        codes[name_of_axis] = axis_codes
        support[name_of_axis] = axis_support
        if name_of_axis not in CLASS_AXES:
            _warn_untested_class_count(name_of_axis, int(k), len(axis_support))

    stored, usable = _project_reference(encoder, matrix, batch_size, progress)
    n_invalid = int((~usable).sum())
    for name_of_axis in axes:
        codes[name_of_axis][~usable] = UNLABELED_CODE
        support[name_of_axis] = {
            class_name: int((codes[name_of_axis] == position).sum())
            for position, class_name in enumerate(class_orders[name_of_axis])
            if class_name != UNKNOWN_LABEL
        }
    n_labeled = {name_of_axis: int((codes[name_of_axis] >= 0).sum()) for name_of_axis in axes}

    metadata = _shard_metadata_fields(
        name=shard_name,
        encoder=encoder,
        class_orders=class_orders,
        support=support,
        n_labeled=n_labeled,
        n_rows=n_rows,
        source=_user_built_source(source),
        license=str(license),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _progress.step(f"Writing the shard to {path}", progress)
    save_file({EMBEDDINGS_TENSOR: stored, **codes}, str(path), metadata=metadata)

    _progress.step(f"Checking the {shard_name} reference against itself", progress)
    info, report = _self_check(
        path, primary, int(k), float(tau), self_check_cells, int(seed), progress
    )
    return ShardBuildReport(
        path=path,
        name=shard_name,
        axis=primary,
        axes=axes,
        class_orders={
            name_of_axis: tuple(name for name in order if name != UNKNOWN_LABEL)
            for name_of_axis, order in class_orders.items()
        },
        n_rows=n_rows,
        n_labeled=n_labeled,
        n_invalid_embeddings=n_invalid,
        class_support=support,
        cell_space_id=encoder.info.cell_space_id,
        user_built=True,
        info=info,
        self_check=report,
    )


def _primary_axis(axis: str, axes: Sequence[str]) -> str:
    """Return the axis the self-check runs on."""
    if axis in axes:
        return axis
    if axis == DEFAULT_AXIS and len(axes) == 1:
        return axes[0]
    raise AlhenaValueError(
        f"Axis {axis!r} is not among the axes the labels cover, {list(axes)}. Name the one the "
        "self-check should run on."
    )


def _self_check(
    path: Path,
    axis: str,
    k: int,
    tau: float,
    max_cells: int | None,
    seed: int,
    progress: bool | None = None,
) -> tuple[ReferenceInfo, SelfCheckReport]:
    """Load the shard back and retrieve it against itself at the build's `k` and `tau`."""
    from alhena.reference import load_reference

    reference = load_reference(path=path, axis=axis, progress=progress)
    arguments: dict[str, Any] = {"k": k, "tau": tau, "seed": seed, "progress": progress}
    if max_cells is not None:
        arguments["max_cells"] = int(max_cells)
    return reference.info, reference.self_check(**arguments)
