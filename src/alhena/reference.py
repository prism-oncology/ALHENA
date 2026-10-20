"""Load a labeled reference shard and report what it can and cannot express.

Rows are upcast to float32 and normalized once, at load, and rows with no class on the axis leave
the retrieval pool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from alhena import _progress
from alhena._constants import (
    CELL_SPACE_DIM,
    CLASS_AXES,
    DEFAULT_AXIS,
    DEFAULT_K,
    DEFAULT_TAU,
    MIN_REFERENCE_CELLS_PER_CLASS,
    REGISTRY_FILE,
    RETRIEVAL_DEPTH,
    UNKNOWN_LABEL,
    label_order,
)
from alhena._constants import SHARD_DIRECTORY as SHARD_DIRECTORY
from alhena._constants import SHARD_SUFFIX as SHARD_SUFFIX
from alhena._coverage import _thinnest_class, _warn_coverage
from alhena._errors import (
    AlhenaValueError,
    CellSpaceMismatchError,
    ReferenceError,
)
from alhena._shards import (
    _UNDECLARED,
    EMBEDDINGS_TENSOR,
    _refuse_file_as_tissue,
    _stamped,
    _stamped_int,
    _stamped_json,
)
from alhena._shards import available_shards as available_shards
from alhena._shards import available_tissues as available_tissues
from alhena._shards import shard_for_tissue as shard_for_tissue
from alhena.annotate import l2_normalize, retrieve, vote
from alhena.metrics import score
from alhena.types import ReferenceInfo, SelfCheckReport

SELF_CHECK_QUANTILES: Final[tuple[float, ...]] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
"""Quantiles of the retrieved similarities a self-check reports."""

DEFAULT_SELF_CHECK_CELLS: Final[int] = 200_000
"""Cells a self-check scores unless the caller asks for fewer."""

# Rows read off the memory map at a time, so a shard is never held whole in two precisions.
_READ_BLOCK: Final[int] = 32768


# --- The reference ------------------------------------------------------------------------


@dataclass
class Reference:
    """A labeled reference in the shared cell space, ready to be retrieved against.

    Attributes:
        info: Name, cell space, provenance and per-class support of the shard.
        embeddings: Unit-norm rows of the scored cells, `(n_scored_cells, cell_dim)`.
        label_ids: Class code of each row, indexing `class_order`.
        class_order: Classes of the axis, those the reference holds no cell of included, in the
            order that fixes their codes and breaks ties in the vote.
    """

    info: ReferenceInfo
    embeddings: np.ndarray
    label_ids: np.ndarray
    class_order: tuple[str, ...]

    def __post_init__(self) -> None:
        self.class_order = tuple(str(name) for name in self.class_order)
        if not self.class_order:
            raise ReferenceError("A reference needs at least one class to retrieve labels from.")
        if len(set(self.class_order)) != len(self.class_order):
            raise ReferenceError(f"The class order repeats a name: {self.class_order}.")

        self.embeddings = np.asarray(self.embeddings)
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] == 0:
            raise ReferenceError(
                f"Reference embeddings must be an (n_cells, cell_dim) array with a positive "
                f"width, got shape {self.embeddings.shape}."
            )
        if self.embeddings.dtype != np.float32:
            raise ReferenceError(
                f"Reference embeddings are read in float32, got {self.embeddings.dtype}. Upcast "
                "them before normalizing, so that the stored precision does not tie neighbors."
            )

        self.label_ids = np.asarray(self.label_ids)
        if not np.issubdtype(self.label_ids.dtype, np.integer):
            raise ReferenceError(
                f"Class codes index the class order and are integers, got {self.label_ids.dtype}."
            )
        if self.label_ids.ndim != 1 or self.label_ids.shape[0] != self.embeddings.shape[0]:
            raise ReferenceError(
                f"The reference holds {self.embeddings.shape[0]} rows and "
                f"{self.label_ids.shape} labels. One class code is needed per row."
            )
        if self.label_ids.size:
            low = int(self.label_ids.min())
            high = int(self.label_ids.max())
            if low < 0 or high >= len(self.class_order):
                raise ReferenceError(
                    f"Class codes run from {low} to {high}, outside the "
                    f"{len(self.class_order)} classes of the reference. A row carrying no class "
                    "leaves the pool when the shard is loaded."
                )

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def n_classes(self) -> int:
        """Number of classes of the axis, those the reference holds no cell of included."""
        return len(self.class_order)

    def require_cell_space(self, cell_space_id: str | None = None) -> None:
        """Check that the reference records the cell space it was projected into, and which.

        Vectors of two runs have the same width, so only the identity tells them apart.

        Args:
            cell_space_id: Cell space identity of the encoder that queries the reference, or of the
                cells it annotates. None checks only that the reference records one.

        Raises:
            AlhenaValueError: If an empty identity is given.
            CellSpaceMismatchError: If the reference records no cell space, or another than given.
        """
        wanted = None if cell_space_id is None else str(cell_space_id)
        if wanted == "":
            raise AlhenaValueError(
                "No cell space identity was given to check the reference against."
            )
        if self.info.cell_space_id == _UNDECLARED:
            raise CellSpaceMismatchError(
                f"The {self.info.name} reference carries no cell space identity, so nothing "
                "shows the cells it annotates lie in its space. Rebuild the shard so that it "
                "stamps the heads that projected it."
            )
        if wanted is not None and self.info.cell_space_id != wanted:
            raise CellSpaceMismatchError(
                f"The {self.info.name} reference lies in cell space {self.info.cell_space_id!r} "
                f"and not in {wanted!r}: the two come from different runs. Use a reference built "
                f"with the encoder of {wanted!r}."
            )

    def self_check(
        self,
        k: int = DEFAULT_K,
        tau: float = DEFAULT_TAU,
        max_cells: int = DEFAULT_SELF_CHECK_CELLS,
        seed: int = 0,
        progress: bool | None = None,
    ) -> SelfCheckReport:
        """Retrieve the reference against itself, every cell excluded from its own neighbors.

        The check is necessary and not sufficient: the query is a reference cell rather than an
        image cell, so its similarity quantiles are not comparable to those of an annotation run. A
        reference of more than `max_cells` cells is scored on a subsample drawn under `seed`; the
        reference retrieved against is never subsampled.

        Args:
            k: Neighbors each cell votes over, once its own row is removed.
            tau: Softmax temperature over neighbor similarities.
            max_cells: Largest number of cells scored.
            seed: Seed of the subsample.
            progress: Whether to report progress on standard error. None follows
                `ALHENA_DISABLE_PROGRESS_BARS`.

        Returns:
            Macro F1, accuracy, per-class recall, the similarity quantiles and the number of cells
            scored, with a note for every reading the numbers depend on.

        Raises:
            AlhenaValueError: If `k`, `tau` or `max_cells` is out of range for the reference.
        """
        k = int(k)
        if k < 1:
            raise AlhenaValueError(f"k must be at least 1, got {k}.")
        tau = float(tau)
        if tau <= 0.0:
            raise AlhenaValueError(f"tau must be positive, got {tau}.")
        max_cells = int(max_cells)
        if max_cells < 1:
            raise AlhenaValueError(f"max_cells must be at least 1, got {max_cells}.")
        n_rows = len(self)
        if k >= n_rows:
            raise AlhenaValueError(
                f"A leave-one-out check over {n_rows} reference cells leaves {n_rows - 1} "
                f"neighbors to vote over, fewer than the {k} asked for."
            )

        subsampled = n_rows > max_cells
        if subsampled:
            rows = np.sort(
                np.random.default_rng(seed).choice(n_rows, size=max_cells, replace=False)
            )
            queries = self.embeddings[rows]
        else:
            rows = np.arange(n_rows)
            queries = self.embeddings

        index, similarity = retrieve(
            queries,
            self.embeddings,
            k=k + 1,
            depth=max(RETRIEVAL_DEPTH, k + 1),
            progress=progress,
        )
        is_self = index == rows[:, None]
        unplaced = ~is_self.any(axis=1)
        n_unplaced = int(np.count_nonzero(unplaced))
        if n_unplaced:
            # Only identical rows push a cell out; dropping the last neighbor keeps k of them.
            is_self[unplaced, -1] = True
        keep = ~is_self
        neighbor_similarity = similarity[keep].reshape(rows.size, k)
        neighbor_labels = self.label_ids[index[keep]].reshape(rows.size, k)

        mass = vote(neighbor_similarity, neighbor_labels, self.n_classes, tau=tau)
        report = score(mass.argmax(axis=1), self.label_ids[rows], self.class_order)
        quantiles = np.quantile(neighbor_similarity.astype(np.float64), SELF_CHECK_QUANTILES)

        return SelfCheckReport(
            reference=self.info.name,
            axis=self.info.axis,
            k=k,
            tau=tau,
            macro_f1=report.macro_f1,
            accuracy=report.accuracy,
            per_class_recall={
                name: float(report.per_class[name]["recall"]) for name in self.class_order
            },
            similarity_quantiles={
                f"p{round(quantile * 100):02d}": float(value)
                for quantile, value in zip(SELF_CHECK_QUANTILES, quantiles, strict=True)
            },
            n_cells_evaluated=int(rows.size),
            warnings=self._self_check_notes(k, rows.size, n_rows, seed, n_unplaced),
        )

    def _self_check_notes(
        self, k: int, n_evaluated: int, n_rows: int, seed: int, n_unplaced: int
    ) -> tuple[str, ...]:
        """Report every reading the self-check numbers depend on."""
        notes: list[str] = []
        if n_evaluated < n_rows:
            notes.append(
                f"Scored {n_evaluated} of {n_rows} reference cells, drawn without replacement "
                f"under seed {seed}."
            )
        if self.info.missing_classes:
            notes.append(
                f"The reference holds no cell of {', '.join(self.info.missing_classes)}, so "
                "their recall is undefined and the macro average is taken over the rest."
            )
        if self.info.thin_classes:
            notes.append(
                f"Fewer than {MIN_REFERENCE_CELLS_PER_CLASS} cells support "
                f"{', '.join(self.info.thin_classes)}."
            )
        thinnest = _thinnest_class(self.info.class_support, k, self.info.name)
        if thinnest is not None:
            notes.append(thinnest)
        if n_unplaced:
            notes.append(
                f"{n_unplaced} cells did not appear among their own {k + 1} nearest rows, the "
                "reference holding at least that many rows identical to them; their last "
                "neighbor was dropped instead."
            )
        return tuple(notes)


# --- Loading ------------------------------------------------------------------------------


def load_reference(
    tissue: str | None = None,
    path: str | Path | None = None,
    axis: str = DEFAULT_AXIS,
    *,
    cache_dir: str | Path | None = None,
    progress: bool | None = None,
) -> Reference:
    """Load a reference shard for one class axis.

    Rows with no class on the axis are dropped, and the rest are upcast to float32 and normalized
    once. Coverage gaps raise `ReferenceCoverageWarning`, and an envelope breach at the default `k`
    raises `EnvelopeWarning`.

    Args:
        tissue: Tissue whose published reference to load, by tissue, alias or shard. Give this or
            `path`.
        path: Location of a shard on disk.
        axis: Class axis to label the reference on: a declared axis, or, for a shard given by
            `path`, an axis it was built on.
        cache_dir: Root a published shard is looked for in before it is downloaded. Defaults to the
            resolved cache root.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        The reference, holding only the rows the axis labels.

    Raises:
        AlhenaValueError: If not exactly one of `tissue` and `path` is given, or no reference fits.
        ReferenceError: If the shard is missing or malformed, or if it holds no row the axis labels.
    """
    if path is None:
        _refuse_file_as_tissue(tissue, "path=")
    file = _shard_file(tissue, path, axis, cache_dir)
    info, positions, label_ids = _read_axis(file, axis)
    _warn_coverage(info.class_support, DEFAULT_K, axis, loading=True)
    embeddings = _read_embeddings(file, positions, info.n_cells, info.name, progress)
    return Reference(
        info=info, embeddings=embeddings, label_ids=label_ids, class_order=info.class_order
    )


def describe_shard(
    tissue: str | None = None,
    path: str | Path | None = None,
    axis: str = DEFAULT_AXIS,
    *,
    cache_dir: str | Path | None = None,
) -> ReferenceInfo:
    """Report a shard's name, cell space, provenance and support without reading its embeddings.

    Only the header and the labels are read. No coverage warning is raised: `missing_classes` and
    `thin_classes` carry the same facts.

    Args:
        tissue: Tissue whose published reference to describe, by tissue, alias or shard. Give this
            or `path`.
        path: Location of a shard on disk.
        axis: Class axis to count the support on: a declared axis, or, for a shard given by `path`,
            an axis it was built on.
        cache_dir: Root a published shard is looked for in before it is downloaded. Defaults to the
            resolved cache root.

    Returns:
        What the shard is and what it can express on that axis.

    Raises:
        AlhenaValueError: If not exactly one of `tissue` and `path` is given, or no reference fits.
        ReferenceError: If the shard is missing or malformed, or carries no label for the axis.
    """
    if path is None:
        _refuse_file_as_tissue(tissue, "path=")
    return _read_axis(_shard_file(tissue, path, axis, cache_dir), axis)[0]


def _shard_file(
    tissue: str | None, path: str | Path | None, axis: str, cache_dir: str | Path | None
) -> Path:
    """Return the shard a reference on an axis is read from; only a path names a custom axis."""
    if axis not in CLASS_AXES and tissue is not None and path is None:
        # A published shard carries the declared axes alone, whatever the axis asked for.
        raise AlhenaValueError(
            f"{tissue!r} names a prebuilt reference, which is labeled on {sorted(CLASS_AXES)} "
            "alone. A reference built on a custom axis is read from its shard: give the "
            "path of the shard instead of a tissue."
        )
    return _resolve_shard_path(tissue, path, cache_dir)


def _stored_order(metadata: dict[str, str], axis: str, path: Path) -> tuple[str, ...]:
    """Return the classes the codes of an axis index: a declared axis's, or those stamped."""
    if axis in CLASS_AXES:
        return label_order(axis)
    orders = _stamped_json(metadata, "class_order", path)
    if axis not in orders:
        raise ReferenceError(
            f"{path} stamps no class order for the axis {axis!r}. Axes stamped in this shard: "
            f"{sorted(orders)}."
        )
    return tuple(str(name) for name in orders[axis])


def _resolve_shard_path(
    tissue: str | None, path: str | Path | None, cache_dir: str | Path | None
) -> Path:
    """Return the file a reference is to be read from."""
    if (tissue is None) == (path is None):
        raise AlhenaValueError(
            "Name a tissue or give the path of a shard. Passing both leaves it ambiguous which "
            "reference is meant, and passing neither names none."
        )
    if path is not None:
        file = Path(path)
        if not file.is_file():
            raise ReferenceError(f"No reference shard at {file}.")
        return file

    shard = shard_for_tissue(str(tissue))
    filename = f"{shard}{SHARD_SUFFIX}"
    from alhena.hub import cache_root, ensure_reference

    # Resolved first, so a cache the environment names is read as one passed here.
    root = Path(cache_root(cache_dir, create=False))
    for candidate in (root / SHARD_DIRECTORY / filename, root / filename):
        if candidate.is_file():
            return candidate
    return Path(ensure_reference(shard, cache_dir=cache_dir))


def _open_shard(path: Path) -> Any:
    """Open a shard for reading, memory-mapped, as a context manager loading no tensor."""
    from safetensors import safe_open

    if not path.is_file():
        raise ReferenceError(f"No reference shard at {path}.")
    try:
        return safe_open(str(path), framework="numpy")
    except Exception as error:
        raise ReferenceError(f"{path} is not readable as a reference shard.") from error


def _read_axis(path: Path, axis: str) -> tuple[ReferenceInfo, np.ndarray, np.ndarray]:
    """Read a shard's header and one label tensor, leaving the embeddings on disk."""
    with _open_shard(path) as handle:
        metadata = dict(handle.metadata() or {})
        tensors = list(handle.keys())
        if EMBEDDINGS_TENSOR not in tensors:
            raise ReferenceError(
                f"{path} holds no {EMBEDDINGS_TENSOR!r} tensor, so it is not a reference shard."
            )
        if axis not in tensors:
            carried = sorted(name for name in tensors if name != EMBEDDINGS_TENSOR)
            raise ReferenceError(
                f"{path} carries no labels for the axis {axis!r}. Axes in this shard: {carried}."
            )
        codes = np.asarray(handle.get_tensor(axis))
        shape = tuple(int(size) for size in handle.get_slice(EMBEDDINGS_TENSOR).get_shape())

    if codes.ndim != 1:
        raise ReferenceError(
            f"The {axis!r} labels of {path} have shape {codes.shape}, where one class code per "
            "reference row is expected."
        )
    if not np.issubdtype(codes.dtype, np.integer):
        raise ReferenceError(
            f"The {axis!r} labels of {path} have dtype {codes.dtype}. Class codes index the "
            "class axis and are integers."
        )
    n_rows = int(codes.shape[0])
    if len(shape) != 2:
        raise ReferenceError(
            f"The embeddings of {path} have shape {shape}, where an (n_cells, cell_dim) array "
            "is expected."
        )
    if shape[0] != n_rows:
        raise ReferenceError(
            f"{path} holds {shape[0]} embeddings against {n_rows} labels on {axis!r}. Every row "
            "of the shard carries one class code."
        )
    if shape[1] != CELL_SPACE_DIM:
        raise ReferenceError(
            f"{path} holds {shape[1]}-dimensional embeddings where the cell space is "
            f"{CELL_SPACE_DIM}-dimensional. This shard was not projected by a reference head."
        )
    stamped_rows = _stamped_int(metadata, "n_cells", path)
    if stamped_rows is not None and stamped_rows != n_rows:
        raise ReferenceError(
            f"{path} declares {stamped_rows} cells and holds {n_rows}. The shard is not the one "
            "its header describes."
        )

    order = _stored_order(metadata, axis, path)
    classes = tuple(name for name in order if name != UNKNOWN_LABEL)
    stored = codes.astype(np.int64)
    if n_rows:
        low = int(stored.min())
        high = int(stored.max())
        if low < -1 or high >= len(order):
            raise ReferenceError(
                f"The {axis!r} class codes of {path} run from {low} to {high}, outside the "
                f"{len(order)} classes of the axis. A row with no class of the axis carries -1."
            )
    # A stored code becomes its class's position, and the unknown class no class at all.
    to_class = np.asarray(
        [classes.index(name) if name in classes else -1 for name in order], dtype=np.int64
    )
    mapped = np.where(stored >= 0, to_class[np.maximum(stored, 0)], -1)
    positions = np.flatnonzero(mapped >= 0)
    if positions.size == 0:
        raise ReferenceError(
            f"{path} holds no cell labeled on {axis!r}, so there is nothing to retrieve "
            "against. Load it on an axis its cells carry a class of."
        )
    label_ids = mapped[positions].astype(np.int32)

    counts = np.bincount(label_ids, minlength=len(classes))
    support = {name: int(counts[position]) for position, name in enumerate(classes)}
    name = _stamped(metadata, "shard")
    name = path.stem if name == _UNDECLARED else name
    cell_space_id = _stamped(metadata, "cell_space_id")
    source, license = _provenance(metadata, path, name, n_rows, cell_space_id)
    info = ReferenceInfo(
        name=name,
        axis=axis,
        cell_space_id=cell_space_id,
        n_cells=n_rows,
        n_scored_cells=int(positions.size),
        class_order=classes,
        class_support=support,
        source=source,
        license=license,
        scconcept_model=_stamped(metadata, "scconcept_model"),
        scconcept_version=_stamped(metadata, "scconcept_version"),
        coverage_verified=_check_support_stamp(metadata, axis, support, path),
    )
    return info, positions, label_ids


def _read_embeddings(
    path: Path,
    positions: np.ndarray,
    n_rows: int,
    name: str,
    progress: bool | None = None,
) -> np.ndarray:
    """Read the labeled rows of a shard block by block, upcast to float32 and normalized."""
    out = np.empty((positions.size, CELL_SPACE_DIM), dtype=np.float32)
    described = f"Reading the {name} reference"
    with (
        _open_shard(path) as handle,
        _progress.bar(n_rows, described, "cell", progress, scaled=True) as bar,
    ):
        payload = handle.get_slice(EMBEDDINGS_TENSOR)
        written = 0
        for start in range(0, n_rows, _READ_BLOCK):
            stop = min(start + _READ_BLOCK, n_rows)
            first = int(np.searchsorted(positions, start))
            last = int(np.searchsorted(positions, stop))
            if last > first:
                wanted = positions[first:last]
                rows = np.asarray(payload[start:stop][wanted - start], dtype=np.float32)
                _reject_unusable_rows(rows, wanted, path)
                out[written : written + rows.shape[0]] = l2_normalize(rows)
                written += rows.shape[0]
            bar.update(stop - start)
    return out


def _reject_unusable_rows(rows: np.ndarray, positions: np.ndarray, path: Path) -> None:
    """Refuse a reference holding a row that has no direction to retrieve against."""
    norms = np.linalg.norm(rows, axis=1)
    unusable = ~np.isfinite(norms) | (norms <= 0.0)
    if unusable.any():
        first = int(positions[np.flatnonzero(unusable)[0]])
        raise ReferenceError(
            f"Row {first} of {path} is either not finite or of zero length, so it has no "
            f"direction, and {int(np.count_nonzero(unusable))} rows of one block are like it. "
            "The shard is corrupt."
        )


def _provenance(
    metadata: dict[str, str], path: Path, name: str, n_rows: int, cell_space_id: str
) -> tuple[str, str]:
    """Return the source and license of a shard: stamped in it, else declared by its registry."""
    source = _stamped(metadata, "source")
    license = _stamped(metadata, "license")
    if _UNDECLARED not in (source, license):
        return source, license
    entry = _registry_entry(path, name, n_rows, cell_space_id)
    if entry is None:
        return source, license
    declared = str(entry.get("source") or "").strip()
    location = str(entry.get("source_url") or "").strip()
    if source == _UNDECLARED and declared:
        source = f"{declared} ({location})" if location else declared
    terms = str(entry.get("license") or "").strip()
    if license == _UNDECLARED and terms:
        license = terms
    return source, license


def _registry_entry(
    path: Path, name: str, n_rows: int, cell_space_id: str
) -> dict[str, Any] | None:
    """Return what the registry beside a shard declares for it, matched by name, cells and space."""
    folders = [path.parent.parent] if path.parent.name == SHARD_DIRECTORY else []
    folders.append(path.parent)
    for folder in folders:
        registry = folder / REGISTRY_FILE
        if not registry.is_file():
            continue
        try:
            payload = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        shards = payload.get("shards") if isinstance(payload, dict) else None
        entry = shards.get(name) if isinstance(shards, dict) else None
        if not isinstance(entry, dict) or entry.get("n_cells") != n_rows:
            continue
        declared = entry.get("cell_space_id")
        if declared is not None and str(declared) != cell_space_id:
            continue
        return entry
    return None


def _check_support_stamp(
    metadata: dict[str, str], axis: str, support: dict[str, int], path: Path
) -> bool:
    """Report whether the shard stamps a support for the axis, refusing one that disagrees."""
    raw = metadata.get("support")
    if not raw:
        return False
    try:
        stamped = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ReferenceError(f"The support stamped in {path} is not readable as JSON.") from error
    if not isinstance(stamped, dict):
        raise ReferenceError(f"The support stamped in {path} must be keyed by class axis.")
    block = stamped.get(axis)
    if block is None:
        return False
    if not isinstance(block, dict):
        raise ReferenceError(
            f"The support stamped in {path} for {axis!r} must be keyed by class name."
        )
    disagreements = [
        f"{name}: stamped {block.get(name)!r}, counted {count}"
        for name, count in support.items()
        if not isinstance(block.get(name), int) or block.get(name) != count
    ]
    if disagreements:
        raise ReferenceError(
            f"The support stamped in {path} for {axis!r} disagrees with its labels: "
            f"{'; '.join(disagreements)}. The shard is not the one its header describes."
        )
    return True
