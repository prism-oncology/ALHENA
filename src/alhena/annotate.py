"""Annotate cells by retrieval against a labeled reference in the shared cell space.

A cell takes the classes of its nearest reference cells, weighted by a softmax over their cosine
similarities.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Any

import numpy as np
import pandas as pd
import torch

from alhena import _progress
from alhena._constants import (
    DEFAULT_AXIS,
    DEFAULT_K,
    DEFAULT_TAU,
    ENTROPY_EPS,
    L2_EPS,
    RETRIEVAL_DEPTH,
    SOFTMAX_EPS,
    _check_class_names,
)
from alhena._coverage import THIN_SUPPORT_WARNING as THIN_SUPPORT_WARNING
from alhena._coverage import _missing_classes, _thin_classes, _warn_coverage
from alhena._defaults import MAX_DEFAULT_WORKERS as MAX_DEFAULT_WORKERS
from alhena._defaults import _retrieval_workers
from alhena._errors import (
    AlhenaValueError,
    BundleError,
    CellSpaceMismatchError,
    ReferenceError,
)
from alhena.types import AnnotationResult, CellEmbeddings

# Reference rows and query cells in one similarity block, so a shard is never upcast whole.
_REFERENCE_BLOCK: int = 16384
_QUERY_BLOCK: int = 2048

# Seconds between two advances of the progress bar while blocks are scored on several threads.
_PROGRESS_POLL_S: float = 0.5

# Bytes of gathered reference rows held at once while the retrieved pool is scored.
_SCORE_BLOCK_BYTES: int = 32 * 1024 * 1024

# Tolerance on the squared norm of a reference row, loose enough for rows stored as float16.
_UNIT_NORM_TOLERANCE: float = 1e-2


def l2_normalize(x: np.ndarray, eps: float = L2_EPS) -> np.ndarray:
    """Scale every row to unit length, leaving a zero row at zero.

    Args:
        x: An `(n, d)` array of embeddings.
        eps: Floor applied to each row norm, so a zero row stays zero instead of becoming NaN.

    Returns:
        An `(n, d)` float32 array.

    Raises:
        AlhenaValueError: If the input is not two-dimensional.
    """
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 2:
        raise AlhenaValueError(f"Expected an (n, d) array of embeddings, got shape {values.shape}.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(eps))


def retrieve(
    query: np.ndarray,
    reference: np.ndarray,
    k: int = DEFAULT_K,
    depth: int = RETRIEVAL_DEPTH,
    chunk_size: int = _REFERENCE_BLOCK,
    query_chunk_size: int = _QUERY_BLOCK,
    workers: int | None = None,
    progress: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the k nearest reference rows of each query cell and their cosine similarities.

    Queries are L2-normalized here; reference rows must be of unit length already. Neighbors are
    ordered by descending similarity, then ascending reference row. The number of threads never
    changes the result, nor do the chunk sizes unless more than `depth - k` rows tie with a cell's
    k-th neighbor.

    Args:
        query: An `(n_cells, d)` array of cell embeddings, not necessarily normalized.
        reference: An `(n_reference, d)` array of unit-norm reference embeddings.
        k: Number of neighbors returned per cell.
        depth: Number of neighbors retrieved before the slice to `k`.
        chunk_size: Reference rows read into one similarity block.
        query_chunk_size: Query cells scored against one reference block.
        workers: Threads scoring blocks of query cells at once. None uses the processors available,
            at most `MAX_DEFAULT_WORKERS`.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        The reference row of each neighbor, `(n_cells, k)` int64, and its cosine similarity,
        `(n_cells, k)` float32.

    Raises:
        AlhenaValueError: If the query is malformed or not finite, or a count is not positive.
        ReferenceError: If the reference is malformed, not of unit length, or smaller than `k`.
    """
    queries = np.asarray(query, dtype=np.float32)
    if queries.ndim != 2:
        raise AlhenaValueError(f"Expected an (n_cells, d) query array, got shape {queries.shape}.")
    unusable = ~np.isfinite(queries).all(axis=1)
    if unusable.any():
        raise AlhenaValueError(
            f"Query cell {int(np.flatnonzero(unusable)[0])} is not finite, so it has no direction "
            "to retrieve neighbors for. Leave it out of the query, as annotate_cells does."
        )
    rows = np.asarray(reference)
    if rows.ndim != 2:
        raise ReferenceError(
            f"Expected an (n_reference, d) reference array, got shape {rows.shape}."
        )
    if rows.shape[1] != queries.shape[1]:
        raise ReferenceError(
            f"The reference is {rows.shape[1]}-dimensional and the query is "
            f"{queries.shape[1]}-dimensional. Both must be in the same cell space."
        )
    k, depth, chunk_size, query_chunk_size = _retrieval_settings(
        k, depth, chunk_size, query_chunk_size, rows.shape[0]
    )
    threads = _retrieval_workers(workers)

    _validate_reference_norms(rows, chunk_size)

    n_cells = queries.shape[0]
    index_out = np.zeros((n_cells, k), dtype=np.int64)
    similarity_out = np.zeros((n_cells, k), dtype=np.float32)
    if n_cells == 0:
        return index_out, similarity_out

    unit_queries = l2_normalize(queries)
    pool = min(depth, rows.shape[0])
    blocks = [
        (start, min(start + query_chunk_size, n_cells))
        for start in range(0, n_cells, query_chunk_size)
    ]
    with _progress.bar(
        n_cells, "Finding nearest reference cells", "cell", progress, scaled=True
    ) as bar:
        if min(threads, len(blocks)) == 1:
            for start, stop in blocks:
                index, similarity = _retrieve_block(
                    unit_queries[start:stop], rows, k, pool, chunk_size, bar.update
                )
                index_out[start:stop] = index
                similarity_out[start:stop] = similarity
        else:
            scored = _Tally()
            with ThreadPoolExecutor(max_workers=min(threads, len(blocks))) as executor:
                futures = {
                    executor.submit(
                        _retrieve_block,
                        unit_queries[start:stop],
                        rows,
                        k,
                        pool,
                        chunk_size,
                        scored.add,
                    ): (start, stop)
                    for start, stop in blocks
                }
                pending = set(futures)
                try:
                    while pending:
                        done, pending = wait(
                            pending, timeout=_PROGRESS_POLL_S, return_when=FIRST_EXCEPTION
                        )
                        for future in done:
                            start, stop = futures[future]
                            index, similarity = future.result()
                            index_out[start:stop] = index
                            similarity_out[start:stop] = similarity
                        # Advanced from this thread alone, as a notebook widget expects.
                        bar.update(scored.total - bar.n)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
    return index_out, similarity_out


def _retrieval_settings(
    k: int, depth: int, chunk_size: int, query_chunk_size: int, n_reference: int
) -> tuple[int, int, int, int]:
    """Return k, the depth and the two chunk sizes as integers, refusing any out of range."""
    k = int(k)
    depth = int(depth)
    if k < 1:
        raise AlhenaValueError(f"k must be at least 1, got {k}.")
    if depth < k:
        raise AlhenaValueError(
            f"depth must be at least k, got depth {depth} and k {k}. Neighbors are retrieved "
            "at depth and then sliced to k."
        )
    chunk_size = int(chunk_size)
    query_chunk_size = int(query_chunk_size)
    if chunk_size < 1 or query_chunk_size < 1:
        raise AlhenaValueError(
            f"Chunk sizes must be at least 1, got {chunk_size} and {query_chunk_size}."
        )
    if n_reference < k:
        raise ReferenceError(
            f"The reference holds {n_reference} rows, fewer than the {k} neighbors asked for. "
            "Lower k or load a reference with more cells."
        )
    return k, depth, chunk_size, query_chunk_size


def _retrieve_block(
    block_query: np.ndarray,
    reference: np.ndarray,
    k: int,
    pool: int,
    chunk_size: int,
    credit: Callable[[int], Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the k nearest reference rows of a block of unit queries, crediting cells as read."""
    queries = np.ascontiguousarray(block_query)
    n_cells = queries.shape[0]
    n_reference = reference.shape[0]
    pool_index: np.ndarray | None = None
    pool_similarity: np.ndarray | None = None
    credited = 0
    for first in range(0, n_reference, chunk_size):
        last = min(first + chunk_size, n_reference)
        block = np.asarray(reference[first:last], dtype=np.float32)
        block_index, block_similarity = _block_top(queries @ block.T, first, pool)
        pool_index, pool_similarity = _merge_top(
            pool_index, pool_similarity, block_index, block_similarity, pool
        )
        reached = n_cells * last // n_reference
        credit(reached - credited)
        credited = reached
    assert pool_index is not None
    scores = _pool_similarity(queries, reference, pool_index)
    order = np.lexsort((pool_index, -scores), axis=1)[:, :k]
    return (
        np.take_along_axis(pool_index, order, axis=1),
        np.take_along_axis(scores, order, axis=1),
    )


class _Tally:
    """A count of cells that several threads add to and the drawing thread reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0

    def add(self, count: int) -> None:
        with self._lock:
            self._total += count

    @property
    def total(self) -> int:
        with self._lock:
            return self._total


def vote(
    neighbor_sim: np.ndarray,
    neighbor_labels: np.ndarray,
    n_classes: int,
    tau: float = DEFAULT_TAU,
) -> np.ndarray:
    """Turn retrieved neighbors into per-class mass, rows summing to one.

    Each neighbor gets a softmax weight over its similarity divided by `tau`, added to the column of
    its class.

    Args:
        neighbor_sim: An `(n_cells, k)` array of neighbor cosine similarities.
        neighbor_labels: An `(n_cells, k)` integer array of neighbor class codes.
        n_classes: Width of the class axis.
        tau: Softmax temperature.

    Returns:
        An `(n_cells, n_classes)` float32 array of class mass.

    Raises:
        AlhenaValueError: If the two inputs disagree in shape, or if `tau` is not positive.
        ReferenceError: If a class code falls outside the class axis.
    """
    similarity = np.asarray(neighbor_sim, dtype=np.float32)
    labels = np.asarray(neighbor_labels)
    if similarity.ndim != 2:
        raise AlhenaValueError(
            f"Expected an (n_cells, k) similarity array, got shape {similarity.shape}."
        )
    if labels.shape != similarity.shape:
        raise AlhenaValueError(
            f"Neighbor labels of shape {labels.shape} do not match neighbor similarities of "
            f"shape {similarity.shape}."
        )
    n_classes = int(n_classes)
    if n_classes < 1:
        raise ReferenceError(f"The class axis must hold at least one class, got {n_classes}.")
    tau = float(tau)
    if tau <= 0.0:
        raise AlhenaValueError(f"tau must be positive, got {tau}.")
    _validate_label_ids(labels, n_classes)
    return _class_mass(_softmax(similarity / np.float32(tau)), labels, n_classes)


def annotate_cells(
    cell_space: np.ndarray | CellEmbeddings,
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    class_order: Sequence[str],
    k: int = DEFAULT_K,
    tau: float = DEFAULT_TAU,
    cell_id: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    axis: str = DEFAULT_AXIS,
    depth: int = RETRIEVAL_DEPTH,
    cell_space_id: str | None = None,
    chunk_size: int = _REFERENCE_BLOCK,
    query_chunk_size: int = _QUERY_BLOCK,
    workers: int | None = None,
    progress: bool | None = None,
) -> AnnotationResult:
    """Predict a cell type for every cell, by retrieval against a labeled reference.

    The reference must be L2-normalized, with no unscorable row. The winning class is the argmax of
    the class mass, ties going to the lowest index. A cell whose embedding is not finite or of zero
    length, or which is marked invalid, gets an empty label, NaN confidence and mass, and `valid`
    False. Coverage gaps raise `ReferenceCoverageWarning`, an envelope breach `EnvelopeWarning`.

    Args:
        cell_space: An `(n_cells, d)` array of cell-space embeddings, or `CellEmbeddings`, whose
            identifiers and validity are then the defaults.
        reference_embeddings: An `(n_reference, d)` array of unit-norm reference embeddings.
        reference_labels: An `(n_reference,)` integer array of class codes into `class_order`.
        class_order: Class names, in the order that indexes the columns of the mass.
        k: Number of neighbors each cell votes over.
        tau: Softmax temperature over neighbor similarities.
        cell_id: Unique identifier of each cell.
        valid: False where the caller has already ruled a cell out.
        axis: Name of the class axis the reference is labeled on.
        depth: Number of neighbors retrieved before the slice to `k`.
        cell_space_id: Identity of the cell space both sides were projected into; `CellEmbeddings`
            recording another are refused.
        chunk_size: Reference rows read into one similarity block.
        query_chunk_size: Query cells scored against one reference block.
        workers: Threads scoring blocks of query cells at once, as `retrieve` reads them.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        The annotations, their mass, confidence and entropies, and the retrieved similarities.
        `meta` records the operating point, `reference_class_support`, `unexpressed_classes`,
        `thinly_supported_classes`, and `effective_k`, the median over the annotated cells of
        `exp(neighbor_entropy)`, or None.

    Raises:
        AlhenaValueError: If the cells, identifiers or hyperparameters are malformed.
        BundleError: If the cells carry no cell-space projection.
        CellSpaceMismatchError: If the cells record a cell space other than `cell_space_id`.
        ReferenceError: If the reference or its labels are malformed, or it is smaller than `k`.
    """
    if (
        isinstance(cell_space, CellEmbeddings)
        and cell_space_id is not None
        and cell_space.cell_space_id is not None
        and str(cell_space.cell_space_id) != str(cell_space_id)
    ):
        raise CellSpaceMismatchError(
            f"These cells lie in cell space {cell_space.cell_space_id!r}, and the call was made "
            f"for cell space {cell_space_id!r}. The two are the same width but not comparable."
        )
    query, identifiers, declared_valid = _resolve_cells(cell_space, cell_id, valid)
    names = tuple(str(name) for name in class_order)
    if not names:
        raise ReferenceError("The class axis is empty, so no annotation can be made.")
    if len(set(names)) != len(names):
        raise ReferenceError(
            f"The class axis repeats a name: {names}. Each column of the mass must be one class."
        )
    _check_class_names(names, ReferenceError)
    n_classes = len(names)

    matrix = np.asarray(reference_embeddings)
    if matrix.ndim != 2:
        raise ReferenceError(
            f"Expected an (n_reference, d) reference array, got shape {matrix.shape}."
        )
    label_ids = np.asarray(reference_labels)
    if label_ids.ndim != 1 or label_ids.shape[0] != matrix.shape[0]:
        raise ReferenceError(
            f"The reference holds {matrix.shape[0]} rows and {label_ids.shape} labels. One "
            "label is needed per reference row."
        )
    _validate_label_ids(label_ids, n_classes)
    tau = float(tau)
    if tau <= 0.0:
        raise AlhenaValueError(f"tau must be positive, got {tau}.")
    # Checked here too, so an out-of-range value is refused even when no cell reaches retrieval.
    k, depth, chunk_size, query_chunk_size = _retrieval_settings(
        k, depth, chunk_size, query_chunk_size, matrix.shape[0]
    )
    threads = _retrieval_workers(workers)

    counts = np.bincount(label_ids.astype(np.int64, copy=False), minlength=n_classes)
    support = {name: int(count) for name, count in zip(names, counts, strict=True)}
    _warn_coverage(support, k, str(axis), loading=False)

    n_cells = query.shape[0]
    usable = declared_valid & np.isfinite(query).all(axis=1)
    usable &= np.linalg.norm(query, axis=1) > 0.0
    called = np.flatnonzero(usable)

    label_values = np.asarray(names)
    labels_out = np.full(n_cells, "", dtype=label_values.dtype)
    mass = np.full((n_cells, n_classes), np.nan, dtype=np.float32)
    confidence = np.full(n_cells, np.nan, dtype=np.float32)
    class_entropy = np.full(n_cells, np.nan, dtype=np.float32)
    neighbor_entropy = np.full(n_cells, np.nan, dtype=np.float32)
    neighbor_similarity = np.full((n_cells, k), np.nan, dtype=np.float32)

    if called.size:
        index, similarity = retrieve(
            query[called],
            matrix,
            k=k,
            depth=depth,
            chunk_size=chunk_size,
            query_chunk_size=query_chunk_size,
            progress=progress,
            workers=threads,
        )
        weights = _softmax(similarity / np.float32(tau))
        called_mass = _class_mass(weights, label_ids[index], n_classes)
        labels_out[called] = label_values[called_mass.argmax(axis=1)]
        mass[called] = called_mass
        confidence[called] = called_mass.max(axis=1)
        class_entropy[called] = _entropy(called_mass)
        neighbor_entropy[called] = _entropy(weights)
        neighbor_similarity[called] = similarity

    meta: dict[str, Any] = {
        "k": k,
        "tau": tau,
        "depth": depth,
        "cell_space_id": cell_space_id,
        "n_cells": int(n_cells),
        "n_called": int(called.size),
        "n_reference_cells": int(matrix.shape[0]),
        "reference_class_support": support,
        "unexpressed_classes": _missing_classes(support),
        "thinly_supported_classes": _thin_classes(support),
        "effective_k": _median_effective_k(neighbor_entropy[called]),
    }
    return AnnotationResult(
        cell_id=identifiers,
        labels=labels_out,
        confidence=confidence,
        mass=mass,
        class_order=names,
        class_entropy=class_entropy,
        neighbor_entropy=neighbor_entropy,
        neighbor_similarity=neighbor_similarity,
        valid=usable,
        axis=str(axis),
        meta=meta,
    )


def _resolve_cells(
    cell_space: np.ndarray | CellEmbeddings,
    cell_id: np.ndarray | None,
    valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the query matrix, the cell identifiers and the caller-declared validity."""
    if isinstance(cell_space, CellEmbeddings):
        if cell_space.cell_space is None:
            raise BundleError(
                "These cells carry no cell-space projection. Pool tokens with a bundle that "
                "has a cell head before annotating them."
            )
        matrix = cell_space.cell_space
        cell_id = cell_space.cell_id if cell_id is None else cell_id
        valid = cell_space.valid if valid is None else valid
    else:
        matrix = cell_space

    query = np.asarray(matrix, dtype=np.float32)
    if query.ndim != 2:
        raise AlhenaValueError(f"Expected an (n_cells, d) array of cells, got shape {query.shape}.")
    if query.shape[1] == 0:
        raise AlhenaValueError(
            "The cell embeddings are zero-dimensional. A cell projection of zero width is an "
            "artifact of a missing cell head, not an empty result."
        )
    n_cells = query.shape[0]

    identifiers = np.arange(n_cells) if cell_id is None else np.asarray(cell_id)
    if identifiers.ndim != 1 or identifiers.shape[0] != n_cells:
        raise AlhenaValueError(
            f"Got {identifiers.shape} cell identifiers for {n_cells} cells. One identifier is "
            "needed per cell."
        )
    # Checked by hashing, so identifiers of several kinds, integers beside text, need no order.
    if not pd.Index(identifiers).is_unique:
        raise AlhenaValueError(
            "Cell identifiers repeat. They must be unique across all the tiles read together, not "
            "only within one tile."
        )

    declared = np.ones(n_cells, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if declared.ndim != 1 or declared.shape[0] != n_cells:
        raise AlhenaValueError(
            f"Got {declared.shape} validity flags for {n_cells} cells. One flag is needed per cell."
        )
    return query, identifiers, declared


def _validate_label_ids(labels: np.ndarray, n_classes: int) -> None:
    """Check that every class code indexes a declared class."""
    if not np.issubdtype(labels.dtype, np.integer):
        raise ReferenceError(
            f"Class codes must be integers indexing the class axis, got dtype {labels.dtype}."
        )
    if labels.size == 0:
        return
    low = int(labels.min())
    high = int(labels.max())
    if low < 0 or high >= n_classes:
        raise ReferenceError(
            f"Class codes run from {low} to {high}, outside the {n_classes} declared classes: rows "
            "with no label leave the retrieval pool at load."
        )


def _validate_reference_norms(reference: np.ndarray, chunk_size: int) -> None:
    """Check that reference rows are finite and of unit length, one block at a time."""
    for first in range(0, reference.shape[0], chunk_size):
        block = np.asarray(reference[first : first + chunk_size], dtype=np.float32)
        squared = np.einsum("ij,ij->i", block, block)
        if not np.isfinite(squared).all():
            raise ReferenceError(
                "A reference row is not finite, so it has no direction to retrieve against. Drop "
                "it from the reference before retrieval."
            )
        worst = float(np.abs(squared - 1.0).max())
        if worst > _UNIT_NORM_TOLERANCE:
            raise ReferenceError(
                f"Reference rows are not of unit length: one row is off by {worst:.3g} in "
                "squared norm. Normalize the reference once when it is loaded, so retrieval "
                "scores cosines rather than inner products."
            )


def _block_top(similarity: np.ndarray, first: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the `count` best neighbors of one reference block, by similarity then global row."""
    width = similarity.shape[1]
    count = min(count, width)
    index = first + np.arange(width, dtype=np.int64)
    if count == width:
        order = np.lexsort((np.broadcast_to(index, similarity.shape), -similarity), axis=1)
        return index[order], np.take_along_axis(similarity, order, axis=1)

    # One candidate more, to detect a tie at the boundary; without one the selection is exact.
    select = count + 1
    values, part = torch.topk(torch.from_numpy(similarity), select, dim=1, sorted=False)
    candidate_similarity = values.numpy()
    candidate_index = first + part.numpy()
    order = np.lexsort((candidate_index, -candidate_similarity), axis=1)
    candidate_index = np.take_along_axis(candidate_index, order, axis=1)
    candidate_similarity = np.take_along_axis(candidate_similarity, order, axis=1)
    out_index = candidate_index[:, :count].copy()
    out_similarity = candidate_similarity[:, :count].copy()

    tied = np.flatnonzero(candidate_similarity[:, count - 1] == candidate_similarity[:, count])
    for row in tied.tolist():
        values = similarity[row]
        threshold = out_similarity[row, count - 1]
        above = np.flatnonzero(values > threshold)
        equal = np.flatnonzero(values == threshold)
        chosen = np.concatenate([above, equal[: count - above.size]])
        chosen_index = first + chosen.astype(np.int64, copy=False)
        chosen_similarity = values[chosen]
        row_order = np.lexsort((chosen_index, -chosen_similarity))
        out_index[row] = chosen_index[row_order]
        out_similarity[row] = chosen_similarity[row_order]
    return out_index, out_similarity


def _merge_top(
    best_index: np.ndarray | None,
    best_similarity: np.ndarray | None,
    new_index: np.ndarray,
    new_similarity: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two ordered candidate sets, keeping the `count` best under the retrieval order."""
    if best_index is None or best_similarity is None:
        return new_index, new_similarity
    index = np.concatenate([best_index, new_index], axis=1)
    similarity = np.concatenate([best_similarity, new_similarity], axis=1)
    order = np.lexsort((index, -similarity), axis=1)[:, : min(count, index.shape[1])]
    return (
        np.take_along_axis(index, order, axis=1),
        np.take_along_axis(similarity, order, axis=1),
    )


def _pool_similarity(query: np.ndarray, reference: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Score each retrieved pair over the embedding width alone, so it depends on the pair only."""
    n_cells, pool = index.shape
    width = query.shape[1]
    scores = np.empty((n_cells, pool), dtype=np.float32)
    if n_cells == 0 or pool == 0:
        return scores
    step = max(1, _SCORE_BLOCK_BYTES // (pool * max(width, 1) * 4))
    for start in range(0, n_cells, step):
        stop = min(start + step, n_cells)
        gathered = np.asarray(reference[index[start:stop]], dtype=np.float32)
        np.multiply(gathered, query[start:stop, None, :], out=gathered)
        scores[start:stop] = gathered.sum(axis=2, dtype=np.float32)
    return scores


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, with the denominator floored so a row cannot divide by zero."""
    shifted = np.asarray(scores, dtype=np.float32)
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    np.exp(shifted, out=shifted)
    total = np.maximum(shifted.sum(axis=1, keepdims=True), np.float32(SOFTMAX_EPS))
    return shifted / total


def _class_mass(weights: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Accumulate neighbor weights into the column of each neighbor's class."""
    n_cells, k = weights.shape
    mass = np.zeros((n_cells, n_classes), dtype=np.float32)
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), k)
    columns = np.asarray(labels, dtype=np.int64).reshape(-1)
    # np.add.at accumulates repeated pairs, where buffered fancy indexing keeps only the last.
    np.add.at(mass, (rows, columns), weights.reshape(-1))
    return mass


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    """Row-wise entropy in nats, with probabilities clipped away from zero."""
    safe = np.clip(np.asarray(probabilities, dtype=np.float32), ENTROPY_EPS, 1.0)
    return (-np.sum(safe * np.log(safe), axis=1)).astype(np.float32)


def _median_effective_k(neighbor_entropy: np.ndarray) -> float | None:
    """Return the median of `exp(neighbor_entropy)`, the effective neighbor count, or None."""
    if neighbor_entropy.size == 0:
        return None
    return float(np.median(np.exp(neighbor_entropy.astype(np.float64))))
