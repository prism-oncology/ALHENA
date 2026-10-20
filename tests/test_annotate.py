"""Tests for retrieval, the class vote and cell-type annotation with a cell typer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import alhena
from alhena.annotate import l2_normalize
from support import (
    CELL_DIM,
    CELL_SPACE,
    CELLS_PER_CLASS,
    LINEAGE,
    N_TILES,
    NUCLEI_PER_TILE,
    class_codes,
    make_label_maps,
    make_shard,
    make_tiles,
    tiny_encoder,
)

CLASS_ORDER = ("B_Plasma", "Myeloid", "T_NK")
DIM = 8
MPP = alhena.REFERENCE_MPP
OTHER_SPACE = "ffffffffffffffff"


def make_reference(
    per_class: int = 400, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit reference rows of separable classes, their class codes and the class centers."""
    rng = np.random.default_rng(seed)
    centers = np.eye(len(CLASS_ORDER), DIM, dtype=np.float32)
    rows = np.repeat(centers, per_class, axis=0)
    rows += 0.05 * rng.normal(size=rows.shape).astype(np.float32)
    codes = np.repeat(np.arange(len(CLASS_ORDER)), per_class)
    return l2_normalize(rows), codes, centers


def noisy_queries(centers: np.ndarray, per_center: int, spread: float, seed: int) -> np.ndarray:
    """Return queries scattered around each class center, center after center."""
    rng = np.random.default_rng(seed)
    queries = np.repeat(centers, per_center, axis=0)
    return queries + spread * rng.normal(size=queries.shape).astype(np.float32)


def brute_force_neighbors(
    query: np.ndarray, reference: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rank every reference row against every query in float64, ties going to the lower row."""
    unit = query / np.linalg.norm(query, axis=1, keepdims=True)
    similarity = unit.astype(np.float64) @ reference.astype(np.float64).T
    index = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    return index, np.take_along_axis(similarity, index, axis=1)


@pytest.fixture
def typer(tmp_path: Path) -> alhena.CellTyper:
    """Return a cell typer pairing the small encoder with a reference in its cell space."""
    reference = alhena.load_reference(path=make_shard(tmp_path))
    return alhena.CellTyper(tiny_encoder(), reference)


def test_l2_normalize_scales_rows_to_unit_length():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0], [-2.0, 0.0]]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0], [-1.0, 0.0]], atol=1e-7)


def test_retrieve_matches_brute_force_search():
    reference, _, _ = make_reference(per_class=120)
    query = np.random.default_rng(3).normal(size=(9, DIM)).astype(np.float32)
    index, similarity = alhena.retrieve(query, reference, k=20, chunk_size=37, query_chunk_size=4)
    expected_index, expected_similarity = brute_force_neighbors(query, reference, k=20)
    assert index.shape == similarity.shape == (9, 20)
    np.testing.assert_array_equal(index, expected_index)
    np.testing.assert_allclose(similarity, expected_similarity, atol=1e-6)


@pytest.mark.parametrize(
    ("malform", "match"),
    [
        pytest.param(lambda rows: rows[:, :-1], "same cell space", id="another width"),
        pytest.param(lambda rows: 3.0 * rows, "unit length", id="not normalized"),
    ],
)
def test_retrieve_rejects_malformed_reference(
    malform: Callable[[np.ndarray], np.ndarray], match: str
):
    reference, _, centers = make_reference()
    with pytest.raises(alhena.ReferenceError, match=match):
        alhena.retrieve(centers, malform(reference))


def test_vote_accumulates_weight_of_repeated_class():
    similarity = np.array([[0.60, 0.54, 0.30]], dtype=np.float32)
    weights = np.exp(similarity[0].astype(np.float64) / 0.1)
    weights /= weights.sum()
    mass = alhena.vote(similarity, np.array([[2, 0, 2]]), n_classes=3, tau=0.1)
    np.testing.assert_allclose(mass[0], [weights[1], 0.0, weights[0] + weights[2]], atol=1e-6)


def test_vote_lower_tau_sharpens_class_mass():
    similarity = np.array([[0.9, 0.6, 0.6]], dtype=np.float32)
    labels = np.array([[0, 1, 1]])
    taus = (1000.0, 1.0, 0.1, 0.001)
    nearest = [float(alhena.vote(similarity, labels, n_classes=2, tau=tau)[0, 0]) for tau in taus]
    assert nearest == sorted(nearest)
    # A high temperature weighs the neighbors alike; a low one gives the nearest the whole vote.
    assert nearest[0] == pytest.approx(1 / 3, abs=1e-3)
    assert nearest[-1] == pytest.approx(1.0)


def test_annotate_cells_assigns_class_of_nearest_cells():
    reference, codes, centers = make_reference()
    query = noisy_queries(centers, per_center=2, spread=0.05, seed=1)
    result = alhena.annotate_cells(
        query, reference, codes, CLASS_ORDER, k=20, tau=0.05, cell_id=np.arange(10, 16)
    )
    assert result.labels.tolist() == [name for name in CLASS_ORDER for _ in range(2)]
    assert result.cell_id.tolist() == list(range(10, 16))
    assert result.valid.all()
    np.testing.assert_allclose(result.mass.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(result.confidence, result.mass.max(axis=1))
    assert (result.meta["k"], result.meta["tau"]) == (20, 0.05)


def test_annotate_cells_records_the_classes_its_reference_lacks():
    reference, codes, centers = make_reference()
    kept = codes < 2
    with pytest.warns(alhena.ReferenceCoverageWarning, match="T_NK"):
        result = alhena.annotate_cells(centers[:2], reference[kept], codes[kept], CLASS_ORDER)
    assert result.missing_classes == ("T_NK",)
    assert alhena.score(result, ["B_Plasma", "Myeloid"]).missing_classes == ("T_NK",)


def test_annotate_cells_mass_is_vote_over_retrieved_neighbors():
    reference, codes, centers = make_reference()
    query = noisy_queries(centers, per_center=2, spread=0.3, seed=8)
    result = alhena.annotate_cells(query, reference, codes, CLASS_ORDER)
    index, similarity = alhena.retrieve(query, reference)
    np.testing.assert_array_equal(result.neighbor_similarity, similarity)
    np.testing.assert_array_equal(
        result.mass, alhena.vote(similarity, codes[index], len(CLASS_ORDER))
    )


def test_annotate_cells_leaves_unusable_cells_unannotated():
    reference, codes, centers = make_reference()
    query = np.stack([centers[0], np.zeros(DIM), np.full(DIM, np.nan), centers[1]])
    declared = np.array([True, True, True, False])
    result = alhena.annotate_cells(query, reference, codes, CLASS_ORDER, valid=declared)
    assert result.valid.tolist() == [True, False, False, False]
    assert result.labels.tolist() == ["B_Plasma", "", "", ""]
    assert np.isnan(result.confidence[1:]).all()
    assert np.isnan(result.mass[1:]).all()
    assert result.meta["n_called"] == 1


@pytest.mark.parametrize(
    ("chunk_size", "query_chunk_size", "workers"),
    [(7, 1, 1), (13, 5, 1), (13, 5, 3), (100_000, 100_000, 2)],
)
def test_annotate_cells_independent_of_chunks_and_workers(
    chunk_size: int, query_chunk_size: int, workers: int
):
    reference, codes, centers = make_reference()
    query = noisy_queries(centers, per_center=4, spread=0.2, seed=9)
    baseline = alhena.annotate_cells(query, reference, codes, CLASS_ORDER, workers=1)
    result = alhena.annotate_cells(
        query,
        reference,
        codes,
        CLASS_ORDER,
        chunk_size=chunk_size,
        query_chunk_size=query_chunk_size,
        workers=workers,
    )
    np.testing.assert_array_equal(result.neighbor_similarity, baseline.neighbor_similarity)
    np.testing.assert_array_equal(result.mass, baseline.mass)
    assert result.labels.tolist() == baseline.labels.tolist()


def test_celltyper_annotates_tiles_and_label_maps(typer: alhena.CellTyper):
    result = typer.annotate(make_tiles(), make_label_maps(), mpp=MPP)
    n_cells = N_TILES * NUCLEI_PER_TILE
    assert result.cell_id.tolist() == list(range(1, n_cells + 1))
    assert result.mass.shape == (n_cells, len(LINEAGE))
    np.testing.assert_allclose(result.mass.sum(axis=1), 1.0, atol=1e-5)
    assert result.labels.tolist() == [LINEAGE[i] for i in result.mass.argmax(axis=1)]
    assert result.valid.all()
    assert result.meta["cell_space_id"] == CELL_SPACE
    assert result.meta["reference"] == "prostate"
    assert result.cells.cell_space.shape == (n_cells, CELL_DIM)


def test_celltyper_matches_annotate_cells_on_its_cells(typer: alhena.CellTyper):
    tiles, maps = make_tiles(), make_label_maps()
    cells = typer.encoder.encode_cells(tiles, maps, mpp=MPP)
    reference = typer.reference
    expected = alhena.annotate_cells(
        cells, reference.embeddings, reference.label_ids, reference.class_order
    )
    result = typer.annotate(tiles, maps, mpp=MPP)
    assert result.cell_id.tolist() == expected.cell_id.tolist()
    assert result.labels.tolist() == expected.labels.tolist()
    np.testing.assert_array_equal(result.mass, expected.mass)


def test_celltyper_preflight_describes_the_geometry(typer: alhena.CellTyper):
    report = typer.preflight(make_tiles(), make_label_maps(), mpp=MPP)
    assert report.tile_size_px == alhena.INPUT_SIZE
    assert report.field_of_view_um == pytest.approx(alhena.INPUT_SIZE * MPP)
    assert not report.warnings
    assert typer.class_order == LINEAGE


def test_celltyper_annotates_stored_cells_without_encoder(tmp_path: Path):
    path = make_shard(tmp_path)
    tiles, maps = make_tiles(), make_label_maps()
    typer = alhena.CellTyper(tiny_encoder(), alhena.load_reference(path=path))
    first = typer.annotate(tiles, maps, mpp=MPP)

    stored = alhena.load_celltyper(reference=path)
    again = stored.annotate(first.cells)
    assert stored.encoder is None
    assert again.cell_id.tolist() == first.cell_id.tolist()
    assert again.labels.tolist() == first.labels.tolist()
    np.testing.assert_array_equal(again.mass, first.mass)


def test_celltyper_refuses_reference_of_another_cell_space(tmp_path: Path):
    path = make_shard(tmp_path, metadata={"cell_space_id": OTHER_SPACE})
    reference = alhena.load_reference(path=path)
    with pytest.raises(alhena.CellSpaceMismatchError, match=OTHER_SPACE):
        alhena.CellTyper(tiny_encoder(), reference)

    cells = tiny_encoder().encode_cells(make_tiles(), make_label_maps(), mpp=MPP)
    with pytest.raises(alhena.CellSpaceMismatchError, match=OTHER_SPACE):
        alhena.CellTyper(None, reference).annotate(cells)


def test_celltyper_reports_thinly_supported_class(tmp_path: Path):
    counts = {name: 100 if name == "B_Plasma" else CELLS_PER_CLASS for name in LINEAGE}
    path = make_shard(tmp_path, codes=class_codes(tuple(counts.values())))
    with (
        pytest.warns(alhena.ReferenceCoverageWarning, match="B_Plasma"),
        pytest.warns(alhena.EnvelopeWarning, match="B_Plasma"),
    ):
        reference = alhena.load_reference(path=path)
    typer = alhena.CellTyper(tiny_encoder(), reference)

    with pytest.warns(alhena.EnvelopeWarning, match="B_Plasma"):
        result = typer.annotate(make_tiles(), make_label_maps(), mpp=MPP)
    assert result.meta["thinly_supported_classes"] == ("B_Plasma",)
    assert result.meta["reference_class_support"] == counts
