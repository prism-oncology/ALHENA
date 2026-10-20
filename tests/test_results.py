"""Tests for the result objects: their tables, the files they are written to, and selections."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from alhena import (
    INPUT_SIZE,
    REFERENCE_MPP,
    AlhenaEncoder,
    AlhenaValueError,
    AnnotationResult,
    CellEmbeddings,
    CellTyper,
    Reference,
    ReferenceCoverageWarning,
    TileEmbeddings,
    annotate_cells,
    label_order,
    load_reference,
)
from alhena.cells import ORIGINS_FRAME
from alhena.io import (
    ANNOTATIONS_KEY,
    BACKBONE_KEY,
    CELL_ID_COLUMN,
    CELL_SPACE_KEY,
    CELLS_PROVENANCE_KEY,
    CONFIDENCE_COLUMN,
    LABEL_COLUMN,
    MASS_KEY,
    NUCLEUS_PX_COLUMN,
    PROVENANCE_KEY,
    SPATIAL_KEY,
    TILE_BOUNDS_COLUMNS,
    TILE_EMBEDDING_KEY,
    TILE_ID_COLUMN,
    TILE_INDEX_COLUMN,
    TILES_PROVENANCE_KEY,
    TileSource,
    add_annotation,
    read_annotations,
    read_cells,
    read_tile_embeddings,
    read_tiles,
)
from support import (
    CELL_DIM,
    CELL_SPACE,
    CELLS_PER_CLASS,
    COMPARTMENT_OF,
    ENCODER_ID,
    LINEAGE,
    NUCLEUS_SIDE,
    class_codes,
    make_centroids,
    make_label_maps,
    make_shard,
    make_tiles,
    tiny_encoder,
)

# Three tiles side by side, four nuclei on each.
ORIGINS = np.array([[0, 0], [INPUT_SIZE, 0], [2 * INPUT_SIZE, 0]])
TILE_OF_EACH_CELL = ["0"] * 4 + ["1"] * 4 + ["2"] * 4


@pytest.fixture
def encoder() -> AlhenaEncoder:
    return tiny_encoder()


@pytest.fixture
def source() -> TileSource:
    return read_tiles(make_tiles(), mpp=REFERENCE_MPP, coords=ORIGINS)


@pytest.fixture
def tiles(encoder: AlhenaEncoder, source: TileSource) -> TileEmbeddings:
    return encoder.encode_tiles(source)


@pytest.fixture
def cells(encoder: AlhenaEncoder, source: TileSource) -> CellEmbeddings:
    return encoder.encode_cells(source, make_label_maps())


@pytest.fixture
def reference(tmp_path) -> Reference:
    return load_reference(path=make_shard(tmp_path))


@pytest.fixture
def result(encoder: AlhenaEncoder, reference: Reference, cells: CellEmbeddings) -> AnnotationResult:
    return CellTyper(encoder, reference).annotate(cells)


def compartment_annotations(cells: CellEmbeddings, reference: Reference) -> AnnotationResult:
    """Annotate cells on the compartment axis, against the lineage reference regrouped."""
    classes = tuple(name for name in label_order("compartment_v1") if name != "Unknown")
    lineage = np.asarray(reference.class_order)[reference.label_ids]
    codes = np.array([classes.index(COMPARTMENT_OF[name]) for name in lineage.tolist()])
    return annotate_cells(
        cells,
        reference.embeddings,
        codes,
        classes,
        axis="compartment_v1",
        cell_space_id=reference.info.cell_space_id,
    )


def assert_same_cells(read: CellEmbeddings, written: CellEmbeddings) -> None:
    """Check that cells read back hold what was written, the tiles read with them aside."""
    assert read.cell_id.dtype == written.cell_id.dtype
    assert read.cell_id.tolist() == written.cell_id.tolist()
    for name in (
        "backbone",
        "cell_space",
        "alpha",
        "tile_index",
        "n_candidate_tiles",
        "valid",
        "centroids",
    ):
        np.testing.assert_array_equal(getattr(read, name), getattr(written, name), err_msg=name)
    for name in (
        "encoder_id",
        "cell_space_id",
        "geometry",
        "membership_source",
        "membership_approximate",
        "n_cells_without_tokens",
        "centroids_frame",
        "centroids_mpp",
    ):
        assert getattr(read, name) == getattr(written, name), name
    assert read.tiles is None


def test_tile_embeddings_to_anndata_holds_bounds_and_centers(tiles):
    table = tiles.to_anndata()

    assert table.shape == (3, 0)
    assert table.obs[TILE_ID_COLUMN].tolist() == ["0", "1", "2"]
    assert table.obs[TILE_INDEX_COLUMN].tolist() == [0, 1, 2]
    np.testing.assert_array_equal(table.obsm[TILE_EMBEDDING_KEY], tiles.embeddings)
    bounds = np.column_stack([table.obs[name].to_numpy() for name in TILE_BOUNDS_COLUMNS])
    np.testing.assert_array_equal(bounds, np.column_stack([ORIGINS, ORIGINS + INPUT_SIZE]))
    np.testing.assert_array_equal(table.obsm[SPATIAL_KEY], tiles.centers)
    record = table.uns[TILES_PROVENANCE_KEY]
    assert record["encoder_id"] == ENCODER_ID
    assert record["bounds_frame"] == ORIGINS_FRAME
    assert record["purpose"] == "tiles"


def test_tile_embeddings_round_trip_through_h5ad(tiles, tmp_path):
    path = tmp_path / "tiles.h5ad"
    tiles.write_h5ad(path)

    back = read_tile_embeddings(path)

    np.testing.assert_array_equal(back.embeddings, tiles.embeddings)
    np.testing.assert_array_equal(back.bounds, tiles.bounds)
    assert back.tile_ids == tiles.tile_ids
    assert back.tile_index.tolist() == tiles.tile_index.tolist()
    assert (back.bounds_frame, back.bounds_mpp) == (ORIGINS_FRAME, REFERENCE_MPP)
    assert back.geometry == tiles.geometry
    assert back.encoder_id == tiles.encoder_id
    assert back.purpose == "tiles"


def test_cell_embeddings_to_anndata_holds_arrays_and_positions(cells):
    table = cells.to_anndata()

    assert table.obs[CELL_ID_COLUMN].tolist() == list(range(1, 13))
    assert table.obs[TILE_INDEX_COLUMN].tolist() == np.repeat([0, 1, 2], 4).tolist()
    assert table.obs[NUCLEUS_PX_COLUMN].tolist() == [NUCLEUS_SIDE**2] * 12
    np.testing.assert_array_equal(table.obsm[BACKBONE_KEY], cells.backbone)
    assert table.obsm[CELL_SPACE_KEY].shape == (12, CELL_DIM)
    points, tile = make_centroids()
    np.testing.assert_allclose(table.obsm[SPATIAL_KEY], points + ORIGINS[tile])
    record = table.uns[CELLS_PROVENANCE_KEY]
    assert record["cell_space_id"] == CELL_SPACE
    assert record["membership_source"] == "labels"
    assert record["centroids_frame"] == ORIGINS_FRAME


@pytest.mark.parametrize("text_ids", [False, True], ids=["integer ids", "text ids"])
def test_cell_embeddings_round_trip_through_h5ad(cells, tmp_path, text_ids):
    if text_ids:
        cells = replace(cells, cell_id=np.array([f"cell-{cell}" for cell in cells.cell_id]))
    path = tmp_path / "cells.h5ad"
    cells.write_h5ad(path)

    assert_same_cells(read_cells(path), cells)


def test_annotation_result_to_anndata_names_fields_by_axis(result):
    table = result.to_anndata()

    label = f"{LABEL_COLUMN}_lineage_v1"
    assert table.obs[label].tolist() == result.labels.tolist()
    np.testing.assert_array_equal(table.obs[f"{CONFIDENCE_COLUMN}_lineage_v1"], result.confidence)
    np.testing.assert_array_equal(table.obsm[f"{MASS_KEY}_lineage_v1"], result.mass)
    np.testing.assert_array_equal(table.obsm[CELL_SPACE_KEY], result.cells.cell_space)
    assert len(table.uns[f"{label}_colors"]) == table.obs[label].cat.categories.size
    record = table.uns[PROVENANCE_KEY][ANNOTATIONS_KEY]["lineage_v1"]
    assert record["axis"] == "lineage_v1"
    assert list(record["class_order"]) == list(LINEAGE)


def test_annotation_result_round_trips_through_h5ad(result, tmp_path):
    path = tmp_path / "annotations.h5ad"
    result.write_h5ad(path)

    back = read_annotations(path)

    assert back.cell_id.tolist() == result.cell_id.tolist()
    assert back.labels.tolist() == result.labels.tolist()
    for name in ("confidence", "mass", "class_entropy", "neighbor_entropy", "valid"):
        np.testing.assert_array_equal(getattr(back, name), getattr(result, name), err_msg=name)
    assert (back.axis, back.class_order) == (result.axis, result.class_order)
    assert (back.meta["k"], back.meta["reference"]) == (result.meta["k"], "prostate")
    assert back.missing_classes == ()
    # The similarities of the neighbors are not written.
    assert back.neighbor_similarity.shape == (12, 0)
    assert_same_cells(back.cells, result.cells)


def test_add_annotation_keeps_two_axes_side_by_side(result, reference, tmp_path):
    compartments = compartment_annotations(result.cells, reference)
    table = result.to_anndata()

    add_annotation(table, compartments)
    path = tmp_path / "annotations.h5ad"
    table.write_h5ad(path)

    lineage = read_annotations(path, name="lineage_v1")
    compartment = read_annotations(path, name="compartment_v1")
    assert (lineage.axis, compartment.axis) == ("lineage_v1", "compartment_v1")
    assert lineage.labels.tolist() == result.labels.tolist()
    assert compartment.labels.tolist() == compartments.labels.tolist()
    assert compartment.class_order == compartments.class_order
    np.testing.assert_array_equal(table.obsm[f"{MASS_KEY}_lineage_v1"], result.mass)
    np.testing.assert_array_equal(table.obsm[f"{MASS_KEY}_compartment_v1"], compartments.mass)
    assert read_cells(path).cell_id.tolist() == result.cell_id.tolist()


def test_add_annotation_refuses_its_arguments_swapped(result):
    table = result.cells.to_anndata()
    with pytest.raises(AlhenaValueError, match="the table first"):
        add_annotation(result, table)


def test_add_annotation_refuses_annotations_of_other_cells(result):
    table = result.cells.to_anndata()

    with pytest.raises(AlhenaValueError, match="do not agree cell for cell"):
        add_annotation(table, result.select(np.arange(6)))
    assert f"{LABEL_COLUMN}_lineage_v1" not in table.obs.columns


def test_annotation_result_select_cuts_its_cells_alike(result):
    kept = result.select(result.cells.tile_index == 1)

    assert kept.cell_id.tolist() == [5, 6, 7, 8]
    assert kept.cells.cell_id.tolist() == [5, 6, 7, 8]
    assert kept.labels.tolist() == result.labels[4:8].tolist()
    np.testing.assert_array_equal(kept.mass, result.mass[4:8])
    np.testing.assert_array_equal(kept.cells.centroids, result.cells.centroids[4:8])
    assert kept.class_order == result.class_order
    assert result.select([7, 0]).cells.cell_id.tolist() == [8, 1]


def test_tile_embeddings_select_keeps_the_order_given(tiles):
    kept = tiles.select([2, 0])

    assert kept.tile_ids == ("2", "0")
    np.testing.assert_array_equal(kept.embeddings, tiles.embeddings[[2, 0]])
    np.testing.assert_array_equal(kept.bounds, tiles.bounds[[2, 0]])
    assert kept.bounds_frame == tiles.bounds_frame


def test_locate_names_the_tile_holding_each_cell(tiles, result):
    assert tiles.locate(result.cells).tolist() == TILE_OF_EACH_CELL
    assert tiles.locate(result).tolist() == TILE_OF_EACH_CELL
    assert tiles.locate(result.cells.to_anndata()).tolist() == TILE_OF_EACH_CELL
    assert tiles.locate(np.array([[5000.0, 50.0]])).tolist() == [None]


def test_to_frame_holds_one_row_per_cell(result):
    frame = result.to_frame()

    assert frame["cell_id"].tolist() == result.cell_id.tolist()
    assert frame["label"].tolist() == result.labels.tolist()
    for position, name in enumerate(result.class_order):
        np.testing.assert_array_equal(frame[f"mass_{name}"], result.mass[:, position])
    np.testing.assert_array_equal(frame[["x", "y"]], result.cells.centroids)
    assert frame["tile_id"].tolist() == TILE_OF_EACH_CELL


def test_missing_classes_names_the_classes_a_reference_lacks(encoder, cells, tmp_path):
    absent = LINEAGE[-1]
    # Cells of every class in class order, and none of the last.
    counts = (CELLS_PER_CLASS,) * (len(LINEAGE) - 1) + (0,)
    shard = make_shard(tmp_path, name="thin", codes=class_codes(counts))
    with pytest.warns(ReferenceCoverageWarning, match=absent):
        reference = load_reference(path=shard)
    with pytest.warns(ReferenceCoverageWarning, match=absent):
        result = CellTyper(encoder, reference).annotate(cells)
    path = tmp_path / "annotations.h5ad"
    result.write_h5ad(path)

    assert result.missing_classes == (absent,)
    assert read_annotations(path).missing_classes == (absent,)
