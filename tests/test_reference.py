"""Tests for loading, describing and building reference shards."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.numpy import save_file

import alhena
from alhena.encoder import build_heads
from alhena.refbuild import encode_labels, read_labels
from support import (
    CELL_SPACE,
    CELLS_PER_CLASS,
    COMPARTMENT_OF,
    EMBED_DIM,
    LINEAGE,
    PixelBackbone,
    bundle_info,
    class_codes,
    make_shard,
)

# Width of the single-cell embeddings the reference head reads.
SCCONCEPT_WIDTH = 512
UNLABELED = "Unknown"
N_UNLABELED = 40
N_LABELED = CELLS_PER_CLASS * len(LINEAGE)
EVEN_SUPPORT = dict.fromkeys(LINEAGE, CELLS_PER_CLASS)


def make_atlas(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return single-cell embeddings of separable classes, exact in half precision, and labels."""
    rng = np.random.default_rng(seed)
    names = np.array([*LINEAGE, UNLABELED])
    counts = [CELLS_PER_CLASS] * len(LINEAGE) + [N_UNLABELED]
    codes = rng.permutation(np.repeat(np.arange(names.size), counts))
    directions = rng.normal(size=(names.size, SCCONCEPT_WIDTH))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    values = directions[codes] + rng.normal(scale=0.01, size=(codes.size, SCCONCEPT_WIDTH))
    return values.astype(np.float16).astype(np.float32), names[codes]


def write_hdf5(path: Path, values: np.ndarray) -> Path:
    """Write embeddings to the `cls_emb` dataset of an HDF5 file, as the embedding step does."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cls_emb", data=values)
    return path


def reference_encoder() -> alhena.AlhenaEncoder:
    """Return an encoder whose reference head reverses the columns of a single-cell embedding."""
    heads = build_heads(
        {"st_node_proj.weight": torch.eye(SCCONCEPT_WIDTH).flip(0)}, embed_dim=EMBED_DIM
    )
    return alhena.AlhenaEncoder(PixelBackbone(), heads, bundle_info())


def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_load_reference_reads_shard_and_drops_unlabeled_rows(tmp_path: Path):
    codes = np.concatenate([class_codes(), np.full(N_UNLABELED, -1, dtype=np.int8)])
    provenance = {"source": "a synthetic atlas", "license": "CC0-1.0"}
    path = make_shard(tmp_path, codes=codes, metadata=provenance)
    reference = alhena.load_reference(path=path, axis="lineage_v1")

    info = reference.info
    assert (info.name, info.axis) == ("prostate", "lineage_v1")
    assert info.cell_space_id == CELL_SPACE
    assert (info.n_cells, info.n_scored_cells) == (N_LABELED + N_UNLABELED, N_LABELED)
    assert info.class_order == reference.class_order == LINEAGE
    assert info.class_support == EVEN_SUPPORT
    assert (info.source, info.license) == ("a synthetic atlas", "CC0-1.0")
    assert info.missing_classes == info.thin_classes == ()
    assert len(reference) == N_LABELED
    assert reference.embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(reference.embeddings, axis=1), 1.0, atol=1e-5)
    assert reference.label_ids.tolist() == class_codes().tolist()


def test_describe_shard_reports_missing_class_without_warning(tmp_path: Path):
    counts = {name: 0 if name == "Stromal_Vascular" else CELLS_PER_CLASS for name in LINEAGE}
    path = make_shard(tmp_path, codes=class_codes(tuple(counts.values())))
    info = alhena.describe_shard(path=path)
    assert info.class_support == counts
    assert info.missing_classes == ("Stromal_Vascular",)

    with pytest.warns(alhena.ReferenceCoverageWarning, match="Stromal_Vascular"):
        reference = alhena.load_reference(path=path)
    assert reference.info == info


def test_load_reference_finds_shard_of_alias_in_cache(tmp_path: Path):
    shards = tmp_path / "shards"
    shards.mkdir()
    make_shard(shards, name="pancreas_cell")
    reference = alhena.load_reference(tissue="Pancreatic", cache_dir=tmp_path)
    assert reference.info.name == "pancreas_cell"
    assert len(reference) == N_LABELED
    assert alhena.describe_shard(tissue="Pancreas", cache_dir=tmp_path) == reference.info


def test_shard_for_tissue_resolves_tissues_and_aliases():
    tissues = alhena.available_tissues()
    assert {alhena.shard_for_tissue(tissue) for tissue in tissues} == set(alhena.available_shards())
    assert alhena.shard_for_tissue("LymphNode") == alhena.shard_for_tissue("Tonsil") == "tonsil"
    assert alhena.shard_for_tissue("lymph_node") == "tonsil"
    assert alhena.shard_for_tissue("Pancreatic") == "pancreas_cell"
    assert "Pancreatic" not in tissues


def test_read_scconcept_embeddings_reads_hdf5(tmp_path: Path):
    values, _ = make_atlas()
    read = alhena.read_scconcept_embeddings(write_hdf5(tmp_path / "atlas.h5", values))
    assert read.dtype == np.float32
    np.testing.assert_array_equal(read, values)


def test_read_labels_reads_column_of_table(tmp_path: Path):
    path = tmp_path / "labels.csv"
    table = pd.DataFrame({"barcode": ["a", "b", "c"], "cell_type": ["T_NK", None, " Myeloid "]})
    table.to_csv(path, index=False)
    assert read_labels(path, n_rows=3, column="cell_type").tolist() == ["T_NK", "", "Myeloid"]


def test_encode_labels_codes_classes_and_unlabeled_rows():
    order = alhena.label_order("lineage_v1")
    codes, counts = encode_labels(["T_NK", UNLABELED, "B_Plasma", "", "T_NK"], order)
    assert codes.dtype == np.int8
    t_nk, b_plasma = order.index("T_NK"), order.index("B_Plasma")
    assert codes.tolist() == [t_nk, -1, b_plasma, -1, t_nk]
    assert counts == {**dict.fromkeys(LINEAGE, 0), "B_Plasma": 1, "T_NK": 2}


def test_build_reference_round_trips_through_load_reference(tmp_path: Path):
    values, labels = make_atlas()
    embeddings = write_hdf5(tmp_path / "atlas.h5", values)
    report = alhena.build_reference(
        embeddings, labels, reference_encoder(), tmp_path / "atlas.safetensors"
    )
    labeled = labels != UNLABELED
    assert isinstance(report, alhena.ShardBuildReport)
    assert (report.name, report.axes, report.n_rows) == ("atlas", ("lineage_v1",), labels.size)
    assert report.n_labeled == {"lineage_v1": N_LABELED}
    assert report.class_support == {"lineage_v1": EVEN_SUPPORT}
    assert report.cell_space_id == CELL_SPACE
    assert report.self_check.accuracy == 1.0

    reference = alhena.load_reference(path=report.path)
    assert reference.info == report.info
    np.testing.assert_allclose(reference.embeddings, unit_rows(values[labeled][:, ::-1]), atol=1e-6)
    assert reference.label_ids.tolist() == [LINEAGE.index(name) for name in labels[labeled]]


def test_build_reference_codes_each_axis_of_the_labels(tmp_path: Path):
    values, labels = make_atlas()
    compartments = np.array([COMPARTMENT_OF.get(name, name) for name in labels])
    report = alhena.build_reference(
        values,
        {"lineage_v1": labels, "compartment_v1": compartments},
        reference_encoder(),
        tmp_path / "atlas.safetensors",
        self_check_cells=200,
    )
    assert report.axes == ("lineage_v1", "compartment_v1")

    reference = alhena.load_reference(path=report.path, axis="compartment_v1")
    assert reference.info.axis == "compartment_v1"
    assert reference.info.class_support == {
        "Epithelial_Melanocyte": CELLS_PER_CLASS,
        "Immune": 3 * CELLS_PER_CLASS,
        "Stromal_Vascular": CELLS_PER_CLASS,
    }
    lineage = alhena.load_reference(path=report.path, axis="lineage_v1")
    assert lineage.class_order == LINEAGE


def test_build_reference_rejects_label_outside_axis(tmp_path: Path):
    values, labels = make_atlas()
    labels[0] = "Neuron"
    output = tmp_path / "atlas.safetensors"
    with pytest.raises(alhena.AlhenaValueError, match="match no class"):
        alhena.build_reference(values, labels, reference_encoder(), output)
    assert not output.exists()


def test_is_user_built_tells_built_shard_from_prebuilt(tmp_path: Path):
    values, labels = make_atlas()
    built = alhena.build_reference(
        values, labels, reference_encoder(), tmp_path / "atlas.safetensors", self_check_cells=200
    )
    prebuilt = make_shard(tmp_path)
    assert alhena.is_user_built(built.path)
    assert alhena.is_user_built(alhena.load_reference(path=built.path))
    assert not alhena.is_user_built(prebuilt)
    assert not alhena.is_user_built(alhena.describe_shard(path=prebuilt))


def test_load_reference_rejects_shard_of_another_width(tmp_path: Path):
    path = tmp_path / "narrow.safetensors"
    codes = class_codes()
    save_file(
        {"embeddings": np.ones((codes.size, 256), np.float16), "lineage_v1": codes}, str(path)
    )
    with pytest.raises(alhena.ReferenceError, match="256-dimensional"):
        alhena.load_reference(path=path)


def test_load_reference_rejects_unknown_tissue():
    with pytest.raises(alhena.AlhenaValueError, match="No reference is published for 'Spleen'"):
        alhena.load_reference(tissue="Spleen")


def test_load_reference_rejects_axis_the_shard_lacks(tmp_path: Path):
    path = make_shard(tmp_path)
    with pytest.raises(alhena.ReferenceError, match="no labels for the axis 'compartment_v1'"):
        alhena.load_reference(path=path, axis="compartment_v1")


def test_require_cell_space_refuses_another_encoder(tmp_path: Path):
    reference = alhena.load_reference(path=make_shard(tmp_path))
    reference.require_cell_space(CELL_SPACE)
    with pytest.raises(alhena.CellSpaceMismatchError):
        reference.require_cell_space("ffffffffffffffff")
