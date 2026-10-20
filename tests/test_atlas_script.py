"""Tests for the atlas embedding script: embedding an atlas, gene identifiers, the command line."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import anndata
import numpy as np
import pandas as pd
import pytest

from alhena.refbuild import read_scconcept_embeddings

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "alhena_atlas_embed.py"

# A special token names no symbol, and CD99 names two genes, a pseudoautosomal pair.
GENE_TABLE = (
    "gene_id,gene_name,token\n"
    "<cls>,,0\n"
    "ENSG00000141510,TP53,1\n"
    "ENSG00000119888,EPCAM,2\n"
    "ENSG00000002586,CD99,3\n"
    "ENSG00000292348,CD99,4\n"
)


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the script, which stands beside the package rather than in it."""
    spec = importlib.util.spec_from_file_location("alhena_atlas_embed", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses look their module up by name while they are built.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class StandInTokenizer:
    """The tokenizer surface the script reads, every gene in its vocabulary."""

    species = ("hsapiens",)
    NOT_FOUND = -1

    def encode(self, gene_ids: np.ndarray, species: str) -> np.ndarray:
        return np.arange(len(gene_ids))


@pytest.fixture
def stand_in_model(script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand a model in for the single-cell model, each cell embedded as its total count."""
    model = SimpleNamespace(
        cfg=SimpleNamespace(
            model=SimpleNamespace(pe_max_len=2048),
            datamodule=SimpleNamespace(normalization="log1p"),
        ),
        tokenizer=StandInTokenizer(),
        device="cpu",
    )

    def embed_chunk(model: Any, chunk: Any, settings: Any, device: Any, on_batch: Any) -> int:
        totals = np.asarray(chunk.X).sum(axis=1)
        for start in range(0, chunk.n_obs, settings.batch_size):
            batch = totals[start : start + settings.batch_size]
            on_batch(start, np.repeat(batch[:, None], script.EMBEDDING_DIM, axis=1))
        return chunk.n_obs

    monkeypatch.setattr(script, "load_model", lambda *args: model)
    monkeypatch.setattr(script, "_embed_chunk", embed_chunk)
    monkeypatch.setattr(script, "_installed_version", lambda name: script.SCCONCEPT_PACKAGE_VERSION)


def write_atlas(path: Path, counts: np.ndarray) -> Path:
    """Write an atlas over genes named by their Ensembl identifiers."""
    n_cells, n_genes = counts.shape
    var = pd.DataFrame(
        {"gene_ids": [f"ENSG{index:011d}" for index in range(n_genes)]},
        index=[f"G{index}" for index in range(n_genes)],
    )
    obs = pd.DataFrame(index=[f"cell{index}" for index in range(n_cells)])
    anndata.AnnData(X=counts.astype(np.float32), obs=obs, var=var).write_h5ad(path)
    return path


def test_embed_atlas_writes_every_cell_in_the_row_order_of_the_atlas(
    script: ModuleType, stand_in_model: None, tmp_path: Path
):
    counts = np.random.default_rng(0).integers(0, 5, size=(10, 4))
    atlas = write_atlas(tmp_path / "atlas.h5ad", counts)
    output = tmp_path / "embeddings.h5"
    # A memory budget of three cells, so the atlas is embedded in four passes.
    budget = 3 * counts.shape[1] * script.BYTES_PER_VALUE / 1024.0**3

    summary = script.embed_atlas(
        atlas, output, model_cache=tmp_path / "models", chunk_memory_gb=budget, batch_size=2
    )
    assert summary["n_cells"] == 10
    np.testing.assert_array_equal(read_scconcept_embeddings(output)[:, 0], counts.sum(axis=1))
    assert not output.with_name(output.name + ".partial").exists()


def test_embed_atlas_refuses_normalized_counts_and_writes_nothing(
    script: ModuleType, tmp_path: Path
):
    atlas = write_atlas(tmp_path / "atlas.h5ad", np.log1p(np.arange(8.0).reshape(2, 4)))
    output = tmp_path / "embeddings.h5"

    with pytest.raises(script.AtlasEmbedValueError, match="not integer-valued"):
        script.embed_atlas(atlas, output, model_cache=tmp_path / "models")
    assert list(tmp_path.glob("embeddings.h5*")) == []


def test_normalize_gene_ids_drops_versions_and_keeps_par_y_tags(script: ModuleType):
    given = ["ENSG00000141510.17", "ENSG00000002586.20_PAR_Y", "ENSG00000000003", "AC000061.1"]
    identifiers, counts = script.normalize_gene_ids(np.array(given))
    assert identifiers.tolist() == [
        "ENSG00000141510",
        "ENSG00000002586_PAR_Y",
        "ENSG00000000003",
        "AC000061.1",
    ]
    assert counts == {"n_gene_ids_normalized": 2, "n_gene_ids_par_y": 1}


def test_map_gene_symbols_maps_a_symbol_naming_one_gene(script: ModuleType, tmp_path: Path):
    path = tmp_path / "hsapiens.csv"
    path.write_text(GENE_TABLE, encoding="utf-8")
    table = script.read_gene_table(path)
    with pytest.warns(script.AtlasEmbedWarning, match="CD99"):
        identifiers, counts = script.map_gene_symbols(
            np.array(["TP53", "CD99", "NOTAGENE", "EPCAM"]), table
        )
    assert identifiers.tolist() == ["ENSG00000141510", "CD99", "NOTAGENE", "ENSG00000119888"]
    assert counts == {
        "n_gene_symbols_mapped": 2,
        "n_gene_symbols_ambiguous": 1,
        "n_gene_symbols_unmatched": 1,
    }


def test_main_refuses_a_gene_table_without_symbols(
    script: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    models = tmp_path / "models"
    table = models / script.MODEL_NAME / script.GENE_MAPPINGS_DIRNAME / f"{script.SPECIES}.csv"
    table.parent.mkdir(parents=True)
    table.write_text("gene_id,token\nENSG00000141510,1\n", encoding="utf-8")
    output = tmp_path / "embeddings.h5"
    arguments = ["--atlas", str(tmp_path / "atlas.h5ad"), "--output", str(output)]
    arguments += ["--model-cache", str(models), "--gene-id-type", "symbol"]
    assert script.main(arguments) == 1
    assert "gene_name" in capsys.readouterr().err
    assert not output.exists()


def test_main_prints_its_summary_as_json(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    given: dict[str, Any] = {}
    output = tmp_path / "embeddings.h5"
    summary = {"output": str(output), "n_cells": 2, "embedding_dim": 512}

    def embed_atlas(atlas: Path, output: Path, **settings: Any) -> dict[str, Any]:
        given.update(settings, atlas=atlas)
        return summary

    monkeypatch.setattr(script, "embed_atlas", embed_atlas)
    arguments = ["--atlas", str(tmp_path / "atlas.h5ad"), "--output", str(output)]
    assert script.main([*arguments, "--gene-id-type", "symbol", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == summary
    assert given["atlas"] == tmp_path / "atlas.h5ad"
    assert given["gene_id_type"] == "symbol"
