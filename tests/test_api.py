"""Tests for loading the encoder and a cell typer, and for the names the package exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import alhena
import alhena.api
import alhena.hub
from alhena import REFERENCE_MPP, AlhenaEncoder, AlhenaValueError
from alhena._constants import WEIGHTS_REPO
from support import (
    CELL_SPACE,
    EMBED_DIM,
    LINEAGE,
    N_TILES,
    NUCLEI_PER_TILE,
    make_label_maps,
    make_shard,
    make_tiles,
    tiny_encoder,
)


def install_download_layer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Replace the downloads and the reading of the weights, recording what each call is given."""
    seen: dict[str, Any] = {}

    def find_backbone(backbone: str | Path | None = None) -> Path:
        seen["backbone"] = backbone
        return tmp_path / "pytorch_model.bin"

    def ensure_bundle(bundle: str | Path, **kwargs: Any) -> Path:
        seen["bundle"] = bundle
        seen["bundle_options"] = kwargs
        return tmp_path / "bundle"

    def ensure_reference(shard: str, **kwargs: Any) -> Path:
        seen["shard"] = shard
        return make_shard(tmp_path, name=shard)

    def load_bundle(bundle: Path, backbone: Path, **kwargs: Any) -> AlhenaEncoder:
        seen["load_bundle"] = (bundle, backbone, kwargs)
        seen["encoder"] = tiny_encoder()
        return seen["encoder"]

    monkeypatch.setattr(alhena.hub, "find_backbone", find_backbone)
    monkeypatch.setattr(alhena.hub, "ensure_bundle", ensure_bundle)
    monkeypatch.setattr(alhena.hub, "ensure_reference", ensure_reference)
    monkeypatch.setattr(alhena.api, "load_bundle", load_bundle)
    return seen


# --- Loaders --------------------------------------------------------------------------------


def test_load_encoder_fetches_the_published_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    seen = install_download_layer(monkeypatch, tmp_path)
    encoder = alhena.load_encoder()
    assert seen["backbone"] is None
    assert seen["bundle"] == WEIGHTS_REPO
    assert seen["load_bundle"][:2] == (tmp_path / "bundle", tmp_path / "pytorch_model.bin")
    assert encoder is seen["encoder"]
    embedded = encoder.encode_tiles(make_tiles(), mpp=REFERENCE_MPP)
    assert embedded.embeddings.shape == (N_TILES, EMBED_DIM)


def test_load_encoder_passes_its_arguments_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    seen = install_download_layer(monkeypatch, tmp_path)
    alhena.load_encoder(
        backbone=tmp_path / "uni",
        bundle="owner/name",
        device="cpu",
        cache_dir=tmp_path / "cache",
        merge=True,
    )
    assert seen["backbone"] == tmp_path / "uni"
    assert seen["bundle"] == "owner/name"
    assert seen["bundle_options"] == {"cache_dir": tmp_path / "cache"}
    options = seen["load_bundle"][2]
    assert options["device"] == "cpu"
    assert options["merge"] is True


def test_load_celltyper_loads_a_tissue_and_its_encoder_on_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seen = install_download_layer(monkeypatch, tmp_path)
    typer = alhena.load_celltyper("Prostate", backbone=tmp_path / "uni", device="cpu")
    assert seen["shard"] == "prostate"
    assert typer.encoder is None
    result = typer.annotate(make_tiles(), make_label_maps(), mpp=REFERENCE_MPP)
    assert typer.encoder is seen["encoder"]
    assert seen["backbone"] == tmp_path / "uni"
    assert seen["load_bundle"][2]["device"] == "cpu"
    assert len(result) == N_TILES * NUCLEI_PER_TILE
    assert set(result.labels.tolist()) <= set(LINEAGE)


def test_load_celltyper_annotates_against_a_shard_file(tmp_path: Path):
    encoder = tiny_encoder()
    typer = alhena.load_celltyper(reference=make_shard(tmp_path), encoder=encoder)
    result = typer.annotate(make_tiles(), make_label_maps(), mpp=REFERENCE_MPP)
    assert typer.encoder is encoder
    assert typer.cell_space_id == CELL_SPACE
    assert result.mass.shape == (N_TILES * NUCLEI_PER_TILE, len(LINEAGE))
    assert result.valid.all()


def test_load_celltyper_refuses_an_encoder_beside_its_options(tmp_path: Path):
    with pytest.raises(AlhenaValueError, match="how to build an encoder"):
        alhena.load_celltyper(reference=make_shard(tmp_path), encoder=tiny_encoder(), device="cpu")


# --- The public surface ---------------------------------------------------------------------


def test_every_exported_name_is_reachable():
    assert [name for name in alhena.__all__ if not hasattr(alhena, name)] == []


def test_missing_extra_error_is_an_import_and_alhena_error():
    assert issubclass(alhena.MissingExtraError, ImportError)
    assert issubclass(alhena.MissingExtraError, alhena.AlhenaError)
