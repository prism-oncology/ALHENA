"""Tests for the alhena command line."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest

import alhena
import alhena.api
import alhena.demo
from alhena import (
    REFERENCE_MPP,
    TISSUE_TO_SHARD,
    AlhenaEncoder,
    BackboneNotFoundError,
    CellTyper,
    _progress,
    load_reference,
)
from alhena.cli import geometry_path, main
from alhena.hub import BUNDLE_FILES, CACHE_ENV_VAR, SHARD_DIRECTORY
from alhena.io import read_annotations, read_cells, read_tile_embeddings
from support import (
    CELL_DIM,
    CELL_SPACE,
    EMBED_DIM,
    ENCODER_ID,
    LINEAGE,
    N_TILES,
    NUCLEI_PER_TILE,
    make_label_maps,
    make_shard,
    make_tiles,
    tiny_encoder,
)

MPP = str(REFERENCE_MPP)
N_CELLS = N_TILES * NUCLEI_PER_TILE


@pytest.fixture(autouse=True)
def typer_loaded_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Stand the small encoder and a shard on disk in for the published models, in every test."""
    seen: dict[str, Any] = {}

    def load_encoder(**kwargs: Any) -> AlhenaEncoder:
        return tiny_encoder()

    def load_celltyper(**kwargs: Any) -> CellTyper:
        seen.update(kwargs)
        shard = make_shard(tmp_path)
        return CellTyper(None, load_reference(path=shard, axis=kwargs["axis"]))

    monkeypatch.setattr(alhena.api, "load_encoder", load_encoder)
    monkeypatch.setattr(alhena.api, "load_celltyper", load_celltyper)
    return seen


def write_stack(directory: Path) -> tuple[Path, Path]:
    """Write stacked tiles and their label maps as two .npy arrays."""
    tiles, nuclei = directory / "tiles.npy", directory / "nuclei.npy"
    np.save(tiles, make_tiles())
    np.save(nuclei, make_label_maps())
    return tiles, nuclei


def write_cached_shard(root: Path, name: str) -> Path:
    """Write a shard where a cache, or the repository filling it, keeps it."""
    (root / SHARD_DIRECTORY).mkdir(parents=True, exist_ok=True)
    return make_shard(root / SHARD_DIRECTORY, name=name)


def install_hub(monkeypatch: pytest.MonkeyPatch, published: Path) -> list[str]:
    """Serve the files of a folder as the model hub would, recording the name of each fetched."""
    fetched: list[str] = []

    def hf_hub_download(repo_id: str, filename: str, cache_dir: str, revision: str | None) -> str:
        fetched.append(filename)
        target = Path(cache_dir) / repo_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(published / filename, target)
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_hub_download)
    return fetched


def refusal(capsys: pytest.CaptureFixture[str]) -> str:
    """Return what a refused command printed on standard error, checking it holds no traceback."""
    err = capsys.readouterr().err
    assert "Traceback" not in err
    return err


# --- Program ---


def test_version_prints_the_package_version(capsys):
    assert main(["--version"]) == 0
    assert alhena.__version__ in capsys.readouterr().out


def test_help_exits_zero_for_every_command(capsys):
    commands = [
        [],
        ["info"],
        ["download"],
        ["encode"],
        ["annotate"],
        ["reference"],
        ["reference", "list"],
        ["reference", "describe"],
        ["reference", "check"],
        ["demo"],
    ]
    for command in commands:
        assert main([*command, "--help"]) == 0
        assert capsys.readouterr().out


def test_unknown_command_is_a_usage_error(capsys):
    assert main(["nosuchcommand"]) == 2
    assert "nosuchcommand" in refusal(capsys)


# --- encode ---


def test_encode_writes_tile_embeddings_from_a_npy_stack(tmp_path):
    tiles, _ = write_stack(tmp_path)
    out = tmp_path / "embeddings.npy"

    assert main(["encode", str(tiles), "--out", str(out), "--mpp", MPP]) == 0
    expected = tiny_encoder().encode(make_tiles(), mpp=REFERENCE_MPP)
    np.testing.assert_allclose(np.load(out), expected, rtol=1e-5)
    record = json.loads(geometry_path(out).read_text())
    assert record["embedding_shape"] == [N_TILES, EMBED_DIM]
    assert record["encoder_id"] == ENCODER_ID


def test_encode_reads_a_directory_of_images(tmp_path):
    folder = tmp_path / "tiles"
    folder.mkdir()
    names = [f"tile_{index}" for index in range(N_TILES)]
    for name, tile in zip(names, make_tiles(), strict=True):
        cv2.imwrite(str(folder / f"{name}.png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
    out = tmp_path / "tiles.h5ad"

    assert main(["encode", str(folder), "--out", str(out), "--mpp", MPP]) == 0
    tiles = read_tile_embeddings(out)
    assert tiles.tile_ids == tuple(names)
    assert tiles.encoder_id == ENCODER_ID
    expected = tiny_encoder().encode(make_tiles(), mpp=REFERENCE_MPP)
    np.testing.assert_allclose(tiles.embeddings, expected, rtol=1e-5)


def test_encode_gives_tile_embeddings_the_bounds_of_their_origins(tmp_path):
    tiles, _ = write_stack(tmp_path)
    side = make_tiles().shape[1]
    origins = tmp_path / "origins.csv"
    names = [str(index) for index in range(N_TILES)]
    x = [index * side for index in range(N_TILES)]
    pd.DataFrame({"tile_id": names, "x": x, "y": 0}).to_csv(origins, index=False)
    out = tmp_path / "tiles.h5ad"

    arguments = ["--coords", str(origins), "--out", str(out), "--mpp", MPP]
    assert main(["encode", str(tiles), *arguments]) == 0
    embedded = read_tile_embeddings(out)
    assert embedded.bounds_frame == "tile origins"
    expected = [[start, 0, start + side, side] for start in x]
    np.testing.assert_array_equal(embedded.bounds, expected)


def test_encode_keeps_the_tile_names_of_an_hdf5_file(tmp_path):
    names = [f"spot_{index}" for index in range(N_TILES)]
    tiles = tmp_path / "tiles.h5"
    with h5py.File(tiles, "w") as handle:
        handle.create_dataset("img", data=make_tiles())
        handle.create_dataset("barcode", data=names, dtype=h5py.string_dtype())
    out = tmp_path / "embeddings.h5"

    assert main(["encode", str(tiles), "--out", str(out), "--mpp", MPP]) == 0
    with h5py.File(out, "r") as handle:
        assert handle["embeddings"].shape == (N_TILES, EMBED_DIM)
        assert handle["barcode"].asstr()[()].tolist() == names


def test_encode_with_nuclei_writes_a_table_of_cells(tmp_path):
    tiles, nuclei = write_stack(tmp_path)
    out = tmp_path / "cells.h5ad"

    code = main(["encode", str(tiles), "--nuclei", str(nuclei), "--out", str(out), "--mpp", MPP])
    assert code == 0
    cells = read_cells(out)
    assert len(cells) == N_CELLS
    assert cells.cell_space.shape == (N_CELLS, CELL_DIM)
    assert cells.cell_space_id == CELL_SPACE


def test_encode_refuses_an_image_without_mpp(tmp_path, capsys):
    image = tmp_path / "region.png"
    cv2.imwrite(str(image), make_tiles(n_tiles=1)[0])

    assert main(["encode", str(image), "--out", str(tmp_path / "embeddings.npy")]) == 1
    assert "--mpp" in refusal(capsys)


def test_encode_refuses_tiles_that_are_not_there(tmp_path, capsys):
    missing = tmp_path / "absent.npy"

    assert main(["encode", str(missing), "--out", str(tmp_path / "embeddings.npy")]) == 1
    assert str(missing) in refusal(capsys)


# --- annotate ---


def test_annotate_writes_annotations_to_csv(tmp_path, typer_loaded_with):
    tiles, nuclei = write_stack(tmp_path)
    out = tmp_path / "annotations.csv"
    arguments = ["--nuclei", str(nuclei), "--tissue", "Prostate", "--mpp", MPP]

    assert main(["annotate", str(tiles), *arguments, "--out", str(out)]) == 0
    frame = pd.read_csv(out)
    assert len(frame) == N_CELLS
    assert set(frame["label"]) <= set(LINEAGE)
    assert typer_loaded_with["tissue"] == "Prostate"


def test_annotate_writes_annotations_with_cells_to_h5ad(tmp_path):
    tiles, nuclei = write_stack(tmp_path)
    out = tmp_path / "annotations.h5ad"
    arguments = ["--nuclei", str(nuclei), "--tissue", "Prostate", "--mpp", MPP]

    assert main(["annotate", str(tiles), *arguments, "--out", str(out)]) == 0
    result = read_annotations(out)
    assert len(result) == N_CELLS
    assert set(result.labels) <= set(LINEAGE)
    assert result.cells.cell_space_id == CELL_SPACE


def test_annotate_reads_cells_without_the_encoder(monkeypatch, tmp_path):
    tiles, nuclei = write_stack(tmp_path)
    cells = tmp_path / "cells.h5ad"
    code = main(["encode", str(tiles), "--nuclei", str(nuclei), "--out", str(cells), "--mpp", MPP])
    assert code == 0

    def no_weights(**kwargs: Any) -> AlhenaEncoder:
        raise BackboneNotFoundError("The UNI2-h weights were not found.")

    monkeypatch.setattr(alhena.api, "load_encoder", no_weights)
    out = tmp_path / "annotations.csv"
    assert main(["annotate", str(cells), "--tissue", "Prostate", "--out", str(out)]) == 0
    assert pd.read_csv(out)["cell_id"].tolist() == read_cells(cells).cell_id.tolist()


def test_annotate_refuses_tiles_without_nuclei(tmp_path, capsys):
    tiles, _ = write_stack(tmp_path)
    arguments = ["--tissue", "Prostate", "--out", str(tmp_path / "annotations.csv")]

    assert main(["annotate", str(tiles), *arguments]) == 1
    assert "--nuclei" in refusal(capsys)


def test_annotate_refuses_a_tissue_without_reference(tmp_path, capsys):
    tiles, nuclei = write_stack(tmp_path)
    out = tmp_path / "annotations.csv"
    arguments = ["--nuclei", str(nuclei), "--tissue", "Duodenum", "--out", str(out)]

    assert main(["annotate", str(tiles), *arguments]) == 1
    assert "Duodenum" in refusal(capsys)
    assert not out.exists()


# --- reference ---


def test_reference_list_names_every_tissue(capsys):
    assert main(["reference", "list"]) == 0
    out = capsys.readouterr().out
    for tissue, shard in TISSUE_TO_SHARD.items():
        assert tissue in out
        assert shard in out


def test_reference_check_scores_a_cached_shard(tmp_path, capsys):
    cache = tmp_path / "cache"
    write_cached_shard(cache, "prostate")
    arguments = ["Prostate", "--cache-dir", str(cache), "--max-cells", "600", "-q"]

    assert main(["reference", "check", *arguments]) == 0
    out = capsys.readouterr().out
    assert "prostate" in out
    assert "600" in out.split()
    # The classes of the shard are separable, so every cell scored is recovered.
    assert "1.0000" in out.split()


# --- download ---


def test_download_fetches_the_bundle_and_a_reference(monkeypatch, tmp_path, capsys):
    published = tmp_path / "published"
    published.mkdir()
    for name in BUNDLE_FILES:
        (published / name).write_bytes(b"weights")
    (published / "config.json").write_text("{}", encoding="utf-8")
    (published / "registry.json").write_text("{}", encoding="utf-8")
    write_cached_shard(published, "breast")
    fetched = install_hub(monkeypatch, published)
    cache = tmp_path / "cache"

    assert main(["download", "--bundle", "--tissue", "Breast", "--cache-dir", str(cache)]) == 0
    assert set(BUNDLE_FILES) <= set(fetched)
    assert f"{SHARD_DIRECTORY}/breast.safetensors" in fetched
    assert str(cache) in capsys.readouterr().out


def test_download_refuses_a_call_naming_nothing(monkeypatch, tmp_path, capsys):
    fetched = install_hub(monkeypatch, tmp_path)

    assert main(["download"]) == 1
    err = refusal(capsys)
    assert "--bundle" in err
    assert "--tissue" in err
    assert fetched == []


# --- info ---


def test_info_reports_the_cache_it_reads(monkeypatch, tmp_path, capsys):
    cache = tmp_path / "cache"
    write_cached_shard(cache, "breast")
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"")
    monkeypatch.setenv(CACHE_ENV_VAR, str(cache))

    assert main(["info", "--backbone", str(weights)]) == 0
    out = capsys.readouterr().out
    assert str(cache) in out
    assert "breast" in out
    assert str(weights) in out


# --- demo ---


def test_demo_forwards_its_arguments(monkeypatch):
    forwarded: list[list[str]] = []

    def demo_main(argv: list[str]) -> int:
        forwarded.append(list(argv))
        return 3

    monkeypatch.setattr(alhena.demo, "main", demo_main)
    assert main(["demo", "Breast", "--n-tiles", "2"]) == 3
    assert main(["demo", "--", "--list-tissues"]) == 3
    assert forwarded == [["Breast", "--n-tiles", "2"], ["--list-tissues"]]


# --- Progress ---


def test_encode_quiet_silences_the_progress_bar(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv(_progress.PROGRESS_ENV_VAR)
    # Bars are drawn from the first tile, as on a run long enough to show them.
    monkeypatch.setattr(_progress, "DELAY_S", 0.0)
    tiles, _ = write_stack(tmp_path)
    arguments = ["encode", str(tiles), "--out", str(tmp_path / "embeddings.npy"), "--mpp", MPP]

    assert main(arguments) == 0
    assert f"{N_TILES}/{N_TILES}" in capsys.readouterr().err
    assert main([*arguments, "-q"]) == 0
    assert capsys.readouterr().err == ""
