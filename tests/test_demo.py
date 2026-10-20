"""Tests for the worked example, alhena.demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import alhena.api
import alhena.hub
from alhena import (
    COMPARTMENT_V1,
    INPUT_SIZE,
    REFERENCE_MPP,
    TISSUE_TO_SHARD,
    AlhenaValueError,
    CellTyper,
    GeometryWarning,
    load_reference,
    prepare_tiles,
)
from alhena.demo import (
    CATEGORIES_COLUMN,
    DEAD_CATEGORY,
    DEFAULT_FOLD,
    DEMO_AXIS,
    HUGGINGFACE_METADATA_KEY,
    IMAGE_COLUMN,
    INSTANCES_COLUMN,
    PANNUKE_CATEGORIES,
    PANNUKE_FOLD_FILES,
    PANNUKE_MPP,
    PANNUKE_REPO,
    PANNUKE_REVISION,
    PANNUKE_TILE_PX,
    PANNUKE_TISSUES,
    PANNUKE_TO_REFERENCE_TISSUE,
    TISSUE_COLUMN,
    canvas_geometry,
    compartment_for_category,
    decode_image,
    decode_mask,
    fetch_pannuke,
    main,
    pannuke_tissue,
    reference_tissue,
    run,
)
from support import CELLS_PER_CLASS, class_codes, make_shard, tiny_encoder

# A Hugging Face image column: the encoded bytes of a file beside its path.
IMAGE_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
FOLD_SCHEMA = pa.schema(
    [
        (IMAGE_COLUMN, IMAGE_STRUCT),
        (INSTANCES_COLUMN, pa.list_(IMAGE_STRUCT)),
        (CATEGORIES_COLUMN, pa.list_(pa.int64())),
        (TISSUE_COLUMN, pa.int64()),
    ]
)


def encode_png(array: np.ndarray) -> bytes:
    """Encode an array as the release stores its images."""
    written, buffer = cv2.imencode(".png", array)
    assert written
    return buffer.tobytes()


def field_image() -> np.ndarray:
    """Return a visual field of random RGB pixels."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(PANNUKE_TILE_PX, PANNUKE_TILE_PX, 3), dtype=np.uint8)


def block_mask(position: int) -> np.ndarray:
    """Return a nucleus large enough to reach tokens once the field is resampled."""
    mask = np.zeros((PANNUKE_TILE_PX, PANNUKE_TILE_PX), dtype=np.uint8)
    top = 8 + 32 * position
    mask[top : top + 24, 8:40] = 255
    return mask


def make_row(tissue: str, categories: list[str]) -> dict[str, Any]:
    """Build one row of a fold: a field, one mask per nucleus and the category of each."""
    return {
        # OpenCV encodes from BGR, so the channels are reversed to store the field as RGB.
        IMAGE_COLUMN: {"bytes": encode_png(field_image()[:, :, ::-1]), "path": "field.png"},
        INSTANCES_COLUMN: [
            {"bytes": encode_png(block_mask(position)), "path": f"mask_{position}.png"}
            for position in range(len(categories))
        ],
        CATEGORIES_COLUMN: [PANNUKE_CATEGORIES.index(name) for name in categories],
        TISSUE_COLUMN: PANNUKE_TISSUES.index(tissue),
    }


def write_fold(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows as a fold file declaring its class names, as the release does."""
    categories = {"_type": "ClassLabel", "names": list(PANNUKE_CATEGORIES)}
    features = {
        TISSUE_COLUMN: {"_type": "ClassLabel", "names": list(PANNUKE_TISSUES)},
        CATEGORIES_COLUMN: {"_type": "Sequence", "feature": categories},
    }
    declared = json.dumps({"info": {"features": features}}).encode("utf-8")
    schema = FOLD_SCHEMA.with_metadata({HUGGINGFACE_METADATA_KEY: declared})
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


@pytest.fixture
def report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> str:
    """Run the example offline over two Breast fields on disk, and return what it printed."""
    rows = [
        make_row("Breast", ["Neoplastic", DEAD_CATEGORY, "Inflammatory"]),
        make_row("Breast", ["Connective", "Epithelial"]),
    ]
    fold = write_fold(tmp_path / "fold.parquet", rows)
    codes = class_codes((CELLS_PER_CLASS,) * len(COMPARTMENT_V1))
    shard = make_shard(tmp_path, "breast", codes=codes, axis=DEMO_AXIS)

    def load_celltyper(**kwargs: Any) -> CellTyper:
        return CellTyper(tiny_encoder(), load_reference(path=shard, axis=kwargs["axis"]))

    monkeypatch.setattr(alhena.hub, "find_backbone", lambda backbone=None: tmp_path / "UNI2-h.bin")
    monkeypatch.setattr(alhena.api, "load_celltyper", load_celltyper)
    run(tissue="Breast", n_tiles=2, parquet=fold, progress=False)
    return capsys.readouterr().out


# --- PanNuke ---


def test_decode_image_and_mask_read_a_pannuke_row():
    row = make_row("Breast", ["Neoplastic"])

    np.testing.assert_array_equal(decode_image(row[IMAGE_COLUMN]["bytes"]), field_image())
    mask = decode_mask(row[INSTANCES_COLUMN][0]["bytes"])
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, block_mask(0) > 0)
    with pytest.raises(AlhenaValueError):
        decode_image(b"not an image")


def test_pannuke_tissue_maps_onto_a_published_reference():
    assert pannuke_tissue("head-neck") == "Head & Neck"
    assert reference_tissue("ovarian") == "Ovary"
    assert reference_tissue("PANCREATIC") == "Pancreas"
    assert set(PANNUKE_TO_REFERENCE_TISSUE.values()) <= set(TISSUE_TO_SHARD)


def test_reference_tissue_refuses_a_tissue_without_reference():
    with pytest.raises(AlhenaValueError, match="No reference is published"):
        reference_tissue("Head & Neck")
    with pytest.raises(AlhenaValueError, match="names no PanNuke tissue"):
        reference_tissue("Tonsil")


def test_compartment_for_category_maps_every_category_but_dead():
    assert compartment_for_category("Neoplastic") == "Epithelial_Melanocyte"
    assert compartment_for_category("epithelial") == "Epithelial_Melanocyte"
    assert compartment_for_category("Inflammatory") == "Immune"
    assert compartment_for_category("Connective") == "Stromal_Vascular"
    with pytest.raises(AlhenaValueError):
        compartment_for_category(DEAD_CATEGORY)


def test_canvas_geometry_matches_preprocessing():
    side, covered = canvas_geometry()
    field = np.zeros((PANNUKE_TILE_PX, PANNUKE_TILE_PX, 3), dtype=np.uint8)
    _, info = prepare_tiles([field], mpp=PANNUKE_MPP)

    assert side == round(PANNUKE_TILE_PX * PANNUKE_MPP / REFERENCE_MPP)
    assert info.resampled_size == (side, side)
    assert covered == pytest.approx(side**2 / INPUT_SIZE**2)


# --- fetch_pannuke ---


def test_fetch_pannuke_reads_the_published_fold(monkeypatch, tmp_path):
    rows = [
        make_row("Breast", ["Neoplastic", "Inflammatory", DEAD_CATEGORY]),
        make_row("Colon", ["Connective"]),
        make_row("Breast", ["Connective", "Epithelial"]),
        make_row("Breast", ["Inflammatory"]),
    ]
    fold = write_fold(tmp_path / "fold.parquet", rows)
    opened: list[str] = []

    class Hub:
        def open(self, path: str, mode: str = "rb") -> Any:
            opened.append(path)
            return fold.open("rb")

    monkeypatch.setattr("huggingface_hub.HfFileSystem", Hub)
    with pytest.warns(GeometryWarning):
        cohort = fetch_pannuke(tissue="breast", n_tiles=2)

    published = f"{PANNUKE_REPO}@{PANNUKE_REVISION}/{PANNUKE_FOLD_FILES[DEFAULT_FOLD]}"
    assert opened == [f"datasets/{published}"]
    assert (cohort.tissue, cohort.fold, cohort.n_tissue_tiles) == ("Breast", DEFAULT_FOLD, 3)
    assert len(cohort.tiles) == 2
    assert (cohort.n_nuclei, cohort.n_dead) == (4, 1)
    truth = sorted(cohort.truth_by_cell().values())
    assert truth == ["Epithelial_Melanocyte", "Epithelial_Melanocyte", "Immune", "Stromal_Vascular"]
    categories = sorted(cohort.categories_by_cell().values())
    assert categories == ["Connective", "Epithelial", "Inflammatory", "Neoplastic"]


# --- run and main ---


def test_run_reports_the_evaluated_nuclei(report):
    evaluated = next(line for line in report.splitlines() if "Evaluated nuclei" in line)
    # Five nuclei, the Dead one left out.
    assert "4" in evaluated.split()


def test_run_prints_its_report_in_plain_ascii(report):
    assert report.isascii()


def test_main_returns_an_exit_code(capsys):
    assert main(["--list-tissues"]) == 0
    assert main(["Head & Neck"]) == 1
    assert "Head & Neck" in capsys.readouterr().err
    assert main(["--n-tiles", "many"]) == 2
