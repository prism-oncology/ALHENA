"""Tests for the figures of alhena.viz, read back from the artists they draw."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alhena import (
    CLASS_COLORS,
    REFERENCE_MPP,
    AlhenaValueError,
    CellTyper,
    MissingExtraError,
    ReferenceInfo,
    TileEmbeddings,
    confidence_sweep,
    load_reference,
    score,
)
from alhena.io import read_polygons, read_tiles
from alhena.viz import (
    MISSING_HATCH,
    class_color,
    class_palette,
    composition,
    confidence_curve,
    confusion,
    legend,
    missing_class_names,
    overlay,
    tile_regions,
)
from support import (
    CELL_SPACE,
    LINEAGE,
    NUCLEUS_SIDE,
    make_label_maps,
    make_polygons,
    make_result,
    make_shard,
    make_tiles,
    nucleus_corners,
    tiny_encoder,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

# Six cells with known classes, the fifth left without a prediction.
TRUTH = ["Myeloid", "Myeloid", "T_NK", "T_NK", "T_NK", "B_Plasma"]
PREDICTED = ["Myeloid", "T_NK", "T_NK", "T_NK", "", "B_Plasma"]
CONFIDENCE = [0.95, 0.3, 0.6, 0.95, 0.0, 0.4]

BOUNDS = np.array([[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]], dtype=np.float64)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close the figures a test draws."""
    yield
    plt.close("all")


def rgb(name: str) -> np.ndarray:
    """Return the color `CLASS_COLORS` gives a class, on the unit scale."""
    return np.asarray(CLASS_COLORS[name], dtype=np.float64) / 255.0


def drawn_images(axes: Any) -> list[np.ndarray]:
    """Return the arrays of the images an axes carries, in the order they were drawn."""
    return [np.asarray(image.get_array()) for image in axes.get_images()]


# --- Palette ----------------------------------------------------------------------------------


def test_class_palette_reads_the_default_palette():
    names = [*CLASS_COLORS, ""]
    expected = [rgb(name) for name in CLASS_COLORS] + [rgb("Unknown")]
    np.testing.assert_allclose(class_palette(names), expected)
    np.testing.assert_allclose(class_color("T_NK"), rgb("T_NK"))


def test_class_color_reads_a_custom_palette():
    colors = {"tumor": (10, 20, 30), "stroma": "#00ff00"}
    np.testing.assert_allclose(class_color("tumor", colors=colors), [10 / 255, 20 / 255, 30 / 255])
    np.testing.assert_allclose(class_color("stroma", colors=colors), [0.0, 1.0, 0.0])


def test_class_color_refuses_a_class_without_a_color():
    with pytest.raises(AlhenaValueError, match="No color is declared"):
        class_color("Hepatocyte")


def test_missing_class_names_reads_a_reference_info():
    support = dict.fromkeys(LINEAGE, 300) | {"Myeloid": 0, "T_NK": 0}
    info = ReferenceInfo(
        name="tonsil",
        axis="lineage_v1",
        cell_space_id=CELL_SPACE,
        n_cells=900,
        n_scored_cells=900,
        class_order=LINEAGE,
        class_support=support,
        source="atlas",
        license="CC-BY-4.0",
        scconcept_model="corpus40M-model30M",
        scconcept_version="v2",
    )
    assert missing_class_names(info, LINEAGE) == ("Myeloid", "T_NK")
    with pytest.raises(AlhenaValueError, match="single name"):
        missing_class_names("Myeloid", LINEAGE)


# --- Legend and composition -------------------------------------------------------------------


def test_legend_colors_each_class_and_hatches_missing_ones():
    handles = legend("lineage_v1", missing_classes=["Myeloid"]).get_legend().legend_handles
    # The last entry stands for the cells without an annotation, in the color of the unknown class.
    expected = [rgb(name) for name in LINEAGE] + [rgb("Unknown")]
    np.testing.assert_allclose([handle.get_facecolor()[:3] for handle in handles], expected)
    hatches = [handle.get_hatch() for handle in handles]
    assert hatches == [MISSING_HATCH if name == "Myeloid" else None for name in LINEAGE] + [None]


def test_composition_draws_fractions_and_marks_missing_classes():
    labels = ["T_NK", "T_NK", "Myeloid", "", "B_Plasma", "T_NK", "", "Myeloid"]
    bars = composition(make_result(labels), missing_classes=["Stromal_Vascular"]).patches
    expected = [labels.count(name) / len(labels) for name in (*LINEAGE, "")]
    assert [bar.get_height() for bar in bars] == pytest.approx(expected)
    hatched = [name for name, bar in zip((*LINEAGE, ""), bars, strict=True) if bar.get_hatch()]
    assert hatched == ["Stromal_Vascular"]


def test_composition_draws_a_table_of_cells_as_its_annotations(tmp_path):
    typer = CellTyper(tiny_encoder(), load_reference(path=make_shard(tmp_path)))
    result = typer.annotate(make_tiles(), make_label_maps(), mpp=REFERENCE_MPP)
    expected = [bar.get_height() for bar in composition(result).patches]
    drawn = [bar.get_height() for bar in composition(result.to_anndata()).patches]
    assert drawn == pytest.approx(expected)


# --- Evaluation -------------------------------------------------------------------------------


def test_confusion_draws_the_counts_normalized_by_row():
    axes = confusion(score(PREDICTED, TRUTH, LINEAGE))
    counts = np.zeros((len(LINEAGE), len(LINEAGE)))
    for known, predicted in zip(TRUTH, PREDICTED, strict=True):
        if predicted:
            counts[LINEAGE.index(known), LINEAGE.index(predicted)] += 1
    with np.errstate(invalid="ignore"):
        expected = counts / counts.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(np.ma.getdata(axes.get_images()[0].get_array()), expected)


def test_confidence_curve_draws_the_sweep():
    sweep = confidence_sweep(make_result(PREDICTED, CONFIDENCE), TRUTH, thresholds=(0.0, 0.5, 0.9))
    axes = confidence_curve(sweep)
    pooled, *recalls = axes.lines
    np.testing.assert_allclose(pooled.get_xdata(), sweep["threshold"])
    np.testing.assert_allclose(pooled.get_ydata(), sweep["macro_f1"])
    for line, name in zip(recalls, LINEAGE, strict=True):
        np.testing.assert_allclose(line.get_ydata(), sweep[f"recall_{name}"])
        np.testing.assert_allclose(line.get_color(), rgb(name))
    retained = next(other for other in axes.get_figure().axes if other is not axes)
    np.testing.assert_allclose(retained.lines[0].get_ydata(), sweep["retained_fraction"])


def test_confidence_curve_marks_the_classes_the_annotations_lack():
    result = make_result(PREDICTED, CONFIDENCE)
    result.meta["unexpressed_classes"] = ("Stromal_Vascular",)
    axes = confidence_curve(confidence_sweep(result, TRUTH, thresholds=(0.0, 0.5)))
    labels = [line.get_label() for line in axes.lines]
    assert [label for label in labels if "not in reference" in label] == [
        "Stromal_Vascular (not in reference)"
    ]


# --- Overlay ----------------------------------------------------------------------------------


def test_overlay_tints_each_nucleus_with_its_class_color():
    label_map = np.zeros((32, 32), dtype=np.int64)
    label_map[2:6, 2:6] = 7
    label_map[20:24, 20:24] = 9
    tile = np.zeros((32, 32, 3), dtype=np.uint8)
    axes = overlay(tile, label_map, make_result(["T_NK", ""], cell_id=[7, 9]), alpha=0.6)
    tint = drawn_images(axes)[1]
    np.testing.assert_allclose(tint[3, 3], [*rgb("T_NK"), 0.6], atol=1e-6)
    np.testing.assert_allclose(tint[21, 21], [*rgb("Unknown"), 0.6], atol=1e-6)
    assert tint[16, 16].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_overlay_draws_a_tile_of_a_source_with_its_outlines():
    polygons, bounds = make_polygons()
    cell_ids = list(range(1, len(polygons) + 1))
    corners = np.concatenate(polygons)
    owners = np.repeat(cell_ids, [len(outline) for outline in polygons])
    vertices = pd.DataFrame({"cell_id": owners, "x": corners[:, 0], "y": corners[:, 1]})
    source = read_tiles(make_tiles(), coords=bounds[:, :2].astype(np.int64))
    labels = [LINEAGE[cell % len(LINEAGE)] for cell in cell_ids]
    result = make_result(labels, cell_id=cell_ids)

    axes = overlay(source, read_polygons(vertices, tiles=source), result, tile_index=1)

    image, tint = drawn_images(axes)
    np.testing.assert_array_equal(image, make_tiles()[1])
    # The second tile holds the fifth to the eighth nucleus, each tinted at its center.
    half = NUCLEUS_SIDE // 2
    for (row, column), label in zip(nucleus_corners(), labels[4:8], strict=True):
        np.testing.assert_allclose(tint[row + half, column + half, :3], rgb(label), atol=1e-6)


# --- Tile regions -----------------------------------------------------------------------------


def test_tile_regions_fills_tiles_with_stored_cluster_colors():
    tiles = TileEmbeddings(
        embeddings=np.zeros((3, 4), dtype=np.float32),
        tile_index=np.arange(3),
        tile_ids=("a", "b", "c"),
        bounds=BOUNDS,
    )
    table = tiles.to_anndata()
    table.obs["tile_cluster"] = pd.Categorical(["1", "0", "1"])
    table.uns["tile_cluster_colors"] = ["#0000ff", "#ff0000"]

    regions = tile_regions(table, "tile_cluster", alpha=0.5).collections[0]

    red, blue = [1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5]
    np.testing.assert_allclose(regions.get_facecolors(), [red, blue, red])
    for path, (x0, y0, x1, y1) in zip(regions.get_paths(), BOUNDS, strict=True):
        assert path.vertices.min(axis=0).tolist() == [x0 - 0.5, y0 - 0.5]
        assert path.vertices.max(axis=0).tolist() == [x1 - 0.5, y1 - 0.5]


def test_tile_regions_draws_numbers_on_a_color_scale():
    regions = tile_regions(BOUNDS, [0.25, np.nan, 0.75]).collections[0]
    assert regions.get_array().tolist() == [0.25, 0.75]
    assert len(regions.get_paths()) == 2


# --- Optional dependency ----------------------------------------------------------------------


def test_composition_without_matplotlib_names_the_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(MissingExtraError, match=r"alhena\[viz\]"):
        composition(make_result(["T_NK"]))
