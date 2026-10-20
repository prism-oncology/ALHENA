"""Tests for scoring annotations against known labels, with their baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alhena
from support import make_result

CLASSES = ("A", "B", "C")
TRUTH = np.array(["A"] * 120 + ["B"] * 100 + ["C"] * 10)
PREDICTED = np.array(["A"] * 100 + ["B"] * 20 + ["A"] * 20 + ["B"] * 80 + ["A"] * 4 + ["C"] * 6)
# Rows are the known classes, columns the predicted ones.
CONFUSION = np.array([[100, 20, 0], [20, 80, 0], [4, 0, 6]])

PER_CLASS = {
    "A": {"support": 120, "predicted_support": 124, "precision": 100 / 124, "recall": 100 / 120},
    "B": {"support": 100, "predicted_support": 100, "precision": 80 / 100, "recall": 80 / 100},
    "C": {"support": 10, "predicted_support": 6, "precision": 6 / 6, "recall": 6 / 10},
}
F1 = {"A": 2 * 100 / (120 + 124), "B": 2 * 80 / (100 + 100), "C": 2 * 6 / (10 + 6)}

# 20 A right and confident, 6 C right but unsure, 4 C confidently predicted A, 1 unannotated.
SWEEP_TRUTH = np.array(["A"] * 20 + ["C"] * 10 + ["A"])
SWEEP_LABELS = np.array(["A"] * 20 + ["C"] * 6 + ["A"] * 4 + [""])
SWEEP_CONFIDENCE = np.array([0.9] * 20 + [0.4] * 6 + [0.9] * 4 + [np.nan])

KNOWN = pd.DataFrame({"cell_id": [2, 3, 1], "label": ["C", "A", "B"]})


def known_labels(form: str, directory: Path) -> object:
    """Return the labels of `KNOWN` in one of the forms `align_truth` reads."""
    if form == "table":
        return KNOWN
    if form == "series":
        return KNOWN.set_index("cell_id")["label"]
    if form == "mapping":
        return dict(zip(KNOWN["cell_id"].tolist(), KNOWN["label"].tolist(), strict=True))
    path = directory / f"known.{form}"
    if form == "csv":
        KNOWN.to_csv(path, index=False)
    else:
        KNOWN.to_parquet(path)
    return path


def test_score_builds_confusion_matrix():
    report = alhena.score(PREDICTED, TRUTH, CLASSES)
    np.testing.assert_array_equal(report.confusion, CONFUSION)
    assert report.class_order == CLASSES
    assert report.n_cells == 230


def test_score_per_class_metrics_match_hand_values():
    report = alhena.score(PREDICTED, TRUTH, CLASSES)
    for name, expected in PER_CLASS.items():
        metrics = report.per_class[name]
        assert {field: metrics[field] for field in expected} == pytest.approx(expected)
        assert metrics["f1"] == pytest.approx(F1[name])


@pytest.mark.parametrize(
    ("macro_min_support", "in_macro"), [(0, ("A", "B", "C")), (50, ("A", "B"))]
)
def test_score_accuracy_and_macro_f1_match_hand_values(
    macro_min_support: int, in_macro: tuple[str, ...]
):
    report = alhena.score(PREDICTED, TRUTH, CLASSES, macro_min_support=macro_min_support)
    assert report.accuracy == pytest.approx(186 / 230)
    assert report.n_present_classes == len(in_macro)
    assert report.macro_f1 == pytest.approx(np.mean([F1[name] for name in in_macro]))
    recalls = [PER_CLASS[name]["recall"] for name in in_macro]
    assert report.balanced_accuracy == pytest.approx(np.mean(recalls))


def test_score_counts_unannotated_cell_as_error():
    predicted = PREDICTED.copy()
    predicted[-6:] = ""
    report = alhena.score(predicted, TRUTH, CLASSES)
    assert report.n_cells == 230
    assert int(report.confusion.sum()) == 224
    assert report.accuracy == pytest.approx(180 / 230)
    assert report.per_class["C"]["recall"] == 0.0


def test_score_reads_labels_and_classes_of_annotations():
    result = make_result(PREDICTED, class_order=CLASSES, axis="letters")
    report = alhena.score(result, TRUTH)
    assert report.class_order == CLASSES
    np.testing.assert_array_equal(report.confusion, CONFUSION)


def test_score_rejects_label_outside_class_order():
    with pytest.raises(alhena.AlhenaValueError, match="not classes of the axis"):
        alhena.score(np.array(["A", "B"]), np.array(["A", "Unknown"]), CLASSES)


@pytest.mark.parametrize(("macro_min_support", "n_in_macro"), [(0, 3), (50, 2)])
def test_chance_floor_commonest_class_matches_hand_values(macro_min_support: int, n_in_macro: int):
    floor = alhena.chance_floor(TRUTH, CLASSES, macro_min_support=macro_min_support)
    assert floor.per_class["A"]["predicted_support"] == 230
    assert floor.accuracy == pytest.approx(120 / 230)
    # Every cell predicted A gives A an F1 of 2 * 120 / (120 + 230), and every other class zero.
    assert floor.macro_f1 == pytest.approx(240 / 350 / n_in_macro)

    report = alhena.score(PREDICTED, TRUTH, CLASSES, macro_min_support=macro_min_support)
    assert report.floor_accuracy == pytest.approx(floor.accuracy)
    assert report.floor_macro_f1 == pytest.approx(floor.macro_f1)


@pytest.mark.parametrize(("macro_min_support", "baseline"), [(0, 1 / 3), (50, 110 / 230)])
def test_chance_floor_frequency_matched_matches_hand_values(
    macro_min_support: int, baseline: float
):
    # Drawing each class at its share of the cells gives it an F1 equal to that share.
    floor = alhena.chance_floor(TRUTH, CLASSES, macro_min_support=macro_min_support)
    report = alhena.score(PREDICTED, TRUTH, CLASSES, macro_min_support=macro_min_support)
    assert floor.frequency_matched_macro_f1 == pytest.approx(baseline)
    assert report.frequency_matched_macro_f1 == pytest.approx(baseline)


def test_confidence_sweep_counts_cells_kept_per_threshold():
    result = make_result(SWEEP_LABELS, SWEEP_CONFIDENCE, class_order=CLASSES, axis="letters")
    frame = alhena.confidence_sweep(result, SWEEP_TRUTH, thresholds=(0.0, 0.5))
    assert frame["threshold"].tolist() == [0.0, 0.5]
    assert frame["n_cells"].tolist() == [31, 31]
    assert frame["n_retained"].tolist() == [30, 24]
    assert frame["retained_fraction"].tolist() == pytest.approx([30 / 31, 24 / 31])
    assert frame["support_C"].tolist() == [10, 4]
    assert frame["recall_C"].tolist() == pytest.approx([0.6, 0.0])


def test_confidence_sweep_scores_cells_kept_at_threshold():
    result = make_result(SWEEP_LABELS, SWEEP_CONFIDENCE, class_order=CLASSES, axis="letters")
    row = alhena.confidence_sweep(result, SWEEP_TRUTH, thresholds=(0.5,)).iloc[0]
    # 24 cells kept, all predicted A: 20 A and 4 C, so C scores an F1 of zero.
    assert row["accuracy"] == pytest.approx(20 / 24)
    assert row["macro_f1"] == pytest.approx((2 * 20 / (20 + 24) + 0.0) / 2)
    assert row["floor_accuracy"] == pytest.approx(20 / 24)


@pytest.mark.parametrize("form", ["table", "series", "mapping", "csv", "parquet"])
def test_align_truth_orders_known_labels_as_cells(form: str, tmp_path: Path):
    result = make_result(["A", "A", "A"], cell_id=[3, 1, 2], class_order=CLASSES, axis="letters")
    aligned = alhena.align_truth(result, known_labels(form, tmp_path))
    assert aligned.tolist() == ["A", "B", "C"]
    assert alhena.score(result, aligned).accuracy == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("known", "match"),
    [
        pytest.param({3: "A", 1: "B"}, "1 of the 3 cells", id="cell without label"),
        pytest.param({"3": "A", "1": "B", "2": "C"}, "3 of the 3 cells", id="identifiers as text"),
    ],
)
def test_align_truth_rejects_cell_without_label(known: dict[object, str], match: str):
    result = make_result(["A", "A", "A"], cell_id=[3, 1, 2], class_order=CLASSES, axis="letters")
    with pytest.raises(alhena.AlhenaValueError, match=match):
        alhena.align_truth(result, known)
