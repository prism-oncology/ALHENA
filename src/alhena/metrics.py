"""Score cell-type predictions against known labels.

Every report carries two baselines of the same cells: the commonest-class baseline and the
frequency-matched baseline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Final, TypeAlias

import numpy as np

from alhena._errors import AlhenaValueError
from alhena._truth import TRUTH_COLUMNS as TRUTH_COLUMNS
from alhena._truth import _aligned_truth, _known_labels
from alhena.types import AnnotationResult, ScoreReport

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Any

    import pandas as pd

LabelInput: TypeAlias = "Sequence[str] | Sequence[int] | np.ndarray"
"""Labels as class names or as codes indexing the class order."""

KnownLabels: TypeAlias = "pd.DataFrame | pd.Series | Mapping[Any, Any] | str | Path"
"""Known labels by cell identifier: a table, a Series, a mapping, or the path of a table."""

NO_CALL_LABEL: Final[str] = ""
"""Label a cell without a prediction carries."""

DEFAULT_CONFIDENCE_THRESHOLDS: Final[tuple[float, ...]] = (
    0.0,
    *(round(0.30 + 0.05 * step, 2) for step in range(14)),
)
"""Default grid of `confidence_sweep`; its first entry retains every annotated cell."""

_TRUTH_ADVICE: Final[str] = (
    "Collapse the labels onto the axis and drop the cells carrying no class of it before "
    "scoring, so the scored population is the one the axis defines."
)
_PREDICTION_ADVICE: Final[str] = (
    "A prediction is a class of the axis, or the empty string where no annotation was assigned."
)


def _validate_class_order(class_order: Sequence[str]) -> tuple[str, ...]:
    order = tuple(class_order)
    if not order:
        raise AlhenaValueError(
            "No classes were given. Pass the class order of the axis being scored."
        )
    if any(not isinstance(name, str) for name in order):
        raise AlhenaValueError(f"Class order must hold class names, got {order!r}.")
    if len(set(order)) != len(order):
        raise AlhenaValueError(f"Class order holds a repeated class name: {order!r}.")
    return order


def _validate_macro_min_support(macro_min_support: int) -> int:
    try:
        value = int(macro_min_support)
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(
            f"macro_min_support must be a whole number of cells, got {macro_min_support!r}."
        ) from error
    if value != macro_min_support or value < 0:
        raise AlhenaValueError(
            "macro_min_support must be a whole number of cells, zero or more, got "
            f"{macro_min_support!r}."
        )
    return value


def _check_length(array: np.ndarray, expected: int, what: str) -> None:
    if array.shape[0] != expected:
        raise AlhenaValueError(
            f"There are {array.shape[0]} {what} for {expected} cells. Every array describing the "
            "same cells must be aligned cell by cell."
        )


def _encode_labels(values: LabelInput, class_order: tuple[str, ...], *, role: str) -> np.ndarray:
    """Map class names or class codes onto codes, with -1 for a prediction that was not made."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise AlhenaValueError(
            f"Expected a one-dimensional array of {role} labels, got shape {array.shape}."
        )

    if array.dtype.kind in "iu":
        codes = array.astype(np.int64)
        lowest = -1 if role == "prediction" else 0
        out_of_range = codes[(codes < lowest) | (codes >= len(class_order))]
        if out_of_range.size:
            raise AlhenaValueError(
                f"{role.capitalize()} class codes index the class order, so they lie in "
                f"[{lowest}, {len(class_order)}), got {sorted(set(out_of_range.tolist()))[:5]}."
            )
        return codes

    index = {name: position for position, name in enumerate(class_order)}
    codes = np.full(array.shape[0], -1, dtype=np.int64)
    unknown: set[str] = set()
    for position, value in enumerate(array):
        name = str(value)
        code = index.get(name, -1)
        codes[position] = code
        if code < 0 and not (role == "prediction" and name == NO_CALL_LABEL):
            unknown.add(name)
    if unknown:
        sample = ", ".join(repr(name) for name in sorted(unknown)[:5])
        advice = _TRUTH_ADVICE if role == "truth" else _PREDICTION_ADVICE
        raise AlhenaValueError(
            f"{len(unknown)} {role} label(s) are not classes of the axis "
            f"{list(class_order)}: {sample}. {advice}"
        )
    return codes


def _majority_prediction(truth_ids: np.ndarray, n_classes: int) -> np.ndarray:
    """Give every cell the commonest class of the population, ties going to the lowest code."""
    if truth_ids.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    counts = np.bincount(truth_ids, minlength=n_classes)
    return np.full(truth_ids.shape[0], int(np.argmax(counts)), dtype=np.int64)


def _score_ids(
    truth_ids: np.ndarray,
    predicted_ids: np.ndarray,
    class_order: tuple[str, ...],
    macro_min_support: int,
    floor_metrics: tuple[float, float] | None,
) -> ScoreReport:
    """Score encoded labels; `floor_metrics` is None when the report is the commonest-class one."""
    n_classes = len(class_order)
    n_cells = int(truth_ids.shape[0])

    called = predicted_ids >= 0
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    # Buffered fancy indexing would keep one write per repeated (truth, prediction) pair.
    np.add.at(confusion, (truth_ids[called], predicted_ids[called]), 1)

    support = np.bincount(truth_ids, minlength=n_classes).astype(np.int64)
    predicted_support = np.bincount(predicted_ids[called], minlength=n_classes).astype(np.int64)
    true_positive = np.diag(confusion).astype(np.float64)

    # Precision of a class never predicted, and recall of an absent class, are undefined, not zero.
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.full(n_classes, np.nan),
        where=predicted_support > 0,
    )
    recall = np.divide(true_positive, support, out=np.full(n_classes, np.nan), where=support > 0)
    # 2*tp + fp + fn, written so a class that was never predicted scores zero rather than nan.
    f1_denominator = support + predicted_support
    f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.full(n_classes, np.nan),
        where=f1_denominator > 0,
    )

    # The one place the support floor acts; every class kept has a cell, so no nan reaches a mean.
    in_macro = support >= max(1, macro_min_support)
    n_present_classes = int(np.count_nonzero(in_macro))

    accuracy = float(true_positive.sum() / n_cells) if n_cells else float("nan")
    if n_present_classes:
        macro_f1 = float(np.mean(f1[in_macro]))
        balanced_accuracy = float(np.mean(recall[in_macro]))
        # Drawing class c at its share p expects p*n_c hits of p*n guesses for n_c cells: F1 = p.
        frequency_matched_macro_f1 = float(np.mean(support[in_macro] / n_cells))
    else:
        macro_f1 = float("nan")
        balanced_accuracy = float("nan")
        frequency_matched_macro_f1 = float("nan")

    per_class = {
        name: {
            "support": int(support[position]),
            "predicted_support": int(predicted_support[position]),
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(f1[position]),
            "in_macro": float(in_macro[position]),
        }
        for position, name in enumerate(class_order)
    }

    floor_macro_f1, floor_accuracy = (
        (macro_f1, accuracy) if floor_metrics is None else floor_metrics
    )
    return ScoreReport(
        macro_f1=macro_f1,
        balanced_accuracy=balanced_accuracy,
        accuracy=accuracy,
        n_cells=n_cells,
        n_present_classes=n_present_classes,
        class_order=class_order,
        per_class=per_class,
        confusion=confusion,
        floor_macro_f1=floor_macro_f1,
        floor_accuracy=floor_accuracy,
        macro_min_support=macro_min_support,
        frequency_matched_macro_f1=frequency_matched_macro_f1,
    )


def _chance_floor_from_ids(
    truth_ids: np.ndarray, class_order: tuple[str, ...], macro_min_support: int
) -> ScoreReport:
    return _score_ids(
        truth_ids,
        _majority_prediction(truth_ids, len(class_order)),
        class_order,
        macro_min_support,
        floor_metrics=None,
    )


def _report_from_ids(
    truth_ids: np.ndarray,
    predicted_ids: np.ndarray,
    class_order: tuple[str, ...],
    macro_min_support: int,
) -> ScoreReport:
    """Score a population and its commonest-class baseline under one support floor."""
    floor = _chance_floor_from_ids(truth_ids, class_order, macro_min_support)
    return _score_ids(
        truth_ids,
        predicted_ids,
        class_order,
        macro_min_support,
        floor_metrics=(floor.macro_f1, floor.accuracy),
    )


def align_truth(result: AnnotationResult, known: KnownLabels) -> np.ndarray:
    """Put known labels in the order of the annotated cells, matching them by cell identifier.

    Identifiers are matched as they are written, never converted, so an integer and its text do not
    meet. Every annotated cell needs a known label. A table holds `TRUTH_COLUMNS`, a Series is
    indexed by identifier, and a mapping goes from identifier to label.

    Args:
        result: Annotations whose cells order the labels.
        known: The known labels: a table, a Series, a mapping, or the path of a `.csv` or
            `.parquet` table.

    Returns:
        One known label per annotated cell, in the order of the result, as `score` and
        `confidence_sweep` take them.

    Raises:
        AlhenaValueError: If the labels cannot be read, or do not name each cell once.
    """
    identifiers, labels, named = _known_labels(known)
    return _aligned_truth(np.asarray(result.cell_id), identifiers, labels, named)


def score(
    predicted: AnnotationResult | LabelInput,
    truth: LabelInput,
    class_order: Sequence[str] | None = None,
    macro_min_support: int = 0,
) -> ScoreReport:
    """Score predicted cell types against known ones.

    Cells without a prediction stay in the population as errors and enter no column of the confusion
    matrix. The macro average runs over the classes the population carries, less those under
    `macro_min_support`, which moves the denominator and removes no cell. Both baselines are
    scored on the same cells under the same floor.

    Args:
        predicted: The annotations, which carry their class order; or predicted class names or
            codes indexing `class_order`, `""` or -1 marking a cell without a prediction.
        truth: Known class names, or codes indexing `class_order`, one per cell, as `align_truth`
            orders them. Every entry must be a class of the axis.
        class_order: Classes of the axis, in the order that fixes their codes and the rows and
            columns of the confusion matrix. Read from annotations when None.
        macro_min_support: Cells a class needs in this population to enter the macro average;
            zero keeps every class it carries.

    Returns:
        The agreement between the two label sets, with the commonest-class and frequency-matched
        baselines of the same population.

    Raises:
        AlhenaValueError: If the class order or the labels are malformed or do not match.
    """
    missing: tuple[str, ...] = ()
    if isinstance(predicted, AnnotationResult):
        given = class_order
        class_order = tuple(predicted.class_order)
        if given is not None and tuple(given) != class_order:
            raise AlhenaValueError(
                f"class_order {tuple(given)!r} is not the class order of the annotations, "
                f"{class_order!r}. Leave class_order out: the annotations carry theirs."
            )
        missing = predicted.missing_classes
        predicted = predicted.labels
    elif class_order is None:
        raise AlhenaValueError(
            "Labels carry no class order. Pass class_order, or the AnnotationResult itself, "
            "which carries it."
        )
    order = _validate_class_order(class_order)
    support_floor = _validate_macro_min_support(macro_min_support)
    truth_ids = _encode_labels(truth, order, role="truth")
    predicted_ids = _encode_labels(predicted, order, role="prediction")
    _check_length(predicted_ids, truth_ids.shape[0], "predicted labels")
    report = _report_from_ids(truth_ids, predicted_ids, order, support_floor)
    return replace(report, missing_classes=missing) if missing else report


def chance_floor(
    truth: LabelInput,
    class_order: Sequence[str],
    macro_min_support: int = 0,
) -> ScoreReport:
    """Score the commonest-class baseline, which gives every cell the commonest class.

    Read against a headline from `score` on the same cells, it takes the same `macro_min_support`.
    Under a macro average it scores zero on every class but the one it predicts.

    Args:
        truth: Known class names, or codes indexing `class_order`.
        class_order: Classes of the axis, in the order that fixes their codes. A tie between two
            equally common classes resolves to the lower code.
        macro_min_support: Cells a class must carry in this population to enter the macro average,
            matching the value the headline was scored with.

    Returns:
        The baseline's own report, whose `floor_macro_f1` and `floor_accuracy` are its scores.

    Raises:
        AlhenaValueError: If the class order is malformed, or a truth label is not of the axis.
    """
    order = _validate_class_order(class_order)
    support_floor = _validate_macro_min_support(macro_min_support)
    truth_ids = _encode_labels(truth, order, role="truth")
    return _chance_floor_from_ids(truth_ids, order, support_floor)


def confidence_sweep(
    result: AnnotationResult,
    truth: LabelInput,
    thresholds: Iterable[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
) -> pd.DataFrame:
    """Score the annotations that survive each of a grid of confidence thresholds.

    A cell is retained when it is annotated and its confidence reaches the threshold; each row is
    `score` on the retained cells, pooled, so no support floor applies. A class whose errors are
    confident loses recall as the threshold rises.

    Args:
        result: Annotations to sweep, supplying the class order, the labels, the confidence and the
            per-cell validity.
        truth: Known class names, or codes indexing the result's class order, one per cell.
        thresholds: Confidence thresholds, reported in the order given.

    Returns:
        One row per threshold: the retained count and fraction, accuracy, macro F1, balanced
        accuracy, the realized macro denominator, both baselines of the retained cells, and the
        support and recall of every class. `attrs["missing_classes"]` names the classes the
        reference held no cell of, as the annotations record them.

    Raises:
        AlhenaValueError: If the grid is empty or not finite, or the labels do not match the cells.
    """
    import pandas as pd

    grid = [float(threshold) for threshold in thresholds]
    if not grid:
        raise AlhenaValueError("No thresholds were given. Pass at least one confidence threshold.")
    if not all(np.isfinite(threshold) for threshold in grid):
        raise AlhenaValueError(f"Confidence thresholds must be finite, got {grid}.")

    order = _validate_class_order(result.class_order)
    truth_ids = _encode_labels(truth, order, role="truth")
    predicted_ids = _encode_labels(result.labels, order, role="prediction")
    n_cells = int(truth_ids.shape[0])
    _check_length(predicted_ids, n_cells, "predicted labels")

    confidence = np.asarray(result.confidence, dtype=np.float64).reshape(-1)
    valid = np.asarray(result.valid, dtype=bool).reshape(-1)
    _check_length(confidence, n_cells, "confidence values")
    _check_length(valid, n_cells, "validity flags")

    # An invalid cell is retained at no threshold, yet counts in the population of the fraction.
    called = valid & np.isfinite(confidence)

    rows: list[dict[str, float]] = []
    for threshold in grid:
        keep = called & (confidence >= threshold)
        n_retained = int(np.count_nonzero(keep))
        report = _report_from_ids(truth_ids[keep], predicted_ids[keep], order, 0)
        row: dict[str, float] = {
            "threshold": threshold,
            "n_cells": n_cells,
            "n_retained": n_retained,
            "retained_fraction": (n_retained / n_cells) if n_cells else float("nan"),
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "balanced_accuracy": report.balanced_accuracy,
            "n_present_classes": report.n_present_classes,
            "floor_accuracy": report.floor_accuracy,
            "floor_macro_f1": report.floor_macro_f1,
            "frequency_matched_macro_f1": report.frequency_matched_macro_f1,
        }
        for name in order:
            row[f"support_{name}"] = report.per_class[name]["support"]
            row[f"recall_{name}"] = report.per_class[name]["recall"]
        rows.append(row)
    sweep = pd.DataFrame(rows)
    sweep.attrs["missing_classes"] = result.missing_classes
    return sweep
