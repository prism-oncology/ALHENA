"""What the demo prints."""

from __future__ import annotations

import textwrap
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from alhena._commands import _fixed, _one_line, _share
from alhena._constants import INPUT_SIZE, UNKNOWN_LABEL, label_order
from alhena.demo._fetch import GEOMETRY_CAVEAT, PanNukeCohort
from alhena.demo._pannuke import (
    DEAD_CATEGORY,
    DEMO_AXIS,
    PANNUKE_LICENSE,
    PANNUKE_MPP,
    PANNUKE_REPO,
    PANNUKE_TO_COMPARTMENT,
    UNRESOLVED_BY_INFLAMMATORY,
    canvas_geometry,
)

if TYPE_CHECKING:
    import pandas as pd

    from alhena.types import AnnotationResult, ReferenceInfo, ScoreReport

_LICENSE_TERMS: Final[str] = (
    f"PanNuke is distributed under {PANNUKE_LICENSE}: non-commercial use only, attribution and "
    "share-alike."
)

_LICENSE_FETCHED: Final[str] = (
    "It is fetched here from its own repository and is redistributed by nothing in this package."
)

_LICENSE_LOCAL: Final[str] = "Nothing in this package redistributes it."

_LICENSE_NOTE: Final[str] = f"{_LICENSE_TERMS} {_LICENSE_FETCHED}"

_WIDTH: Final[int] = 80
_LABEL_WIDTH: Final[int] = 20
_CLASS_WIDTH: Final[int] = 22


def _heading(text: str) -> None:
    """Open a section of the report."""
    print(f"\n{text}")


def _wrapped(text: str, first: str, rest: str) -> list[str]:
    """Wrap text to the width of the report, never inside a word or a path."""
    return textwrap.wrap(
        text,
        width=_WIDTH,
        initial_indent=first,
        subsequent_indent=rest,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _note(text: str) -> None:
    """Print a paragraph of the section, wrapped to the width of the report."""
    for line in _wrapped(text, "  ", "  "):
        print(line)


def _field(name: str, value: object) -> None:
    """Print one labeled line, a long value wrapping under itself."""
    label = f"  {name:<{_LABEL_WIDTH - 2}}  "
    for line in _wrapped(str(value), label, " " * len(label)) or [label.rstrip()]:
        print(line)


def _row(text: str) -> None:
    """Print one line of a table as it is laid out."""
    print(f"  {text}".rstrip())


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """Return a count with its noun, singular for one."""
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _listing(names: Sequence[str]) -> str:
    """Join names as prose, the last two with 'and'."""
    items = [str(name) for name in names]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _classes(axis: str) -> tuple[str, ...]:
    """Return the classes of an axis, in the order a result holds them."""
    return tuple(name for name in label_order(axis) if name != UNKNOWN_LABEL)


def _fields_read(cohort: PanNukeCohort) -> str:
    """Say which fields were read: how many, of how many, and from where."""
    where = (
        f"of fold {cohort.fold}" if cohort.fold is not None else f"in {Path(cohort.source).name}"
    )
    return (
        f"the first {len(cohort.tiles)} of the {cohort.n_tissue_tiles} {cohort.tissue} fields "
        f"{where}, in file order"
    )


@contextmanager
def _collected_warnings(sink: list[str]) -> Iterator[None]:
    """Collect a block's warnings once each, in order, to print beside the numbers they qualify."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            for entry in caught:
                # A ResourceWarning says nothing about the numbers it would be printed beside.
                if issubclass(entry.category, ResourceWarning):
                    continue
                text = _one_line(entry.message)
                if text not in sink:
                    sink.append(text)


def _print_introduction(tissue: str, reference: str, n_tiles: int, axis: str) -> None:
    """Say what the demo does and what it does not measure, before anything is read."""
    print(f"ALHENA DEMO: PanNuke {tissue}")
    _heading("WHAT THIS DEMO DOES")
    _note(
        "ALHENA performs zero-shot cell-type annotation of H&E nuclei using similarity-weighted "
        "k-nearest neighbors (kNN) against a labeled single-cell RNA-seq reference."
    )
    print()
    _note(
        f"In this demo, ALHENA uses the {reference} reference to predict cell classes for nuclei "
        f"in {_plural(n_tiles, f'PanNuke {tissue} image field')}."
    )
    if axis != DEMO_AXIS:
        classes = _classes(axis)
        text = f"The axis {axis} has {len(classes)} classes: {_listing(classes)}."
        covered = UNRESOLVED_BY_INFLAMMATORY.get(axis)
        if covered:
            text += (
                f" Compared with {DEMO_AXIS}, it separates "
                f"{PANNUKE_TO_COMPARTMENT['Inflammatory']} into {_listing(covered)}."
            )
        print()
        _note(text)
    print()
    if axis == DEMO_AXIS:
        _note(
            "PanNuke labels do not determine the predicted classes. Nuclei labeled "
            f"{DEAD_CATEGORY} are excluded, and the remaining labels are used afterwards to "
            "evaluate agreement with ALHENA's predictions."
        )
    else:
        _note(
            "PanNuke labels do not determine the predicted classes. Nuclei labeled "
            f"{DEAD_CATEGORY} are excluded. No comparison with PanNuke is performed on this axis."
        )
    print()
    _note("No training or fine-tuning is performed during this demo.")

    _heading("IMPORTANT - NOT A PERFORMANCE BENCHMARK")
    _note(
        f"This demo uses only the first {_plural(n_tiles, f'{tissue} field')} in file order, not "
        "a random or representative sample."
    )
    print()
    if axis == DEMO_AXIS:
        _note(
            "The metrics describe this small sample only. They must not be interpreted as "
            f"ALHENA's overall performance on {tissue} tissue."
        )
    else:
        _note(
            f"This run demonstrates annotation on {axis}; it does not measure prediction accuracy."
        )


def _print_input(cohort: PanNukeCohort, shard: str, axis: str) -> None:
    """Report which fields were read, and the reference and axis they are annotated on."""
    first = cohort.tiles[0].image
    width, height = int(first.shape[1]), int(first.shape[0])
    _heading("INPUT")
    _field("PanNuke fields", _fields_read(cohort))
    _field(
        "Field size",
        f"{width} by {height} pixels at {PANNUKE_MPP} micron per pixel, "
        f"{width * PANNUKE_MPP:.0f} microns across",
    )
    _field("Reference", f"{cohort.reference_tissue} single-cell reference, shard {shard}")
    _field("Axis", f"{axis}, {_plural(len(_classes(axis)), 'class', 'classes')}")


@dataclass(frozen=True)
class NucleusCounts:
    """Where every annotated nucleus went. The five counts partition `annotated`.

    Attributes:
        annotated: Nuclei the release annotates on the fields read.
        dead: Nuclei left out as `Dead`.
        lost: Nuclei left with no pixel once overlaps were resolved.
        without_tokens: Nuclei that reached no token of the model canvas.
        unusable: Nuclei that reached tokens but got no prediction, their embedding unusable.
        typed: Nuclei that carry a prediction.
    """

    annotated: int
    dead: int
    lost: int
    without_tokens: int
    unusable: int
    typed: int

    @property
    def accounted(self) -> int:
        """The five counts summed, which equals `annotated`."""
        return self.dead + self.lost + self.without_tokens + self.unusable + self.typed

    @property
    def attempted(self) -> int:
        """Nuclei a prediction was attempted for, the ones an evaluation scores."""
        return self.unusable + self.typed


def _nucleus_counts(cohort: PanNukeCohort, result: AnnotationResult) -> NucleusCounts:
    """Count where every annotated nucleus went."""
    meta = dict(result.meta or {})
    valid = np.asarray(result.valid, dtype=bool)
    predicted = int(np.count_nonzero(valid))
    counts = NucleusCounts(
        annotated=cohort.n_nuclei + cohort.n_dead,
        dead=cohort.n_dead,
        lost=cohort.n_lost,
        without_tokens=int(meta.get("n_cells_without_tokens") or 0),
        unusable=int(valid.shape[0] - predicted),
        typed=predicted,
    )
    assert counts.accounted == counts.annotated, "the five counts partition the annotated nuclei"
    return counts


def _count_row(name: str, count: int, reason: str = "", depth: int = 0) -> None:
    """Print one line of the nucleus count, with what the count holds."""
    _row(f"{'  ' * depth + name:<29}{count:>6}   {reason}")


def _print_nuclei(counts: NucleusCounts, evaluated: bool) -> None:
    """Report where every annotated nucleus went, and which ones the evaluation counts."""
    _heading("NUCLEI")
    _count_row("Annotated in PanNuke", counts.annotated)
    _row("Excluded before prediction:")
    _count_row(DEAD_CATEGORY, counts.dead, f"labeled {DEAD_CATEGORY} in PanNuke", 1)
    _count_row("No pixels after overlap", counts.lost, "no pixels remain after overlaps", 1)
    _count_row(
        "Not captured after resizing",
        counts.without_tokens,
        "the model cannot represent the nucleus",
        1,
    )
    _count_row("Prediction attempts", counts.attempted)
    _count_row("Valid predictions", counts.typed, depth=1)
    _count_row("Failed predictions", counts.unusable, "no usable image embedding", 1)
    if evaluated:
        _count_row("Evaluated nuclei", counts.attempted, "valid + failed predictions")
        print()
        _note(
            "A failed prediction has no assigned cell class. It is nevertheless counted as an "
            "error when evaluating ALHENA against PanNuke labels."
        )


def _print_labels(cohort: PanNukeCohort, result: AnnotationResult) -> None:
    """Report the PanNuke labels of the evaluated nuclei, and the class each is compared with."""
    known = cohort.categories_by_cell()
    counts = dict.fromkeys(PANNUKE_TO_COMPARTMENT, 0)
    for cell_id in np.asarray(result.cell_id):
        counts[known[int(cell_id)]] += 1
    _heading("PANNUKE LABELS AND THE CLASSES THEY ARE COMPARED WITH")
    _row(f"{'PanNuke label':<16}{'Evaluated nuclei':>18}   Compared with")
    for name, count in counts.items():
        _row(f"{name:<16}{count:>18}   {PANNUKE_TO_COMPARTMENT[name]}")
    print()
    _note(
        "PanNuke distinguishes Neoplastic and Epithelial nuclei, but both are compared with "
        f"{PANNUKE_TO_COMPARTMENT['Neoplastic']} here."
    )
    print()
    _note(
        "This mapping assumes that Neoplastic nuclei are epithelial or melanocytic tumor cells, "
        "such as carcinoma or melanoma cells. ALHENA does not predict malignancy, so agreement "
        "on this class does not demonstrate tumor recognition."
    )


def _print_classes(report: ScoreReport) -> None:
    """Report, class by class, the PanNuke labels, the predictions and where the two agree."""
    _heading("PREDICTIONS AGAINST THE PANNUKE LABELS")
    _row(
        f"{'Class':<{_CLASS_WIDTH}}{'PanNuke':>9}{'Predicted':>11}{'Correct':>9}"
        f"{'Precision':>11}{'Recall':>8}{'F1':>7}"
    )
    predicted_total = 0
    for position, name in enumerate(report.class_order):
        row = report.per_class[name]
        predicted = int(row["predicted_support"])
        predicted_total += predicted
        _row(
            f"{name:<{_CLASS_WIDTH}}{int(row['support']):>9}{predicted:>11}"
            f"{int(report.confusion[position, position]):>9}{_fixed(row['precision'], 11)}"
            f"{_fixed(row['recall'], 8)}{_fixed(row['f1'], 7)}"
        )
    correct = int(np.trace(report.confusion))
    _row(f"{'All classes':<{_CLASS_WIDTH}}{report.n_cells:>9}{predicted_total:>11}{correct:>9}")
    print()
    _note("PanNuke counts the evaluated nuclei belonging to each class.")
    _note("Predicted counts nuclei assigned to that class by ALHENA.")
    _note("Correct counts nuclei where the prediction matches the PanNuke label.")
    print()
    _note("Precision is Correct / Predicted; recall is Correct / PanNuke.")
    _note("F1 combines precision and recall.")


def _print_agreement(report: ScoreReport) -> None:
    """Report ALHENA's agreement with the PanNuke labels beside the two baselines."""
    supports = {name: int(report.per_class[name]["support"]) for name in report.class_order}
    # The first of equally common classes, as the baseline breaks a tie.
    commonest = max(report.class_order, key=supports.__getitem__)
    correct = int(np.trace(report.confusion))
    _heading("AGREEMENT WITH THE PANNUKE LABELS")
    _row(f"{'':<30}{'Accuracy':>10}{'Macro F1':>11}")
    _row(f"{'ALHENA':<30}{_fixed(report.accuracy, 10, 4)}{_fixed(report.macro_f1, 11, 4)}")
    _row("Baselines, simple rules that are not ALHENA predictions:")
    _row(
        f"{'  Commonest-class baseline':<30}{_fixed(report.floor_accuracy, 10, 4)}"
        f"{_fixed(report.floor_macro_f1, 11, 4)}"
    )
    _row(
        f"{'  Frequency-matched baseline':<30}{'-':>10}"
        f"{_fixed(report.frequency_matched_macro_f1, 11, 4)}"
    )
    _field(
        "Balanced accuracy",
        f"{report.balanced_accuracy:.4f} for ALHENA, the mean of the per-class recalls",
    )
    print()
    _note(
        f"Accuracy is the share of evaluated nuclei predicted correctly: {correct} of "
        f"{report.n_cells} for ALHENA. Macro F1 averages the F1 of the classes, each weighing the "
        "same, so a rare class counts as much as a common one."
    )
    if report.n_present_classes < len(report.class_order):
        print()
        _note(
            f"Macro F1 averages over the {report.n_present_classes} classes present among the "
            "evaluated nuclei."
        )
    print()
    _note(
        f"The commonest-class baseline predicts {commonest}, the commonest class among these "
        "PanNuke labels, for every nucleus. The frequency-matched baseline predicts each class at "
        "its PanNuke frequency and is scored from the expected counts, not from random draws. "
        "Neither uses the images or the reference."
    )


def _print_predictions(result: AnnotationResult) -> None:
    """Report how many valid predictions each class received."""
    labels = np.asarray(result.labels)
    valid = int(np.count_nonzero(np.asarray(result.valid, dtype=bool)))
    _heading("PREDICTIONS")
    _row(f"{'Class':<{_CLASS_WIDTH}}{'Predicted':>11}{'% of valid predictions':>30}")
    for name in result.class_order:
        count = int(np.count_nonzero(labels == name))
        _row(f"{name:<{_CLASS_WIDTH}}{count:>11}{_share(count, valid):>30}")
    _row(f"{'All classes':<{_CLASS_WIDTH}}{valid:>11}{_share(valid, valid):>30}")


def _print_no_comparison(axis: str) -> None:
    """State why the predictions on this axis are not compared with the PanNuke labels."""
    _heading("NO COMPARISON WITH THE PANNUKE LABELS")
    covered = UNRESOLVED_BY_INFLAMMATORY.get(axis)
    if covered:
        _note(
            f"PanNuke has one Inflammatory label for {_listing(covered)} nuclei, so it cannot say "
            f"whether a prediction of one of them is right, and the predictions on {axis} are not "
            f"scored. Run with --axis {DEMO_AXIS}, which groups these {len(covered)} classes as "
            f"{PANNUKE_TO_COMPARTMENT['Inflammatory']}, to compare the predictions with PanNuke."
        )
    else:
        _note(
            f"The PanNuke labels do not match the classes of {axis}, so the predictions on it are "
            f"not scored. Run with --axis {DEMO_AXIS} to compare the predictions with PanNuke."
        )


def _print_reference(
    info: ReferenceInfo,
    result: AnnotationResult,
    shard_file: Path | None,
    notes: Sequence[str],
) -> None:
    """Report the reference the predictions come from, and how its cells vote."""
    meta = dict(result.meta or {})
    k = meta.get("k")
    pool = sum(int(count) for count in info.class_support.values())
    _heading("REFERENCE")
    _field("Source", f"{info.source}, shard {info.name}")
    if shard_file is not None:
        _note(f"A shard already on disk, read from {shard_file} instead of the cache.")
    _field("Cells", f"{pool:,} with a class of {result.axis}, of the {info.n_cells:,} it holds")
    print()
    _row(f"{'Class':<{_CLASS_WIDTH}}{'Reference cells':>17}{'% of labeled reference cells':>30}")
    for name in result.class_order:
        held = int(info.class_support.get(name, 0))
        _row(f"{name:<{_CLASS_WIDTH}}{held:>17,}{_share(held, pool):>30}")
    print()
    _note(
        "Reference class percentages are calculated over the labeled reference cells. "
        "They are not the PanNuke class percentages or the percentages among the "
        f"{k} neighbors selected for one nucleus."
    )
    print()
    _field("Neighbors", f"k = {k}, the reference cells considered per nucleus")
    _field(
        "Temperature",
        f"tau = {meta.get('tau')}, controls how strongly similarity influences the weighted vote",
    )
    effective = meta.get("effective_k")
    if effective is not None and np.isfinite(effective):
        _field(
            "Effective k",
            f"median {float(effective):.1f}, indicating how many neighbors effectively "
            "contribute to the weighted vote",
        )
    unexpressed = tuple(meta.get("unexpressed_classes") or ())
    if unexpressed:
        print()
        _note(
            f"The reference holds no cell of {_listing(unexpressed)}: no nucleus can be predicted "
            "as such."
        )
    thin = tuple(meta.get("thinly_supported_classes") or ())
    if thin:
        print()
        _note(
            f"The reference holds too few cells of {_listing(thin)} for predictions of them to be "
            "read as the others are."
        )
    for note in notes:
        print()
        _note(note)


def _print_confidence(result: AnnotationResult) -> None:
    """Report how confident the valid predictions are, over every one of them."""
    confidence = np.asarray(result.confidence, dtype=np.float64)
    scores = confidence[np.asarray(result.valid, dtype=bool)]
    assert np.isfinite(scores).all(), "every valid prediction carries a confidence"
    _heading("CONFIDENCE OF THE VALID PREDICTIONS")
    if scores.size == 0:
        _note("No nucleus received a valid prediction.")
        return
    _note(
        "Confidence is the fraction of the similarity-weighted kNN vote assigned to the predicted "
        "class."
    )
    print()
    _note(
        "A high confidence score means strong support from the reference neighbors. It does not "
        "mean the prediction is necessarily correct, and it is not a calibrated probability of "
        "correctness."
    )
    print()
    quantiles = np.percentile(scores, [0, 25, 50, 75, 100])
    _field(
        "Quantiles",
        "  ".join(
            f"{name} {value:.3f}"
            for name, value in zip(("min", "p25", "median", "p75", "max"), quantiles, strict=True)
        ),
    )
    print()
    _note(
        "Each percentage below is calculated over nuclei with a valid prediction, not over all "
        "nuclei annotated in PanNuke."
    )
    edges = np.linspace(1.0 / max(len(result.class_order), 1), 1.0, 6)
    # A float32 vote can land a rounding step outside [1/C, 1]: every score is still counted.
    counts, _ = np.histogram(np.clip(scores, edges[0], edges[-1]), bins=edges)
    assert int(counts.sum()) == scores.size, "the histogram counts every valid prediction"
    _row(f"{'Confidence':<16}{'Nuclei':>8}{'% of valid predictions':>30}")
    for position, count in enumerate(counts):
        span = f"{edges[position]:.2f} to {edges[position + 1]:.2f}"
        _row(f"{span:<16}{int(count):>8}{_share(int(count), scores.size):>30}")
    _row(f"{'All':<16}{scores.size:>8}{_share(scores.size, scores.size):>30}")


def _print_sweep(sweep: pd.DataFrame, class_order: Sequence[str]) -> None:
    """Report what each confidence threshold keeps, scored with the baselines of the same nuclei."""
    _heading("KEEPING ONLY CONFIDENT PREDICTIONS")
    _note(
        "This analysis shows what happens when only predictions meeting a minimum confidence "
        "threshold are kept."
    )
    print()
    _note(
        "A higher threshold retains fewer or the same number of nuclei. Accuracy and macro F1 are "
        "recalculated using only those retained."
    )
    print()
    _note(
        "The table shows the number retained, the fraction of evaluated nuclei retained, "
        "ALHENA's performance, and the two baselines recalculated on that same set."
    )
    print()
    _note("Nuclei without valid predictions are never retained, even at threshold 0.00.")
    print()
    _row(f"{'':<9}{'Nuclei kept':>18}{'ALHENA':>20}{'Baselines, macro F1':>31}")
    _row(
        f"{'Threshold':<9}{'Count':>8}{'Fraction':>10}{'Accuracy':>10}{'Macro F1':>10}"
        f"{'Classes':>9}{'Commonest':>11}{'Frequency':>11}"
    )
    for row in sweep.itertuples():
        _row(
            f"{row.threshold:>9.2f}{row.n_retained:>8}{_fixed(row.retained_fraction, 10)}"
            f"{_fixed(row.accuracy, 10)}{_fixed(row.macro_f1, 10)}{row.n_present_classes:>9}"
            f"{_fixed(row.floor_macro_f1, 11)}{_fixed(row.frequency_matched_macro_f1, 11)}"
        )
    print()
    _note("Classes is the number of classes included in the macro F1.")
    _note("Results across thresholds use different sets of nuclei.")
    _note("Changes in accuracy or F1 must be read alongside the fraction retained.")
    print()

    thresholds = [float(value) for value in sweep["threshold"]]
    _row(f"{'Recall by class':<{_CLASS_WIDTH}}" + "".join(f"{value:>8.2f}" for value in thresholds))
    for name in class_order:
        recalls = [float(value) for value in sweep[f"recall_{name}"]]
        _row(f"{name:<{_CLASS_WIDTH}}" + "".join(_fixed(value, 8) for value in recalls))
    print()
    _note("Recall for each class is calculated among its retained nuclei.")
    _note("A dash means no nucleus of that class was retained.")
    print()
    _note(
        "Increasing the confidence threshold does not necessarily improve recall for every "
        "class. Filtering removes both correct and incorrect predictions, and the effect "
        "depends on which cells are retained."
    )

    leaving: list[str] = []
    for name in class_order:
        supports = [int(value) for value in sweep[f"support_{name}"]]
        gone = [edge for support, edge in zip(supports, thresholds, strict=True) if support == 0]
        if gone and supports[0] > 0:
            leaving.append(f"{name} at {gone[0]:.2f}")
    if leaving:
        print()
        _note(
            f"Some classes lose every nucleus as the threshold rises: {'; '.join(leaving)}. From "
            "there, the macro F1 averages over fewer classes and both baselines rise, so a score "
            "that climbs gains from the smaller set of classes, not from better predictions."
        )


def _print_data(cohort: PanNukeCohort, notes: Sequence[str]) -> None:
    """Report where the fields come from, under what terms, and how the model sees them."""
    _heading("DATA AND GEOMETRY")
    published = cohort.fold is not None
    if published:
        _field("Dataset", f"{PANNUKE_REPO}, fold {cohort.fold}")
        _field("File", cohort.source)
    else:
        _field("Dataset", cohort.source)
        _note("A fold file already on disk, read instead of the published one.")
    _field("License", f"{_LICENSE_TERMS} {_LICENSE_FETCHED if published else _LICENSE_LOCAL}")
    side, covered = canvas_geometry(int(cohort.tiles[0].image.shape[1]))
    _field(
        "Model canvas",
        f"each field fills {side} by {side} pixels of the {INPUT_SIZE} by {INPUT_SIZE} the model "
        f"reads, {covered:.1%} of its area",
    )
    print()
    _note(
        f"PanNuke declares a resolution of {PANNUKE_MPP} micron per pixel. Each field fills "
        f"{covered:.1%} of the model input; the remaining {1 - covered:.1%} is white padding."
    )
    print()
    _note(
        "This changes the image context seen by the model and may affect predictions. The "
        "declared resolution is nominal rather than guaranteed native optical resolution."
    )
    for note in notes:
        # The paragraph above states what this warning details.
        if note != _one_line(GEOMETRY_CAVEAT):
            print()
            _note(note)
