"""The annotate command."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from alhena._commands import _fixed, _ignoring, _share
from alhena._constants import MIN_REFERENCE_CELLS_PER_CLASS
from alhena._defaults import _batch_size
from alhena._errors import (
    PACKAGE_PREFIXES,
    AlhenaError,
    AlhenaValueError,
    EnvelopeWarning,
    GeometryWarning,
)
from alhena._truth import TRUTH_COLUMNS as TRUTH_COLUMNS
from alhena._truth import _aligned_truth, _read_known_table
from alhena._wording import PROGRAM
from alhena.cli._reading import _NUCLEI_READING, _open_tiles, _read_nuclei, _require_suffix
from alhena.cli._shared import (
    _CLASS_WIDTH,
    _UNDECLARED_MPP,
    CALL_SUFFIXES,
    CELLS_SUFFIXES,
    _failure,
    _field,
    _file_size,
    _given,
    _no_token_note,
    _note,
    _print_resolution,
    _progress_of,
    _reported,
    _reported_warnings,
    _write_calls,
)

if TYPE_CHECKING:
    import pandas as pd

    from alhena.api import CellTyper
    from alhena.io import TileSource
    from alhena.types import AnnotationResult, ScoreReport


# Options annotate refuses beside a table of cells, which is annotated by retrieval alone.
_TILE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--nuclei", "nuclei"),
    ("--mpp", "mpp"),
    ("--tile-px", "tile_px"),
    ("--stride-px", "stride_px"),
    ("--edge", "edge"),
    ("--coords", "coords"),
    ("--coords-columns", "coords_columns"),
    ("--coords-scale", "coords_scale"),
    ("--region", "region"),
    ("--h5-datasets", "h5_datasets"),
    *_NUCLEI_READING,
    ("--batch-size", "batch_size"),
    ("--backbone", "backbone"),
    ("--bundle", "bundle"),
    ("--device", "device"),
)


def _align_truth(result: AnnotationResult, truth: pd.DataFrame, path: Path) -> np.ndarray:
    """Order the known labels as the result, matching identifiers as written, never converted."""
    return _aligned_truth(
        np.asarray(result.cell_id),
        truth["cell_id"].to_numpy(),
        truth["label"].to_numpy(),
        path.name,
        advice="Score against a table covering every annotated cell.",
    )


def _retrieval(meta: dict[str, Any]) -> str:
    """Return the retrieval operating point, and the median effective number of neighbors."""
    text = f"k={meta.get('k')}, tau={meta.get('tau')}"
    effective = meta.get("effective_k")
    if effective is not None and np.isfinite(effective):
        text += f"; median effective k {float(effective):.1f}"
    return text


def _print_composition(result: AnnotationResult, reference: str) -> None:
    """Report how many cells were annotated, with which classes, and what the reference holds."""
    labels = np.asarray(result.labels)
    valid = np.asarray(result.valid, dtype=bool)
    total = int(labels.shape[0])
    called = int(np.count_nonzero(valid))
    meta = dict(result.meta or {})
    support = dict(meta.get("reference_class_support") or {})
    pool = sum(int(count) for count in support.values())

    _field("reference", meta.get("reference", reference))
    _field("axis", result.axis)
    _field("retrieval", _retrieval(meta))
    _note("k: reference neighbors each cell votes over; tau: vote softmax temperature.")
    effective = meta.get("effective_k")
    if effective is not None and np.isfinite(effective):
        _note("effective k: how many of those neighbors effectively contribute to the vote.")
    _field("cells annotated", f"{called} of {total}; {total - called} without a valid prediction")
    without_tokens = meta.get("n_cells_without_tokens")
    if without_tokens:
        _note(_no_token_note(int(without_tokens)))
    if meta.get("membership_approximate"):
        _note("Nucleus shapes are approximated, so the pooling weights are approximate.")

    _note(f"{'':<{_CLASS_WIDTH}}{'cells':>10}{'% annotated':>14}{'% reference':>14}")
    for name in result.class_order:
        count = int(np.count_nonzero(labels == name))
        held = _share(int(support.get(name, 0)), pool)
        _note(f"{name:<{_CLASS_WIDTH}}{count:>10}{_share(count, called):>14}{held:>14}")
    _note(
        "Percentages use valid annotations and labeled reference cells as their respective totals."
    )

    unexpressed = tuple(meta.get("unexpressed_classes") or ())
    if unexpressed:
        _note(
            f"The reference contains no cells of {', '.join(unexpressed)}; these classes "
            "cannot be predicted with this reference."
        )
    thin = tuple(meta.get("thinly_supported_classes") or ())
    if thin:
        _note(
            f"Thinly supported (<{MIN_REFERENCE_CELLS_PER_CLASS} labeled reference cells "
            f"each): {', '.join(thin)}. Interpret predictions with caution."
        )


def _print_class_table(report: ScoreReport) -> None:
    """Report the support, recall and F1 of every class of the axis."""
    for name in report.class_order:
        row = report.per_class[name]
        _note(
            f"{name:<{_CLASS_WIDTH}}{int(row['support']):>10}"
            f"  recall {_fixed(row['recall'], 5)}  F1 {_fixed(row['f1'], 5)}"
        )


def _print_sweep(sweep: pd.DataFrame, class_order: Sequence[str]) -> None:
    """Report the predictions each confidence threshold retains, and whose recall falls with it."""
    print("results by minimum confidence")
    _note(
        "Only valid predictions at or above each threshold are scored; "
        "the evaluated cell set changes between rows."
    )
    _note("% kept is the share of all cells, including cells without a valid prediction.")
    _note(
        "Confidence is a weighted reference-vote share, not a calibrated "
        "probability of correctness."
    )
    _note(f"{'threshold':>9}{'cells kept':>12}{'% kept':>10}{'accuracy':>10}{'macro F1':>10}")
    for row in sweep.itertuples():
        fraction = f"{row.retained_fraction:.1%}" if np.isfinite(row.retained_fraction) else "-"
        _note(
            f"{row.threshold:>9.2f}{row.n_retained:>12}{fraction:>10}"
            f"{_fixed(row.accuracy, 10)}{_fixed(row.macro_f1, 10)}"
        )
    first = sweep.iloc[0]
    last = sweep.iloc[-1]
    falling = []
    for name in class_order:
        low = float(first[f"recall_{name}"])
        high = float(last[f"recall_{name}"])
        if np.isfinite(low) and np.isfinite(high) and high < low:
            falling.append(f"{name} {low:.3f} to {high:.3f}")
    if falling:
        _note(f"Recall falls as the threshold rises for {'; '.join(falling)}.")
        _note(
            "For each listed class, the excluded predictions were more often correct "
            "than those retained."
        )


def _print_score(result: AnnotationResult, truth: np.ndarray, path: Path) -> None:
    """Report the agreement with known labels, its two baselines, and the abstention curve."""
    from alhena.metrics import confidence_sweep, score

    report = score(result, truth)
    macro = f"{report.macro_f1:.4f}"
    print(f"comparison with known labels: {path.name}")
    _note("Cells without a valid prediction count as errors in the overall scores below.")
    _field("  macro F1", f"{macro}  (commonest-class baseline {report.floor_macro_f1:.4f})")
    _field(
        "",
        f"{'':<{len(macro)}}  (frequency-matched baseline {report.frequency_matched_macro_f1:.4f})",
    )
    _field("  balanced accuracy", f"{report.balanced_accuracy:.4f}")
    _field(
        "  accuracy",
        f"{report.accuracy:.4f}  (commonest-class baseline {report.floor_accuracy:.4f})",
    )
    _field("  macro F1 over", f"{report.n_present_classes} of {len(report.class_order)} classes")
    _print_class_table(report)
    _print_sweep(confidence_sweep(result, truth), report.class_order)


def _cell_typer(args: argparse.Namespace, progress: bool | None, **encoder: Any) -> CellTyper:
    """Load the reference the command names, with the encoder options given."""
    from alhena.api import load_celltyper

    # The load warns of the thinnest class at the default k; the run warns at its own.
    with _ignoring(EnvelopeWarning):
        return load_celltyper(
            tissue=args.tissue,
            reference=args.reference,
            axis=args.axis,
            cache_dir=args.cache_dir,
            progress=progress,
            **encoder,
        )


def _annotate(args: argparse.Namespace) -> int:
    """Annotate the nuclei of a set of tiles, or a table of cells, and write the annotations out."""
    from alhena._shards import _require_declared_axis, shard_for_tissue
    from alhena.io import read_cells
    from alhena.preprocess import UNDECLARED_RESOLUTION_WARNING

    progress = _progress_of(args)
    _require_suffix(args.out, CALL_SUFFIXES, "A table of annotations is written")
    # A misspelled tissue, or an axis no prebuilt reference has, is refused before any read.
    if args.tissue is not None:
        shard_for_tissue(args.tissue)
        _require_declared_axis(args.axis)
    stored = args.tiles.suffix.lower() in CELLS_SUFFIXES
    if stored:
        given = _given(args, _TILE_OPTIONS)
        if given:
            one = len(given) == 1
            raise AlhenaValueError(
                f"{', '.join(given)} {'describes' if one else 'describe'} tiles and the encoder "
                f"that reads them, and a {CELLS_SUFFIXES[0]} file is read as a table of cells "
                "encoded already, annotated by retrieval alone. Drop "
                f"{'it' if one else 'them'}, or pass the tiles the nuclei were segmented in."
            )
    elif args.nuclei is None:
        raise AlhenaValueError(
            "--nuclei is required to annotate tiles. A table of cells written by "
            f"`{PROGRAM} encode --nuclei` is annotated without it."
        )
    else:
        from alhena.encoder._inference import _resolve_device

        # Before any read, so a batch size or a device out of range costs no reading.
        if args.batch_size is not None:
            _batch_size(args.batch_size)
        if args.device is not None:
            _resolve_device(args.device)
    source: TileSource | None = None
    with _reported_warnings(collapse=(UNDECLARED_RESOLUTION_WARNING,)):
        if stored:
            cells = read_cells(args.tiles)
            truth = None if args.truth is None else _read_known_table(args.truth)
            # Cells encoded already need no encoder, so no UNI2-h weights and no bundle.
            typer = _cell_typer(args, progress)
            result = typer.annotate(
                cells, k=args.k, tau=args.tau, progress=progress, workers=args.workers
            )
        else:
            with _open_tiles(args) as source:
                if source.mpp is None:
                    warnings.warn(
                        _UNDECLARED_MPP, GeometryWarning, skip_file_prefixes=PACKAGE_PREFIXES
                    )
                nuclei = _read_nuclei(args, source, progress)
                # Read before the encoder is built, so a bad table costs a read, not a forward pass.
                truth = None if args.truth is None else _read_known_table(args.truth)
                typer = _cell_typer(
                    args, progress, backbone=args.backbone, bundle=args.bundle, device=args.device
                )
                result = typer.annotate(
                    source,
                    nuclei,
                    mpp=source.mpp,
                    k=args.k,
                    tau=args.tau,
                    batch_size=args.batch_size,
                    progress=progress,
                    workers=args.workers,
                )
    _write_calls(result, args.out)

    # A table of cells records its own resolution, so only tiles report one here.
    if source is not None:
        _print_resolution(source)
    _print_composition(result, args.tissue or str(args.reference))
    _field("written", f"{args.out} ({_file_size(args.out)})")
    if truth is not None:
        # The annotations are on disk already: a table that cannot be aligned ends only the scoring.
        try:
            _print_score(result, _align_truth(result, truth, args.truth), args.truth)
        except AlhenaError as error:
            _failure(
                _reported(error),
                f"The annotations were written to {args.out} and are complete. Only the scoring "
                f"against {args.truth.name} was skipped.",
            )
            return 1
    return 0
