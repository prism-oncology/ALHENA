"""A demo run: fetching, annotating and scoring."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from alhena import _progress
from alhena._commands import _ignoring
from alhena._constants import CLASS_AXES, DEFAULT_K, DEFAULT_TAU
from alhena._errors import AlhenaValueError, BundleError, EnvelopeWarning
from alhena._hub import _repo_id
from alhena.demo._fetch import (
    DEFAULT_FOLD,
    DEFAULT_N_TILES,
    DEFAULT_TISSUE,
    _fold_file,
    fetch_pannuke,
)
from alhena.demo._pannuke import (
    DEMO_AXIS,
    PANNUKE_MPP,
    PANNUKE_REVISION,
    pannuke_tissue,
    reference_tissue,
)
from alhena.demo._report import (
    _collected_warnings,
    _nucleus_counts,
    _plural,
    _print_agreement,
    _print_classes,
    _print_confidence,
    _print_data,
    _print_input,
    _print_introduction,
    _print_labels,
    _print_no_comparison,
    _print_nuclei,
    _print_predictions,
    _print_reference,
    _print_sweep,
)

DEMO_THRESHOLDS: Final[tuple[float, ...]] = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""Confidence thresholds the abstention curve is reported at."""


def _shard_on_disk(path: str | Path, shard: str, tissue: str, target: str, axis: str) -> Path:
    """Return a shard on disk once its header names it the tissue's, before an embedding is read."""
    from alhena.reference import describe_shard

    file = Path(path)
    found = describe_shard(path=file, axis=axis).name
    if found != shard:
        raise AlhenaValueError(
            f"The shard at {file} is the {found!r} reference, and the PanNuke {tissue} fields "
            f"are annotated against {shard!r}, the reference published for {target}. Pass that "
            "shard, or none to read the published one; a custom reference is used with alhena "
            "annotate or alhena.load_celltyper."
        )
    return file


def run(
    tissue: str = DEFAULT_TISSUE,
    n_tiles: int = DEFAULT_N_TILES,
    axis: str = DEMO_AXIS,
    fold: int = DEFAULT_FOLD,
    parquet: str | Path | None = None,
    revision: str | None = PANNUKE_REVISION,
    k: int = DEFAULT_K,
    tau: float = DEFAULT_TAU,
    bundle: str | Path | None = None,
    backbone: str | Path | None = None,
    reference: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device: str | None = None,
    progress: bool | None = None,
) -> None:
    """Fetch a few PanNuke fields, annotate their nuclei, and report the run on standard output.

    Given `parquet`, `bundle` as a directory, `reference` and `backbone`, the run reads every file
    from disk and reaches no network.

    Args:
        tissue: PanNuke tissue to read.
        n_tiles: How many fields of that tissue to read.
        axis: Class axis the predictions are made on.
        fold: Fold to read, ignored when `parquet` names a file already on disk.
        parquet: A fold file already on disk, read instead of the published one.
        revision: Branch, tag or commit of the dataset repository; None reads its default branch.
        k: Reference neighbors each nucleus votes over.
        tau: Softmax temperature over the neighbor similarities.
        bundle: Directory holding the bundle files, or a repository `owner/name`. Defaults to the
            published repository.
        backbone: The UNI2-h weight file, or a directory holding one.
        reference: The reference file published for the tissue, read from disk instead of the cache.
        cache_dir: Cache the bundle and the reference are fetched into.
        device: Device the encoder runs on.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Raises:
        AlhenaError: If an argument is out of range, the tissue has no reference, a reference file
            is absent or not the tissue's, or the fold, the encoder or the reference cannot be read.
    """
    # Checked before any import or fetch, so a bad argument costs a lookup, not a download.
    if axis not in CLASS_AXES:
        raise AlhenaValueError(f"Unknown class axis {axis!r}. Declared axes: {sorted(CLASS_AXES)}.")
    if int(k) < 1:
        raise AlhenaValueError(f"k must be at least 1, got {k}.")
    if float(tau) <= 0.0:
        raise AlhenaValueError(f"tau must be positive, got {tau}.")
    if int(n_tiles) < 1:
        raise AlhenaValueError(f"n_tiles must be at least 1, got {n_tiles}.")
    name = pannuke_tissue(tissue)
    target = reference_tissue(name)

    from alhena.encoder._inference import _resolve_device
    from alhena.hub import find_backbone
    from alhena.reference import shard_for_tissue

    shard = shard_for_tissue(target)
    shard_file = None
    if reference is not None:
        shard_file = _shard_on_disk(reference, shard, name, target, axis)
    _fold_file(fold, parquet)
    if bundle is not None and not Path(bundle).is_dir():
        _repo_id(bundle, BundleError)
    if device is not None:
        _resolve_device(device)
    # Found before PanNuke is read, so weights that are missing cost no read and no download.
    weights = find_backbone(backbone)

    _print_introduction(name, target, n_tiles, axis)
    fetch_notes: list[str] = []
    _progress.step(f"Reading {_plural(n_tiles, 'PanNuke field')} of {name}", progress)
    with _collected_warnings(fetch_notes):
        cohort = fetch_pannuke(
            tissue=tissue, n_tiles=n_tiles, fold=fold, parquet=parquet, revision=revision
        )
    _print_input(cohort, shard, axis)
    # Shown before the long stages, even where standard output is a pipe.
    sys.stdout.flush()

    from alhena.api import load_celltyper

    call_notes: list[str] = []
    with _collected_warnings(call_notes):
        # The load warns of the thinnest class at the default k; the run warns at its own.
        with _ignoring(EnvelopeWarning):
            # The loader takes a tissue or the path of a shard and refuses the two together.
            typer = load_celltyper(
                tissue=target if shard_file is None else None,
                reference=shard_file,
                axis=axis,
                bundle=bundle,
                backbone=weights,
                device=device,
                cache_dir=cache_dir,
                progress=progress,
            )
        result = typer.annotate(
            cohort.images, cohort.label_maps, mpp=PANNUKE_MPP, k=k, tau=tau, progress=progress
        )

    counts = _nucleus_counts(cohort, result)
    report = sweep = None
    if axis == DEMO_AXIS:
        from alhena.metrics import align_truth, confidence_sweep, score

        truth = align_truth(result, cohort.truth_by_cell())
        report = score(result, truth)
        sweep = confidence_sweep(result, truth, thresholds=DEMO_THRESHOLDS)

    _print_nuclei(counts, evaluated=report is not None)
    if report is None:
        _print_predictions(result)
        _print_no_comparison(axis)
    else:
        _print_labels(cohort, result)
        _print_classes(report)
        _print_agreement(report)
    _print_reference(typer.reference.info, result, shard_file, call_notes)
    _print_confidence(result)
    if report is not None and sweep is not None:
        _print_sweep(sweep, report.class_order)
    _print_data(cohort, fetch_notes)
