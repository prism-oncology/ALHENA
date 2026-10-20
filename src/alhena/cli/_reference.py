"""The reference commands: list, describe and check."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from alhena._commands import _fixed, _ignoring, _one_line, _share
from alhena._constants import CLASS_AXES, MIN_REFERENCE_CELLS_PER_CLASS, TISSUE_TO_SHARD
from alhena._errors import EnvelopeWarning, ReferenceCoverageWarning
from alhena._wording import PROGRAM
from alhena.cli._shared import (
    _CLASS_WIDTH,
    _cached_shards,
    _field,
    _note,
    _progress_of,
    _reported_warnings,
)

if TYPE_CHECKING:
    from alhena.types import ReferenceInfo


def _reference_list(args: argparse.Namespace) -> int:
    """List the tissues a reference is published for, their aliases, and the shards on disk."""
    from alhena._constants import TISSUE_ALIASES
    from alhena.hub import cache_root

    _field("tissue", "shard")
    for tissue in sorted(TISSUE_TO_SHARD):
        _field(f"  {tissue}", TISSUE_TO_SHARD[tissue])
    _field("tissue", "also accepted as")
    for tissue in sorted(TISSUE_TO_SHARD):
        aliases = sorted(alias for alias, named in TISSUE_ALIASES.items() if named == tissue)
        if aliases:
            _field(f"  {tissue}", ", ".join(aliases))
    _field("axes", ", ".join(sorted(CLASS_AXES)))

    shards = _cached_shards(Path(cache_root(args.cache_dir, create=False)))
    if shards:
        _field("cached", ", ".join(shards))
    else:
        _field("cached", "none")
        _note(f"Fetch one with `{PROGRAM} download --tissue Breast`.")
    return 0


def _print_reference_info(info: ReferenceInfo) -> None:
    """Report a shard's identity, provenance, class support and what it cannot express."""
    _field("shard", info.name)
    _field("axis", info.axis)
    _field("cell space id", info.cell_space_id)
    _field("cells", f"{info.n_scored_cells} labeled of {info.n_cells} total")
    _field("source", info.source)
    _field("license", info.license)
    _field("single cell model", f"{info.scconcept_model} {info.scconcept_version}")
    _field("support metadata", "verified" if info.coverage_verified else "not verified")
    _field("class support", "")
    pool = sum(int(info.class_support[name]) for name in info.class_order)
    _note(f"{'':<{_CLASS_WIDTH}}{'cells':>10}{'% of labeled reference cells':>30}")
    for name in info.class_order:
        count = int(info.class_support[name])
        _note(f"{name:<{_CLASS_WIDTH}}{count:>10}{_share(count, pool):>30}")

    missing = info.missing_classes
    if missing:
        _field("cannot express", ", ".join(missing))
        _note(
            "These classes cannot be predicted with this reference; their absence from results "
            "is not evidence of absence from the tissue."
        )
    else:
        _field("cannot express", "none")
    if info.thin_classes:
        _field("thinly supported", ", ".join(info.thin_classes))
        _note(
            f"Fewer than {MIN_REFERENCE_CELLS_PER_CLASS} labeled reference cells per listed "
            "class; interpret predictions with caution."
        )


def _reference_describe(args: argparse.Namespace) -> int:
    """Report what one shard is and what it can express, without reading its embeddings."""
    from alhena.reference import describe_shard

    with _reported_warnings():
        info = describe_shard(
            tissue=args.name, path=args.path, axis=args.axis, cache_dir=args.cache_dir
        )
    _print_reference_info(info)
    return 0


def _reference_check(args: argparse.Namespace) -> int:
    """Retrieve one shard against itself and report what it recovers."""
    from alhena.reference import load_reference

    progress = _progress_of(args)
    with _reported_warnings():
        # The report states what the reference lacks in its own notes, at the k of the check.
        with _ignoring(ReferenceCoverageWarning, EnvelopeWarning):
            reference = load_reference(
                tissue=args.name,
                path=args.path,
                axis=args.axis,
                cache_dir=args.cache_dir,
                progress=progress,
            )
        report = reference.self_check(
            k=args.k, tau=args.tau, max_cells=args.max_cells, seed=args.seed, progress=progress
        )

    _field("shard", report.reference)
    _field("axis", report.axis)
    _note(
        "Internal leave-one-out check on reference cells, not a benchmark of "
        "H&E annotation accuracy."
    )
    _field("retrieval", f"k={report.k}, tau={report.tau}")
    _field("cells scored", report.n_cells_evaluated)
    _field("macro F1", f"{report.macro_f1:.4f}")
    _field("accuracy", f"{report.accuracy:.4f}")
    _field("recall", "")
    for name, value in report.per_class_recall.items():
        _note(f"{name:<{_CLASS_WIDTH}}{_fixed(value, 10)}")
    _field(
        "similarity",
        ", ".join(f"{name} {value:.3f}" for name, value in report.similarity_quantiles.items()),
    )
    _note("Cosine-similarity quantiles for the retrieved reference neighbors.")
    for note in report.warnings:
        _note(_one_line(note))
    _note(
        "Failure signals a problem with the reference; passing does not establish "
        "performance on H&E-derived cells."
    )
    return 0
