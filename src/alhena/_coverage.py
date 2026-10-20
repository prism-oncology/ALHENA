"""What a reference cannot express, and whether `k` suits the cells of its classes."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

from alhena._constants import (
    MIN_K_TO_CLASSES_RATIO,
    MIN_REFERENCE_CELLS_PER_CLASS,
    MIN_SUPPORT_TO_K_RATIO,
)
from alhena._errors import PACKAGE_PREFIXES, EnvelopeWarning, ReferenceCoverageWarning

THIN_SUPPORT_WARNING: str = "The thinnest class the"
"""Opening of the warning that the thinnest class of a reference has too few cells for `k`."""


def _missing_classes(support: Mapping[str, int]) -> tuple[str, ...]:
    """Return the classes no reference cell carries, in class order."""
    return tuple(name for name, count in support.items() if count == 0)


def _thin_classes(support: Mapping[str, int]) -> tuple[str, ...]:
    """Return the classes too few reference cells carry for retrieval to be reliable."""
    return tuple(
        name for name, count in support.items() if 0 < count < MIN_REFERENCE_CELLS_PER_CLASS
    )


def _thinnest_class(support: Mapping[str, int], k: int, reference: str | None = None) -> str | None:
    """Return the sentence on the thinnest class a reference expresses, or None if `k` suits it."""
    present = {name: count for name, count in support.items() if count > 0}
    if not present or min(present.values()) / k >= MIN_SUPPORT_TO_K_RATIO:
        return None
    minimum = min(present.values())
    thinnest = next(name for name, count in present.items() if count == minimum)
    named = "reference" if reference is None else f"{reference} reference"
    return (
        f"{THIN_SUPPORT_WARNING} {named} expresses, {thinnest}, has {minimum} cells, fewer than "
        f"the {k * MIN_SUPPORT_TO_K_RATIO:g} recommended for k={k}. Interpret predictions of "
        f"{thinnest} with caution."
    )


def _warn_coverage(support: Mapping[str, int], k: int, axis: str, *, loading: bool) -> None:
    """Warn of missing and thinnest classes, thin ones at load, and k over classes at annotation."""
    missing = _missing_classes(support)
    if missing:
        warnings.warn(
            f"The reference holds no cell of {', '.join(missing)} on the {axis} axis, so these "
            "classes cannot be predicted with this reference. Their absence from the annotations "
            "does not show that they are absent from the tissue.",
            ReferenceCoverageWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
    thinly = _thin_classes(support) if loading else ()
    if thinly:
        counted = ", ".join(f"{name} ({support[name]})" for name in thinly)
        warnings.warn(
            f"The reference has fewer than {MIN_REFERENCE_CELLS_PER_CLASS} labeled cells for "
            f"{counted} on the {axis} axis. Interpret predictions for each listed class with "
            "caution.",
            ReferenceCoverageWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
    sentence = _thinnest_class(support, k)
    if sentence is not None:
        warnings.warn(sentence, EnvelopeWarning, skip_file_prefixes=PACKAGE_PREFIXES)
    if not loading and k / len(support) < MIN_K_TO_CLASSES_RATIO:
        warnings.warn(
            f"k={k} across {len(support)} classes is below the {MIN_K_TO_CLASSES_RATIO:g} "
            "reference neighbors per class recommended by ALHENA. Interpret confidence values "
            "with caution.",
            EnvelopeWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
