"""What the commands and the worked example share: their common options, and how they print."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def _add_cache_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="directory downloaded artifacts are cached in, overriding ALHENA_CACHE",
    )


def _add_quiet_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help=(
            "report no progress on standard error, download bars included; the results are "
            "printed as usual"
        ),
    )


def _add_backbone_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backbone",
        type=Path,
        default=None,
        help=(
            "the UNI2-h weight file, or a directory holding one; ALHENA_BACKBONE, then the "
            "Hugging Face cache, by default"
        ),
    )


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    _add_backbone_option(parser)
    parser.add_argument(
        "--bundle",
        default=None,
        help=(
            "directory holding the bundle files, or a repository owner/name; the published one "
            "by default"
        ),
    )
    _add_cache_option(parser)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "device to run on: cpu, cuda, or mps for the GPU of an Apple silicon Mac; CUDA "
            "where available, the processor otherwise"
        ),
    )


def _one_line(text: object) -> str:
    """Collapse a message onto one line, so a report never breaks in the middle of a field."""
    return " ".join(str(text).split())


def _share(count: int, total: int) -> str:
    """Return a count as a share of a total, or a dash where the total is zero."""
    return f"{count / total:.1%}" if total else "-"


def _fixed(value: float, width: int, decimals: int = 3) -> str:
    """Return a right-aligned number, or a dash where it is undefined."""
    return f"{value:>{width}.{decimals}f}" if np.isfinite(value) else f"{'-':>{width}}"


@contextmanager
def _ignoring(*categories: type[Warning]) -> Iterator[None]:
    """Leave the warnings of the given categories a block raises unreported."""
    with warnings.catch_warnings():
        for category in categories:
            warnings.simplefilter("ignore", category)
        yield
