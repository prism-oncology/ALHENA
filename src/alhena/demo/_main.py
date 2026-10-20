"""The demo's command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from alhena import _progress
from alhena._commands import _add_model_options, _add_quiet_option, _one_line
from alhena._constants import CLASS_AXES, DEFAULT_K, DEFAULT_TAU
from alhena._errors import AlhenaError
from alhena._wording import PROGRAM, _as_command_line
from alhena.demo._fetch import DEFAULT_FOLD, DEFAULT_N_TILES, DEFAULT_TISSUE
from alhena.demo._pannuke import (
    DEMO_AXIS,
    PANNUKE_FOLD_FILES,
    PANNUKE_REVISION,
    PANNUKE_TISSUES,
    PANNUKE_TO_REFERENCE_TISSUE,
)
from alhena.demo._report import _LICENSE_NOTE, _field, _heading, _note
from alhena.demo._run import run


def _list_tissues() -> int:
    """Report which PanNuke tissues carry a published reference and which do not."""
    _heading("PANNUKE TISSUES CARRYING A PUBLISHED REFERENCE")
    for name in PANNUKE_TISSUES:
        target = PANNUKE_TO_REFERENCE_TISSUE.get(name)
        if target is not None:
            _field(name, target)
    _heading("PANNUKE TISSUES CARRYING NONE")
    for name in PANNUKE_TISSUES:
        if name not in PANNUKE_TO_REFERENCE_TISSUE:
            _field(name, "no reference")
    _note(
        "A tissue with no reference is not annotated against the reference of a neighboring organ. "
        "A reference expresses the classes of the organ it was built from and no others."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the worked example."""
    parser = argparse.ArgumentParser(
        prog=f"{PROGRAM} demo",
        description=(
            "Predict the cell class of the nuclei of a few PanNuke fields from the reference "
            "published for their tissue, and compare the predictions with the PanNuke labels."
        ),
        epilog=_LICENSE_NOTE,
    )
    parser.add_argument(
        "tissue",
        nargs="?",
        default=DEFAULT_TISSUE,
        help=f"PanNuke tissue to read; one of {list(PANNUKE_TO_REFERENCE_TISSUE)}",
    )
    parser.add_argument(
        "--list-tissues",
        action="store_true",
        help="report which PanNuke tissues carry a published reference, and stop",
    )
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=DEFAULT_N_TILES,
        help="how many fields of that tissue to read, the first ones in file order",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=DEFAULT_FOLD,
        choices=sorted(PANNUKE_FOLD_FILES),
        help="PanNuke fold to read; it selects nothing when --parquet is given",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help=("a fold file already on disk, read instead of the published one"),
    )
    parser.add_argument(
        "--revision",
        default=PANNUKE_REVISION,
        help="branch, tag or commit of the dataset repository",
    )
    parser.add_argument(
        "--axis",
        default=DEMO_AXIS,
        choices=sorted(CLASS_AXES),
        help=(
            f"class axis the predictions are made on; only {DEMO_AXIS} is compared with the "
            "PanNuke labels"
        ),
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="reference neighbors each nucleus votes over"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
        help="softmax temperature over the neighbor similarities",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="the reference file published for the tissue, read from disk instead of the cache",
    )
    _add_model_options(parser)
    _add_quiet_option(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the worked example and return its exit code.

    A package error is reported in one line on standard error, without a traceback.

    Args:
        argv: Arguments to parse, the process arguments by default.

    Returns:
        Zero when the example ran, and a non-zero code when it did not.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(None if argv is None else list(argv))
    except SystemExit as request:
        # Usage errors and --help raise SystemExit; its code is returned instead.
        code = request.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 1

    if args.list_tissues:
        return _list_tissues()

    try:
        with _progress.silenced_downloads(args.quiet):
            run(
                tissue=args.tissue,
                n_tiles=args.n_tiles,
                axis=args.axis,
                fold=args.fold,
                parquet=args.parquet,
                revision=args.revision,
                k=args.k,
                tau=args.tau,
                bundle=args.bundle,
                backbone=args.backbone,
                reference=args.reference,
                cache_dir=args.cache_dir,
                device=args.device,
                progress=False if args.quiet else None,
            )
    except (AlhenaError, ImportError) as error:
        print(f"{PROGRAM} demo: {_as_command_line(_one_line(error))}", file=sys.stderr)
        return 1
    return 0
