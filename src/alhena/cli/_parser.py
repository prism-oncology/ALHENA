"""The parser of the command line, and its entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from alhena import _progress
from alhena._commands import (
    _add_backbone_option,
    _add_cache_option,
    _add_model_options,
    _add_quiet_option,
)
from alhena._constants import (
    CLASS_AXES,
    DEFAULT_AXIS,
    DEFAULT_K,
    DEFAULT_TAU,
    HUB_REVISION,
    TISSUE_TO_SHARD,
)
from alhena._defaults import DEFAULT_BATCH_SIZE, MAX_DEFAULT_WORKERS
from alhena._errors import AlhenaError
from alhena._truth import _LABEL_TABLE_SUFFIXES
from alhena._version import __version__
from alhena._wording import PROGRAM
from alhena.cli._annotate import _annotate
from alhena.cli._demo import _demo, _run_demo
from alhena.cli._download import _download
from alhena.cli._encode import _encode
from alhena.cli._info import _info
from alhena.cli._options import _add_nuclei_options, _add_shard_options, _add_tile_options
from alhena.cli._reference import _reference_check, _reference_describe, _reference_list
from alhena.cli._shared import (
    CELLS_SUFFIXES,
    EMBEDDING_SUFFIXES,
    _failure,
    _reported,
)
from alhena.io import IMAGE_SUFFIXES, SLIDE_SUFFIXES

# The suffixes a whole slide is read from, as the help of the tiles argument lists them.
_SLIDE_FORMS: str = f"{', '.join(SLIDE_SUFFIXES[:-1])} or {SLIDE_SUFFIXES[-1]}"

# Image suffixes the help of the tiles lists; a TIFF file is read as a slide.
_IMAGE_ONLY: tuple[str, ...] = tuple(
    suffix for suffix in IMAGE_SUFFIXES if suffix not in SLIDE_SUFFIXES
)
_IMAGE_FORMS: str = f"{', '.join(_IMAGE_ONLY[:-1])} or {_IMAGE_ONLY[-1]}"

# Subcommand whose arguments belong to the worked example rather than to this parser.
_DEMO_COMMAND: str = "demo"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command line."""
    from alhena.reference import DEFAULT_SELF_CHECK_CELLS

    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Embed H&E tiles with ALHENA, and annotate cells against a labeled reference with "
            "nothing fitted at deployment."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    info = commands.add_parser(
        "info", help="report the version, the cache, and what is and is not downloaded"
    )
    _add_backbone_option(info)
    info.add_argument(
        "--bundle",
        default=None,
        help=(
            "the bundle to report on, a directory or a repository owner/name; the published "
            "bundle in the cache by default"
        ),
    )
    _add_cache_option(info)
    info.set_defaults(handler=_info)

    download = commands.add_parser(
        "download",
        help="fetch the encoder bundle and references",
        description=(
            "Fetch the encoder bundle, the references of one or more tissues, or both. At least "
            "one of --bundle and --tissue is required."
        ),
    )
    download.add_argument("--bundle", action="store_true", help="fetch the encoder bundle")
    download.add_argument(
        "--tissue",
        action="extend",
        nargs="+",
        default=[],
        metavar="TISSUE",
        help="fetch the reference of one or more tissues, by tissue name, alias or reference name",
    )
    download.add_argument(
        "--revision",
        default=None,
        help=(
            f"branch, tag or commit to fetch; {HUB_REVISION}, the release this version of ALHENA "
            "reads, by default"
        ),
    )
    _add_cache_option(download)
    _add_quiet_option(download)
    download.set_defaults(handler=_download)

    encode = commands.add_parser(
        "encode",
        help="compute tile embeddings, or the cell embeddings of their nuclei",
        description=(
            "Compute tile embeddings. Given --nuclei, compute the embeddings of their cells "
            "instead, reading each tile once, and write them as a table of cells, which annotate "
            "reads without the encoder."
        ),
    )
    encode.add_argument(
        "tiles",
        type=Path,
        help=(
            "a directory of images, a .npy array of shape (n_tiles, height, width, 3), an HDF5 "
            "file holding an img dataset and optionally barcode and coords, a whole slide, a "
            f"{_SLIDE_FORMS} file: any TIFF is read as a slide, OME-TIFF included; or one large "
            f"image cut into a grid, a {_IMAGE_FORMS} file or a .npy array of shape (height, "
            "width, 3)"
        ),
    )
    encode.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            f"file the embeddings are written to: {' or '.join(EMBEDDING_SUFFIXES)} for tile "
            f"embeddings, {' or '.join(CELLS_SUFFIXES)} for the cells of --nuclei"
        ),
    )
    encode.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="tiles per forward pass"
    )
    _add_nuclei_options(encode, "embed the cells of these nuclei rather than the tiles")
    encode.add_argument(
        "--all-tiles",
        action="store_true",
        help="with --nuclei, read every tile, including those no cell is read from",
    )
    encode.add_argument(
        "--tile-embeddings",
        type=Path,
        default=None,
        help=(
            "with --nuclei, also write the embeddings of the tiles read, to a "
            f"{' or '.join(EMBEDDING_SUFFIXES)} file"
        ),
    )
    _add_tile_options(encode)
    _add_model_options(encode)
    _add_quiet_option(encode)
    encode.set_defaults(handler=_encode)

    annotate = commands.add_parser(
        "annotate", help="annotate every nucleus, or every cell of a table, against a reference"
    )
    annotate.add_argument(
        "tiles",
        type=Path,
        help=(
            "the tiles the nuclei were segmented in, in any form encode takes, or a table of "
            f"cells written by `{PROGRAM} encode --nuclei`, annotated without the encoder"
        ),
    )
    _add_nuclei_options(annotate, "required with tiles")
    against = annotate.add_mutually_exclusive_group(required=True)
    against.add_argument(
        "--tissue",
        default=None,
        help=(
            "tissue whose published reference is retrieved against: one of "
            f"{', '.join(sorted(TISSUE_TO_SHARD))}, or another name `{PROGRAM} reference list` "
            "gives it"
        ),
    )
    against.add_argument(
        "--reference", type=Path, default=None, help="a reference file on disk, instead of a tissue"
    )
    annotate.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            f"file the annotations are written to: {' or '.join(_LABEL_TABLE_SUFFIXES)} for the "
            f"annotations alone, {' or '.join(CELLS_SUFFIXES)} for the annotations with their "
            "cells and provenance"
        ),
    )
    annotate.add_argument(
        "--axis",
        default=DEFAULT_AXIS,
        help=(
            f"class axis the cells are annotated on: one of {', '.join(sorted(CLASS_AXES))}, or, "
            "with --reference, an axis the file was built on"
        ),
    )
    annotate.add_argument(
        "--k", type=int, default=DEFAULT_K, help="reference neighbors each cell votes over"
    )
    annotate.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
        help="softmax temperature over the neighbor similarities",
    )
    annotate.add_argument(
        "--truth",
        type=Path,
        default=None,
        help=(
            "a table of known labels with cell_id and label columns, which turns on the scoring "
            "report"
        ),
    )
    annotate.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "threads retrieving neighbors at once; the processors available, at most "
            f"{MAX_DEFAULT_WORKERS}, by default"
        ),
    )
    annotate.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"tiles per forward pass; {DEFAULT_BATCH_SIZE} by default",
    )
    _add_tile_options(annotate)
    _add_model_options(annotate)
    _add_quiet_option(annotate)
    annotate.set_defaults(handler=_annotate)

    reference = commands.add_parser("reference", help="list, describe or check references")
    actions = reference.add_subparsers(dest="action", required=True, metavar="ACTION")

    listing = actions.add_parser("list", help="list the published references")
    _add_cache_option(listing)
    listing.set_defaults(handler=_reference_list)

    describe = actions.add_parser(
        "describe", help="report a reference's support and the classes it cannot express"
    )
    _add_shard_options(describe)
    _add_quiet_option(describe)
    describe.set_defaults(handler=_reference_describe)

    check = actions.add_parser("check", help="retrieve a reference against itself")
    _add_shard_options(check)
    check.add_argument(
        "--k", type=int, default=DEFAULT_K, help="neighbors each cell votes over in the check"
    )
    check.add_argument(
        "--tau", type=float, default=DEFAULT_TAU, help="softmax temperature in the check"
    )
    check.add_argument(
        "--max-cells",
        type=int,
        default=DEFAULT_SELF_CHECK_CELLS,
        help="cells the check scores, %(default)s by default; a smaller reference is scored entire",
    )
    check.add_argument("--seed", type=int, default=0, help="seed of the subsample the check scores")
    _add_quiet_option(check)
    check.set_defaults(handler=_reference_check)

    demo = commands.add_parser(
        "demo", help=f"run the worked example; `{PROGRAM} demo --help` lists its options"
    )
    demo.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help=(
            "arguments of the worked example, forwarded to it unchanged; a bare -- may be "
            "written before them and is dropped"
        ),
    )
    demo.set_defaults(handler=_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code.

    A package error is reported in one line on standard error, without a traceback; anything else
    propagates with its traceback.

    Args:
        argv: Arguments to parse, the process arguments by default.

    Returns:
        Zero when the command succeeded, and a non-zero code when it did not.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == _DEMO_COMMAND:
        # The example owns its options: everything after the command is handed to it unread.
        return _run_demo(arguments[1:])

    parser = build_parser()
    try:
        args = parser.parse_args(arguments)
    except SystemExit as request:
        # Usage errors, --help and --version raise SystemExit; its code is returned instead.
        code = request.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 1

    try:
        # A command owns its process, so --quiet silences huggingface-hub's download bars too.
        with _progress.silenced_downloads(getattr(args, "quiet", False)):
            return int(args.handler(args))
    except AlhenaError as error:
        _failure(_reported(error))
        return 1
