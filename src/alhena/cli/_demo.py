"""The demo command."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from alhena import demo


def _forwarded(arguments: Sequence[str]) -> list[str]:
    """Return the arguments meant for the demo, without a separator written before them."""
    forwarded = list(arguments)
    # A bare separator is dropped: the example's options are handed on unread.
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return forwarded


def _demo(args: argparse.Namespace) -> int:
    """Run the worked example with the arguments the command line carried for it."""
    return _run_demo(args.args)


def _run_demo(arguments: Sequence[str]) -> int:
    """Run the worked example, which owns its own options."""
    return demo.main(_forwarded(arguments))
