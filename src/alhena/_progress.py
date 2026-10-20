"""Report on standard error what a long call is doing, and how far it has got.

A call's `progress` wins; else `ALHENA_DISABLE_PROGRESS_BARS`, set true, turns reporting off. The
outermost loop knowing the total owns the bar, and inner calls get `progress=False`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Final, TypeVar

if TYPE_CHECKING:
    from tqdm import tqdm

_Item = TypeVar("_Item")

# Environment variable that, set to a true value, turns progress reporting off.
PROGRESS_ENV_VAR: Final[str] = "ALHENA_DISABLE_PROGRESS_BARS"

# Seconds a stage runs before its bar is drawn, so a call that returns quickly draws nothing.
DELAY_S: float = 2.0

# Seconds before a bar is first drawn and between redraws, where every redraw is kept.
LOG_REFRESH_S: float = 10.0

# tqdm's own variable for the seconds between redraws, which wins over `LOG_REFRESH_S`.
_REFRESH_ENV_VAR: Final[str] = "TQDM_MININTERVAL"

# Values of the environment variable read as true, compared without case or surrounding space.
_TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def enabled(progress: bool | None = None) -> bool:
    """Return whether progress is reported: the call's choice, else the environment's."""
    if progress is not None:
        return bool(progress)
    return os.environ.get(PROGRESS_ENV_VAR, "").strip().lower() not in _TRUE_VALUES


def bar(
    total: int,
    desc: str,
    unit: str,
    progress: bool | None = None,
    scaled: bool = False,
) -> tqdm:
    """Return a bar over the units of one stage, drawn on standard error.

    A notebook widget under Jupyter and text elsewhere; disabled, it accepts the same calls and
    draws nothing.

    Args:
        total: Units the stage runs over.
        desc: What the stage is doing, shown ahead of the bar.
        unit: Name of one unit, shown in the rate.
        progress: The call's own choice. None defers to the environment.
        scaled: Show counts with a metric prefix, for stages running over many units.

    Returns:
        The bar, to be advanced with `update` and closed, or used as a context manager.
    """
    from tqdm.auto import tqdm

    return tqdm(**_options(total, desc, unit, progress, scaled))


def track(
    items: Iterable[_Item],
    total: int,
    desc: str,
    unit: str,
    progress: bool | None = None,
) -> tqdm:
    """Return the items of a loop behind a bar that counts one unit per item.

    An item is counted once the body of the loop has finished with it, a `continue` included.

    Args:
        items: What the loop runs over, one item per unit.
        total: Number of items, for a loop over an iterable that does not carry its length.
        desc: What the stage is doing, shown ahead of the bar.
        unit: Name of one unit, shown in the rate.
        progress: The call's own choice. None defers to the environment.

    Returns:
        The items, to be iterated inside the bar used as a context manager.
    """
    from tqdm.auto import tqdm

    return tqdm(items, **_options(total, desc, unit, progress, scaled=False))


def _options(
    total: int, desc: str, unit: str, progress: bool | None, scaled: bool
) -> dict[str, Any]:
    """Return the settings every bar of the package is drawn with."""
    options: dict[str, Any] = {
        "total": total,
        "desc": desc,
        "unit": unit,
        "unit_scale": scaled,
        "delay": DELAY_S,
        "dynamic_ncols": True,
        "file": sys.stderr,
    }
    # tqdm's defaults hold; a stream keeping every redraw refreshes slower, unless tqdm's is set.
    if not enabled(progress):
        options["disable"] = True
    elif _keeps_every_redraw(sys.stderr) and _REFRESH_ENV_VAR not in os.environ:
        options["mininterval"] = LOG_REFRESH_S
    return options


def _keeps_every_redraw(stream: Any) -> bool:
    """Return whether a stream keeps every redraw of a bar, as a log file or a pipe does."""
    if "ipykernel" in sys.modules:
        return False
    try:
        return not stream.isatty()
    except (AttributeError, ValueError):
        return True


def step(message: str, progress: bool | None = None) -> None:
    """Announce a long step that has no loop to measure, on one line of standard error.

    Args:
        message: What the step is doing.
        progress: The call's own choice. None defers to the environment.
    """
    if not enabled(progress):
        return
    from tqdm.auto import tqdm

    # Written through tqdm, so a line printed while a bar is drawn does not break the bar.
    tqdm.write(message, file=sys.stderr)


@contextmanager
def silenced_downloads(active: bool = True) -> Iterator[None]:
    """Silence the download bars huggingface-hub draws, for the length of a block.

    The process-wide switch is restored when the block ends, so bars already off stay off.

    Args:
        active: Whether to silence them. The block runs unchanged when this is False.
    """
    if not active:
        yield
        return
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    already_off = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        yield
    finally:
        if not already_off:
            enable_progress_bars()
