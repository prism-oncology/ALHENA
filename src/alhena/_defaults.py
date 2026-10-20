"""The defaults a call falls back on where it does not say."""

from __future__ import annotations

import os

from alhena._errors import AlhenaValueError

DEFAULT_BATCH_SIZE: int = 32
"""Tiles per forward pass where a call does not say."""

MAX_DEFAULT_WORKERS: int = 8
"""Default cap on retrieval threads, each holding about 0.4 GB at the default block sizes."""


def _batch_size(value: int) -> int:
    """Return a batch size as an integer, refusing one below one."""
    size = int(value)
    if size < 1:
        raise AlhenaValueError(f"batch_size must be at least one, got {value}.")
    return size


def _retrieval_workers(workers: int | None = None) -> int:
    """Return the threads retrieval runs on: `workers`, or the usable processors up to the cap."""
    if workers is None:
        affinity = getattr(os, "sched_getaffinity", None)
        available = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
        return max(1, min(available, MAX_DEFAULT_WORKERS))
    count = int(workers)
    if count < 1:
        raise AlhenaValueError(
            f"workers must be at least 1, got {workers}. Pass 1 to retrieve on one thread."
        )
    return count
