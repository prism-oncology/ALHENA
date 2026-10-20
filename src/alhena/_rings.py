"""Boxes, areas and centroids of vertex rings, computed for many rings at once."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _stacked(rings: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the vertices of every ring stacked, with the first row and the length of each."""
    lengths = np.fromiter((ring.shape[0] for ring in rings), dtype=np.int64, count=len(rings))
    stacked = np.concatenate(rings, axis=0)
    starts = np.zeros(lengths.shape[0], dtype=np.int64)
    starts[1:] = np.cumsum(lengths)[:-1]
    return stacked, starts, lengths


def _relative(
    stacked: np.ndarray, starts: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each vertex relative to its ring's first, the row of the next, and their cross."""
    # Relative to the first vertex of each ring, so the products stay small.
    local = stacked - np.repeat(stacked[starts], lengths, axis=0)
    following = np.arange(1, stacked.shape[0] + 1, dtype=np.int64)
    following[starts + lengths - 1] = starts
    cross = local[:, 0] * local[following, 1] - local[following, 0] * local[:, 1]
    return local, following, cross


def _ring_boxes(rings: Sequence[np.ndarray]) -> np.ndarray:
    """Return the smallest x and y, then the largest, of the vertices of every ring."""
    if not len(rings):
        return np.zeros((0, 4), dtype=np.float64)
    stacked, starts, _ = _stacked(rings)
    return np.column_stack(
        [np.minimum.reduceat(stacked, starts, axis=0), np.maximum.reduceat(stacked, starts, axis=0)]
    )


def _ring_areas(rings: Sequence[np.ndarray]) -> np.ndarray:
    """Return the signed area each ring encloses, whether or not it repeats its first vertex."""
    if not len(rings):
        return np.zeros(0, dtype=np.float64)
    stacked, starts, lengths = _stacked(rings)
    _, _, cross = _relative(stacked, starts, lengths)
    return 0.5 * np.add.reduceat(cross, starts)


def _ring_centroids(rings: Sequence[np.ndarray]) -> np.ndarray:
    """Return the area centroid of each ring, or the mean of its vertices where it encloses none."""
    if not len(rings):
        return np.zeros((0, 2), dtype=np.float64)
    stacked, starts, lengths = _stacked(rings)
    local, following, cross = _relative(stacked, starts, lengths)
    x, y = local[:, 0], local[:, 1]
    doubled_area = np.add.reduceat(cross, starts)
    moment_x = np.add.reduceat((x + x[following]) * cross, starts)
    moment_y = np.add.reduceat((y + y[following]) * cross, starts)
    mean = np.add.reduceat(local, starts, axis=0) / lengths[:, None]
    extent = np.maximum.reduceat(np.abs(local).max(axis=1), starts)
    flat = np.abs(doubled_area) <= 1e-12 * np.maximum(extent, 1.0) ** 2
    denominator = 3.0 * np.where(flat, 1.0, doubled_area)
    area_centroid = np.column_stack([moment_x / denominator, moment_y / denominator])
    return stacked[starts] + np.where(flat[:, None], mean, area_centroid)
