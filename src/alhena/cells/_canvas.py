"""The token grid of the canvas, and nucleus masks fitted onto it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import pandas as pd

from alhena._constants import INPUT_SIZE, NUM_TOKENS, PATCH_SIZE, REFERENCE_MPP
from alhena._errors import GeometryError, MembershipError
from alhena._frames import _as_tile_scale, _window_centers
from alhena._geometry import _checked_mpp, _positive, _target_scale
from alhena.cells._table import _concatenate

_MASK_INTERPOLATION = cv2.INTER_NEAREST
_MASK_BACKGROUND = 0

# Tolerance, absolute and relative, under which two positions computed apart count as one.
_POSITION_SLACK = 1e-9

# Largest label resized as it is: OpenCV keeps int32 values and narrows wider ones.
_LARGEST_RESIZED_LABEL = int(np.iinfo(np.int32).max)


def token_index_map(size: int = INPUT_SIZE, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Return the token index of every pixel of the model input canvas, row-major over the grid.

    Args:
        size: Side of the canvas, in pixels.
        patch_size: Side of one token, in pixels.

    Returns:
        A `(size, size)` int32 array holding `(y // patch_size) * grid + (x // patch_size)`.

    Raises:
        GeometryError: If the canvas is not a whole number of tokens across.
    """
    if size <= 0 or patch_size <= 0 or size % patch_size != 0:
        raise GeometryError(
            f"A {size} pixel canvas does not divide into {patch_size} pixel tokens."
        )
    grid = size // patch_size
    rows = (np.arange(size, dtype=np.int32) // patch_size)[:, None]
    columns = (np.arange(size, dtype=np.int32) // patch_size)[None, :]
    return (rows * grid + columns).astype(np.int32)


def fit_mask(
    mask: np.ndarray,
    mpp: float | None = None,
    target_mpp: float = REFERENCE_MPP,
    size: int = INPUT_SIZE,
) -> np.ndarray:
    """Carry an integer label mask onto the model input canvas, as the image of its tile is.

    Nearest-neighbor resampling keeps label values, and the padding is background. Declare the same
    `mpp` as for the image.

    Args:
        mask: A `(H, W)` integer array, zero for background and one identifier per nucleus.
        mpp: Microns per pixel of the tile the mask belongs to.
        target_mpp: Microns per pixel to resample to.
        size: Side of the model input canvas.

    Returns:
        A `(size, size)` array of the dtype of the input.

    Raises:
        AlhenaValueError: If a resolution is not positive or exceeds `MAX_MICROSCOPE_MPP`.
        GeometryError: If the mask is not a non-empty 2-D array of non-negative integers.
    """
    target_mpp = _positive(target_mpp, "target_mpp")
    if mpp is not None:
        mpp = _checked_mpp(mpp, warn=True, target_mpp=target_mpp)
    return _fit_mask(mask, mpp, target_mpp, size)


def _fit_mask(mask: np.ndarray, mpp: float | None, target_mpp: float, size: int) -> np.ndarray:
    """Carry a label mask onto the canvas as `fit_mask` does, at a resolution checked already."""
    working = _as_label_mask(mask)
    if mpp is None:
        return _resize_labels(working, size, size)

    # The factor of the image path, so the mask and its image describe one field of view.
    scale = _target_scale(mpp, target_mpp)
    if scale != 1.0:
        height = round(working.shape[0] * scale)
        width = round(working.shape[1] * scale)
        if height < 1 or width < 1:
            raise GeometryError(
                f"Resampling a {working.shape[1]} by {working.shape[0]} mask by {scale:.4f} "
                "leaves no pixels. Check the mpp you declared."
            )
        working = _resize_labels(working, height, width)
    return _fit_labels_to_size(working, size)


def _as_label_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or not array.size:
        raise GeometryError(
            f"Expected a (H, W) label mask of at least one pixel, got shape {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise GeometryError(f"Expected an integer label mask, got {array.dtype}.")
    if array.size and int(array.min()) < 0:
        raise GeometryError(
            "Label masks hold zero for background and positive identifiers for nuclei; a "
            "negative label has no meaning."
        )
    return np.ascontiguousarray(array)


def _resize_labels(labels: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a label mask with nearest-neighbor sampling, preserving the label values."""
    if labels.shape[0] == height and labels.shape[1] == width:
        return labels
    # Nearest-neighbor sampling keeps values, so int32 labels are resized as they are.
    if labels.size and int(labels.max()) <= _LARGEST_RESIZED_LABEL:
        resized = cv2.resize(
            labels.astype(np.int32, copy=False), (width, height), interpolation=_MASK_INTERPOLATION
        )
        return resized.astype(labels.dtype, copy=False)
    # OpenCV silently narrows 64-bit labels to int32, so larger ones are resized as codes.
    values, codes = np.unique(labels, return_inverse=True)
    compact = codes.reshape(labels.shape).astype(np.int32, copy=False)
    resized = cv2.resize(compact, (width, height), interpolation=_MASK_INTERPOLATION)
    return values[resized.astype(np.intp, copy=False)]


def _fit_labels_to_size(labels: np.ndarray, size: int) -> np.ndarray:
    """Center-crop or pad a label mask to a square, exactly as the image path does."""
    height, width = labels.shape[:2]
    if height < size or width < size:
        canvas = np.full(
            (max(height, size), max(width, size)), _MASK_BACKGROUND, dtype=labels.dtype
        )
        top = (canvas.shape[0] - height) // 2
        left = (canvas.shape[1] - width) // 2
        canvas[top : top + height, left : left + width] = labels
        labels = canvas
        height, width = labels.shape[:2]
    if height > size or width > size:
        top = (height - size) // 2
        left = (width - size) // 2
        labels = labels[top : top + size, left : left + size]
    return labels


def _tokenize_mask(
    fitted: np.ndarray, token_map: np.ndarray, n_tokens: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the label, token index and pixel count of every occupied (label, token) pair."""
    flat = fitted.reshape(-1)
    occupied = flat > 0
    if not occupied.any():
        empty_labels = np.empty(0, dtype=fitted.dtype)
        return empty_labels, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    labels = flat[occupied]
    tokens = token_map.reshape(-1)[occupied].astype(np.int64, copy=False)
    values, codes = np.unique(labels, return_inverse=True)
    key = codes.reshape(-1).astype(np.int64) * np.int64(n_tokens) + tokens
    unique_key, counts = np.unique(key, return_counts=True)
    return (
        values[unique_key // n_tokens],
        (unique_key % n_tokens).astype(np.int32),
        counts.astype(np.int32),
    )


def _as_tile_indices(tile_indices: Sequence[int] | np.ndarray, expected: int, what: str):
    tiles = np.asarray(tile_indices)
    if tiles.ndim != 1 or tiles.shape[0] != expected:
        raise MembershipError(
            f"tile_indices holds {tiles.size} entries for {expected} {what}; one tile index per "
            "entry is required."
        )
    if not np.issubdtype(tiles.dtype, np.integer):
        raise MembershipError(f"tile_indices must be integers, got {tiles.dtype}.")
    if tiles.size and int(tiles.min()) < 0:
        raise MembershipError("tile_indices must not be negative.")
    return tiles.astype(np.int64, copy=False)


def _as_cell_ids(
    cell_ids: Sequence[Any] | np.ndarray | None, expected: int, require_unique: bool = False
) -> np.ndarray:
    if cell_ids is None:
        return np.arange(expected, dtype=np.int64)
    identifiers = np.asarray(cell_ids)
    if identifiers.ndim != 1 or identifiers.shape[0] != expected:
        raise MembershipError(
            f"cell_ids holds {identifiers.size} entries for {expected} nuclei; one identifier "
            "per nucleus is required."
        )
    if pd.isna(identifiers).any():
        raise MembershipError("cell_ids holds a missing value; every nucleus must be named.")
    if require_unique and len(pd.unique(identifiers)) != expected:
        raise MembershipError("cell_ids repeats an identifier. Name each cell once.")
    return identifiers


def _clear_past_edge(mask: np.ndarray, inside: np.ndarray | None, tile: int) -> np.ndarray:
    """Return the mask of a tile cleared past the edge of its image, where the tile is white."""
    if inside is None:
        return mask
    height, width = (int(value) for value in inside[tile])
    if height >= mask.shape[0] and width >= mask.shape[1]:
        return mask
    cleared = np.array(mask)
    cleared[height:, :] = _MASK_BACKGROUND
    cleared[:, width:] = _MASK_BACKGROUND
    return cleared


def _as_origins(
    tile_origins: np.ndarray | None, tile_scale: float, tiles: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Return the tile origins a builder is given, with their scale, or None where none were."""
    scale = _as_tile_scale(tile_scale)
    if tile_origins is None:
        if scale != 1.0:
            raise MembershipError(
                "tile_scale scales the tile origins, and no tile_origins were given."
            )
        return None
    try:
        origins = np.asarray(tile_origins, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MembershipError(
            "tile_origins is an (n_tiles, 2) numeric array of the x, y origin of every tile."
        ) from error
    if origins.ndim != 2 or origins.shape[1] != 2:
        raise MembershipError(f"Expected tile_origins of shape (n_tiles, 2), got {origins.shape}.")
    if not np.isfinite(origins).all():
        raise MembershipError("Tile origins must be finite.")
    if tiles.size and int(tiles.max()) >= origins.shape[0]:
        raise MembershipError(
            f"tile_origins holds {origins.shape[0]} rows and tile index {int(tiles.max())} is "
            "used; one origin per tile is required, the tile index being the row index."
        )
    return origins, scale


def _frame_position(origin: np.ndarray, local: np.ndarray, scale: float) -> np.ndarray:
    """Carry positions in tile pixels into the frame of the origin, pixel centers at integers."""
    if scale == 1.0:
        return origin + local
    return origin + scale * (local + 0.5) - 0.5


def _tile_centers(origins: np.ndarray, shape: tuple[int, int], scale: float) -> np.ndarray:
    """Return the center of every tile of one shape, in the frame of the origins."""
    height, width = shape
    extent = scale * np.array([width, height], dtype=np.float64)
    return _window_centers(np.column_stack([origins, origins + extent]))


class _TokenRows:
    """The rows of a membership table, gathered from the mask drawn on each tile in turn."""

    def __init__(self, mpp: float | None, target_mpp: float, inside: np.ndarray | None) -> None:
        self._mpp = mpp
        self._target_mpp = target_mpp
        self._inside = inside
        self._token_map = token_index_map()
        self._tiles: list[np.ndarray] = []
        self._cells: list[np.ndarray] = []
        self._tokens: list[np.ndarray] = []
        self._pixels: list[np.ndarray] = []

    def add(self, tile: int, mask: np.ndarray, names: np.ndarray | None = None) -> None:
        """Fit the mask of a tile to the canvas; code k names `names[k - 1]`, or itself."""
        cleared = _clear_past_edge(mask, self._inside, tile)
        fitted = _fit_mask(cleared, self._mpp, self._target_mpp, INPUT_SIZE)
        codes, tokens, pixels = _tokenize_mask(fitted, self._token_map, NUM_TOKENS)
        if codes.size == 0:
            return
        self._tiles.append(np.full(codes.shape, tile, dtype=np.int64))
        if names is None:
            self._cells.append(codes.astype(np.int64, copy=False))
        else:
            self._cells.append(names[codes.astype(np.intp) - 1])
        self._tokens.append(tokens)
        self._pixels.append(pixels)

    def columns(self, dtype: Any) -> dict[str, np.ndarray]:
        """Return the tile, cell, token and pixel columns; an empty cell column takes `dtype`."""
        return {
            "tiles": _concatenate(self._tiles, np.int64),
            "cells": _concatenate(self._cells, dtype),
            "tokens": _concatenate(self._tokens, np.int32),
            "pixels": _concatenate(self._pixels, np.int32),
        }
