"""Turn raw H&E tiles into the tensor the encoder reads.

The model reads a 224 by 224 RGB tile normalized with ImageNet statistics, resampled to the
distillation resolution where the resolution of the tiles is declared.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import torch

from alhena._constants import INPUT_SIZE, PATCH_SIZE, REFERENCE_MPP
from alhena._errors import AlhenaValueError
from alhena._geometry import UNDECLARED_RESOLUTION_WARNING as UNDECLARED_RESOLUTION_WARNING
from alhena._geometry import (
    _checked_mpp,
    _crop_note,
    _format_mpp,
    _positive,
    _prepare,
    _resolved_mpp,
    _warn_resolution,
)
from alhena._geometry import channel_lut as channel_lut
from alhena._geometry import fit_to_size as fit_to_size
from alhena._geometry import normalize_tile as normalize_tile
from alhena._geometry import plan_resample as plan_resample
from alhena._geometry import resample as resample
from alhena.types import GeometryReport, ResampleInfo


def prepare_tiles(
    images: np.ndarray | Sequence[np.ndarray],
    mpp: float | None = None,
    target_mpp: float = REFERENCE_MPP,
    progress: bool | None = None,
) -> tuple[torch.Tensor, ResampleInfo]:
    """Prepare tiles for the encoder.

    With `mpp` declared, tiles are resampled to `target_mpp` and center-cropped or padded to the
    model input size, so the field of view is physical; without it, they are resized to the input
    size as they are, with a warning. A tile source carrying a resolution declares it itself, and an
    `mpp` more than `MPP_TOLERANCE` from it is refused. Tiles are read one at a time.

    Args:
        images: An `(N, H, W, 3)` uint8 array, or a sequence of `(H, W, 3)` uint8 tiles which need
            not share a size.
        mpp: Microns per pixel of the tiles. Defaults to the resolution a tile source carries.
        target_mpp: Microns per pixel to resample to.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        An `(N, 3, 224, 224)` float32 tensor, and the `ResampleInfo` record of what was done to
        reach it.

    Raises:
        AlhenaValueError: If there are no tiles, or `mpp` is out of range or contradicts the source.
        GeometryError: If a tile is not uint8 RGB or holds no pixel.
    """
    target_mpp = _positive(target_mpp, "target_mpp")
    tiles = images if isinstance(images, np.ndarray | Sequence) else list(images)
    if len(tiles) == 0:
        raise AlhenaValueError("No tiles were given.")
    if mpp is not None:
        mpp = _checked_mpp(mpp, warn=True, target_mpp=target_mpp)
    mpp = _resolved_mpp(images, mpp)
    _warn_resolution(mpp, target_mpp)
    return _prepare(tiles, mpp, target_mpp, progress)


def describe_geometry(
    mpp: float | None,
    tile_size_px: int = INPUT_SIZE,
    median_nucleus_area_px: float | None = None,
    target_mpp: float = REFERENCE_MPP,
) -> GeometryReport:
    """Describe the physical geometry a set of tiles presents to the model.

    Read from shapes alone, it agrees with `prepare_tiles` on `pad_fraction` and `crop_fraction`,
    both zero without a declared resolution. A crop discarding more than `CROP_WARNING_FRACTION` of
    each field is a note of the report; the pad is reported, not noted.

    Args:
        mpp: Microns per pixel of the tiles, refused where `prepare_tiles` refuses it and never
            warned about.
        tile_size_px: Side of the tiles, in pixels.
        median_nucleus_area_px: Median nucleus area in the tiles, in square pixels, used to report
            how many tokens a typical nucleus spans.
        target_mpp: Microns per pixel the model reads at.

    Returns:
        A record of the field of view, the microns one token covers, the share of the canvas the
        tiles pad with white, the share of each field the center crop discards, and any geometry
        concern.

    Raises:
        AlhenaValueError: If a resolution is not positive or exceeds `MAX_MICROSCOPE_MPP`.
    """
    target_mpp = _positive(target_mpp, "target_mpp")
    notes: list[str] = []
    pad_fraction = crop_fraction = 0.0
    nucleus_note: str | None = None
    if mpp is None:
        notes.append("Tile resolution was not declared; the field of view is unknown.")
        fov = float("nan")
        token_um = float("nan")
        ratio = None
        nucleus_tokens = None
    else:
        mpp = _checked_mpp(mpp, target_mpp=target_mpp)
        fov = tile_size_px * mpp
        token_um = PATCH_SIZE * target_mpp
        ratio = mpp / target_mpp
        plan = plan_resample((tile_size_px, tile_size_px), mpp, target_mpp)
        pad_fraction, crop_fraction = plan.pad_fraction, plan.crop_fraction
        if ratio < 0.5 or ratio > 2.0:
            notes.append(
                f"Tiles are at {_format_mpp(mpp)} micron per pixel against the "
                f"{_format_mpp(target_mpp)} the encoder "
                f"reads, a factor of {ratio:.2f}. Resampling covers this, but the result is "
                "far from the resolution the adapters were distilled at."
            )
        nucleus_tokens = None
        if median_nucleus_area_px is not None and median_nucleus_area_px > 0:
            diameter_um = float(np.sqrt(median_nucleus_area_px)) * mpp
            nucleus_tokens = diameter_um / token_um
            if nucleus_tokens > 2.0:
                nucleus_note = (
                    f"A typical nucleus spans {nucleus_tokens:.1f} tokens per side. Cell "
                    "embeddings pool over more context than they did at the distillation "
                    "resolution."
                )
    report = GeometryReport(
        mpp=mpp,
        tile_size_px=tile_size_px,
        field_of_view_um=fov,
        token_footprint_um=token_um,
        reference_mpp=target_mpp,
        mpp_ratio=ratio,
        median_nucleus_tokens=nucleus_tokens,
        pad_fraction=pad_fraction,
        crop_fraction=crop_fraction,
    )
    notes.extend(note for note in (_crop_note(report), nucleus_note) if note is not None)
    return replace(report, warnings=tuple(notes))
