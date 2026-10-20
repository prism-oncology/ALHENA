"""Resolution checks, and resampling, normalizing and fitting tiles to the encoder's canvas."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import replace

import cv2
import numpy as np
import torch

from alhena import _progress
from alhena._constants import (
    CROP_WARNING_FRACTION,
    IMAGE_MEAN,
    IMAGE_STD,
    INPUT_SIZE,
    MAX_MICROSCOPE_MPP,
    MIN_MICROSCOPE_MPP,
    MPP_TOLERANCE,
    NUM_TOKENS,
    REFERENCE_MPP,
)
from alhena._errors import PACKAGE_PREFIXES, AlhenaValueError, GeometryError, GeometryWarning
from alhena.types import GeometryReport, ResampleInfo

UNDECLARED_RESOLUTION_WARNING: str = "Tile resolution was not declared, so tiles are resized to"
"""Opening of the warning raised where no tile resolution is declared, to match it in a filter."""

_INTERPOLATION = cv2.INTER_LINEAR
# Bilinear down to a twofold shrink; past it, area averaging keeps texture from aliasing.
_SHRINK_INTERPOLATION = cv2.INTER_AREA
_BILINEAR_MIN_SCALE = 0.5
_PAD_VALUE = 255


def channel_lut(
    mean: Sequence[float] = IMAGE_MEAN,
    std: Sequence[float] = IMAGE_STD,
) -> np.ndarray:
    """Return the per-channel lookup table mapping a byte value to its normalized float.

    The reciprocal is taken in float32, as in the distillation, rather than in double and cast,
    which rounds differently.

    Args:
        mean: Per-channel mean, on the unit scale.
        std: Per-channel standard deviation, on the unit scale.

    Returns:
        A `(3, 256)` float32 table indexed by channel and byte value.
    """
    offset = np.asarray(mean, dtype=np.float32) * 255.0
    inverse = np.reciprocal(np.asarray(std, dtype=np.float32) * 255.0, dtype=np.float32)
    levels = np.arange(256, dtype=np.float32)[None, :]
    return (levels - offset[:, None]) * inverse[:, None]


def normalize_tile(image: np.ndarray, lut: np.ndarray | None = None) -> np.ndarray:
    """Normalize one RGB tile and return it in channel-first layout.

    Args:
        image: A (H, W, 3) uint8 RGB tile.
        lut: A table from `channel_lut`, built once when normalizing many tiles.

    Returns:
        A (3, H, W) float32 array.

    Raises:
        GeometryError: If the tile is not (H, W, 3) uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise GeometryError(f"Expected an (H, W, 3) RGB tile, got shape {image.shape}.")
    if image.dtype != np.uint8:
        raise GeometryError(
            f"Expected a uint8 tile, got {image.dtype}. Convert to bytes before normalizing "
            "so the lookup table applies."
        )
    table = channel_lut() if lut is None else lut
    out = np.empty(image.shape, dtype=np.float32)
    for channel in range(3):
        out[..., channel] = table[channel][image[..., channel]]
    return out.transpose(2, 0, 1)


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a tile bilinearly, or by averaging over pixel area past a twofold shrink."""
    across = width / image.shape[1]
    down = height / image.shape[0]
    # OpenCV averages area only where no side grows, so a tile growing on one side stays bilinear.
    shrinking = max(across, down) <= 1.0 and min(across, down) < _BILINEAR_MIN_SCALE
    interpolation = _SHRINK_INTERPOLATION if shrinking else _INTERPOLATION
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resample(image: np.ndarray, scale: float) -> np.ndarray:
    """Resample a tile by a scale factor, preserving its aspect ratio.

    Bilinear, or averaged over pixel area past a twofold shrink, where bilinear would alias fine
    texture.

    Args:
        image: A `(H, W, 3)` uint8 RGB tile.
        scale: Multiplier applied to both sides. A value above one enlarges.

    Returns:
        The resampled tile.

    Raises:
        GeometryError: If the scale would collapse the tile to nothing.
    """
    height = round(image.shape[0] * scale)
    width = round(image.shape[1] * scale)
    if height < 1 or width < 1:
        raise GeometryError(
            f"Resampling a {image.shape[1]} by {image.shape[0]} tile by {scale:.4f} leaves no "
            "pixels. Check the mpp you declared."
        )
    if (height, width) == image.shape[:2]:
        return image
    return _resize(image, width, height)


def _target_scale(mpp: float, target_mpp: float) -> float:
    """Return the factor tiles are resized by, one inside the tolerance around the target."""
    scale = mpp / target_mpp
    return 1.0 if abs(scale - 1.0) <= MPP_TOLERANCE else scale


def _checked_mpp(mpp: float, warn: bool = False, target_mpp: float = REFERENCE_MPP) -> float:
    """Return a declared resolution as a float, refusing one no canvas can be read at."""
    value = _positive(mpp, "mpp")
    target_mpp = _positive(target_mpp, "target_mpp")
    field_um = INPUT_SIZE * target_mpp
    if value > MAX_MICROSCOPE_MPP:
        raise AlhenaValueError(
            f"mpp={value:g} declares pixels coarser than the {MAX_MICROSCOPE_MPP:g} microns of a "
            f"microscope image. The {INPUT_SIZE} pixel canvas the encoder reads spans "
            f"{field_um:.1f} microns, {field_um / value:.2g} pixels of such an image across, so "
            f"each of its {NUM_TOKENS} tokens would read a fraction of one pixel. Such a value "
            "can come from a length in nanometers interpreted as microns, or from a "
            "dots-per-inch value used as the resolution."
        )
    if value < MIN_MICROSCOPE_MPP and warn:
        warnings.warn(
            f"mpp={value:g} declares pixels finer than the {MIN_MICROSCOPE_MPP:g} micron of a "
            "microscope image. This can happen when a length in millimeters is interpreted as "
            f"microns. It is used as declared, and reaching the {target_mpp} micron per pixel "
            f"the encoder reads shrinks each tile {target_mpp / value:.0f} times.",
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
    return value


def _positive(value: float, name: str) -> float:
    """Return a resolution as a float, refusing one that is not a finite positive number."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(f"{name} must be a number, got {value!r}.") from error
    if not math.isfinite(number) or number <= 0.0:
        raise AlhenaValueError(f"{name} must be finite and positive, got {value}.")
    return number


def _format_mpp(value: float) -> str:
    """Format microns per pixel for a human-readable message only."""
    displayed = f"{float(value):.4f}".rstrip("0").rstrip(".")
    if displayed in {"0", "-0"} and value != 0:
        return f"{float(value):.4g}"
    return displayed


def _carried_mpp(tiles: object) -> tuple[float, str] | None:
    """Return the resolution a tile source carries and where it was read from, or None."""
    value = getattr(tiles, "mpp", None)
    if not isinstance(value, float | int) or isinstance(value, bool):
        return None
    where = getattr(tiles, "mpp_source", None)
    return float(value), "the tiles" if where is None else str(where)


def _resolved_mpp(tiles: object, mpp: float | None) -> float | None:
    """Return the resolution to read tiles at; a declared one must agree with the one they carry."""
    carried = _carried_mpp(tiles)
    if carried is None:
        return mpp
    own, where = carried
    if mpp is None:
        return own
    if abs(float(mpp) - own) <= MPP_TOLERANCE * own:
        return float(mpp)
    raise AlhenaValueError(
        f"These tiles are at {own:.4g} micron per pixel, resolved from {where} when they were "
        f"read, and mpp={float(mpp):g} was declared for them, more than {MPP_TOLERANCE:.0%} "
        "apart, so at most one of them is the resolution of the tiles. Declare the resolution of "
        "the tiles themselves, coarser for a pyramid level than for its full-resolution level, or "
        "leave mpp out."
    )


def _filling_side(mpp: float, target_mpp: float = REFERENCE_MPP) -> int:
    """Return the side, in pixels at `mpp`, of the smallest square tile covering the canvas."""
    # Within MPP_TOLERANCE no tile is resampled, so the canvas side itself covers the canvas.
    if _target_scale(mpp, target_mpp) == 1.0:
        return INPUT_SIZE
    # Taken a hair low, so a quotient landing on a whole number (200 at 0.476) is not pushed up.
    return math.ceil(INPUT_SIZE * target_mpp / mpp - 1e-9)


def _advised_cut(mpp: float, target_mpp: float = REFERENCE_MPP) -> int | None:
    """Return the widest cut at a resolution whose center crop stays within the warned share."""
    if not mpp > 0.0:
        return None
    side = _filling_side(mpp, target_mpp)
    while side > 1 and _crop_of_cut(side, mpp, target_mpp) > CROP_WARNING_FRACTION:
        side -= 1
    return None if _crop_of_cut(side, mpp, target_mpp) > CROP_WARNING_FRACTION else side


def _crop_of_cut(side: int, mpp: float, target_mpp: float) -> float:
    """Return the share of a square tile of that side the center crop discards."""
    return plan_resample((side, side), mpp, target_mpp).crop_fraction


def _crop_advice(mpp: float, target_mpp: float = REFERENCE_MPP) -> str:
    """Return what to do about a center crop discarding too much, at a resolution."""
    side = _advised_cut(mpp, target_mpp)
    if side is None:
        return (
            f"No cut keeps the crop below {CROP_WARNING_FRACTION:.0%} at this resolution: the "
            "canvas holds so few of its pixels that every cut either overshoots it or leaves it "
            "nearly empty. Read the tissue at a finer resolution."
        )
    return (
        f"Cut tiles of about {side} pixels at this resolution, with a stride below the tile side "
        "so that a nucleus at the edge of one tile is read from a neighboring one."
    )


def _kept_pixels(height: int, width: int, size: int = INPUT_SIZE) -> int:
    """Return how many pixels of a tile being fitted reach the size by size output."""
    top = max(size - height, 0) // 2
    left = max(size - width, 0) // 2
    crop_top = (max(height, size) - size) // 2
    crop_left = (max(width, size) - size) // 2
    rows = max(min(top + height, crop_top + size) - max(top, crop_top), 0)
    columns = max(min(left + width, crop_left + size) - max(left, crop_left), 0)
    return rows * columns


def _pad_pixels(height: int, width: int, size: int = INPUT_SIZE) -> int:
    """Return how many of the size by size output pixels fitting a tile fills with pad."""
    return size * size - _kept_pixels(height, width, size)


def _crop_fraction(height: int, width: int, size: int = INPUT_SIZE) -> float:
    """Return the share of a tile, at its fitted size, that the center crop discards."""
    area = int(height) * int(width)
    if area <= 0:
        return 0.0
    return (area - _kept_pixels(height, width, size)) / float(area)


def fit_to_size(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, bool, bool, int]:
    """Center-crop or pad a tile to a square of the given side.

    Padding uses white, the background of an H&E section.

    Args:
        image: A (H, W, 3) uint8 RGB tile.
        size: Side of the output square.

    Returns:
        The fitted tile, whether it was padded, whether it was cropped, and how many pixels of
        the output came from padding rather than from the tile.
    """
    height, width = image.shape[:2]
    pad_px = _pad_pixels(height, width, size)
    padded = height < size or width < size
    if padded:
        canvas = np.full((max(height, size), max(width, size), 3), _PAD_VALUE, dtype=image.dtype)
        top = (canvas.shape[0] - height) // 2
        left = (canvas.shape[1] - width) // 2
        canvas[top : top + height, left : left + width] = image
        image = canvas
        height, width = image.shape[:2]

    cropped = height > size or width > size
    if cropped:
        top = (height - size) // 2
        left = (width - size) // 2
        image = image[top : top + size, left : left + size]
    return image, padded, cropped, pad_px


def _warn_resolution(mpp: float | None, target_mpp: float) -> None:
    """Raise the concerns a declared or undeclared resolution carries, once for a set of tiles."""
    if mpp is None:
        warnings.warn(
            f"{UNDECLARED_RESOLUTION_WARNING} "
            f"{INPUT_SIZE} by {INPUT_SIZE} pixels as they are, without matching their "
            f"physical scale to the {_format_mpp(target_mpp)} micron per pixel the encoder "
            "expects. "
            "Pass mpp= with the tiles' source resolution in microns per pixel.",
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
        return
    mpp = _checked_mpp(mpp, target_mpp=target_mpp)
    scale = _target_scale(mpp, target_mpp)
    if scale > 1.0:
        warnings.warn(
            f"Tiles are at {_format_mpp(mpp)} micron per pixel, coarser than the "
            f"{_format_mpp(target_mpp)} the "
            f"encoder reads, so reaching that geometry enlarges them by {scale:.2f}. "
            "Upsampling cannot recover image detail; the adapters were trained at the "
            "finer resolution.",
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )


def _prepare(
    tiles: np.ndarray | Sequence[np.ndarray],
    mpp: float | None,
    target_mpp: float,
    progress: bool | None,
) -> tuple[torch.Tensor, ResampleInfo]:
    """Prepare tiles as `prepare_tiles` does, raising none of its warnings."""
    if len(tiles) == 0:
        raise AlhenaValueError("No tiles were given.")
    scale = 1.0
    if mpp is not None:
        mpp = _checked_mpp(mpp, target_mpp=target_mpp)
        scale = _target_scale(mpp, target_mpp)

    lut = channel_lut()
    prepared: list[np.ndarray] = []
    padded_any = False
    cropped_any = False
    # One tile stands for the batch: the one that padded most, else cropped most, else the first.
    padded_most: tuple[int, tuple[int, int], tuple[int, int]] | None = None
    cropped_most: tuple[float, tuple[int, int], tuple[int, int]] | None = None

    with _progress.track(tiles, len(tiles), "Preparing tiles", "tile", progress) as items:
        for tile in items:
            if tile.ndim != 3 or tile.shape[2] != 3 or tile.dtype != np.uint8 or not tile.size:
                raise GeometryError(
                    "Every tile must be an (H, W, 3) uint8 RGB array of at least one pixel, got "
                    f"shape {tile.shape} of dtype {tile.dtype}."
                )
            arrived = (int(tile.shape[0]), int(tile.shape[1]))
            working = np.ascontiguousarray(tile)
            if mpp is None:
                if working.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
                    working = _resize(working, INPUT_SIZE, INPUT_SIZE)
                reached = arrived
                padded = cropped = False
                pad_px = 0
                crop = 0.0
            else:
                if scale != 1.0:
                    working = resample(working, scale)
                reached = (int(working.shape[0]), int(working.shape[1]))
                crop = _crop_fraction(*reached)
                working, padded, cropped, pad_px = fit_to_size(working, INPUT_SIZE)
            padded_any |= padded
            cropped_any |= cropped
            if padded_most is None or pad_px > padded_most[0]:
                padded_most = (pad_px, arrived, reached)
            if cropped_most is None or crop > cropped_most[0]:
                cropped_most = (crop, arrived, reached)
            prepared.append(normalize_tile(working, lut=lut))

    assert padded_most is not None and cropped_most is not None
    pad_px, source_size, resampled_size = padded_most
    if pad_px == 0:
        _, source_size, resampled_size = cropped_most
    batch = torch.from_numpy(np.stack(prepared, axis=0))
    info = ResampleInfo(
        source_mpp=mpp,
        target_mpp=target_mpp,
        scale=scale,
        source_size=source_size,
        resampled_size=resampled_size,
        pad_fraction=pad_px / float(INPUT_SIZE * INPUT_SIZE),
        crop_fraction=cropped_most[0],
        padded=padded_any,
        cropped=cropped_any,
        mpp_declared=mpp is not None,
    )
    return batch, info


def plan_resample(
    source_size: tuple[int, int],
    mpp: float | None = None,
    target_mpp: float = REFERENCE_MPP,
) -> ResampleInfo:
    """Return what `prepare_tiles` would do to tiles of a given size, touching no pixel.

    Every field agrees with the record `prepare_tiles` returns for tiles of that size. A scale
    leaving no pixels is reported as a canvas of pure pad, where `prepare_tiles` refuses it, and a
    resolution finer than `MIN_MICROSCOPE_MPP` is described without a warning.

    Args:
        source_size: Height and width of the tiles as they arrive.
        mpp: Microns per pixel of the tiles. Without one, tiles are resized to the input size as
            they are, so nothing is padded and nothing is cropped.
        target_mpp: Microns per pixel to resample to.

    Returns:
        The record `prepare_tiles` would return for tiles of that size.

    Raises:
        AlhenaValueError: If a resolution is not positive or exceeds `MAX_MICROSCOPE_MPP`.
    """
    target_mpp = _positive(target_mpp, "target_mpp")
    height, width = int(source_size[0]), int(source_size[1])
    if mpp is None:
        return ResampleInfo(
            source_mpp=None,
            target_mpp=target_mpp,
            scale=1.0,
            source_size=(height, width),
            resampled_size=(height, width),
            pad_fraction=0.0,
            crop_fraction=0.0,
            padded=False,
            cropped=False,
            mpp_declared=False,
        )
    mpp = _checked_mpp(mpp, target_mpp=target_mpp)
    scale = _target_scale(mpp, target_mpp)
    resampled = (round(height * scale), round(width * scale))
    return ResampleInfo(
        source_mpp=mpp,
        target_mpp=target_mpp,
        scale=scale,
        source_size=(height, width),
        resampled_size=resampled,
        pad_fraction=_pad_pixels(*resampled) / float(INPUT_SIZE * INPUT_SIZE),
        crop_fraction=_crop_fraction(*resampled),
        padded=min(resampled) < INPUT_SIZE,
        cropped=max(resampled) > INPUT_SIZE,
        mpp_declared=True,
    )


def _merge_records(records: Sequence[ResampleInfo]) -> ResampleInfo | None:
    """Return one record for tiles prepared in several groups, as one call would have made it."""
    if not records:
        return None
    # max keeps the first of equal candidates, so the record is the one a single call would keep.
    described = max(records, key=lambda record: record.pad_fraction)
    if described.pad_fraction == 0.0:
        described = max(records, key=lambda record: record.crop_fraction)
    return replace(
        described,
        crop_fraction=max(record.crop_fraction for record in records),
        padded=any(record.padded for record in records),
        cropped=any(record.cropped for record in records),
    )


def _crop_note(report: GeometryReport) -> str | None:
    """Return the concern a report raises about the center crop, or None where it raises none."""
    if report.mpp is None or report.crop_fraction <= CROP_WARNING_FRACTION:
        return None
    window = INPUT_SIZE * report.reference_mpp
    return (
        f"Tiles of {report.tile_size_px} pixels at {_format_mpp(report.mpp)} micron per pixel "
        "cover "
        f"{report.field_of_view_um:.1f} microns across, of which the {INPUT_SIZE} pixel canvas "
        f"holds the central {window:.1f}, so the center crop discards "
        f"{report.crop_fraction:.1%} of each field. A tile embedding describes that center "
        "alone, and a nucleus outside it reaches no token and is never read. "
        + _crop_advice(report.mpp, report.reference_mpp)
    )
