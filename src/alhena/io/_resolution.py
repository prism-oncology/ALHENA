"""The resolution a slide declares, read from its properties, TIFF tags or OME metadata."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple
from xml.etree import ElementTree

from alhena._constants import MAX_MICROSCOPE_MPP, MIN_MICROSCOPE_MPP, MPP_TOLERANCE, REFERENCE_MPP
from alhena._errors import PACKAGE_PREFIXES, GeometryError, GeometryWarning
from alhena._geometry import _checked_mpp
from alhena.io._formats import MPP_X_PROPERTIES, MPP_Y_PROPERTIES, OME_LENGTH_UNITS
from alhena.io._sources import _DECLARED_MPP_SOURCE, _OME_MPP_SOURCE, _TAGS_MPP_SOURCE

# Other spellings of the micrometer, each read as the symbol the OME schema writes.
_MICROMETER_SPELLINGS: dict[str, str] = {"μm": "µm", "um": "µm"}

# Properties holding the TIFF resolution tags of level zero, and the microns in each unit.
_RESOLUTION_TAG_PROPERTIES: tuple[str, str, str] = (
    "tiff.XResolution",
    "tiff.YResolution",
    "tiff.ResolutionUnit",
)
_RESOLUTION_UNIT_MICRONS: dict[str, float] = {
    "INCH": 25_400.0,
    "CENTIMETER": 10_000.0,
    "MILLIMETER": 1_000.0,
    "MICROMETER": 1.0,
}

# Properties a slide reader derives from the TIFF tags: one holding their figure is read as them.
_TAG_DERIVED_PROPERTIES: frozenset[str] = frozenset({"tiffslide.mpp-x", "openslide.mpp-x"})


class _Declaration(NamedTuple):
    """A resolution a slide declares for its level zero, and where it declares it."""

    across: float
    # Microns per pixel down, or None where the source gives one figure only.
    down: float | None
    # The source as `SlideTileSource.mpp_source` names it.
    source: str
    # The source as a message names it.
    where: str
    # What the source holds, where that is not the figure itself.
    written: str = ""


def _resolve_slide_mpp(slide: Any, path: Path, declared: float | None) -> tuple[float, str]:
    """Return the microns per pixel of level zero of a slide, and where they were read from."""
    if declared is not None:
        value = _checked_mpp(declared, warn=True)
        try:
            own = _slide_resolution(slide, path)
        except GeometryError:
            own = None
        if (
            own is not None
            and _plausible(own)
            and abs(value - own.across) > MPP_TOLERANCE * own.across
        ):
            warnings.warn(
                f"{path.name} declares {own.across:.4g} micron per pixel {own.where}, and "
                f"mpp={value:g} was passed: the two differ by more than {MPP_TOLERANCE:.0%}, and "
                "the value passed is used.",
                GeometryWarning,
                skip_file_prefixes=PACKAGE_PREFIXES,
            )
        return value, _DECLARED_MPP_SOURCE

    own = _slide_resolution(slide, path)
    if own is None:
        keys = MPP_X_PROPERTIES
        raise GeometryError(
            f"{path.name} declares no resolution: no OME PhysicalSizeX describes its "
            f"full-resolution level, none of {', '.join(keys[:-1])} or {keys[-1]} is set, and no "
            "TIFF resolution tags give one in a unit of length. Pass mpp= to declare the "
            "resolution of its full-resolution level."
        )
    _check_plausible(own, path)
    if own.down is not None and abs(own.down - own.across) > MPP_TOLERANCE * own.across:
        warnings.warn(
            f"{path.name} declares {own.across:.4g} micron per pixel across and {own.down:.4g} "
            f"down {own.where}. Tiles are resampled by the value across in both directions, so "
            "their vertical axis is off by the ratio of the two.",
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
    return own.across, own.source


def _slide_resolution(slide: Any, path: Path) -> _Declaration | None:
    """Return the resolution a slide declares for level zero: OME, then properties, then tags."""
    declared = _ome_resolution(slide, path)
    if declared is not None:
        return declared
    properties = dict(getattr(slide, "properties", {}) or {})
    tags = _tag_resolution(properties)
    for across_key, down_key in zip(MPP_X_PROPERTIES, MPP_Y_PROPERTIES, strict=True):
        across = _property_mpp(properties, across_key, path)
        if across is None:
            continue
        derived = across_key in _TAG_DERIVED_PROPERTIES
        if derived and tags is not None and math.isclose(across, tags.across, rel_tol=1e-9):
            continue
        down = _property_mpp(properties, down_key, path)
        return _Declaration(across, down, across_key, f"under {across_key}")
    return tags


def _property_mpp(properties: Mapping[str, Any], key: str, path: Path) -> float | None:
    """Return the microns per pixel a slide property declares, or None where it is not set."""
    value = properties.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    number = _positive_number(value)
    if number is None:
        raise GeometryError(
            f"{path.name} sets {key} to {value!r}, which is not a positive number of microns per "
            "pixel. Pass mpp= to declare the resolution of its full-resolution level."
        )
    return number


def _positive_number(value: Any) -> float | None:
    """Return a value as a finite positive number, or None where it is not one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _tag_resolution(properties: Mapping[str, Any]) -> _Declaration | None:
    """Return the resolution the TIFF tags of level zero give, or None without a unit of length."""
    across_key, down_key, unit_key = _RESOLUTION_TAG_PROPERTIES
    unit = str(properties.get(unit_key, "")).upper()
    microns = _RESOLUTION_UNIT_MICRONS.get(unit)
    across = _positive_number(properties.get(across_key))
    if microns is None or across is None:
        return None
    down = _positive_number(properties.get(down_key))
    return _Declaration(
        microns / across,
        None if down is None else microns / down,
        _TAGS_MPP_SOURCE,
        "in its TIFF resolution tags",
        f"XResolution {across:g} pixels per {unit.lower()}",
    )


def _ome_resolution(slide: Any, path: Path) -> _Declaration | None:
    """Return the resolution the OME images of the size of level zero declare, or None."""
    text = getattr(getattr(slide, "ts_tifffile", None), "ome_metadata", None)
    if not text:
        return None
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise GeometryError(
            f"{path.name} carries OME metadata that does not parse as XML ({error}). Pass mpp= "
            "to declare the resolution of its full-resolution level."
        ) from error
    width, height = (int(value) for value in slide.level_dimensions[0])
    found: list[_Declaration] = []
    for image in root:
        if _local_name(image.tag) != "Image":
            continue
        for pixels in image:
            if _local_name(pixels.tag) != "Pixels" or _ome_size(pixels) != (width, height):
                continue
            declared = _ome_pixel_size(pixels, path)
            if declared is not None:
                found.append(declared)
    if not found:
        return None
    first = found[0]
    if any(abs(other.across - first.across) > MPP_TOLERANCE * first.across for other in found):
        sizes = ", ".join(sorted({f"{declared.across:.4g}" for declared in found}))
        raise GeometryError(
            f"{path.name} carries OME metadata describing {len(found)} images of {width} by "
            f"{height} pixels, the size of its full-resolution level, at different pixel sizes: "
            f"{sizes} micron. Pass mpp= to declare the resolution of its full-resolution level."
        )
    return first


def _local_name(tag: str) -> str:
    """Return the name of an XML element without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _ome_size(pixels: ElementTree.Element) -> tuple[int, int] | None:
    """Return the width and height an OME `Pixels` element gives, or None where it gives none."""
    try:
        return int(pixels.get("SizeX", "")), int(pixels.get("SizeY", ""))
    except ValueError:
        return None


def _ome_pixel_size(pixels: ElementTree.Element, path: Path) -> _Declaration | None:
    """Return the pixel size an OME `Pixels` element declares, or None where it declares none."""
    across = _ome_length(pixels, "X", path)
    if across is None:
        return None
    unit = pixels.get("PhysicalSizeXUnit", "µm")
    return _Declaration(
        across,
        _ome_length(pixels, "Y", path),
        _OME_MPP_SOURCE,
        "in its OME metadata",
        f"PhysicalSizeX {pixels.get('PhysicalSizeX')} {unit}",
    )


def _ome_length(pixels: ElementTree.Element, axis: str, path: Path) -> float | None:
    """Return the size of a pixel along one axis in microns, or None where it is not given."""
    name = f"PhysicalSize{axis}"
    value = pixels.get(name)
    if value is None:
        return None
    unit = pixels.get(f"{name}Unit", "µm")
    microns = OME_LENGTH_UNITS.get(_MICROMETER_SPELLINGS.get(unit, unit))
    if microns is None:
        units = list(OME_LENGTH_UNITS)
        raise GeometryError(
            f"{path.name} gives {name} in {unit!r}, where a pixel size is read in a unit of "
            f"length, {', '.join(units[:-1])} or {units[-1]}. Pass mpp= to declare the "
            "resolution of its full-resolution level."
        )
    number = _positive_number(value)
    if number is None:
        raise GeometryError(
            f"{path.name} gives {name} as {value!r}, which is not a positive length. Pass mpp= "
            "to declare the resolution of its full-resolution level."
        )
    return number * microns


def _plausible(declared: _Declaration) -> bool:
    """Report whether a resolution lies in the range of a microscope image along both axes."""
    return all(
        MIN_MICROSCOPE_MPP <= value <= MAX_MICROSCOPE_MPP
        for value in (declared.across, declared.down)
        if value is not None
    )


def _check_plausible(declared: _Declaration, path: Path) -> None:
    """Refuse a resolution a slide declares outside the range of a microscope image."""
    if _plausible(declared):
        return
    figures = [value for value in (declared.across, declared.down) if value is not None]
    outside = next(
        value for value in figures if not MIN_MICROSCOPE_MPP <= value <= MAX_MICROSCOPE_MPP
    )
    figure = f"{declared.across:.4g} micron per pixel"
    if declared.down is not None and not math.isclose(declared.down, declared.across):
        figure += f" across and {declared.down:.4g} down"
    change = "enlarge" if outside > REFERENCE_MPP else "shrink"
    factor = max(outside / REFERENCE_MPP, REFERENCE_MPP / outside)
    written = f", {declared.written}" if declared.written else ""
    default = ""
    if declared.source == _TAGS_MPP_SOURCE:
        default = (
            " Resolution tags often hold a default of the program that wrote the file, a "
            "screen's or a printer's dots per inch, rather than a calibration."
        )
    raise GeometryError(
        f"{path.name} declares {figure} {declared.where}{written}, outside the "
        f"{MIN_MICROSCOPE_MPP:g} to {MAX_MICROSCOPE_MPP:g} micron per pixel of a microscope "
        f"image: reaching the {REFERENCE_MPP} micron per pixel the encoder reads would {change} "
        f"each tile {factor:.0f} times.{default} Pass mpp= to declare the resolution of its "
        "full-resolution level."
    )
