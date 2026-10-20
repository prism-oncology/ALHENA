"""Nuclei read from the shapes of a SpatialData object."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alhena._errors import AlhenaValueError, MembershipError
from alhena._frames import _as_tile_scale, _windows_inside
from alhena._geometry import _checked_mpp
from alhena.cells._polygons import _polygons_membership
from alhena.cells._table import _ATTR_FRAME
from alhena.io import MULTIPART_RULES
from alhena.io._files import _choice
from alhena.io._polygons import (
    _check_boundaries_meet,
    _geometry_area,
    _geometry_parts,
    _read_geometry_parts,
    _select_parts,
    _warn_holes,
)
from alhena.spatial._elements import (
    _carrier,
    _declares_frames,
    _Frame,
    _frame,
    _placements,
    _read,
    _same,
    _shapes,
)
from alhena.spatial._tiles import _level_zero
from alhena.spatial._windows import PATCHES_KEY, _array_windows, _element_windows


def cells_from_shapes(
    sdata: Any,
    shapes_key: str,
    *,
    patches: str | np.ndarray = PATCHES_KEY,
    coordinate_system: str | None = None,
    image_key: str | None = None,
    multipart: str = "refuse",
    mpp: float | None = None,
    tile_scale: float = 1.0,
    progress: bool | None = None,
) -> pd.DataFrame:
    """Build a token membership table from nucleus boundaries held in a SpatialData object.

    Each boundary is rasterized into the window of every tile it falls in, and carried onto the
    canvas as the image of the tile is. The boundaries and the windows are read in the level-zero
    pixels of `image_key`, else in `coordinate_system` or as stored; windows given as an array are
    refused where an image sits in that frame other than by the identity. `cells_to_spatialdata`
    cannot write the polygons `multipart="split"` yields, which are not instances of the element.

    Args:
        sdata: The object holding the boundaries and the tile windows.
        shapes_key: Shapes element holding one polygon per nucleus, indexed by cell identifier.
        patches: Shapes element holding one box per tile, read as `patch_bounds` reads it, or an
            `(n_tiles, 4)` array of windows.
        coordinate_system: Coordinate system both elements are read in, or carried through into the
            pixels of `image_key`.
        image_key: Image the tiles are read from, in whose level-zero pixels both elements are read.
        multipart: One of `alhena.io.MULTIPART_RULES`, applied as `alhena.io.read_polygons` applies
            it.
        mpp: Microns per pixel of the tiles, as the call annotating the cells declares. Without it,
            each raster is stretched onto the whole canvas.
        tile_scale: Pixels of the frame one tile pixel spans, the `coords_scale` of the tiles.
        progress: Whether to report progress on standard error. None follows
            `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        A table with `MEMBERSHIP_COLUMNS`, recording the windows and the resolution, with positions
        in the frame the boundaries were read in, named such as `image 'he', level 0`,
        `coordinate system 'global'` or `polygons`.

    Raises:
        AlhenaValueError: If an element is missing or empty, or no frame holds both elements.
        MembershipError: If the boundaries and the windows are placed in different frames.
        MissingExtraError: If the spatial extra is not installed.
    """
    multipart = _choice(multipart, MULTIPART_RULES, "multipart")
    if mpp is not None:
        mpp = _checked_mpp(mpp, warn=True)
    frame = _frame(sdata, coordinate_system, image_key)
    stored = _shapes(sdata, shapes_key)
    if len(stored.geometry) == 0:
        raise AlhenaValueError(f"The element {shapes_key!r} holds no nucleus boundary.")

    if isinstance(patches, str):
        element = _shapes(sdata, patches)
        _check_stored_frames(sdata, stored, shapes_key, element, patches, frame)
        windows = _element_windows(sdata, element, patches, frame)
    else:
        windows = _array_windows(patches)
        _check_array_frame(sdata, stored, shapes_key, frame)

    boundaries = _read(stored, frame, _carrier(sdata, stored, shapes_key, frame))
    vertices, identifiers = _geometry_vertices(
        boundaries.geometry,
        np.asarray(boundaries.index.to_numpy()),
        shapes_key,
        multipart,
        progress=progress,
    )
    _check_boundaries_meet(
        vertices,
        windows,
        repr(shapes_key),
        f"in {frame.described}",
        "Pass image_key naming the image the tiles are read from, which reads both in its pixels, "
        "or name the coordinate system both are read in.",
    )

    # A window past the edge of the image is a tile padded white there, showing no nucleus.
    inside = None
    if frame.image is not None:
        shape = _level_zero(frame.image, str(frame.image_key)).shape
        inside = _windows_inside(
            windows, (int(shape[-1]), int(shape[-2])), _as_tile_scale(tile_scale)
        )
    membership = _polygons_membership(
        vertices,
        windows,
        cell_ids=identifiers,
        mpp=mpp,
        progress=progress,
        tile_scale=tile_scale,
        inside=inside,
    )
    if frame.named is not None:
        # The positions are in the frame the boundaries were read in, which the call names.
        membership.attrs[_ATTR_FRAME] = frame.named
    return membership


def _geometry_vertices(
    geometries: Any,
    identifiers: np.ndarray,
    described: str,
    multipart: str,
    progress: bool | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return the outline of every nucleus, read as `alhena.io.read_polygons` reads GeoParquet."""
    named = repr(described)
    parts = _geometry_parts(geometries, len(geometries), progress)
    kept, cells = _select_parts(parts, identifiers, multipart, named, _geometry_area)
    polygons = _read_geometry_parts(kept)
    _warn_holes(polygons, named)
    return [outline for outline, _ in polygons], cells


def _check_array_frame(sdata: Any, boundaries: Any, name: str, frame: _Frame) -> None:
    """Refuse windows given as an array where the object places an image in another frame."""
    if frame.image is not None or not _declares_frames(sdata):
        return
    images = getattr(sdata, "images", {})
    elsewhere: list[str] = []
    if frame.coordinate_system is not None:
        for key, image in images.items():
            placed = _placements(image)
            if frame.coordinate_system in placed and not _same(
                placed[frame.coordinate_system], np.eye(3)
            ):
                elsewhere.append(str(key))
        placed_by = "there by a transformation other than the identity"
    else:
        own = _placements(boundaries) if len(images) else {}
        for key, image in images.items():
            placed = _placements(image)
            if any(not _same(placed[system], own[system]) for system in set(placed) & set(own)):
                elsewhere.append(str(key))
        placed_by = "otherwise than those boundaries"
    if not elsewhere:
        return
    elsewhere.sort()
    named = f"the image {elsewhere[0]!r}" if len(elsewhere) == 1 else f"the images {elsewhere}"
    source = "its file" if len(elsewhere) == 1 else "the file of one of them"
    raise MembershipError(
        "The tile windows are an array, taken to be in the frame the boundaries of "
        f"{name!r} are read in, {frame.described}. The object places {named} {placed_by}, so "
        f"windows cut from {source} lie elsewhere: pass image_key naming the image the tiles are "
        "read from, or give the windows as a shapes element the object places."
    )


def _check_stored_frames(
    sdata: Any, boundaries: Any, name: str, patch_element: Any, patches: str, frame: _Frame
) -> None:
    """Refuse boundaries and windows read as stored where the object places them differently."""
    if not frame.stored or not _declares_frames(sdata):
        return
    own = _placements(boundaries)
    theirs = _placements(patch_element)
    differ = sorted(
        system for system in set(own) & set(theirs) if not _same(own[system], theirs[system])
    )
    if differ:
        raise MembershipError(
            f"The boundaries of {name!r} and the windows of {patches!r} are read as they are "
            f"stored, and the object places the two differently in {differ}. Pass image_key "
            "naming the image the tiles are read from, or coordinate_system naming one both are "
            "placed in."
        )
