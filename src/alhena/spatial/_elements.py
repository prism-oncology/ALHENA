"""Elements of a SpatialData object, and the transformations that place them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from alhena._errors import AlhenaValueError, MissingExtraError
from alhena._layout import CELL_ID_COLUMN
from alhena.io._carry import _invertible

_SPATIAL_EXTRA: Final[str] = 'pip install "alhena[spatial]"'

CELL_TABLE_KEY: Final[str] = "alhena_cells"
"""Name the table of cells is stored under by default."""
REGION_KEY: Final[str] = "region"
"""Column of the table of cells naming the element it annotates, by default."""
INSTANCE_KEY: Final[str] = CELL_ID_COLUMN
"""Column of the table of cells holding the instance identifiers: the cell identifiers."""

# Element groups of a SpatialData object, in the order an element is looked up.
_GROUPS: Final[tuple[str, ...]] = ("images", "labels", "points", "shapes", "tables")

# Axes a transformation is read on: the plane boundaries and tile windows lie in.
_PLANE: Final[tuple[str, str]] = ("x", "y")

# Matrices whose entries agree within this, absolutely or relatively, are one transformation.
_SAME_TRANSFORMATION: Final[float] = 1e-9


def _require(module: str) -> Any:
    """Import one optional dependency of the spatial extra."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise MissingExtraError(
            f"{module} is needed to read ALHENA through a SpatialData object and is not "
            f"installed. Install it with {_SPATIAL_EXTRA}."
        ) from error


@dataclass(frozen=True)
class _Frame:
    """The frame a call reads nucleus boundaries and tile windows in."""

    # The coordinate system read in, or carried through into the image's pixels; None as stored.
    coordinate_system: str | None = None
    image_key: str | None = None
    image: Any = None

    @property
    def stored(self) -> bool:
        """Whether every element is read in the frame it is stored in."""
        return self.coordinate_system is None and self.image_key is None

    @property
    def described(self) -> str:
        """The frame, as a message names it."""
        if self.image_key is not None:
            return f"the pixels of the image {self.image_key!r}"
        if self.coordinate_system is not None:
            return f"the coordinate system {self.coordinate_system!r}"
        return "the frame they are stored in"

    @property
    def named(self) -> str | None:
        """The frame, as the positions read in it record it, or None for the stored frame."""
        if self.image_key is not None:
            return _image_element_frame(self.image_key)
        if self.coordinate_system is not None:
            return f"coordinate system {self.coordinate_system!r}"
        return None


def _image_element_frame(image_key: str) -> str:
    """Return the name of the frame of the level-zero pixels of an image of an object."""
    return f"image {image_key!r}, level 0"


def _frame(sdata: Any, coordinate_system: str | None, image_key: str | None) -> _Frame:
    """Return the frame a call names, refusing an image key that names no image."""
    if image_key is None:
        return _Frame(coordinate_system=coordinate_system)
    images = getattr(sdata, "images", {})
    if image_key not in images:
        group = next((name for name in _GROUPS if image_key in getattr(sdata, name, {})), None)
        found = (
            "names no element of this object"
            if group is None
            else f"this object holds among its {group} and not among its images"
        )
        held = sorted(str(key) for key in images)
        images_held = f"the images of this object are {held}" if held else "this object holds none"
        raise AlhenaValueError(
            f"image_key is {image_key!r}, which {found}. It names the image the tiles are read "
            f"from, and {images_held}."
        )
    return _Frame(coordinate_system, str(image_key), images[image_key])


def _declares_frames(sdata: Any) -> bool:
    """Return whether an object declares coordinate systems; one declaring none is read as given."""
    return bool(getattr(sdata, "coordinate_systems", None))


def _placements(element: Any) -> dict[str, np.ndarray]:
    """Return the matrix placing an element's own coordinates in each of its coordinate systems."""
    transformations = _require("spatialdata.transformations")
    placed = transformations.get_transformation(element, get_all=True)
    return {
        str(system): np.asarray(
            transformation.to_affine_matrix(input_axes=_PLANE, output_axes=_PLANE),
            dtype=np.float64,
        )
        for system, transformation in placed.items()
    }


def _placement_in(placed: dict[str, np.ndarray], described: str, system: str) -> np.ndarray:
    """Return the placement in one coordinate system, refusing an element placed elsewhere."""
    if system not in placed:
        raise AlhenaValueError(
            f"{described} is placed in the coordinate systems {sorted(placed)} and not in "
            f"{system!r}. Name a coordinate system it is placed in."
        )
    return placed[system]


def _same(first: np.ndarray, second: np.ndarray) -> bool:
    """Return whether two affine matrices are one transformation, up to rounding."""
    return bool(np.allclose(first, second, rtol=_SAME_TRANSFORMATION, atol=_SAME_TRANSFORMATION))


def _onto_image(sdata: Any, element: Any, name: str, frame: _Frame) -> np.ndarray:
    """Return the matrix carrying an element into the level-zero pixels of an image."""
    key = frame.image_key
    if not _declares_frames(sdata):
        raise AlhenaValueError(
            f"This object declares no coordinate system, so nothing in it places {name!r} "
            f"against the image {key!r}. Read both from a SpatialData object, whose elements "
            "carry the transformations that place them."
        )
    own = _placements(element)
    image = _placements(frame.image)
    if frame.coordinate_system is not None:
        through = [frame.coordinate_system]
        _placement_in(own, repr(name), frame.coordinate_system)
        _placement_in(image, f"The image {key!r}", frame.coordinate_system)
    else:
        through = sorted(set(own) & set(image))
        if not through:
            raise AlhenaValueError(
                f"{name!r} is placed in the coordinate systems {sorted(own)} and the image "
                f"{key!r} in {sorted(image)}. No coordinate system places both: place them "
                "in one."
            )

    carried: dict[str, np.ndarray] = {}
    for system in through:
        for described, matrix in ((repr(name), own[system]), (f"The image {key!r}", image[system])):
            if not _invertible(matrix):
                raise AlhenaValueError(
                    f"{described} is placed in {system!r} by a transformation that collapses the "
                    f"plane, so {name!r} cannot be carried onto the pixels of the image {key!r} "
                    "through it."
                )
        carried[system] = np.linalg.solve(image[system], own[system])
    first = carried[through[0]]
    if any(not _same(carried[system], first) for system in through[1:]):
        raise AlhenaValueError(
            f"{name!r} and the image {key!r} are both placed in {through}, and these carry "
            f"{name!r} onto different pixels of the image. Pass coordinate_system naming the one "
            "to carry it through."
        )
    return first


def _carrier(sdata: Any, element: Any, name: str, frame: _Frame) -> np.ndarray | None:
    """Return the matrix carrying an element into the frame named, or None where read as stored."""
    if frame.image is not None:
        return _onto_image(sdata, element, name, frame)
    if frame.coordinate_system is None or not _declares_frames(sdata):
        return None
    return _placement_in(_placements(element), repr(name), frame.coordinate_system)


def _read(element: Any, frame: _Frame, carrier: np.ndarray | None) -> Any:
    """Return a shapes element with its coordinates in the frame named."""
    if frame.image is not None and carrier is not None:
        return _moved(element, carrier)
    if frame.coordinate_system is not None:
        spatialdata = _require("spatialdata")
        return spatialdata.transform(element, to_coordinate_system=frame.coordinate_system)
    return element


def _moved(element: Any, matrix: np.ndarray) -> Any:
    """Return a shapes element with every coordinate carried by an affine matrix."""
    if _same(matrix, np.eye(3)):
        return element
    spatialdata = _require("spatialdata")
    transformations = _require("spatialdata.transformations")
    affine = transformations.Affine(matrix, input_axes=_PLANE, output_axes=_PLANE)
    # Applied with the placements kept; only the coordinates are read afterwards.
    return spatialdata.transform(element, transformation=affine, maintain_positioning=True)


def _element(sdata: Any, name: str) -> Any:
    """Return one element of a SpatialData object, naming what it holds when it has no such key."""
    try:
        return sdata[name]
    except KeyError as error:
        available: list[str] = []
        for group in _GROUPS:
            available.extend(str(key) for key in getattr(sdata, group, {}))
        raise AlhenaValueError(
            f"{name!r} names no element of this object, which holds {sorted(available)}."
        ) from error


def _shapes(sdata: Any, name: str) -> Any:
    """Return a shapes element, refusing an element that holds no geometry."""
    element = _element(sdata, name)
    if not hasattr(element, "geometry") or not hasattr(element, "bounds"):
        raise AlhenaValueError(
            f"The element {name!r} is of type {type(element).__name__} and holds no geometry, "
            "where a shapes element of boundaries or of tile windows is expected."
        )
    return element
