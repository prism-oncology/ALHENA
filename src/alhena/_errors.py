"""Exception and warning types raised by ALHENA."""

from __future__ import annotations

import os
from pathlib import Path

# Passed as skip_file_prefixes, so a warning points at the first line outside the package.
PACKAGE_PREFIXES: tuple[str, ...] = (f"{Path(__file__).parent}{os.sep}",)


class AlhenaError(Exception):
    """Base class for every error raised by ALHENA."""


class AlhenaValueError(AlhenaError, ValueError):
    """An argument value is outside the range the call accepts.

    Subclasses `ValueError` as well, so a caller may catch either.
    """


class BundleError(AlhenaError):
    """The encoder bundle is missing, incomplete or inconsistent with the backbone."""


class BackboneNotFoundError(AlhenaError):
    """The UNI2-h weights could not be located."""


class ReferenceError(AlhenaError):
    """A reference shard is missing, malformed or unusable for the requested axis."""


class CellSpaceMismatchError(AlhenaError):
    """The encoder and the reference were produced by different training runs.

    Their cell projections are not aligned, though the vector widths agree.
    """


class GeometryError(AlhenaError):
    """Tile geometry cannot be reconciled with the geometry the model reads."""


class MembershipError(AlhenaError):
    """The nucleus-to-token membership table is malformed or inconsistent with the tiles."""


class MissingExtraError(AlhenaError, ImportError):
    """An optional dependency is not installed. The message names the extra that installs it.

    Subclasses `ImportError` as well, so a caller may catch either.
    """


class AlhenaWarning(UserWarning):
    """Base class for warnings that change how a result should be read."""


class GeometryWarning(AlhenaWarning):
    """The geometry tiles present to the model changes how a result reads.

    Raised for an undeclared or differing resolution, white pad, a center crop, nuclei spanning
    more tokens than at the distillation resolution, nuclei drawn as disks around their centroids,
    an identifier naming nuclei apart, and a label map an overlay draws as the map of tile 0.
    """


class ReferenceCoverageWarning(AlhenaWarning):
    """Some classes of the axis have no cell in the reference, or are thinly supported."""


class EnvelopeWarning(AlhenaWarning):
    """Retrieval is running outside the declared envelope of its hyperparameters."""
