"""The program name, and the library's messages put in the words of the command line."""

from __future__ import annotations

import re

PROGRAM: str = "alhena"
"""Name the command is installed under."""

# Library phrases naming a reader, constant or call the command line has no use for, reworded.
_LIBRARY_PHRASES: tuple[tuple[str, str], ...] = (
    (
        "is read by read_polygons, or by read_nuclei with kind='polygons'",
        "is read with --nuclei-kind polygons",
    ),
    ("under the aliases TISSUE_ALIASES declares", "under other names"),
    (
        "under the aliases alhena.TISSUE_ALIASES declares",
        f"under the other names `{PROGRAM} reference list` prints",
    ),
    ("is read one region at a time by read_slide", "is read one region at a time as a slide"),
    (
        "were not cut from one image by read_slide or read_image",
        "were not cut from one image, a slide or an image file named as the tiles",
    ),
    (
        "take one label map each, which read_label_maps reads",
        "take one label map each, from a directory, an HDF5 file or a .npy array of them",
    ),
    (
        "A stack of one label map per tile is read by read_label_maps.",
        "A stack of one label map per tile is read from a .npy array of three dimensions.",
    ),
    ("which read_label_image cuts along the tiles", "which is cut along the tiles"),
    (
        "make them unique first: io.read_label_maps renumbers them",
        "drop --nuclei-ids cohort, which keeps them as they are",
    ),
    (
        "positions with a radius are read by alhena.cells.membership_from_centroids, which "
        "reports its own approximation",
        "positions with a radius are read from a table of centroids",
    ),
    ("or pass coords= when building the source", "or give their origins with --coords"),
    (
        "or give them coords=, for instance from read_tile_origins",
        "or give their origins with --coords",
    ),
    (", or read the tiles with coords_dataset=None", ""),
    (
        "Pass barcode_dataset=None to read the tiles without their names.",
        "Name the dataset holding their names with --h5-datasets barcode=NAME.",
    ),
    (
        "Pass coords_dataset=None to read the tiles without their origins.",
        "Name the dataset holding their origins with --h5-datasets coords=NAME.",
    ),
    ("with image_dataset=.", "with --h5-datasets img=NAME."),
    ("with coords_dataset=.", "with --h5-datasets coords=NAME."),
    (
        "Write cells with CellEmbeddings.write_h5ad, or annotations with "
        "AnnotationResult.write_h5ad.",
        f"`{PROGRAM} encode --nuclei` writes one, and so does `{PROGRAM} annotate` to a .h5ad "
        "file.",
    ),
    ("transform carries coordinates", "--nuclei-transform carries coordinates"),
    ("leave units at 'px'", "leave --nuclei-units at px"),
)

# Library arguments by the name messages give them, and the option passing each on.
_LIBRARY_ARGUMENTS: dict[str, str] = {
    "mpp": "--mpp",
    "tile_px": "--tile-px",
    "stride_px": "--stride-px",
    "edge": "--edge",
    "coords": "--coords",
    "coords_scale": "--coords-scale",
    "radius_px": "--radius-px",
    "frame": "--nuclei-frame",
    "units": "--nuclei-units",
    "transform": "--nuclei-transform",
    "kind": "--nuclei-kind",
    "multipart": "--nuclei-multipart",
    "id_property": "--nuclei-id-property",
    "geometry_property": "--nuclei-geometry-property",
    "parquet": "--parquet",
    "batch_size": "--batch-size",
}

# Arguments read as their option even alone: none is an English word or part of a quoted name.
_BARE_ARGUMENTS: tuple[str, ...] = (
    "mpp",
    "tile_px",
    "stride_px",
    "coords_scale",
    "radius_px",
    "id_property",
    "geometry_property",
    "batch_size",
)

_ARGUMENT_WITH_VALUE = re.compile(
    rf"(?<![\w.-])({'|'.join(_LIBRARY_ARGUMENTS)})="
    r"(?:'([^']*)'|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?))?"
)
_BARE_ARGUMENT = re.compile(rf"(?<![\w.-])({'|'.join(_BARE_ARGUMENTS)})(?![\w=-])")


def _as_command_line(text: str) -> str:
    """Return a library message in the words of the command line: phrases first, then arguments."""
    for phrase, wording in _LIBRARY_PHRASES:
        text = text.replace(phrase, wording)

    def option(match: re.Match[str]) -> str:
        value = match.group(2) if match.group(2) is not None else match.group(3)
        flag = _LIBRARY_ARGUMENTS[match.group(1)]
        return flag if value is None else f"{flag} {value}"

    text = _ARGUMENT_WITH_VALUE.sub(option, text)
    return _BARE_ARGUMENT.sub(lambda match: _LIBRARY_ARGUMENTS[match.group(1)], text)
