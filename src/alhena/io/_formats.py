"""File suffixes, dataset and column names, and other values the readers recognize."""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
"""Suffixes read as tile images."""

LABEL_SUFFIXES: tuple[str, ...] = (".png", ".tif", ".tiff", ".npy")
"""Suffixes read as integer label maps."""

HDF5_SUFFIXES: tuple[str, ...] = (".h5", ".hdf5")
"""Suffixes read as HDF5."""

TABLE_SUFFIXES: tuple[str, ...] = (".csv", ".tsv", ".txt", ".parquet", ".pq")
"""Suffixes read as tables, of centroids or of tile origins."""

GEOJSON_SUFFIXES: tuple[str, ...] = (".geojson", ".json")
"""Suffixes read as GeoJSON collections of polygons."""

SLIDE_SUFFIXES: tuple[str, ...] = (".svs", ".ndpi", ".scn", ".bif", ".tif", ".tiff", ".btf")
"""Suffixes of whole slides, read by `read_slide`: TIFF and the slide formats built on it."""

UNREAD_SLIDE_FORMATS: dict[str, str] = {
    ".mrxs": "MIRAX",
    ".vms": "Hamamatsu VMS",
    ".vmu": "Hamamatsu VMU",
    ".dcm": "DICOM",
    ".zvi": "Zeiss ZVI",
    ".czi": "Zeiss CZI",
}
"""Slide formats the slide reader cannot open, by suffix, refused before they are opened."""

EDGE_RULES: tuple[str, ...] = ("drop", "shift", "pad")
"""Rules for tiles cut by the image edge: leave out, shift inward, or pad with white."""

IMAGE_FRAME: str = "image, level 0"
"""Name of the frame of the tiles `read_image` cuts from an array: the pixels of that array."""

OME_LENGTH_UNITS: dict[str, float] = {
    "m": 1e6,
    "dm": 1e5,
    "cm": 1e4,
    "mm": 1e3,
    "µm": 1.0,
    "nm": 1e-3,
    "pm": 1e-6,
    "Å": 1e-4,
    "in": 25_400.0,
}
"""Microns in each unit an OME pixel size may be written in; no unit means micrometers."""

H5_IMAGE_DATASET: str = "img"
"""HDF5 dataset holding the tiles, an `(n_tiles, height, width, 3)` uint8 array."""
H5_COORDS_DATASET: str = "coords"
"""HDF5 dataset holding the top-left origin of each tile, `(n_tiles, 2)` integers, x then y."""
H5_BARCODE_DATASET: str = "barcode"
"""HDF5 dataset holding the name of each tile, an `(n_tiles, 1)` array of UTF-8 strings."""

H5_FRAME_SIDE_ATTRIBUTE: str = "patch_size_level0"
"""Attribute of `coords` recording the side of a tile in the pixels of the frame of `coords`."""
H5_TILE_SIDE_ATTRIBUTE: str = "patch_size"
"""Attribute of `coords` recording the side of a tile in the pixels of the tile."""

H5_SCALE_RECORDS: tuple[tuple[str, str, str], ...] = (
    (H5_COORDS_DATASET, H5_FRAME_SIDE_ATTRIBUTE, H5_TILE_SIDE_ATTRIBUTE),
    (H5_IMAGE_DATASET, "patch_size_src", "patch_size_target"),
)
"""Where an HDF5 file records its tile side: dataset, side in the frame, side in the tile."""

H5_LABEL_DATASET: str = "labels"
"""HDF5 dataset holding the label maps of every tile, stacked into one array."""

CELL_ID_STRIDE: int = 100_000
"""Number of identifiers in the block each tile owns when `global_cell_ids` renumbers."""

NUCLEI_KINDS: tuple[str, ...] = ("label_maps", "centroids", "polygons")
"""Kinds of nucleus input `read_nuclei` dispatches to."""

CENTROID_FRAMES: tuple[str, ...] = ("tile", "origins")
"""Frames a table of centroids is read in: its tile's pixels, or the frame of the tile origins."""

COORDINATE_UNITS: tuple[str, ...] = ("px", "um")
"""Units nucleus coordinates are read in: pixels of their frame, or microns."""

MULTIPART_RULES: tuple[str, ...] = ("refuse", "split", "largest")
"""Rules for a boundary of several polygons: refuse it, split it, or keep its largest polygon."""

PART_SEPARATOR: str = ":"
"""Separator between a boundary's identifier and a polygon's position, under `"split"`."""

BOUNDARY_TABLE_SUFFIXES: tuple[str, ...] = TABLE_SUFFIXES
"""Suffixes of boundary files read as tables of vertices or GeoParquet; others are GeoJSON."""

TRANSFORM_SUFFIXES: tuple[str, ...] = (".csv", ".npy")
"""Suffixes an affine matrix is read from: comma-separated numbers, or a NumPy array."""

MPP_X_PROPERTIES: tuple[str, ...] = ("aperio.MPP", "tiffslide.mpp-x", "openslide.mpp-x")
"""Slide properties read, in order, for the microns per pixel of level zero across."""
MPP_Y_PROPERTIES: tuple[str, ...] = ("aperio.MPP", "tiffslide.mpp-y", "openslide.mpp-y")
"""Slide properties read, in order, for the microns per pixel of level zero down."""

CENTROID_COLUMNS: dict[str, tuple[str, ...]] = {
    "x": ("x", "centroid_x", "x_centroid"),
    "y": ("y", "centroid_y", "y_centroid"),
    "tile_index": ("tile_index", "tile", "patch_index"),
    "cell_id": ("cell_id", "cell", "id"),
    "radius_px": ("radius_px", "radius"),
}
"""Column names tried for each field of a table of centroids, unless `columns` names one."""

VERTEX_COLUMNS: dict[str, tuple[str, ...]] = {
    "cell_id": ("cell_id", "cell", "id"),
    "x": ("vertex_x", "x"),
    "y": ("vertex_y", "y"),
}
"""Column names tried for each field of a table of vertices, unless `columns` names one."""

ORIGIN_COLUMNS: dict[str, tuple[str, ...]] = {
    "tile_id": ("tile_id", "tile"),
    "x": ("x",),
    "y": ("y",),
}
"""Column names tried for each field of a table of tile origins, unless `columns` names one."""


def _unread_format_message(path: Path) -> str:
    """Return the refusal of a slide in one of `UNREAD_SLIDE_FORMATS`."""
    return (
        f"{path.name} is a slide in the {UNREAD_SLIDE_FORMATS[path.suffix.lower()]} format, which "
        "the slide reader, tiffslide, cannot open. Export it to a pyramidal TIFF, with mpp= if the "
        "export carries no resolution."
    )
