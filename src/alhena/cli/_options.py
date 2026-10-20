"""Options several commands share."""

from __future__ import annotations

import argparse
from pathlib import Path

from alhena._commands import _add_cache_option
from alhena._constants import CLASS_AXES, DEFAULT_AXIS
from alhena.cli._reading import NUCLEI_ID_SCOPES
from alhena.io import EDGE_RULES


def _add_tile_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mpp",
        type=float,
        default=None,
        help=(
            "microns per pixel of the tiles; for a whole slide, of its full-resolution level, "
            "overriding the resolution the slide declares"
        ),
    )
    parser.add_argument(
        "--tile-px",
        type=int,
        default=None,
        help=(
            "side of the tiles cut from a whole slide or an image, in its pixels; by default the "
            "side filling the canvas, at the resolution of the level read for a slide, at --mpp "
            "for an image"
        ),
    )
    parser.add_argument(
        "--stride-px",
        type=int,
        default=None,
        help="step between the tiles cut from a whole slide or an image; the tile side by default",
    )
    parser.add_argument(
        "--edge",
        default=None,
        choices=EDGE_RULES,
        help=(
            "what becomes of the tiles the edge of a slide or an image cuts: drop leaves them "
            "out, shift moves them inward, pad pads them white; pad for a slide, shift for an "
            "image by default"
        ),
    )
    parser.add_argument(
        "--coords",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "for a slide or an image, the level-zero origins of the tiles to cut instead of the "
            "grid: a table of x and y, a .npy array or an HDF5 file; for a directory or a .npy "
            "array of tiles, a .csv, .tsv or .parquet table giving each tile_id its top-left x "
            "and y in the frame the tiles were cut from"
        ),
    )
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help=(
            "for a slide or an image, keep only the tiles lying wholly inside this rectangle, in "
            "level-zero pixels, upper bounds excluded"
        ),
    )
    parser.add_argument(
        "--coords-columns",
        action="extend",
        nargs="+",
        default=None,
        metavar="FIELD=COLUMN",
        help=(
            "the columns of the --coords table carrying x and y, and tile_id in a table of the "
            "origin of every tile, where the usual names are not found"
        ),
    )
    parser.add_argument(
        "--coords-scale",
        type=float,
        default=None,
        help=(
            "pixels of the frame of the tile origins one tile pixel spans, for tiles stored at "
            "another size than they were cut at; what the file records by default, one otherwise"
        ),
    )
    parser.add_argument(
        "--h5-datasets",
        action="extend",
        nargs="+",
        default=None,
        metavar="DATASET=NAME",
        help=(
            "the names an HDF5 file of tiles holds img, barcode and coords under, or a file of "
            "origins --coords names holds coords under"
        ),
    )


def _add_nuclei_options(parser: argparse.ArgumentParser, described: str) -> None:
    from alhena.io import CENTROID_FRAMES, COORDINATE_UNITS, MULTIPART_RULES, NUCLEI_KINDS

    parser.add_argument(
        "--nuclei",
        type=Path,
        default=None,
        help=(
            "label maps in a directory, an HDF5 file or a .npy array of shape (n_tiles, height, "
            "width); the label image of the slide or the image the tiles are cut from, a .png, "
            ".tif or .tiff file or a .npy array of shape (height, width); centroids in a table; "
            f"or boundaries in GeoJSON, GeoParquet or a table of vertices; {described}"
        ),
    )
    parser.add_argument(
        "--nuclei-kind",
        default=None,
        choices=NUCLEI_KINDS,
        help=(
            "the form --nuclei holds, where its suffix does not say it: a table other than "
            "GeoParquet is read as centroids unless named polygons, as a table of boundary "
            "vertices is"
        ),
    )
    parser.add_argument(
        "--nuclei-ids",
        default=None,
        choices=NUCLEI_ID_SCOPES,
        help=(
            "whether a label value names a cell within its tile, renumbered across the tiles, "
            f"or one cell across all of them; {NUCLEI_ID_SCOPES[0]} by default"
        ),
    )
    parser.add_argument(
        "--radius-px",
        type=float,
        default=None,
        help=(
            "nucleus radius standing in for the segmented shape, for centroids with no radius, in "
            "the pixels of the tiles whatever the units of the table"
        ),
    )
    parser.add_argument(
        "--nuclei-frame",
        default=None,
        choices=CENTROID_FRAMES,
        help=(
            "frame a table of centroids is written in: tile, the pixels of the tile each row "
            f"names, or origins, the frame of the tile origins; {CENTROID_FRAMES[0]} by default"
        ),
    )
    parser.add_argument(
        "--nuclei-units",
        default=None,
        choices=COORDINATE_UNITS,
        help=(
            "units of the coordinates of centroids or boundaries: px, or um, divided by the "
            f"microns one pixel of their frame spans; {COORDINATE_UNITS[0]} by default"
        ),
    )
    parser.add_argument(
        "--nuclei-transform",
        type=Path,
        default=None,
        metavar="MATRIX",
        help=(
            "a .csv or .npy file holding a 2 by 3 or 3 by 3 affine matrix carrying the "
            "coordinates of centroids or boundaries into the pixels of the frame of the tile "
            "origins; not with --nuclei-units um"
        ),
    )
    parser.add_argument(
        "--nuclei-columns",
        action="extend",
        nargs="+",
        default=None,
        metavar="FIELD=COLUMN",
        help=(
            "the columns of a table of centroids or of vertices carrying each field, where the "
            "usual names are not found: x, y and cell_id, and tile_index and radius_px for "
            "centroids"
        ),
    )
    parser.add_argument(
        "--nuclei-id-property",
        default=None,
        metavar="NAME",
        help=(
            "property of a GeoJSON feature naming its cell; cell_id by default, then the feature "
            "id, then its position"
        ),
    )
    parser.add_argument(
        "--nuclei-geometry-property",
        default=None,
        metavar="NAME",
        help=(
            "member of a GeoJSON feature holding the nucleus outline, such as nucleusGeometry, "
            "read instead of its geometry"
        ),
    )
    parser.add_argument(
        "--nuclei-multipart",
        default=None,
        choices=MULTIPART_RULES,
        help=(
            "how a boundary holding several polygons is read: refuse it, split it into one "
            f"nucleus per polygon, or keep the largest; {MULTIPART_RULES[0]} by default"
        ),
    )


def _add_shard_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help=(
            "tissue name, alias or reference name of a published reference; give this or "
            "--path, not both"
        ),
    )
    parser.add_argument(
        "--path", type=Path, default=None, help="a reference file on disk, instead of a name"
    )
    parser.add_argument(
        "--axis",
        default=DEFAULT_AXIS,
        help=(
            f"class axis the reference is read on: one of {', '.join(sorted(CLASS_AXES))}, or, "
            "with --path, an axis the file was built on"
        ),
    )
    _add_cache_option(parser)
