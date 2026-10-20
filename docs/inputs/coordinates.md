# Coordinates

No additional coordinate information is needed when nuclei are expressed in the same pixel
coordinate system as the H&E image or tile in which they were segmented.

This covers the common cases:

- polygons, or whole-image centroids read with `frame="origins"`, expressed in full-resolution
  image pixels;
- one label map per tile;
- a label image covering the same image as the H&E input;
- centroids expressed in the pixels of their corresponding tile.

The sections below cover cases where tile positions are missing or nuclei are expressed in microns
or in the coordinate system of another registered image.

!!! tip "Which coordinate option do I need?"
    - **Nuclei already use the pixels of the corresponding H&E image or tile:** no additional
      coordinate conversion is needed.
    - **Stored tiles have no positions in their parent image:** provide tile origins with `coords=`.
    - **Stored tiles were resized after extraction:** describe the image-pixel to tile-pixel
      relationship with `coords_scale=`.
    - **Whole-image centroids are given in image pixels rather than per-tile pixels:** use
      `frame="origins"`.
    - **Coordinates are expressed in microns:** use `units="um"`.
    - **Coordinates come from another registered image:** provide the registration with
      `transform=`.

## Coordinate systems used by ALHENA

In this documentation, a *frame* means a coordinate system with a defined origin, axes, and unit.
Four types are commonly encountered:

| Frame | Coordinates are | Typical source |
|---|---|---|
| **Tile pixels** | pixels of one tile, from its top-left corner | label maps; centroids given per tile |
| **Image pixels** | full-resolution pixels of the image the tiles are cut from | outlines from a segmentation of the whole slide |
| **Microns** | physical distances on the section | tables of measurements |
| **Another image's pixels** | pixels of a second image registered to this one | the image of a spatial assay |

For integrated spatial analysis, ALHENA converts positions into the image-pixel coordinate system
used by the tile origins. Coordinates expressed in microns or in another registered image are
transformed into this frame when they are loaded.

Pixel centers lie at whole coordinates: the pixel in row `r` and column `c` is centered at
`x = c`, `y = r`.

## Tile origins

A tile origin is the `(x, y)` position of its top-left pixel in the parent image. Tile origins
place all tiles in a shared image coordinate system. This allows `encode_cells()` to retain cell
positions and resolve nuclei that occur in overlapping tiles, while `encode_tiles()` can retain
the spatial bounds of each tile.

Most tile sources provide their origins:

| Source | `coords` | `coords_frame` |
|---|---|---|
| `read_slide` | the grid or the tiles selected, in level-zero pixels | `"slide.svs, level 0"` |
| `read_image` | the grid or the tiles selected, in the pixels of the image | `"region.png, level 0"`, or `"image, level 0"` for an array |
| `spatial.tile_source` | the grid or the windows given, in the pixels of the image | `"image 'he', level 0"` |
| an HDF5 file holding `coords` | that dataset | `"tiles.h5, coords"` |
| `read_tiles(..., coords=...)` | the origins given | `"tile origins"` |
| a directory or an array, a file without `coords` | `None` | `None` |

The frame name travels with every position recorded in it, so a table always says what its
positions are in. An origin is a whole number of pixels: one that is not, or is not finite, is
refused rather than rounded.

!!! warning "Tile origins use `(x, y)` top-left coordinates"
    A coordinate array does not encode whether values represent tile centers or top-left corners,
    nor whether the order is `(x, y)` or `(y, x)`. ALHENA interprets `coords` as `(x, y)` top-left
    origins. If your source uses another convention, convert the coordinates before loading them.

### Tiles without origins

A directory of images or an array has no origins. Give them in a table, one row per tile: its
name under `tile_id`, and its top-left pixel under `x` and `y`.

```text
tile_id,x,y
x0_y0,0,0
x224_y0,224,0
x0_y224,0,224
```

```python
source = io.read_tiles("tiles/", mpp=0.5)
source = io.read_tiles("tiles/", mpp=0.5, coords=io.read_tile_origins("origins.csv", tiles=source))
print(source.coords_frame)      # tile origins
```

The table is a `.csv`, `.tsv`, `.txt` or `.parquet` file, and `columns=` names other columns.
Rows are matched to tiles by name, so their order does not matter. A tile of a directory is named
after its file, and a tile of an array by its position, from `0`. Names are compared as text, so
`007` does not name tile `7`.

Every tile must have exactly one origin. ALHENA refuses a row that names no tile, a tile that
appears more than once, a tile with no matching row, or an origin that falls between pixels. On the
command line, `--coords origins.csv` does the same for a directory or a `.npy` array of tiles.

### Tiles stored at another size

Tile origins are expressed in image pixels, while stored tile arrays may have been resized after
extraction. `coords_scale` describes this relationship by giving the number of image pixels
represented by one stored tile pixel.

- **An HDF5 file may record it**, as the side of a tile in image pixels and, optionally, in tile
  pixels, under either of two conventions:

    | Attributes on | Side in the image | Side in the tile |
    |---|---|---|
    | `coords` | `patch_size_level0` | `patch_size` |
    | `img` | `patch_size_src` | `patch_size_target` |

    The scale is the image-space side length divided by the stored tile side length:
    `163 / 224 = 0.728` for a 224 pixel stored tile that was cut from a 163 pixel-wide region of the
    image. A record that contradicts the tiles is refused.

- **`coords_scale=` states it**, on `read_tiles`, and `--coords-scale` on the command line. A
  value that contradicts the file's record is refused.
- **A slide read at a coarser level** uses the downsample of that level.
- **Otherwise it is one.**

```python
source = io.read_tiles("patches.h5", mpp=0.5, coords_scale=2.0)
print(source.coords_scale, source.coords_frame)   # 2.0 patches.h5, coords
```

`coords_scale` is used when converting tile bounds, tile centers, and cell positions into image
coordinates. If resized tiles are loaded without the correct scale, their spatial positions will be
systematically incorrect even when the tile origins themselves are valid.

## Centroids over a whole image

A table with one row per nucleus, positions in image pixels and no tile column, is read with
`frame="origins"`:

```python
nuclei = io.read_centroids("cells.csv", tiles=source, radius_px=8.0, frame="origins")
```

Each centroid is assigned to the tile or tiles that contain it. This requires tile origins.
Centroids outside all retained tiles are omitted and reported in a warning. If none of the
centroids overlaps any tile, ALHENA raises an error because the coordinate system or units are
likely incorrect.

## Coordinates in microns

`units="um"` reads coordinates in microns. Each coordinate is divided by the microns one pixel
of its frame spans, which the tiles declare:

```python
nuclei = io.read_centroids(
    "cells.parquet", tiles=source, radius_px=8.0, frame="origins", units="um"
)
```

In image pixels, that divisor is `mpp / coords_scale`. For a slide, it is the resolution of level
zero, whatever level is read. For positions in tile pixels, it is the `mpp` of the tiles. Tiles
that declare no `mpp` are refused.

A table of measurements often names its columns otherwise. `columns=` maps them:

```python
nuclei = io.read_centroids(
    "measurements.tsv", tiles=source, frame="origins", units="um", radius_px=8.0,
    columns={"x": "Centroid X µm", "y": "Centroid Y µm"},
)
```

The radius stays in tile pixels, whatever the units of the positions.

!!! warning "Coordinate units must be specified correctly"
    Numeric coordinates do not indicate whether they represent pixels or microns. ALHENA cannot
    infer this reliably from the values alone. Supplying the wrong units changes the spatial
    location of the nuclei and may still produce geometrically plausible coordinates.

`units="um"` assumes the microns were measured on this image. Nuclei measured on another image,
registered to this one, need the registration as well: use `transform=`, below.

## Coordinates of another image

Nuclei segmented on another image of the same section are written in that image's frame.
`transform=` carries them into the frame of the tile origins. It takes a 2×3 or 3×3 affine
matrix, as an array or in a `.csv` or `.npy` file, and applies it to the coordinates as written:

```python
nuclei = io.read_polygons("nuclei.geojson", tiles=source, transform="matrix.csv")
```

The matrix already holds any change of units, so `transform=` and `units="um"` are never given
together. A 3×3 matrix whose last row is not `0, 0, 1` is projective, and is refused, as is a
matrix that collapses the plane. Centroids are carried only with `frame="origins"`.

### Constructing the transform from a registration matrix

A registration is usually recorded the other way round: as an alignment matrix `A`, which
carries the pixels of the tiles' image into the pixels of the other image. The matrix to pass
composes the other image's unit scale with the inverse of `A`:

```python
import numpy as np

alignment = np.loadtxt("alignment.csv", delimiter=",")  # tiles' image -> other image, pixels
to_pixels = np.diag([1 / 0.5, 1 / 0.5, 1.0])            # microns -> pixels of the other image
transform = np.linalg.inv(alignment) @ to_pixels         # microns -> pixels of the tiles' image

nuclei = io.read_centroids(
    "cells.csv", tiles=source, radius_px=8.0, frame="origins", transform=transform
)
```

For example, take the alignment `[[0.5, 0, 100], [0, 0.5, 40], [0, 0, 1]]`, with pixels of
0.5 µm on the other image. A cell written at 60, 30 µm lies on pixel 120, 60 of the other image,
and on pixel 40, 40 of the tiles' image.

The alignment must start from the level-zero pixels of the file the tiles are cut from. Check which
direction the recorded registration uses. If it maps the tiles' image pixels to the other image
pixels, invert it as above. If it maps the other image pixels to the tiles' image pixels, use that
matrix directly in place of `np.linalg.inv(alignment)`.

Outlines are carried before they are placed, so their positions are recorded in the frame of the
tile origins.

## Coordinates in a SpatialData object

A SpatialData object stores transformations between its elements and coordinate systems.
`spatial.cells_from_shapes()` uses these transformations to express nucleus outlines in the
coordinate system required by the H&E image.

- **With `image_key`**, outlines are transformed into the level-zero pixel coordinates of the
  selected image. This is the preferred mode when the ALHENA tiles were read from that image.
  `image_key` must identify the image whose pixels underlie the tiles: a downsampled copy of the
  slide has a different frame.
- **With `coordinate_system` but no `image_key`**, outlines are transformed into the selected
  SpatialData coordinate system instead. With neither argument, they are read in their stored
  coordinates.

If several valid transformations would place the outlines differently, ALHENA does not choose one
automatically. Specify `coordinate_system=` explicitly.

The resulting frame is recorded with the cell positions so downstream results retain their
spatial provenance.

??? note "Windows from a shapes element"
    The windows of a shapes element, such as the patches a pipeline writes, are carried into the
    frame the outlines are read in. They are refused where that would rotate, mirror or shear
    them. A pipeline may clip its windows to the tissue outline and keep the full window of each
    tile in a `bboxes` column; the windows are then read from that column. Without it, a window
    that does not fill its bounding box is refused, since it describes a smaller field than its
    tile.

## Where positions end up

When spatial coordinates are available, ALHENA stores them with the resulting tile and cell
embeddings:

| Result | Positions | Frame | Microns per unit |
|---|---|---|---|
| `CellEmbeddings` | `centroids`; `obsm["spatial"]` in its table | `centroids_frame` | `centroids_mpp` |
| `TileEmbeddings` | `bounds` and `centers`; `obsm["spatial"]` holds the centers | `bounds_frame` | `bounds_mpp` |

Multiply by the microns per unit to get microns: `cells.centroids * cells.centroids_mpp`. For a
slide read at a coarser level, that is the resolution of level zero, not of the tiles.

Tile and cell results derived from the same image source use the same spatial frame, allowing
their positions to be compared directly in multi-scale analyses. See
[Across scales](../outputs/downstream-analysis.md#across-scales).

Tiles without origins give label maps and centroids no position, and their tables carry no
`spatial`.

## On the command line

The main coordinate options are also available from `alhena encode` and `alhena annotate`:

| Flag | Python argument |
|---|---|
| `--coords` | tile origins for a directory or `.npy` of tiles; tiles to cut, for a slide or an image |
| `--coords-scale` | `coords_scale=` |
| `--nuclei-frame origins` | `frame="origins"` |
| `--nuclei-units um` | `units="um"` |
| `--nuclei-transform matrix.csv` | `transform=` |
| `--nuclei-columns x=... y=...` | `columns=` |

```bash
alhena annotate slide.svs --nuclei cells.parquet --nuclei-frame origins --nuclei-units um --radius-px 8 --tissue Breast --out annotations.h5ad
```

```bash
alhena annotate slide.svs --nuclei nuclei.geojson --nuclei-transform matrix.csv --tissue Breast --out annotations.h5ad
```

See [How nuclei are read from the command line](../cli.md#how-the-nuclei-are-read) for the
complete set of options.

---

**Next:** [References](references.md)
