# Nuclei

ALHENA uses nuclei to determine which image features belong to each cell. Nuclei can be provided
as segmented shapes or, when full segmentation is unavailable, as approximate positions. Each
nucleus must also have a cell identifier.

ALHENA does not perform nucleus segmentation itself. Provide the output of an external
segmentation method in one of the supported formats below. For example, polygons expressed in the
same image coordinates as the H&E slide can be loaded with:

```python
from alhena import io

nuclei = io.read_polygons("nuclei.geojson", tiles=source)
```

The resulting `nuclei` object can be passed directly to `encode_cells()` or `annotate()`, and to
`viz.overlay()`, which draws each nucleus in the color of its annotation.

!!! tip "No segmentation yet"
    sopa can segment a whole slide with StarDist, and ALHENA reads the outlines from sopa's
    SpatialData object. See
    [Segment a slide with sopa and StarDist](#segment-a-slide-with-sopa-and-stardist).

## Supported nucleus representations

ALHENA accepts six nucleus representations:

| Form | What it is | Reader |
|---|---|---|
| Polygons | the outline of each nucleus | `read_polygons` |
| Label maps | one integer image per tile; a nucleus is the set of pixels holding its identifier | `read_label_maps` |
| A label image | one integer image over the whole image the tiles are cut from | `read_label_image` |
| One mask per nucleus | one binary image per nucleus | `label_map_from_masks`, then a label-map reader |
| Centroids | one position per nucleus, and a radius | `read_centroids` |
| A membership table | nuclei already reduced to the tokens they cover | none: passed as it is |

The representation affects only how the nuclei are loaded. Polygons, label maps, label images, and
masks preserve the segmented nuclear shape. Centroids instead approximate each nucleus as a disk
with a user-defined radius.

### Polygons

```python
nuclei = io.read_polygons("nuclei.geojson", tiles=source)
```

`read_polygons()` accepts nucleus outlines expressed in the coordinate system of the image used to
create the tiles. Three file formats are supported:

- **GeoJSON**: Polygon and MultiPolygon features. `id_property=` names the property holding the
  identifier, `cell_id` by default. `geometry_property=` reads the outline from another member of
  each feature. Use it for files that store the nucleus outline alongside the outline of the whole
  cell, as `geometry_property="nucleusGeometry"`.
- **A table of vertices**, `.csv`, `.tsv`, `.parquet` or `.pq`: one row per vertex, under
  `cell_id`, `vertex_x` and `vertex_y`, or the columns `columns=` names. The rows of a cell are
  consecutive and follow its outline. A cell whose rows are split across the table is refused.
- **GeoParquet**: a parquet file that declares itself GeoParquet, with a geometry column of
  polygons. Reading it needs geopandas, which the `spatial` extra installs.

A nucleus covers the pixels whose centers lie inside its outline or on it, the rule ALHENA's
training masks were drawn with. An outline traced along pixel edges, as contour tracers draw it,
covers exactly the pixels it encloses. A nucleus is placed at the area centroid of its outline. A
file whose outlines all lie outside the tiles is refused, since a wrong frame or unit is then
likelier than an empty segmentation.

??? note "Several polygons under one identifier"
    An outline holding several polygons is refused unless `multipart=` says what the parts are:

    - `multipart="split"` reads each part as a nucleus. Part `k` of feature `a1` is named
      `a1:k`, counting from zero. Every identifier of such a read is text. Use it for files
      that group many nuclei into one feature, such as every nucleus of a class.
    - `multipart="largest"` keeps the part enclosing the largest area, and warns. Use it for
      one nucleus cut into pieces.

    In a table of vertices, several closed rings under one identifier are several polygons of
    one outline, and follow the same rule. A hole in an outline is filled when it is drawn, and
    a warning says so.

### Label maps

```python
nuclei = io.read_label_maps("masks/", tiles=source)
```

A label map is an integer image in which `0` represents background and each positive value
identifies one nucleus. `read_label_maps()` accepts one map per tile from a directory, HDF5 file,
array, list, or stacked `.npy` file.

When maps are stored as individual files, or as one dataset per tile in an HDF5 file, they are
matched to tiles by name, so `tiles=` is required. A stacked array, including the `labels`
dataset of an HDF5 file, does not contain tile names, so its maps are matched to the tiles by
order and the number of maps must equal the number of tiles. The same applies to a list of maps.

A map shares the pixel grid of its tile: one map pixel per tile pixel. A map of another size is
refused once a resolution is declared, or when the tiles carry origins.

### A label image

```python
source = io.read_image("region.png", mpp=0.25)
nuclei = io.read_label_image("labels.tif", tiles=source)
```

One integer image over the whole image the tiles are cut from, naming each nucleus once across
it. `read_label_image` cuts it along the tiles, pixel for pixel. It keeps the identifiers as they
are, so a nucleus that two overlapping tiles hold is one cell, read once.

It can be an array in memory, or an array read by slicing, such as a NumPy memmap, a zarr array
or an HDF5 dataset, for a label image larger than memory. Each map is read from it when the map is
used, so keep an HDF5 file holding it open until the cells are encoded. It can also be a `.npy`
file, read as a memory map, or a `.png`, `.tif` or `.tiff` file, decoded whole. A 16-bit PNG holds
identifiers up to 65,535. It must have the shape of the image. Tiles read at a coarser level of a
slide are refused: give their nuclei as polygons or centroids in level-zero pixels instead.

### One mask per nucleus

Nuclei given as one binary mask each are painted into a label map first:

```python
import numpy as np

masks = np.zeros((2, 8, 8), dtype=bool)        # one mask per nucleus
masks[0, 0:6, 0:6] = True
masks[1, 4:6, 4:6] = True
label_map, lost = io.label_map_from_masks(masks, cell_ids=[11, 12])
print(label_map[5, 5], label_map[0, 0], lost)  # 12 11 []
```

Where masks overlap, a pixel goes to the smaller nucleus, and between equal areas to the lower
identifier. The larger nucleus loses a rim rather than vanishing. A nucleus left with no pixel is
returned in `lost` instead of being dropped silently.

The map carries the identifiers you give, so read it with `renumber=False`. For tiles that each
come with their masks:

```python
maps, next_id = [], 1
for masks in masks_per_tile:                          # (n_nuclei, height, width) booleans
    ids = np.arange(next_id, next_id + len(masks))    # unique across all tiles
    label_map, lost = io.label_map_from_masks(masks, ids, shape=masks.shape[1:])
    maps.append(label_map)
    next_id += len(masks)

nuclei = io.read_label_maps(np.stack(maps), tiles=source, renumber=False)  # in tile order
```

Masks painted over a whole image give a label image instead, read with `read_label_image`.

### Centroids

```python
nuclei = io.read_centroids("nuclei.csv", tiles=source, radius_px=8.0)
```

A `.csv`, `.tsv`, `.txt`, `.parquet` or `.pq` table of `x` and `y`. Because a centroid has no
extent, its radius must come from either a column or `radius_px=`; the radius is measured in tile
pixels. Use `columns=` when the relevant columns have different names.

By default each row names its tile in a `tile_index` column, and gives its position in that
tile's pixels. The first pixel is centered at `0`, so a position runs from `-0.5` up to
`width - 0.5`, and the same along y. A table with one row per nucleus over the whole image
is read with `frame="origins"`, as
[Coordinates](coordinates.md#centroids-over-a-whole-image) shows.

!!! warning "Centroids approximate the nuclear shape"
    A centroid does not contain the shape of the nucleus. ALHENA therefore represents it as a disk
    with the supplied radius before pooling image features. When a full segmentation is available,
    prefer polygons or label maps.

### A membership table

A membership table directly specifies which image tokens contribute to each nucleus and how many
nuclear pixels overlap each token. Other nucleus representations are converted to this form
internally by `encode_cells()`. Supplying a membership table directly is mainly useful for
advanced workflows that need to reuse or control the pooling step. See
[Cell pooling](../concepts/cell-pooling.md).

## Reading any form

`read_nuclei()` selects a reader from the input type and file extension:

| Source | Read as |
|---|---|
| a directory, an HDF5 file, a stacked `.npy` file or array of shape `(n_tiles, height, width)`, or a list of maps | label maps |
| a single `.png`, `.tif` or `.tiff` file, or a two-dimensional `.npy` file or array | a label image |
| a `.geojson` or `.json` file, or a GeoParquet file | polygons |
| any other `.csv`, `.tsv`, `.txt`, `.parquet` or `.pq` file | centroids |

A `GeoDataFrame` held in memory is read as polygons, and any other `DataFrame` as centroids.

Use `kind=` when the format is ambiguous. A table of vertices has the suffix of any table, so it
is read with `kind="polygons"`:

```python
nuclei = io.read_nuclei("boundaries.parquet", tiles=source, kind="polygons")
```

Other arguments are forwarded to the reader selected by `kind`. An argument that the selected reader
does not accept is refused rather than ignored.

**Pass `tiles=` whenever possible.** The tile source provides the geometry needed to match nuclei
with image regions and validate their coordinates. Several nucleus representations, including
whole-image label images and coordinates expressed in microns, require it explicitly.

??? note "Placed by name and frame"
    Label map files are matched to tiles by name, so a map that names no tile is an error rather
    than a guess:

    ```text
    Nucleus map 'x0_y9999' names no tile of this source. Its tiles are named ['x0_y0', 'x0_y348'].
    ```

    Centroids are checked against the tile size. Positions written over a whole slide are
    refused where the size is known, and the message names `frame="origins"`, which reads them.

## Identifiers

!!! warning "Cell identifiers must be globally consistent"
    The same identifier is interpreted as the same biological cell wherever it appears. This allows
    one nucleus present in overlapping tiles to be recognized as a single cell, but it also means
    that identifiers that restart independently in each tile can incorrectly merge unrelated
    nuclei. Give each distinct cell its own identifier, across all the tiles read together.

In practice, an identifier may appear in several overlapping tiles only when those occurrences
represent the same biological cell. Two distinct cells must never share an identifier.

Most segmenters number nuclei independently within each tile, so the same local instance number
can refer to different cells in different tiles. `read_label_maps` therefore renumbers instances
by default as `tile_index * 100000 + instance`: tile 0 keeps `1`, while tile 1's `1` becomes
`100001`. An instance number too large for its block raises an error rather than colliding with the
next tile's. Use `renumber=False` only when the segmentation identifiers are already unique
across the tiles.

```python
nuclei = io.read_label_maps("masks/", tiles=source)                  # renumbers
nuclei = io.read_label_maps("masks/", tiles=source, renumber=False)  # keeps your identifiers
```

`read_label_maps` is the only reader that renumbers. `global_cell_ids` applies the same rule to
a table, before it is read:

```python
import pandas as pd

table = pd.read_csv("nuclei.csv")
table["cell_id"] = io.global_cell_ids(table["tile_index"], table["cell_id"])
table.to_csv("nuclei_cohort.csv", index=False)
```

??? note "How identifiers are assigned for other nucleus inputs"
    - **A label image** is never renumbered. Its values name the nuclei of one image, so a
      cohort read one image at a time needs values unique across the images.
    - **Masks** take the `cell_ids` you give: one positive identifier per mask, with no repeat.
    - **Centroids** take a `cell_id`, `cell` or `id` column, or the one `columns=` names, as they
      stand. A table without one gets identifiers unique across its tiles. Under
      `frame="origins"`, such a table's rows are named by their position instead, from `0`.
    - **Polygons** take the `id_property` of a GeoJSON feature, then the feature's `id`, then
      its position in the file. A table of vertices takes a `cell_id`, `cell` or `id` column,
      and GeoParquet such a column or its index. Two outlines naming one cell in one file are
      refused. Positions restart at zero in every file.
    - **A membership table** you build is read as it stands. Apply `global_cell_ids` first.

    Wherever identifiers come from positions in one file, a cohort read one file at a time needs
    identifiers of its own, unique across the files.

An identifier is an integer or text. Text stored as bytes, as in a parquet column written without
a string type, is read as the text it spells. Identifiers of any other kind, such as decimal
numbers, are refused before any tile is embedded.

Labels you join to the cells must use the same identifiers. Another dataset may rename its cells, in
which case the same identifier can refer to different cells in the two systems. Joining on that
identifier would silently pair the wrong cells. Join on the identifiers of the file the nuclei were
read from, or map one identifier system onto the other first.

??? note "Identifiers, not colors"
    A label map is a two-dimensional integer array. A map of floats is refused rather than
    rounded, since labels are never interpolated. Identifiers written as three color channels
    cannot be read back. Files are decoded without narrowing, so a 16-bit identifier stays whole.

## Nuclei in a SpatialData object

Nucleus outlines stored as a SpatialData shapes element can be converted directly into the
membership representation used by `encode_cells()` and `annotate()` with
`spatial.cells_from_shapes()`:

```python
import alhena
from alhena import io, spatial

source = io.read_slide("slide.svs")
nuclei = spatial.cells_from_shapes(
    sdata, "nucleus_boundaries", patches=io.tile_bounds(source), image_key="he", mpp=source.mpp
)
typer = alhena.load_celltyper(tissue="Breast")
result = typer.annotate(source, nuclei)
```

- `patches` contains the spatial bounds of the ALHENA tiles.
- `image_key` names the image whose level zero holds the pixels of that source. The outlines are
  read in those pixels, through the transformations the object stores. See
  [Coordinates in a SpatialData object](coordinates.md#coordinates-in-a-spatialdata-object).
- `mpp` is the resolution of the tiles, so each mask is mapped onto the model canvas with the same
  geometry as its tile. A table built without `mpp` stretches each mask to the full canvas and is
  refused with tiles that do not fill the canvas exactly after resampling.

`multipart=` follows the same rules as `read_polygons()`. This workflow requires the `spatial`
extra.

## If you have no nuclei yet

ALHENA does not segment nuclei. For an H&E slide with no segmentation, sopa runs StarDist's H&E
model over the tissue. sopa stores the outlines in a SpatialData object, and ALHENA reads them
from there.

Tile embeddings need no nuclei at all. See
[Compute tile embeddings](../start/choose-your-task.md#compute-tile-embeddings).

### Segment a slide with sopa and StarDist

[sopa](https://prism-oncology.github.io/sopa/tutorials/he/) outlines the tissue, cuts it into
patches, and runs StarDist on each patch. This example was checked with sopa 2.2.

Install sopa with its slide reader and StarDist, beside ALHENA. StarDist runs on TensorFlow, on
the CPU.

```bash
pip install "alhena[wsi,spatial]" "sopa[wsi,stardist]"
```

Outline the tissue, cut it into patches for the segmenter, segment, and encode:

```python
import sopa

import alhena
from alhena import io, spatial

encoder = alhena.load_encoder()
source = io.read_slide("slide.svs")

sdata = sopa.io.wsi("slide.svs")
sopa.segmentation.tissue(sdata)
sopa.make_image_patches(sdata, patch_width=2000, patch_overlap=50)
sopa.segmentation.stardist(sdata, model_type="2D_versatile_he")

nuclei = spatial.cells_from_shapes(
    sdata,
    "stardist_boundaries",
    patches=io.tile_bounds(source),
    image_key="wsi",
    mpp=source.mpp,
)
cells = encoder.encode_cells(source, nuclei)
```

The patches sopa segments are not ALHENA's tiles. `patches=io.tile_bounds(source)` draws each
outline into the tiles ALHENA reads, as in
[Nuclei in a SpatialData object](#nuclei-in-a-spatialdata-object).

!!! warning "Check the outlines before you encode"
    Look at the outlines over the slide, at the tissue edge especially. sopa runs StarDist with
    `prob_thresh=0.2` by default, and it can outline marks that are not nuclei, such as ink. If
    it does, raise `prob_thresh=` or set `min_area=` in `sopa.segmentation.stardist`, and
    segment again.

To keep the segmentation, write the outlines as GeoParquet, in the pixels of the slide:

```python
sopa.utils.to_intrinsic(sdata, "stardist_boundaries", "wsi").to_parquet("nuclei.parquet")
```

ALHENA reads that file as polygons, without sopa. In Python:
`io.read_polygons("nuclei.parquet", tiles=source)`. On the command line:

```bash
alhena encode slide.svs --nuclei nuclei.parquet --out cells.h5ad
```

To run ALHENA inside sopa's own patch pipeline instead, see
[Advanced: sopa patch pipeline](../outputs/tile-embeddings.md#advanced-sopa-patch-pipeline).

### Other segmenters

Any other segmentation method works too, as long as its output takes one of the
[supported nucleus representations](#supported-nucleus-representations). Export it in the pixels
of the H&E image when possible: no coordinate transformation is then needed. Otherwise, describe
the relationship as [Coordinates](coordinates.md) explains.

## A nucleus in several tiles

A nucleus may appear in more than one overlapping tile. When tile coordinates are available,
ALHENA keeps a single representation using the tile that contains the largest fraction of the
nucleus. If the tiles do not share a coordinate system, the ambiguity cannot be resolved
automatically, so the input is refused. See
[A nucleus in several tiles](../concepts/cell-pooling.md#a-nucleus-in-several-tiles).

---

**Next:** [Coordinates](coordinates.md)
