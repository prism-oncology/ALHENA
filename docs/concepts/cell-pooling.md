# Cell pooling

ALHENA derives cell embeddings by pooling image tokens over each nucleus rather than by cropping a
separate image around every cell.

The nucleus segmentation determines which tokens contribute and how strongly they are weighted.
Because each image token was computed in the context of the surrounding tile, the resulting cell
representation combines nucleus-associated information with local tissue context.

See [Nuclei](../inputs/nuclei.md) for the supported input representations.

## From a nucleus to its tokens

The 224 × 224 model canvas is divided into a 16 × 16 grid of 256 tokens. At the 0.425 µm/px
resolution the model reads, each token covers approximately 5.95 × 5.95 µm.

A nucleus may overlap several tokens. ALHENA records this relationship in a **membership table**
with one row for each nucleus–token pair of a tile and the number of nuclear pixels overlapping
that token.

```python
import alhena
import numpy as np

maps = np.zeros((2, 256, 256), dtype=np.int32)
maps[0, 40:60, 40:60] = 1
maps[1, 40:60, 40:60] = 100_001

frame = alhena.membership_from_labels(maps, tile_indices=[0, 1], mpp=0.25)
print(frame.columns.tolist())
# ['tile_index', 'cell_id', 'token_index', 'nucleus_px']
```

Three builders produce it, one per kind of nucleus input:

| Builder | Takes | Pixel counts |
|---|---|---|
| `membership_from_labels` | one integer map per tile | the segmented shape |
| `membership_from_centroids` | positions and a radius | a disk, so **approximate** |
| `membership_from_polygons` | outlines and tile windows | the drawn outline |

`encode_cells` and `annotate` call the right one for you, from the nuclei you pass. Call one
yourself to inspect the table before a run, or to build it once and reuse it.
See [Advanced: inspect a membership table](#advanced-inspect-a-membership-table).

For each cell, ALHENA computes a weighted average of the image tokens overlapping its nucleus. The
number of nuclear pixels in each token provides the pooling weight.

This produces the 1536-dimensional **pre-projection cell embedding** exposed through the API as
`backbone` / `X_alhena_backbone`. `img_tok_proj` then maps this representation into the
512-dimensional shared cell space used for reference-based annotation. The
[annotation workflow](cell-typing.md#annotation-workflow) lists every step in the order the code
runs them.

## What information enters a cell embedding

Pooling weights are specific to each nucleus. If two nuclei overlap the same image token, that
token can contribute to both cell embeddings with different weights.

Consequently, nearby cells may share some image tokens and their embeddings may be correlated.
Very small nuclei contained entirely within the same token can even receive nearly identical
pooled representations.

Importantly, an image token is contextualized by the entire tile through the image encoder. A
cell embedding therefore represents the nucleus **within its local image context**, rather than an
isolated nuclear crop.

The token grid also limits the spatial granularity of this pooling: segmentation boundaries finer
than the approximately 5.95 µm token scale do not create correspondingly finer image features.

## Masks on the canvas

Nucleus masks must remain spatially aligned with the H&E image as both are mapped onto the model
canvas. `alhena.cells.fit_mask()` therefore applies the same scaling and center crop or padding
used for the image, with two mask-specific differences:

| | Image | Mask |
|---|---|---|
| Resampling | `cv2.INTER_LINEAR`, or `cv2.INTER_AREA` past a twofold shrink | `cv2.INTER_NEAREST`, so no identifier is blended into one that does not exist |
| Padding | white, 255, the background of an H&E section | 0, the background label |

Masks remain integer-valued, with `0` reserved for background. Nearest-neighbor interpolation
prevents the creation of artificial nucleus identifiers during resampling. A float mask raises
`GeometryError` rather than being rounded.

Padding with the background label is why white pad gets no direct pooling weight: a pad pixel
belongs to no nucleus, so it adds weight to no cell. The pad still reaches a cell as context,
through the attention of the tokens under its nucleus.

!!! warning "Image and nucleus geometry must match"
    A nucleus mask and its corresponding H&E image must use compatible resolution and spatial
    geometry. Otherwise, nucleus pixels would be assigned to the wrong image tokens. Membership
    tables created by ALHENA record their geometry so this consistency can be checked before
    encoding. See [Advanced: membership geometry checks](#advanced-membership-geometry-checks).

## A nucleus in several tiles

Overlapping tiles can contain the same biological nucleus. A nucleus is recognized across tiles by
its identifier, so identifiers must be unique across all the tiles read together, not only
within a tile. When tile coordinates are available, ALHENA resolves these duplicate observations
and keeps a single cell representation. The tile it is read from is chosen in this order:

1. prefer the tile containing the largest number of nucleus pixels after mapping to the model
   canvas;
2. if several tiles contain the same amount, prefer the tile whose center is closest to the
   nucleus position.

This choice is made from the membership information before image encoding. It requires the tiles
to share a coordinate system: polygons carry one, and label maps and centroid tables are placed in
one by the [tile origins](../inputs/coordinates.md#tile-origins). Without a shared coordinate
system, a nucleus found in several tiles is refused rather than resolved arbitrarily.

When the tiles share a coordinate system, each nucleus is also placed once, at one position:

| Form | Position |
|---|---|
| Polygons | centroid of the polygon in the shared image coordinate system |
| Label maps | centroid of the labeled pixels, transformed through the tile origin |
| Centroids | supplied centroid, transformed through the tile origin |

??? note "Nuclei near the edge of a tile"
    The image context available to a cell is limited by the tile used to compute it. Overlapping
    tiles can reduce edge truncation by allowing a nucleus to be represented from a neighboring
    tile that contains more of it.

## Approximate shapes and contested pixels

!!! warning "Centroids approximate nucleus shape"
    `membership_from_centroids()` represents each centroid as a disk with the supplied radius.
    Pooling therefore follows this approximate disk rather than a segmented nuclear boundary.
    Prefer polygons or label maps when full segmentations are available.

When two rasterized nuclei overlap, the contested pixels can belong to only one nucleus in the
final membership mask. If a nucleus loses all of its pixels, it cannot contribute any tokens and
therefore has no cell-embedding row.

## Advanced: membership geometry checks

Membership tables created by ALHENA record the image resolution and pixel grid used to rasterize
the nuclei. Before encoding, ALHENA verifies that these masks map onto the model canvas in the
same way as the corresponding H&E tiles. `encode_cells`, `annotate` and `preflight` refuse a table
whose masks and images would reach different canvas sizes.

This catches the following cases.

- a table built at another resolution than the call declares, such as 0.2125 µm/px against 0.25;
- a table built on another pixel grid than its tile, as label maps of another size are, or the
  windows of a slide read at a coarser level drawn without its `coords_scale`;
- a table built with no resolution, which stretches every mask onto the whole canvas, used with
  tiles that do not fill the canvas exactly after resampling.

A tile of 224 pixels at 0.425 µm/px fills the canvas as it is. A tile of 448 pixels at 0.2125 µm/px
also fills the canvas after resampling. A tile of 224 pixels at 0.2125 µm/px reaches only the
central 112 pixels of the canvas. A table built without a resolution would stretch its masks across
all 224 pixels and is therefore refused with that tile.

A table assembled by hand records none of this. It is read as it stands, at the resolution the
call declares.

`membership_from_labels` and `membership_from_centroids` take `tile_origins=`, which gives the
origins of the tiles, and `tile_scale=`, which gives the number of frame pixels spanned by one tile
pixel. Given the origins, a table created by these builders records the tile centers and the
position of every nucleus, as `membership_from_polygons` does from its windows. If `mpp=` is also
given, every builder records the microns per pixel of that frame. `encode_cells` and `annotate`
pass the origins and scale of a tile source for you.

```python
frame = alhena.membership_from_labels(
    maps, tile_indices=[0, 1], mpp=0.25, tile_origins=[[0, 0], [256, 0]]
)
```

`alhena.spatial.cells_from_shapes` builds the same table from outlines held in a SpatialData
object. See [Nuclei in a SpatialData object](../inputs/nuclei.md#nuclei-in-a-spatialdata-object).

## Advanced: inspect a membership table

Three questions can be answered from a table a builder made, before any encoder is loaded:

```python
from alhena.cells import membership_is_approximate, membership_source, missing_cells

alhena.validate_membership(frame, n_tiles=2)   # raises, or returns None
print(membership_source(frame))                # 'labels'
print(membership_is_approximate(frame))        # False
print(missing_cells(frame))                    # []
```

- `validate_membership()` checks the required columns, valid tile/token indices, positive pixel
  counts, and duplicate `(tile_index, cell_id, token_index)` entries.
- `membership_source()` reports the nucleus representation from which the table was built.
- `membership_is_approximate()` reports whether nucleus shapes were approximated, for example from
  centroids.
- `missing_cells()` identifies input nuclei that did not reach any image token.

??? example "Lost to the center crop"
    A 512 pixel tile at 0.425 µm/px is cropped to its central 224 pixels. The nucleus near the
    corner does not survive the crop; the one near the center does.

    ```python
    maps = np.zeros((1, 512, 512), dtype=np.int32)
    maps[0, 2:12, 2:12] = 1          # near the corner
    maps[0, 250:262, 250:262] = 2    # near the center

    frame = alhena.membership_from_labels(maps, tile_indices=[0], mpp=0.425)
    print(frame["cell_id"].unique().tolist())  # [2]
    print(missing_cells(frame))                # [1]
    ```

    The crop discards 80.9 percent of this field, which `describe_geometry` reports as
    `crop_fraction`. The builder says nothing, and `encode_cells` and `annotate` warn.
    See [A large field is cropped](geometry.md#a-large-field-is-cropped).

??? note "A table built by hand has no provenance"
    `membership_is_approximate` and `missing_cells` need a table a builder made. A table
    assembled by hand records no input and no provenance, so both raise rather than guess, and
    `membership_source` returns `'unknown'`. It records no resolution either: it is read at the
    resolution the call declares, which its masks must already match.

---

**Next:** [Cell typing](cell-typing.md)
