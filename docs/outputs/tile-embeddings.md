# Tile embeddings

`encode_tiles()` returns a `TileEmbeddings` object with one row per tissue tile. Each row contains
a 1536-dimensional tile embedding, a tile identifier, and spatial bounds when the tile can be
located in the source image.

!!! tip "What can I learn from this?"
    Regional tissue morphology and its variation across the slide.

```python
tiles = encoder.encode_tiles(source)

print(tiles.embeddings.shape)   # (n_tiles, 1536)
print(tiles.tile_ids[:2])       # ('x0_y0', 'x381_y0')
print(tiles.bounds[:2])         # x0, y0, x1, y1 of each tile, in the pixels of the slide
print(tiles.bounds_frame)       # slide.svs, level 0
```

## What a tile embedding represents

Each tile embedding is a 1536-dimensional pre-projection representation of one H&E region. The
model processes a physical field of view of approximately 95.2 µm, after any required resampling,
cropping, or padding.

The representation is computed from H&E alone, but the encoder was adapted under molecular
supervision during training. It therefore remains an image-derived representation and should not
be interpreted as gene expression.

Tiles that are close in this embedding space have similar representations according to ALHENA.
The embeddings can be used for downstream analyses such as nearest-neighbor graphs, dimensionality
reduction, and clustering.

See [What distillation changes](../concepts/how-it-works.md#what-distillation-changes) and
[Downstream analysis](downstream-analysis.md).

## What TileEmbeddings holds

```text
TileEmbeddings
├── embeddings       the tile embedding, one row per tile
├── tile_ids         the name of each tile
├── tile_index       its position among the tiles of the source
├── bounds           its spatial bounds in the source image, when available
├── centers          the center of those bounds
├── bounds_frame     the frame bounds is written in
├── bounds_mpp       the microns one unit of that frame spans
├── image_shape      the size of the image the tiles were cut from
├── geometry         what preprocessing did to reach the model's canvas
├── encoder_id       the encoder that produced the embeddings
├── backbone_digest  the digest of the UNI2-h model weights
└── purpose          "tiles", or "cell_context" for tiles read by a cell pass
```

| Field | Type | Holds |
|---|---|---|
| `embeddings` | `(n_tiles, 1536)` float32 | one embedding per tile |
| `tile_ids` | tuple of str | the name the source gives each tile |
| `tile_index` | `(n_tiles,)` int | the position of each tile among the tiles given |
| `bounds` | `(n_tiles, 4)`, or `None` | `x0, y0, x1, y1` of each tile in the image, the upper bound excluded |
| `centers` | `(n_tiles, 2)`, or `None` | the center of each tile's spatial bounds, `(x, y)` |
| `bounds_frame` | str, or `None` | the frame of `bounds`, such as `"slide.svs, level 0"` |
| `bounds_mpp` | float, or `None` | microns per unit of that frame |
| `image_shape` | `(height, width)`, or `None` | the image the tiles were cut from |
| `geometry` | `ResampleInfo`, or `None` | what preprocessing did; also read as `pad_fraction`, `crop_fraction` and `mpp` |
| `encoder_id` | str, or `None` | the identity of the encoder |
| `backbone_digest` | str, or `None` | the digest of the UNI2-h model weights the tiles were encoded with |
| `purpose` | str | `"tiles"` or `"cell_context"` |

## Spatial and non-spatial tile collections

When tiles originate from the same image and their coordinates are known, `TileEmbeddings`
retains their spatial bounds, centers, coordinate frame, and spatial scale.

If the input consists of unrelated crops with no shared coordinate system, the embeddings remain
valid but `bounds`, `centers`, `bounds_frame`, and `bounds_mpp` are `None`. ALHENA does not infer
spatial positions that were not provided by the input.

## Spatial bounds and centers

`bounds` records the region of the source image represented by each tile as `(x0, y0, x1, y1)`.
These coordinates refer to the source image rather than the 224 × 224 model canvas.

If a tile is resampled, cropped, or padded before inference, its source-image bounds remain
unchanged while `geometry` records the preprocessing applied before the model input was created.

`centers` contains the center of each tile's bounds and can be used by tools that expect one
spatial point per observation. `bounds_mpp` converts these image-coordinate values to microns.

`viz.tile_regions()` draws a value per tile, such as a tile embedding cluster, over the slide
region `io.read_region()` reads, each region in its place. `tiles.locate()` names the tile whose
region holds each cell, which carries a tile-level value to the cells. See
[Across scales](downstream-analysis.md#across-scales).

`tiles.select()` keeps some of the tiles, given as a boolean mask or as positions, with their
bounds and provenance.

## Encoding geometry

`TileEmbeddings` records how the input tiles were mapped onto the model canvas. Two summary values
are particularly useful:

- **`pad_fraction`** reports the fraction of the model canvas filled by padding when an input tile
  covers less than the expected physical field of view.
- **`crop_fraction`** reports the fraction of the input field removed by center cropping when the
  tile covers a larger physical region than expected.

The values stored on `TileEmbeddings` report the maximum observed across the encoded tiles.

`encode_tiles()` warns when padding occurs and when more than 5% of an input field is removed by
cropping. With the default tile size selected by the ALHENA readers, substantial padding or
cropping is normally avoided.

!!! warning "Compare embeddings generated with consistent image geometry"
    Differences in padding or cropping can affect the resulting embeddings independently of
    tissue morphology. Before combining tile embeddings from different slides or cohorts, check
    that their image resolution and preprocessing geometry are comparable. See
    [Can I pool tile embeddings from two cohorts?](../faq.md#can-i-pool-tile-embeddings-from-two-cohorts).

## Dedicated tiles and cell-context tiles

ALHENA produces two kinds of tile embeddings, and `purpose` records which one you hold:

| `purpose` | Produced by | What the tiles are |
|---|---|---|
| `"tiles"` | `encode_tiles()` | a deliberate sampling of the image, for tile-level analysis |
| `"cell_context"` | `encode_cells()`, on `cells.tiles` | the tiles the cells were read from, which give the cells their context |

`purpose` records how the tile embeddings were generated. It is provenance metadata rather than a
parameter chosen by the user.

By default, `encode_cells()` processes only tiles required to represent the supplied nuclei.
Consequently, `cells.tiles` may cover only part of the image and should not automatically be
treated as a systematic tile-level analysis.

For an independent tile-level analysis, use `encode_tiles()`. Alternatively,
`encode_cells(..., all_tiles=True)` processes every tile during the same forward pass; these
embeddings still retain `purpose="cell_context"` to record their origin.

When a slide needs both tile and cell embeddings, the single pass is the efficient choice. At the
same `batch_size`, its tile embeddings, tile names and spatial bounds equal those of
`encode_tiles()`, and each tile runs through the model once instead of twice:

```python
cells = encoder.encode_cells(source, nuclei, all_tiles=True)
tiles = cells.tiles  # every tile of the source
```

## As an AnnData table

`tiles.to_anndata()` returns the tiles as an AnnData table. One observation is one tile.

| Where | Key | Holds |
|---|---|---|
| `obsm` | `X_alhena_tile` | the tile embeddings, `(n_tiles, 1536)` |
| `obsm` | `spatial` | the center of each tile, x then y, where the tiles are placed |
| `obs` | `tile_id` | the name of each tile, which is also the index |
| `obs` | `alhena_tile_index` | the position of each tile among the tiles given |
| `obs` | `alhena_tile_x0`, `alhena_tile_y0`, `alhena_tile_x1`, `alhena_tile_y1` | the tile bounds, where the tiles are placed |
| `uns` | `alhena_tiles` | the provenance: encoder, purpose, geometry, frame, microns per unit, image shape |
| `X` | | nothing: zero columns wide |

The embedding dimensions are representation features rather than measured variables such as genes,
so they are stored in `obsm["X_alhena_tile"]` rather than in `X`. For example, Scanpy can use them
directly with `sc.pp.neighbors(table, use_rep="X_alhena_tile")`.

## In a SpatialData object

The tiles can be written into the same SpatialData object as the image they were cut from:

```python
from alhena import spatial

spatial.tiles_to_spatialdata(sdata, tiles, image_key="he")
```

The function adds one rectangle per tile together with the tile table. It requires the `spatial`
extra and tiles with spatial bounds. When `image_key` is provided, the tile rectangles inherit the
transformations of that image and remain spatially registered with it. See
[Writing into a SpatialData object](persistence.md#writing-into-a-spatialdata-object).

## Advanced: low-level `encode()`

`encoder.encode(tiles, mpp=...)` is the low-level operation used beneath the higher-level tile
workflow. It accepts tiles already held in memory and returns only the `(n_tiles, 1536)` embedding
matrix, without tile identifiers, spatial coordinates, or provenance.

`encode()` prepares its entire input before the forward pass, so `batch_size` limits the
forward-pass batch but does not bound the memory used by the prepared input. `encode_tiles()`
instead reads a source one batch at a time and releases each batch, so the source itself can be
larger than memory. Use `encode()` when a matrix for a few tiles in memory is all you need.

## Advanced: sopa patch pipeline

Use this route only when sopa already owns the patching step: sopa chooses and extracts the image
patches, and you want the resulting embeddings stored in sopa's SpatialData layout.

In that workflow:

1. load an ALHENA encoder;
2. call `spatial.tile_embeddings(sdata, encoder, ...)`;
3. continue with the embeddings written into the SpatialData object by the sopa pipeline.

`spatial.tile_embeddings(...)` uses `spatial.Alhena` as the module called by sopa. The wrapper
requires 224-pixel-wide patches and pixel values on the unit scale. sopa controls the patch
geometry and pyramid level, and ALHENA does not resample the patches on this route. Pass `mpp=`
to have that geometry checked. The forward pass runs on the device on which the encoder was
loaded.

This route does **not** return a `TileEmbeddings` object.

If you want an ALHENA `TileEmbeddings` object instead, do not use the sopa patch pipeline. Read the
SpatialData image with `spatial.tile_source(...)` and pass that source to
`encoder.encode_tiles(...)`. To encode the same windows that sopa selected, pass those windows
through `patches=` when creating the tile source.

---

**Next:** [Cell embeddings](cell-embeddings.md)
