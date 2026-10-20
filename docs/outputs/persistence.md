# Persistence and interchange

ALHENA results can be saved in several formats depending on whether you need only the embedding
matrix, the complete analysis table, or an integrated spatial object. The formats below preserve
different amounts of metadata and spatial information.

## Tile embeddings

| Format | Holds | Written by | Read by |
|---|---|---|---|
| `.h5ad` | the full table: embeddings, identifiers, spatial bounds and centers when available, and provenance | `tiles.write_h5ad(path)`; `alhena encode --out tiles.h5ad` | `io.read_tile_embeddings(path)` |
| SpatialData | the tile table together with one spatial rectangle per tile | `spatial.tiles_to_spatialdata(sdata, tiles)`; `tiles.to_spatialdata()` | `io.read_tile_embeddings(sdata.tables[key])` |
| `.h5`, `.hdf5` | the matrix, the tile names, and the geometry of the run | `alhena encode --out tiles.h5` | h5py |
| `.npy` | the matrix alone, with the geometry of the run in a `.geometry.json` file beside it | `alhena encode --out tiles.npy` | NumPy |

`.h5ad` is the analysis-ready form, and the one to use by default:

```python
tiles.write_h5ad("tiles.h5ad")

tiles = alhena.io.read_tile_embeddings("tiles.h5ad")
print(tiles.purpose, tiles.bounds_frame, tiles.mpp)
```

`read_tile_embeddings()` reconstructs the `TileEmbeddings` object, including spatial bounds,
`purpose`, and encoding geometry when these were stored. The lighter `.h5` and `.npy` formats
preserve the embedding matrix and geometry information but not the full spatial metadata.
[`alhena encode`](../cli.md#alhena-encode) lists what each tile-output format preserves.

A `.h5` file holds two datasets, `embeddings`, one row per tile, and `barcode`, the name of each
tile. The geometry of the run is stored as attributes of the file: `encoder_id`, the fields of
`ResampleInfo` and `embedding_shape`, with NaN for a field that has no value. The
`.geometry.json` file beside a `.npy` holds the same record, with null for a missing value.

## Cell embeddings and annotations

| Result | Format | Holds | Written by | Read by |
|---|---|---|---|---|
| `CellEmbeddings` | `.h5ad` | both embeddings, the positions, the provenance | `cells.write_h5ad(path)`; `alhena encode --nuclei ... --out cells.h5ad` | `io.read_cells(path)` |
| `CellEmbeddings` | SpatialData | the `.h5ad` table, annotating the element the nuclei came from | `spatial.cells_to_spatialdata(sdata, cells, region=...)`; `cells.to_spatialdata(sdata, region=...)` | `io.read_cells(table)` |
| `AnnotationResult` | `.h5ad` | one or several annotations, the cells they were made from, and their provenance | `result.write_h5ad(path)`; `io.add_annotation(table, result)` for each further annotation; `alhena annotate --out annotations.h5ad` | `io.read_annotations(path, name=...)` for one annotation with its cells; `io.read_cells(path)` for the cells alone |
| `AnnotationResult` | `.parquet`, `.csv` | the annotations alone, laid out as `to_frame()` lays them | `result.to_frame()`; `alhena annotate --out annotations.parquet` | pandas |
| `AnnotationResult` | SpatialData | the `.h5ad` table, annotating the element the nuclei came from | `spatial.cells_to_spatialdata(sdata, result, region=...)` or `result.to_spatialdata(sdata, region=...)`, once per annotation | `io.read_annotations(table, name=...)`; `io.read_cells(table)` for the cells alone |

Save an `AnnotationResult` as `.h5ad` when the underlying cell embeddings may need to be reused.
`io.read_annotations()` reads an annotation back from this file, with its cells, so it can be
drawn or evaluated again; `name=` chooses one where the file holds several. `io.read_cells()`
recovers the `CellEmbeddings` alone, allowing them to be annotated again without another
image-encoder pass.

A table records the identifier and the position of each cell, not the outline of its nucleus. The
outlines stay in the segmentation they were read from, and the identifiers link each row to its
nucleus there. `spatial.cells_to_spatialdata()` keeps the table and the nuclei together in one
SpatialData object. `.parquet` and `.csv` exports contain the annotation
table only and do not preserve the embeddings or full provenance.

### Several annotations of the same cells

One table holds the cells once, with their embeddings, and any number of annotations of them.
Each annotation adds its own fields, named after the annotation. The name is the axis unless
another is given, so annotations on `lineage_v1` and on `compartment_v1` share one table:

```python
from alhena import io

table = lineage.to_anndata()              # the cells, and alhena_celltype_lineage_v1
io.add_annotation(table, broad)           # adds alhena_celltype_compartment_v1
table.write_h5ad("annotations.h5ad")
```

Here `lineage` and `broad` are the results of annotating the same `CellEmbeddings` on the
two axes. Annotations on one axis against two references take a name each, as
`io.add_annotation(table, result, name="lineage_v1_second")`.

`add_annotation()` refuses an annotation whose name the table already holds, unless
`overwrite=True`, which replaces that annotation alone. It also refuses annotations of other
cells: identifiers that differ from those of the table, in its order, or another cell space
identity. To add an annotation to a saved table, read it with `anndata.read_h5ad()`, add the
annotation, and write the table back.

## The table layouts

ALHENA stores embeddings in AnnData `obsm`, while `obs` contains per-tile or per-cell metadata and
`uns` stores provenance. `X` is intentionally empty because the embedding dimensions are
representation features rather than measured variables. Every key below is a constant of
`alhena.io`, or for the fields of an annotation a constant followed by the annotation name, so
code reading a table need not spell one out.

### Tiles

| Where | Key | Constant | Holds |
|---|---|---|---|
| `obs` index | | | the name of each tile |
| `obs` | `tile_id` | `TILE_ID_COLUMN` | the same names, as a column |
| `obs` | `alhena_tile_index` | `TILE_INDEX_COLUMN` | the position of each tile among the tiles given |
| `obs` | `alhena_tile_x0`, `alhena_tile_y0`, `alhena_tile_x1`, `alhena_tile_y1` | `TILE_BOUNDS_COLUMNS` | tile bounds `(x0, y0, x1, y1)`, upper bounds excluded |
| `obsm` | `X_alhena_tile` | `TILE_EMBEDDING_KEY` | the embeddings, `(n_tiles, 1536)` float32 |
| `obsm` | `spatial` | `SPATIAL_KEY` | center of each tile's spatial bounds, `(x, y)` |
| `uns` | `alhena_tiles` | `TILES_PROVENANCE_KEY` | the provenance: encoder, UNI2-h model weights, purpose, geometry, frame, microns per unit, image shape |

The bound columns and `obsm["spatial"]` are absent when the tiles have no shared coordinate frame.

### Cells

| Where | Key | Constant | Holds |
|---|---|---|---|
| `obs` index | | | each identifier, as text |
| `obs` | `cell_id` | `CELL_ID_COLUMN` | the identifiers, in the type they were given: integer or text |
| `obs` | `alhena_tile_index` | `TILE_INDEX_COLUMN` | the tile each cell was read from |
| `obs` | `alhena_nucleus_px` | `NUCLEUS_PX_COLUMN` | the nucleus pixels supporting it there |
| `obs` | `alhena_n_candidate_tiles` | `CANDIDATE_TILES_COLUMN` | how many tiles held the nucleus |
| `obs` | `alhena_embedding_valid` | `EMBEDDING_VALID_COLUMN` | whether the embedding may be annotated |
| `obsm` | `X_alhena_backbone` | `BACKBONE_KEY` | the pre-projection cell embeddings, `(n_cells, 1536)` float32 |
| `obsm` | `X_alhena_cell` | `CELL_SPACE_KEY` | projected embeddings in the shared cell space, `(n_cells, 512)` float32 |
| `obsm` | `spatial` | `SPATIAL_KEY` | the position of each nucleus, x then y, where recorded |
| `uns` | `alhena_cells` | `CELLS_PROVENANCE_KEY` | the provenance of the cells |

The cell provenance records the encoder and cell-space identities, the digest of the UNI2-h model
weights and the geometry of the run. It also records where the nuclei came from, whether their
shapes were approximated, how many nuclei reached no token, the frame and microns per unit of
`spatial`, and the identifier type.

### Annotations

Each annotation adds the following fields to the cell table. `<name>` is the name of the
annotation: its axis unless another name was given. The annotation on the default axis therefore
lands in `alhena_celltype_lineage_v1`.

| Where | Key | Constant | Holds |
|---|---|---|---|
| `obs` | `alhena_celltype_<name>` | `LABEL_COLUMN` | predicted cell type; `""` when unassigned |
| `obs` | `alhena_confidence_<name>` | `CONFIDENCE_COLUMN` | the winning class's share of the vote |
| `obs` | `alhena_valid_<name>` | `VALID_COLUMN` | whether an annotation was assigned |
| `obs` | `alhena_class_entropy_<name>` | `CLASS_ENTROPY_COLUMN` | the entropy of the class mass |
| `obs` | `alhena_neighbor_entropy_<name>` | `NEIGHBOR_ENTROPY_COLUMN` | the entropy of the neighbor weights |
| `obsm` | `alhena_mass_<name>` | `MASS_KEY` | the class mass, in class order |
| `uns` | `alhena_celltype_<name>_colors` | | the color of each cell type in the default palette, as Scanpy reads it; absent where the palette lacks one of the cell types |
| `uns` | `alhena`, then `annotations`, then `<name>` | `PROVENANCE_KEY`, `ANNOTATIONS_KEY` | the provenance of the annotation: reference, axis, class order, `k`, `tau`, cell-space identity |

A table holding no annotation has no `alhena` entry in `uns`.

A provenance entry whose value is unknown is omitted because these file formats have no null
representation for it. `read_cells` and `read_tile_embeddings` read such a missing entry as
unknown, and refuse a table lacking an entry every writer records.

## Writing into a SpatialData object

A SpatialData object can store the H&E image, nuclei, tile regions, and ALHENA result tables while
preserving their coordinate transformations. This functionality requires the `spatial` extra.

**Tiles.** `spatial.tiles_to_spatialdata` adds one rectangle per tile, and the tile table:

```python
from alhena import spatial

spatial.tiles_to_spatialdata(sdata, tiles, image_key="he")
```

With `image_key`, the tile rectangles inherit the transformations of the selected image and remain
registered with it. Tiles that `spatial.tile_source()` read from an image of the object inherit
its transformations without `image_key`. Other tiles without `image_key` are added in the `global`
coordinate system with an identity transformation.

`tiles.to_spatialdata()` provides the object-oriented equivalent, and can also create a new
SpatialData object containing only the tile regions and tile table when no existing object is
supplied. By default, the shapes and table are stored as `alhena_tiles` and
`alhena_tile_embeddings`.

**Cells and their annotations.** `spatial.cells_to_spatialdata()` adds the table of cells to an
existing SpatialData object and associates each row with the nucleus element supplied through
`region=`. The first call writes the cells, alone or with an annotation. Each later call adds the
annotations of the same cells to that table:

```python
spatial.cells_to_spatialdata(sdata, lineage, region="nucleus_boundaries")
spatial.cells_to_spatialdata(sdata, broad, region="nucleus_boundaries")
```

`cells.to_spatialdata(sdata, region=...)` and `result.to_spatialdata(sdata, region=...)` are the
object-oriented equivalents.

The resulting table contains the same ALHENA information as the `.h5ad` representation, together
with the SpatialData region information required to link each row to its nucleus. Cell
identifiers that do not match the selected nucleus element are refused, and nothing is written.
Each annotation is named after its axis unless `name=` is given, and annotations of other cells
are refused, as [Several annotations of the same cells](#several-annotations-of-the-same-cells)
describes. The table is stored as `alhena_cells` unless another name is specified.

**Saving.** Use `sdata.write(path)` to write a new SpatialData object. If the object was opened from
an existing store and only a newly added element needs to be persisted, write that element
explicitly:

```python
sdata.write_element("alhena_cells")
```

After an annotation is added to a table already in the store, write the table again.
`write_element()` does not overwrite an element of the store the object was read from, so delete
the copy on disk first. The table is held in memory, so nothing is lost:

```python
sdata.delete_element_from_disk("alhena_cells")
sdata.write_element("alhena_cells")
```

## Moving results between environments

The `.h5ad` files written by ALHENA use standard AnnData storage and can be transferred between
compatible Python environments. This allows, for example, cell embeddings to be computed on one
machine and annotated later on another without rerunning the image encoder.

For the supported package versions, see [Installation](../installation.md).

---

**Next:** [Tutorials](../tutorials/index.md)
