# Reading data

Functions for converting supported image, nucleus, coordinate, and saved-result formats into the
objects used by ALHENA.

Image readers return a `TileSource`, which reads tiles on demand. Nucleus readers convert
supported segmentation or centroid representations into the formats used by `encode_cells()` and
`annotate()`. Saved tile embeddings, cell embeddings and annotations can be reconstructed with
`read_tile_embeddings()`, `read_cells()` and `read_annotations()`, and `add_annotation()` adds an
annotation to a table of cells. `read_region()` reads a rectangle of a slide or an image to draw
results over it.

!!! warning "Cell identifiers must refer to unique cells"
    Reusing the same identifier for different cells can cause ALHENA to interpret them as
    repeated observations of one cell. `read_label_maps()` renumbers per-tile identifiers by
    default.

::: alhena.io.read_tiles

::: alhena.io.read_slide

::: alhena.io.read_region

::: alhena.io.read_image

::: alhena.io.read_tile_origins

::: alhena.io.read_nuclei

::: alhena.io.read_label_maps

::: alhena.io.read_label_image

::: alhena.io.label_map_from_masks

::: alhena.io.read_centroids

::: alhena.io.read_polygons

::: alhena.io.tile_bounds

::: alhena.io.as_tile_source

::: alhena.io.global_cell_ids

::: alhena.io.TileSource

::: alhena.io.ArrayTileSource

::: alhena.io.DirectoryTileSource

::: alhena.io.HDF5TileSource

::: alhena.io.SlideTileSource

::: alhena.io.ImageTileSource

::: alhena.io.NUCLEI_KINDS

::: alhena.io.CENTROID_FRAMES

::: alhena.io.COORDINATE_UNITS

::: alhena.io.MULTIPART_RULES

::: alhena.io.EDGE_RULES

::: alhena.io.IMAGE_FRAME

::: alhena.io.PART_SEPARATOR

::: alhena.io.CELL_ID_STRIDE

::: alhena.io.IMAGE_SUFFIXES

::: alhena.io.LABEL_SUFFIXES

::: alhena.io.HDF5_SUFFIXES

::: alhena.io.TABLE_SUFFIXES

::: alhena.io.GEOJSON_SUFFIXES

::: alhena.io.BOUNDARY_TABLE_SUFFIXES

::: alhena.io.TRANSFORM_SUFFIXES

::: alhena.io.SLIDE_SUFFIXES

::: alhena.io.UNREAD_SLIDE_FORMATS

::: alhena.io.H5_IMAGE_DATASET

::: alhena.io.H5_COORDS_DATASET

::: alhena.io.H5_BARCODE_DATASET

::: alhena.io.H5_FRAME_SIDE_ATTRIBUTE

::: alhena.io.H5_TILE_SIDE_ATTRIBUTE

::: alhena.io.H5_SCALE_RECORDS

::: alhena.io.H5_LABEL_DATASET

::: alhena.io.CENTROID_COLUMNS

::: alhena.io.VERTEX_COLUMNS

::: alhena.io.ORIGIN_COLUMNS

::: alhena.io.MPP_X_PROPERTIES

::: alhena.io.MPP_Y_PROPERTIES

::: alhena.io.OME_LENGTH_UNITS

## Tile embeddings on disk

One table layout holds tile embeddings: the embedding of every tile, its spatial bounds in the
source image and their center when available, and the provenance of the pass.
`TileEmbeddings.write_h5ad` writes it, `to_anndata` returns it in memory, and
`read_tile_embeddings` reads it back.
[The table layouts](../outputs/persistence.md#tiles) set it out.

::: alhena.io.read_tile_embeddings

::: alhena.io.TILES_SUFFIX

::: alhena.io.TILE_ID_COLUMN

::: alhena.io.TILE_BOUNDS_COLUMNS

::: alhena.io.TILE_EMBEDDING_KEY

::: alhena.io.TILES_PROVENANCE_KEY

## Cells on disk

One table layout holds cells alone, annotations together with their underlying cell embeddings,
and the table `cells_to_spatialdata` stores. `CellEmbeddings.write_h5ad` and
`AnnotationResult.write_h5ad` write it, and `to_anndata` returns it in memory. `read_cells` reads
the cells back out of it with their provenance, and `read_annotations` one annotation together with
its cells. The cells are held once, and `add_annotation` adds further annotations of them, each
under fields named after it.
[The table layouts](../outputs/persistence.md#cells) set it out.

::: alhena.io.read_cells

::: alhena.io.read_annotations

::: alhena.io.add_annotation

::: alhena.io.CELLS_SUFFIX

::: alhena.io.CELL_ID_COLUMN

::: alhena.io.TILE_INDEX_COLUMN

::: alhena.io.NUCLEUS_PX_COLUMN

::: alhena.io.CANDIDATE_TILES_COLUMN

::: alhena.io.EMBEDDING_VALID_COLUMN

::: alhena.io.LABEL_COLUMN

::: alhena.io.CONFIDENCE_COLUMN

::: alhena.io.VALID_COLUMN

::: alhena.io.CLASS_ENTROPY_COLUMN

::: alhena.io.NEIGHBOR_ENTROPY_COLUMN

::: alhena.io.BACKBONE_KEY

::: alhena.io.CELL_SPACE_KEY

::: alhena.io.SPATIAL_KEY

::: alhena.io.MASS_KEY

::: alhena.io.CELLS_PROVENANCE_KEY

::: alhena.io.PROVENANCE_KEY

::: alhena.io.ANNOTATIONS_KEY

---

**Next:** [Preprocessing](preprocess.md)
