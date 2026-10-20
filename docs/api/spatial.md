# SpatialData

Functions connecting ALHENA inputs and outputs with SpatialData.

- `tile_source()` exposes an image stored in SpatialData through the standard ALHENA `TileSource`
  interface.
- `cells_from_shapes()` converts a nucleus shapes element into the membership representation used
  for cell encoding.
- `tiles_to_spatialdata()` adds tile-level results to an existing SpatialData object.
- `cells_to_spatialdata()` adds the table of cells associated with a nucleus element, from the
  cells alone or from their annotations, then each further annotation of the same cells.
- `tile_embeddings()` and `Alhena` support the alternative sopa patch workflow. See
  [Advanced: sopa patch pipeline](../outputs/tile-embeddings.md#advanced-sopa-patch-pipeline).

See [Writing into a SpatialData object](../outputs/persistence.md#writing-into-a-spatialdata-object).

ALHENA tables retain the same embedding and annotation fields used by the AnnData representations,
with the additional SpatialData region/instance metadata required to connect observations to
spatial elements.

!!! note "Needs the spatial extra"
    `pip install "alhena[spatial]"` brings in spatialdata and sopa. See
    [optional extras](../installation.md#optional-extras).

::: alhena.spatial.tile_source

::: alhena.spatial.tiles_to_spatialdata

::: alhena.spatial.tile_embeddings

::: alhena.spatial.cells_from_shapes

::: alhena.spatial.cells_to_spatialdata

::: alhena.spatial.patch_bounds

::: alhena.spatial.Alhena

::: alhena.spatial.TILE_EMBEDDINGS_KEY

::: alhena.spatial.PATCHES_KEY

::: alhena.spatial.TILE_SHAPES_KEY

::: alhena.spatial.TILE_TABLE_KEY

::: alhena.spatial.DEFAULT_COORDINATE_SYSTEM

::: alhena.spatial.CELL_TABLE_KEY

::: alhena.spatial.REGION_KEY

::: alhena.spatial.INSTANCE_KEY

---

**Next:** [Plotting](viz.md)
