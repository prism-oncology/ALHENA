# Types

The objects ALHENA hands back. Three are the analytical results:

| Type | One row is | Returned by | Explained on |
|---|---|---|---|
| `TileEmbeddings` | a tile | `encode_tiles` | [Tile embeddings](../outputs/tile-embeddings.md) |
| `CellEmbeddings` | a cell | `encode_cells` | [Cell embeddings](../outputs/cell-embeddings.md) |
| `AnnotationResult` | a cell with its reference-based annotation | `annotate` | [Cell annotations](../outputs/cell-annotations.md) |

The others describe geometry, references, scores and inputs. Every class here is a dataclass, so
a result can be read field by field, and every field is documented below.

!!! note "Geometry metadata"
    `ResampleInfo`, `GeometryReport`, and `TileEmbeddings` expose `pad_fraction` and
    `crop_fraction`, which summarize how input fields were fitted to the model canvas. See
    [Geometry](../concepts/geometry.md).

!!! note "Tile spatial metadata"
    `TileEmbeddings` records the encoding geometry and the encoder identity. When tile
    coordinates are available, it also records spatial bounds, centers, and the coordinate frame,
    with its physical scale when the resolution is declared. `purpose` distinguishes dedicated
    tile-level results from tile embeddings generated as cell context.

!!! note "Cell provenance"
    `CellEmbeddings` records the encoder identity, `cell_space_id`, image geometry, membership
    provenance, and spatial positions when available. This allows saved cells to be annotated
    later without rerunning the image encoder. `AnnotationResult.cells` retains the cells used for
    the annotation.

::: alhena.types.TileEmbeddings

::: alhena.types.CellEmbeddings

::: alhena.types.AnnotationResult

::: alhena.types.TILE_PURPOSES

::: alhena.types.BundleInfo

::: alhena.types.ResampleInfo

::: alhena.types.GeometryReport

::: alhena.types.ReferenceInfo

::: alhena.types.SelfCheckReport

::: alhena.types.ScoreReport

::: alhena.types.MEMBERSHIP_COLUMNS

::: alhena.types.TileInput

::: alhena.types.NucleiInput

---

**Next:** [Errors and warnings](errors.md)
