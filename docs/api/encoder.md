# Encoder

`AlhenaEncoder` combines the UNI2-h image encoder with the ALHENA low-rank adaptation and
projection heads.

Three methods cover the main encoding levels:

| Method | Level | Returns |
|---|---|---|
| `encode_tiles()` | tile | `TileEmbeddings` with embeddings, identifiers, spatial bounds when available, and provenance |
| `encode_cells()` | cell | `CellEmbeddings` with pre-projection and projected cell representations, positions when available, and provenance |
| `encode()` | low-level tile operation | embedding matrix only |

`encode_tiles()` and `encode_cells()` are the high-level analysis workflows. `encode()` is the
lower-level operation for in-memory arrays when only the embedding matrix is required.

!!! note "Two sets of tiles, told apart"
    The tiles `encode_tiles` reads are the tile-level result: one embedding per tile of a sampling
    of the image. The tiles `encode_cells` reads provide context for the cells, so by default they
    cover only the tiles those cells require rather than forming a sampling of the image. Both
    come back as `TileEmbeddings`, and `purpose` says which is which, in memory and in the table
    `write_h5ad` writes. See
    [Dedicated tiles and cell-context tiles](../outputs/tile-embeddings.md#dedicated-tiles-and-cell-context-tiles).

!!! note "Image geometry is recorded with the result"
    Tile padding and cropping can affect the representations produced by the encoder. High-level
    result objects retain `pad_fraction` and `crop_fraction`, and the relevant encoding methods
    emit geometry warnings when appropriate.

    See [Geometry](../concepts/geometry.md) for interpretation.

::: alhena.encoder.AlhenaEncoder

::: alhena.encoder.load_bundle

::: alhena.encoder.LoRAQKV

::: alhena.encoder.insert_adapters

::: alhena.encoder.load_adapter_weights

::: alhena.encoder.merge_adapters

::: alhena.encoder.build_backbone

::: alhena.encoder.build_heads

::: alhena.encoder.bundle_identity

::: alhena.encoder.adapter_key

::: alhena.encoder.ADAPTER_NAMES

::: alhena.encoder.DEFAULT_BATCH_SIZE

::: alhena.encoder.TILE_EMBEDDING_CONTRACT

---

**Next:** [Cell typer](celltyper.md)
