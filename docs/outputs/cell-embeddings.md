# Cell embeddings

`encode_cells()` returns a `CellEmbeddings` object with one row per cell. Each row contains a
pre-projection image-derived cell embedding, its projected representation in the shared cell space
used for annotation, and spatial information when available.

!!! tip "Which representation should I use?"
    - Use the **pre-projection embedding** (`X_alhena_backbone`) for general downstream analysis
      such as neighbors, UMAP, and clustering.
    - Use the **projected cell-space embedding** (`X_alhena_cell`) to inspect the representation
      used for reference-based annotation.

```python
cells = encoder.encode_cells(source, nuclei)

print(len(cells), int(cells.valid.sum()))       # cells, and how many can be annotated
print(cells.backbone.shape)                     # (n_cells, 1536)
print(cells.cell_space.shape)                   # (n_cells, 512)
print(cells.centroids_frame, cells.centroids_mpp)
```

## What a cell embedding represents

ALHENA derives each cell representation by pooling image tokens according to the pixels covered by
the nucleus. Each token corresponds to approximately 5.95 µm of the image and is itself computed in
the context of the surrounding tile.

The resulting cell representation therefore combines information from the nucleus with local
tissue context. Cells whose nuclei overlap some of the same image tokens can consequently have
correlated representations.

Each cell stores two related embeddings:

| Representation | Width | Role |
|---|---:|---|
| `backbone` / `X_alhena_backbone` | 1536 | pre-projection image-derived cell embedding used for general downstream analysis |
| `cell_space` / `X_alhena_cell` | 512 | projection of the cell embedding into the shared space used for reference-based annotation |

`X_alhena_cell` is obtained from the pre-projection cell representation through the learned
projection head `img_tok_proj`.

See [The shared cell space](../concepts/how-it-works.md#the-shared-cell-space) and
[Cell pooling](../concepts/cell-pooling.md).

## Which cell representation should I use?

For general analysis of image-derived cellular variation, use `X_alhena_backbone`. Despite its API
name, this is best understood as ALHENA's **pre-projection cell embedding**. It is derived from H&E
features learned under molecular supervision, but it is not gene expression.

Typical uses include:

- nearest-neighbor graphs;
- dimensionality reduction;
- clustering into cell embedding clusters;
- analysis of cellular heterogeneity.

Use `X_alhena_cell` when the analysis specifically concerns the shared representation used for
cell-type annotation. This space is where image-derived cells are compared with the single-cell
reference.

`X_alhena_cell` is a projection of `X_alhena_backbone`; the two representations are therefore
related rather than independent.

## What CellEmbeddings holds

```text
CellEmbeddings
├── cell_id                  the identifier of each cell
├── backbone                 pre-projection cell embedding
├── cell_space               projected embedding used for annotation
├── valid                    whether the cell can be used for annotation
├── centroids                the position of each nucleus, where it can be placed
├── centroids_frame          the frame of those positions
├── centroids_mpp            the microns one unit of that frame spans
├── tile_index, alpha,       the tile each cell was read from, its nucleus pixels there,
│   n_candidate_tiles        and how many tiles held the nucleus
├── encoder_id,              the encoder and cell space identities
│   cell_space_id
├── backbone_digest          the digest of the UNI2-h model weights
├── geometry                 the geometry the tiles presented to the model
├── membership_source,       how the nuclei were given, and whether their shapes were
│   membership_approximate   approximated
├── n_cells_without_tokens   how many nuclei reached no token
└── tiles                    the embeddings of the tiles the cells were read from
```

| Field | Holds |
|---|---|
| `cell_id` | the identifier of each cell; rows are ordered by it |
| `backbone` | pre-projection cell embeddings, `(n_cells, 1536)` float32 |
| `cell_space` | projected embeddings in the shared cell space, `(n_cells, 512)` float32 |
| `valid` | `False` where the embedding cannot be annotated: no nucleus pixel was pooled, or it is not finite |
| `centroids` | `(n_cells, 2)`, x then y, or `None` |
| `centroids_frame` | the frame of `centroids`, such as `"slide.svs, level 0"` |
| `centroids_mpp` | microns per unit of that frame |
| `tile_index` | the tile each cell was read from |
| `alpha` | the nucleus pixels supporting the cell in that tile |
| `n_candidate_tiles` | how many tiles held the nucleus |
| `encoder_id`, `cell_space_id` | the identities that pair the cells with a reference |
| `backbone_digest` | the digest of the UNI2-h model weights the cells were encoded with, or `None` |
| `geometry` | the geometry the tiles presented, as `describe_geometry` reports it; `cells.mpp` reads its resolution |
| `membership_source` | how the nuclei were given: `labels`, `centroids`, `polygons`, or `unknown` |
| `membership_approximate` | whether the nucleus shapes were approximated, as centroid disks are |
| `n_cells_without_tokens` | how many nuclei reached no token |
| `tiles` | a `TileEmbeddings` with `purpose="cell_context"` |

## Cells omitted or marked invalid

A nucleus that cannot be associated with any image token is omitted from the returned rows.
`cells.n_cells_without_tokens` records how many nuclei were excluded for this reason.

A cell may also remain present while being marked `valid=False`, for example when its embedding
cannot be used safely for annotation. Keeping the row allows the cell and its provenance to remain
inspectable.

See [Some of my nuclei have no row](../faq.md#some-of-my-nuclei-have-no-row-in-the-result-where-did-they-go).

## Cell positions

When the nucleus can be located in a shared image coordinate system, `CellEmbeddings` stores its
position in `centroids`.

Depending on the nucleus representation, this position corresponds to the centroid of the
segmented shape, the centroid of the labeled pixels, or the supplied centroid itself.
`centroids_frame` identifies the coordinate system and `centroids_mpp` gives its physical scale.

For label maps and centroids, spatial positions require tiles with known origins. See
[Where positions end up](../inputs/coordinates.md#where-positions-end-up).

## Tile context retained during cell encoding

`cells.tiles` contains the tile embeddings computed during the same forward pass as the cell
embeddings, with `purpose="cell_context"`.

By default, these are only the tiles needed for the supplied nuclei, so they may not provide
systematic coverage of the full image. For an independent tile-level analysis, use
`encode_tiles()`.

If `encode_cells(..., all_tiles=True)` is used, all tiles are processed during the cell pass,
avoiding a second encoder pass while retaining `purpose="cell_context"` as provenance. At the same
`batch_size`, these tile embeddings equal those of `encode_tiles()`. See
[Dedicated tiles and cell-context tiles](tile-embeddings.md#dedicated-tiles-and-cell-context-tiles).

## As an AnnData table

`cells.to_anndata()` returns the cells as an AnnData table. One observation is one cell.

| Where | Key | Holds |
|---|---|---|
| `obsm` | `X_alhena_backbone` | the pre-projection cell embeddings, `(n_cells, 1536)` |
| `obsm` | `X_alhena_cell` | the projected embeddings in the shared cell space, `(n_cells, 512)` |
| `obsm` | `spatial` | the position of each nucleus, x then y, where recorded |
| `obs` | `cell_id` | the identifiers, as integers or text |
| `obs` | `alhena_tile_index` | the tile each cell was read from |
| `obs` | `alhena_nucleus_px` | the nucleus pixels behind the cell there |
| `obs` | `alhena_n_candidate_tiles` | how many tiles held the nucleus |
| `obs` | `alhena_embedding_valid` | whether the embedding can be annotated |
| `uns` | `alhena_cells` | the provenance, including `centroids_frame` and `centroids_mpp` |
| `X` | | nothing: zero columns wide |

The index holds the identifiers as text, which AnnData requires. As for tiles, the embeddings sit
in `obsm`, not in `X`: they are representation features rather than measured variables such as
genes. The tile embeddings on `cells.tiles` are not part of this table.

`cells.to_spatialdata(sdata, region="nuclei")` writes this table into a SpatialData object that
holds the nuclei, tied to their element, before any annotation. See
[Writing into a SpatialData object](persistence.md#writing-into-a-spatialdata-object).

## Encode once, annotate many times

Cell encoding depends on the H&E image and nuclei, not on the single-cell reference. The resulting
`CellEmbeddings` can therefore be saved once and reused with different compatible references or
annotation axes without running the image encoder again:

```python
cells.write_h5ad("cells.h5ad")

cells = alhena.io.read_cells("cells.h5ad")
result = alhena.load_celltyper(tissue="Breast").annotate(cells)
```

`read_cells()` also restores the `cell_space_id`, allowing ALHENA to verify that the saved cells
and the selected reference use the same shared cell space.

---

**Next:** [Cell annotations](cell-annotations.md)
