# Tutorials

Six tutorials cover the main ALHENA workflows, from tile and cell embeddings to cell-type
annotation, multi-scale analysis, custom references, and annotation evaluation.

If you are new to ALHENA, run the [demo](../start/demo.ipynb) first to verify that the
installation works end to end.

| Tutorial | Task | Needs | Returns | Key concept |
|---|---|---|---|---|
| [1. Tile embeddings](01_tile_embeddings.ipynb) | describe the slide at the tile level | an image source | `TileEmbeddings` | a tile embedding and its spatial location |
| [2. Cell embeddings](02_cell_embeddings.ipynb) | describe the slide at the cell level | an image source and its nuclei | `CellEmbeddings` | pooling image tokens by nucleus |
| [3. Cell annotation](03_cell_annotation.ipynb) | give each cell a type | `CellEmbeddings` and a reference | `AnnotationResult` | retrieval against a reference; confidence |
| [4. Multi-scale slide analysis](04_multiscale_analysis.ipynb) | relate tile- and cell-level results on one slide | `TileEmbeddings`, `CellEmbeddings`, `AnnotationResult` | tile embedding clusters, cell embedding clusters, cell types, spatial neighborhoods, and cross-scale summaries | embedding space versus physical space |
| [5. Custom reference](05_custom_reference.ipynb) | build and use a custom single-cell reference | a single-cell atlas, and cells encoded earlier | a reference file, and cells annotated against it | building a reference |
| [6. Annotation evaluation](06_annotation_evaluation.ipynb) | evaluate annotations against independent labels | images, nuclei and known labels | `ScoreReport` and confidence sweep | confusions and confidence–coverage |

## How they connect

Tutorials 1 to 3 build the three main ALHENA outputs for one slide. Tutorial 4 combines these
outputs for multi-scale analysis. Tutorial 5 reuses previously computed cell embeddings and
changes only the single-cell reference used for annotation.

Tutorial 6 uses a separate, independently labeled example to show how annotation results can be
evaluated. It does not depend on the embeddings saved in Tutorials 1–5.

## The example inputs

Tutorials 1–5 follow a shared H&E slide example. Tutorial 1 encodes image tiles, while the
subsequent tutorials reuse nucleus outlines, saved embeddings, or annotation results as needed.
An optional example in Tutorial 2 introduces label maps and centroids. Tutorial 6 uses small
image fields with nucleus label maps and known labels. These are example input formats rather than
requirements. Other supported image and nucleus representations
change only the loading step; the downstream ALHENA calls remain the same. See
[Inputs](../inputs/index.md).

Set the paths in the first code cell of each tutorial to your own data. Workflows that compute
embeddings from H&E require the UNI2-h model weights. See
[Set up the UNI2-h model weights](../start/getting-started.md#2-set-up-the-uni2-h-model-weights).

---

**Next:** [1. Tile embeddings](01_tile_embeddings.ipynb)
