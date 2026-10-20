# Python API

This section documents the public Python API directly from the ALHENA docstrings.

Most workflows use three pages:

- [Reading data](io.md) for H&E images and nuclei;
- [Encoder](encoder.md) for tile and cell embeddings;
- [Cell typer](celltyper.md) for reference-based cell annotation.

The returned `TileEmbeddings`, `CellEmbeddings`, and `AnnotationResult` objects are documented
under [Types](types.md).

| Page | Open it when |
|---|---|
| [Encoder](encoder.md) | You compute tile or cell embeddings, or want the adapters underneath. |
| [Cell typer](celltyper.md) | You annotate cells, from tiles and nuclei or from cells encoded earlier. |
| [Reading data](io.md) | Your images or nuclei sit in files or arrays, or you read saved results. |
| [Preprocessing](preprocess.md) | You check what a resolution does to a tile. |
| [Cells and pooling](cells.md) | You build the nucleus-to-token table, or pool yourself. |
| [Retrieval](annotate.md) | You run the lower-level reference-retrieval operator. |
| [References](reference.md) | You load a prebuilt or custom reference and inspect its class support. |
| [Building a reference](refbuild.md) | You build a custom reference from atlas embeddings and labels. |
| [Scoring](metrics.md) | You evaluate annotations against known labels. |
| [SpatialData](spatial.md) | Your data is in a SpatialData object, or you run sopa's patch pipeline. |
| [Plotting](viz.md) | You draw overlays, tile regions, compositions, matrices or legends. |
| [Types](types.md) | You need the fields of a result: `TileEmbeddings`, `CellEmbeddings`, `AnnotationResult`. |
| [Errors and warnings](errors.md) | You catch an ALHENA error, or filter its warnings. |
| [Constants](constants.md) | You need a fixed quantity: resolution, k, tau, the axes. |

Each heading names the public module to import a symbol from. On the Errors and Constants
pages, it names the private module that defines the symbol, and the page says where to import
it from. A re-exported symbol reaches either way, so `alhena.load_celltyper` and
`alhena.api.load_celltyper` are one function.

---

**Next:** [FAQ](../faq.md)
