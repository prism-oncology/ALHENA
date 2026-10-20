# Choose your task

Choose the output that matches your analysis. The sections below show which inputs are required,
which ALHENA call to use, and what it returns.

## Compute tile embeddings

Compute one embedding for each tissue tile, while retaining its position on the slide when
spatial coordinates are available.

```python
import alhena
from alhena import io

encoder = alhena.load_encoder()
source = io.read_slide("slide.svs")
tiles = encoder.encode_tiles(source)
tiles.write_h5ad("tiles.h5ad")
```

| | |
|---|---|
| **Needs** | an image source |
| **Returns** | `TileEmbeddings`: one embedding per tile, with its spatial bounds when available |
| **Command line** | `alhena encode slide.svs --out tiles.h5ad` |
| **Read next** | [Images](../inputs/images.md) · [Tile embeddings](../outputs/tile-embeddings.md) · [Tutorial 1](../tutorials/01_tile_embeddings.ipynb) |

Tile embeddings require neither nuclei nor a single-cell reference. `encode_tiles()` can process a
whole-slide image, a large image, pre-extracted tiles, or an image stored in a SpatialData object.

## Compute cell embeddings

Compute one embedding for each cell by pooling image features over its nucleus while retaining
the surrounding tissue context.

```python
import alhena
from alhena import io

encoder = alhena.load_encoder()
source = io.read_slide("slide.svs")
nuclei = io.read_polygons("nuclei.geojson", tiles=source)
cells = encoder.encode_cells(source, nuclei)
cells.write_h5ad("cells.h5ad")
```

| | |
|---|---|
| **Needs** | an image source and nuclei; if they use different coordinate systems, the transformation between them |
| **Returns** | `CellEmbeddings`: per cell, one pre-projection cell embedding for downstream analysis, one projected embedding in the shared cell space for annotation, and the associated spatial and provenance metadata |
| **Command line** | `alhena encode slide.svs --nuclei nuclei.geojson --out cells.h5ad` |
| **Read next** | [Nuclei](../inputs/nuclei.md) · [Coordinates](../inputs/coordinates.md) · [Cell embeddings](../outputs/cell-embeddings.md) · [Cell pooling](../concepts/cell-pooling.md) · [Tutorial 2](../tutorials/02_cell_embeddings.ipynb) |

Nuclei can be provided as polygons, label maps, or centroids. Without a segmentation, see
[If you have no nuclei yet](../inputs/nuclei.md#if-you-have-no-nuclei-yet).

## Annotate cells

Assign cell types to previously encoded cells using a single-cell reference. If `CellEmbeddings`
are already available, the H&E image does not need to be encoded again:

```python
import alhena

cells = alhena.io.read_cells("cells.h5ad")
typer = alhena.load_celltyper(tissue="Breast")
result = typer.annotate(cells)
```

To encode and annotate cells in one step, provide the image, nuclei, and reference:

```python
import alhena
from alhena import io

source = io.read_slide("slide.svs")
nuclei = io.read_polygons("nuclei.geojson", tiles=source)
typer = alhena.load_celltyper(tissue="Breast")
result = typer.annotate(source, nuclei)
cells = result.cells
```

| | |
|---|---|
| **Needs** | a reference, and either `CellEmbeddings` or an image source with its nuclei |
| **Returns** | `AnnotationResult`: a cell-type annotation, confidence, and validity flag for each cell, together with the `CellEmbeddings` used for annotation |
| **Command line** | `alhena annotate cells.h5ad --tissue Breast --out annotations.h5ad`, or `alhena annotate slide.svs --nuclei nuclei.geojson --tissue Breast --out annotations.h5ad` |
| **Read next** | [References](../inputs/references.md) · [Cell annotations](../outputs/cell-annotations.md) · [Tutorial 3](../tutorials/03_cell_annotation.ipynb) |

Once cells have been encoded, the same embeddings can be annotated against different references
or class axes without running the image encoder again.

When known labels are available, [Tutorial 6](../tutorials/06_annotation_evaluation.ipynb) shows
how to evaluate the annotations.

## Analyze the slide across scales

Combine tile embeddings, cell embeddings, and cell-type annotations to study the slide at multiple
spatial scales.

Tile and cell embeddings can be used for downstream analyses such as neighborhood graphs,
dimensionality reduction, and clustering. When spatial coordinates are available, these results
can also be mapped back to the H&E image and analyzed together with cell-type annotations.

| Result | Main use |
|---|---|
| `TileEmbeddings` | pre-projection tile embeddings for downstream analysis |
| `CellEmbeddings` | pre-projection cell embeddings for downstream analysis and projected cell-space embeddings used for annotation |
| `AnnotationResult` | reference-based cell-type annotations |

If the H&E image and nuclei are already stored in a SpatialData object, ALHENA can add the tile-
and cell-level results directly to the same object:

```python
from alhena import spatial

spatial.tiles_to_spatialdata(sdata, tiles, image_key="he")
spatial.cells_to_spatialdata(sdata, result, region="nuclei")
```

`image_key` names the image the tiles were cut from, and `region` names the nucleus element. With
`image_key`, the tile rectangles take the transformations of that image. This keeps the image,
nuclei, tile-level results, and cell-level results aligned in one coordinate system.

| | |
|---|---|
| **Needs** | tile- and cell-level results from the same slide |
| **Uses** | the AnnData representation of individual results with tools such as Scanpy and Squidpy, or SpatialData for integrated spatial analysis |
| **Read next** | [Downstream analysis](../outputs/downstream-analysis.md) · [Tutorial 4](../tutorials/04_multiscale_analysis.ipynb) |

## Build a custom reference

Build a reference from a labeled single-cell atlas to annotate cells using your own taxonomy.

```python
import alhena

typer = alhena.load_celltyper(reference="my_reference.safetensors", axis="my_taxonomy_v1")
cells = alhena.io.read_cells("cells.h5ad")
result = typer.annotate(cells)
```

A custom reference is built from a labeled single-cell atlas and can then be loaded in the same
way as a prebuilt reference. Previously computed `CellEmbeddings` can be annotated with the new
reference directly, without re-encoding the H&E image. Tutorial 5 walks through the
reference-building workflow.

| | |
|---|---|
| **Needs** | a single-cell atlas of raw counts and one label per cell, the UNI2-h model weights, and the atlas-embedding script of the [ALHENA repository](https://github.com/prism-oncology/ALHENA) |
| **Returns** | a reference file, used like a prebuilt reference |
| **Read next** | [References](../inputs/references.md#build-a-custom-reference) · [Tutorial 5](../tutorials/05_custom_reference.ipynb) |

## Common requirements

- **UNI2-h model weights.** Computing embeddings from H&E, and building a custom reference,
  require the UNI2-h weights. They are not needed when annotating previously computed
  `CellEmbeddings`. See
  [Set up the UNI2-h model weights](getting-started.md#2-set-up-the-uni2-h-model-weights).
- **Image resolution.** Image arrays without reliable resolution metadata require `mpp`, the
  physical pixel size in microns per pixel. See [Image resolution](../inputs/images.md#resolution).
- **A single-cell reference for annotation.** Cell annotation requires a compatible reference.
  The prebuilt ALHENA references are derived from human single-cell atlases; computing
  embeddings does not use a reference. See [References](../inputs/references.md).

---

**Next:** [Inputs](../inputs/index.md)
