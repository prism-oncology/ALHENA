# FAQ

Short answers to common questions about ALHENA inputs, outputs, installation, geometry,
annotation, and performance.

Most answers link to the page containing the full explanation.

!!! tip "Useful checks before a large run"
    - `alhena info` checks the installation and available resources.
    - `alhena.describe_geometry()` checks the physical image geometry.
    - `alhena reference describe` reports class support for a reference.

## Tiles, cells and analysis

### What is the difference between encode() and encode_tiles()?

`encode_tiles()` is the high-level tile workflow. It accepts a `TileSource`, processes it in
batches, and returns `TileEmbeddings` with identifiers, spatial bounds when available, and
provenance.

`encode()` is a lower-level array operation that returns only the embedding matrix.

See [Advanced: low-level `encode()`](outputs/tile-embeddings.md#advanced-low-level-encode).

### Why does my TileEmbeddings have no spatial coordinates?

Spatial coordinates are available only when the input tells ALHENA where each tile lies in a
shared image coordinate system. A directory or raw tile array has no such information by default.

Use a whole-slide or large-image source, or provide tile origins explicitly. See
[Tiles without origins](inputs/coordinates.md#tiles-without-origins).

### What is the difference between purpose="tiles" and purpose="cell_context"?

`purpose="tiles"` identifies a dedicated result from `encode_tiles()`.

`purpose="cell_context"` identifies tile embeddings generated during `encode_cells()` to provide
context for cells. These tiles may cover only the regions required by the supplied nuclei. With
`all_tiles=True` they cover every tile, and equal the embeddings of `encode_tiles()` at the same
`batch_size`.

`purpose` is provenance metadata, not a parameter selected by the user. See
[Dedicated tiles and cell-context tiles](outputs/tile-embeddings.md#dedicated-tiles-and-cell-context-tiles).

### Why did tile encoding warn about padding or cropping?

ALHENA operates on a fixed physical field of view. Smaller input fields require white padding and
larger fields require center cropping.

These transformations can affect the resulting representations, so check the recorded
`pad_fraction` and `crop_fraction`. See [Geometry](concepts/geometry.md).

### Should I use AnnData or SpatialData?

Use **AnnData** when you want an analysis table for Scanpy/Squidpy or a portable `.h5ad` file.

Use **SpatialData** when you want the H&E image, nuclei, tile regions, and ALHENA tables
registered together in shared coordinate systems.

SpatialData is optional; ALHENA does not automatically return one integrated object. See
[Persistence and interchange](outputs/persistence.md).

### Can I analyze ALHENA embeddings with Scanpy without genes?

Yes. The embeddings are in `obsm`, and Scanpy reads them with `use_rep`:
`sc.pp.neighbors(table, use_rep="X_alhena_backbone")` for cells, `use_rep="X_alhena_tile"` for
tiles. `X` is empty, so skip the steps meant for gene expression, such as normalization or the
selection of variable genes. See [Downstream analysis](outputs/downstream-analysis.md).

### Which cell embedding should I use?

Use `X_alhena_backbone`, ALHENA's **pre-projection cell embedding**, for general analysis such as
neighbors, UMAP, and clustering.

Use `X_alhena_cell`, the **projected shared cell-space embedding**, when examining the
representation used for reference-based annotation. See
[Which cell representation should I use?](outputs/cell-embeddings.md#which-cell-representation-should-i-use).

### When should I use Squidpy?

Use Squidpy for questions involving physical organization on the slide, such as cell-type
neighborhoods, distance-dependent co-occurrence, or the spatial distribution of tile or cell
embedding clusters. See
[Cell types in physical space](outputs/downstream-analysis.md#cell-types-in-physical-space).

### What is the difference between embedding space and physical space?

**Embedding space** describes similarity between ALHENA representations. **Physical space**
describes where observations lie on the slide.

Neighbors or clusters in one space do not imply proximity in the other. See
[Embedding space and physical space](outputs/downstream-analysis.md#embedding-space-and-physical-space).

### How are tile-level and cell-level analyses related?

Tile- and cell-level outputs can be related when they come from the same spatial context.

Cells retain the tile used to compute their representation, and both levels can retain positions
in the slide coordinate system. This allows predicted cell types and cell embedding clusters to be
summarized within tile embedding clusters or spatial regions.

See [Across scales](outputs/downstream-analysis.md#across-scales).

## Install, weights and cache

### ALHENA cannot find the UNI2-h weights. Where does it look?

ALHENA searches, in order:

1. an explicit `backbone=` / `--backbone` path;
2. `ALHENA_BACKBONE`;
3. the local Hugging Face model cache.

To use a copy stored elsewhere, set `ALHENA_BACKBONE` to the path of the weight file:

```bash
export ALHENA_BACKBONE=/path/to/UNI2-h/pytorch_model.bin
```

`find_backbone()` returns the file this lookup finds, and raises `BackboneNotFoundError` when it
finds none:

```python
from alhena.hub import find_backbone

print(find_backbone())
```

UNI2-h is distributed separately by its authors. See
[UNI2-h model weights](installation.md#the-uni2-h-model-weights).

### Do I need a GPU?

No. ALHENA can run on CPU, although image encoding is faster on a supported GPU. CUDA is selected
automatically when available; Apple silicon can use `device="mps"` explicitly.

### Do I get the same results on another computer?

Almost. Between processor families, such as Intel or AMD and Apple silicon, the resizing of a tile
can differ by one gray level, which can change a small share of cell-type annotations, mostly
low-confidence ones. Compare runs from one processor family when exact agreement matters.

### Can I run with no network at all?

Yes, if the required resources are already available locally.

Image encoding requires the UNI2-h weights and ALHENA encoder bundle. Cell annotation additionally
requires the selected reference. Annotating previously saved `CellEmbeddings` requires only the
compatible reference.

`alhena download` can populate the ALHENA cache in advance, or local resources can be supplied
directly with `--bundle`, `--reference`, and `--backbone`. See
[The cache](installation.md#the-cache).

### Where does the cache live, and how do I move it?

ALHENA resources use the first location that is set among `cache_dir`, `ALHENA_CACHE`,
`$XDG_CACHE_HOME/alhena`, and the platform-specific user cache.

UNI2-h uses a separate lookup mechanism and is not stored in the ALHENA cache.

```bash
export ALHENA_CACHE=/scratch/alhena
```

See [The cache](installation.md#the-cache).

### A package I installed still reports as missing. What now?

Optional dependencies are imported only when the corresponding functionality is used. If an
installed dependency still appears to be missing, try importing it directly to expose the
underlying import error:

```bash
python -c "import tiffslide"
```

See [Optional extras](installation.md#optional-extras).

## Resolution and tiles

### What do I pass as mpp?

Pass the physical pixel size of the image arrays being supplied to ALHENA, in microns per pixel.

If tiles were extracted from a downsampled pyramid level or resized, use the resolution of those
resulting pixels rather than the original slide resolution or scanner magnification.

See [Input resolution (`mpp`)](concepts/geometry.md#input-resolution-mpp).

### My slide declares a resolution and read_slide refuses it. What do I pass?

If the slide metadata is missing or incorrect, pass the verified full-resolution pixel size
explicitly:

```python
from alhena.io import read_slide

tiles = read_slide("export.tif", mpp=0.25)
```

An explicit `mpp=` overrides the slide metadata. See the advanced slide-resolution section on
[Images](inputs/images.md#how-a-slides-resolution-is-read).

### Every run on my 20x slides warns that the tiles are coarser. Can I silence it?

The warning concerns **pixel resolution**, not tile width. A 0.5 µm/px image is coarser than
ALHENA's 0.425 µm/px input resolution regardless of the number of pixels in each tile.

Use a finer pyramid level when one is available. Otherwise, the image must be upsampled and the
warning records that fact. Once that is understood, the warning can be silenced by its wording:

```python
import warnings

from alhena import GeometryWarning

warnings.filterwarnings("ignore", message="Tiles are at .* coarser than", category=GeometryWarning)
```

See [Advanced: geometry warnings](concepts/geometry.md#advanced-geometry-warnings).

### My fields are small in microns. Does that matter?

A field smaller than 95.2 µm does not fill the model canvas and is padded with white pixels.

`describe_geometry()` reports this as `pad_fraction`. Prefer tiles covering approximately the full
model field rather than altering `mpp` to avoid padding.

See [A small field pads with white](concepts/geometry.md#a-small-field-pads-with-white).

### My fields are large in microns. Can I pass one whole?

A field larger than 95.2 µm is center-cropped before inference, so tissue and nuclei outside the
retained central region do not reach the model.

If the input is one large image, use `read_image()` and let ALHENA divide it into appropriate
tiles rather than passing the complete image as one tile.

See [A large field is cropped](concepts/geometry.md#a-large-field-is-cropped).

### Can I pool tile embeddings from two cohorts?

Tile embeddings from different cohorts can be compared only with care. Differences in image
resolution, padding, cropping, stain characteristics, and tissue selection can introduce
systematic variation unrelated to biology.

In particular, compare `pad_fraction` and `crop_fraction` and use consistent tile geometry.

Geometry consistency is necessary but does not by itself guarantee that cohort effects are
absent. See
[Advanced: comparing slides and cohorts](outputs/downstream-analysis.md#advanced-comparing-slides-and-cohorts).

### Can I work on a canvas other than 224 pixels?

Not with the released ALHENA model. Its geometry and 16 × 16 token grid are defined for the
224 × 224 model canvas. Changing the canvas would change the token geometry used by cell pooling.

See
[Advanced: why the canvas size is fixed](concepts/geometry.md#advanced-why-the-canvas-size-is-fixed).

## Nuclei and cells

### What happens when two cells carry the same identifier?

Cell identifiers are interpreted globally. Reusing an identifier for two different biological
cells can cause ALHENA to treat them as duplicate observations of the same cell and keep only one
of them.

`read_label_maps()` and the command line renumber per-tile label maps by default to prevent this.
Label maps passed to `encode_cells()` or `annotate()` as arrays are read as they are. Without tile
origins, an identifier found in two tiles is refused. With them, a `GeometryWarning` names an
identifier whose pixels in two tiles do not meet. If identifiers are already globally unique, pass
`renumber=False` to `read_label_maps()`, or `--nuclei-ids cohort` on the command line.

See [Identifiers](inputs/nuclei.md#identifiers).

### Why are my cell identifiers refused before anything is embedded?

Cell identifiers must be consistently represented as integers or text. Mixed or unsupported
identifier types are refused before image encoding begins.

See [Identifiers](inputs/nuclei.md#identifiers).

### My membership table is refused for its geometry. What do I pass?

A membership table created by ALHENA records the image resolution and grid used when nucleus
masks were mapped onto image tokens. Reusing that table with incompatible tile geometry is
refused because the nuclei would no longer align with the same image tokens.

Build the membership table using the same image resolution and tile geometry that will be used for
encoding. See
[Advanced: membership geometry checks](concepts/cell-pooling.md#advanced-membership-geometry-checks).

### Some of my nuclei have no row in the result. Where did they go?

A nucleus has no result row when it cannot be associated with any image token. This can happen
after substantial cropping or mask downsampling, or when overlapping nuclei leave it no pixel of
its own.

`result.meta["n_cells_without_tokens"]` records the count. If many nuclei are missing, check image
geometry first.

See
[Advanced: inspect a membership table](concepts/cell-pooling.md#advanced-inspect-a-membership-table).

### Some rows have an empty label and NaN confidence. What does that mean?

These cells have an embedding row but did not receive a valid annotation. They remain in the
result with `valid=False`, an empty label, and `confidence=NaN`.

This is different from an input nucleus that has no output row at all. See
[Cells without an annotation](concepts/cell-typing.md#cells-without-an-annotation).

### Do I need a segmentation?

Cell embeddings and cell-type annotation require nucleus locations or segmentation; tile
embeddings do not.

ALHENA does not perform segmentation itself. Polygons, label maps, masks, and centroids can all be
used. See [Supported nucleus representations](inputs/nuclei.md#supported-nucleus-representations).

If you have no segmentation yet, sopa can segment a whole slide with StarDist, and ALHENA reads
the outlines directly. See
[Segment a slide with sopa and StarDist](inputs/nuclei.md#segment-a-slide-with-sopa-and-stardist).

### My nuclei are written in microns, or against another image. How do I read them?

If nuclei are not expressed in the same pixel coordinate system as the H&E image, describe their
coordinate system explicitly.

- use `units="um"` for coordinates in microns;
- use `transform=` for coordinates registered to another image;
- use `frame="origins"` for one set of whole-image centroids expressed in the tile-origin frame.

See [Coordinates](inputs/coordinates.md).

### What does a cell embedding capture?

A cell embedding is obtained by pooling contextualized image tokens according to the pixels
covered by the nucleus.

The pooling weights are nucleus-specific, but each image token has already incorporated
information from the surrounding tile. The resulting representation therefore reflects the
nucleus in local tissue context rather than an isolated nuclear crop.

Use `X_alhena_backbone` as the pre-projection cell representation for downstream analysis. See
[What information enters a cell embedding](concepts/cell-pooling.md#what-information-enters-a-cell-embedding).

## Reading cell-type annotations

### Why is the lowest confidence I ever see 0.2?

`confidence` is the fraction of the normalized reference vote assigned to the winning class. Its
minimum is therefore `1 / number_of_classes`.

This gives a floor of 0.2 for `lineage_v1` and 1/3 for `compartment_v1`. Confidence thresholds
should therefore be interpreted separately for the two axes. See
[Confidence is a class mass, not a similarity](concepts/cell-typing.md#confidence-is-a-class-mass-not-a-similarity).

### Can I threshold on neighbor_similarity instead?

Do not use the absolute `neighbor_similarity` value as a universal annotation threshold. It is a
cross-modal cosine similarity used for retrieval and is not calibrated as a probability or general
quality score.

Use `confidence` to summarize agreement among the retrieved classes, and calibrate any filtering
threshold on appropriate labeled data.

### How does the composition of the reference atlas affect cell-type annotation?

Annotation depends on the local reference neighbors retrieved for each query cell rather than
directly on the global atlas class proportions.

Reference composition can still influence predictions through local density in the shared cell
space: a more densely represented class may contribute more of the nearest neighbors in an
overlapping region.

Inspect reference class support when interpreting annotation composition. See
[How reference composition affects annotation](concepts/references.md#how-reference-composition-affects-annotation).

### A class I expected never appears in the annotations. Why?

Check whether that class is represented in the selected reference. A class with no reference cells
cannot be predicted; thinly supported classes have more limited neighbor support.

```python
import alhena

info = alhena.describe_shard(tissue="Tonsil", axis="lineage_v1")
print(info.missing_classes, info.thin_classes)
```

See [Class support](concepts/references.md#class-support).

### What happens to a cell whose type the reference does not have?

ALHENA chooses among the classes represented by the selected reference and does not automatically
introduce an `Unknown` class.

Confidence measures agreement among available reference classes and is not a novelty detector.
Biological properties absent from the annotation taxonomy therefore require a reference whose
labels explicitly represent them.

See
[Categories not represented by an axis](concepts/references.md#categories-not-represented-by-an-axis).

### Can I compare a compartment_v1 number with a lineage_v1 one?

No. `lineage_v1` and `compartment_v1` define different classification tasks and have different
class sets and confidence floors. Interpret their metrics separately and always report which axis
was used. See [The two class axes](concepts/cell-typing.md#the-two-class-axes).

### My scores are near the commonest-class baseline but the annotations look right. What should I check?

First verify that the external ground-truth labels are aligned to `result.cell_id`. `score()`
checks vector length but cannot determine whether the labels are in the correct row order.
`align_truth()` orders them by cell identifier:

```python
truth = alhena.align_truth(result, known)
```

See
[Advanced: evaluation with known labels](concepts/cell-typing.md#advanced-evaluation-with-known-labels).

## References and the encoder

### What is CellSpaceMismatchError telling me?

The reference belongs to a different learned shared cell space than the encoded cells, or than the
encoder.

ALHENA checks compatibility through `cell_space_id` because embedding width alone is
insufficient: incompatible releases may still both produce 512-dimensional vectors.

Use a reference compatible with the encoder or build the reference using the corresponding ALHENA
release. See [Reference compatibility](concepts/references.md#reference-compatibility).

### Can I retrieve against two references at once?

No. ALHENA performs each annotation run against one reference. Concatenating references would
change the retrieval pool and make results depend on how the references were combined.

Run separate annotations when several references need to be compared. See
[Use one reference for each annotation run](concepts/references.md#use-one-reference-for-each-annotation-run).

### An EnvelopeWarning fired. Do I have to stop?

Not necessarily. `EnvelopeWarning` indicates that the selected reference or retrieval
configuration lies outside conservative operating bounds declared by ALHENA. It is a condition to inspect, not
an automatic reason to discard the result.

See
[Reference coverage and retrieval warnings](concepts/references.md#reference-coverage-and-retrieval-warnings).

### How do I turn ALHENA's warnings into errors?

Every warning ALHENA raises subclasses `AlhenaWarning`, and therefore `UserWarning`. Filter on the
base class to catch every one, or on `GeometryWarning`, `ReferenceCoverageWarning` and
`EnvelopeWarning` separately. Each changes how a result should be read, so a pipeline is a good
place to raise them.

```python
import warnings

from alhena import AlhenaWarning

warnings.simplefilter("error", AlhenaWarning)
```

[Errors and warnings](api/errors.md) lists every one, with what it signals.

### Why does the same warning come back on every pass of a loop?

Python's record of which warnings it has already shown can be reset by code executed inside other
libraries. If a known warning is repeatedly emitted inside a loop, filter that specific warning
category or message rather than disabling all ALHENA warnings.

```python
import warnings

from alhena import GeometryWarning

warnings.filterwarnings("ignore", message="Tiles are at", category=GeometryWarning)
```

### My tissue is not one of the fifteen. What can I do?

ALHENA provides prebuilt references for fifteen human tissue names. Use:

```python
import alhena

print(alhena.available_tissues())
```

Some tissues also have accepted aliases. See
[Tissue aliases](concepts/references.md#tissue-aliases).

If no appropriate prebuilt reference exists, you can still compute tile and cell embeddings, or
build a custom reference from a labeled single-cell atlas. Do not substitute an unrelated
neighboring organ solely because a reference is available for it.

### Can I use ALHENA on tissue from another species?

The prebuilt references are human single-cell atlases and should not be used as
species-appropriate references for other organisms.

Tile and cell embeddings do not require a reference, but ALHENA itself has not been validated on
non-human H&E tissue.

Building a custom reference for another species is not automatically sufficient because the
released single-cell projection was trained with the human single-cell representation pipeline.
See [Prebuilt references](concepts/references.md#prebuilt-references) and
[Step 1: embed the single-cell atlas](concepts/references.md#step-1-embed-the-single-cell-atlas).

## Memory and time

### How much memory does a run take?

Image encoding is batched, so memory used by the forward pass scales mainly with `batch_size`
rather than with the total number of tiles.

Cell-level workflows additionally retain their returned cell embeddings, so very large numbers of
cells can still require substantial host memory. Each cell stores a 1536-dimensional
pre-projection embedding and a 512-dimensional projected embedding.

Reference retrieval runs on several threads, and each thread holds about 0.4 GB while it works.

Reduce `batch_size` when forward-pass memory is limiting, and persist large results to disk when
the returned embedding matrices dominate memory. Pass a smaller `workers` to `annotate()` when
retrieval memory is limiting.

??? note "Advanced: memory per tile and per cell"
    In `encode_cells()` and tile-based `annotate()`, the current batch holds these arrays for each
    of its tiles. `encode_tiles()` holds only the first two.

    | Held | Shape | Per tile |
    |---|---|---|
    | Prepared tiles | `(batch, 3, 224, 224)` float32 | 0.57 MiB |
    | Features of the forward pass | `(batch, 265, 1536)` float32 | 1.55 MiB |
    | Patch tokens, copied out for pooling | `(batch, 256, 1536)` float32 | 1.5 MiB |

    That is about 3.6 MiB per tile, or about 116 MiB at the default `batch_size` of 32, whatever
    the number of tiles. The model weights and the intermediate activations inside the forward
    pass are not included.

    The returned embeddings grow with the input: 6 KiB per cell for the pre-projection
    embedding, 2 KiB per cell for the projected embedding, and 6 KiB per tile read for the tile
    embeddings. A slide of 500,000 nuclei therefore returns about 4 GiB of cell embeddings. The
    membership table adds a few tens of bytes per nucleus and token it touches. A loaded
    reference adds 2 KiB per labeled cell, because its half-precision rows are converted to
    float32 once, at load. Retrieval adds 384 MiB per thread: each thread holds the similarities
    of 2,048 cells to 16,384 reference cells, 128 MiB, and their ranking, 256 MiB, at a time.

### I lowered batch_size and memory did not move. Why?

Lowering `batch_size` reduces the amount of data processed simultaneously by `encode_tiles()`,
`encode_cells()`, and tile-based `annotate()`.

The low-level `encode()` function is different because the complete in-memory input is prepared
before its batched forward pass. Reducing `batch_size` therefore does not remove the memory
occupied by that prepared input.

### Can I annotate the same cells with another reference without running the encoder again?

Yes. Cell encoding is independent of the single-cell reference.

Save `CellEmbeddings` once and reuse them with any compatible reference or annotation axis:

```python
import alhena

cells = alhena.load_encoder().encode_cells(source, nuclei)
cells.write_h5ad("cells.h5ad")

cells = alhena.io.read_cells("cells.h5ad")
result = alhena.load_celltyper(tissue="Breast", axis="compartment_v1").annotate(cells)
```

```bash
alhena encode tiles/ --nuclei masks/ --mpp 0.425 --out cells.h5ad
alhena annotate cells.h5ad --tissue Breast --axis compartment_v1 --out annotations.csv
```

Compatibility is checked through `cell_space_id`. See
[Annotating cells again](concepts/cell-typing.md#annotating-cells-again).

The annotations of the same cells can then share one table, each under columns named after its
axis. See
[Several annotations of the same cells](outputs/persistence.md#several-annotations-of-the-same-cells).

### How long does a run take?

Runtime depends mainly on:

- image encoding, which scales with the number of tiles and benefits from GPU acceleration;
- reference retrieval, which scales with the number of cells and reference size, and runs on
  several processor threads at once: the processors available, at most eight, unless
  `workers=` or `--workers` sets another number;
- initial resource downloads when files are not already available locally.

Long-running stages report progress, most of them with an estimated completion time. Exact
runtime depends strongly on hardware and input size, so the documentation does not prescribe a
universal expected duration.

### Can I turn the progress bars off?

Set `progress=False` on an individual Python call, `ALHENA_DISABLE_PROGRESS_BARS=1` for ALHENA
progress process-wide, or `--quiet` for one CLI invocation. See [Progress](cli.md#progress).

---

**Next:** [Cite](cite.md)
