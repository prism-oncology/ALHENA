# Command line

Installing ALHENA adds the `alhena` command. It exposes the main workflows of the Python API:

- `alhena encode` computes tile embeddings or, with `--nuclei`, cell embeddings;
- `alhena annotate` performs reference-based cell-type annotation;
- the remaining commands inspect, download, validate, or demonstrate ALHENA resources.

Run:

```bash
alhena --help
```

or `-h` after any subcommand to see its available options.

## The six commands

| Command | Purpose | Needs |
|---|---|---|
| [`info`](#alhena-info) | report the installation, cache, UNI2-h weights, and available ALHENA resources | nothing |
| [`download`](#alhena-download) | download the encoder bundle and selected prebuilt references | the network |
| [`encode`](#alhena-encode) | compute tile embeddings or cell embeddings | the bundle, the UNI2-h weights |
| [`annotate`](#alhena-annotate) | assign reference-based cell types to nuclei or previously encoded cells | a reference; the bundle and the UNI2-h weights for tiles |
| [`reference`](#alhena-reference) | list, inspect, or validate references | a reference, for two of its three actions |
| [`demo`](#alhena-demo) | run the small PanNuke demonstration | the bundle, the UNI2-h weights, a reference |

Successful commands exit with status `0`. Every refusal is written to standard error: errors the
parser detects, such as an unknown option, exit with status `2`, and every other refusal with
status `1`. A missing optional Python dependency is reported the same way, and its message
identifies the extra that is required.

A command whose output needs nothing downloaded is shown with the block it prints. Every other
block is a field layout, with each value written as a placeholder, or a message `alhena` prints
literally.

??? note "The meaning of `--bundle` depends on the command"
    On `alhena download`, `--bundle` is a switch requesting that the published encoder bundle be
    downloaded.

    On `encode`, `annotate`, `info`, and `demo`, `--bundle` instead identifies the bundle to use,
    either as a local directory or repository name.

    See the option table of the individual command for the exact meaning.

??? tip "Running with no network"
    `--bundle` accepts either a local directory or an `owner/name` repository. If you provide a
    local bundle directory, a local reference file and the UNI2-h model weights, `encode` and
    `annotate` make no network requests. With those files on disk as well, `demo` is fully offline
    when a PanNuke fold is supplied with `--parquet`. An air-gapped machine therefore needs no
    download step once the required files are present. Annotating an existing table of cells
    requires only the reference file.

## Options several commands share

Five options recur, with the same meaning wherever they appear. The exception is `--bundle`: a
switch on `download`, and on `info` the bundle to report on, which it never downloads.

| Option | Takes | Default |
|---|---|---|
| `--backbone` | the UNI2-h weight file, or a directory holding one | `ALHENA_BACKBONE`, then the Hugging Face cache |
| `--bundle` | a directory holding the bundle files, or a repository `owner/name` | `prism-oncology/ALHENA` |
| `--cache-dir` | a directory, overriding `ALHENA_CACHE` | `ALHENA_CACHE`, then `$XDG_CACHE_HOME/alhena`, then the per-user cache |
| `--device` | a torch device string: `cpu`, `cuda`, or `mps` for the GPU of an Apple silicon Mac | a CUDA device where one is available, the processor otherwise |
| `-q`, `--quiet` | nothing; a switch | off: progress is reported |

The lookup of the UNI2-h weights and the cache order are set out in
[The UNI2-h model weights](installation.md#the-uni2-h-model-weights) and
[The cache](installation.md#the-cache).

`--cache-dir` names a cache using the standard Hugging Face repository layout. This is the same
layout populated by `download`. The bundle and registry are read from that cache only in this
layout. A directory that directly contains the bundle files is not a cache. Pass it with
`--bundle` instead.

Prebuilt reference files are the exception. `annotate`, `reference` and `demo` first look for the
reference file of a named tissue directly under the cache directory, as
`shards/<shard>.safetensors` or `<shard>.safetensors`, then fall back to the Hugging Face layout.
The cache directory is either the path supplied with `--cache-dir` or the path resolved from
`ALHENA_CACHE` when that flag is absent. `alhena info` reports the resolved location.

## Progress

Long-running stages report progress on standard error, leaving command results on standard
output. Depending on the operation, progress can include tile reading, image encoding,
nucleus-to-token matching, reference loading, and retrieval.

`--quiet` disables ALHENA and download progress for one command.
`ALHENA_DISABLE_PROGRESS_BARS=1` disables ALHENA progress reporting process-wide.

```text
Loading the UNI2-h weights from /data/UNI2-h/pytorch_model.bin
Encoding tiles:  45%|████▌     | 450/1000 [00:31<00:38, 14.2tile/s]
```

??? note "Advanced: progress in log files and notebooks"
    A step with no loop to measure, such as reading the UNI2-h weights, prints one line as it
    starts. A stage that finishes within two seconds draws nothing. Downloads show
    huggingface-hub's own bars.

    Because the two streams are separate, `alhena encode ... > run.log` keeps the report and
    leaves the bars on the terminal. An error is printed on standard error too, so the reason a
    run failed stays on the terminal. Where standard error is not a terminal, as with
    `2> run.err` or a pipe, every redraw of a bar stays as a line. There, a bar is first drawn
    after 10 seconds and redrawn at most every 10 seconds. In a notebook, `out = !alhena ...`
    captures the output through a pipe, so the same rule applies. Without the assignment,
    `!alhena ...` runs in a pseudo-terminal, and its bars redraw in place, as in a terminal.

---

## alhena info

Reports the current ALHENA setup without downloading or modifying anything:

```bash
alhena info
```

The report includes:

- the ALHENA version;
- resolved ALHENA cache location;
- availability of the UNI2-h model weights;
- availability of the encoder bundle;
- cached references;
- when the bundle is available, its encoder identity and `cell_space_id`.

The command succeeds even when resources are missing, making it useful for diagnosing an
installation before running an analysis.

| Option | Takes | Default | What it does |
|---|---|---|---|
| `--backbone` | a file or a directory | `ALHENA_BACKBONE`, then the Hugging Face cache | the weight file to report on |
| `--bundle` | a directory, or a repository of the form `owner/name` | the published bundle in the cache | the bundle to report on |
| `--cache-dir` | a directory | `ALHENA_CACHE`, then `$XDG_CACHE_HOME/alhena`, then the per-user cache | the cache to report on |

---

## alhena download

Downloads the ALHENA encoder bundle and/or selected prebuilt references into the ALHENA cache. Run
it on a login node, or before going offline, ahead of a run that would fetch them itself.

```bash
alhena download --bundle --tissue Breast Skin
```

| Option | Takes | Default | What it does |
|---|---|---|---|
| `--bundle` | nothing; a switch | off | fetch the encoder bundle |
| `--tissue` | one or more tissue names, aliases or reference names | none | fetch the reference of one or more tissues |
| `--revision` | a branch, tag or commit | `v0.1.0`, the release this version of ALHENA reads | which revision of the repositories to fetch |
| `--cache-dir` | a directory | `ALHENA_CACHE`, then `$XDG_CACHE_HOME/alhena`, then the per-user cache | where the artifacts land |
| `-q`, `--quiet` | nothing; a switch | off | draw no download bars |

The bundle is three files: `config.json`, `adapters.safetensors` and `heads.safetensors`. Each
selected tissue resolves to one prebuilt reference file.

Sizes vary substantially between reference files, so the command first prints a
`requested resources` block with each reference's declared size. A requested resource may already
be cached, so this is not necessarily a list of new downloads. What is downloaded, and how each
file is checked on arrival, is described in
[Downloaded ALHENA resources](installation.md#downloaded-alhena-resources).

Tissue names, accepted aliases, and underlying reference names resolve through the same tissue
mapping that `alhena reference list` prints. Multiple names that map to the same reference are
downloaded only once.

---

## alhena encode

Computes image-derived embeddings and writes them to disk.

Without `--nuclei`, the command runs the tile-level workflow corresponding to `encode_tiles()`.
With `--nuclei`, it runs `encode_cells()` and writes a cell-level result.

```bash
alhena encode tiles/ --mpp 0.25 --out prostate_embeddings.h5
```

Seven fields are printed. Every value in the layout below is a placeholder.

```text
encoder id          <sixteen hex characters>
resolution          <mpp> micron per pixel
  <where the resolution came from>
white pad           <share> of the canvas
cropped away        <share> of the field
  Both percentages are the maximum observed across tiles.
embeddings          <tiles read> tiles x 1536 dimensions
tile bounds         in the frame of <frame>
written             <path> (<size on disk>)
```

- `encoder id` fingerprints the bundle. `alhena info` reports the same value once the bundle is
  cached.
- `resolution` gives the microns per pixel the run worked at. The line under it says where that
  resolution came from: `declared with --mpp`, or, for a whole slide, `read from` and the source, such
  as `aperio.MPP`, `OME PhysicalSizeX` or `TIFF resolution tags`. It reads `undeclared`, with no
  line under it, only for tiles with no resolution of their own, read without `--mpp`.
- `white pad` is the share of the 224 pixel canvas filled with white instead of tissue.
  `cropped away` is the share of the field the center crop discarded. Each is the largest any tile
  reached, and both read `unknown` where no resolution was declared.
- `embeddings` gives the number of tiles and the dimensions of each tile embedding. The stored
  matrix has one row per tile and 1536 columns.
- `tile bounds` reports the coordinate frame of the stored tile bounds. This field reads
  `unavailable (tile origins not provided)` when the input tiles have no shared spatial
  coordinates.
  Spatial bounds are preserved only by the `.h5ad` output.
- `written` names the file and its size. The lines under it say what the format carries: where
  the geometry went, that a `.npy` array carries no tile names, or what a `.h5ad` table holds.

The command also emits geometry warnings when input tiles require padding or substantial center
cropping. The report records `white pad` and `cropped away` so the geometry can be inspected after
the run. See [Geometry](concepts/geometry.md).

??? note "What each tile-output format preserves"
    - `.npy` stores the embedding matrix and a separate geometry sidecar.
    - `.h5` / `.hdf5` stores the datasets `embeddings` and `barcode`, the tile names, with the
      geometry as attributes of the file.
    - `.h5ad` stores the complete analysis-ready tile table, including embeddings, spatial bounds
      when available, and provenance.

    See [Persistence and interchange](outputs/persistence.md) for the complete layouts.

| Argument | Takes | Default | What it does |
|---|---|---|---|
| `tiles` | a directory of images, a `.npy` array of shape `(n_tiles, height, width, 3)`, an HDF5 file holding an `img` dataset and optionally `barcode` and `coords`, a whole slide: a `.svs`, `.ndpi`, `.scn` or `.bif` file, or any TIFF, `.tif`, `.tiff` or `.btf`, OME-TIFF included; or one large image cut into a grid: a `.png`, `.jpg`, `.jpeg`, `.bmp` or `.webp` file, or a `.npy` array of shape `(height, width, 3)` | required | the tiles to embed |
| `--out` | a path ending `.npy`, `.h5`, `.hdf5` or `.h5ad` | required | file the embeddings are written to; with `--nuclei` it is the `.h5ad` table of cells |
| `--batch-size` | an integer | `32` | tiles per forward pass |
| `--nuclei` | the nuclei, in the forms `annotate` takes | none | embed the cells of these nuclei rather than the tiles |
| `--nuclei-ids` | `per_tile` or `cohort` | `per_tile` | with `--nuclei`, how label values name cells, as on `annotate` |
| `--radius-px` | a positive float | none | with `--nuclei`, the nucleus radius for centroids with no radius, in the pixels of the tiles |
| `--nuclei-kind`, `--nuclei-frame`, `--nuclei-units`, `--nuclei-transform`, `--nuclei-columns`, `--nuclei-id-property`, `--nuclei-geometry-property`, `--nuclei-multipart` | as on `annotate` | as on `annotate` | with `--nuclei`, [how the nuclei are read](#how-the-nuclei-are-read) |
| `--all-tiles` | nothing; a switch | off | with `--nuclei`, read every tile, including those no cell is read from |
| `--tile-embeddings` | a path ending `.npy`, `.h5`, `.hdf5` or `.h5ad` | none | with `--nuclei`, also write the embeddings of the tiles read |
| `--mpp` | a positive float | the resolution a whole slide declares; undeclared for other tiles | microns per pixel of the tiles, which is what makes the field of view physical; for a whole slide, of its full-resolution level, overriding the resolution the slide declares |
| `--tile-px` | an integer | the side that fills the canvas: at the resolution of the level read for a slide, at `--mpp` for an image | side of the tiles cut from a whole slide or an image, in its pixels; a side given by hand fills the canvas only where it covers 95.2 µm |
| `--stride-px` | an integer | the tile side | step between the tiles cut from a whole slide or an image |
| `--edge` | `drop`, `shift` or `pad` | `pad` for a slide, `shift` for an image | what becomes of the tiles the edge of a slide or an image cuts: [left out, moved inward, or padded white](inputs/images.md#the-edges-of-a-grid) |
| `--coords` | a table, a `.npy` array or an HDF5 file | none | for a slide or an image, the [origins of the tiles to cut](inputs/images.md#choosing-which-tiles) instead of the grid; for a directory or a `.npy` array of tiles, a `.csv`, `.tsv` or `.parquet` table with `tile_id`, `x` and `y` columns giving the [origin of every tile](inputs/coordinates.md#tiles-without-origins), which gives the tiles their bounds and places the nuclei in one frame |
| `--region` | four numbers, `X0 Y0 X1 Y1` | none | for a slide or an image, keep only the [tiles lying wholly inside this rectangle](inputs/images.md#choosing-which-tiles), in level-zero pixels, upper bounds excluded |
| `--coords-columns` | `FIELD=COLUMN` pairs | the usual names | the columns of the `--coords` table holding `x` and `y`, and `tile_id` for the origin of every tile, where they are named otherwise |
| `--coords-scale` | a positive float | what the file records, one otherwise | the [pixels of the frame of the tile origins](inputs/coordinates.md#tiles-stored-at-another-size) one tile pixel spans, for tiles stored at another size than they were cut at: the origins of an HDF5 file, or those `--coords` gives |
| `--h5-datasets` | `DATASET=NAME` pairs | `img`, `barcode` and `coords` | the names an HDF5 file of tiles holds `img`, `barcode` and `coords` under, or a file of origins `--coords` names holds `coords` under |
| `--backbone` | a file or a directory | `ALHENA_BACKBONE`, then the Hugging Face cache | UNI2-h model weights |
| `--bundle` | a directory or `owner/name` | `prism-oncology/ALHENA` | where the adapters and heads are read from |
| `--cache-dir` | a directory | `ALHENA_CACHE`, then `$XDG_CACHE_HOME/alhena`, then the per-user cache | where downloads land |
| `--device` | `cpu`, `cuda` or `mps` | CUDA where available, the processor otherwise | device the forward pass runs on |
| `-q`, `--quiet` | nothing; a switch | off | report no [progress](#progress) |

!!! warning "`--tile-px` refers to input-image pixels"
    `--tile-px=224` does not necessarily correspond to the 224 × 224 model canvas. The physical
    field of view also depends on the image resolution.

    In most cases, leave `--tile-px` unset and allow ALHENA to choose a tile side that
    approximately fills the 95.2 µm model field of view. See
    [How large to cut](concepts/geometry.md#how-large-to-cut).

Tiles carrying no resolution of their own and given none warn once and report
`resolution undeclared`. What that costs is [tile geometry](concepts/geometry.md). For whole-slide
inputs, ALHENA reads the resolution from slide metadata when available.

Declaring a resolution the encoder does not read is not an error either. The run reports what it
did with the tiles, in a warning above the fields.

The `--out` suffix is checked before the encoder is built, so a typo costs a second rather than a
forward pass.

```text
alhena: Tile embeddings are written as .npy or .h5 or .hdf5 or .h5ad, and embeddings.txt names .txt.
```

A slide is read at level zero using the resolution it declares. `--mpp` takes precedence over that
declaration and can therefore correct a wrong value. A slide that declares no resolution, or a
resolution outside the 0.02 to 20 µm/px range of a microscope image, is refused until `--mpp`
declares one. If `--mpp` departs from a plausible declared resolution, a warning appears above the
fields.

`--mpp` itself is held to the same range for every form of `tiles`: values coarser than 20 µm/px
are refused, and values finer than 0.02 µm/px are warned about.
[Images](inputs/images.md#a-whole-slide) sets out which formats are read, where the resolution is
read from, and how a broken `tiffslide` install is reported. `--batch-size` bounds the forward
pass; see [How much memory does a run take?](faq.md#how-much-memory-does-a-run-take).

### Image input formats

`alhena encode` accepts the same image sources as the Python API:

- supported whole-slide formats;
- one large image that ALHENA divides into a grid;
- directories or arrays of pre-extracted tiles;
- HDF5 tile collections.

For whole slides and large images, `--tile-px`, `--stride-px`, `--edge`, `--coords` and
`--region` control how image regions are selected.

For pre-extracted tiles, grid options do not apply because the tiles already exist. ALHENA refuses
them rather than ignoring them.

See [Images](inputs/images.md) for format-specific behavior and
[Coordinates](inputs/coordinates.md) for tile origins and coordinate scaling.

### Cell embeddings with `--nuclei`

With `--nuclei`, `alhena encode` computes `CellEmbeddings` rather than a tile-level result. Nuclei
are first matched to the image geometry, then the required tiles are encoded in batches and their
patch tokens are pooled over each nucleus.

```bash
alhena encode slide.svs --nuclei nuclei.geojson --out cells.h5ad
```

The `.h5ad` result contains the pre-projection and projected cell embeddings together with cell
identifiers, spatial positions when available, and provenance required for later annotation.

The report is one field per line. Every value in the layout below is a placeholder.

```text
encoder id          <sixteen hex characters>
cell space id       <sixteen hex characters>
resolution          <mpp> micron per pixel
  <where the resolution came from>
tiles read          <read> of <tiles in the source>
cells               <cells> total; <usable> usable for annotation
positions           in the frame of <frame>
written             <path> (<size on disk>)
```

- `resolution`, and the line under it, read as they do for tile embeddings above.
- `tiles read` reports how many image tiles were required for the supplied nuclei. By default,
  only tiles contributing to cells are processed; `--all-tiles` processes the complete tile
  source.
- `cells` reports the returned cell rows and how many are usable for annotation. Nuclei that did
  not overlap any image patch token are reported separately.
- `positions` reports the coordinate frame of the stored cell positions when available.

The table is written in the layout [the table layouts](outputs/persistence.md#cells) set out.
Where the resolution is known, the table also records the microns one pixel of the frame spans,
under `uns["alhena_cells"]["centroids_mpp"]`. [Tile origins](inputs/coordinates.md#tile-origins)
sets out the frame.

```bash
alhena encode tiles/ --nuclei masks/ --coords origins.csv --mpp 0.5 --out cells.h5ad
```

`--coords` reads one row per tile, with the tile name under `tile_id` and its top-left pixel under
`x` and `y`. Rows are matched by tile name: the file stem for a directory or the tile's position for
an array. ALHENA refuses a row that names no tile, a tile that appears more than once, a tile with
no matching row, or an origin that falls between pixels.
[Tiles without origins](inputs/coordinates.md#tiles-without-origins) sets out the rule.

`--tile-embeddings` optionally saves the tile embeddings generated during cell encoding. These
have `purpose="cell_context"` because they were produced to provide context for the cells rather
than as an independent tile-level sampling of the slide. A `.npy` path is refused unless
`--all-tiles` is given, because an array carries no tile names. See
[Dedicated tiles and cell-context tiles](outputs/tile-embeddings.md#dedicated-tiles-and-cell-context-tiles).

A crop discarding more than 5 percent of each field reaches the cells, since a nucleus outside the
central window the canvas holds is never read. The run warns about it above the report, and the
warning counts the nuclei that reached no token.

`--nuclei` and the options that read it work exactly as on `annotate`:
[how the nuclei are read](#how-the-nuclei-are-read). Some options are refused where they would
describe nothing:

- without `--nuclei`: `--all-tiles`, `--tile-embeddings` and every option reading the nuclei, since
  there are no cells to describe.
- with an HDF5 file: `--coords`, since the file is the only record of its tiles, and its origins
  come from its `coords` dataset.
- `--coords-scale`, for tiles with no origins, and for a whole slide or an image, which set their
  own scale: the downsample of the level read.

---

## alhena annotate

Assigns reference-based cell types and writes one row per encoded cell.

The command accepts either:

- an H&E image together with nuclei, in which case it first computes `CellEmbeddings`;
- a previously saved `.h5ad` cell table, in which case no image encoding is repeated.

```bash
alhena annotate slide.svs \
    --nuclei nuclei.geojson \
    --tissue Breast \
    --out annotations.h5ad
```

Previously encoded cells can be annotated directly:

```bash
alhena annotate cells.h5ad \
    --tissue Breast \
    --axis compartment_v1 \
    --out annotations.csv
```

A custom reference is supplied through `--reference` together with the axis on which it was
built.

The report is one field per line, with one indented line per class of the axis. Every value in the
layout below is a placeholder.

```text
resolution          <mpp> micron per pixel
  <where the resolution came from>
reference           <reference>
axis                <axis>
retrieval           k=<k>, tau=<tau>; median effective k <effective k>
  k: reference neighbors each cell votes over; tau: vote softmax temperature.
  effective k: how many of those neighbors effectively contribute to the vote.
cells annotated     <annotated> of <in the table>; <without prediction> without a valid prediction
                               cells   % annotated   % reference
  <class>                    <count>       <share>       <share>
  <class>                    <count>       <share>       <share>
  Percentages use valid annotations and labeled reference cells as their respective totals.
written             <path> (<size on disk>)
```

When annotating tiles, the report opens with the resolution they were read at, and where it came
from, as [`encode`](#alhena-encode) reports them. A table of cells records its own resolution, so
its report opens at `reference`.

`retrieval` gives the operating point, then the median effective number of neighbors over the
annotated cells, which is left out when no cell received an annotation.
[Effective number of neighbors](concepts/cell-typing.md#effective-number-of-neighbors) says what
it measures.

`cells annotated` reports how many rows received a valid prediction and how many did not. The
table gives each class's cell count, its percentage among valid annotations (`% annotated`), and
its percentage among labeled cells of the retrieved reference (`% reference`). Nuclei excluded
before the cell table is formed because they overlap no patch token are reported separately.

This comparison can help interpret differences in reference class support but should not be read
as a direct correction for reference composition. See
[How reference composition affects annotation](concepts/references.md#how-reference-composition-affects-annotation).

Warnings the reference raises print above the report and change how it reads. A class the
reference cannot express can never be returned as an annotation, and a thinly supported class
competes for the vote on fewer neighbors than the others. Both are properties of the reference,
visible before a run through [`alhena reference describe`](#alhena-reference-describe), and read
in [References](concepts/references.md).

| Argument | Takes | Default | What it does |
|---|---|---|---|
| `tiles` | the forms `encode` takes, or a `.h5ad` table of cells | required | the tiles the nuclei were segmented in, or the cells to annotate |
| `--nuclei` | label maps in a directory, an HDF5 file or a `.npy` array of shape `(n_tiles, height, width)`; the label image of the slide or the image the tiles are cut from, a `.png`, `.tif` or `.tiff` file or a `.npy` array of shape `(height, width)`; centroids in a table; or boundaries in GeoJSON, GeoParquet or a table of vertices | required with tiles | the nuclei to annotate |
| `--tissue` | one of the fifteen tissue names, an alias of one, or a reference name | — | tissue whose reference is retrieved against |
| `--reference` | the path of a reference file on disk | — | a reference file, instead of a tissue |
| `--out` | a path ending `.parquet`, `.csv` or `.h5ad` | required | file the annotations are written to |
| `--axis` | `compartment_v1` or `lineage_v1`, or the axis a custom reference was built on | `lineage_v1` | annotation axis |
| `--k` | an integer | `30` | reference neighbors each cell votes over |
| `--tau` | a positive float | `0.03` | softmax temperature over the neighbor similarities |
| `--nuclei-ids` | `per_tile` or `cohort` | `per_tile` | `per_tile` reads a label value as naming a cell within its own tile and renumbers it to be unique across the tiles; `cohort` takes the values as they are, already naming one cell each. A label image is read as it is |
| `--radius-px` | a positive float | none | nucleus radius standing in for the segmented shape, for centroids with no radius, in the pixels of the tiles whatever the units of the table |
| `--nuclei-kind` | `label_maps`, `centroids` or `polygons` | from the suffix | the form `--nuclei` holds; a table other than GeoParquet is read as centroids unless named `polygons`, as a table of boundary vertices is |
| `--nuclei-frame` | `tile` or `origins` | `tile` | the frame a table of centroids is written in: the pixels of the tile each row names, or one row per nucleus in the frame of the tile origins |
| `--nuclei-units` | `px` or `um` | `px` | the units of the coordinates of centroids or boundaries; microns are divided by the microns one pixel of their frame spans |
| `--nuclei-transform` | a `.csv` or `.npy` file holding a 2 by 3 or 3 by 3 affine matrix | none | carries the coordinates of centroids or boundaries, as they are written, into the pixels of the frame of the tile origins; not given with `--nuclei-units um` |
| `--nuclei-columns` | `FIELD=COLUMN` pairs | the usual names | the columns of a table of centroids or of vertices carrying `x`, `y` and `cell_id`, and `tile_index` and `radius_px` for centroids |
| `--nuclei-id-property` | a property name | `cell_id`, then the feature `id`, then its position | the property of a GeoJSON feature naming its cell |
| `--nuclei-geometry-property` | a member name | the geometry | the member of a GeoJSON feature holding the nucleus outline, such as `nucleusGeometry` |
| `--nuclei-multipart` | `refuse`, `split` or `largest` | `refuse` | how a boundary holding several polygons is read |
| `--truth` | a `.parquet` or `.csv` with `cell_id` and `label` columns | none | known labels, which turn on the scoring report |
| `--workers` | an integer | the processors available, at most `8` | threads retrieving reference neighbors at once; the annotations are the same for any number |
| `--batch-size` | an integer | `32` | tiles per forward pass, which bounds the memory a run holds |
| `--mpp` | a positive float | the resolution a whole slide declares; undeclared for other tiles | microns per pixel of the tiles; for a whole slide, of its full-resolution level, overriding the resolution the slide declares |
| `--tile-px` | an integer | the side that fills the canvas: at the resolution of the level read for a slide, at `--mpp` for an image | side of the tiles cut from a whole slide or an image; a side given by hand fills the canvas only where it covers 95.2 µm |
| `--stride-px` | an integer | the tile side | step between the tiles cut from a whole slide or an image |
| `--edge` | `drop`, `shift` or `pad` | `pad` for a slide, `shift` for an image | what becomes of the tiles the edge of a slide or an image cuts, as on `encode` |
| `--coords` | a table, a `.npy` array or an HDF5 file | none | the [origins of the tiles to cut](inputs/images.md#choosing-which-tiles) from a slide or an image, or the [origin of every tile](inputs/coordinates.md#tiles-without-origins) of a directory or a `.npy` array of tiles, as on `encode` |
| `--region` | four numbers, `X0 Y0 X1 Y1` | none | the tiles of a slide or an image to keep, those lying wholly inside the rectangle, as on `encode` |
| `--coords-columns` | `FIELD=COLUMN` pairs | the usual names | the columns of the `--coords` table, as on `encode` |
| `--coords-scale` | a positive float | what the file records, one otherwise | the [pixels of the frame of the tile origins](inputs/coordinates.md#tiles-stored-at-another-size) one tile pixel spans, as on `encode` |
| `--h5-datasets` | `DATASET=NAME` pairs | `img`, `barcode` and `coords` | the names the datasets of an HDF5 file are held under, as on `encode` |
| `--backbone` | a file or a directory | `ALHENA_BACKBONE`, then the Hugging Face cache | UNI2-h model weights |
| `--bundle` | a directory or `owner/name` | `prism-oncology/ALHENA` | where the adapters and heads are read from |
| `--cache-dir` | a directory | `ALHENA_CACHE`, then `$XDG_CACHE_HOME/alhena`, then the per-user cache | where downloads land, and where a reference file named by `--tissue` is looked for first |
| `--device` | `cpu`, `cuda` or `mps` | CUDA where available, the processor otherwise | device the forward pass runs on |
| `-q`, `--quiet` | nothing; a switch | off | report no [progress](#progress) |

!!! warning "Check tile geometry when annotating directly from images"
    A manually supplied `--tile-px` may produce padding or cropping if its physical field of view
    differs substantially from the 95.2 µm model field. `annotate` does not warn about padding,
    because pad tokens carry no direct pooling weight on the cell path.

    Prefer the automatically selected tile size unless a specific geometry is required. See
    [Geometry](concepts/geometry.md).

Exactly one of `--tissue` and `--reference` is required, and that rule is declared to the parser.
The usage line shows `(--tissue TISSUE | --reference REFERENCE)`, and passing both is a usage
error. `--tissue` is resolved before anything is read, as on `download`, so a misspelled tissue
costs no read of the tiles, the nuclei or a table of cells.

`--mpp` reaches the mask path as well as the tile path, because one array has one resolution;
[tile geometry](concepts/geometry.md) says what follows from that. `--batch-size` bounds what a
run holds, one batch of tiles at a time; see
[How much memory does a run take?](faq.md#how-much-memory-does-a-run-take). `--workers` sets
how many threads retrieve reference neighbors at once. Each thread holds about 0.4 GB while it
works, and the annotations do not depend on the number.

When `annotate` receives a table of cells, options that describe tiles, nuclei or the encoder are
refused: `--nuclei` and all nucleus-reader options, `--mpp`, `--tile-px`, `--stride-px`, `--edge`,
`--coords`, `--coords-columns`, `--coords-scale`, `--region`, `--h5-datasets`, `--batch-size`,
`--backbone`, `--bundle` and `--device`. The stored cells already record their nucleus provenance, resolution and
positions, and they can be annotated only against a reference in their recorded cell space.

### How the nuclei are read

The file format and `--nuclei-kind` determine how nucleus inputs are interpreted. The remaining
nucleus options map to the corresponding Python reader arguments.

Use `--nuclei-frame`, `--nuclei-units`, and `--nuclei-transform` when nuclei are not already
expressed in the same image-pixel coordinate system as the H&E input. See
[Coordinates](inputs/coordinates.md).

| Option | Argument of the reader | Read for |
|---|---|---|
| `--nuclei-ids` | `renumber`, off with `cohort` | label maps, one per tile |
| `--radius-px` | `radius_px` | centroids |
| `--nuclei-frame` | `frame` | centroids |
| `--nuclei-units` | `units` | centroids and boundaries |
| `--nuclei-transform` | `transform` | centroids and boundaries |
| `--nuclei-columns` | `columns` | tables of centroids and of vertices |
| `--nuclei-id-property` | `id_property` | GeoJSON |
| `--nuclei-geometry-property` | `geometry_property` | GeoJSON |
| `--nuclei-multipart` | `multipart` | boundaries |

A table of centroids written one row per nucleus over a whole slide, in microns, and a table of
boundary vertices whose columns carry other names, read over the same slide:

```bash
alhena annotate slide.svs --nuclei cells.parquet --nuclei-frame origins --nuclei-units um --radius-px 8 --tile-px 381 --tissue Breast --out annotations.h5ad
alhena annotate slide.svs --nuclei boundaries.csv --nuclei-kind polygons --nuclei-columns cell_id=nucleus_id x=x_px y=y_px --tile-px 381 --tissue Breast --out annotations.h5ad
```

`--nuclei-columns` takes every pair up to the next option, so the tiles are named before it.
[Constructing the transform from a registration matrix](inputs/coordinates.md#constructing-the-transform-from-a-registration-matrix)
builds the matrix `--nuclei-transform` reads from an alignment between two images.

A single integer label image can be used when it covers the same large image or whole slide as the
H&E input. ALHENA applies the same tile grid while preserving global nucleus identifiers. See
[A label image](inputs/nuclei.md#a-label-image).

```bash
alhena annotate region.png --nuclei labels.tif --mpp 0.25 --tissue Breast --out annotations.h5ad
```

### Annotation output

`.parquet` and `.csv` outputs contain the per-cell annotation table, including the predicted
label, confidence, validity flag, entropy diagnostics, and class masses. The table also holds the
position of each cell where the cells record one, and the index of its tile. Tiles annotated
directly also give the name of the tile. See
[Cell annotations](outputs/cell-annotations.md#cell-type-predictions) for every column.

`.h5ad` additionally preserves the underlying `CellEmbeddings` and provenance. Its annotation
columns are named after the axis, as `alhena_celltype_lineage_v1`. These cells can later be
annotated with another compatible reference without re-encoding the H&E image.

See [Persistence and interchange](outputs/persistence.md#cell-embeddings-and-annotations).

With `--truth`, ALHENA evaluates the annotations against an external table of known labels. The
report includes accuracy, balanced accuracy, macro F1, per-class metrics, baselines, and
`results by minimum confidence`. Cells without a valid prediction count as errors in the overall
scores. Each confidence-threshold row instead scores only cells retained at that threshold;
`% kept` is the percentage of all evaluated cells, including cells without a valid prediction.
Confidence reflects weighted agreement among retrieved reference cells, not a calibrated
probability of correctness. Metrics from different threshold rows are computed on different cell
sets and should not be compared as if the evaluated population were unchanged.

Ground-truth rows are matched to the annotations by `cell_id`; identifier types must match
exactly. See
[Advanced: evaluation with known labels](concepts/cell-typing.md#advanced-evaluation-with-known-labels).

!!! warning "Ground-truth identifiers must match exactly"
    `cell_id` values and types must match between the annotations and truth table. ALHENA does not
    automatically convert integer identifiers to strings or vice versa. For label maps,
    `--nuclei-ids cohort` preserves the original label values, while `per_tile` renumbers them.

??? note "Failed scoring still writes annotations"
    The known labels are read and checked for their columns before the encoder is built, so a
    missing column or a repeated `cell_id` costs a read. Whether the table *covers* the cells that
    were annotated can only be known once they are annotated. That failure therefore arrives after
    the annotations are made, with the annotations already on disk. The command names the file,
    says that only the scoring was skipped, and exits 1. The annotations are complete; score them
    later against a table that covers them.

---

## alhena reference

Inspects the available single-cell references. The three subcommands list the prebuilt references,
describe class support, or run a self-retrieval consistency check. They do not require the H&E image
encoder.

### alhena reference list

Lists the prebuilt tissue references, accepted tissue aliases, underlying reference names,
supported axes, and references currently available in the local cache.

```bash
alhena reference list
```

```text
tissue              shard
  BoneMarrow        bone_marrow
  Breast            breast
  Cervix            cervix
  Colon             intestine
  Heart             heart
  Kidney            kidney
  Liver             liver_cell
  Lung              lung
  LymphNode         tonsil
  Ovary             ovary
  Pancreas          pancreas_cell
  Prostate          prostate
  Skin              skin
  Tonsil            tonsil
  Uterus            uterus
tissue              also accepted as
  Breast            Mammary
  Cervix            Cervix uteri, Uterine cervix
  Colon             Bowel, Colonic, Colorectal, Large intestine, Rectum, Small intestine
  Kidney            Renal
  Liver             Hepatic
  Lung              Pulmonary
  Ovary             Ovarian
  Pancreas          Pancreatic
  Prostate          Prostate gland, Prostatic
  Skin              Cutaneous
  Tonsil            Tonsillar
  Uterus            Corpus uteri, Endometrium, Myometrium, Uterine
axes                compartment_v1, lineage_v1
cached              none
  Fetch one with `alhena download --tissue Breast`.
```

That block needs nothing downloaded. The tables are compiled into the package, so only the
`cached` line depends on the machine. The command writes nothing and reaches no network. See
[Tissue aliases](concepts/references.md#tissue-aliases).

| Option | Takes | Default | What it does |
|---|---|---|---|
| `--cache-dir` | a directory | `ALHENA_CACHE`, then `$XDG_CACHE_HOME/alhena`, then the per-user cache | the cache whose reference files are listed |

### alhena reference describe

Reports metadata and class support for one reference on the selected annotation axis.

```bash
alhena reference describe Prostate
```

The report is one field per line, with the class support indented beneath its own field. Every
value in the layout below is a placeholder.

```text
shard               <shard>
axis                <axis>
cell space id       <sixteen hex characters>
cells               <labeled> labeled of <held> total
source              <source>
license             <license>
single cell model   <model> <version>
support metadata    <verified or not verified>
class support
                               cells  % of labeled reference cells
  <class>                  <support>                       <share>
  <class>                  <support>                       <share>
cannot express      <classes, or none>
thinly supported    <classes>
```

`cell space id` is the identity that must match the encoder bundle, as
[Reference compatibility](concepts/references.md#reference-compatibility) explains. `cells`
reports the number of cells carrying a label on this axis and, after `of`, the total number of
cells held by the reference file. `source` and `license` come either from the reference file
itself or from the registry associated with a prebuilt reference. They read `undeclared` when
neither source provides a value. `class support` gives one line per class with its cell count and
its share of the labeled cells. Those labeled cells form the retrieval pool for this axis.
`support metadata` indicates whether the per-class counts recorded in the reference metadata were
verified against the actual labels (`verified`) or were not verified (`not verified`). It does not
measure annotation accuracy. `cannot express` lists classes with no labeled cells in the reference
file, or reads `none` when all classes on the axis are represented. `thinly supported` appears only
when one or more classes have insufficient support. An indented explanation then states the
consequence for a run.

The command inspects the reference metadata and class labels without loading the full embedding
matrix.

| Argument | Takes | Default | What it does |
|---|---|---|---|
| `name` | a tissue name, an alias or a reference name | — | the reference to describe |
| `--path` | the path of a reference file on disk | — | a reference file, instead of a name |
| `--axis` | `compartment_v1` or `lineage_v1`, or, with `--path`, the axis a custom reference was built on | `lineage_v1` | class axis the reference is read on |
| `--cache-dir` | a directory | `ALHENA_CACHE`, then `$XDG_CACHE_HOME/alhena`, then the per-user cache | where a named reference is looked for |
| `-q`, `--quiet` | nothing; a switch | off | draw no download bar |

Passing both `name` and `--path` leaves it ambiguous which reference is meant, and the command
says so. Reading the same reference on the other axis reports a different set of classes; which
axis to read on is [cell typing](concepts/cell-typing.md).

### alhena reference check

Performs leave-one-out self-retrieval within the selected reference and reports how well the
reference recovers its own atlas labels.

```bash
alhena reference check Prostate
```

The report names the reference, the axis, the operator it ran, and how many cells were scored. It
then gives the agreement with the reference's own labels, the recall of every class, and quantiles
of the similarities it retrieved. Notes under it say what the numbers depend on: a subsample, the
classes the reference holds no or few cells of, and a thinnest class too thin for the `--k` of the
check.

The reported similarity quantiles describe cosine similarities between reference cells and their
retrieved reference neighbors.

This is an internal consistency check for the reference. It does not measure performance on
H&E-derived cells. See
[Reference self-check](concepts/references.md#reference-self-check).

| Argument | Takes | Default | What it does |
|---|---|---|---|
| `name` | a tissue name, an alias or a reference name | — | the reference to check |
| `--path` | the path of a reference file on disk | — | a reference file, instead of a name |
| `--axis` | `compartment_v1` or `lineage_v1`, or, with `--path`, the axis a custom reference was built on | `lineage_v1` | class axis the reference is read on |
| `--k` | an integer | `30` | neighbors each cell votes over in the check |
| `--tau` | a positive float | `0.03` | softmax temperature in the check |
| `--max-cells` | an integer | `200000` | cells the check scores; a smaller reference is scored entire |
| `--seed` | an integer | `0` | seed of the subsample the check scores |
| `--cache-dir` | a directory | `ALHENA_CACHE`, then `$XDG_CACHE_HOME/alhena`, then the per-user cache | where a named reference is looked for |
| `-q`, `--quiet` | nothing; a switch | off | report no [progress](#progress) |

The whole report is printed and nothing is written. The same one-of-`name`-or-`--path` rule
applies, with the same message.

---

## alhena demo

Runs the small PanNuke example used to verify an ALHENA installation.

```bash
alhena demo Breast
```

ALHENA performs zero-shot cell-type annotation of H&E nuclei using
similarity-weighted k-nearest neighbors (kNN) against a labeled
single-cell RNA-seq reference.

The demo uses PanNuke images and compares the predicted cell classes
with PanNuke labels.

**Important:** Only the first fields of the selected tissue are used,
in file order. This is a workflow demonstration, not a representative
performance benchmark.

See [Understanding the results](start/demo.ipynb#understanding-the-results)
for an explanation of the evaluation, baselines, confidence and limitations.

The default axis is `compartment_v1`, whose three broad cell classes
can be compared with the mapped PanNuke labels.

| Argument | Takes | Default | What it does |
|---|---|---|---|
| `tissue` | one of `Breast`, `Cervix`, `Colon`, `Kidney`, `Liver`, `Lung`, `Ovarian`, `Pancreatic`, `Prostate`, `Skin`, `Uterus` | `Breast` | PanNuke tissue to read |
| `--list-tissues` | nothing; a switch | off | report which PanNuke tissues carry a reference, and stop |
| `--n-tiles` | an integer | `8` | how many fields of that tissue to read, the first ones in file order |
| `--fold` | `1`, `2` or `3` | `1` | PanNuke fold to read; it selects nothing when `--parquet` is given |
| `--parquet` | the path of a fold file on disk | none | a fold already on disk, read instead of the published one |
| `--revision` | a branch, tag or commit | a fixed commit, `alhena.demo.PANNUKE_REVISION` | which revision of the dataset repository to read |
| `--axis` | `compartment_v1` or `lineage_v1` | `compartment_v1` | annotation axis |
| `--k` | an integer | `30` | reference neighbors each nucleus votes over |
| `--tau` | a positive float | `0.03` | softmax temperature over the neighbor similarities |
| `--backbone` | a file or a directory | `ALHENA_BACKBONE`, then the Hugging Face cache | UNI2-h model weights |
| `--bundle` | a directory or `owner/name` | `prism-oncology/ALHENA` | where the adapters and heads are read from, as on `annotate` |
| `--reference` | the path of a reference file on disk | none | the prebuilt reference file of the tissue, read from disk instead of the cache; it must be the file published for that tissue |
| `--cache-dir` | a directory | `ALHENA_CACHE`, then `$XDG_CACHE_HOME/alhena`, then the per-user cache | where downloads land, and where the reference file of the tissue is looked for first |
| `--device` | `cpu`, `cuda` or `mps` | CUDA where available, the processor otherwise | device the forward pass runs on |
| `-q`, `--quiet` | nothing; a switch | off | report no [progress](#progress) |

The example writes no file: its report is printed. The bundle and the one reference it needs land
in the cache, as they would for any run, unless `--bundle` and `--reference` name them on disk.

`--parquet` can be used with a local PanNuke fold. `--bundle`, `--reference`, and `--backbone` can
likewise point to local resources when needed.

PanNuke tissue names are not always identical to ALHENA reference names; the demo resolves the
supported mappings internally. Use `--list-tissues` to inspect them.

??? warning "The demo axis default differs"
    `alhena demo` defaults to `compartment_v1`, where every other command defaults to
    `lineage_v1`. PanNuke's labels map to the three broad cell classes of `compartment_v1`, and no
    comparison against them is available on another axis, so the example defaults to the axis its
    own comparison needs.

??? note "Options after demo are forwarded"
    Everything written after `demo` is handed to the example unread. That is what makes
    `alhena demo --help` the example's own help rather than the wrapper's, and it lets an option
    follow the command with no separator between them. A bare `--` written first is accepted and
    means nothing.

---

**Next:** [The Python API](api/index.md)
