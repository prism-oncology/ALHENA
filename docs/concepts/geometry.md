# Geometry

ALHENA analyzes H&E at a fixed physical scale. Input tiles may arrive at different pixel
resolutions and sizes, but before encoding they are resampled and fitted to the same 224 × 224
pixel model canvas.

The released model uses:

- **0.425 µm/px** as the resolution it reads;
- a **224 × 224 pixel** canvas;
- a **95.2 µm** field of view;
- a **16 × 16** token grid;
- approximately **5.95 µm per token**.

`mpp` tells ALHENA how many microns each input pixel represents. This value determines how much
physical tissue reaches the model.

Use `describe_geometry()` to inspect the transformation before running a large analysis:

```python
import alhena

report = alhena.describe_geometry(mpp=0.25, tile_size_px=256)

print(report.field_of_view_um)
print(report.pad_fraction)
print(report.crop_fraction)
```

See [Check before you run](#check-before-you-run) for a complete example.

## Geometry at both scales

Image geometry affects both ALHENA scales:

- **Tile embeddings** describe the physical region that reaches the model canvas. Padding or
  cropping changes the image context represented by the embedding.
- **Cell embeddings** additionally depend on the alignment between nucleus masks and image tokens.
  The image and nucleus representation must therefore undergo compatible geometric
  transformations.

The same image resolution should describe both the H&E pixels and the corresponding nucleus
coordinates or masks.

## Input resolution (mpp)

`mpp` is the physical size of the image pixels supplied to ALHENA, in microns per pixel. It should
describe the actual arrays being encoded.

For example, if tiles are extracted from a downsampled pyramid level or resized after extraction,
`mpp` must describe the resulting tile pixels rather than the resolution of the original
full-resolution slide. `read_slide` is the exception: its `mpp=` declares level zero, and the tiles
it reads carry the resolution of their level.

Whole-slide readers obtain this value from the slide metadata when possible. For image arrays or
pre-extracted tiles, provide it explicitly.

!!! warning "Verify the physical image resolution"
    ALHENA can refuse clearly implausible microscopy resolutions, but it cannot determine from RGB
    values whether a plausible `mpp` is actually correct. An incorrect resolution changes the
    physical field of view seen by the model without necessarily raising an error.

You declare `mpp` once per call, and it governs the image and the nuclei together. The value given
to `encode_cells` or `annotate` reaches the tile path and the mask path alike, so the two describe
one field of view by construction. A tile source carries its own resolution, and a call over it
needs none.

The membership builders take an `mpp` of their own, as do `alhena.spatial.cells_from_shapes` and
`alhena.cells.fit_mask`. Give them the value you gave the images. A table created by a builder
records both that resolution and the pixel grid on which its masks were drawn. A table that brings
its masks onto the canvas differently from the images is refused. See
[Advanced: membership geometry checks](cell-pooling.md#advanced-membership-geometry-checks). On the
command line, `--mpp` governs the nuclei as well as the tiles.

The values ALHENA refuses or warns about are listed in
[Advanced: resolution validation](#advanced-resolution-validation).

## The canvas and the token grid

At the 0.425 µm/px resolution the model reads, the 224-pixel canvas spans **95.2 µm**. UNI2-h
divides this canvas into 14 × 14 pixel patches, producing a **16 × 16 grid of 256 image tokens**.

Each token therefore corresponds to approximately **5.95 × 5.95 µm** of the input image. Cell
pooling operates on this token grid, so the token side sets the granularity of a cell embedding.
See [What information enters a cell embedding](cell-pooling.md#what-information-enters-a-cell-embedding).

These values are properties of the released model rather than user-configurable analysis
parameters. [Advanced: model operating resolutions](#advanced-model-operating-resolutions) lists
the constants that hold them.

## What happens to a tile

Given an input resolution, ALHENA maps each tile to the model canvas in three steps:

1. **Determine the scale** required to reach 0.425 µm/px.
2. **Resample the tile** while preserving its aspect ratio.
3. **Fit the result to the 224 × 224 canvas** by center-cropping a larger field or white-padding a
   smaller one.

A scale close to one is left alone. When `abs(scale - 1)` is at most `MPP_TOLERANCE`, 0.02, no
resampling happens at all. A declared resolution from about 0.417 to about 0.433 µm/px passes
through untouched.

Resampling is bilinear. A tile that shrinks more than twofold, such as one scanned finer than
0.2125 µm/px, is averaged over pixel area instead. Bilinear interpolation would skip pixels there
and distort fine texture.

`prepare_tiles()` returns the transformed batch together with a `ResampleInfo` object describing
the operation.

```python
import numpy as np
import alhena

tiles = np.full((4, 381, 381, 3), 200, dtype=np.uint8)
batch, info = alhena.prepare_tiles(tiles, mpp=0.25)

print(batch.shape)            # torch.Size([4, 3, 224, 224])
print(info.scale)             # 0.588...
print(info.source_size, "to", info.resampled_size)
print("padded", info.padded, "cropped", info.cropped)
print(info.pad_fraction)      # 0.0 here; the share of the canvas that is white pad
print(info.crop_fraction)     # 0.0 here; the share of the field the crop discards
```

- `pad_fraction` is the fraction of the model canvas filled by padding rather than source-image
  pixels.
- `crop_fraction` is the fraction of the resampled input field that lies outside the model canvas
  and is discarded.

For a batch containing different tile sizes, the result records whether any tile was padded or
cropped and summarizes the largest observed fractions. The API reference of
[`ResampleInfo`](../api/types.md) says which tile the recorded sizes describe.

## A small field pads with white

If a tile covers less than the model's 95.2 µm field of view, resampling leaves part of the
224 × 224 canvas empty and ALHENA fills that area with white pixels.

!!! warning "Padding changes the image context"
    A 256-pixel tile at 0.25 µm/px covers 64 µm. After resampling, it occupies only part of the
    model canvas and the remaining area is white padding (`pad_fraction ≈ 0.546`).

    Padding affects tile and cell outputs differently. Tile embeddings summarize the entire
    canvas, so the white region contributes to the tile representation. Cell pooling assigns no
    direct pooling weight to padded regions because nucleus masks use background label `0`,
    although padding may still influence the contextualized image tokens through the encoder.

Tile-level encoding warns when padding is present. Cell-level workflows retain the same geometry
information in their result metadata. Check `pad_fraction` before comparing representations
produced with different tile geometries. [Advanced: geometry warnings](#advanced-geometry-warnings)
lists which calls warn and where the share is recorded.

!!! warning "Use consistent geometry when comparing datasets"
    Differences in padding can introduce systematic differences in tile embeddings that are
    unrelated to tissue morphology. When comparing slides or cohorts, use consistent image
    resolution and tile geometry, and inspect the recorded `pad_fraction`. Tile embeddings
    computed at different geometries are not comparable and must not be pooled.

??? example "The arithmetic on your numbers"
    `describe_geometry` reads it off your shapes, before any forward pass:

    ```python
    from alhena import INPUT_SIZE, REFERENCE_MPP, describe_geometry

    tile_px, mpp = 256, 0.25
    report = describe_geometry(mpp=mpp, tile_size_px=tile_px)

    print(f"field           {report.field_of_view_um:.1f} um")
    print(f"canvas window   {INPUT_SIZE * REFERENCE_MPP:.1f} um")
    print(f"resampled to    {round(tile_px * mpp / REFERENCE_MPP)} of {INPUT_SIZE}")
    print(f"white pad       {report.pad_fraction:.1%} of the canvas")
    print(f"tissue          {1 - report.pad_fraction:.1%} of the canvas")
    ```

    Run it once for every new source of tiles, before a large run. `alhena.preprocess.plan_resample`
    answers the same question as a whole `ResampleInfo`, and handles tiles that are not square.

To avoid padding, choose a tile size that covers approximately the model's 95.2 µm field of view
at the input resolution. See [How large to cut](#how-large-to-cut).

## A large field is cropped

If an input tile covers more than 95.2 µm, it resamples to an image larger than the model canvas.
ALHENA then keeps the central 224 × 224 pixels and discards the surrounding region.

!!! warning "Cropping removes part of the input field"
    A 1000-pixel image at 0.25 µm/px covers 250 µm. Only the central 95.2 µm reaches the model,
    corresponding to `crop_fraction ≈ 0.855`.

    A tile embedding therefore represents only this central region. For cell-level encoding,
    nuclei outside the retained region cannot contribute image tokens and may consequently have
    no output row.

ALHENA raises a `GeometryWarning` when center cropping removes more than 5% of the input field.
Small crops introduced by integer rounding can occur even with the automatically selected tile
size, which is why the threshold is not zero.

The exact cropped fraction remains available as `crop_fraction` in the geometry metadata even
below the warning threshold. [Advanced: geometry warnings](#advanced-geometry-warnings) lists
where it is recorded.

The remedy is to pass the field as several tiles of the size given in
[how large to cut](#how-large-to-cut), rather than as one large tile. `alhena.io.read_image` does
this for a region, crop, PNG file or array. It creates a grid whose tile side fills the canvas at
the declared `mpp`, records each tile's origin in image pixels, and shifts the final row and column
inward so that every image pixel is read with full context. `read_label_image` cuts the
corresponding label image along exactly the same tiles while preserving its identifiers, so a
nucleus present in two overlapping tiles is still treated as one cell.
See [One large image](../inputs/images.md#one-large-image).

```python
from alhena.io import read_image, read_label_image

tiles = read_image(region, mpp=0.25)              # tiles of 381 pixels, every pixel read
nuclei = read_label_image(labels, tiles=tiles)    # the label image of the region, cut along them
```

`read_slide` cuts a slide into the same grid, but by default it pads the last row and column
with white where `read_image` shifts them inward. `edge=` sets the rule on either reader.
`alhena encode` and `alhena annotate` cut an image file given as their tiles. With a stride below
the tile side, `stride_px` on either reader, neighboring tiles overlap, and a nucleus at the edge
of one tile is read from a tile holding more of it. For tiles you cut yourself, give their
origins, so that a nucleus found in two tiles is resolved to one of them.
See [Tile origins](../inputs/coordinates.md#tile-origins).

??? example "The arithmetic on your numbers"
    `describe_geometry` reads it off your shapes, before any forward pass:

    ```python
    from alhena import INPUT_SIZE, REFERENCE_MPP, describe_geometry

    tile_px, mpp = 1000, 0.25
    report = describe_geometry(mpp=mpp, tile_size_px=tile_px)

    print(f"field           {report.field_of_view_um:.1f} um")
    print(f"canvas window   {INPUT_SIZE * REFERENCE_MPP:.1f} um")
    print(f"resampled to    {round(tile_px * mpp / REFERENCE_MPP)}, of which {INPUT_SIZE} kept")
    print(f"cropped away    {report.crop_fraction:.1%} of the field")
    for note in report.warnings:
        print(note)
    ```

    `alhena.preprocess.plan_resample` answers the same question for tiles that are not square.

## How large to cut

To fill the 95.2 µm field of view without padding, cut tiles of the smallest whole number of
pixels that covers it:

`tile_px = ceil(95.2 / mpp)`

The ALHENA image readers perform this calculation automatically when `tile_px` is not specified.

At 0.25 µm/px this gives 381 pixels, and at 0.5 µm/px 191 pixels. The latter requires slight
cropping after resampling because of integer rounding.

Within 2% of 0.425 µm/px, tiles are not resampled, so the readers use the canvas side itself, 224
pixels, rather than the formula.

```python
from alhena.io import read_image, read_slide

tiles = read_image("region.png", mpp=0.25)    # tiles of 381 pixels
slide = read_slide("slide.svs")               # the side the slide's own resolution implies
print(tiles.tile_shape, slide.mpp, slide.tile_shape)
```

Both sizes take `pad_fraction` to zero and keep `crop_fraction` below 1%. A field larger than
that is cut into several tiles of this size, as
[a large field is cropped](#a-large-field-is-cropped) sets out.

Read the resolution off the slide rather than assuming it from the scanner magnification.
`alhena.io.read_slide` does that for you, and the source carries it on to `encode_tiles`,
`encode_cells` or `annotate`.
See [How a slide's resolution is read](../inputs/images.md#how-a-slides-resolution-is-read).

## Masks

Nucleus masks undergo the same spatial transformation as their corresponding H&E tile so that
nucleus pixels remain aligned with image tokens. Masks use nearest-neighbor interpolation and
background label `0`. See [Masks on the canvas](cell-pooling.md#masks-on-the-canvas).

## Check before you run

Before processing a new image source, inspect at least:

- `field_of_view_um`: physical width represented by each input tile;
- `pad_fraction`: fraction of the model canvas filled by padding;
- `crop_fraction`: fraction of the input field removed before inference;
- `median_nucleus_tokens`: side of a typical nucleus, in tokens, when nucleus sizes are available.

```python
import alhena

report = alhena.describe_geometry(mpp=0.25, tile_size_px=256, median_nucleus_area_px=180.0)

print(report.field_of_view_um)      # 64.0
print(report.token_footprint_um)    # 5.95
print(report.mpp_ratio)             # 0.588...
print(report.pad_fraction)          # 0.545...
print(report.crop_fraction)         # 0.0
print(report.median_nucleus_tokens)
for note in report.warnings:
    print(note)
```

`CellTyper.preflight()` performs the same geometry checks together with consistency checks
between tiles and nuclei:

```python
typer = alhena.load_celltyper(tissue="Breast")
report = typer.preflight(tiles, nuclei=masks, mpp=0.425)
```

`encode_cells()` and `annotate()` perform these checks automatically before the image-encoder
forward pass. [When the checks run](#when-the-checks-run) gives their exact order.

The remaining sections are reference material for advanced resolution validation, warning behavior,
and model operating geometry. Most users can stop here once the reported geometry is appropriate
for their data.

## Advanced: resolution validation

One rule checks a declared resolution, wherever it is declared: to a reader, to `prepare_tiles`,
`encode`, `encode_tiles`, `encode_cells` or `annotate`, or to a membership builder or `fit_mask`.

- `mpp` must be a finite positive number.
- Values above 20 µm/px are refused because the model field of view would contain too few source
  pixels.
- Values below 0.02 µm/px are accepted but generate a warning because they are unusually fine for
  microscopy and can indicate a unit error, such as a length in millimeters given where microns
  are meant.

`describe_geometry` refuses what the others refuse, and describes the rest without warning. A
resolution a slide declares for itself is refused outside the same range, at either end.
See [How a slide's resolution is read](../inputs/images.md#how-a-slides-resolution-is-read).

## Advanced: geometry warnings

Geometry concerns are raised as `GeometryWarning`. It is a `UserWarning`, so
`warnings.catch_warnings` catches it and `-W error` turns it into an error.

`prepare_tiles` raises it in three situations, which depend on the resolution alone:

??? warning "No resolution declared"
    Tiles are resized to 224 as they are, and nothing then ensures they cover the field the model
    reads. The warning is raised once per call, and the run records the resolution as
    undeclared, on `info.mpp_declared`.

    ```text
    Tile resolution was not declared, so tiles are resized to 224 by 224 pixels as they are,
    without matching their physical scale to the 0.425 micron per pixel the encoder expects.
    Pass mpp= with the tiles' source resolution in microns per pixel.
    ```

??? warning "Coarser than the reference"
    Reaching 0.425 µm/px from a coarser resolution enlarges the tile, which invents no detail.
    The adapters were distilled at the finer resolution only.

    ```text
    Tiles are at 0.5 micron per pixel, coarser than the 0.425 the encoder reads, so reaching
    that geometry enlarges them by 1.18. Upsampling cannot recover image detail; the adapters
    were trained at the finer resolution.
    ```

    The warning reports the resolution of the pixels, not the size of the field, so no tile size
    silences it: tiles at 0.5 µm/px raise it on every call.

??? warning "Finer than a microscope image"
    A resolution below 0.02 µm/px is used as declared, and warned about wherever it is declared.
    This can happen when a length expressed in millimeters is interpreted as though it were in microns.

    ```text
    mpp=0.01 declares pixels finer than the 0.02 micron of a microscope image. This can happen
    when a length in millimeters is interpreted as microns. It is used as declared, and reaching
    the 0.425 micron per pixel the encoder reads shrinks each tile 42 times.
    ```

Reading a slide can raise geometry warnings of its own, such as this one:

??? warning "Axes that disagree"
    A slide whose two axes have resolutions more than the tolerance apart emits a
    `GeometryWarning`. Tiles are resampled by the resolution across in both directions, so their
    vertical axis is off by the ratio of the two.

Two further warnings are raised only where the geometry reaches a result:

- **White pad.** `encode` and `encode_tiles` raise a `GeometryWarning` on any pad at all, because
  nothing after them keeps the pad out of a tile embedding. There is no threshold. `encode_cells`,
  `annotate` and `encode_with_tokens` stay silent on the same tiles, because the pad carries no
  direct pooling weight there. The tile embeddings `encode_cells` returns alongside the cells do
  carry the pad, so read `cells.tiles.pad_fraction` before using them elsewhere.
- **Center crop.** `encode` and `encode_tiles` raise a `GeometryWarning` when the crop discards
  more than `CROP_WARNING_FRACTION`, 5 percent, of a field. `describe_geometry` then carries a
  note as well. `preflight` raises that note, and `encode_cells` and `annotate` raise it once the
  membership is built, adding how many of the nuclei given reached no token.

`prepare_tiles` and `describe_geometry` raise no warning about either share. Both shares are
recorded whether or not a warning is raised:

- as `pad_fraction` and `crop_fraction` on `ResampleInfo` and on `GeometryReport`;
- on cells, as `cells.geometry.pad_fraction` and `cells.geometry.crop_fraction`;
- on the tile embeddings of a cell pass, as `cells.tiles.pad_fraction` and
  `cells.tiles.crop_fraction`;
- on an annotation result, as `result.meta["pad_fraction"]` and `result.meta["crop_fraction"]`,
  alongside `mpp` and `field_of_view_um`.

`alhena encode` records both with the geometry of every file of tile embeddings it writes, and
prints them as `white pad` and `cropped away`.

`pad_fraction` never becomes a note of a geometry report, so it does not reach the warnings of
`preflight`. Check it yourself whenever the embeddings are used outside the cell path, and
whenever their padded image context matters.

The crop threshold is not zero because rounding alone crops a little. The tile size from
[how large to cut](#how-large-to-cut) crops nothing at 0.25 µm/px, and one row and one column at
0.5 µm/px, 0.9 percent of the field. Across resolutions up to 2.3 µm/px, rounding alone crops at
most 4.3 percent. A crop above 5 percent therefore reflects the size at which the tiles were cut
rather than this rounding effect.

A report also carries a note when a typical nucleus spans more than two tokens per side: cell
embeddings then pool over more context than they did at the distillation resolution.

### When the checks run

`describe_geometry` and `preflight` build their report from shapes alone, so their cost is
negligible beside a forward pass. `preflight` needs no encoder: a typer loads its encoder only
when it is first given tiles to embed.

`encode_cells` and `annotate` perform the same checks automatically before building the
membership table, loading an encoder or starting the forward pass. Geometry or nucleus-input
errors are therefore reported early. The crop warning is delayed until the membership table exists
because it also reports how many nuclei reached no token, but it is still raised before the
encoder is loaded.

`preflight`, `encode_cells` and `annotate` build the report for square tiles as tall as the
first one. Where tiles differ in size, or are not square, `plan_resample` gives the shares of
each shape, and the tile embeddings of a run record the largest crop over the tiles read, as
`cells.tiles.crop_fraction`.

## Advanced: model operating resolutions

| Quantity | Value | What it is |
|---|---|---|
| `REFERENCE_MPP` | **0.425 µm/px** | The resolution the adapters were distilled at. Everything is resampled to it. |
| `BACKBONE_NATIVE_MPP` | **0.5 µm/px** | The UNI2-h backbone's own native resolution. |

`REFERENCE_MPP` is exported at the top level. Masks are fitted to it and geometry reports are
built against it, so checking a run needs no encoder. A loaded encoder carries the same value as
`encoder.info.reference_mpp`.

`BACKBONE_NATIVE_MPP` is a documented operating point only. Nothing in the package resamples to
it: `REFERENCE_MPP` is the only target the code aims at.

The patient-level evaluations reported in the paper read 224-pixel tiles at 0.5 µm/px for every
encoder, ALHENA included, without resampling. The package resamples to `REFERENCE_MPP` instead,
so it does not reproduce the geometry of those evaluations. The [Cite page](../cite.md) lists the
paper.

The canvas and its token grid are fixed by four constants:

| Constant | Value |
|---|---|
| `INPUT_SIZE` | 224 pixels |
| `PATCH_SIZE` | 14 pixels |
| `TOKEN_GRID` | 224 / 14 = **16** |
| `NUM_TOKENS` | 16 x 16 = **256** |

The physical size of the canvas follows from the resolution it is read at:

| Read at | Field of view of one tile | Side of one token |
|---|---|---|
| 0.425 µm/px | 224 x 0.425 = **95.2 µm** | 14 x 0.425 = **5.95 µm** |
| 0.5 µm/px | 224 x 0.5 = **112.0 µm** | 14 x 0.5 = **7.0 µm** |

Only the first row is the geometry the package works in. Tiles are resampled to `REFERENCE_MPP`
before the backbone sees them, so a token covers **5.95 µm**, whatever the tiles declare.
`describe_geometry` returns that number as `token_footprint_um`. The second row is what the same
canvas would cover at the backbone's native resolution, which nothing in the package reads at.

Compute them yourself rather than trusting the table:

```python
from alhena import BACKBONE_NATIVE_MPP, INPUT_SIZE, PATCH_SIZE, REFERENCE_MPP

for mpp in (REFERENCE_MPP, BACKBONE_NATIVE_MPP):
    print(f"{mpp} um/px: a 224 pixel tile covers {INPUT_SIZE * mpp:.2f} um")
print(f"a token covers {PATCH_SIZE * REFERENCE_MPP:.2f} um, whatever the tiles declare")
```

Changing one of these quantities gives a different model, not a different configuration, which is
why they are constants. `REFERENCE_MPP`, `BACKBONE_NATIVE_MPP`, `INPUT_SIZE` and `PATCH_SIZE` are
exported at the top level. `TOKEN_GRID` and `NUM_TOKENS` follow from `INPUT_SIZE` and
`PATCH_SIZE`, and are not. A loaded encoder carries three of them on `encoder.info`, as
`reference_mpp`, `input_size` and `patch_size`.

## Advanced: why the canvas size is fixed

The package pins the canvas rather than relying on the model to object.

- `prepare_tiles` always produces `(N, 3, 224, 224)`. A batch you prepared yourself is refused
  unless it has exactly that shape.
- `encode_with_tokens` checks the grid it got back, and raises `BundleError` when the token
  count is not 256: the canvas is fixed, so only another backbone can return another count.
- `pool_cells` refuses tokens that do not carry 256 positions per tile.

!!! warning "Changing the canvas changes the token grid"
    The backbone is built with `dynamic_img_size=True`. A model built that way accepts any input
    whose side is divisible by 14, without complaint, and changes the token grid to match. A 280
    pixel input is 20 patches across, so it yields a 20 by 20 grid of 400 tokens, not 16 by 16
    of 256. The tile embedding still comes out at the right width, so nothing in the shape of the
    result shows the change.

    Token indices are geometric. A membership table built on a 16 by 16 grid addresses different
    tissue on a 20 by 20 one. [Cell pooling](cell-pooling.md) sets out the mapping from pixels to
    tokens.

---

**Next:** [Cell pooling](cell-pooling.md)
