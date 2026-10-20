# Cell typer

`CellTyper` combines a compatible single-cell reference with ALHENA's cell-annotation workflow.
`annotate()` accepts either an H&E image plus nuclei or previously computed `CellEmbeddings`.

When previously encoded cells are supplied, the image encoder is not run again.

!!! tip "The image encoder is loaded only when needed"
    A `CellTyper` can annotate previously computed `CellEmbeddings` using only the compatible
    reference. UNI2-h weights and the encoder bundle are required only when H&E images still need
    to be encoded.

!!! tip "Preflight before a long run"
    `CellTyper.preflight` builds the same geometry report as `describe_geometry`, from shapes
    alone, and raises its notes as `GeometryWarning` before the forward pass starts. It also
    checks that the tiles and nuclei describe the same set of tiles, that a membership table maps
    its masks onto the canvas with the same geometry as the images, and that cell identifiers are
    integers or text. It evaluates geometry and input consistency without running the image
    encoder.

::: alhena.api.CellTyper

::: alhena.api.load_encoder

::: alhena.api.load_celltyper

The forms tiles and nuclei take are `TileInput` and `NucleiInput`, on [Types](types.md).

`load_encoder` finds the UNI2-h weights with `find_backbone`, and both loaders download into the
directory `cache_root` returns.

::: alhena.hub.find_backbone

::: alhena.hub.cache_root

---

**Next:** [Reading data](io.md)
