# Preprocessing

`alhena.preprocess` contains the image-geometry operations applied before the encoder.

`prepare_tiles()` resamples input tiles to ALHENA's model resolution and maps them to the
224 × 224 canvas. Without a declared resolution, tiles are resized to the canvas as they are, so
nothing is padded or cropped. `describe_geometry()` and `plan_resample()` report the
corresponding transformation from image shapes alone.

!!! warning "Padding"
    An input field smaller than the model field of view is centered and padded with white
    pixels. `pad_fraction` records the affected fraction of the model canvas.

!!! warning "Cropping"
    An input field larger than the model field of view is center-cropped. `crop_fraction`
    records the fraction of the resampled field discarded outside the model canvas.

See [Geometry](../concepts/geometry.md) for interpretation and recommended tile sizes.

!!! note "What normalization means here"
    `normalize_tile()` applies fixed channel normalization; it does not perform stain
    normalization.

::: alhena.preprocess.prepare_tiles

::: alhena.preprocess.describe_geometry

::: alhena.preprocess.plan_resample

::: alhena.preprocess.channel_lut

::: alhena.preprocess.normalize_tile

::: alhena.preprocess.resample

::: alhena.preprocess.fit_to_size

::: alhena.preprocess.UNDECLARED_RESOLUTION_WARNING

---

**Next:** [Cells and pooling](cells.md)
