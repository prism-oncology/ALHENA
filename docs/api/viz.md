# Plotting

Plotting utilities for ALHENA tile embeddings, annotations and evaluation results. Each drawing
function accepts a Matplotlib `Axes` object and returns the same object, allowing the plots to be
composed into larger figures. The functions that draw annotations take an `AnnotationResult`, or
the table of cells it was written to, read as `io.read_annotations()` reads it.

Install the plotting dependencies with `pip install "alhena[viz]"`. They are imported only when
the plotting functions are used.

See them in use: `overlay()` in
[Tutorial 3](../tutorials/03_cell_annotation.ipynb#see-the-annotations-on-he), `tile_regions()` in
[Tutorial 4](../tutorials/04_multiscale_analysis.ipynb), custom class colors
in [Tutorial 5](../tutorials/05_custom_reference.ipynb#visualize-the-custom-annotation), and
`confusion()` and `confidence_curve()` in [Tutorial 6](../tutorials/06_annotation_evaluation.ipynb).

::: alhena.viz.overlay

::: alhena.viz.tile_regions

::: alhena.viz.composition

::: alhena.viz.confusion

::: alhena.viz.confidence_curve

::: alhena.viz.legend

::: alhena.viz.class_color

::: alhena.viz.class_palette

::: alhena.viz.missing_class_names

::: alhena.viz.NO_CALL_NAME

::: alhena.viz.NOT_IN_REFERENCE

::: alhena.viz.MISSING_HATCH

::: alhena.viz.RGB

---

**Next:** [Types](types.md)
