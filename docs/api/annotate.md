# Retrieval

Lower-level operators for reference retrieval and voting in the shared cell space.

Each query cell retrieves nearby labeled reference cells. Their cosine similarities are converted
into normalized neighbor weights, and those weights are aggregated by class to obtain the class
masses used for annotation.

`CellTyper.annotate()` provides the high-level workflow and performs compatibility and provenance
checks around these lower-level functions.

::: alhena.annotate.annotate_cells

::: alhena.annotate.retrieve

::: alhena.annotate.vote

::: alhena.annotate.l2_normalize

::: alhena.annotate.THIN_SUPPORT_WARNING

::: alhena.annotate.MAX_DEFAULT_WORKERS

---

**Next:** [References](reference.md)
