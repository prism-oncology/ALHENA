# Cell annotations

`annotate()` returns an `AnnotationResult` containing:

- the predicted cell type, confidence, and validity flag for each cell;
- the `CellEmbeddings` used for annotation, available as `result.cells`;
- annotation provenance and reference information in `result.meta`.

!!! tip "What can I learn from this?"
    The cellular composition of the tissue and, when cell positions are available, the spatial
    organization of the predicted cell types.

```python
result = typer.annotate(cells)

frame = result.to_frame()                       # one row per cell
print(frame[["cell_id", "label", "confidence", "valid"]].head())
```

## Cell-type predictions

Three fields summarize the prediction for each cell:

| Field | Meaning |
|---|---|
| `labels` | predicted cell type, or `""` when no annotation was assigned |
| `confidence` | fraction of the weighted reference-neighbor vote assigned to the winning class |
| `valid` | whether a cell-type annotation was assigned |

Annotation is based on retrieval in the shared cell space. Nearby reference cells contribute a
weighted vote, and the class with the largest total mass is selected.

Two parameters control this retrieval. `k` is the number of reference neighbors retained for each
cell, and `tau` is the softmax temperature used when their similarities are converted into voting
weights. The defaults are `k=30` and `tau=0.03`. See
[Retrieval operator](../concepts/cell-typing.md#retrieval-operator) for the full definition and
retrieval-envelope checks.

`confidence` is the fraction of this class mass assigned to the winning class. Its minimum possible
value depends on the number of classes: 0.2 for the five-class `lineage_v1` axis and one third for
the three-class `compartment_v1` axis.

A value near the class-count floor indicates a divided vote, while a value near 1 indicates strong
agreement among the retrieved neighbors. It should not be interpreted as the probability that the
prediction is correct.

ALHENA does not prescribe a universal confidence threshold. If thresholding is needed, calibrate it
on labeled data relevant to the intended application. See
[Confidence is a class mass, not a similarity](../concepts/cell-typing.md#confidence-is-a-class-mass-not-a-similarity).

`to_frame()` returns the annotation results as one row per cell:

| Column | Holds |
|---|---|
| `cell_id` | the identifier of the cell |
| `label` | the predicted class, or `""` when no annotation was assigned |
| `confidence` | the winning class's share of the vote |
| `class_entropy` | the entropy of the class mass |
| `neighbor_entropy` | the entropy of the neighbor weights |
| `valid` | whether an annotation was assigned |
| `mass_<class>` | the class mass for that class, one column per class, with all mass columns summing to one for each annotated cell |

When the result carries the cells it was made from, as `annotate()` returns it, the table also
holds what each cell records apart from its embeddings:

| Column | Holds |
|---|---|
| `x`, `y` | the position of the nucleus, in the frame `result.cells.centroids_frame` names; only where the cells record positions |
| `tile_index` | the index of the tile the cell was read from |
| `tile_id` | the name of that tile; only where the cells carry the tiles they were read from, which cells read back with `read_cells()` do not |
| `nucleus_px` | the nucleus pixels supporting the cell |
| `n_candidate_tiles` | how many tiles contained the nucleus |
| `embedding_valid` | whether the cell embedding is usable |

This table is much smaller than the embeddings, so it is the form to keep when a run annotates
many cells.

## Diagnostics

Additional fields describe the retrieval and class-mass distribution behind each prediction:

- **`mass`**, `(n_cells, n_classes)`: the class mass for every class, in the order of
  `result.class_order`.
- **`class_entropy`**: how evenly the vote spread over the classes.
- **`neighbor_entropy`**: how evenly the neighbors were weighted. Its exponential is the
  effective number of neighbors, and `meta["effective_k"]` gives its median.
- **`neighbor_similarity`**, `(n_cells, k)`: cosine similarity of each retrieved reference
  neighbor. Its absolute value is not calibrated across images or references and should not be
  used as a universal threshold.

## Provenance

`result.meta` records the settings and data provenance needed to interpret or reproduce the
annotation:

- the reference and annotation axis identify which labeled reference population and taxonomy were
  used;
- `k` and `tau` record the retrieval settings used to turn reference neighbors into the annotation;
- the encoder and cell-space identities identify the learned representations used for the H&E
  cells and reference, and `backbone_digest` identifies the UNI2-h model weights;
- the declared resolution, field of view, and geometry note record the image geometry presented to
  the model;
- the nucleus-input fields record how nuclei were supplied and whether their shapes were
  approximated;
- `unexpressed_classes` records classes that the reference cannot predict, while
  `thinly_supported_classes` records classes with limited reference support;
- `reference_class_support` records the number of cells available for retrieval in each class;
- `n_cells_without_tokens` records how many input nuclei produced no cell-embedding row because
  they reached no image token.

Preserve this metadata when saving annotations; `result.write_h5ad()` does so automatically.

## Missing rows and unassigned annotations

Two situations should be distinguished:

- **The nucleus has no output row.** A nucleus that could not be associated with any image token is
  omitted earlier, when the `CellEmbeddings` are created. It therefore never reaches the annotation
  step. `meta["n_cells_without_tokens"]` records how many input nuclei were excluded for this
  reason.
- **A `CellEmbeddings` row exists but cannot be annotated.** Annotation preserves rows that already
  exist in the input `CellEmbeddings`. If such a row is marked `valid=False` or its embedding
  cannot be used for retrieval, the cell remains in the `AnnotationResult` so its identifier and
  provenance are preserved, but it receives an empty `label`, `confidence=NaN`, and
  `valid=False`.

Exclude rows with `valid=False` when summarizing cell-type composition. See
[Cells without an annotation](../concepts/cell-typing.md#cells-without-an-annotation).

## Cell embeddings used for annotation

`result.cells` contains the `CellEmbeddings` used to generate the annotations, in the same row
order as the `AnnotationResult`. `result.cell_id` and `result.cells.cell_id` therefore match
directly without an additional join.

`result.select()` keeps the annotations of some cells, given as a boolean mask or as positions,
with those cells attached in the same order. `CellEmbeddings.select()` does the same for cells
alone.

`result.missing_classes` names the classes the reference holds no cell of, as the run recorded
them. `viz.composition()` and `viz.legend()` mark them by default, and so does `viz.confusion()`
for a report `score()` made from the annotations. `viz.confidence_curve()` marks them too, for a
sweep `confidence_sweep()` made from the annotations.

## As an AnnData table

`result.to_anndata()` returns the table of [cell embeddings](cell-embeddings.md#as-an-anndata-table),
with the annotation fields added. Each field is named after the annotation: by default its axis,
written `<name>` below. An annotation on `lineage_v1` therefore adds `alhena_celltype_lineage_v1`.

| Where | Key | Holds |
|---|---|---|
| `obs` | `alhena_celltype_<name>` | predicted cell type; `""` when no annotation was assigned |
| `obs` | `alhena_confidence_<name>` | the winning class's share of the vote |
| `obs` | `alhena_valid_<name>` | whether an annotation was assigned |
| `obs` | `alhena_class_entropy_<name>`, `alhena_neighbor_entropy_<name>` | the two entropies |
| `obsm` | `alhena_mass_<name>` | the class mass, in class order |
| `uns` | `alhena`, then `annotations`, then `<name>` | annotation provenance, axis, and class order |
| `uns` | `alhena_celltype_<name>_colors` | the color of each cell type, which Scanpy uses; for the default palette only |

`io.read_annotations` reads an annotation back from this table, with its cells, to draw or
evaluate it later. Because the underlying `CellEmbeddings` are retained, `io.read_cells` can also
read them back alone, and they can later be annotated with another compatible reference without
running the image encoder again.

The same table can hold several annotations of its cells. `io.add_annotation(table, result)` adds
one beside those the table already holds, such as an annotation of the same cells on another
axis. Annotations on one axis against two references take a name each through `name=`. See
[Several annotations of the same cells](persistence.md#several-annotations-of-the-same-cells).

## Continue with downstream analysis

Cell-type annotations can be analyzed together with the pre-projection cell embeddings, cell
positions when available, and tile-level representations. See
[Downstream analysis](downstream-analysis.md) for clustering, spatial neighborhoods, and
multi-scale analyses.

---

**Next:** [Downstream analysis](downstream-analysis.md)
