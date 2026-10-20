# Cell typing

ALHENA assigns cell types by retrieving labeled single-cell reference cells in the shared cell
space.

```python
import alhena

typer = alhena.load_celltyper(tissue="Breast")
result = typer.annotate(cells)
frame = result.to_frame()
```

For each image-derived cell, the retrieved reference neighbors contribute a similarity-weighted
vote over the classes represented by the selected reference. The winning class becomes the
annotation, and `confidence` reports the fraction of vote mass assigned to that class.

Two parameters define this retrieval operator. `k` is the number of reference neighbors retained
for each image-derived cell, and `tau` is the softmax temperature used when their similarities are
converted into voting weights. ALHENA uses `k=30` and `tau=0.03` by default. See
[Retrieval operator](#retrieval-operator) for the detailed behavior and envelope checks.

Confidence measures agreement among retrieved reference cells; it is not a calibrated probability
of correctness. See [Confidence](#confidence-is-a-class-mass-not-a-similarity).

Previously computed `CellEmbeddings` can be annotated without running the image encoder again.
See [Annotating cells again](#annotating-cells-again).

The final sections describe evaluation when independent ground-truth cell labels are available.

## The output

`annotate()` returns an `AnnotationResult` with one row for each encoded cell. It contains the
predicted labels, confidence values, validity flags, class masses, retrieval diagnostics, the
underlying `CellEmbeddings`, and annotation provenance.

See [Cell annotations](../outputs/cell-annotations.md) for the complete result schema.

```python
print(result.axis, result.class_order)
print(result.meta["reference"], result.meta["k"], result.meta["tau"])
print(result.meta["effective_k"], "neighbors carry the vote, as a median")
print(result.meta["n_cells"], "cells,", result.meta["n_called"], "annotated")
print(result.meta["n_cells_without_tokens"], "nuclei reached no token")
print(result.meta["unexpressed_classes"])
print(result.meta["reference_class_support"])
```

The mass columns of `to_frame()` are named after the classes of the axis the reference was loaded
on, with three on `compartment_v1` and five on `lineage_v1`. The result also carries
`neighbor_similarity`, an `(n_cells, k)` array of the retrieved cosines. It is a diagnostic, and the
next section explains what it is not.

!!! tip "Save annotations together with their embeddings and provenance"
    `result.write_h5ad("annotations.h5ad")` preserves the annotation together with the underlying
    `CellEmbeddings` and provenance. This information is useful for interpreting or reproducing the
    annotation later. See [Persistence and interchange](../outputs/persistence.md).

## Confidence is a class mass, not a similarity

The retrieved neighbors contribute a normalized vote over the available classes. These class
masses sum to 1 for each valid cell.

`confidence` is the largest class mass: the fraction assigned to the winning class.

With `C` classes, confidence lies between `1/C` and `1`. The floor is therefore:

- **0.2** for the five-class `lineage_v1` axis;
- **1/3** for the three-class `compartment_v1` axis.

A value near the floor indicates a divided vote. A value near 1 indicates strong agreement among
the retrieved reference neighbors.

```python
import alhena

for axis in ("lineage_v1", "compartment_v1"):
    classes = [name for name in alhena.label_order(axis) if name != "Unknown"]
    print(axis, len(classes), "classes, confidence in", (1 / len(classes), 1.0))
```

!!! warning "Do not use raw similarity as a universal confidence threshold"
    `neighbor_similarity` records the cross-modal cosine similarities used to rank reference cells.
    Their absolute magnitude is not calibrated as a probability or universal quality score. Use
    `confidence` to summarize agreement among the retrieved classes, and calibrate any filtering
    threshold on labeled data relevant to the application.

### What a confidence value tells you

- **Near the class-count floor:** the retrieved neighbors distribute their vote across several
  classes.
- **Near 1:** most of the weighted vote is assigned to the same class.
- **Neither value establishes correctness.** Confidence describes agreement among the retrieved
  reference cells, not the probability that the prediction matches biological ground truth.

ALHENA does not define a universal confidence cutoff. If annotations need to be filtered, choose
the threshold on labeled data from a relevant cohort, tissue, and annotation axis.
[`confidence_sweep()`](#evaluating-confidence-based-filtering) can be used to examine the
consequences of different thresholds.

```python
# Illustrative threshold only; calibrate on labeled data for your application.
threshold = 0.6
confident = frame[frame["valid"] & (frame["confidence"] >= threshold)]
print(f"{len(confident)} of {len(frame)} cells kept")
print(confident["label"].value_counts())
```

## Cells without an annotation

Two situations can produce an input nucleus without a cell-type annotation.

**No cell-embedding row.** A nucleus that cannot be associated with any image token is omitted
before annotation. `result.meta["n_cells_without_tokens"]` records how many input nuclei were
excluded this way.

**Cell retained but annotation invalid.** A cell can remain in the result with `valid=False`, an
empty label, and `NaN` confidence if its embedding cannot be used for retrieval or the cell was
marked invalid before annotation. It then carries:

| Field | Value |
|---|---|
| `labels` | `""`, the empty string |
| `confidence` | `NaN` |
| `mass` | a row of `NaN` |
| `class_entropy`, `neighbor_entropy` | `NaN` |
| `neighbor_similarity` | a row of `NaN` |
| `valid` | `False` |

```python
print(int((~result.valid).sum()), "cells have no annotation")
```

??? note "Why a nucleus reaches no token"
    A nucleus can fail to reach an image token if it lies outside the retained image crop, is
    removed during mask resampling, or loses all of its rasterized pixels through overlap
    resolution. See [Cell pooling](cell-pooling.md) and [Geometry](geometry.md).

    `result.meta["n_cells_without_tokens"]` is `None` when the membership table was assembled by
    hand, since such a table records no set of input nuclei to compare against.

## The two class axes

| Axis | Classes |
|---|---|
| `lineage_v1` (default) | `B_Plasma`, `Epithelial_Melanocyte`, `Myeloid`, `Stromal_Vascular`, `T_NK` |
| `compartment_v1` | `Epithelial_Melanocyte`, `Immune`, `Stromal_Vascular` |

```python
typer = alhena.load_celltyper(tissue="Breast", axis="compartment_v1")
```

Every prebuilt reference codes both axes, and a reference is loaded on exactly one. The axis then
determines the confidence floor, the denominator of a macro average, and the commonest-class
baseline. What the choice does to the retrieval pool is in [References](references.md).

!!! warning "Interpret the two axes separately"
    `lineage_v1` and `compartment_v1` define different classification tasks with different class
    sets and confidence floors. Performance or composition summaries from one axis should not be
    combined directly with those from the other.

??? note "An axis is chosen once"
    `load_celltyper` refuses a reference already loaded on one axis when another is asked for,
    rather than relabeling it.

??? note "The axis moves the support"
    An axis changes the class support, not the cells. The same reference reports three classes
    on `compartment_v1` where `lineage_v1` reports five, because the three immune classes are one
    class there. A thinly supported class can stop being thin, because it is no longer a class.

??? warning "Class order is part of the protocol"
    `class_order` determines the correspondence between class names, class-mass columns, and
    internal integer codes. It also provides the deterministic tie-breaking order when two classes
    receive exactly the same vote mass. Preserve the class order recorded by the reference.

## What the reference can and cannot say

A reference can predict only classes represented on the selected annotation axis. Classes with no
reference cells cannot be returned, and classes with limited reference support should be
interpreted with additional caution.

Each annotation result records these conditions in `meta["unexpressed_classes"]` and
`meta["thinly_supported_classes"]`.

A biological category not represented by the annotation axis is not automatically detected as
unknown. For example, the built-in axes describe broad cell classes rather than malignancy or
activation state. See
[Categories not represented by an axis](references.md#categories-not-represented-by-an-axis).

## Annotation workflow

1. **Validate image and nucleus geometry.**
2. **Map nuclei onto the model token grid.**
3. **Resolve nuclei represented in overlapping tiles.**
4. **Encode the required H&E tiles.**
5. **Pool image tokens to obtain pre-projection cell embeddings.**
6. **Project the cells into the shared cell space with `img_tok_proj`.**
7. **Retrieve the nearest labeled reference cells.**
8. **Convert neighbor similarities into normalized weights using the retrieval operator.**
9. **Sum the weighted vote by class and select the class with the largest mass.**

Steps 1–6 produce `CellEmbeddings`. Steps 7–9 are the reference-based annotation step.

## Annotating cells again

Cell encoding depends on the H&E image and nuclei but not on the single-cell reference. Save
`CellEmbeddings` once and reuse them with another compatible reference or annotation axis:

```python
import alhena

encoder = alhena.load_encoder()
cells = encoder.encode_cells(source, nuclei)
cells.write_h5ad("cells.h5ad")

cells = alhena.io.read_cells("cells.h5ad")
typer = alhena.load_celltyper(tissue="Breast", axis="compartment_v1")
result = typer.annotate(cells)
```

When `CellTyper` receives previously computed `CellEmbeddings`, it performs only projection-space
retrieval and annotation; the image encoder is not run again.

Compatibility is checked through `cell_space_id`. Cells and references from incompatible learned
cell spaces raise `CellSpaceMismatchError`.

Annotations of the same cells on several axes, or against several references, can share one
table. `alhena.io.add_annotation(table, result)` adds each beside the others, under fields named
after it. See
[Several annotations of the same cells](../outputs/persistence.md#several-annotations-of-the-same-cells).

## Retrieval operator

ALHENA uses `k=30` retrieved neighbors and a softmax temperature `tau=0.03` by default. These
parameters belong to the retrieval operator rather than to a particular tissue reference.

| | Value | Constant |
|---|---|---|
| Neighbors voted over | **30** | `alhena.DEFAULT_K` |
| Softmax temperature | **0.03** | `alhena.DEFAULT_TAU` |

`k` and `tau` can be changed, but doing so changes the annotation procedure, and changing `k` may
trigger `EnvelopeWarning` when the configuration moves outside the range assumed by ALHENA.

??? warning "Moving k trips an envelope check"
    Raise `k` until the thinnest class of the reference no longer has ten cells per neighbor,
    and an `EnvelopeWarning` names the ratio it crossed. Lower `k` below three times the number
    of classes, and a second `EnvelopeWarning` fires.

    `tau` has no envelope check beyond being positive. Moving it is entirely up to you, and
    [the effective number of neighbors](#effective-number-of-neighbors) shows what a value did to
    the weights.

### Effective number of neighbors

The `k` retrieved neighbors do not necessarily contribute equally. Their weights are obtained by
applying a softmax to the similarities scaled by `tau`.

`neighbor_entropy` summarizes how evenly these weights are distributed. `exp(neighbor_entropy)` can
be interpreted as an effective number of contributing neighbors: it approaches `k` when weights are
uniform and approaches 1 when one neighbor dominates.

This quantity describes the distribution of neighbor weights, not agreement between classes. Class
agreement is summarized by `confidence`.

## Advanced: retrieval implementation

Retrieval first gathers, for each query, the 200 most similar reference cells
(`alhena.RETRIEVAL_DEPTH`), and then keeps the `k` nearest. A reference of several hundred
megabytes is walked block by block, and only those 200 cells per query are carried from one block
to the next. `chunk_size` and `query_chunk_size` set the size of a block; they are arguments of
`annotate_cells`, not of `CellTyper.annotate`.

Those 200 cells are then scored a second time, each pair on its own over the width of the
embedding, and that score orders them. A matrix product, which scores the blocks, sums in an order
that depends on the shape it is given, so a cosine can move by a unit in the last place: enough to
swap two neighbors of nearly equal similarity. The second score depends on the pair alone.

Ordering is by descending similarity, then by ascending reference row, at every stage. A tie at the
`k`-th position therefore resolves on the reference row, not on the block size. This holds while at
most `RETRIEVAL_DEPTH - k` reference rows, 170 at the default `k`, tie with the `k`-th neighbor.
Beyond that, as with many identical rows, the block size can choose among them.

## Advanced: evaluation with known labels

Evaluation requires one ground-truth label for each row of the `AnnotationResult`.
`align_truth()` puts external labels in the order of the annotated cells, matching them by
`cell_id`:

```python
truth = alhena.align_truth(result, known)
```

`known` is a table with `cell_id` and `label` columns, a Series of labels indexed by identifier, a
mapping from identifier to label, or the path of a `.csv` or `.parquet` table. Identifiers are
matched as written, so the integer `12` and the text `"12"` name different cells, and an annotated
cell without a label is refused. `score()` and `confidence_sweep()` verify the vector length but
cannot determine whether rows are correctly aligned.

When only some cells carry a known label, keep those cells with `select()` first. It takes a
boolean mask, or the positions of the cells to keep, and cuts the annotations and their cells
alike:

```python
import numpy as np

labeled = result.select(np.isin(result.cell_id, known["cell_id"]))
truth = alhena.align_truth(labeled, known)
report = alhena.score(labeled, truth)
sweep = alhena.confidence_sweep(labeled, truth)
```

`score()` and `confidence_sweep()` take the annotations themselves, which carry their class order.

!!! warning "Align ground truth by cell identifier"
    Never assume that external labels are already in the same row order as the ALHENA result.

### Evaluating confidence-based filtering

`confidence_sweep()` evaluates how retained-cell fraction and classification metrics change as
increasingly strict confidence thresholds are applied.

Filtering low-confidence predictions can improve some aggregate metrics, but it does not guarantee
improvement for every class. Inspect per-class recall together with pooled metrics.

```python
sweep = alhena.confidence_sweep(result, truth)
print(sweep[["threshold", "n_retained", "retained_fraction", "accuracy", "macro_f1"]])
print(sweep[[column for column in sweep.columns if column.startswith("recall_")]])
```

The sweep has one row per threshold, on the default grid: `0.0`, then `0.30` to `0.95` in steps of
`0.05`. The first row keeps every cell with a valid annotation. Each row also carries
`n_present_classes`, the number of classes entering the macro average, together with
`floor_accuracy` and `floor_macro_f1`, the scores of the commonest-class baseline on the same
retained cells. `frequency_matched_macro_f1` reports the frequency-matched baseline for that same
population, which [Macro metrics and baselines](#macro-metrics-and-baselines) describes.

`alhena annotate --truth` prints this sweep, and names at its foot every class whose recall is lower
at the last threshold than at the first. See [Annotation output](../cli.md#annotation-output).

Per-class metrics can move differently from aggregate accuracy because confidence distributions and
error patterns differ between classes. Use the per-class columns returned by `confidence_sweep()`
rather than assuming that one threshold benefits every class equally.

### Macro metrics and baselines

`score()` reports accuracy, balanced accuracy, macro F1, the number of classes included in the
macro denominator, and two baselines:

- **commonest-class baseline:** always predict the most frequent class;
- **frequency-matched baseline:** the macro F1 of predictions that follow the observed class
  frequencies but not the cells. It is computed from the expected counts, not sampled: each class
  scores an F1 equal to its fraction of the evaluated cells.

Baselines are computed on the same evaluated population and with the same macro-support rule as the
reported metric. Compare metrics only with the corresponding baseline returned by the same report.

Cells retained in the evaluated population but lacking a valid annotation count as errors rather
than being silently removed.

### Ground-truth classes absent from the reference

Ground-truth data may contain a class that the selected reference cannot represent. Two evaluation
populations answer different questions:

- **Include those cells** to evaluate the complete deployed system, including limitations of the
  selected reference.
- **Exclude those cells** to evaluate performance only on classes the reference can express.

Neither choice is universally preferable. Report which population was used and do not combine
metrics computed under different definitions.

---

**Next:** [References](references.md)
