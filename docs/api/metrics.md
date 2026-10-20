# Scoring

Functions for evaluating ALHENA annotations when independent ground-truth labels are available.

`align_truth()` puts known labels in the order of the annotated cells, `score()` returns
classification metrics with both baselines, `chance_floor()` computes the commonest-class baseline
from the known labels alone, and `confidence_sweep()` evaluates performance across confidence
thresholds. `score()` and `confidence_sweep()` take the `AnnotationResult` itself.

[Tutorial 6](../tutorials/06_annotation_evaluation.ipynb) shows them on an example.

::: alhena.metrics.align_truth

::: alhena.metrics.score

::: alhena.metrics.chance_floor

::: alhena.metrics.confidence_sweep

::: alhena.metrics.NO_CALL_LABEL

::: alhena.metrics.DEFAULT_CONFIDENCE_THRESHOLDS

::: alhena.metrics.LabelInput

::: alhena.metrics.KnownLabels

::: alhena.metrics.TRUTH_COLUMNS

---

**Next:** [SpatialData](spatial.md)
