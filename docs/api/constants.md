# Constants

Fixed quantities used by the released ALHENA model, including image geometry, retrieval defaults,
annotation axes, and the tissue-reference registry.

!!! warning "The built-in annotation axes are distinct"
    `lineage_v1` and `compartment_v1` define different classification tasks. Interpret annotations
    and evaluation metrics separately for each axis.

!!! note "Not importable from the package"
    Thirteen names here live in `alhena._constants` alone: `TOKEN_GRID`, `NUM_TOKENS`,
    `UNKNOWN_LABEL`, `MPP_TOLERANCE`, `CROP_WARNING_FRACTION`, `MIN_MICROSCOPE_MPP`,
    `MAX_MICROSCOPE_MPP`, `MIN_REFERENCE_CELLS_PER_CLASS`, `MIN_SUPPORT_TO_K_RATIO`,
    `MIN_K_TO_CLASSES_RATIO`, `BACKBONE_REPO`, `WEIGHTS_REPO` and `REFERENCES_REPO`. The remaining
    constants are re-exported from the top-level package, so `alhena.REFERENCE_MPP`, for example,
    refers to the same value.

::: alhena._constants.REFERENCE_MPP

::: alhena._constants.BACKBONE_NATIVE_MPP

::: alhena._constants.INPUT_SIZE

::: alhena._constants.PATCH_SIZE

::: alhena._constants.DEFAULT_K

::: alhena._constants.DEFAULT_TAU

::: alhena._constants.RETRIEVAL_DEPTH

::: alhena._constants.LINEAGE_V1

::: alhena._constants.COMPARTMENT_V1

::: alhena._constants.CLASS_AXES

::: alhena._constants.DEFAULT_AXIS

::: alhena._constants.label_order

::: alhena._constants.CLASS_COLORS

::: alhena._constants.TISSUE_TO_SHARD

::: alhena._constants.TISSUE_ALIASES

::: alhena.__version__

::: alhena._constants.TOKEN_GRID

::: alhena._constants.NUM_TOKENS

::: alhena._constants.UNKNOWN_LABEL

::: alhena._constants.MPP_TOLERANCE

::: alhena._constants.CROP_WARNING_FRACTION

::: alhena._constants.MIN_MICROSCOPE_MPP

::: alhena._constants.MAX_MICROSCOPE_MPP

::: alhena._constants.MIN_REFERENCE_CELLS_PER_CLASS

::: alhena._constants.MIN_SUPPORT_TO_K_RATIO

::: alhena._constants.MIN_K_TO_CLASSES_RATIO

::: alhena._constants.BACKBONE_REPO

::: alhena._constants.WEIGHTS_REPO

::: alhena._constants.REFERENCES_REPO

---

**Next:** [FAQ](../faq.md)
