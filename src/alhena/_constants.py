"""Fixed quantities of the released ALHENA model and its reference registry.

Nothing here is a tunable: changing a value produces a different model, not a different setting.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from alhena._errors import AlhenaError, AlhenaValueError

# --- Distribution -----------------------------------------------------------------------

WEIGHTS_REPO: Final[str] = "prism-oncology/ALHENA"
"""Model hub repository publishing the encoder bundle."""
REFERENCES_REPO: Final[str] = "prism-oncology/ALHENA-references"
"""Model hub repository publishing the reference shards and their registry."""
HUB_REVISION: Final[str] = "v0.1.0"
"""Release tag of the weights and references repositories this version of the package reads."""
BACKBONE_REPO: Final[str] = "MahmoodLab/UNI2-h"
"""Model hub repository of the UNI2-h weights, gated by their authors."""

ADAPTERS_FILE: Final[str] = "adapters.safetensors"
HEADS_FILE: Final[str] = "heads.safetensors"
BUNDLE_CONFIG_FILE: Final[str] = "config.json"
REGISTRY_FILE: Final[str] = "registry.json"

SHARD_SUFFIX: Final[str] = ".safetensors"
"""Suffix of a reference shard file."""
SHARD_DIRECTORY: Final[str] = "shards"
"""Folder holding the shards, in the references repository and in a cache root."""

# --- Backbone and adapters --------------------------------------------------------------

BACKBONE_ARCHITECTURE: Final[str] = "vit_giant_patch14_224"

BACKBONE_TIMM_KWARGS: Final[dict[str, object]] = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}

# Two layer classes of the architecture, resolved by name in encoder.build_backbone.
BACKBONE_MLP_LAYER: Final[str] = "SwiGLUPacked"
BACKBONE_ACT_LAYER: Final[str] = "SiLU"

LORA_RANK: Final[int] = 64
LORA_ALPHA: Final[float] = 32.0

BACKBONE_DIM: Final[int] = 1536
CELL_SPACE_DIM: Final[int] = 512
SCCONCEPT_DIM: Final[int] = 512

NUM_PREFIX_TOKENS: Final[int] = 9

# --- Image geometry ---------------------------------------------------------------------

INPUT_SIZE: Final[int] = 224
"""Side of the square canvas the encoder reads, in pixels."""
PATCH_SIZE: Final[int] = 14
"""Side of one token, in pixels of the canvas."""
TOKEN_GRID: Final[int] = INPUT_SIZE // PATCH_SIZE
"""Tokens along each side of the canvas."""
NUM_TOKENS: Final[int] = TOKEN_GRID * TOKEN_GRID
"""Patch tokens of one canvas."""

IMAGE_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGE_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)
IMAGE_INTERPOLATION: Final[str] = "bilinear"

# Microns per pixel the adapters were distilled at.
DISTILLATION_MPP: Final[float] = 0.425
BACKBONE_NATIVE_MPP: Final[float] = 0.5
"""Native microns per pixel of UNI2-h, an operating point nothing resamples to."""
REFERENCE_MPP: Final[float] = DISTILLATION_MPP
"""Microns per pixel the encoder reads, which tiles of a declared resolution are brought to."""

MPP_TOLERANCE: Final[float] = 0.02
"""Relative difference between a declared mpp and the target above which tiles are resampled."""

CROP_WARNING_FRACTION: Final[float] = 0.05
"""Share of a field the center crop may discard before a `GeometryWarning` is raised."""

MIN_MICROSCOPE_MPP: Final[float] = 0.02
"""Finest microns per pixel of a microscope image: a tenth of the 0.2 micron a lens resolves."""
MAX_MICROSCOPE_MPP: Final[float] = 20.0
"""Coarsest microns per pixel of a microscope image, between a 1.25x scan and 600 dpi."""

# --- Retrieval --------------------------------------------------------------------------

DEFAULT_K: Final[int] = 30
"""Reference neighbors each cell votes over."""
DEFAULT_TAU: Final[float] = 0.03
"""Softmax temperature over neighbor similarities."""

RETRIEVAL_DEPTH: Final[int] = 200
"""Neighbors retrieved, then scored again pair by pair, before the slice to k."""

L2_EPS: Final[float] = 1e-12
SOFTMAX_EPS: Final[float] = 1e-12
ENTROPY_EPS: Final[float] = 1e-12
POOL_EPS: Final[float] = 1e-6

# --- Class axes -------------------------------------------------------------------------

UNKNOWN_LABEL: Final[str] = "Unknown"
"""Name of the unknown class, last in every class order, which no row is coded with."""

LINEAGE_V1: Final[tuple[str, ...]] = (
    "B_Plasma",
    "Epithelial_Melanocyte",
    "Myeloid",
    "Stromal_Vascular",
    "T_NK",
)
"""Classes of the `lineage_v1` axis, in the order of their codes (`label_order`)."""

COMPARTMENT_V1: Final[tuple[str, ...]] = (
    "Epithelial_Melanocyte",
    "Immune",
    "Stromal_Vascular",
)
"""Classes of the `compartment_v1` axis, in the order of their codes (`label_order`)."""

CLASS_AXES: Final[dict[str, tuple[str, ...]]] = {
    "lineage_v1": LINEAGE_V1,
    "compartment_v1": COMPARTMENT_V1,
}
"""Declared class axes, by name."""

DEFAULT_AXIS: Final[str] = "lineage_v1"
"""Class axis used where none is named."""

CLASS_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "T_NK": (255, 15, 15),
    "B_Plasma": (255, 143, 15),
    "Myeloid": (255, 211, 15),
    "Immune": (255, 123, 15),
    "Stromal_Vascular": (113, 191, 18),
    "Epithelial_Melanocyte": (39, 188, 178),
    "Unknown": (143, 143, 143),
}
"""RGB color of each class in the figures of `alhena.viz`."""

# --- Reference thresholds ---------------------------------------------------------------

MIN_REFERENCE_CELLS_PER_CLASS: Final[int] = 200
"""Reference cells below which a class is reported as thinly supported."""

MIN_SUPPORT_TO_K_RATIO: Final[float] = 10.0
"""Floor of the thinnest class's support over k, below which `EnvelopeWarning` is raised."""
MIN_K_TO_CLASSES_RATIO: Final[float] = 3.0
"""Floor of k over the number of classes, below which `EnvelopeWarning` is raised."""

# --- Reference registry -----------------------------------------------------------------

TISSUE_TO_SHARD: Final[dict[str, str]] = {
    "Breast": "breast",
    "BoneMarrow": "bone_marrow",
    "Cervix": "cervix",
    "Colon": "intestine",
    "Heart": "heart",
    "Kidney": "kidney",
    "Liver": "liver_cell",
    "Lung": "lung",
    "LymphNode": "tonsil",
    "Ovary": "ovary",
    "Pancreas": "pancreas_cell",
    "Prostate": "prostate",
    "Skin": "skin",
    "Tonsil": "tonsil",
    "Uterus": "uterus",
}
"""Shard each tissue is served by, under its canonical organ name; tissues may share one."""

# Once case, spacing and separators are ignored, no alias matches a tissue, a shard or another.
TISSUE_ALIASES: Final[dict[str, str]] = {
    # Adjectives of the organ, but not Cardiac or Cervical, which also name the cardia and the neck.
    "Colonic": "Colon",
    "Cutaneous": "Skin",
    "Hepatic": "Liver",
    "Mammary": "Breast",
    "Ovarian": "Ovary",
    "Pancreatic": "Pancreas",
    "Prostatic": "Prostate",
    "Pulmonary": "Lung",
    "Renal": "Kidney",
    "Tonsillar": "Tonsil",
    "Uterine": "Uterus",
    # The fuller anatomical name of the organ.
    "Cervix uteri": "Cervix",
    "Prostate gland": "Prostate",
    "Uterine cervix": "Cervix",
    # Parts or regions of the organ the atlas behind the shard samples, by the site of each cell.
    "Bowel": "Colon",
    "Colorectal": "Colon",
    "Large intestine": "Colon",
    "Rectum": "Colon",
    # The intestine atlas holds ileum, cecum, appendix, colon and rectum, no duodenum or jejunum.
    "Small intestine": "Colon",
    "Corpus uteri": "Uterus",
    "Endometrium": "Uterus",
    "Myometrium": "Uterus",
}
"""Other names a tissue is accepted under, each mapped to its tissue in `TISSUE_TO_SHARD`."""

SCCONCEPT_MODEL: Final[str] = "corpus40M-model30M"
SCCONCEPT_VERSION: Final[str] = "v2"


def label_order(axis: str) -> tuple[str, ...]:
    """Return the class order of an axis, scorable classes sorted, then the unknown class.

    The order fixes the code of every class and breaks ties in the vote.

    Args:
        axis: A key of `CLASS_AXES`.

    Returns:
        Class names in their canonical order.

    Raises:
        AlhenaValueError: If the axis is not one of the declared axes.
    """
    if axis not in CLASS_AXES:
        raise AlhenaValueError(f"Unknown class axis {axis!r}. Declared axes: {sorted(CLASS_AXES)}.")
    return (*sorted(CLASS_AXES[axis]), UNKNOWN_LABEL)


def _check_class_names(names: Sequence[str], error: type[AlhenaError] = AlhenaValueError) -> None:
    """Refuse a class name that no key of an .h5ad or zarr file can hold."""
    for name in names:
        if "/" in name:
            raise error(
                f"The class name {name!r} holds '/', which no key of an .h5ad or zarr file can "
                "hold, so its results could not be written. Rename the class, for example "
                f"{name.replace('/', '_')!r}."
            )
