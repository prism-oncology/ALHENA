"""PanNuke: its tiles, classes and tissues, and how they map onto ALHENA's."""

from __future__ import annotations

import json
import warnings
from typing import Any, Final

import cv2
import numpy as np

from alhena._constants import INPUT_SIZE, REFERENCE_MPP
from alhena._errors import PACKAGE_PREFIXES, AlhenaValueError, AlhenaWarning
from alhena._geometry import _target_scale


class ClassCodeWarning(AlhenaWarning):
    """A fold declares no class label names, so its integer codes are read positionally."""


PANNUKE_REPO: Final[str] = "RationAI/PanNuke"
"""Hugging Face dataset repository the example reads."""

PANNUKE_REVISION: Final[str] = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
"""Commit of the dataset repository read by default, so every run reads the same fields."""

PANNUKE_FOLD_FILES: Final[dict[int, str]] = {
    1: "data/fold1-00000-of-00001.parquet",
    2: "data/fold2-00000-of-00001.parquet",
    3: "data/fold3-00000-of-00001.parquet",
}
"""The three folds of the release, each one parquet file."""

PANNUKE_LICENSE: Final[str] = "CC BY-NC-SA 4.0"
"""Terms the dataset is distributed under."""

PANNUKE_MPP: Final[float] = 0.25
"""Microns per pixel the release declares: nominal, part of the fields enlarged from 20x."""

PANNUKE_TILE_PX: Final[int] = 256
"""Side of a PanNuke visual field, in pixels."""

PANNUKE_CATEGORIES: Final[tuple[str, ...]] = (
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
)
"""Nucleus categories, in the index order the `categories` column encodes."""

PANNUKE_TISSUES: Final[tuple[str, ...]] = (
    "Adrenal Gland",
    "Bile Duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "Head & Neck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
)
"""Tissues, in the index order the `tissue` column encodes, in the release spelling."""

IMAGE_COLUMN: Final[str] = "image"
"""Column of the parquet holding the visual field."""

INSTANCES_COLUMN: Final[str] = "instances"
"""Column holding one binary mask per annotated nucleus."""

CATEGORIES_COLUMN: Final[str] = "categories"
"""Column holding one category code per mask, aligned with `instances`."""

TISSUE_COLUMN: Final[str] = "tissue"
"""Column holding the tissue code of the field, read alone over the whole file."""

_TILE_COLUMNS: Final[tuple[str, ...]] = (IMAGE_COLUMN, INSTANCES_COLUMN, CATEGORIES_COLUMN)

HUGGINGFACE_METADATA_KEY: Final[bytes] = b"huggingface"
"""Key of the parquet schema metadata a Hugging Face dataset writes its features under."""

DEMO_AXIS: Final[str] = "compartment_v1"
"""Class axis the PanNuke categories are comparable on."""

# Dead is absent, so nuclei a caller forgot to drop are refused, not given a wrong class.
PANNUKE_TO_COMPARTMENT: Final[dict[str, str]] = {
    "Neoplastic": "Epithelial_Melanocyte",  # carcinoma and melanoma cells
    "Epithelial": "Epithelial_Melanocyte",
    "Inflammatory": "Immune",
    "Connective": "Stromal_Vascular",
}
"""PanNuke category to the class of `compartment_v1` it is compared with."""

DEAD_CATEGORY: Final[str] = "Dead"
"""Category no class of an axis matches, so its nuclei are left out."""

UNRESOLVED_BY_INFLAMMATORY: Final[dict[str, tuple[str, ...]]] = {
    "lineage_v1": ("T_NK", "B_Plasma", "Myeloid"),
}
"""Classes `Inflammatory` lumps together, by axis; no comparison is offered on an axis here."""

PANNUKE_TO_REFERENCE_TISSUE: Final[dict[str, str]] = {
    "Breast": "Breast",
    "Cervix": "Cervix",
    "Colon": "Colon",
    "Kidney": "Kidney",
    "Liver": "Liver",
    "Lung": "Lung",
    "Ovarian": "Ovary",
    "Pancreatic": "Pancreas",
    "Prostate": "Prostate",
    "Skin": "Skin",
    "Uterus": "Uterus",
}
"""PanNuke tissue to its reference tissue; one absent here has no reference and no proxy."""

POSITIONAL_CODE_NOTE: Final[str] = (
    "The fold declares no class label names, so its tissue and category codes are read "
    "positionally, in the order this example holds them. A file whose codes were written in "
    "another order names another tissue and another category for the same code, and nothing "
    "here can tell."
)
"""Warning a fold declaring no class label names raises: its codes are read by position."""


def canvas_geometry(field_px: int = PANNUKE_TILE_PX) -> tuple[int, float]:
    """Return what one visual field covers of the canvas the encoder reads.

    The field is resampled to the distillation resolution and centered on the canvas; a field
    smaller than the canvas leaves white pad around it.

    Args:
        field_px: Side of one visual field, in pixels.

    Returns:
        The side the field occupies on the canvas, and the share of the canvas area it covers.
    """
    # The factor preprocessing resizes by, so the canvas reported is the one the encoder reads.
    scale = _target_scale(PANNUKE_MPP, REFERENCE_MPP)
    side = min(round(field_px * scale), INPUT_SIZE)
    return side, (side * side) / float(INPUT_SIZE * INPUT_SIZE)


def _fold_name(name: str) -> str:
    """Reduce a tissue or category name to the letters and digits that identify it."""
    return "".join(character for character in str(name).lower() if character.isalnum())


def pannuke_tissue(name: str) -> str:
    """Return the release spelling of a PanNuke tissue.

    Matching ignores case, spacing and punctuation: `headneck` names `Head & Neck`.

    Args:
        name: Tissue as written by the caller.

    Returns:
        The tissue in the spelling the release uses.

    Raises:
        AlhenaValueError: If the name is not a tissue the release holds.
    """
    wanted = _fold_name(name)
    for tissue in PANNUKE_TISSUES:
        if _fold_name(tissue) == wanted:
            return tissue
    raise AlhenaValueError(
        f"{name!r} names no PanNuke tissue. The release holds {list(PANNUKE_TISSUES)}."
    )


def reference_tissue(tissue: str) -> str:
    """Return the tissue whose published reference covers a PanNuke tissue.

    Args:
        tissue: A PanNuke tissue, in any spelling `pannuke_tissue` accepts.

    Returns:
        The tissue name a prebuilt reference is published under.

    Raises:
        AlhenaValueError: If the tissue is not a PanNuke tissue, or if no reference is published
            for it.
    """
    name = pannuke_tissue(tissue)
    target = PANNUKE_TO_REFERENCE_TISSUE.get(name)
    if target is None:
        raise AlhenaValueError(
            f"No reference is published for the PanNuke tissue {name!r}, and the reference of a "
            "neighboring organ does not stand in for it. References cover "
            f"{list(PANNUKE_TO_REFERENCE_TISSUE)}."
        )
    return target


def compartment_for_category(category: str) -> str:
    """Return the class of `compartment_v1` a PanNuke nucleus category is compared with.

    Args:
        category: A PanNuke nucleus category, in any spelling `PANNUKE_CATEGORIES` matches.

    Returns:
        The class of `compartment_v1` the category is compared with.

    Raises:
        AlhenaValueError: If the category is `Dead`, which no class matches, or if it is not a
            PanNuke category at all.
    """
    wanted = _fold_name(category)
    for name in PANNUKE_CATEGORIES:
        if _fold_name(name) == wanted:
            if name == DEAD_CATEGORY:
                raise AlhenaValueError(
                    f"{DEAD_CATEGORY} nuclei carry no class of {DEMO_AXIS}, so they are left out "
                    "of the comparison. Drop them before mapping the categories."
                )
            return PANNUKE_TO_COMPARTMENT[name]
    raise AlhenaValueError(
        f"{category!r} names no PanNuke nucleus category. The release holds "
        f"{list(PANNUKE_CATEGORIES)}."
    )


def _category_name(code: int) -> str:
    """Return the category a code of the `categories` column names."""
    if not 0 <= code < len(PANNUKE_CATEGORIES):
        raise AlhenaValueError(
            f"Category code {code} is outside the {len(PANNUKE_CATEGORIES)} codes PanNuke "
            f"encodes, which are {list(PANNUKE_CATEGORIES)}."
        )
    return PANNUKE_CATEGORIES[code]


def _class_label_names(feature: Any) -> tuple[str, ...] | None:
    """Return the names a class label feature declares, unwrapping any sequence around it."""
    if isinstance(feature, list):
        return _class_label_names(feature[0]) if len(feature) == 1 else None
    if not isinstance(feature, dict):
        return None
    if feature.get("_type") == "ClassLabel" and isinstance(feature.get("names"), list):
        return tuple(str(name) for name in feature["names"])
    return _class_label_names(feature.get("feature"))


def declared_class_labels(
    parquet_file: Any,
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """Return the tissue and category orders a fold declares in its schema metadata.

    Args:
        parquet_file: An open parquet file of the release.

    Returns:
        The names the tissue column and the categories column index, each None where the file
        declares none.
    """
    metadata = parquet_file.schema_arrow.metadata or {}
    raw = metadata.get(HUGGINGFACE_METADATA_KEY)
    if raw is None:
        return None, None
    try:
        block = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, None
    features: dict[str, Any] | None = None
    for holder in (block.get("info") if isinstance(block, dict) else None, block):
        if isinstance(holder, dict) and isinstance(holder.get("features"), dict):
            features = holder["features"]
            break
    if features is None:
        return None, None
    return (
        _class_label_names(features.get(TISSUE_COLUMN)),
        _class_label_names(features.get(CATEGORIES_COLUMN)),
    )


def _check_class_codes(parquet_file: Any, described: str) -> None:
    """Check the class orders a fold declares against those read here; warn where none is."""
    tissues, categories = declared_class_labels(parquet_file)
    for declared, read, column in (
        (tissues, PANNUKE_TISSUES, TISSUE_COLUMN),
        (categories, PANNUKE_CATEGORIES, CATEGORIES_COLUMN),
    ):
        if declared is not None and declared != tuple(read):
            raise AlhenaValueError(
                f"The {column} codes of {described} index {list(declared)}, and this example "
                f"reads them as {list(read)}. The same code names a different class in the two "
                "orders, so the fold cannot be read here."
            )
    if tissues is None or categories is None:
        warnings.warn(POSITIONAL_CODE_NOTE, ClassCodeWarning, skip_file_prefixes=PACKAGE_PREFIXES)


def _payload(entry: Any, described: str) -> bytes:
    """Return the encoded bytes of an image struct, which carries them beside a path."""
    if isinstance(entry, (bytes, bytearray, memoryview)):
        return bytes(entry)
    if isinstance(entry, dict) and isinstance(entry.get("bytes"), (bytes, bytearray)):
        return bytes(entry["bytes"])
    raise AlhenaValueError(
        f"{described} carries no encoded bytes. A Hugging Face image column holds a struct with "
        "a 'bytes' field, and the file read here does not."
    )


def decode_image(payload: bytes) -> np.ndarray:
    """Decode an encoded tile into an (H, W, 3) uint8 RGB array.

    Args:
        payload: The encoded bytes of one visual field.

    Returns:
        The tile in RGB channel order, which is the order the encoder reads.

    Raises:
        AlhenaValueError: If the bytes do not decode to an image.
    """
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AlhenaValueError(
            f"{len(payload)} bytes of the image column did not decode to an image."
        )
    # The decoder returns BGR, which would silently swap two of the ImageNet channel statistics.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_mask(payload: bytes) -> np.ndarray:
    """Decode an encoded nucleus mask into an (H, W) boolean array.

    Args:
        payload: The encoded bytes of one instance mask.

    Returns:
        True where the nucleus covers the pixel.

    Raises:
        AlhenaValueError: If the bytes do not decode to an image.
    """
    mask = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise AlhenaValueError(
            f"{len(payload)} bytes of the instance column did not decode to a mask."
        )
    return mask > 0
