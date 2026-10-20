"""What a bundle declares about its weights, and the identity computed from them."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import torch

from alhena._constants import SCCONCEPT_MODEL, SCCONCEPT_VERSION
from alhena._errors import BundleError

_CELL_SPACE_TAG = "alhena-cell-space-v1"
_IDENTITY_DIGITS = 16

# What each object of a bundle configuration holds, as the refusal of a malformed one names it.
_BUNDLE_SECTIONS: dict[str, str] = {
    "adapters": "the rank and the scaling of the low-rank adapters",
    "dims": "the widths of the embedding and of the cell space",
    "teacher": "the single-cell model the cell space is aligned to",
    "backbone": "the architecture and the repository of the backbone",
}

_Declared = TypeVar("_Declared", int, float, str)


def _tensor_digest(tensor: torch.Tensor | None) -> str:
    """Return the digest of a tensor's float32 bytes, or an empty field when it is absent."""
    if tensor is None:
        return ""
    values = tensor.detach().to(dtype=torch.float32).cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def bundle_identity(
    adapter_bytes: bytes,
    head_tensors: Mapping[str, torch.Tensor],
    scconcept_model: str = SCCONCEPT_MODEL,
    scconcept_version: str = SCCONCEPT_VERSION,
) -> tuple[str, str]:
    """Return the encoder identity and the cell space identity of a bundle.

    Each is the first sixteen hexadecimal digits of a SHA-256 digest: of the adapter file, and of
    `alhena-cell-space-v1`, the adapter digest, the digests of the cell and reference head weights,
    the single-cell model name and its version, joined by `|`.

    Args:
        adapter_bytes: Contents of the adapter file.
        head_tensors: Head weights, keyed as in the bundle.
        scconcept_model: Single-cell model the reference embeddings come from.
        scconcept_version: Version of that model.

    Returns:
        The encoder identity and the cell space identity.
    """
    adapter_digest = hashlib.sha256(adapter_bytes).hexdigest()
    fields = (
        _CELL_SPACE_TAG,
        adapter_digest,
        _tensor_digest(head_tensors.get("img_tok_proj.weight")),
        _tensor_digest(head_tensors.get("st_node_proj.weight")),
        scconcept_model,
        scconcept_version,
    )
    cell_space_digest = hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()
    return adapter_digest[:_IDENTITY_DIGITS], cell_space_digest[:_IDENTITY_DIGITS]


def _declared_property(
    directory: Path,
    config: Mapping[str, Any],
    section: str,
    name: str,
    default: _Declared,
) -> _Declared:
    """Read a quantity a bundle states about itself, falling back to the package's own value."""
    declared = config.get(section)
    if declared is None:
        return default
    if not isinstance(declared, dict):
        raise BundleError(
            f"The bundle at {directory} states `{section}` as a value of type "
            f"{type(declared).__name__} where an object naming {_BUNDLE_SECTIONS[section]} is "
            "expected. Load a bundle whose configuration follows the published layout."
        )
    if name not in declared:
        return default
    value = declared[name]
    if isinstance(default, str):
        if not isinstance(value, str):
            raise BundleError(
                f"The bundle at {directory} states {section}.{name} as {value!r} where a string "
                "is expected. Load a bundle whose configuration follows the published layout."
            )
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleError(
            f"The bundle at {directory} states {section}.{name} as {value!r} where a number is "
            "expected. Load a bundle whose configuration follows the published layout."
        )
    if isinstance(default, int) and not float(value).is_integer():
        raise BundleError(
            f"The bundle at {directory} states {section}.{name} as {value!r} where an integer "
            "is expected. Load a bundle whose configuration follows the published layout."
        )
    return type(default)(value)


def _check_stated_identity(
    directory: Path, config: Mapping[str, Any], encoder_id: str, cell_space_id: str
) -> None:
    """Refuse a bundle whose configuration states an identity its weights do not produce."""
    produced = {"encoder_id": encoder_id, "cell_space_id": cell_space_id}
    for name, value in produced.items():
        stated = config.get(name)
        if isinstance(stated, str) and stated != value:
            raise BundleError(
                f"The bundle at {directory} states {name} {stated!r} where its weights produce "
                f"{value!r}. The configuration and the weights come from different runs."
            )
