"""Reference shards: the tissues the prebuilt ones serve, and the fields a shard stamps."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from safetensors import safe_open

from alhena._constants import CLASS_AXES, TISSUE_ALIASES, TISSUE_TO_SHARD
from alhena._errors import AlhenaValueError, ReferenceError

# Recorded for a provenance field the shard does not stamp.
_UNDECLARED: Final[str] = "undeclared"


def available_tissues() -> tuple[str, ...]:
    """Return the tissue names a reference is published for.

    The canonical names, one per organ; the aliases `TISSUE_ALIASES` declares are not listed.
    """
    return tuple(sorted(TISSUE_TO_SHARD))


def available_shards() -> tuple[str, ...]:
    """Return the published reference shards. Several tissues may share one shard."""
    return tuple(sorted(set(TISSUE_TO_SHARD.values())))


def shard_for_tissue(tissue: str) -> str:
    """Return the shard published for a tissue.

    A tissue is named by its canonical name, an alias `TISSUE_ALIASES` declares, or its shard,
    ignoring case, spacing and separators: `LymphNode`, `lymph_node` and `tonsil` name one shard.

    Args:
        tissue: Tissue the reference is wanted for, an alias of it, or its shard.

    Returns:
        Name of the shard, without a suffix.

    Raises:
        AlhenaValueError: If the name is empty, or no reference is published under it.
    """
    if not isinstance(tissue, str) or not tissue.strip():
        raise AlhenaValueError(
            "A reference is named by its tissue, an alias of it or its shard, as a non-empty "
            f"string, got {tissue!r}."
        )
    shard = _SHARD_BY_NAME.get(_fold(tissue))
    if shard is None:
        raise AlhenaValueError(
            f"No reference is published for {tissue!r}. The tissues are "
            f"{', '.join(available_tissues())}. Each is also accepted under the name of its "
            "shard, and under the aliases TISSUE_ALIASES declares, such as Pancreatic for "
            "Pancreas; `alhena reference list` prints them all."
        )
    return shard


def _refuse_file_as_tissue(tissue: Any, instead: str) -> None:
    """Refuse the name of a file given as a tissue, saying where the file goes instead."""
    text = "" if tissue is None else str(tissue)
    if "/" in text or "\\" in text or Path(text).suffix:
        raise AlhenaValueError(
            f"{text!r} names a file, and a tissue is named by its name. Give the file as {instead}."
        )


def _fold(name: str) -> str:
    """Reduce a tissue, alias or shard name to the letters and digits that identify it."""
    return "".join(character for character in str(name).lower() if character.isalnum())


# Every folded name a published shard answers to: its own, its tissues' and their aliases'.
_SHARD_BY_NAME: Final[dict[str, str]] = {
    **{_fold(shard): shard for shard in TISSUE_TO_SHARD.values()},
    **{_fold(tissue): shard for tissue, shard in TISSUE_TO_SHARD.items()},
    **{_fold(alias): TISSUE_TO_SHARD[tissue] for alias, tissue in TISSUE_ALIASES.items()},
}


def _require_declared_axis(axis: str) -> None:
    """Refuse an axis no prebuilt reference is labeled on, pointing at a shard built on it."""
    if axis not in CLASS_AXES:
        raise AlhenaValueError(
            f"Unknown class axis {axis!r}. The prebuilt references are labeled on "
            f"{sorted(CLASS_AXES)}. A reference built on a custom axis is read from its "
            "shard: give the path of the shard instead of a tissue."
        )


EMBEDDINGS_TENSOR: Final[str] = "embeddings"
"""Name of the embedding tensor in a shard; every other tensor codes the classes of an axis."""


def shard_metadata(path: str | Path) -> dict[str, str]:
    """Read the metadata of a shard without reading its tensors.

    Args:
        path: Path of the shard.

    Returns:
        The metadata fields, all of them strings.

    Raises:
        ReferenceError: If the file is missing or is not a shard.
    """
    file = Path(path)
    if not file.is_file():
        raise ReferenceError(f"No reference shard at {file}.")
    try:
        with safe_open(str(file), framework="numpy") as handle:
            metadata = handle.metadata()
    except Exception as error:
        raise ReferenceError(f"{file} is not a reference shard: {error}") from error
    return dict(metadata or {})


def _stamped_json(metadata: Mapping[str, str], key: str, path: Path) -> dict[str, Any]:
    """Read a JSON field of a shard's metadata."""
    raw = metadata.get(key)
    if not raw:
        raise ReferenceError(
            f"{path} stamps no {key}, which a shard on a custom axis carries. Build the shard with "
            "alhena.refbuild.build_reference, which stamps it."
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ReferenceError(f"The {key} stamped in {path} is not readable as JSON.") from error
    if not isinstance(parsed, dict):
        raise ReferenceError(f"The {key} stamped in {path} must be keyed by class axis.")
    return parsed


def _stamped(metadata: dict[str, str], key: str) -> str:
    """Return a field the shard stamps, or `undeclared` where it stamps none."""
    value = str(metadata.get(key, "")).strip()
    return value if value else _UNDECLARED


def _stamped_int(metadata: dict[str, str], key: str, path: Path) -> int | None:
    """Return a whole number the shard stamps, or None where it stamps none."""
    raw = metadata.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(str(raw))
    except ValueError as error:
        raise ReferenceError(
            f"{path} stamps {key}={raw!r}, which is not a whole number."
        ) from error
