"""The model hub's names and cache layout, the files it serves, and their declared digests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alhena._constants import ADAPTERS_FILE, HEADS_FILE, HUB_REVISION, REFERENCES_REPO, WEIGHTS_REPO
from alhena._errors import AlhenaError, BundleError

# Weight files a bundle directory holds, beside the configuration declaring their digests.
_BUNDLE_FILES: tuple[str, ...] = (ADAPTERS_FILE, HEADS_FILE)

_DIGEST_PREFIX = "sha256:"
_DIGEST_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")


def _normalize_digest(declared: str, described: str, error: type[AlhenaError]) -> str:
    text = declared.strip().lower()
    if text.startswith(_DIGEST_PREFIX):
        text = text[len(_DIGEST_PREFIX) :]
    if len(text) != _DIGEST_LENGTH or not set(text) <= _HEX_DIGITS:
        raise error(
            f"The SHA-256 declared for {described} is {declared!r}, which is not a "
            f"{_DIGEST_LENGTH} digit hexadecimal digest."
        )
    return text


def _hub_folder(repo_id: str) -> str:
    """Return the folder the model hub's cache layout keeps the files of a repository under."""
    return "models--" + repo_id.replace("/", "--")


def _is_repo_id(name: str) -> bool:
    """Report whether a name is a repository `owner/name` as the model hub accepts one."""
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import validate_repo_id

    if name.count("/") != 1:
        return False
    try:
        validate_repo_id(name)
    except HFValidationError:
        return False
    return True


def _repo_id(value: str | Path, error: type[AlhenaError]) -> str:
    """Return a name as a repository owner/name, raising `error` for one that is not."""
    name = str(value)
    if not _is_repo_id(name):
        raise error(
            f"{name!r} is neither a directory on disk nor a repository of the form owner/name."
        )
    return name


def _hub_revision(repo_id: str, revision: str | None) -> str | None:
    """Return the revision named, else the release tag for a published repository."""
    if revision is None and repo_id in (WEIGHTS_REPO, REFERENCES_REPO):
        return HUB_REVISION
    return revision


def _hf_cache_lookup(
    repo_id: str,
    filename: str,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> Path | None:
    """Return the cached path of a repository file, or None, reading the cache alone."""
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        cache_dir=None if cache_dir is None else str(cache_dir),
        revision=revision,
    )
    # A file the cache records as absent comes back as a sentinel object, not a path.
    if isinstance(found, str) and Path(found).is_file():
        return Path(found)
    return None


def _read_json_object(path: Path, error: type[AlhenaError]) -> dict[str, Any]:
    """Read a JSON file holding an object, raising `error` on anything else."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as failure:
        raise error(f"{path} is not readable as JSON.") from failure
    if not isinstance(payload, dict):
        raise error(
            f"{path} must hold an object, and holds a value of type {type(payload).__name__}."
        )
    return payload


def _check_bundle_files(directory: Path) -> None:
    """Refuse a bundle directory lacking one of its weight files."""
    missing = [name for name in _BUNDLE_FILES if not (directory / name).is_file()]
    if missing:
        raise BundleError(
            f"The bundle at {directory} is missing {', '.join(missing)}. A bundle holds "
            f"{', '.join(_BUNDLE_FILES)}."
        )
