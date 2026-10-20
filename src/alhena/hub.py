"""Locate the released ALHENA artifacts, fetching them into a cache in the model hub's layout.

Downloaded files are checked against the digests they are declared with. The UNI2-h weights are
never redistributed: they are looked for on disk.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

from alhena import _shards
from alhena._constants import (
    ADAPTERS_FILE,
    BACKBONE_REPO,
    BUNDLE_CONFIG_FILE,
    HEADS_FILE,
    REFERENCES_REPO,
    REGISTRY_FILE,
    SCCONCEPT_MODEL,
    SCCONCEPT_VERSION,
    SHARD_SUFFIX,
    WEIGHTS_REPO,
)
from alhena._constants import SHARD_DIRECTORY as SHARD_DIRECTORY
from alhena._errors import (
    AlhenaError,
    BackboneNotFoundError,
    BundleError,
    ReferenceError,
)
from alhena._hub import (
    _BUNDLE_FILES,
    _check_bundle_files,
    _hf_cache_lookup,
    _hub_folder,
    _hub_revision,
    _normalize_digest,
    _read_json_object,
    _repo_id,
)

CACHE_ENV_VAR: str = "ALHENA_CACHE"
"""Environment variable holding the directory downloaded artifacts are cached in."""

BACKBONE_ENV_VAR: str = "ALHENA_BACKBONE"
"""Environment variable holding the path of the UNI2-h weights."""

BUNDLE_FILES: tuple[str, ...] = _BUNDLE_FILES
"""Weight files a bundle directory holds, beside the `config.json` declaring their digests."""

BACKBONE_WEIGHT_FILES: tuple[str, ...] = ("pytorch_model.bin", "model.safetensors")
"""Names UNI2-h publishes its weights under, in the order they are looked for."""

_CACHE_DIRECTORY_NAME = "alhena"
_DIGEST_CHUNK = 1 << 20


# --- Cache ----------------------------------------------------------------------------------


def _cache_location(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    declared = os.environ.get(CACHE_ENV_VAR, "").strip()
    if declared:
        return Path(declared).expanduser()
    shared = os.environ.get("XDG_CACHE_HOME", "").strip()
    if shared:
        return Path(shared).expanduser() / _CACHE_DIRECTORY_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / _CACHE_DIRECTORY_NAME
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local).expanduser() if local else Path.home() / "AppData" / "Local"
        return base / _CACHE_DIRECTORY_NAME / "Cache"
    return Path.home() / ".cache" / _CACHE_DIRECTORY_NAME


def cache_root(cache_dir: str | Path | None = None, create: bool = True) -> Path:
    """Return the directory downloaded artifacts are cached in.

    The first of `cache_dir`, `ALHENA_CACHE` and `XDG_CACHE_HOME/alhena` that is set, else a
    per-user cache directory.

    Args:
        cache_dir: Directory to use, overriding the environment.
        create: Create the directory and its parents when they do not exist.

    Returns:
        The cache directory.

    Raises:
        AlhenaError: If the directory has to be created and cannot be.
    """
    root = _cache_location(cache_dir)
    if create:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise AlhenaError(
                f"The cache directory {root} could not be created. Point {CACHE_ENV_VAR} at a "
                "writable directory, or pass one as cache_dir."
            ) from error
    return root


# --- Digests --------------------------------------------------------------------------------


def file_digest(path: str | Path, chunk_size: int = _DIGEST_CHUNK) -> str:
    """Return the SHA-256 digest of a file, read in chunks.

    Args:
        path: File to read.
        chunk_size: Bytes read per step, so a large shard is never held in memory whole.

    Returns:
        The digest, as lower case hexadecimal.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _check_digest(
    path: Path,
    declared: str | None,
    described: str,
    error: type[AlhenaError],
    folder: Path,
) -> None:
    """Check a cached file against a declared digest, doing nothing when none is declared."""
    if declared is None:
        return
    expected = _normalize_digest(declared, described, error)
    found = file_digest(path)
    if found != expected:
        # The cache links a fetched file to the copy it holds, so the whole folder must go.
        raise error(
            f"{described} has SHA-256 {found} where {expected} is declared: the copy at {path} is "
            f"not the published one. Delete {folder}, which holds the cached files of that "
            "repository, and fetch it again."
        )


# --- Download layer -------------------------------------------------------------------------


def _local_route(error: type[AlhenaError]) -> str:
    """Return how the artifact an error class describes is read from disk without a cache."""
    if issubclass(error, BundleError):
        return (
            "A directory holding the bundle files themselves is not read as a cache but passed "
            "as the bundle, which reaches no network: bundle= in Python, --bundle to encode, "
            "annotate and demo."
        )
    if issubclass(error, ReferenceError):
        return (
            "A shard on disk is read from its path instead, which reaches no network: "
            "reference= or path= in Python, --reference to annotate and demo, --path to "
            "reference describe and check."
        )
    return ""


def _download(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    revision: str | None,
    error: type[AlhenaError],
) -> Path:
    """Fetch one repository file into the cache and return its path, raising `error` on failure."""
    from huggingface_hub import errors, hf_hub_download

    try:
        resolved = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            revision=revision,
        )
    except errors.LocalEntryNotFoundError as failure:
        message = (
            f"{filename} of {repo_id} is not in the cache at {cache_dir} and could not be "
            "downloaded. Connect once to fill it, or name a cache holding the files with "
            f"{CACHE_ENV_VAR}, cache_dir= or --cache-dir: a cache is read in the layout `alhena "
            f"download` fills, which keeps them under {Path(cache_dir) / _hub_folder(repo_id)}."
        )
        raise error(f"{message} {_local_route(error)}".rstrip()) from failure
    except errors.EntryNotFoundError as failure:
        where = "on its default branch" if revision is None else f"at revision {revision}"
        raise error(f"{repo_id} publishes no {filename} {where}.") from failure
    except errors.GatedRepoError as failure:
        raise error(
            f"{repo_id} is gated. Accept its conditions on the model hub and log in with "
            "`hf auth login` before fetching from it."
        ) from failure
    except errors.HfHubHTTPError as failure:
        raise error(f"{filename} of {repo_id} could not be fetched: {failure}") from failure
    return Path(resolved)


def _fetch(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    revision: str | None,
    error: type[AlhenaError],
    declared_digest: str | None = None,
) -> Path:
    """Fetch one file and check it against the digest declared for it, if any."""
    path = _download(repo_id, filename, cache_dir, revision, error)
    folder = cache_dir / _hub_folder(repo_id)
    _check_digest(path, declared_digest, f"{filename} of {repo_id}", error, folder)
    return path


# --- Bundles --------------------------------------------------------------------------------


def _one_directory(paths: list[Path], repo_id: str) -> Path:
    parents = {path.parent for path in paths}
    if len(parents) != 1:
        raise BundleError(
            f"The files of {repo_id} were fetched into {len(parents)} directories, so no single "
            "directory holds the bundle. Clear the cache and fetch it again."
        )
    return parents.pop()


def _declared_digests(config: dict[str, Any], described: str) -> dict[str, str]:
    declared = config.get("sha256")
    if declared is None:
        return {}
    if not isinstance(declared, dict):
        raise BundleError(
            f"{described} declares `sha256` as a value of type {type(declared).__name__} where "
            "an object mapping file name to digest is expected."
        )
    return {str(name): str(digest) for name, digest in declared.items()}


def bundle_fingerprint(directory: str | Path) -> tuple[str, str]:
    """Return the encoder identity and the cell space identity a bundle's weights produce.

    Both are computed from the weights, not read from the configuration, so a stated identity can be
    checked.

    Args:
        directory: Directory holding the bundle files.

    Returns:
        The encoder identity and the cell space identity.

    Raises:
        BundleError: If the directory does not hold a bundle, or its files cannot be read.
    """
    from safetensors import SafetensorError
    from safetensors.torch import load_file

    # Reused, not recomputed: a second hashing could report a matched pair as two cell spaces.
    from alhena._bundle import _declared_property, bundle_identity

    path = Path(directory)
    _check_bundle_files(path)
    config_path = path / BUNDLE_CONFIG_FILE
    config = _read_json_object(config_path, BundleError) if config_path.is_file() else {}
    try:
        head_tensors = load_file(str(path / HEADS_FILE))
    except (OSError, SafetensorError) as error:
        raise BundleError(
            f"{path / HEADS_FILE} could not be read as a safetensors file of head weights."
        ) from error
    return bundle_identity(
        (path / ADAPTERS_FILE).read_bytes(),
        head_tensors,
        _declared_property(path, config, "teacher", "scconcept_model", SCCONCEPT_MODEL),
        _declared_property(path, config, "teacher", "scconcept_version", SCCONCEPT_VERSION),
    )


def _check_declared_identity(directory: Path, config: dict[str, Any]) -> None:
    """Check the identities a bundle states against the ones its weights produce."""
    from alhena._bundle import _check_stated_identity

    if any(isinstance(config.get(name), str) for name in ("encoder_id", "cell_space_id")):
        _check_stated_identity(directory, config, *bundle_fingerprint(directory))


def ensure_bundle(
    bundle: str | Path = WEIGHTS_REPO,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> Path:
    """Return a local directory holding the encoder bundle, fetching it where needed.

    A directory is returned once its contents are checked. A string that names no directory is a
    repository `owner/name`, fetched into the cache and checked against the digests and identities
    its configuration declares.

    Args:
        bundle: Directory holding the bundle files, or the repository publishing them.
        cache_dir: Cache to fetch into. Defaults to the resolved cache root.
        revision: Branch, tag or commit to fetch. Defaults to `HUB_REVISION`, the release this
            version of the package reads, and to the default branch of any other repository.

    Returns:
        A directory holding the adapter and head weights.

    Raises:
        BundleError: If a path names no directory or an incomplete one, if the repository cannot be
            read, or if a file or an identity does not match what the configuration declares.
    """
    directory = Path(bundle)
    if directory.is_dir():
        _check_bundle_files(directory)
        config_path = directory / BUNDLE_CONFIG_FILE
        if config_path.is_file():
            _check_declared_identity(directory, _read_json_object(config_path, BundleError))
        return directory
    if isinstance(bundle, Path):
        raise BundleError(
            f"No bundle directory at {directory}. Pass the directory holding the bundle files, "
            "or the repository publishing them as a string owner/name."
        )

    repo_id = _repo_id(bundle, BundleError)
    revision = _hub_revision(repo_id, revision)
    root = cache_root(cache_dir)
    try:
        config_path = _fetch(repo_id, BUNDLE_CONFIG_FILE, root, revision, BundleError)
    except BundleError as failure:
        if all((root / name).is_file() for name in BUNDLE_FILES):
            raise BundleError(
                f"{failure} The cache directory itself holds {' and '.join(BUNDLE_FILES)}, as a "
                "bundle directory does: pass it as the bundle."
            ) from failure
        raise
    config = _read_json_object(config_path, BundleError)
    digests = _declared_digests(config, f"{BUNDLE_CONFIG_FILE} of {repo_id}")

    paths = [config_path]
    paths.extend(
        _fetch(repo_id, name, root, revision, BundleError, digests.get(name))
        for name in BUNDLE_FILES
    )

    fetched = _one_directory(paths, repo_id)
    _check_bundle_files(fetched)
    _check_declared_identity(fetched, config)
    return fetched


# --- References -----------------------------------------------------------------------------


def ensure_registry(cache_dir: str | Path | None = None, revision: str | None = None) -> Path:
    """Return the path of the reference registry, fetching it when it is not in the cache.

    Args:
        cache_dir: Cache to fetch into. Defaults to the resolved cache root.
        revision: Branch, tag or commit to fetch. Defaults to `HUB_REVISION`, the release this
            version of the package reads.

    Returns:
        The path of the registry file.

    Raises:
        ReferenceError: If the registry cannot be fetched.
    """
    root = cache_root(cache_dir)
    revision = _hub_revision(REFERENCES_REPO, revision)
    return _fetch(REFERENCES_REPO, REGISTRY_FILE, root, revision, ReferenceError)


def _shard_entry(registry: dict[str, Any], shard: str) -> tuple[str, str | None]:
    """Return the file a shard is published as and the digest declared for it, if any."""
    default = f"{SHARD_DIRECTORY}/{shard}{SHARD_SUFFIX}"
    shards = registry.get("shards")
    if shards is None:
        return default, None
    if not isinstance(shards, dict):
        raise ReferenceError(
            f"The reference registry declares `shards` as a value of type {type(shards).__name__} "
            "where an object keyed by shard name is expected."
        )
    entry = shards.get(shard)
    if entry is None:
        raise ReferenceError(
            f"The reference registry publishes no shard named {shard!r}. It holds {sorted(shards)}."
        )
    if isinstance(entry, str):
        return entry, None
    if not isinstance(entry, dict):
        raise ReferenceError(
            f"The reference registry declares shard {shard!r} as a value of type "
            f"{type(entry).__name__} where a file name or an object is expected."
        )
    filename = entry.get("file")
    digest = entry.get("sha256")
    return (
        default if filename is None else str(filename),
        None if digest is None else str(digest),
    )


def ensure_reference(
    tissue: str,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> Path:
    """Return the path of one reference shard, fetching it when it is not in the cache.

    The registry is read first, and only the shard asked for is fetched.

    Args:
        tissue: A tissue name, an alias of one, or a shard name.
        cache_dir: Cache to fetch into. Defaults to the resolved cache root.
        revision: Branch, tag or commit to fetch. Defaults to `HUB_REVISION`, the release this
            version of the package reads.

    Returns:
        The path of the shard file.

    Raises:
        AlhenaValueError: If the name is empty, or matches no tissue, alias or shard.
        ReferenceError: If the registry or the shard cannot be fetched, or the shard does not match
            the digest declared for it.
    """
    shard = _shards.shard_for_tissue(tissue)
    root = cache_root(cache_dir)
    revision = _hub_revision(REFERENCES_REPO, revision)
    registry_path = ensure_registry(cache_dir=root, revision=revision)
    registry = _read_json_object(registry_path, ReferenceError)
    filename, declared = _shard_entry(registry, shard)
    return _fetch(REFERENCES_REPO, filename, root, revision, ReferenceError, declared)


# --- Backbone -------------------------------------------------------------------------------


def _backbone_at(path: Path, source: str) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for filename in BACKBONE_WEIGHT_FILES:
            candidate = path / filename
            if candidate.is_file():
                return candidate
        raise BackboneNotFoundError(
            f"{source}: directory {path} contains none of {', '.join(BACKBONE_WEIGHT_FILES)}."
        )
    raise BackboneNotFoundError(f"{source}: {path} does not exist.")


def _hub_cache_directory() -> Path:
    """Return the model hub cache huggingface-hub reads, resolved once at its first import."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def find_backbone(backbone: str | Path | None = None) -> Path:
    """Return the path of the UNI2-h weight file.

    The weights are looked for at `backbone`, then at `ALHENA_BACKBONE`, either naming a file or a
    directory holding one, then in the model hub cache: `HF_HUB_CACHE`, else `HF_HOME/hub`, with
    `HF_HOME` defaulting to `XDG_CACHE_HOME/huggingface`, then `~/.cache/huggingface`. A named
    location holding no weights is reported, not passed over.

    Args:
        backbone: Path of the weight file, or of a directory holding it.

    Returns:
        The path of the weight file.

    Raises:
        BackboneNotFoundError: If no location holds the weights.
    """
    if backbone is not None:
        return _backbone_at(Path(backbone), "Specified UNI2-h weights")

    declared = os.environ.get(BACKBONE_ENV_VAR, "").strip()
    if declared:
        return _backbone_at(Path(declared), BACKBONE_ENV_VAR)

    # Resolved once, so the directory a failure names is the one searched.
    searched = _hub_cache_directory()
    for filename in BACKBONE_WEIGHT_FILES:
        cached = _hf_cache_lookup(BACKBONE_REPO, filename, cache_dir=searched)
        if cached is not None:
            return cached

    raise BackboneNotFoundError(
        f"The UNI2-h weights were not found. They are gated and not redistributed with ALHENA: "
        f"accept the conditions on {BACKBONE_REPO}, download {BACKBONE_WEIGHT_FILES[0]}, and pass "
        f"its path or set {BACKBONE_ENV_VAR} to it. A copy already in the model hub cache is found "
        f"without either; the cache searched was {searched}. huggingface-hub reads it from "
        "HF_HUB_CACHE, else HF_HOME/hub, with HF_HOME defaulting to XDG_CACHE_HOME/huggingface, "
        "then ~/.cache/huggingface; a copy downloaded before one of these was set is found again "
        "by setting HF_HUB_CACHE to its directory."
    )
