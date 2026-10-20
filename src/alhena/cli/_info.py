"""The info command."""

from __future__ import annotations

import argparse
from pathlib import Path

from alhena._constants import ADAPTERS_FILE, HEADS_FILE, REFERENCES_REPO, WEIGHTS_REPO
from alhena._errors import AlhenaError
from alhena._hub import _is_repo_id
from alhena._version import __version__
from alhena._wording import PROGRAM
from alhena.cli._shared import _cached_bundle, _cached_shards, _field, _note, _reported


def _info(args: argparse.Namespace) -> int:
    """Report the version, the cache, the UNI2-h weights, the bundle and the references."""
    from alhena.hub import bundle_fingerprint, cache_root, find_backbone

    print(f"{PROGRAM} {__version__}")
    root = Path(cache_root(args.cache_dir, create=False))
    _field("cache", root if root.is_dir() else f"{root} (not created yet)")

    try:
        _field("UNI2-h weights", find_backbone(args.backbone))
    except (AlhenaError, ImportError) as error:
        _field("UNI2-h weights", "not found")
        _note(_reported(error))

    directory, missing = _bundle_to_report(args.bundle, root)
    if directory is None:
        _field("bundle", "not downloaded" if args.bundle is None else "not found")
        _note(missing)
    else:
        _field("bundle", directory)
        try:
            encoder_id, cell_space_id = bundle_fingerprint(directory)
            _field("  encoder id", encoder_id)
            _field("  cell space id", cell_space_id)
        except (AlhenaError, ImportError) as error:
            _note(f"identity unavailable: {_reported(error)}")

    shards = _cached_shards(root)
    if shards:
        _field("references", ", ".join(shards))
    else:
        _field("references", "none cached")
        _note(f"Fetch one from {REFERENCES_REPO} with `{PROGRAM} download --tissue Breast`.")
    return 0


def _bundle_to_report(named: str | None, root: Path) -> tuple[Path | None, str]:
    """Return the bundle directory `info` reports, or None and how to obtain one."""
    if named is None:
        found = _cached_bundle(root)
        return found, (
            f"Download the encoder bundle from {WEIGHTS_REPO} with "
            f"`{PROGRAM} download --bundle`, or pass --bundle with a directory "
            f"containing {ADAPTERS_FILE} and {HEADS_FILE}."
        )
    path = Path(named).expanduser()
    if path.is_dir():
        return path, ""
    if _is_repo_id(str(named)) and not path.exists():
        found = _cached_bundle(root, str(named))
        return found, (
            f"{named} is not cached at {root}. `{PROGRAM} encode` and `{PROGRAM} annotate` "
            "download it on their first run. Alternatively, pass --bundle with a directory "
            f"containing {ADAPTERS_FILE} and {HEADS_FILE}."
        )
    return None, (
        f"Bundle directory not found: {path}. Pass --bundle with a directory containing "
        f"{ADAPTERS_FILE} and {HEADS_FILE}, or a repository name in owner/name format."
    )
