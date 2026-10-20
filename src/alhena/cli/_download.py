"""The download command."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from alhena._constants import REFERENCES_REPO, TISSUE_TO_SHARD, WEIGHTS_REPO
from alhena._errors import AlhenaValueError, ReferenceError
from alhena._hub import _read_json_object
from alhena._wording import PROGRAM
from alhena.cli._shared import _CLASS_WIDTH, _field, _file_size, _format_size, _note


def _declared_sizes(registry: Path, shards: Sequence[str]) -> dict[str, int | None]:
    """Return the size the registry declares for each shard, where it declares one."""
    entries = _read_json_object(registry, ReferenceError).get("shards")
    sizes: dict[str, int | None] = dict.fromkeys(shards)
    if not isinstance(entries, dict):
        return sizes
    for shard in shards:
        entry = entries.get(shard)
        if isinstance(entry, dict) and isinstance(entry.get("bytes"), int):
            sizes[shard] = entry["bytes"]
    return sizes


def _download(args: argparse.Namespace) -> int:
    """Fetch the encoder bundle and the named references into the cache."""
    from alhena._shards import shard_for_tissue
    from alhena.hub import BUNDLE_FILES, ensure_bundle, ensure_reference, ensure_registry

    if not args.bundle and not args.tissue:
        raise AlhenaValueError(
            "Name what to download: --bundle for the encoder bundle, --tissue for one or "
            f"more tissues. The tissues are {', '.join(sorted(TISSUE_TO_SHARD))}; "
            f"`{PROGRAM} reference list` prints the other names each is accepted under."
        )
    # Resolved first, so a misspelled tissue costs no download; a shard named twice is fetched once.
    shards = list(dict.fromkeys(shard_for_tissue(name) for name in args.tissue))

    print("requested resources")
    if args.bundle:
        _note(f"{WEIGHTS_REPO}: {', '.join(BUNDLE_FILES)}")
    if shards:
        registry = Path(ensure_registry(cache_dir=args.cache_dir, revision=args.revision))
        sizes = _declared_sizes(registry, shards)
        for shard in shards:
            size = sizes[shard]
            _note(
                f"{REFERENCES_REPO}: {shard:<{_CLASS_WIDTH}}"
                f"{'size not declared' if size is None else _format_size(size)}"
            )

    if args.bundle:
        directory = Path(
            ensure_bundle(WEIGHTS_REPO, cache_dir=args.cache_dir, revision=args.revision)
        )
        weights = [directory / name for name in BUNDLE_FILES]
        total = sum(file.stat().st_size for file in weights if file.is_file())
        _field("bundle", f"{directory} ({_format_size(total)})")
    for shard in shards:
        path = Path(ensure_reference(shard, cache_dir=args.cache_dir, revision=args.revision))
        _field(shard, f"{path} ({_file_size(path)})")
    return 0
