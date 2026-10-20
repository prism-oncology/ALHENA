"""What the commands share: printing reports, writing outputs and finding cached files."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from alhena._commands import _one_line
from alhena._constants import (
    REFERENCES_REPO,
    SHARD_DIRECTORY,
    SHARD_SUFFIX,
    TISSUE_TO_SHARD,
    WEIGHTS_REPO,
)
from alhena._geometry import _format_mpp
from alhena._hub import _BUNDLE_FILES, _hf_cache_lookup, _hub_revision
from alhena._layout import CELLS_SUFFIX
from alhena._truth import _LABEL_TABLE_SUFFIXES
from alhena._wording import PROGRAM, _as_command_line

if TYPE_CHECKING:
    import pandas as pd

    from alhena.io import TileSource
    from alhena.types import AnnotationResult, ResampleInfo, TileEmbeddings

EMBEDDING_SUFFIXES: tuple[str, ...] = (".npy", ".h5", ".hdf5", CELLS_SUFFIX)
"""Formats tile embeddings are written to, from an array alone to a table with provenance."""

CELLS_SUFFIXES: tuple[str, ...] = (CELLS_SUFFIX,)
"""Format cells are written to by `encode`, and read from by `annotate`."""

CALL_SUFFIXES: tuple[str, ...] = (*_LABEL_TABLE_SUFFIXES, *CELLS_SUFFIXES)
"""Formats `annotate` writes to: the annotations alone, or with their cells and provenance."""

EMBEDDINGS_DATASET: str = "embeddings"
"""Dataset the embeddings are written under, in an HDF5 output."""

GEOMETRY_SUFFIX: str = ".geometry.json"
"""Suffix of the geometry file beside a `.npy` output; an HDF5 output keeps it on its root."""

_NPY_SUFFIX: str = ".npy"
_SIZE_UNITS: tuple[str, ...] = ("B", "kB", "MB", "GB", "TB")
_LABEL_WIDTH: int = 20
_CLASS_WIDTH: int = 24

# The command line's wording of UNDECLARED_RESOLUTION_WARNING, opening alike so the two collapse.
_UNDECLARED_MPP: str = (
    "Tile resolution was not declared, so tiles are resized to the model input whatever "
    "their physical scale, and may not cover the field of view the encoder expects. "
    "Pass --mpp with the tiles' microns per pixel."
)


def _field(name: str, value: object) -> None:
    """Print one labeled line of a report."""
    print(f"{name:<{_LABEL_WIDTH}}{value}")


def _note(text: str) -> None:
    """Print an indented line qualifying the field above it."""
    print(f"  {text}")


def _failure(text: str, *notes: str) -> None:
    """Print why a command failed on standard error, apart from the report on standard output."""
    print(f"{PROGRAM}: {text}", file=sys.stderr)
    for note in notes:
        print(f"  {note}", file=sys.stderr)


def _no_token_note(count: int) -> str:
    """Say how many nuclei reached no token, and so have no row in the table."""
    if count == 1:
        return "1 nucleus did not overlap any image patch token and was excluded from the output."
    return (
        f"{count} nuclei did not overlap any image patch token and were excluded from the output."
    )


def _reported(message: object) -> str:
    """Return a message of the library as a report prints it: on one line, as options name it."""
    return _as_command_line(_one_line(message))


def _format_size(n_bytes: float) -> str:
    """Return a byte count in the largest unit that leaves it above one."""
    size = float(n_bytes)
    for unit in _SIZE_UNITS[:-1]:
        if size < 1000.0:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000.0
    return f"{size:.1f} {_SIZE_UNITS[-1]}"


def _file_size(path: Path) -> str:
    """Return the size of a file on disk, or a plain note where it cannot be read."""
    try:
        return _format_size(path.stat().st_size)
    except OSError:
        return "size unavailable"


def _given(args: argparse.Namespace, options: Sequence[tuple[str, str]]) -> list[str]:
    """Return the options of a list that the command line set, by the name it spells them."""
    return [
        flag
        for flag, attribute in options
        if getattr(args, attribute, None) is not None and getattr(args, attribute) is not False
    ]


def _progress_of(args: argparse.Namespace) -> bool | None:
    """Return False under --quiet, else None, so `ALHENA_DISABLE_PROGRESS_BARS` still applies."""
    return False if getattr(args, "quiet", False) else None


@contextmanager
def _reported_warnings(collapse: Sequence[str] = ()) -> Iterator[None]:
    """Print a block's warnings as report lines: once each, and once per opening in `collapse`."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            printed: set[str] = set()
            for entry in caught:
                # A ResourceWarning says nothing about the numbers it would be printed beside.
                if issubclass(entry.category, ResourceWarning):
                    continue
                text = _one_line(entry.message)
                key = next((opening for opening in collapse if text.startswith(opening)), text)
                if key not in printed:
                    printed.add(key)
                    raw = str(entry.message)
                    if "\n\n" not in raw:
                        _note(f"warning: {_as_command_line(text)}")
                    else:
                        paragraphs = raw.split("\n\n")
                        for index, paragraph in enumerate(paragraphs):
                            if index:
                                print("  ")
                            initial = "  warning: " if index == 0 else "           "
                            print(
                                textwrap.fill(
                                    _as_command_line(_one_line(paragraph)),
                                    width=96,
                                    initial_indent=initial,
                                    subsequent_indent="           ",
                                    break_long_words=False,
                                    break_on_hyphens=False,
                                )
                            )


def geometry_path(out: Path) -> Path:
    """Return where the geometry of a `.npy` output is written.

    Args:
        out: The file the embeddings were written to.

    Returns:
        The sidecar beside it, named for the same run.
    """
    return out.with_suffix(GEOMETRY_SUFFIX)


def _geometry_record(
    info: ResampleInfo, encoder_id: str, embedding_shape: tuple[int, int]
) -> dict[str, Any]:
    """Return the geometry record a file of tile embeddings is written with, in written order."""
    record: dict[str, Any] = {"encoder_id": encoder_id}
    record.update({item.name: getattr(info, item.name) for item in fields(info)})
    record["embedding_shape"] = [int(embedding_shape[0]), int(embedding_shape[1])]
    return record


def _h5_attribute(value: Any) -> Any:
    """Return one record value as HDF5 holds it, which has no null of its own."""
    if value is None:
        return float("nan")
    return list(value) if isinstance(value, tuple) else value


def _write_embeddings(tiles: TileEmbeddings, out: Path, geometry: dict[str, Any]) -> None:
    """Write tile embeddings in the format the file name asks for, with the geometry of the run."""
    from alhena.io import H5_BARCODE_DATASET

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == CELLS_SUFFIX:
        tiles.write_h5ad(out)
        return
    if out.suffix.lower() == _NPY_SUFFIX:
        np.save(out, tiles.embeddings)
        geometry_path(out).write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")
        return
    import h5py

    with h5py.File(out, "w") as handle:
        handle.create_dataset(EMBEDDINGS_DATASET, data=tiles.embeddings)
        handle.create_dataset(
            H5_BARCODE_DATASET,
            data=np.asarray(list(tiles.tile_ids), dtype=object),
            dtype=h5py.string_dtype(),
        )
        for name, value in geometry.items():
            handle.attrs[name] = _h5_attribute(value)


def _write_table(frame: pd.DataFrame, out: Path) -> None:
    """Write a table of annotations."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        frame.to_parquet(out, index=False)
    else:
        frame.to_csv(out, index=False)


def _write_calls(result: AnnotationResult, out: Path) -> None:
    """Write annotations as a table alone, or as `.h5ad` together with the cells behind them."""
    if out.suffix.lower() in CELLS_SUFFIXES:
        out.parent.mkdir(parents=True, exist_ok=True)
        result.write_h5ad(out)
        return
    _write_table(result.to_frame(), out)


def _cached_bundle(root: Path, repo: str = WEIGHTS_REPO) -> Path | None:
    """Return the cache directory the loaders read a repository's bundle from, or None."""
    revision = _hub_revision(repo, None)
    try:
        paths = [_hf_cache_lookup(repo, name, root, revision) for name in _BUNDLE_FILES]
    except OSError:
        return None
    folders = {path.parent for path in paths if path is not None}
    if None in paths or len(folders) != 1:
        return None
    return folders.pop()


def _cached_shards(root: Path) -> list[str]:
    """Return the published shards the loaders would find in the cache, and no others."""
    revision = _hub_revision(REFERENCES_REPO, None)
    found = []
    for name in sorted(set(TISSUE_TO_SHARD.values())):
        filename = f"{name}{SHARD_SUFFIX}"
        if any((folder / filename).is_file() for folder in (root / SHARD_DIRECTORY, root)):
            found.append(name)
            continue
        with suppress(OSError):
            cached = _hf_cache_lookup(
                REFERENCES_REPO, f"{SHARD_DIRECTORY}/{filename}", root, revision
            )
            if cached is not None:
                found.append(name)
    return found


def _print_resolution(source: TileSource) -> None:
    """Report the resolution of the tiles, and below it where it was read from."""
    from alhena.io._sources import _DECLARED_MPP_SOURCE

    if source.mpp is None:
        _field("resolution", "undeclared")
        return
    _field("resolution", f"{_format_mpp(source.mpp)} micron per pixel")
    if source.mpp_source == _DECLARED_MPP_SOURCE:
        _note("declared with --mpp")
    elif source.mpp_source is not None:
        _note(f"read from {source.mpp_source}")
