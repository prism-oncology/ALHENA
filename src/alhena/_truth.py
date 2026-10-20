"""Known labels from a table, a Series or a mapping, put in the order of the annotations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alhena._errors import AlhenaValueError

TRUTH_COLUMNS: tuple[str, ...] = ("cell_id", "label")
"""Columns a table of known labels must carry."""

# Formats a table of known labels is read from, and the command line writes its annotations to.
_LABEL_TABLE_SUFFIXES: tuple[str, ...] = (".parquet", ".csv")

# What to do where only some annotated cells carry a known label, in Python.
_SELECT_ADVICE: str = (
    "Keep the labeled cells with the select() of the annotations before aligning, or give every "
    "annotated cell a known label."
)


def _read_known_table(path: Path) -> pd.DataFrame:
    """Read a table of known labels, one row per cell, checking its columns and its cells."""
    if not path.is_file():
        raise AlhenaValueError(f"No known labels at {path}.")
    suffix = path.suffix.lower()
    if suffix not in _LABEL_TABLE_SUFFIXES:
        raise AlhenaValueError(
            f"A table of known labels is written as {' or '.join(_LABEL_TABLE_SUFFIXES)}, and "
            f"{path.name} names {suffix if suffix else 'no suffix'}."
        )
    frame = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
    _check_known_table(frame, path.name)
    return frame


def _check_known_table(frame: pd.DataFrame, named: str) -> None:
    """Refuse a table of known labels lacking a column, or naming a cell twice."""
    missing = [name for name in TRUTH_COLUMNS if name not in frame.columns]
    if missing:
        raise AlhenaValueError(
            f"{named} carries no {', '.join(missing)} column. A table of known labels holds "
            f"{', '.join(TRUTH_COLUMNS)}."
        )
    if bool(frame["cell_id"].duplicated().any()):
        raise AlhenaValueError(
            f"{named} names a cell twice. An identifier names one cell across all the tiles read "
            "together, not one cell within a tile."
        )


def _known_labels(known: Any) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the identifiers and labels of known labels in any form read, and how to name them."""
    if isinstance(known, str | Path):
        path = Path(known)
        frame = _read_known_table(path)
        return frame["cell_id"].to_numpy(), frame["label"].to_numpy(), path.name
    if isinstance(known, pd.DataFrame):
        named = "the table of known labels"
        _check_known_table(known, named)
        return known["cell_id"].to_numpy(), known["label"].to_numpy(), named
    if isinstance(known, pd.Series):
        named = "the Series of known labels"
        if bool(known.index.duplicated().any()):
            raise AlhenaValueError(
                f"{named} names a cell twice in its index. An identifier names one cell across "
                "all the tiles read together, not one cell within a tile."
            )
        return known.index.to_numpy(), known.to_numpy(), named
    if isinstance(known, Mapping):
        identifiers = list(known)
        return (
            # Kept as written: an array of integers beside text would turn the integers to text.
            pd.Index(identifiers).to_numpy(),
            np.asarray([known[key] for key in identifiers], dtype=object),
            "the mapping of known labels",
        )
    raise AlhenaValueError(
        f"Known labels of type {type(known).__name__} are not read. Pass a table with columns "
        f"{list(TRUTH_COLUMNS)}, a Series of labels indexed by cell identifier, a mapping from "
        f"identifier to label, or the path of a {' or '.join(_LABEL_TABLE_SUFFIXES)} table."
    )


def _aligned_truth(
    cell_id: np.ndarray,
    identifiers: np.ndarray,
    labels: np.ndarray,
    named: str,
    advice: str = _SELECT_ADVICE,
) -> np.ndarray:
    """Order known labels as the annotated cells, matching identifiers as written."""
    from alhena.cells._inputs import _identifier_kind

    # Each identifier names one cell, which the readers checked, so each row is found by hashing.
    rows = pd.Index(identifiers).get_indexer(cell_id)
    absent = rows < 0
    if bool(absent.any()):
        typed = _identifier_kind(cell_id)
        written = _identifier_kind(identifiers)
        opening = (
            f"{int(absent.sum())} of the {cell_id.shape[0]} cells that were annotated carry no "
            f"label in {named}."
        )
        if typed != written and bool(absent.all()):
            example = f", such as {identifiers.tolist()[0]!r}" if identifiers.size else ""
            raise AlhenaValueError(
                f"{opening} The annotations name their cells by {typed}, such as "
                f"{cell_id.tolist()[0]!r}, and {named} names them by {written}{example}. Name "
                "the cells of the known labels as the annotation table writes its identifiers."
            )
        kinds = (
            ""
            if typed == written
            else f" The annotations name their cells by {typed} and {named} by {written}."
        )
        raise AlhenaValueError(
            f"{opening}{kinds} The first cell with no label is {cell_id[absent].tolist()[0]!r}. "
            f"{advice}"
        )
    return labels[rows]
