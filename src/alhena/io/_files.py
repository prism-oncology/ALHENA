"""Reading arrays, images and tables, from a file or a frame in memory, and picking columns."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from alhena._errors import AlhenaValueError
from alhena._geometry import _PAD_VALUE
from alhena.io._formats import IMAGE_SUFFIXES, TABLE_SUFFIXES

_PARQUET_SUFFIXES: tuple[str, ...] = (".parquet", ".pq")

# File metadata key a GeoParquet file declares itself by; a parquet file without it is a table.
_GEOPARQUET_KEY: bytes = b"geo"

# Delimiters a `.txt` table is read under, the first of them where its header line holds none.
_TEXT_DELIMITERS: tuple[str, ...] = (",", "\t", ";", "|")


def _list_files(root: Path, pattern: str | None, suffixes: Sequence[str], what: str) -> list[Path]:
    """Return the files of a directory that carry one of the given suffixes, in name order."""
    if not root.is_dir():
        raise AlhenaValueError(f"No directory of {what}s at {root}.")
    entries = sorted(root.glob(pattern)) if pattern is not None else sorted(root.iterdir())
    accepted = {suffix.lower() for suffix in suffixes}
    # Hidden files are skipped: file systems write sidecars named after the files they shadow.
    files = [
        entry
        for entry in entries
        if entry.is_file() and not entry.name.startswith(".") and entry.suffix.lower() in accepted
    ]
    if not files:
        raise AlhenaValueError(
            f"{root} holds no {what}. Files are taken by suffix, one of {sorted(accepted)}"
            + (f", matching {pattern!r}." if pattern is not None else ".")
        )
    return files


def _read_image_file(path: Path) -> np.ndarray:
    """Read one image file as a uint8 RGB array."""
    if not path.is_file():
        raise AlhenaValueError(f"No tile image at {path}.")
    buffer = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if buffer.size == 0:
        raise AlhenaValueError(f"{path} is empty.")
    # IMREAD_COLOR would decode one plane per channel as the first plane tripled, losing the color.
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AlhenaValueError(
            f"{path} could not be decoded as an image. Tiles are read from "
            f"{list(IMAGE_SUFFIXES)} files."
        )
    return _as_rgb_pixels(image, path)


def _as_rgb_pixels(decoded: np.ndarray, path: Path) -> np.ndarray:
    """Return decoded pixels as uint8 RGB, a fourth channel read as alpha and laid on white."""
    pixels = np.asarray(decoded)
    channels = pixels.shape[2] if pixels.ndim == 3 else 0
    if channels not in (3, 4) or pixels.dtype != np.uint8:
        raise AlhenaValueError(
            f"{path.name} decodes to shape {tuple(pixels.shape)} of {pixels.dtype}, where bytes "
            "of color interleaved one pixel at a time, three channels or four, are expected. A "
            "pyramidal TIFF of interleaved color is read one region at a time by read_slide."
        )
    # OpenCV decodes into BGR, turned into RGB here and nowhere else, before the normalization.
    if channels == 4:
        return _on_white(cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGBA))
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)


def _on_white(rgba: np.ndarray) -> np.ndarray:
    """Lay uint8 RGBA pixels, straight alpha, on white and return them as RGB."""
    alpha = rgba[..., 3]
    if bool((alpha == 255).all()):
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    weight = alpha.astype(np.uint16)
    # Each blend stays within 255 * 255 + 127, which sixteen bits hold.
    background = (255 - weight) * _PAD_VALUE + 127
    rgb = np.empty((*rgba.shape[:2], 3), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = (rgba[..., channel] * weight + background) // 255
    return rgb


def _is_geoparquet(path: Path) -> bool:
    """Return whether a file is parquet declaring itself GeoParquet; an unreadable one is not."""
    if path.suffix.lower() not in _PARQUET_SUFFIXES or not path.is_file():
        return False
    import pyarrow.parquet as parquet

    try:
        metadata = parquet.read_schema(path).metadata or {}
    except (OSError, ValueError):
        return False
    return _GEOPARQUET_KEY in metadata


def _is_geodataframe(source: Any) -> bool:
    """Return whether a source is a GeoDataFrame, without importing geopandas to tell."""
    geopandas = sys.modules.get("geopandas")
    return geopandas is not None and isinstance(source, geopandas.GeoDataFrame)


def _named(source: Any) -> str:
    """Return how messages name a table: by its file name, or as the DataFrame given."""
    if isinstance(source, pd.DataFrame):
        return f"the {type(source).__name__}"
    return Path(source).name


def _table(
    source: Any,
    what: str = "nuclei",
    as_text: bool = False,
    id_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return a table given as a DataFrame, or read from its path by `_read_table`."""
    if isinstance(source, pd.DataFrame):
        return source
    if not isinstance(source, str | os.PathLike):
        raise AlhenaValueError(
            f"A table of {what} of type {type(source).__name__} is not read. Pass a DataFrame or "
            f"the path of a {', '.join(TABLE_SUFFIXES)} file."
        )
    return _read_table(Path(source), what, as_text, id_columns)


def _read_table(
    path: Path,
    what: str = "nuclei",
    as_text: bool = False,
    id_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Read a table; `as_text` and `id_columns` read columns as text, so `007` keeps its zeros."""
    if not path.is_file():
        raise AlhenaValueError(f"No table of {what} at {path}.")
    suffix = path.suffix.lower()
    text: dict[str, Any] = {}
    if as_text:
        text = {"dtype": str, "keep_default_na": False, "na_values": [""]}
    elif id_columns:
        text = {"dtype": dict.fromkeys(id_columns, str)}
    try:
        if suffix == ".csv":
            return pd.read_csv(path, **text)
        if suffix == ".txt":
            return pd.read_csv(path, sep=_text_delimiter(path), **text)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", **text)
        if suffix in _PARQUET_SUFFIXES:
            return pd.read_parquet(path)
    # The parsers raise ValueError for an empty, malformed, non-utf-8 or non-parquet file.
    except ValueError as error:
        raise AlhenaValueError(f"{path} could not be read as a table: {error}") from error
    raise AlhenaValueError(
        f"Tables of {what} are read from {list(TABLE_SUFFIXES)} files, and {path.name} has suffix "
        f"{path.suffix!r}."
    )


def _text_delimiter(path: Path) -> str:
    """Return the delimiter of a `.txt` table: the one its header holds most of, else a comma."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    except OSError:
        return _TEXT_DELIMITERS[0]
    counts = {candidate: header.count(candidate) for candidate in _TEXT_DELIMITERS}
    found = max(counts, key=lambda candidate: counts[candidate])
    return found if counts[found] else _TEXT_DELIMITERS[0]


def _identifier_columns(
    chosen: Mapping[str, str], looked_for: Mapping[str, Sequence[str]]
) -> tuple[str, ...]:
    """Return the names the column of cell identifiers may go under, for `_read_table`."""
    named = chosen.get("cell_id")
    return (named,) if named is not None else tuple(looked_for["cell_id"])


def _pick_column(
    frame: pd.DataFrame,
    named: str | None,
    candidates: Sequence[str],
    described: str,
    required: bool = True,
    advice: str = "",
) -> Any:
    """Return the label of the column carrying one field, matching labels as text."""
    labels = list(frame.columns)
    names = [str(label) for label in labels]
    if named is not None:
        if str(named) not in names:
            raise AlhenaValueError(f"The table holds no column {named!r}. It holds {names}.")
        return _only_label(labels, names, str(named))
    found = [candidate for candidate in candidates if candidate in names]
    if len(found) > 1:
        raise AlhenaValueError(
            f"The table holds {found}, and which of them carries the {described} cannot be "
            "settled. Name it through columns=."
        )
    if not found:
        if required:
            raise AlhenaValueError(
                f"The table holds no {described} column among {names}, looked for under "
                f"{list(candidates)}. Name it through columns=." + (f" {advice}" if advice else "")
            )
        return None
    return _only_label(labels, names, found[0])


def _only_label(labels: list[Any], names: list[str], name: str) -> Any:
    """Return the one column label spelled `name`, refusing a name two columns share."""
    if names.count(name) > 1:
        raise AlhenaValueError(
            f"The table holds {names.count(name)} columns named {name!r}. Give each column a name "
            "of its own."
        )
    return labels[names.index(name)]


def _choice(value: Any, allowed: Sequence[str], name: str) -> str:
    """Return an option given as one of the values it may take, refusing any other."""
    if not isinstance(value, str) or value not in allowed:
        raise AlhenaValueError(f"{name} is {value!r}, where one of {list(allowed)} is expected.")
    return value


def _column_names(
    columns: Mapping[str, str] | None, fields_read: Mapping[str, Any], what: str
) -> dict[str, str]:
    """Return the columns a caller names, refusing a field the table it reads does not have."""
    chosen = dict(columns or {})
    unknown = sorted(str(name) for name in set(chosen) - set(fields_read))
    if unknown:
        raise AlhenaValueError(
            f"columns names {unknown}, which are not fields of {what}. Its fields are "
            f"{list(fields_read)}."
        )
    return chosen


def _coordinates(table: pd.DataFrame, names: tuple[Any, Any], named: str, what: str) -> np.ndarray:
    """Return the x and y of every row of a table, refusing a position that is not two numbers."""
    if len(table) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    values = np.column_stack(
        [
            pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
            for name in names
        ]
    )
    unreadable = ~np.isfinite(values).all(axis=1)
    if unreadable.any():
        row = int(np.flatnonzero(unreadable)[0])
        written = (table[name].iloc[row] for name in names)
        shown = ["missing" if pd.isna(value) else repr(str(value)) for value in written]
        count = int(unreadable.sum())
        raise AlhenaValueError(
            f"{count} {'row' if count == 1 else 'rows'} of {named} "
            f"{'gives' if count == 1 else 'give'} no {what} position that can be read, first "
            f"row {row}, whose x is {shown[0]} and y {shown[1]}, where two finite numbers are "
            "expected."
        )
    return values


def _text_identifiers(values: Any, described: str) -> np.ndarray:
    """Return identifiers with any stored as bytes read as the UTF-8 text it spells."""
    array = np.asarray(values)
    if array.dtype.kind not in "OS":
        return array
    items = array.reshape(-1).tolist()
    if not any(isinstance(item, bytes | bytearray) for item in items):
        return array
    decoded: list[Any] = []
    for item in items:
        if isinstance(item, bytes | bytearray):
            try:
                item = bytes(item).decode("utf-8")
            except UnicodeDecodeError as error:
                raise AlhenaValueError(
                    f"{described} holds the identifier {bytes(item)!r}, stored as bytes that are "
                    "not utf-8 text. Write the identifiers as text or as integers."
                ) from error
        decoded.append(item)
    text = np.empty(len(decoded), dtype=object)
    text[:] = decoded
    return text.reshape(array.shape)


def _load_npy(path: Path, mapped: bool = True) -> np.ndarray:
    """Return the array a `.npy` file holds, mapped from disk unless `mapped` is false."""
    try:
        return np.load(path, mmap_mode="r" if mapped else None, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise AlhenaValueError(f"{path} could not be read as a NumPy array: {error}") from error
