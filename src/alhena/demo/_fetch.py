"""Fetching PanNuke tiles."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from alhena._errors import PACKAGE_PREFIXES, AlhenaError, AlhenaValueError, GeometryWarning
from alhena.demo._pannuke import (
    _TILE_COLUMNS,
    CATEGORIES_COLUMN,
    DEAD_CATEGORY,
    IMAGE_COLUMN,
    INSTANCES_COLUMN,
    PANNUKE_FOLD_FILES,
    PANNUKE_MPP,
    PANNUKE_REPO,
    PANNUKE_REVISION,
    PANNUKE_TILE_PX,
    PANNUKE_TISSUES,
    PANNUKE_TO_COMPARTMENT,
    TISSUE_COLUMN,
    _category_name,
    _check_class_codes,
    _payload,
    decode_image,
    decode_mask,
    pannuke_tissue,
    reference_tissue,
)
from alhena.io import label_map_from_masks

DEFAULT_TISSUE: Final[str] = "Breast"
"""Tissue the example reads by default."""
DEFAULT_N_TILES: Final[int] = 8
"""Fields the example reads by default."""
DEFAULT_FOLD: Final[int] = 1
"""Fold the example reads by default."""

GEOMETRY_CAVEAT: Final[str] = (
    "PanNuke declares 0.25 microns per pixel, which is nominal and resampled rather than native "
    "optical: fields scanned at 20x were enlarged to 40x by the authors, and the release marks "
    "none of them, so they cannot be filtered out. The declared resolution is passed on as it "
    "stands."
)
"""The `GeometryWarning` every fetch raises: the resolution PanNuke declares is nominal."""

_REMOTE_ADVICE: Final[str] = (
    "Download one fold file from the repository and pass it as parquet=; nothing here "
    "downloads a whole fold, several hundred megabytes for a handful of fields."
)


@dataclass(frozen=True)
class PanNukeTile:
    """One PanNuke visual field and its nuclei, `Dead` ones left out.

    Attributes:
        tile_id: Name of the field, carrying the fold and the row it was read from.
        image: The field, (H, W, 3) uint8 RGB.
        label_map: One identifier per nucleus pixel, zero for background.
        cell_ids: Identifier of each nucleus kept, in annotation order.
        categories: PanNuke category of each nucleus kept.
        truth: Class of `compartment_v1` each nucleus kept is compared with.
        lost_cell_ids: Nuclei left with no pixel once overlaps were resolved.
        n_dead: Nuclei of the field left out as `Dead`.
    """

    tile_id: str
    image: np.ndarray
    label_map: np.ndarray
    cell_ids: np.ndarray
    categories: np.ndarray
    truth: np.ndarray
    lost_cell_ids: np.ndarray
    n_dead: int


@dataclass(frozen=True)
class PanNukeCohort:
    """The fields of one tissue that the example runs on.

    Attributes:
        tissue: Tissue in the PanNuke spelling.
        reference_tissue: Tissue whose published reference the predictions are made against.
        fold: Fold the fields were read from, None where a file already on disk was read, whose
            contents the fold number does not describe.
        source: The parquet the fields were read from.
        n_tissue_tiles: Fields the fold holds for that tissue, of which the first were read.
        tiles: The fields, in the order they were read.
    """

    tissue: str
    reference_tissue: str
    fold: int | None
    source: str
    n_tissue_tiles: int
    tiles: tuple[PanNukeTile, ...]

    @property
    def images(self) -> list[np.ndarray]:
        """The fields, one uint8 RGB array each."""
        return [tile.image for tile in self.tiles]

    @property
    def label_maps(self) -> list[np.ndarray]:
        """The label maps, in the order of the fields."""
        return [tile.label_map for tile in self.tiles]

    @property
    def n_nuclei(self) -> int:
        """Nuclei other than `Dead` ones, before overlaps and tokens remove any."""
        return int(sum(tile.cell_ids.shape[0] for tile in self.tiles))

    @property
    def n_dead(self) -> int:
        """Nuclei left out as `Dead`."""
        return int(sum(tile.n_dead for tile in self.tiles))

    @property
    def n_lost(self) -> int:
        """Nuclei left with no pixel once overlaps were resolved."""
        return int(sum(tile.lost_cell_ids.shape[0] for tile in self.tiles))

    def truth_by_cell(self) -> dict[int, str]:
        """The class of `compartment_v1` of every nucleus kept, keyed by identifier."""
        known: dict[int, str] = {}
        for tile in self.tiles:
            for cell_id, label in zip(tile.cell_ids, tile.truth, strict=True):
                known[int(cell_id)] = str(label)
        return known

    def categories_by_cell(self) -> dict[int, str]:
        """The PanNuke category of every nucleus kept, keyed by identifier."""
        known: dict[int, str] = {}
        for tile in self.tiles:
            for cell_id, name in zip(tile.cell_ids, tile.categories, strict=True):
                known[int(cell_id)] = str(name)
        return known


def _open_remote(filename: str, revision: str | None) -> Any:
    """Open a file of the dataset repository for reading, without downloading it whole."""
    from huggingface_hub import HfFileSystem

    repo = PANNUKE_REPO if revision is None else f"{PANNUKE_REPO}@{revision}"
    try:
        return HfFileSystem().open(f"datasets/{repo}/{filename}", "rb")
    except Exception as error:
        raise AlhenaError(
            f"{filename} of {PANNUKE_REPO} could not be opened: {error}. {_REMOTE_ADVICE}"
        ) from error


def _tissue_rows(parquet_file: Any, tissue: str, n_tiles: int) -> tuple[np.ndarray, int]:
    """Return the first rows of one tissue, and how many the fold holds, from that column alone."""
    index = PANNUKE_TISSUES.index(tissue)
    codes = parquet_file.read(columns=[TISSUE_COLUMN])[TISSUE_COLUMN].to_pylist()
    rows = np.flatnonzero(np.asarray(codes, dtype=np.int64) == index)
    if rows.shape[0] < n_tiles:
        raise AlhenaValueError(
            f"The fold holds {rows.shape[0]} fields of {tissue} and {n_tiles} were asked for. "
            "Ask for fewer, or read another fold."
        )
    return rows[:n_tiles], int(rows.shape[0])


def _read_rows(parquet_file: Any, rows: np.ndarray) -> Any:
    """Read only the row groups holding the wanted rows, and only the columns that are read."""
    metadata = parquet_file.metadata
    sizes = np.asarray(
        [metadata.row_group(group).num_rows for group in range(metadata.num_row_groups)],
        dtype=np.int64,
    )
    starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sizes)))
    holder = np.searchsorted(starts, rows, side="right") - 1
    groups = sorted({int(group) for group in holder})

    # Rows are addressed within the groups read, the skipped groups left out of the count.
    base: dict[int, int] = {}
    running = 0
    for group in groups:
        base[group] = running
        running += int(sizes[group])
    inside = [
        base[int(group)] + int(row - starts[group]) for row, group in zip(rows, holder, strict=True)
    ]
    return parquet_file.read_row_groups(groups, columns=list(_TILE_COLUMNS)).take(inside)


def _tiles_from_table(table: Any, rows: np.ndarray, prefix: str) -> tuple[PanNukeTile, ...]:
    """Turn the read rows into fields, label maps and per-nucleus truth."""
    images = table.column(IMAGE_COLUMN).to_pylist()
    instances = table.column(INSTANCES_COLUMN).to_pylist()
    categories = table.column(CATEGORIES_COLUMN).to_pylist()

    tiles: list[PanNukeTile] = []
    # Identifiers run across the cohort: a per-tile ordinal names a different nucleus per field.
    next_cell_id = 1
    for position, row in enumerate(rows):
        tile_id = f"{prefix}_{int(row):05d}"
        image = decode_image(_payload(images[position], f"{tile_id} image"))
        height, width = int(image.shape[0]), int(image.shape[1])
        if (height, width) != (PANNUKE_TILE_PX, PANNUKE_TILE_PX):
            raise AlhenaValueError(
                f"{tile_id} is {width} by {height} pixels, where a PanNuke visual field is "
                f"{PANNUKE_TILE_PX} by {PANNUKE_TILE_PX}. The declared resolution of "
                f"{PANNUKE_MPP} micron per pixel describes fields of that size, so a field of "
                "another size would be resampled to a physical extent it does not have."
            )
        codes = list(categories[position] or ())
        masks_encoded = list(instances[position] or ())
        if len(codes) != len(masks_encoded):
            raise AlhenaValueError(
                f"{tile_id} carries {len(masks_encoded)} masks and {len(codes)} categories, "
                "which are one per mask."
            )

        masks: list[np.ndarray] = []
        names: list[str] = []
        n_dead = 0
        for encoded, code in zip(masks_encoded, codes, strict=True):
            name = _category_name(int(code))
            if name == DEAD_CATEGORY:
                n_dead += 1
                continue
            masks.append(decode_mask(_payload(encoded, f"{tile_id} mask")))
            names.append(name)

        cell_ids = np.arange(next_cell_id, next_cell_id + len(masks), dtype=np.int64)
        next_cell_id += len(masks)
        label_map, lost = label_map_from_masks(masks, cell_ids, image.shape[:2], progress=False)
        tiles.append(
            PanNukeTile(
                tile_id=tile_id,
                image=image,
                label_map=label_map,
                cell_ids=cell_ids,
                categories=np.asarray(names, dtype=object),
                truth=np.asarray([PANNUKE_TO_COMPARTMENT[name] for name in names], dtype=object),
                lost_cell_ids=lost,
                n_dead=n_dead,
            )
        )
    return tuple(tiles)


def _fold_file(fold: int, parquet: str | Path | None) -> Path | None:
    """Return the fold file on disk, or None for the published fold; refuse an unknown fold."""
    # Checked whatever the source, so a fold number that selects nothing is still refused.
    if fold not in PANNUKE_FOLD_FILES:
        raise AlhenaValueError(
            f"Fold {fold} is not published. PanNuke publishes folds {sorted(PANNUKE_FOLD_FILES)}."
        )
    local = None if parquet is None else Path(parquet)
    if local is not None and not local.is_file():
        raise AlhenaValueError(
            f"No PanNuke fold file at {local}. Give parquet= a fold file already on disk, or "
            "leave it unset to read the published fold."
        )
    return local


def fetch_pannuke(
    tissue: str = DEFAULT_TISSUE,
    n_tiles: int = DEFAULT_N_TILES,
    fold: int = DEFAULT_FOLD,
    parquet: str | Path | None = None,
    revision: str | None = PANNUKE_REVISION,
) -> PanNukeCohort:
    """Read the first fields of one tissue out of one PanNuke fold.

    The fold is read where it lies, never downloaded whole: its tissue column, then the row groups
    holding the wanted fields.

    Args:
        tissue: PanNuke tissue to read, in any spelling `pannuke_tissue` accepts.
        n_tiles: How many fields of that tissue to read.
        fold: Fold to read, one of `PANNUKE_FOLD_FILES`. It selects nothing when `parquet` names a
            file already on disk, and the cohort then carries no fold.
        parquet: A fold file already on disk, read instead of the published one.
        revision: Branch, tag or commit of the dataset repository; None reads its default branch.

    Returns:
        The fields, their label maps, and the class of `compartment_v1` of every nucleus other
        than `Dead` ones.

    Raises:
        AlhenaValueError: If the tissue has no reference, the fold or the count is out of range, a
            local file is absent, the class codes are declared in another order, or the fold holds
            too few fields of the tissue.
        AlhenaError: If the published fold cannot be read where it lies.
    """
    name = pannuke_tissue(tissue)
    target = reference_tissue(name)
    if n_tiles < 1:
        raise AlhenaValueError(f"n_tiles must be at least 1, got {n_tiles}.")
    local = _fold_file(fold, parquet)
    source = str(local) if local is not None else f"{PANNUKE_REPO}/{PANNUKE_FOLD_FILES[fold]}"
    # Named after its file, so a local file the fold number does not describe gets accurate names.
    prefix = local.stem if local is not None else f"fold{fold}"

    import pyarrow.parquet as pq

    handle = None
    try:
        if local is None:
            handle = _open_remote(PANNUKE_FOLD_FILES[fold], revision)
        opened = pq.ParquetFile(handle if local is None else local)
        _check_class_codes(opened, source)
        rows, available = _tissue_rows(opened, name, n_tiles)
        table = _read_rows(opened, rows)
    except AlhenaError:
        raise
    except Exception as error:
        advice = _REMOTE_ADVICE if local is None else "Check the file is a PanNuke fold."
        raise AlhenaError(f"{source} could not be read: {error}. {advice}") from error
    finally:
        if handle is not None:
            handle.close()

    cohort = PanNukeCohort(
        tissue=name,
        reference_tissue=target,
        fold=None if local is not None else fold,
        source=source,
        n_tissue_tiles=available,
        tiles=_tiles_from_table(table, rows, prefix),
    )
    if cohort.n_nuclei == 0:
        raise AlhenaValueError(
            f"The {n_tiles} fields read of {name} carry no nucleus that can be scored. Read more "
            "fields, or another fold."
        )
    warnings.warn(GEOMETRY_CAVEAT, GeometryWarning, skip_file_prefixes=PACKAGE_PREFIXES)
    return cohort
