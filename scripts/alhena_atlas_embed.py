#!/usr/bin/env python3
"""Embed a single-cell atlas into the cell space ALHENA retrieves against.

The first of two steps, run in the environment of `scripts/requirements-atlas.txt`; it never imports
alhena. Its output feeds `alhena.refbuild.build_reference`, which matches labels to cells by row
position.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# --- The released cell space ----------------------------------------------------------------

MODEL_NAME = "corpus40M-model30M"
"""Single-cell model the released references were embedded with."""

SCCONCEPT_VERSION = "v2"
"""Release of the single-cell model family, as the package records it in a shard."""

SCCONCEPT_DISTRIBUTION = "sc-concept"
"""Distribution providing the single-cell model."""

SCCONCEPT_PACKAGE_VERSION = "0.2.4"
"""Release of that distribution the released references were embedded under."""

UNKNOWN_VERSION = "unknown"
"""Value standing for a version that cannot be measured in this environment."""

SPECIES = "hsapiens"
"""Species key of the tokenizer vocabulary."""

ADAPTATION_STEPS = 0
"""Adaptation steps taken before extraction: none, so the pretrained model is loaded as it is."""

PRETRAINED_VOCABULARY = False
"""Whether a pretrained vocabulary loads beside the checkpoint: not for the released space."""

PRETRAINED_VOCABULARY_DIRNAME = "pretrained_vocabulary"
"""Directory of the model assets holding a pretrained vocabulary, where there is one."""

MAX_TOKENS = 1000
"""Gene tokens read per cell."""

GENE_SAMPLING_STRATEGY = "top-nonzero"
"""Rule selecting which genes of a panel fill the token budget."""

EMBEDDING_DIM = 512
"""Width of the cls embedding the model returns, the width the reference head reads."""

CLS_KEY = "cls_cell_emb"
"""Key `predict_step` returns the cls embedding under."""

# --- Environment and output -------------------------------------------------------------------

ENVIRONMENT_NAME = "alhena_atlas"
"""Environment this script runs in: the model pins torch and a huggingface-hub below 1.0."""

ENVIRONMENT_FILE = "scripts/requirements-atlas.txt"
"""Declaration of that environment, relative to the repository root."""

DEFAULT_DATASET = "cls_emb"
"""Dataset the package reads the embeddings from."""

OBS_NAME_DATASET = "obs_name"
"""Dataset recording the cell names of the source atlas."""

OBS_NAME_ROLE = "provenance"
"""What the cell names are for: recognizing the atlas, never matching on."""

ROW_ORDER = "source_atlas"
"""Value of the metadata field recording that rows keep the order of the source atlas."""

# --- Defaults ---------------------------------------------------------------------------------

DEFAULT_GENE_ID_COLUMN = "gene_ids"
"""Column of the gene table holding the gene identifiers the corpus vocabulary is keyed on."""

VAR_NAMES = "var_names"
"""Value of `--gene-id-column` selecting `adata.var_names` instead of a column."""

DEFAULT_BATCH_SIZE = 256
"""Cells per forward pass."""

DEFAULT_NUM_WORKERS = 4
"""Processes tokenizing cells alongside the forward pass."""

DEFAULT_CHUNK_MEMORY_GB = 2.0
"""Gigabytes one materialized block of the expression matrix may occupy."""

DEFAULT_COUNT_CHECK_CELLS = 2048
"""Cells sampled to check that the expression matrix holds raw counts."""

DEFAULT_MIN_VOCABULARY_FRACTION = 0.3
"""Fraction of the panel that must be in the corpus vocabulary."""

DEFAULT_MIN_GENES_IN_VOCABULARY = 1
"""Genes of the panel that must be in the corpus vocabulary."""

# --- Gene identifiers -------------------------------------------------------------------------

GENE_ID_TYPES = ("ensembl", "symbol")
"""What the gene identifier column is read as: Ensembl identifiers, the default, or symbols."""

ENSEMBL_GENE_ID = re.compile(r"(ENS[A-Z]*G\d{11})(\.\d+)?(_PAR_Y)?")
"""An Ensembl gene identifier as GENCODE writes it, with an optional version and `_PAR_Y` tag."""

GENE_MAPPINGS_DIRNAME = "gene_mappings"
"""Directory of the model assets holding one gene table per species, as `<species>.csv`."""

GENE_TABLE_ID_COLUMN = "gene_id"
"""Column of a gene table holding the identifiers the vocabulary is keyed on."""

GENE_TABLE_SYMBOL_COLUMN = "gene_name"
"""Column of a gene table holding the symbol of each gene."""

EXAMPLES_NAMED = 3
"""Entries a message names where it reports on many."""

COMFORTABLE_VOCABULARY_FRACTION = 0.8
"""Fraction of a panel below which the tokenizer reads much less than the atlas carries."""

# --- Chunks and storage -----------------------------------------------------------------------

MAX_ROWS_PER_CHUNK = 100_000
"""Cells materialized at once, whatever the memory budget allows."""

COUNT_CHECK_BLOCKS = 3
"""Blocks the raw-count check samples, so a normalized tail cannot hide behind a raw head."""

BYTES_PER_VALUE = 4
"""Bytes one expression value occupies once a chunk is materialized."""

STORAGE_CHUNK_ROWS = 4096
"""Rows per chunk of the stored embedding matrix."""

STORAGE_COMPRESSION = 4
"""gzip level the stored embedding matrix is written at."""

STORAGE_DTYPES = {"float16": np.float16, "float32": np.float32}
"""Precisions the embeddings may be stored at, by name."""


# --- Errors and warnings ----------------------------------------------------------------------


class AtlasEmbedError(RuntimeError):
    """The atlas, the model assets or the output path cannot be used as given."""


class AtlasEmbedValueError(AtlasEmbedError, ValueError):
    """An argument or an input value is outside the range this step accepts.

    Subclasses `ValueError` as well, so a caller may catch either.
    """


class AtlasEmbedWarning(UserWarning):
    """Something changes how the resulting embeddings should be read."""


def _report(message: str) -> None:
    """Write one progress line to standard error, separately from the JSON result."""
    print(message, file=sys.stderr, flush=True)


# --- Chunks, checks and the gene panel --------------------------------------------------------


def rows_per_chunk(n_genes: int, budget_gb: float = DEFAULT_CHUNK_MEMORY_GB) -> int:
    """Return the number of cells materialized at once under a memory budget.

    Args:
        n_genes: Columns of the expression matrix.
        budget_gb: Gigabytes one materialized chunk may occupy.

    Returns:
        Cells per chunk, at least one and at most `MAX_ROWS_PER_CHUNK`.

    Raises:
        AtlasEmbedValueError: If the gene count is not positive.
    """
    if n_genes < 1:
        raise AtlasEmbedValueError(f"The atlas must carry at least one gene, got {n_genes}.")
    fits = int((budget_gb * 1024.0**3) // (n_genes * BYTES_PER_VALUE))
    return max(1, min(MAX_ROWS_PER_CHUNK, fits))


def chunk_bounds(n_rows: int, rows: int) -> list[tuple[int, int]]:
    """Return contiguous row ranges covering the atlas in order.

    Args:
        n_rows: Rows to cover.
        rows: Rows per range.

    Returns:
        Half-open ranges, in ascending order.

    Raises:
        AtlasEmbedValueError: If either count is not positive.
    """
    if n_rows < 1:
        raise AtlasEmbedValueError(f"The atlas holds no cell to embed, got {n_rows} rows.")
    if rows < 1:
        raise AtlasEmbedValueError(f"A chunk must hold at least one row, got {rows}.")
    return [(start, min(start + rows, n_rows)) for start in range(0, n_rows, rows)]


def pass_line(
    number: int,
    n_passes: int,
    bounds: tuple[int, int],
    done: int,
    total: int,
    elapsed_s: float,
) -> str:
    """Return the progress line announcing one pass, with the time spent and the time left.

    The first pass, with nothing run yet, announces no estimate.

    Args:
        number: Pass about to run, counted from one.
        n_passes: Passes in the run.
        bounds: First row and the row past the last of the pass.
        done: Cells embedded by the passes already run.
        total: Cells in the atlas.
        elapsed_s: Seconds spent embedding so far.

    Returns:
        The line, ready for `_report`.
    """
    line = f"[embed]   pass {number} of {n_passes}, cells {bounds[0] + 1} to {bounds[1]}"
    if done < 1 or elapsed_s <= 0.0:
        return line
    left_s = elapsed_s * (total - done) / done
    return f"{line}, {elapsed_s / 60.0:.1f} min elapsed, about {left_s / 60.0:.1f} min left"


def sample_blocks(n_rows: int, n_sampled: int) -> list[tuple[int, int]]:
    """Return `COUNT_CHECK_BLOCKS` row ranges spread over the atlas, `n_sampled` rows at most.

    Args:
        n_rows: Rows of the atlas.
        n_sampled: Rows to sample in total.

    Returns:
        Disjoint half-open ranges, in ascending order.

    Raises:
        AtlasEmbedValueError: If the atlas is empty.
    """
    if n_rows < 1:
        raise AtlasEmbedValueError(f"The atlas holds no cell to sample, got {n_rows} rows.")
    if n_rows <= n_sampled:
        return [(0, n_rows)]
    blocks = max(1, min(COUNT_CHECK_BLOCKS, n_sampled))
    size = max(1, n_sampled // blocks)
    if blocks == 1:
        return [(0, size)]
    span = n_rows - size
    starts = sorted({round(index * span / (blocks - 1)) for index in range(blocks)})
    return [(start, start + size) for start in starts]


# Closing of the refusals of a matrix that holds no raw counts.
_RAW_COUNTS_ADVICE = (
    "Pass the raw counts: through --layer if they live in a layer, or --use-raw if they live in "
    "adata.raw."
)


def check_raw_counts(values: np.ndarray, described: str) -> dict[str, float]:
    """Check that a block of the expression matrix holds raw counts.

    The model normalizes counts itself, so a matrix normalized beforehand yields embeddings that
    look valid and are not.

    Args:
        values: A block of the expression matrix.
        described: How to name the matrix in an error message.

    Returns:
        The values seen, their range, and the fraction of them that is non-zero.

    Raises:
        AtlasEmbedValueError: If the block holds a value that is not finite, is negative, or is not
            a whole number.
    """
    block = np.asarray(values, dtype=np.float32)
    if block.size == 0:
        raise AtlasEmbedValueError(f"{described} holds no value to check.")
    if not bool(np.isfinite(block).all()):
        raise AtlasEmbedValueError(
            f"{described} holds values that are not finite. The model reads a count matrix."
        )
    smallest = float(block.min())
    largest = float(block.max())
    if smallest < 0.0:
        raise AtlasEmbedValueError(
            f"{described} holds negative values, the smallest being {smallest:g}, so it has been "
            f"scaled or centered. {_RAW_COUNTS_ADVICE}"
        )
    if not bool(np.array_equal(block, np.rint(block))):
        raise AtlasEmbedValueError(
            f"{described} is not integer-valued, so it has been normalized, log-transformed or "
            "scaled, and the model, which normalizes counts itself, would embed it wrongly. "
            f"{_RAW_COUNTS_ADVICE}"
        )
    return {
        "n_values": float(block.size),
        "minimum": smallest,
        "maximum": largest,
        "nonzero_fraction": float(np.count_nonzero(block)) / float(block.size),
    }


def check_tokenizer(tokenizer: Any) -> None:
    """Check that a tokenizer is the multi-species one this recipe reads.

    Args:
        tokenizer: Tokenizer of the loaded model.

    Raises:
        AtlasEmbedError: If it carries no species vocabulary.
    """
    if not hasattr(tokenizer, "species") or not hasattr(tokenizer, "NOT_FOUND"):
        raise AtlasEmbedError(
            "The loaded model carries a tokenizer without a species vocabulary, so it is not the "
            f"multi-species model this step reads. Load {MODEL_NAME}."
        )


def check_obs_names(obs_names: np.ndarray) -> bool:
    """Report whether the cell names of an atlas identify its cells.

    The names are provenance, never a join key, so repeated names are reported rather than refused.

    Args:
        obs_names: Cell names, in the row order of the atlas.

    Returns:
        Whether every name occurs once.
    """
    names = np.asarray(obs_names, dtype=str)
    distinct = int(np.unique(names).shape[0])
    if distinct == int(names.shape[0]):
        return True
    warnings.warn(
        f"The atlas holds {int(names.shape[0])} cells under {distinct} distinct names, so the "
        f"{OBS_NAME_DATASET!r} dataset records names that identify no cell. The embeddings are "
        "unaffected, as labels are matched by position, but labels joined on those names are not.",
        AtlasEmbedWarning,
        stacklevel=2,
    )
    return False


def check_embedding_width(width: int) -> None:
    """Check that the model returns embeddings of the width the reference head reads.

    Args:
        width: Columns of one block of embeddings.

    Raises:
        AtlasEmbedError: If the width is not the one the released cell space is defined at.
    """
    if width != EMBEDDING_DIM:
        raise AtlasEmbedError(
            f"The model returned {width}-dimensional embeddings where the reference head reads "
            f"{EMBEDDING_DIM}. The released cell space is {EMBEDDING_DIM} wide and is produced by "
            f"{MODEL_NAME}."
        )


def _listed(values: Sequence[str]) -> str:
    """Return up to `EXAMPLES_NAMED` values as a message lists them."""
    named = [repr(str(value)) for value in values[:EXAMPLES_NAMED]]
    return named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"


def normalize_gene_ids(identifiers: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Read Ensembl gene identifiers without the version a GENCODE-based table appends.

    `ENSG00000141510.17` is read as `ENSG00000141510`. A `_PAR_Y` copy keeps its tag and is left
    unread beside its X copy; any other entry is left as it is.

    Args:
        identifiers: Gene identifiers, in column order.

    Returns:
        The identifiers the tokenizer is given, in column order, and the counts read without a
        version and carrying the tag, under `n_gene_ids_normalized` and `n_gene_ids_par_y`.
    """
    given = np.asarray(identifiers, dtype=str)
    normalized = given.astype(object)
    unversioned = 0
    tagged = 0
    for position, value in enumerate(given.tolist()):
        match = ENSEMBL_GENE_ID.fullmatch(value)
        if match is None:
            continue
        stable, version, tag = match.groups()
        tagged += int(tag is not None)
        if version is not None:
            unversioned += 1
            normalized[position] = stable + (tag or "")
    counts = {"n_gene_ids_normalized": unversioned, "n_gene_ids_par_y": tagged}
    return np.asarray(normalized, dtype=str), counts


def read_gene_table(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read a gene table of the model assets, as the identifiers each gene symbol names.

    One row per gene of the vocabulary, under `gene_id` and `gene_name`; a row naming no symbol, as
    a special token's, is skipped.

    Args:
        path: Path of the table, `gene_mappings/<species>.csv` among the model assets.

    Returns:
        The identifiers each symbol names, in the order the table lists them.

    Raises:
        AtlasEmbedError: If there is no table at `path`, or it holds no column of identifiers or no
            column of symbols.
    """
    import csv

    file = Path(path)
    if not file.is_file():
        raise AtlasEmbedError(
            f"No gene table at {file}, so gene symbols cannot be mapped. Pass --model-cache the "
            "directory holding the one named after the model, and --species a species it carries."
        )
    with file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = [
            name for name in (GENE_TABLE_ID_COLUMN, GENE_TABLE_SYMBOL_COLUMN) if name not in columns
        ]
        if missing:
            raise AtlasEmbedError(
                f"The gene table {file} holds the columns {columns} and no {missing}, so gene "
                "symbols cannot be mapped through it. Map them to Ensembl gene identifiers in a "
                "column of adata.var, and name it with --gene-id-column."
            )
        named: dict[str, list[str]] = {}
        for row in reader:
            symbol = row[GENE_TABLE_SYMBOL_COLUMN] or ""
            identifier = row[GENE_TABLE_ID_COLUMN] or ""
            if symbol and identifier:
                named.setdefault(symbol, []).append(identifier)
    return {symbol: tuple(dict.fromkeys(found)) for symbol, found in named.items()}


def map_gene_symbols(
    symbols: np.ndarray, table: Mapping[str, Sequence[str]]
) -> tuple[np.ndarray, dict[str, int]]:
    """Replace each gene symbol by the Ensembl gene identifier the gene table gives it.

    Symbols match exactly, case included. A symbol the table gives several identifiers, or none, is
    passed on unchanged, so the tokenizer finds no gene for it; the ambiguous ones are warned about.

    Args:
        symbols: Gene symbols, in column order.
        table: The identifiers each symbol names, as `read_gene_table` returns them.

    Returns:
        The identifiers, in column order, with the numbers of symbols mapped, ambiguous and
        unmatched, under `n_gene_symbols_mapped`, `n_gene_symbols_ambiguous` and
        `n_gene_symbols_unmatched`.
    """
    given = np.asarray(symbols, dtype=str)
    mapped = given.astype(object)
    ambiguous: list[str] = []
    replaced = 0
    unmatched = 0
    for position, symbol in enumerate(given.tolist()):
        found = table.get(symbol, ())
        if len(found) == 1:
            mapped[position] = found[0]
            replaced += 1
        elif found:
            ambiguous.append(symbol)
        else:
            unmatched += 1
    if ambiguous:
        warnings.warn(
            f"{len(ambiguous)} of the {given.shape[0]} panel symbols, such as "
            f"{_listed(ambiguous)}, name several genes in the gene table of the model, so "
            "nothing says which of them a column counts, and the model does not read those "
            "columns. Map these symbols to Ensembl gene identifiers in a column of adata.var, and "
            "name it with --gene-id-column, to have them read.",
            AtlasEmbedWarning,
            stacklevel=2,
        )
    counts = {
        "n_gene_symbols_mapped": replaced,
        "n_gene_symbols_ambiguous": len(ambiguous),
        "n_gene_symbols_unmatched": unmatched,
    }
    return np.asarray(mapped, dtype=str), counts


def read_panel(
    given: np.ndarray,
    described: str,
    symbols: Mapping[str, Sequence[str]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the identifiers the tokenizer is given for a panel, and a record of how.

    Symbols are mapped first where a table is given, then versions are dropped. Two columns reaching
    one identifier are refused: whether to keep one or sum them is the caller's choice.

    Args:
        given: The panel as the atlas writes it, in column order.
        described: How to name the panel in an error message.
        symbols: The identifiers each symbol names, to read the panel as gene symbols, or None to
            read it as Ensembl gene identifiers.

    Returns:
        The identifiers, in column order, and the record the output keeps: `gene_id_type`,
        `gene_ids_normalized` and the counts of `normalize_gene_ids` and `map_gene_symbols`, zero
        where they did not apply.

    Raises:
        AtlasEmbedValueError: If two columns reach one identifier.
    """
    written = np.asarray(given, dtype=str)
    record: dict[str, Any] = {
        "gene_id_type": GENE_ID_TYPES[0] if symbols is None else GENE_ID_TYPES[1],
        "n_gene_symbols_mapped": 0,
        "n_gene_symbols_ambiguous": 0,
        "n_gene_symbols_unmatched": 0,
    }
    identifiers = written
    if symbols is not None:
        identifiers, mapped = map_gene_symbols(written, symbols)
        record.update(mapped)
    identifiers, normalized = normalize_gene_ids(identifiers)
    record.update(normalized)
    record["gene_ids_normalized"] = normalized["n_gene_ids_normalized"] > 0

    reached, counts = np.unique(identifiers, return_counts=True)
    shared = reached[counts > 1]
    if shared.size:
        columns = np.flatnonzero(identifiers == shared[0])
        how = "once versions are removed"
        if symbols is not None:
            how = "once symbols are mapped and versions removed"
        found = (
            f"One identifier of {described} is"
            if shared.size == 1
            else f"{shared.size} identifiers of {described} are each"
        )
        raise AtlasEmbedValueError(
            f"{found} reached by more than one column {how}: "
            f"{_listed(written[columns].tolist())} read as {str(shared[0])!r}, one gene read "
            "twice. Keep one of them, or sum them into one column, and run this step again."
        )
    return identifiers, record


def vocabulary_overlap(gene_ids: np.ndarray, tokenizer: Any, species: str) -> dict[str, float]:
    """Measure how much of a panel the corpus vocabulary holds.

    A gene outside the vocabulary is masked out inside the model, so a panel barely in it is
    embedded from almost nothing.

    Args:
        gene_ids: Gene identifiers as the tokenizer is given them, in column order.
        tokenizer: Tokenizer of the loaded model.
        species: Species key routing the per-species vocabulary.

    Returns:
        The panel size, the genes found in the vocabulary, and their fraction.

    Raises:
        AtlasEmbedError: If the tokenizer is not the multi-species one this recipe reads.
    """
    check_tokenizer(tokenizer)
    tokens = np.asarray(tokenizer.encode(gene_ids, species))
    n_genes = int(gene_ids.shape[0])
    found = int(np.count_nonzero(tokens != tokenizer.NOT_FOUND))
    return {
        "n_genes": float(n_genes),
        "n_genes_in_vocabulary": float(found),
        "fraction": (found / n_genes) if n_genes else 0.0,
    }


def check_vocabulary_overlap(
    overlap: dict[str, float],
    min_fraction: float = DEFAULT_MIN_VOCABULARY_FRACTION,
    min_genes: int = DEFAULT_MIN_GENES_IN_VOCABULARY,
) -> None:
    """Refuse a panel the corpus vocabulary barely holds, and report a thin one.

    Args:
        overlap: The measurement returned by `vocabulary_overlap`.
        min_fraction: Fraction of the panel that must be in the vocabulary.
        min_genes: Genes of the panel that must be in the vocabulary.

    Raises:
        AtlasEmbedValueError: If the overlap is below either floor.
    """
    found = int(overlap["n_genes_in_vocabulary"])
    n_genes = int(overlap["n_genes"])
    fraction = float(overlap["fraction"])
    if fraction < min_fraction or found < min_genes:
        raise AtlasEmbedValueError(
            f"Only {found} of the {n_genes} panel genes ({100.0 * fraction:.1f}%) are in the "
            f"corpus vocabulary, below the {100.0 * min_fraction:.1f}% and {min_genes} gene "
            "floors, so the embeddings would be degenerate. Name the column of Ensembl gene "
            "identifiers with --gene-id-column, or read gene symbols with --gene-id-type symbol."
        )
    if fraction < COMFORTABLE_VOCABULARY_FRACTION:
        warnings.warn(
            f"{found} of the {n_genes} panel genes ({100.0 * fraction:.1f}%) are in the corpus "
            "vocabulary, so the model reads a smaller panel than the atlas carries. The "
            "embeddings are computed from the genes it can read, and the cells are described by "
            "that much of their transcriptome.",
            AtlasEmbedWarning,
            stacklevel=2,
        )


def vocabulary_state(loaded: bool) -> str:
    """Return how a pretrained-vocabulary decision reads in a message."""
    return "loaded" if loaded else "not loaded"


def check_released_settings(
    model: str,
    species: str,
    max_tokens: int,
    strategy: str,
    pretrained_vocabulary: bool,
    package_version: str,
) -> None:
    """Report settings that leave the cell space the encoder's reference head was trained against.

    Args:
        model: Single-cell model being run.
        species: Species key of the tokenizer vocabulary.
        max_tokens: Gene tokens read per cell.
        strategy: Rule filling the token budget.
        pretrained_vocabulary: Whether the checkpoint is loaded with a pretrained vocabulary
            beside it, which changes the tokens the model reads and so the space it embeds into.
        package_version: Installed release of the distribution providing the model, or
            `UNKNOWN_VERSION` where this environment cannot report it.
    """
    if package_version == UNKNOWN_VERSION:
        warnings.warn(
            f"The installed {SCCONCEPT_DISTRIBUTION} release cannot be read in this environment, "
            f"so it cannot be checked against the {SCCONCEPT_PACKAGE_VERSION} the released cell "
            "space was built under. Install the model through its distribution rather than from a "
            "source tree, so the release is recorded in the output alongside the embeddings.",
            AtlasEmbedWarning,
            stacklevel=2,
        )
    declared = {
        "model": (model, MODEL_NAME),
        "species": (species, SPECIES),
        "token budget": (str(max_tokens), str(MAX_TOKENS)),
        "gene sampling strategy": (strategy, GENE_SAMPLING_STRATEGY),
        "pretrained vocabulary": (
            vocabulary_state(pretrained_vocabulary),
            vocabulary_state(PRETRAINED_VOCABULARY),
        ),
    }
    if package_version != UNKNOWN_VERSION:
        declared[f"{SCCONCEPT_DISTRIBUTION} release"] = (
            package_version,
            SCCONCEPT_PACKAGE_VERSION,
        )
    moved = [
        f"{name} {given!r} rather than {expected!r}"
        for name, (given, expected) in declared.items()
        if given != expected
    ]
    if moved:
        listed = moved[0] if len(moved) == 1 else f"{', '.join(moved[:-1])} and {moved[-1]}"
        warnings.warn(
            f"This run uses {listed}, so the embeddings land in a different cell "
            "space from the one the encoder's reference head was trained against. Nothing "
            "downstream detects it: a shard records the encoder that projected it, not this model.",
            AtlasEmbedWarning,
            stacklevel=2,
        )


# --- Reading the atlas ------------------------------------------------------------------------


@dataclass(frozen=True)
class Atlas:
    """The parts of an atlas this step reads, with its expression matrix left on disk.

    Attributes:
        path: Path of the h5ad file.
        n_cells: Rows of the expression matrix.
        n_genes: Columns of the expression matrix.
        gene_ids: Gene identifiers the tokenizer is given, in column order.
        obs_names: Cell names, in row order.
        obs: A cell table carrying the names and no column.
        var: A gene table carrying the identifiers the tokenizer is given, in the identifier
            column or, for the gene index, as its index.
        var_column: Column the identifiers were taken from, or None for the gene index.
        layer: Layer the counts are read from, or None for the main matrix.
        panel: How the identifiers were read from the atlas, as `read_panel` records it.
        use_raw: Whether the counts and the gene table are those of `adata.raw`.
    """

    path: Path
    n_cells: int
    n_genes: int
    gene_ids: np.ndarray
    obs_names: np.ndarray
    obs: Any
    var: Any
    var_column: str | None
    layer: str | None
    panel: dict[str, Any]
    use_raw: bool = False


def open_atlas(
    path: str | Path,
    gene_id_column: str = DEFAULT_GENE_ID_COLUMN,
    layer: str | None = None,
    symbols: Mapping[str, Sequence[str]] | None = None,
    use_raw: bool = False,
) -> Atlas:
    """Read an atlas header and check that its genes can be tokenized.

    Only the cell names, the gene table and the shape of the matrix are read; the matrix stays on
    disk, read in chunks later. The gene identifiers are read as `read_panel` reads them.

    Args:
        path: Path of the h5ad file.
        gene_id_column: Column of the gene table holding the gene identifiers, or `var_names` to
            use the gene index.
        layer: Layer holding the raw counts, or None to read the main matrix.
        symbols: The identifiers each gene symbol names, as `read_gene_table` returns them, to read
            the identifiers as gene symbols; None to read them as Ensembl gene identifiers.
        use_raw: Read the matrix and the gene table of `adata.raw`, where CELLxGENE files keep
            the raw counts.

    Returns:
        The atlas header.

    Raises:
        AtlasEmbedValueError: If the file, the layer, `adata.raw` or the column is missing, if both
            a layer and `adata.raw` are asked for, if an identifier is missing or repeated as
            written or once read, or if the atlas holds no cell.
        AtlasEmbedError: If the file keeps its tables in the layout anndata wrote before 0.7.
        ImportError: If anndata or h5py is not installed.
    """
    try:
        import h5py
        import pandas as pd
        from anndata.io import read_elem
    except ImportError as error:
        raise _scconcept_import_error(error) from error

    file = Path(path)
    if not file.is_file():
        raise AtlasEmbedValueError(f"No atlas at {file}.")
    if layer is not None and use_raw:
        raise AtlasEmbedValueError(
            "The counts are read from one matrix: pass --layer for a layer, or --use-raw for "
            "adata.raw, not both."
        )

    # Read with h5py: anndata reads every layer whole, even in backed mode.
    with h5py.File(file, "r") as handle:
        n_cells, n_genes = _matrix_shape(_matrix_node(handle, layer, file, use_raw), file)
        if n_cells < 1 or n_genes < 1:
            raise AtlasEmbedValueError(
                f"{file} holds {n_cells} cells and {n_genes} genes, so there is nothing to embed."
            )
        obs = handle.get("obs")
        genes_table = handle.get("raw/var" if use_raw else "var")
        if not all(
            isinstance(table, h5py.Group) and "_index" in table.attrs
            for table in (obs, genes_table)
        ):
            raise AtlasEmbedError(
                f"{file} keeps its cell and gene tables in the layout anndata wrote before its 0.7 "
                "release, which this step does not read. Write it again with a recent anndata, as "
                "anndata.read_h5ad(path).write_h5ad(new_path) does, and run this step on the new "
                "file."
            )
        cells = pd.Index(read_elem(obs[_text(obs.attrs["_index"])]))
        var = read_elem(genes_table)
    table = "adata.raw.var" if use_raw else "adata.var"
    var_column, gene_ids, genes, panel = _resolve_gene_ids(
        var, gene_id_column, file, symbols, table
    )

    return Atlas(
        path=file,
        n_cells=n_cells,
        n_genes=n_genes,
        gene_ids=gene_ids,
        obs_names=np.asarray(cells, dtype=str),
        obs=pd.DataFrame(index=cells),
        var=genes,
        var_column=var_column,
        layer=layer,
        panel=panel,
        use_raw=use_raw,
    )


def _resolve_gene_ids(
    var: Any,
    gene_id_column: str,
    file: Path,
    symbols: Mapping[str, Sequence[str]] | None,
    table: str,
) -> tuple[str | None, np.ndarray, Any, dict[str, Any]]:
    """Read the identifiers of a gene table as `read_panel` does, with their column and record."""
    columns = [str(name) for name in var.columns]
    if gene_id_column == VAR_NAMES:
        if VAR_NAMES in columns:
            raise AtlasEmbedValueError(
                f"{file} holds a column named {VAR_NAMES!r} in {table}, which is also the value "
                "that selects the gene index. Rename the column, or name it through another "
                "value of --gene-id-column."
            )
        described = f"{table}_names"
        given = np.asarray(var.index, dtype=str)
        _check_gene_ids(given, described, file)
        identifiers, panel = read_panel(given, f"{described} of {file}", symbols)
        genes = var.iloc[:, :0].copy()
        genes.index = identifiers.astype(object)
        return None, identifiers, genes, panel

    if gene_id_column not in columns:
        raise AtlasEmbedValueError(
            f"{file} holds no column {gene_id_column!r} in {table}. It holds {columns}. Pass "
            f"--gene-id-column {VAR_NAMES} to tokenize the gene index instead."
        )
    values = var[gene_id_column]
    described = f"{table}[{gene_id_column!r}]"
    if bool(values.isna().any()):
        raise AtlasEmbedValueError(
            f"{described} of {file} holds missing values, so those genes carry no identifier the "
            "corpus vocabulary can be looked up with."
        )
    given = np.asarray(values, dtype=str)
    _check_gene_ids(given, described, file)
    identifiers, panel = read_panel(given, f"{described} of {file}", symbols)
    genes = var[[gene_id_column]].copy()
    genes[gene_id_column] = identifiers.astype(object)
    return gene_id_column, identifiers, genes, panel


def _check_gene_ids(identifiers: np.ndarray, described: str, file: Path) -> None:
    """Check that gene identifiers are present and distinct."""
    if np.unique(identifiers).shape[0] != identifiers.shape[0]:
        raise AtlasEmbedValueError(
            f"{described} of {file} repeats a gene identifier. Each column of the matrix must "
            "carry an identifier of its own for the tokenizer to read the panel."
        )
    if bool((identifiers == "").any()):
        raise AtlasEmbedValueError(f"{described} of {file} holds an empty gene identifier.")


def read_matrix_rows(
    path: Path,
    start: int,
    stop: int,
    layer: str | None = None,
    use_raw: bool = False,
) -> np.ndarray:
    """Read a contiguous block of rows of an h5ad expression matrix, densely.

    Args:
        path: Path of the h5ad file.
        start: First row of the block.
        stop: Row the block stops before.
        layer: Layer to read, or None for the main matrix.
        use_raw: Read the matrix of `adata.raw` instead.

    Returns:
        A (stop - start, n_genes) float32 array.

    Raises:
        AtlasEmbedValueError: If the range is outside the matrix.
        AtlasEmbedError: If the matrix is stored in a layout this step does not read.
        ImportError: If h5py is not installed.
    """
    try:
        import h5py
    except ImportError as error:
        raise _scconcept_import_error(error) from error

    with h5py.File(path, "r") as handle:
        node = _matrix_node(handle, layer, path, use_raw)
        n_rows, n_genes = _matrix_shape(node, path)
        if start < 0 or stop > n_rows or start >= stop:
            raise AtlasEmbedValueError(
                f"Rows {start} to {stop} are outside the {n_rows} rows of {path}."
            )
        if isinstance(node, h5py.Dataset):
            return np.asarray(node[start:stop, :], dtype=np.float32)
        return _read_sparse_rows(node, start, stop, n_rows, n_genes, path)


def _matrix_node(handle: Any, layer: str | None, path: Path, use_raw: bool = False) -> Any:
    """Return the HDF5 node holding the expression matrix."""
    if use_raw:
        if "raw" not in handle or "X" not in handle["raw"]:
            raise AtlasEmbedValueError(
                f"{path} holds no adata.raw. Drop --use-raw, and pass --layer if the raw counts "
                "live in a layer."
            )
        return handle["raw"]["X"]
    if layer is None:
        if "X" not in handle:
            raise AtlasEmbedError(f"{path} holds no expression matrix at X.")
        return handle["X"]
    layers = handle.get("layers")
    if layers is None or layer not in layers:
        held = [] if layers is None else sorted(layers)
        raise AtlasEmbedValueError(f"{path} holds no layer {layer!r}. Its layers are {held}.")
    return layers[layer]


def _matrix_shape(node: Any, path: Path) -> tuple[int, int]:
    """Return the two-dimensional shape of an expression matrix node."""
    shape = getattr(node, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if "shape" in node.attrs:
        stored = node.attrs["shape"]
        return int(stored[0]), int(stored[1])
    if "shape" in node:
        stored = node["shape"][()]
        return int(stored[0]), int(stored[1])
    raise AtlasEmbedError(f"The expression matrix of {path} declares no shape.")


def _text(value: Any) -> str:
    """Return an HDF5 attribute as a string, whichever way the writer stored it."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_sparse_rows(
    node: Any,
    start: int,
    stop: int,
    n_rows: int,
    n_genes: int,
    path: Path,
) -> np.ndarray:
    """Materialize a row block of a compressed sparse matrix stored row by row."""
    missing = {"data", "indices", "indptr"}.difference(node.keys())
    if missing:
        raise AtlasEmbedError(
            f"The expression matrix of {path} is a group missing {sorted(missing)}, so it is "
            "neither a dense array nor a compressed sparse matrix."
        )
    encoding = _text(node.attrs.get("encoding-type"))
    if not encoding:
        encoding = "csr_matrix" if int(node["indptr"].shape[0]) == n_rows + 1 else "csc_matrix"
    if encoding not in {"csr_matrix", "csr_array"}:
        raise AtlasEmbedError(
            f"The expression matrix of {path} is stored as {encoding!r}, which is compressed by "
            "gene rather than by cell, so a block of cells cannot be read from it in one pass. "
            "Write the atlas with a matrix compressed by cell, which is what "
            "`adata.X = adata.X.tocsr()` produces, and run this step again."
        )
    offsets = np.asarray(node["indptr"][start : stop + 1], dtype=np.int64)
    first = int(offsets[0])
    last = int(offsets[-1])
    columns = np.asarray(node["indices"][first:last], dtype=np.int64)
    values = np.asarray(node["data"][first:last], dtype=np.float32)
    dense = np.zeros((stop - start, n_genes), dtype=np.float32)
    if columns.size:
        rows = np.repeat(np.arange(stop - start, dtype=np.int64), np.diff(offsets))
        # A coordinate stored twice sums its entries; indexed assignment would keep only the last.
        np.add.at(dense, (rows, columns), values)
    return dense


# --- The model --------------------------------------------------------------------------------


def _scconcept_import_error(error: ImportError) -> ImportError:
    """Return the error to raise when this environment is not the one that runs this step."""
    return ImportError(
        f"{error}. This step runs in its own environment, declared in {ENVIRONMENT_FILE}: "
        f"`python3.12 -m venv {ENVIRONMENT_NAME}`, `source {ENVIRONMENT_NAME}/bin/activate`, "
        f"`pip install -r {ENVIRONMENT_FILE}`, then run this script from it."
    )


def resolve_device(name: str | None = None) -> Any:
    """Return the device the model runs on.

    Args:
        name: A device name, or None for CUDA where available, the processor otherwise.

    Returns:
        A `torch.device`.

    Raises:
        ImportError: If torch is not installed.
    """
    try:
        import torch
    except ImportError as error:
        raise _scconcept_import_error(error) from error
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device: Any) -> Any:
    """Return the precision context the released references were embedded under."""
    import torch

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def pretrained_vocabulary_available(model_dir: str | Path) -> bool:
    """Report whether the model assets ship the vocabulary a checkpoint may be loaded against.

    Args:
        model_dir: Directory holding the model assets.

    Returns:
        Whether a pretrained vocabulary directory holding at least one table is present.
    """
    vocabulary = Path(model_dir) / PRETRAINED_VOCABULARY_DIRNAME
    return vocabulary.is_dir() and any(vocabulary.glob("*.csv"))


def load_model(model_dir: Path, species: str, device: Any, pretrained_vocabulary: bool) -> Any:
    """Load the pretrained single-cell model from its downloaded assets, ready for inference.

    No adaptation: the released teacher took none, so the pretrained weights define the cell space.

    Args:
        model_dir: Directory holding `config.yaml`, `model.ckpt`, `gene_mappings/` and `panels/`.
        species: Species key that must be in the tokenizer vocabulary.
        device: Device the model is moved to.
        pretrained_vocabulary: Load the checkpoint against the pretrained vocabulary beside it,
            which changes the tokens the model reads; recorded in the output.

    Returns:
        The loaded model, in evaluation mode with its weights frozen.

    Raises:
        AtlasEmbedError: If an asset is missing, the tokenizer is not the multi-species one, or the
            species is not one the model knows.
        ImportError: If sc-concept is not installed.
    """
    try:
        from concept import scConcept
    except ImportError as error:
        raise _scconcept_import_error(error) from error

    assets = {
        "config.yaml": model_dir / "config.yaml",
        "model.ckpt": model_dir / "model.ckpt",
        "gene_mappings": model_dir / "gene_mappings",
        "panels": model_dir / "panels",
    }
    missing = sorted(name for name, item in assets.items() if not item.exists())
    if missing:
        raise AtlasEmbedError(
            f"{model_dir} is missing {missing}. Place the single-cell model assets, as "
            f"sc-concept distributes them, under a directory named after the model, so that "
            f"{model_dir.name}/config.yaml and {model_dir.name}/model.ckpt resolve, and pass "
            "--model-cache the directory holding it."
        )

    # The checkpoint's vocabulary loads exactly as trained, or the strict load fails.
    vocabulary = model_dir / PRETRAINED_VOCABULARY_DIRNAME
    model = scConcept(cache_dir=str(model_dir.parent))
    model.load_config_and_model(
        config=str(assets["config.yaml"]),
        model_path=str(assets["model.ckpt"]),
        gene_mappings_path=str(assets["gene_mappings"]),
        panels_dir=str(assets["panels"]),
        pretrained_vocabulary_path=str(vocabulary) if pretrained_vocabulary else None,
    )
    check_tokenizer(getattr(model, "tokenizer", None))
    known = list(model.tokenizer.species)
    if species not in known:
        raise AtlasEmbedError(f"The model does not know the species {species!r}. It knows {known}.")

    model.model = model.model.to(device)
    model.device = next(model.model.parameters()).device
    model.model.eval()
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    return model


def check_token_budget(model: Any, max_tokens: int) -> None:
    """Check the token budget against the positional encoding the model was built with.

    Args:
        model: The loaded model.
        max_tokens: Gene tokens read per cell.

    Raises:
        AtlasEmbedValueError: If the budget does not fit the positional encoding.
    """
    limit = int(model.cfg.model.pe_max_len)
    if max_tokens >= limit:
        raise AtlasEmbedValueError(
            f"A token budget of {max_tokens} does not fit the {limit} positions the model encodes. "
            f"Pass --max-tokens below {limit}."
        )


# --- Inference --------------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceSettings:
    """How the model is run over the atlas.

    Attributes:
        species: Species key routing the per-species vocabulary.
        var_column: Column the gene identifiers come from, or None for the gene index.
        max_tokens: Gene tokens read per cell.
        gene_sampling_strategy: Rule filling the token budget.
        batch_size: Cells per forward pass.
        num_workers: Processes tokenizing cells alongside the forward pass.
        dtype: Precision the embeddings are stored at.
    """

    species: str
    var_column: str | None
    max_tokens: int
    gene_sampling_strategy: str
    batch_size: int
    num_workers: int
    dtype: str


def _embed_chunk(
    model: Any,
    chunk: Any,
    settings: InferenceSettings,
    device: Any,
    on_batch: Any,
) -> int:
    """Run the model over a chunk in row order, giving `on_batch` each batch as a NumPy array."""
    try:
        import torch
        from concept.dataset import MultiSpeciesTokenizedDataset
        from lamin_dataloader import BaseCollate, InMemoryCollection
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise _scconcept_import_error(error) from error

    collection = InMemoryCollection([chunk], var_column=settings.var_column)
    dataset = MultiSpeciesTokenizedDataset(
        metadata={"species": [settings.species]},
        collection=collection,
        tokenizer=model.tokenizer,
        normalization=model.cfg.datamodule.normalization,
    )
    collate = BaseCollate(
        model.tokenizer.PAD_TOKEN,
        max_tokens=int(settings.max_tokens),
        gene_sampling_strategy=str(settings.gene_sampling_strategy),
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=False,
    )
    stored = torch.float16 if settings.dtype == "float16" else torch.float32

    written = 0
    with torch.inference_mode(), _autocast(device):
        for index, batch in enumerate(loader):
            moved = {
                key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
                for key, value in batch.items()
            }
            output = model.model.predict_step(moved, index)
            if CLS_KEY not in output:
                raise AtlasEmbedError(
                    f"The model returned {sorted(output)} and no {CLS_KEY!r}, so this is not the "
                    "single-cell model this step reads."
                )
            block = output[CLS_KEY].to(device="cpu", dtype=stored).numpy()
            on_batch(written, block)
            written += int(block.shape[0])
    return written


# --- Embedding the atlas ----------------------------------------------------------------------


def embed_atlas(
    atlas: str | Path,
    output: str | Path,
    *,
    model_cache: str | Path,
    model_name: str = MODEL_NAME,
    species: str = SPECIES,
    gene_id_column: str = DEFAULT_GENE_ID_COLUMN,
    gene_id_type: str = GENE_ID_TYPES[0],
    layer: str | None = None,
    use_raw: bool = False,
    max_tokens: int = MAX_TOKENS,
    gene_sampling_strategy: str = GENE_SAMPLING_STRATEGY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    chunk_memory_gb: float = DEFAULT_CHUNK_MEMORY_GB,
    dtype: str = "float16",
    device: str | None = None,
    dataset: str = DEFAULT_DATASET,
    min_vocabulary_fraction: float = DEFAULT_MIN_VOCABULARY_FRACTION,
    min_genes_in_vocabulary: int = DEFAULT_MIN_GENES_IN_VOCABULARY,
    count_check_cells: int = DEFAULT_COUNT_CHECK_CELLS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Embed every cell of an atlas and write the embeddings in the row order of the atlas.

    Every cell is written, labeled or not. The atlas, its gene identifiers, the raw counts and the
    output path are checked before the model loads, the vocabulary overlap once the tokenizer
    exists, and the width on the first batch. On CUDA the model runs under bfloat16 autocast, as the
    released references were embedded; elsewhere in float32, which differs slightly.

    Args:
        atlas: Path of the h5ad file holding the cells.
        output: Path of the HDF5 file to write.
        model_cache: Directory holding the model assets under a directory named after the model.
        model_name: Single-cell model to run.
        species: Species key of the tokenizer vocabulary.
        gene_id_column: Column of the gene table holding the gene identifiers, or `var_names`.
        gene_id_type: What that column holds, one of `GENE_ID_TYPES`: `"ensembl"` for Ensembl gene
            identifiers, with or without their version, or `"symbol"` for gene symbols.
        layer: Layer holding the raw counts, or None for the main matrix.
        use_raw: Read the counts and the gene table of `adata.raw`, where CELLxGENE files keep
            the raw counts.
        max_tokens: Gene tokens read per cell.
        gene_sampling_strategy: Rule filling the token budget.
        batch_size: Cells per forward pass.
        num_workers: Processes tokenizing cells alongside the forward pass.
        chunk_memory_gb: Gigabytes one materialized block of the matrix may occupy.
        dtype: Precision the embeddings are stored at.
        device: Device to run on, or None to take CUDA where it is available.
        dataset: Name of the dataset the embeddings are written to.
        min_vocabulary_fraction: Fraction of the panel that must be in the corpus vocabulary.
        min_genes_in_vocabulary: Genes of the panel that must be in the corpus vocabulary.
        count_check_cells: Cells sampled to check that the matrix holds raw counts.
        overwrite: Replace a file already at `output`.

    Returns:
        What was written, the settings it was written under, and the realized panel overlap.

    Raises:
        AtlasEmbedValueError: If an argument or an input is outside the range this step accepts.
        AtlasEmbedError: If the model assets or the expression matrix cannot be read.
        ImportError: If this environment does not carry the single-cell model.
    """
    if dtype not in STORAGE_DTYPES:
        raise AtlasEmbedValueError(
            f"Embeddings are stored as {sorted(STORAGE_DTYPES)}, not {dtype!r}."
        )
    if batch_size < 1:
        raise AtlasEmbedValueError(f"The batch size must be at least one, got {batch_size}.")
    if num_workers < 0:
        raise AtlasEmbedValueError(f"The worker count cannot be negative, got {num_workers}.")
    if max_tokens < 1:
        raise AtlasEmbedValueError(f"The token budget must be at least one, got {max_tokens}.")
    if chunk_memory_gb <= 0.0:
        raise AtlasEmbedValueError(
            f"The chunk memory budget must be positive, got {chunk_memory_gb}."
        )
    if count_check_cells < 1:
        raise AtlasEmbedValueError(
            f"At least one cell must be checked for raw counts, got {count_check_cells}."
        )
    if not 0.0 <= min_vocabulary_fraction <= 1.0:
        raise AtlasEmbedValueError(
            "The vocabulary fraction is a fraction of the panel, so it lies between zero and "
            f"one, got {min_vocabulary_fraction}."
        )
    if min_genes_in_vocabulary < 0:
        raise AtlasEmbedValueError(
            f"The gene floor cannot be negative, got {min_genes_in_vocabulary}."
        )
    if not str(dataset).strip():
        raise AtlasEmbedValueError(
            "The name of the dataset the embeddings are written to is empty."
        )
    if dataset == OBS_NAME_DATASET:
        raise AtlasEmbedValueError(
            f"The cell names are written to the dataset {OBS_NAME_DATASET!r}. Name the dataset "
            "of the embeddings otherwise."
        )
    if gene_id_type not in GENE_ID_TYPES:
        raise AtlasEmbedValueError(
            f"The gene identifier column is read as one of {list(GENE_ID_TYPES)}, not "
            f"{gene_id_type!r}."
        )

    started = time.monotonic()
    target, partial = _prepare_output(output, overwrite)
    model_dir = Path(model_cache) / model_name
    symbols = None
    if gene_id_type == GENE_ID_TYPES[1]:
        symbols = read_gene_table(model_dir / GENE_MAPPINGS_DIRNAME / f"{species}.csv")
    header = open_atlas(
        atlas, gene_id_column=gene_id_column, layer=layer, symbols=symbols, use_raw=use_raw
    )
    _report(f"[atlas]   {header.path}")
    _report(f"[cells]   {header.n_cells} cells, {header.n_genes} genes")
    _report(_genes_line(header.panel, header.n_genes))
    unique_names = check_obs_names(header.obs_names)

    counts = _check_atlas_counts(header, count_check_cells, chunk_memory_gb)
    _report(
        f"[counts]  raw counts on {int(counts['n_cells_checked'])} sampled cells, values "
        f"{counts['minimum']:g} to {counts['maximum']:g}"
    )

    # sc-concept's download may bring a vocabulary, which the released cell space does not load.
    shipped = pretrained_vocabulary_available(model_dir)
    uses_vocabulary = shipped and PRETRAINED_VOCABULARY
    package_version = _installed_version(SCCONCEPT_DISTRIBUTION)
    check_released_settings(
        model_name,
        species,
        max_tokens,
        gene_sampling_strategy,
        uses_vocabulary,
        package_version,
    )
    unit = resolve_device(device)
    _configure_device(unit)
    _report(
        f"[model]   {model_name}, species {species}, {ADAPTATION_STEPS} adaptation steps, "
        f"pretrained vocabulary {vocabulary_state(uses_vocabulary)}"
        + (", the one in the model folder left aside" if shipped and not uses_vocabulary else "")
    )
    model = load_model(model_dir, species, unit, uses_vocabulary)
    check_token_budget(model, max_tokens)
    precision = "bfloat16 autocast" if unit.type == "cuda" else "float32"
    _report(f"[device]  {unit}, {precision}")

    overlap = vocabulary_overlap(header.gene_ids, model.tokenizer, species)
    check_vocabulary_overlap(overlap, min_vocabulary_fraction, min_genes_in_vocabulary)
    _report(
        f"[panel]   {int(overlap['n_genes_in_vocabulary'])} of {int(overlap['n_genes'])} genes in "
        f"the corpus vocabulary ({100.0 * overlap['fraction']:.1f}%)"
    )

    settings = InferenceSettings(
        species=species,
        var_column=header.var_column,
        max_tokens=int(max_tokens),
        gene_sampling_strategy=str(gene_sampling_strategy),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        dtype=dtype,
    )
    rows = rows_per_chunk(header.n_genes, chunk_memory_gb)
    bounds = chunk_bounds(header.n_cells, rows)
    _report(f"[chunk]   {rows} cells per pass, {len(bounds)} passes")

    attributes = _provenance(
        header,
        model,
        model_name,
        settings,
        overlap,
        unit,
        precision,
        pretrained_vocabulary=uses_vocabulary,
        package_version=package_version,
        obs_names_unique=unique_names,
    )
    try:
        written, width = _stream(header, model, settings, bounds, partial, dataset, attributes)
        partial.replace(target)
    finally:
        # Gone once replaced; a run that fails leaves no half-written file behind.
        partial.unlink(missing_ok=True)

    elapsed = (time.monotonic() - started) / 60.0
    _report(f"[written] {target}, {written} by {width} {dtype}, {elapsed:.1f} min")
    _report(
        "[next]    build a shard from it with alhena.refbuild.build_reference, giving one label "
        f"per cell in the row order of {header.path.name}"
    )
    return {
        "output": str(target),
        "n_cells": written,
        "embedding_dim": width,
        "attributes": attributes,
        "vocabulary_overlap": overlap,
        "counts": counts,
        "minutes": elapsed,
    }


def _prepare_output(output: str | Path, overwrite: bool) -> tuple[Path, Path]:
    """Return the output path and its `.partial`, both proven writable."""
    target = Path(output)
    if target.is_dir():
        raise AtlasEmbedValueError(f"{target} is a directory. Pass the path of the file to write.")
    if target.exists() and not overwrite:
        raise AtlasEmbedValueError(
            f"{target} already holds embeddings. Pass --overwrite to replace them, or name "
            "another file."
        )
    partial = target.with_name(target.name + ".partial")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("wb"):
            pass
        partial.unlink()
    except OSError as error:
        raise AtlasEmbedValueError(f"Nothing can be written at {target}: {error}") from error
    return target, partial


def _check_atlas_counts(
    header: Atlas,
    count_check_cells: int,
    chunk_memory_gb: float = DEFAULT_CHUNK_MEMORY_GB,
) -> dict[str, float]:
    """Check raw counts over blocks spread across the atlas, read within the memory budget."""
    read = "raw.X" if header.use_raw else f"layers[{header.layer!r}]" if header.layer else "X"
    described = f"adata.{read} of {header.path}"
    rows = rows_per_chunk(header.n_genes, chunk_memory_gb)
    smallest = float("inf")
    largest = 0.0
    values = 0.0
    nonzero = 0.0
    checked = 0
    for start, stop in sample_blocks(header.n_cells, count_check_cells):
        block_largest = 0.0
        for lower, upper in chunk_bounds(stop - start, rows):
            block = read_matrix_rows(
                header.path, start + lower, start + upper, header.layer, header.use_raw
            )
            seen = check_raw_counts(block, described)
            smallest = min(smallest, seen["minimum"])
            largest = max(largest, seen["maximum"])
            block_largest = max(block_largest, seen["maximum"])
            values += seen["n_values"]
            nonzero += seen["nonzero_fraction"] * seen["n_values"]
            checked += upper - lower
        if block_largest == 0.0:
            raise AtlasEmbedValueError(
                f"{described} is entirely zero over cells {start + 1} to {stop}, so those cells "
                f"carry no expression at all. {_RAW_COUNTS_ADVICE}"
            )
    return {
        "n_cells_checked": float(checked),
        "n_values": values,
        "minimum": smallest,
        "maximum": largest,
        "nonzero_fraction": nonzero / values if values else 0.0,
    }


def _configure_device(device: Any) -> None:
    """Set the matrix-multiplication precision the released references were embedded under."""
    import torch

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def _installed_version(distribution: str) -> str:
    """Return a distribution's installed release, or `UNKNOWN_VERSION` where it cannot be read."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return str(version(distribution))
    except PackageNotFoundError:
        return UNKNOWN_VERSION


def _genes_line(panel: Mapping[str, Any], n_genes: int) -> str:
    """Return the progress line saying how the gene identifiers of the atlas were read."""
    if panel["gene_id_type"] == GENE_ID_TYPES[1]:
        mapped = panel["n_gene_symbols_mapped"]
        parts = [f"{n_genes} gene symbols, {mapped} mapped to Ensembl gene identifiers"]
        left = [
            f"{panel[key]} {what}"
            for key, what in (
                ("n_gene_symbols_ambiguous", "naming several genes"),
                ("n_gene_symbols_unmatched", "absent from the gene table"),
            )
            if panel[key]
        ]
        if left:
            parts.append(f"{' and '.join(left)} left as they are")
    else:
        parts = [f"{n_genes} Ensembl gene identifiers"]
    if panel["n_gene_ids_normalized"]:
        parts.append(f"{panel['n_gene_ids_normalized']} read without their version")
    if panel["n_gene_ids_par_y"]:
        parts.append(f"{panel['n_gene_ids_par_y']} _PAR_Y copies left unread")
    return "[genes]   " + ", ".join(parts)


def _provenance(
    header: Atlas,
    model: Any,
    model_name: str,
    settings: InferenceSettings,
    overlap: dict[str, float],
    device: Any,
    precision: str,
    *,
    pretrained_vocabulary: bool,
    package_version: str,
    obs_names_unique: bool,
) -> dict[str, Any]:
    """Assemble what the output file records about how it was produced, gene reading included."""
    return {
        "model_name": str(getattr(model, "model_name", "") or model_name),
        "scconcept_version": SCCONCEPT_VERSION,
        "scconcept_package_version": package_version,
        "species": settings.species,
        "adaptation_steps": ADAPTATION_STEPS,
        "pretrained_vocabulary_used": bool(pretrained_vocabulary),
        "obs_names_unique": bool(obs_names_unique),
        "source": str(header.path.resolve()),
        "layer": "raw.X" if header.use_raw else header.layer or "X",
        "gene_id_column": header.var_column or VAR_NAMES,
        **header.panel,
        "max_tokens": settings.max_tokens,
        "gene_sampling_strategy": settings.gene_sampling_strategy,
        "normalization": str(model.cfg.datamodule.normalization),
        "embedding_dtype": settings.dtype,
        "n_cells": header.n_cells,
        "n_genes_panel": int(overlap["n_genes"]),
        "n_genes_in_vocabulary": int(overlap["n_genes_in_vocabulary"]),
        "vocabulary_fraction": float(overlap["fraction"]),
        "row_order": ROW_ORDER,
        "device": str(device),
        "precision": precision,
    }


def _stream(
    header: Atlas,
    model: Any,
    settings: InferenceSettings,
    bounds: Sequence[tuple[int, int]],
    partial: Path,
    dataset: str,
    attributes: dict[str, Any],
) -> tuple[int, int]:
    """Embed chunk by chunk, writing each batch at its atlas row."""
    try:
        import anndata
        import h5py
    except ImportError as error:
        raise _scconcept_import_error(error) from error

    stored = STORAGE_DTYPES[settings.dtype]
    written = 0
    width = 0

    with h5py.File(partial, "w") as handle:
        for key, value in attributes.items():
            handle.attrs[key] = value
        names = handle.create_dataset(
            OBS_NAME_DATASET,
            data=np.asarray(header.obs_names, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        names.attrs["role"] = OBS_NAME_ROLE
        names.attrs["description"] = (
            "Cell names of the source atlas, in row order. Recorded so a reader can recognize the "
            "atlas the embeddings came from. The reference build matches labels to cells by "
            "position, so this is never a join key and the names need not be distinct."
        )
        matrix: Any = None

        began = time.monotonic()
        for number, (start, stop) in enumerate(bounds, start=1):
            _report(
                pass_line(
                    number,
                    len(bounds),
                    (start, stop),
                    written,
                    header.n_cells,
                    time.monotonic() - began,
                )
            )
            values = read_matrix_rows(header.path, start, stop, header.layer, header.use_raw)
            chunk = anndata.AnnData(
                X=values,
                obs=header.obs.iloc[start:stop].copy(),
                var=header.var.copy(),
            )

            # `base` is this chunk's first row, so a batch offset becomes the cell's atlas row.
            def _write(offset: int, block: np.ndarray, base: int = start) -> None:
                nonlocal matrix, width
                check_embedding_width(int(block.shape[1]))
                if matrix is None:
                    width = int(block.shape[1])
                    matrix = handle.create_dataset(
                        dataset,
                        shape=(header.n_cells, width),
                        dtype=stored,
                        chunks=(min(STORAGE_CHUNK_ROWS, header.n_cells), width),
                        compression="gzip",
                        compression_opts=STORAGE_COMPRESSION,
                        shuffle=True,
                    )
                    handle.attrs["embedding_dim"] = width
                matrix[base + offset : base + offset + block.shape[0]] = block

            produced = _embed_chunk(model, chunk, settings, model.device, _write)
            if produced != stop - start:
                raise AtlasEmbedError(
                    f"The model returned {produced} embeddings for the {stop - start} cells of "
                    f"rows {start} to {stop}, so the output would not be in the row order of the "
                    "atlas."
                )
            written += produced
            del chunk, values

        if written != header.n_cells:
            raise AtlasEmbedError(
                f"{written} of the {header.n_cells} cells of {header.path} were embedded, so the "
                "output does not cover the atlas."
            )
    return written, width


# --- Command line -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of this step."""
    parser = argparse.ArgumentParser(
        prog="alhena_atlas_embed.py",
        description=(
            "Embed a single-cell atlas into the cell space ALHENA retrieves against. The output "
            "feeds alhena.refbuild.build_reference, which runs in the package environment and "
            "matches labels to cells by position, so the row order of the atlas is kept."
        ),
    )
    parser.add_argument(
        "--atlas", type=Path, required=True, help="h5ad file holding the cells, with raw counts"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="HDF5 file the embeddings are written to"
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path.home() / ".cache" / "sc-concept",
        help="directory holding the model assets under a directory named after the model",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME, help="single-cell model the atlas is embedded with"
    )
    parser.add_argument(
        "--species", default=SPECIES, help="species key of the tokenizer vocabulary"
    )
    parser.add_argument(
        "--gene-id-column",
        default=DEFAULT_GENE_ID_COLUMN,
        help=(
            "column of adata.var, or of adata.raw.var with --use-raw, holding the gene "
            f"identifiers; {VAR_NAMES} to tokenize the gene index"
        ),
    )
    parser.add_argument(
        "--gene-id-type",
        default=GENE_ID_TYPES[0],
        choices=GENE_ID_TYPES,
        help=(
            "what that column holds: ensembl, Ensembl gene identifiers, with or without a version; "
            "or symbol, gene symbols, mapped through the model's "
            f"{GENE_MAPPINGS_DIRNAME}/<species>.csv where it gives a symbol one identifier"
        ),
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="layer holding the raw counts, where adata.X holds normalized values instead",
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help=(
            "read the counts and the gene table of adata.raw, where CELLxGENE files keep the raw "
            "counts"
        ),
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS, help="gene tokens read per cell"
    )
    parser.add_argument(
        "--gene-sampling-strategy",
        default=GENE_SAMPLING_STRATEGY,
        help="rule selecting which genes of a panel fill the token budget",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="cells per forward pass"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="processes tokenizing cells alongside the forward pass",
    )
    parser.add_argument(
        "--chunk-memory-gb",
        type=float,
        default=DEFAULT_CHUNK_MEMORY_GB,
        help="gigabytes one materialized block of the expression matrix may occupy",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=sorted(STORAGE_DTYPES),
        help="precision the embeddings are stored at",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device to run on, cuda or cpu; cuda where available, the processor otherwise",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="dataset the embeddings are written to, which the package reads by name",
    )
    parser.add_argument(
        "--min-vocabulary-fraction",
        type=float,
        default=DEFAULT_MIN_VOCABULARY_FRACTION,
        help="fraction of the panel that must be in the corpus vocabulary for the run to proceed",
    )
    parser.add_argument(
        "--min-genes-in-vocabulary",
        type=int,
        default=DEFAULT_MIN_GENES_IN_VOCABULARY,
        help="genes of the panel that must be in the corpus vocabulary for the run to proceed",
    )
    parser.add_argument(
        "--count-check-cells",
        type=int,
        default=DEFAULT_COUNT_CHECK_CELLS,
        help="cells sampled to check that the expression matrix holds raw counts",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace a file already at the output path"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="write a one-line JSON summary to standard output; progress goes to standard error",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the step and report any failure as one line rather than as a traceback.

    Args:
        argv: Command line arguments, or None to read them from the process.

    Returns:
        Zero when the atlas was embedded, one otherwise.
    """
    arguments = build_parser().parse_args(argv)
    try:
        summary = embed_atlas(
            arguments.atlas,
            arguments.output,
            model_cache=arguments.model_cache,
            model_name=arguments.model,
            species=arguments.species,
            gene_id_column=arguments.gene_id_column,
            gene_id_type=arguments.gene_id_type,
            layer=arguments.layer,
            use_raw=arguments.use_raw,
            max_tokens=arguments.max_tokens,
            gene_sampling_strategy=arguments.gene_sampling_strategy,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            chunk_memory_gb=arguments.chunk_memory_gb,
            dtype=arguments.dtype,
            device=arguments.device,
            dataset=arguments.dataset,
            min_vocabulary_fraction=arguments.min_vocabulary_fraction,
            min_genes_in_vocabulary=arguments.min_genes_in_vocabulary,
            count_check_cells=arguments.count_check_cells,
            overwrite=arguments.overwrite,
        )
    except (AtlasEmbedError, ImportError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    if arguments.json:
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
