"""The `alhena` command line.

Only the commands and `alhena.demo` print results, on standard output; progress goes to standard
error. Every command returns an exit code and reports a package error in one line.
"""

from alhena._wording import PROGRAM
from alhena.cli._annotate import TRUTH_COLUMNS
from alhena.cli._parser import build_parser, main
from alhena.cli._reading import NUCLEI_ID_SCOPES
from alhena.cli._shared import (
    CALL_SUFFIXES,
    CELLS_SUFFIXES,
    EMBEDDING_SUFFIXES,
    EMBEDDINGS_DATASET,
    GEOMETRY_SUFFIX,
    geometry_path,
)

__all__ = [
    "CALL_SUFFIXES",
    "CELLS_SUFFIXES",
    "EMBEDDINGS_DATASET",
    "EMBEDDING_SUFFIXES",
    "GEOMETRY_SUFFIX",
    "NUCLEI_ID_SCOPES",
    "PROGRAM",
    "TRUTH_COLUMNS",
    "build_parser",
    "geometry_path",
    "main",
]
