"""Pool encoder tokens into one embedding per cell, under the geometry the image was read at.

Nuclei become a membership table of tokens and pixel counts, which `pool_cells` averages over.
"""

from alhena._frames import ORIGINS_FRAME
from alhena.cells._canvas import fit_mask, token_index_map
from alhena.cells._centroids import membership_from_centroids
from alhena.cells._labels import membership_from_labels
from alhena.cells._polygons import POLYGONS_FRAME, membership_from_polygons
from alhena.cells._pooling import GROUP_COLUMNS, pool_cells, pool_tokens, resolve_main_tile
from alhena.cells._table import (
    membership_is_approximate,
    membership_source,
    missing_cells,
    validate_membership,
)

__all__ = [
    "GROUP_COLUMNS",
    "ORIGINS_FRAME",
    "POLYGONS_FRAME",
    "fit_mask",
    "membership_from_centroids",
    "membership_from_labels",
    "membership_from_polygons",
    "membership_is_approximate",
    "membership_source",
    "missing_cells",
    "pool_cells",
    "pool_tokens",
    "resolve_main_tile",
    "token_index_map",
    "validate_membership",
]
