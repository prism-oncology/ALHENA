"""Read ALHENA's inputs from a SpatialData object, and write its results back.

`tile_embeddings` runs the encoder inside sopa's patch pipeline.
"""

from alhena.spatial._cells import cells_to_spatialdata
from alhena.spatial._elements import CELL_TABLE_KEY, INSTANCE_KEY, REGION_KEY
from alhena.spatial._nuclei import cells_from_shapes
from alhena.spatial._sopa import TILE_EMBEDDINGS_KEY, Alhena, tile_embeddings
from alhena.spatial._tiles import (
    DEFAULT_COORDINATE_SYSTEM,
    TILE_SHAPES_KEY,
    TILE_TABLE_KEY,
    tile_source,
    tiles_to_spatialdata,
)
from alhena.spatial._windows import PATCHES_KEY, patch_bounds

__all__ = [
    "CELL_TABLE_KEY",
    "DEFAULT_COORDINATE_SYSTEM",
    "INSTANCE_KEY",
    "PATCHES_KEY",
    "REGION_KEY",
    "TILE_EMBEDDINGS_KEY",
    "TILE_SHAPES_KEY",
    "TILE_TABLE_KEY",
    "Alhena",
    "cells_from_shapes",
    "cells_to_spatialdata",
    "patch_bounds",
    "tile_embeddings",
    "tile_source",
    "tiles_to_spatialdata",
]
