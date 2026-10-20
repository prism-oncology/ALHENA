"""ALHENA, an H&E encoder distilled from transcriptomic foundation models
for multi-scale slide analysis.
"""

from __future__ import annotations

# Loaded with the package so the readers are reached as alhena.io; it imports nothing new.
from alhena import io as io
from alhena._constants import (
    BACKBONE_NATIVE_MPP,
    CLASS_AXES,
    CLASS_COLORS,
    COMPARTMENT_V1,
    DEFAULT_AXIS,
    DEFAULT_K,
    DEFAULT_TAU,
    INPUT_SIZE,
    LINEAGE_V1,
    PATCH_SIZE,
    REFERENCE_MPP,
    RETRIEVAL_DEPTH,
    TISSUE_ALIASES,
    TISSUE_TO_SHARD,
    label_order,
)
from alhena._errors import (
    AlhenaError,
    AlhenaValueError,
    AlhenaWarning,
    BackboneNotFoundError,
    BundleError,
    CellSpaceMismatchError,
    EnvelopeWarning,
    GeometryError,
    GeometryWarning,
    MembershipError,
    MissingExtraError,
    ReferenceCoverageWarning,
    ReferenceError,
)
from alhena._version import __version__ as __version__
from alhena.annotate import annotate_cells, retrieve, vote
from alhena.api import CellTyper, load_celltyper, load_encoder
from alhena.cells import (
    membership_from_centroids,
    membership_from_labels,
    membership_from_polygons,
    pool_cells,
    validate_membership,
)
from alhena.encoder import AlhenaEncoder, load_bundle
from alhena.metrics import align_truth, chance_floor, confidence_sweep, score
from alhena.preprocess import describe_geometry, prepare_tiles
from alhena.refbuild import (
    ShardBuildReport,
    build_reference,
    is_user_built,
    read_scconcept_embeddings,
)
from alhena.reference import (
    Reference,
    available_shards,
    available_tissues,
    describe_shard,
    load_reference,
    shard_for_tissue,
)
from alhena.types import (
    AnnotationResult,
    BundleInfo,
    CellEmbeddings,
    GeometryReport,
    ReferenceInfo,
    ResampleInfo,
    ScoreReport,
    SelfCheckReport,
    TileEmbeddings,
)

__all__ = [
    "BACKBONE_NATIVE_MPP",
    "CLASS_AXES",
    "CLASS_COLORS",
    "COMPARTMENT_V1",
    "DEFAULT_AXIS",
    "DEFAULT_K",
    "DEFAULT_TAU",
    "INPUT_SIZE",
    "LINEAGE_V1",
    "PATCH_SIZE",
    "REFERENCE_MPP",
    "RETRIEVAL_DEPTH",
    "TISSUE_ALIASES",
    "TISSUE_TO_SHARD",
    "AlhenaEncoder",
    "AlhenaError",
    "AlhenaValueError",
    "AlhenaWarning",
    "AnnotationResult",
    "BackboneNotFoundError",
    "BundleError",
    "BundleInfo",
    "CellEmbeddings",
    "CellSpaceMismatchError",
    "CellTyper",
    "EnvelopeWarning",
    "GeometryError",
    "GeometryReport",
    "GeometryWarning",
    "MembershipError",
    "MissingExtraError",
    "Reference",
    "ReferenceCoverageWarning",
    "ReferenceError",
    "ReferenceInfo",
    "ResampleInfo",
    "ScoreReport",
    "SelfCheckReport",
    "ShardBuildReport",
    "TileEmbeddings",
    "__version__",
    "align_truth",
    "annotate_cells",
    "available_shards",
    "available_tissues",
    "build_reference",
    "chance_floor",
    "confidence_sweep",
    "describe_geometry",
    "describe_shard",
    "is_user_built",
    "label_order",
    "load_bundle",
    "load_celltyper",
    "load_encoder",
    "load_reference",
    "membership_from_centroids",
    "membership_from_labels",
    "membership_from_polygons",
    "pool_cells",
    "prepare_tiles",
    "read_scconcept_embeddings",
    "retrieve",
    "score",
    "shard_for_tissue",
    "validate_membership",
    "vote",
]
