"""Build the adapted backbone from the UNI2-h weights, and read embeddings out of it.

The geometry is the package's, and a bundle stating another is refused; the adapter rank and scaling
and the cell space are the bundle's.
"""

from alhena._bundle import bundle_identity
from alhena._defaults import DEFAULT_BATCH_SIZE
from alhena.encoder._adapters import (
    ADAPTER_NAMES,
    LoRAQKV,
    adapter_key,
    insert_adapters,
    load_adapter_weights,
    merge_adapters,
)
from alhena.encoder._inference import TILE_EMBEDDING_CONTRACT, AlhenaEncoder
from alhena.encoder._loading import build_backbone, build_heads, load_bundle

__all__ = [
    "ADAPTER_NAMES",
    "DEFAULT_BATCH_SIZE",
    "TILE_EMBEDDING_CONTRACT",
    "AlhenaEncoder",
    "LoRAQKV",
    "adapter_key",
    "build_backbone",
    "build_heads",
    "bundle_identity",
    "insert_adapters",
    "load_adapter_weights",
    "load_bundle",
    "merge_adapters",
]
