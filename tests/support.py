"""Synthetic inputs, a small encoder and a reference the tests share."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import nn

from alhena import (
    INPUT_SIZE,
    PATCH_SIZE,
    REFERENCE_MPP,
    AlhenaEncoder,
    AnnotationResult,
    BundleInfo,
    label_order,
)
from alhena._constants import (
    IMAGE_MEAN,
    IMAGE_STD,
    NUM_PREFIX_TOKENS,
    SCCONCEPT_MODEL,
    SCCONCEPT_VERSION,
)
from alhena.encoder import build_heads

N_TILES = 3
NUCLEI_PER_TILE = 4
NUCLEUS_SIDE = 20

# Widths of the stand-in backbone's tokens, of the shared cell space and of atlas embeddings.
EMBED_DIM = 4
CELL_DIM = 512
ATLAS_DIM = 5
ENCODER_ID = "0123456789abcdef"
CELL_SPACE = "0f1e2d3c4b5a6978"

LINEAGE = tuple(name for name in label_order("lineage_v1") if name != "Unknown")
CELLS_PER_CLASS = 320

# The class of compartment_v1 each class of lineage_v1 falls under.
COMPARTMENT_OF = {
    "B_Plasma": "Immune",
    "Myeloid": "Immune",
    "T_NK": "Immune",
    "Epithelial_Melanocyte": "Epithelial_Melanocyte",
    "Stromal_Vascular": "Stromal_Vascular",
}


def make_tiles(n_tiles: int = N_TILES, size: int = INPUT_SIZE, seed: int = 5) -> np.ndarray:
    """Return stacked uint8 RGB tiles of random pixels."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n_tiles, size, size, 3), dtype=np.uint8)


def nucleus_corners(per_tile: int = NUCLEI_PER_TILE) -> list[tuple[int, int]]:
    """Return the top-left corner (row, column) of each nucleus of a tile."""
    return [(20 + (index // 2) * 90, 20 + (index % 2) * 90) for index in range(per_tile)]


def make_label_maps(
    n_tiles: int = N_TILES, per_tile: int = NUCLEI_PER_TILE, size: int = INPUT_SIZE
) -> np.ndarray:
    """Return one label map per tile, square nuclei numbered from 1 across the tiles."""
    maps = np.zeros((n_tiles, size, size), dtype=np.int32)
    identifier = 1
    for tile in range(n_tiles):
        for row, column in nucleus_corners(per_tile):
            maps[tile, row : row + NUCLEUS_SIDE, column : column + NUCLEUS_SIDE] = identifier
            identifier += 1
    return maps


def make_centroids(
    n_tiles: int = N_TILES, per_tile: int = NUCLEI_PER_TILE
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (x, y) center of each nucleus in its own tile, and the tile holding it."""
    half = (NUCLEUS_SIDE - 1) / 2
    corners = nucleus_corners(per_tile)
    points = [[column + half, row + half] for _ in range(n_tiles) for row, column in corners]
    tiles = np.repeat(np.arange(n_tiles), per_tile)
    return np.asarray(points, dtype=np.float64), tiles.astype(np.int64)


def make_polygons(
    n_tiles: int = N_TILES, per_tile: int = NUCLEI_PER_TILE, size: int = INPUT_SIZE
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return square nucleus outlines and tile windows in one frame, the tiles side by side."""
    polygons = []
    for tile in range(n_tiles):
        for row, column in nucleus_corners(per_tile):
            x, y = tile * size + column, row
            corners = [[x, y], [x + NUCLEUS_SIDE, y], [x + NUCLEUS_SIDE, y + NUCLEUS_SIDE]]
            polygons.append(np.asarray([*corners, [x, y + NUCLEUS_SIDE]], dtype=np.float64))
    bounds = np.asarray(
        [[tile * size, 0, (tile + 1) * size, size] for tile in range(n_tiles)], dtype=np.float64
    )
    return polygons, bounds


class PixelBackbone(nn.Module):
    """A backbone reading one pixel per token, standing in for the transformer."""

    embed_dim = EMBED_DIM

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.5, 1.5, EMBED_DIM), requires_grad=False)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        corners = images[:, :, ::PATCH_SIZE, ::PATCH_SIZE].flatten(2).transpose(1, 2)
        tokens = torch.cat([corners, corners[..., :1] * corners[..., 1:2]], dim=-1) * self.scale
        prefix = torch.zeros(images.shape[0], NUM_PREFIX_TOKENS, EMBED_DIM)
        prefix[:, 0] = tokens[:, 0]
        return torch.cat([prefix, tokens], dim=1)

    def forward_head(self, features: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return features[:, 0]


def bundle_info() -> BundleInfo:
    """Return the description of the small encoder below."""
    return BundleInfo(
        encoder_id=ENCODER_ID,
        cell_space_id=CELL_SPACE,
        backbone="test-backbone",
        embed_dim=EMBED_DIM,
        cell_dim=CELL_DIM,
        lora_rank=4,
        lora_alpha=2.0,
        input_size=INPUT_SIZE,
        patch_size=PATCH_SIZE,
        reference_mpp=REFERENCE_MPP,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        interpolation="bilinear",
        has_image_proj=False,
        has_cell_proj=True,
        scconcept_model=SCCONCEPT_MODEL,
        scconcept_version=SCCONCEPT_VERSION,
        version="0.1.0",
    )


def tiny_encoder() -> AlhenaEncoder:
    """Return a whole encoder over the stand-in backbone, with both heads of the cell space."""
    generator = torch.Generator().manual_seed(0)
    heads = build_heads(
        {
            "img_tok_proj.weight": torch.randn((CELL_DIM, EMBED_DIM), generator=generator),
            "st_node_proj.weight": torch.randn((CELL_DIM, ATLAS_DIM), generator=generator),
        },
        embed_dim=EMBED_DIM,
        cell_dim=CELL_DIM,
        scconcept_dim=ATLAS_DIM,
    )
    return AlhenaEncoder(PixelBackbone(), heads, bundle_info())


def class_codes(counts: tuple[int, ...] = (CELLS_PER_CLASS,) * len(LINEAGE)) -> np.ndarray:
    """Return the code of each reference cell, class after class."""
    return np.repeat(np.arange(len(counts)), counts).astype(np.int8)


def class_embeddings(codes: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return one direction of the cell space per class, with a little noise."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(int(codes.max()) + 1, CELL_DIM))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    noise = rng.normal(scale=0.01, size=(codes.size, CELL_DIM))
    return (directions[codes] + noise).astype(np.float16)


def make_shard(
    directory: Path,
    name: str = "prostate",
    codes: np.ndarray | None = None,
    metadata: dict[str, str] | None = None,
    axis: str = "lineage_v1",
) -> Path:
    """Write a reference shard of separable classes, in the cell space of the encoder."""
    codes = class_codes() if codes is None else np.asarray(codes, dtype=np.int8)
    header = {
        "shard": name,
        "n_cells": str(codes.size),
        "scconcept_model": SCCONCEPT_MODEL,
        "scconcept_version": SCCONCEPT_VERSION,
        "row_order": "source_atlas",
        "cell_space_id": CELL_SPACE,
        **(metadata or {}),
    }
    path = Path(directory) / f"{name}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {"embeddings": class_embeddings(codes), axis: codes}
    save_file(tensors, str(path), metadata=header)
    return path


def make_result(
    labels: list[str] | np.ndarray,
    confidence: list[float] | np.ndarray | None = None,
    cell_id: list[int] | np.ndarray | None = None,
    *,
    class_order: tuple[str, ...] = LINEAGE,
    axis: str = "lineage_v1",
) -> AnnotationResult:
    """Return annotations of the given labels, the empty label marking a cell without one."""
    names = np.asarray(labels, dtype=object)
    n_cells = names.shape[0]
    valid = names != ""
    confidence = np.full(n_cells, 0.9) if confidence is None else np.asarray(confidence, float)
    return AnnotationResult(
        cell_id=np.arange(1, n_cells + 1) if cell_id is None else np.asarray(cell_id),
        labels=names,
        confidence=np.where(valid, confidence, np.nan),
        mass=np.zeros((n_cells, len(class_order)), dtype=np.float32),
        class_order=class_order,
        class_entropy=np.zeros(n_cells),
        neighbor_entropy=np.zeros(n_cells),
        neighbor_similarity=np.zeros((n_cells, 1)),
        valid=valid,
        axis=axis,
    )
