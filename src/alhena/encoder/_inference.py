"""The encoder: tile and cell embeddings from tiles."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from alhena import _progress
from alhena._constants import (
    CROP_WARNING_FRACTION,
    IMAGE_MEAN,
    IMAGE_STD,
    INPUT_SIZE,
    NUM_PREFIX_TOKENS,
    NUM_TOKENS,
    REFERENCE_MPP,
)
from alhena._defaults import DEFAULT_BATCH_SIZE, _batch_size
from alhena._errors import (
    PACKAGE_PREFIXES,
    AlhenaValueError,
    BundleError,
    GeometryError,
    GeometryWarning,
)
from alhena._frames import _frame_mpp
from alhena._geometry import (
    _crop_advice,
    _crop_fraction,
    _format_mpp,
    _merge_records,
    _prepare,
    _resolved_mpp,
    _warn_resolution,
)
from alhena.cells._path import _encode_plan, _plan_cells
from alhena.io import TileSource, as_tile_source, tile_bounds
from alhena.preprocess import prepare_tiles
from alhena.types import (
    BundleInfo,
    CellEmbeddings,
    NucleiInput,
    ResampleInfo,
    TileEmbeddings,
    TileInput,
)

TILE_EMBEDDING_CONTRACT: str = (
    "A tile embedding is the class token over every token of the canvas, taken unmasked and "
    "unweighted, so it moves with the geometry the tiles reach. Tile embeddings computed at "
    "different geometries are not comparable and must not be pooled."
)
"""What a tile embedding is, as the warnings on white pad and on patch resolution state it."""


def _prepare_batch(
    tiles: np.ndarray | Sequence[np.ndarray] | torch.Tensor,
    mpp: float | None,
    progress: bool | None = None,
) -> tuple[torch.Tensor, ResampleInfo | None]:
    """Bring tiles to the batch the model reads, with the record of the fit, or None if prepared."""
    if isinstance(tiles, torch.Tensor):
        if torch.is_floating_point(tiles):
            if mpp is not None:
                raise GeometryError(
                    "These tiles are already prepared, so there is nothing left for mpp to "
                    "resample. Pass raw uint8 tiles with their mpp, or prepared tiles without it."
                )
            if tiles.ndim != 4 or tuple(tiles.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
                raise GeometryError(
                    f"Prepared tiles must have shape (N, 3, {INPUT_SIZE}, {INPUT_SIZE}), got "
                    f"{tuple(tiles.shape)}. Use prepare_tiles to reach it."
                )
            if tiles.shape[0] == 0:
                raise AlhenaValueError("No tiles were given.")
            return tiles.detach().to(dtype=torch.float32), None
        tiles = tiles.detach().cpu().numpy()
    return prepare_tiles(tiles, mpp=mpp, progress=progress)


def _white_pad_warning(info: ResampleInfo) -> str:
    """Return what white pad on the canvas means for a tile embedding, at a declared resolution."""
    mpp = 0.0 if info.source_mpp is None else float(info.source_mpp)
    field = tuple(side * mpp for side in info.source_size)
    return (
        f"Tiles at {_format_mpp(mpp)} micron per pixel cover {field[0]:.1f} by {field[1]:.1f} "
        f"microns and land on {info.resampled_size[0]} by {info.resampled_size[1]} of the "
        f"{INPUT_SIZE} pixel canvas, so {info.pad_fraction:.1%} of it is white pad. "
        f"{TILE_EMBEDDING_CONTRACT} Cut fields of at least {INPUT_SIZE * info.target_mpp:.1f} "
        "microns to fill the canvas."
    )


def _crop_warning(info: ResampleInfo) -> str:
    """Explain excessive center cropping of tile embeddings."""
    mpp = 0.0 if info.source_mpp is None else float(info.source_mpp)
    window = INPUT_SIZE * info.target_mpp
    height, width = info.resampled_size
    if _crop_fraction(height, width) == info.crop_fraction:
        field = tuple(side * mpp for side in info.source_size)
        kept = tuple(min(side, INPUT_SIZE) * info.target_mpp for side in info.resampled_size)
        opening = (
            f"Tiles at {_format_mpp(mpp)} micron per pixel cover "
            f"{field[0]:.1f} by {field[1]:.1f} microns and resample to "
            f"{height} by {width} pixels. The {INPUT_SIZE}-pixel canvas keeps "
            f"the central {kept[0]:.1f} by {kept[1]:.1f} microns; "
            f"{info.crop_fraction:.1%} of each field is cropped away."
        )
    else:
        opening = (
            f"Tiles at {_format_mpp(mpp)} micron per pixel differ in size. "
            f"The {INPUT_SIZE}-pixel canvas keeps at most the central "
            f"{window:.1f} by {window:.1f} microns of each field; "
            f"up to {info.crop_fraction:.1%} of a field is cropped away."
        )
    meaning = (
        "Tile embeddings use the class token across the retained canvas, "
        "without masking or weighting its image tokens. Each embedding "
        "describes only the retained central area. Tile embeddings computed "
        "at different geometries must not be compared or pooled."
    )
    return f"{opening}\n\n{meaning}\n\n{_crop_advice(mpp, info.target_mpp)}"


def _devices_here() -> str:
    """Name the devices torch can run on here, as a call takes them."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices += [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        devices.append("mps")
    named = [repr(device) for device in devices]
    return named[0] if len(named) == 1 else f"{', '.join(named[:-1])} or {named[-1]}"


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Return the device to run on, refusing a CUDA or Apple silicon GPU torch does not see."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise AlhenaValueError(f"{device!r} names no device. Pass {_devices_here()}.") from error
    hint = ""
    if resolved.type == "cuda":
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        available = (resolved.index or 0) < count
        if not torch.backends.cuda.is_built():
            hint = ", or install a build of torch with CUDA support"
    else:
        available = resolved.type != "mps" or (
            torch.backends.mps.is_available() and not resolved.index
        )
    if not available:
        raise AlhenaValueError(
            f"{device!r} is not a device torch sees here. Pass {_devices_here()}{hint}."
        )
    return resolved


def _warn_fit(info: ResampleInfo) -> None:
    """Warn of white pad on the canvas, and of a center crop past `CROP_WARNING_FRACTION`."""
    if info.pad_fraction > 0.0:
        warnings.warn(
            _white_pad_warning(info), GeometryWarning, skip_file_prefixes=PACKAGE_PREFIXES
        )
    if info.crop_fraction > CROP_WARNING_FRACTION:
        warnings.warn(_crop_warning(info), GeometryWarning, skip_file_prefixes=PACKAGE_PREFIXES)


def _batch_slices(count: int, batch_size: int) -> list[slice]:
    return [slice(start, min(start + batch_size, count)) for start in range(0, count, batch_size)]


def _as_matrix(values: np.ndarray | torch.Tensor, described: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        matrix = values.detach().to(dtype=torch.float32)
    else:
        matrix = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    if matrix.ndim != 2:
        raise AlhenaValueError(
            f"Expected a two-dimensional array of {described}, got shape {tuple(matrix.shape)}."
        )
    return matrix


def _autocast_off(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    """Turn autocast off on CPU and CUDA; torch 2.2 has no autocast context for MPS."""
    if device.type in ("cpu", "cuda"):
        return torch.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


class _TileEmbedder(nn.Module):
    """Pre-projection tile embeddings from tiles on the unit scale, normalized and in eval mode."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        device = next(backbone.parameters()).device
        for name, values in (("mean", IMAGE_MEAN), ("std", IMAGE_STD)):
            self.register_buffer(
                name, torch.tensor(values, dtype=torch.float32, device=device).view(1, 3, 1, 1)
            )
        self.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise GeometryError(
                "Expected a (B, 3, H, W) tensor with values in [0, 1], got shape "
                f"{tuple(images.shape)}."
            )
        with _autocast_off(images.device):
            normalized = (images.to(dtype=torch.float32) - self.mean) / self.std
            features = self.backbone.forward_features(normalized)
            return self.backbone.forward_head(features, pre_logits=True)


class AlhenaEncoder:
    """The adapted backbone together with the projection heads that read out of it.

    `load_encoder` and `load_bundle` build it. Every method runs in float32 under inference mode,
    with autocast disabled on CPU and CUDA, and no output is normalized.

    Args:
        backbone: UNI2-h from `build_backbone`, with adapters inserted by `insert_adapters`.
        heads: The projection heads, as `build_heads` makes them.
        info: The identity and geometry of the bundle, a `BundleInfo`.
        device: Device to run on, such as `"cpu"`, `"cuda"` or `"mps"`.

    Attributes:
        info: Identity and geometry of the bundle.
        device: Device the backbone and the heads sit on.

    Raises:
        AlhenaValueError: If `device` names no device this machine has.
        BundleError: If the backbone is not as wide as the bundle declares.
    """

    def __init__(
        self,
        backbone: nn.Module,
        heads: nn.Module,
        info: BundleInfo,
        device: torch.device | str = "cpu",
    ) -> None:
        backbone_dim = getattr(backbone, "embed_dim", None)
        if backbone_dim is not None and int(backbone_dim) != info.embed_dim:
            raise BundleError(
                f"The bundle declares an embedding width of {info.embed_dim} where the backbone "
                f"produces {int(backbone_dim)}."
            )
        self.device = _resolve_device(device)
        self.info = info
        self.backbone = backbone.to(self.device)
        self.heads = heads.to(self.device)
        self.backbone.eval()
        self.heads.eval()
        self.backbone.requires_grad_(False)
        self.heads.requires_grad_(False)

    def __repr__(self) -> str:
        return (
            f"AlhenaEncoder(encoder_id={self.info.encoder_id!r}, "
            f"embed_dim={self.info.embed_dim}, device={self.device.type!r})"
        )

    def _forward(
        self, batch: torch.Tensor, want_tokens: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        with torch.inference_mode(), _autocast_off(self.device):
            features = self.backbone.forward_features(
                batch.to(device=self.device, dtype=torch.float32)
            )
            if features.ndim != 3 or features.shape[1] <= NUM_PREFIX_TOKENS:
                raise BundleError(
                    f"The backbone returned features of shape {tuple(features.shape)}, which hold "
                    f"no patch tokens behind the {NUM_PREFIX_TOKENS} prefix tokens."
                )
            # The normalized class token, copied out so that holding it does not hold the batch.
            pooled = self.backbone.forward_head(features, pre_logits=True)
            embeddings = pooled.cpu().numpy().copy()
            if not want_tokens:
                return embeddings, None
            tokens = features[:, NUM_PREFIX_TOKENS:, :]
            if tokens.shape[1] != NUM_TOKENS:
                raise BundleError(
                    f"The backbone returned {tokens.shape[1]} patch tokens for the "
                    f"{INPUT_SIZE} pixel canvas, where the bundle reads {NUM_TOKENS}: it is not "
                    "the backbone the bundle was trained on."
                )
            return embeddings, tokens.cpu().numpy()

    def encode(
        self,
        tiles: np.ndarray | Sequence[np.ndarray] | torch.Tensor,
        mpp: float | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress: bool | None = None,
    ) -> np.ndarray:
        """Embed tiles with the backbone.

        Any white pad on the canvas raises a `GeometryWarning`, and so does a center crop discarding
        more than `CROP_WARNING_FRACTION` of a field; prepared tiles raise neither.

        Args:
            tiles: Raw uint8 RGB tiles, or a float tensor already prepared for the model.
            mpp: Microns per pixel of the tiles. Refused for prepared tiles.
            batch_size: Tiles per forward pass.
            progress: Whether to report progress on standard error. None follows
                `ALHENA_DISABLE_PROGRESS_BARS`.

        Returns:
            An `(n_tiles, embed_dim)` float32 array, not normalized.

        Raises:
            AlhenaValueError: If `mpp` or `batch_size` is out of range, or no tile is given.
            GeometryError: If the tiles cannot reach the geometry the model reads.
        """
        batch_size = _batch_size(batch_size)
        batch, info = _prepare_batch(tiles, mpp, progress)
        if info is not None:
            _warn_fit(info)
        chunks: list[np.ndarray] = []
        with _progress.bar(len(batch), "Encoding tiles", "tile", progress) as bar:
            for window in _batch_slices(len(batch), batch_size):
                chunks.append(self._forward(batch[window], want_tokens=False)[0])
                bar.update(window.stop - window.start)
        return np.concatenate(chunks, axis=0)

    def encode_tiles(
        self,
        tiles: TileSource | TileInput,
        mpp: float | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress: bool | None = None,
    ) -> TileEmbeddings:
        """Embed the tiles of a source, with where each one was read from.

        The entry point for tile-level analysis: the names, bounds and frame of the source travel
        with the result, and arrays come back named by position and without bounds. Tiles are read
        one batch at a time. The warnings on white pad and the center crop come once: with the first
        batch where every tile has one size, after the pass where sizes differ.

        Args:
            tiles: A tile source, raw uint8 RGB tiles as one array per tile or one stacked array, or
                a float tensor already prepared for the model.
            mpp: Microns per pixel of the tiles. Defaults to the resolution the source carries.
                Refused for prepared tiles.
            batch_size: Tiles per forward pass, which bounds the memory the pass holds.
            progress: Whether to report progress on standard error. None follows
                `ALHENA_DISABLE_PROGRESS_BARS`.

        Returns:
            One embedding per tile of the source, in its order, with the tile bounds and the
            provenance of the pass; `purpose` reads `tiles`.

        Raises:
            AlhenaValueError: If `mpp` or `batch_size` is out of range, or the tiles are unreadable.
            GeometryError: If the tiles cannot reach the geometry the model reads.
        """
        batch_size = _batch_size(batch_size)
        if isinstance(tiles, torch.Tensor) and torch.is_floating_point(tiles):
            # Prepared tiles carry no source, geometry or window; `encode` refuses an mpp for them.
            prepared = self.encode(tiles, mpp=mpp, batch_size=batch_size, progress=progress)
            return TileEmbeddings(
                embeddings=prepared,
                tile_index=np.arange(prepared.shape[0], dtype=np.int64),
                tile_ids=tuple(str(position) for position in range(prepared.shape[0])),
                encoder_id=str(self.info.encoder_id),
                backbone_digest=self.info.backbone_digest,
            )
        if isinstance(tiles, torch.Tensor):
            tiles = tiles.detach().cpu().numpy()
        source = as_tile_source(tiles, mpp=mpp)
        # Checked by the source or by as_tile_source already, so resolved here without a check.
        resolution = _resolved_mpp(source, None if mpp is None else float(mpp))
        _warn_resolution(resolution, REFERENCE_MPP)

        # Tiles of one size fit the canvas alike, so the first batch tells of every tile.
        uniform = source.tile_shape is not None
        chunks: list[np.ndarray] = []
        records: list[ResampleInfo] = []
        shapes: list[tuple[int, int]] = []
        with _progress.bar(len(source), "Encoding tiles", "tile", progress) as bar:
            for group in source.batches(batch_size):
                shapes.extend((int(tile.shape[0]), int(tile.shape[1])) for tile in group)
                batch, record = _prepare(group, resolution, REFERENCE_MPP, progress=False)
                records.append(record)
                if uniform and len(records) == 1:
                    _warn_fit(record)
                chunks.append(self._forward(batch, want_tokens=False)[0])
                bar.update(len(group))
        geometry = _merge_records(records)
        if not uniform and geometry is not None:
            _warn_fit(geometry)

        bounds = frame = bounds_mpp = None
        if source.coords is not None:
            # From the sizes just read, so a source whose tiles differ in size is not read again.
            bounds = tile_bounds(source, tile_shapes=np.asarray(shapes, dtype=np.float64))
            frame = source.coords_frame
            bounds_mpp = _frame_mpp(resolution, float(source.coords_scale))
        return TileEmbeddings(
            embeddings=np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0, int(self.info.embed_dim)), dtype=np.float32),
            tile_index=np.arange(len(source), dtype=np.int64),
            tile_ids=tuple(str(name) for name in source.tile_ids),
            geometry=geometry,
            encoder_id=str(self.info.encoder_id),
            bounds=bounds,
            bounds_frame=frame,
            bounds_mpp=bounds_mpp,
            image_shape=source.image_shape,
            backbone_digest=self.info.backbone_digest,
        )

    def encode_with_tokens(
        self,
        tiles: np.ndarray | Sequence[np.ndarray] | torch.Tensor,
        mpp: float | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Embed tiles and return their patch tokens as well.

        The tokens follow the final layer normalization, without the class and register tokens, in
        row-major order of the grid; pooling tokens read before that normalization gives another
        cell embedding. No warning is raised here about white pad or the center crop: the cell
        path keeps the pad out of pooling and warns about the crop itself.

        Args:
            tiles: Raw uint8 RGB tiles, or a float tensor already prepared for the model.
            mpp: Microns per pixel of the tiles. Refused for prepared tiles.
            batch_size: Tiles per forward pass.
            progress: Whether to report progress on standard error. None follows
                `ALHENA_DISABLE_PROGRESS_BARS`.

        Returns:
            An `(n_tiles, embed_dim)` array of tile embeddings and an
            `(n_tiles, n_tokens, embed_dim)` array of patch tokens, neither normalized.

        Raises:
            AlhenaValueError: If `mpp` or `batch_size` is out of range, or no tile is given.
            BundleError: If the backbone yields another token grid than the bundle reads.
            GeometryError: If the tiles cannot reach the geometry the model reads.
        """
        batch_size = _batch_size(batch_size)
        batch, _ = _prepare_batch(tiles, mpp, progress)
        pooled: list[np.ndarray] = []
        tokens: list[np.ndarray] = []
        with _progress.bar(len(batch), "Encoding tiles", "tile", progress) as bar:
            for window in _batch_slices(len(batch), batch_size):
                chunk_pooled, chunk_tokens = self._forward(batch[window], want_tokens=True)
                pooled.append(chunk_pooled)
                tokens.append(chunk_tokens)
                bar.update(window.stop - window.start)
        return np.concatenate(pooled, axis=0), np.concatenate(tokens, axis=0)

    def encode_cells(
        self,
        tiles: TileInput,
        nuclei: NucleiInput,
        mpp: float | None = None,
        *,
        cell_id: Sequence[Any] | np.ndarray | None = None,
        all_tiles: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress: bool | None = None,
    ) -> CellEmbeddings:
        """Embed the cells of a set of tiles in one pass, each tile embedded once.

        The tiles and the nuclei are checked, and every nucleus resolved to the tile it is read
        from, before the tiles are embedded one batch at a time. By default a tile no cell is read
        from is skipped, and the embeddings of the tiles read come back on `CellEmbeddings.tiles`.
        The cells equal those `encode_with_tokens` and `pool_cells` give, to float32 rounding.

        Args:
            tiles: Raw uint8 RGB tiles, one array per tile or one stacked array, a tile source, or a
                float tensor already prepared for the model.
            nuclei: Label maps, the mapping an `alhena.io` reader of nuclei returns, or a membership
                table.
            mpp: Microns per pixel of the tiles, declared once for the images and the nuclei.
                Refused for prepared tiles.
            cell_id: Unique identifier of each nucleus, for the builders that take positions.
            all_tiles: Read every tile, including those no cell is read from.
            batch_size: Tiles per forward pass, which bounds the memory the pass holds.
            progress: Whether to report progress on standard error. None follows
                `ALHENA_DISABLE_PROGRESS_BARS`.

        Returns:
            One pre-projection and one cell-space embedding per cell that reached a token, ordered
            by identifier, with their provenance and the embeddings of the tiles read.

        Raises:
            AlhenaValueError: If `mpp` is out of range, or the tiles and the nuclei do not match.
            BundleError: If the bundle carries no cell head.
            GeometryError: If the tiles cannot reach the canvas, or the nuclei lie on another grid.
            MembershipError: If the nuclei and the tiles cannot be reconciled.
        """
        batch_size = _batch_size(batch_size)
        plan = _plan_cells(
            tiles,
            nuclei,
            mpp,
            cell_id=cell_id,
            all_tiles=all_tiles,
            progress=progress,
        )
        return _encode_plan(self, plan, batch_size=batch_size, progress=progress)

    def _project(
        self,
        values: np.ndarray | torch.Tensor,
        head: nn.Linear | None,
        head_name: str,
        described: str,
    ) -> np.ndarray:
        if head is None:
            raise BundleError(
                f"This bundle carries no {head_name} head, so {described} cannot be projected."
            )
        matrix = _as_matrix(values, described)
        if matrix.shape[1] != head.in_features:
            raise BundleError(
                f"These {described} are {matrix.shape[1]} columns wide where {head_name} reads "
                f"{head.in_features}."
            )
        with torch.inference_mode(), _autocast_off(self.device):
            projected = head(matrix.to(device=self.device, dtype=torch.float32))
            return projected.cpu().numpy().copy()

    def project_tile(self, embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
        """Project tile embeddings with the tile head.

        Args:
            embeddings: A (n_tiles, embed_dim) array of pre-projection tile embeddings.

        Returns:
            A (n_tiles, cell_dim) float32 array, not normalized.

        Raises:
            AlhenaValueError: If the array is not two-dimensional.
            BundleError: If the bundle carries no tile head, or the array is the wrong width.
        """
        return self._project(embeddings, self.heads.image_proj, "image_proj", "tile embeddings")

    def project_cell(self, pooled_tokens: np.ndarray | torch.Tensor) -> np.ndarray:
        """Project pooled patch tokens with the cell head, the image side of the retrieval space.

        Args:
            pooled_tokens: An `(n_cells, embed_dim)` array of nucleus-pooled patch tokens.

        Returns:
            An `(n_cells, cell_dim)` float32 array, not normalized.

        Raises:
            AlhenaValueError: If the array is not two-dimensional.
            BundleError: If the bundle carries no cell head, or if the array is the wrong width.
        """
        return self._project(
            pooled_tokens, self.heads.img_tok_proj, "img_tok_proj", "pooled tokens"
        )

    def project_reference(self, scconcept_embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
        """Project single-cell embeddings with the reference head, the reference side of retrieval.

        Args:
            scconcept_embeddings: An `(n_cells, scconcept_dim)` array of single-cell embeddings.

        Returns:
            An `(n_cells, cell_dim)` float32 array, not normalized.

        Raises:
            AlhenaValueError: If the array is not two-dimensional.
            BundleError: If the bundle carries no reference head, or the array is the wrong width.
        """
        return self._project(
            scconcept_embeddings, self.heads.st_node_proj, "st_node_proj", "reference embeddings"
        )

    def as_module(self) -> nn.Module:
        """Return a module mapping tiles on the unit scale to pre-projection tile embeddings.

        The module takes a `(B, 3, 224, 224)` float tensor in [0, 1], normalizes it, and is in
        evaluation mode, as the tile pipelines that call it expect. It shares the backbone with the
        encoder.
        """
        return _TileEmbedder(self.backbone)
