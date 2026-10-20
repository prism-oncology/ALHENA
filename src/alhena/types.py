"""Data structures exchanged between ALHENA's stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from alhena._coverage import _missing_classes, _thin_classes
from alhena._errors import AlhenaValueError
from alhena._frames import _containing_tiles, _window_centers
from alhena._layout import (
    CELLS_PROVENANCE_KEY,
    SPATIAL_KEY,
    _annotation_table,
    _cell_columns,
    _cell_table,
    _h5ad_path,
    _tile_table,
)

if TYPE_CHECKING:
    import pandas as pd
    import torch
    from anndata import AnnData

MEMBERSHIP_COLUMNS: tuple[str, ...] = ("tile_index", "cell_id", "token_index", "nucleus_px")
"""Columns of a nucleus-to-token membership table, in order."""

TileInput: TypeAlias = "np.ndarray | Sequence[np.ndarray] | torch.Tensor"
"""Raw uint8 RGB tiles, one array per tile or one stacked array, or a prepared batch."""

NucleiInput: TypeAlias = "np.ndarray | Sequence[np.ndarray] | Mapping[str, Any] | pd.DataFrame"
"""Label maps, the mapping an `alhena.io` reader of nuclei returns, or a membership table."""


@dataclass(frozen=True)
class BundleInfo:
    """Identity and geometry of an encoder bundle.

    Attributes:
        encoder_id: Identity of the encoder: the first sixteen hexadecimal digits of the SHA-256 of
            the adapter file.
        cell_space_id: Identity of the cell space the cell and reference heads project into, which
            a reference must record to annotate the encoder's cells.
        backbone: Repository of the UNI2-h weights the adapters were trained on.
        embed_dim: Width of the backbone's tokens and embeddings.
        cell_dim: Width of the shared cell space.
        lora_rank: Rank of the low-rank adapters.
        lora_alpha: Scaling of the adapters, whose updates are multiplied by
            `lora_alpha / lora_rank`.
        input_size: Side of the canvas the encoder reads, in pixels.
        patch_size: Side of one token, in pixels of the canvas.
        reference_mpp: Microns per pixel the encoder reads, to which tiles of a declared resolution
            are resampled.
        image_mean: Mean of each color channel, subtracted from a tile on the unit scale.
        image_std: Standard deviation of each color channel, which the centered tile is divided by.
        interpolation: Interpolation tiles are resampled with.
        has_image_proj: Whether the bundle carries the tile head, `image_proj`.
        has_cell_proj: Whether it carries both heads of the cell space, `img_tok_proj` and
            `st_node_proj`.
        scconcept_model: Single-cell model the reference embeddings come from.
        scconcept_version: Version of that model.
        version: Version of ALHENA the bundle states it was published for, else this one.
        backbone_digest: SHA-256 of the UNI2-h tensors, each read as float32 in the order of their
            names, so either file they are distributed in gives one digest; None where not computed.
    """

    encoder_id: str
    cell_space_id: str
    backbone: str
    embed_dim: int
    cell_dim: int
    lora_rank: int
    lora_alpha: float
    input_size: int
    patch_size: int
    reference_mpp: float
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    interpolation: str
    has_image_proj: bool
    has_cell_proj: bool
    scconcept_model: str
    scconcept_version: str
    version: str
    backbone_digest: str | None = None


@dataclass(frozen=True)
class ResampleInfo:
    """What preprocessing did to a batch of tiles to reach the model input geometry.

    Attributes:
        source_mpp: Microns per pixel the tiles were declared at, or None where none was.
        target_mpp: Microns per pixel the tiles were resampled toward.
        scale: Factor the tiles were resized by, one where they passed through untouched.
        source_size: Height and width of the tiles as they arrived; for tiles of several sizes,
            those of the tile that padded most, else cropped most, else the first.
        resampled_size: Height and width that same tile reached before being fitted to the canvas.
        pad_fraction: Share of the canvas filled with pad rather than the image, over the tile that
            padded most. Zero where nothing was padded.
        crop_fraction: Share of the resampled tile the center crop discards, over the tile that lost
            most, whichever tile the sizes describe. Zero where nothing was cropped.
        padded: Whether any tile was padded.
        cropped: Whether any tile was cropped.
        mpp_declared: Whether a resolution was declared for the tiles.
    """

    source_mpp: float | None
    target_mpp: float
    scale: float
    source_size: tuple[int, int]
    resampled_size: tuple[int, int]
    pad_fraction: float
    crop_fraction: float
    padded: bool
    cropped: bool
    mpp_declared: bool


@dataclass(frozen=True)
class GeometryReport:
    """Physical geometry a batch of tiles presents to the model.

    Attributes:
        mpp: Microns per pixel of the tiles, or None where none was declared.
        tile_size_px: Side of the tiles, in pixels.
        field_of_view_um: Microns one tile covers, NaN where the resolution is undeclared.
        token_footprint_um: Microns one token covers on the canvas.
        reference_mpp: Microns per pixel the model reads at.
        mpp_ratio: Ratio of `mpp` to `reference_mpp`, None where the resolution is undeclared.
        median_nucleus_tokens: Side of a typical nucleus, in tokens.
        pad_fraction: Share of the canvas the tiles pad with white; zero where the resolution is
            undeclared, as tiles are then resized to the canvas.
        crop_fraction: Share of each field the center crop discards; zero where the resolution is
            undeclared.
        warnings: Concerns the geometry raises, one or a few sentences each.
    """

    mpp: float | None
    tile_size_px: int
    field_of_view_um: float
    token_footprint_um: float
    reference_mpp: float
    mpp_ratio: float | None
    median_nucleus_tokens: float | None
    pad_fraction: float = 0.0
    crop_fraction: float = 0.0
    warnings: tuple[str, ...] = ()


TILE_PURPOSES: tuple[str, ...] = ("tiles", "cell_context")
"""What a set of tile embeddings was read for: `tiles` or `cell_context`."""


@dataclass
class TileEmbeddings:
    """Tile embeddings, with where in its source image each tile was read from.

    `bounds` is the region of the source image each embedding stands for, in the frame the tiles
    were cut from, and None where the tiles share no frame. `geometry` says what happened to a tile
    on its way to the canvas, padded or cropped, and `purpose` what it was read for.

    Attributes:
        embeddings: One embedding per tile read, `(n_tiles_read, embed_dim)` float32, not
            normalized.
        tile_index: Index of each of those tiles among the tiles given, ascending.
        tile_ids: Name of each of those tiles, as the tile source names it, or its position.
        geometry: What preprocessing did to reach the canvas, as `prepare_tiles` records it over the
            tiles read, or None where the tiles arrived prepared or none was read.
        encoder_id: Identity of the encoder that produced the embeddings, or None where unknown.
        purpose: What the tiles were read for, one of `TILE_PURPOSES`.
        bounds: Spatial bounds of each tile, `(n_tiles_read, 4)` holding x0, y0, x1, y1, upper
            bounds excluded, in the pixels of the frame the tiles were cut from.
        bounds_frame: Name of the frame `bounds` is written in, as the tile source names its
            origins, such as `"slide.svs, level 0"` or `"tile origins"`. None where unknown.
        bounds_mpp: Microns one pixel of that frame spans, so `bounds * bounds_mpp` is in microns.
            None where there are no bounds or no declared resolution.
        image_shape: Height and width of the image the tiles were cut from, in the pixels of that
            frame, or None for tiles that arrive already cut.
        backbone_digest: Digest of the UNI2-h model weights the embeddings were produced with, as
            `BundleInfo.backbone_digest` records it, or None where unknown.
    """

    embeddings: np.ndarray
    tile_index: np.ndarray
    tile_ids: tuple[str, ...]
    geometry: ResampleInfo | None = None
    encoder_id: str | None = None
    purpose: str = "tiles"
    bounds: np.ndarray | None = None
    bounds_frame: str | None = None
    bounds_mpp: float | None = None
    image_shape: tuple[int, int] | None = None
    backbone_digest: str | None = None

    def __post_init__(self) -> None:
        """Refuse a purpose outside `TILE_PURPOSES`, which a table would carry on unread."""
        if self.purpose not in TILE_PURPOSES:
            raise AlhenaValueError(
                f"{self.purpose!r} is no purpose tile embeddings are read for. Use one of "
                f"{list(TILE_PURPOSES)}."
            )

    def __len__(self) -> int:
        return int(self.tile_index.shape[0])

    @property
    def pad_fraction(self) -> float | None:
        """Largest share of the canvas filled with white pad, or None where it is not recorded."""
        return None if self.geometry is None else float(self.geometry.pad_fraction)

    @property
    def crop_fraction(self) -> float | None:
        """Largest share of a field the center crop discarded, or None where it is not recorded."""
        return None if self.geometry is None else float(self.geometry.crop_fraction)

    @property
    def mpp(self) -> float | None:
        """Microns per pixel the tiles were declared at, None where undeclared or unrecorded."""
        return None if self.geometry is None else self.geometry.source_mpp

    @property
    def centers(self) -> np.ndarray | None:
        """Center of each tile, `(n_tiles_read, 2)`, x then y in the frame of `bounds`, or None.

        The mean of the tile's pixel centers, which lie at whole coordinates as for cell positions.
        """
        if self.bounds is None:
            return None
        return _window_centers(self.bounds)

    def to_anndata(self) -> AnnData:
        """Return the tiles as an AnnData table, one row per tile.

        The layout is the one `alhena.io` describes, and `alhena.io.read_tile_embeddings` reads it
        back.

        Raises:
            AlhenaValueError: If the arrays disagree in length.
        """
        return _tile_table(self)

    def write_h5ad(self, path: str | Path) -> None:
        """Write the tiles to an `.h5ad` file, with their spatial bounds and their provenance.

        Args:
            path: File to write. An existing file is replaced.

        Raises:
            AlhenaValueError: If the path does not end in `.h5ad`, or `to_anndata` refuses them.
        """
        target = _h5ad_path(path)
        self.to_anndata().write_h5ad(target)

    def to_spatialdata(
        self,
        sdata: Any | None = None,
        shapes_key: str | None = None,
        table_key: str | None = None,
        image_key: str | None = None,
        overwrite: bool = False,
    ) -> Any:
        """Return the tiles as a SpatialData object: one rectangle and one row per tile.

        Without `sdata` the object holds the rectangles and the table alone. As in
        `alhena.spatial.tiles_to_spatialdata`, the rectangles take the transformations of the image
        the tiles were cut from where `sdata` holds it, and the identity in
        `alhena.spatial.DEFAULT_COORDINATE_SYSTEM` otherwise.

        Args:
            sdata: Object the rectangles and the table are added to, or None to build one.
            shapes_key: Name the rectangles are stored under, or None for the default
                `alhena.spatial.tiles_to_spatialdata` declares.
            table_key: Name the table is stored under, or None for that call's default.
            image_key: Image of `sdata` the tiles were cut from. Defaults to the image of `sdata`
                their bounds are written in.
            overwrite: Replace elements already stored under those names.

        Returns:
            The object, with the rectangles and the table in it.

        Raises:
            AlhenaValueError: If the tiles lack bounds, a name is taken, or `image_key` is wrong.
            MissingExtraError: If the spatial extra is not installed.
        """
        from alhena.spatial import tiles_to_spatialdata

        return tiles_to_spatialdata(
            sdata,
            self,
            shapes_key=shapes_key,
            table_key=table_key,
            image_key=image_key,
            overwrite=overwrite,
        )

    def select(self, keep: Any) -> TileEmbeddings:
        """Return the tiles `keep` selects, every array cut alike and the provenance as recorded.

        Args:
            keep: A boolean mask, one entry per tile, or the positions of the tiles to keep, in the
                order they are kept in.

        Returns:
            The tiles kept, with the geometry and the provenance of the run that read them.

        Raises:
            AlhenaValueError: If a mask does not fit, or a position repeats or is out of range.
        """
        rows = _selected_rows(keep, len(self), "tile")
        return replace(
            self,
            embeddings=np.asarray(self.embeddings)[rows],
            tile_index=np.asarray(self.tile_index)[rows],
            tile_ids=tuple(self.tile_ids[row] for row in rows.tolist()),
            bounds=None if self.bounds is None else np.asarray(self.bounds)[rows],
        )

    def locate(self, cells: Any) -> np.ndarray:
        """Name the tile whose region holds each cell, wherever the cells were encoded.

        A region runs from the first pixel of a tile's bounds to its last, pixel centers at integer
        coordinates, as `alhena.viz.tile_regions` draws it. Where regions overlap, the tile whose
        center lies nearest holds the cell. The frames of the tiles and of the cells must agree
        where both are recorded.

        Args:
            cells: `CellEmbeddings`, the `AnnotationResult` carrying them, a table of cells ALHENA
                wrote, or an `(n_cells, 2)` array of x and y in the frame of `bounds`.

        Returns:
            The name of each cell's tile, as `tile_ids` names it, in the order of the cells; None
            for a cell outside every tile.

        Raises:
            AlhenaValueError: If the tiles lack bounds, the cells positions, or their frames differ.
        """
        if self.bounds is None:
            raise AlhenaValueError(
                "These tile embeddings carry no spatial bounds, so no cell can be placed in them. "
                "Read the tiles with their origins, as the readers of alhena.io place them."
            )
        positions, frame = _cell_positions(cells)
        if frame is not None and self.bounds_frame is not None and frame != self.bounds_frame:
            raise AlhenaValueError(
                f"The cells are placed in {frame!r} and the tiles in {self.bounds_frame!r}, so "
                "their positions cannot be compared. Read both from the same image, or pass the "
                "positions of the cells as an array once they are in the frame of the tiles."
            )
        rows = _containing_tiles(np.asarray(self.bounds), positions)
        names = np.full(rows.shape[0], None, dtype=object)
        inside = rows >= 0
        names[inside] = np.asarray(self.tile_ids, dtype=object)[rows[inside]]
        return names


@dataclass
class CellEmbeddings:
    """Cell embeddings pooled from tile tokens, with the provenance needed to annotate them later.

    `AlhenaEncoder.encode_cells` fills every field; `pool_cells` fills those a membership table
    records and leaves the others None.

    Attributes:
        cell_id: Identifier of each cell, in the order of every other array here.
        backbone: Pooled pre-projection cell embeddings, `(n_cells, embed_dim)`.
        cell_space: Embeddings after the cell projection, `(n_cells, cell_dim)`, or None when the
            bundle carries no cell head.
        alpha: Nucleus pixels supporting each cell in the tile it was read from.
        tile_index: Index of the tile each cell was read from.
        n_candidate_tiles: How many tiles contained the nucleus.
        valid: False where the embedding is unusable and no label may be emitted.
        encoder_id: Identity of the encoder that produced the embeddings, or None where unknown.
        cell_space_id: Identity of the space `cell_space` lies in, or None where unknown; only a
            reference of the same identity annotates the cells.
        geometry: The geometry the tiles presented to the model, as `describe_geometry` reports it,
            or None where unknown.
        membership_source: Kind of nucleus input the membership was built from: `labels`,
            `centroids`, `polygons`, or `unknown` for a table assembled by hand.
        membership_approximate: Whether the pooling weights approximate the nucleus shape, or None
            where the membership records no provenance.
        n_cells_without_tokens: Nuclei of the input that reached no token and so have no row here,
            or None where the membership records no set of input cells.
        centroids: Position of each nucleus, `(n_cells, 2)` x then y in the pixels of the frame the
            tiles share, pixel centers at integer coordinates; None without a shared frame.
        centroids_frame: Name of that frame, such as `"slide.svs, level 0"`, `"tile origins"` or
            `"polygons"`. None where unknown.
        centroids_mpp: Microns one pixel of that frame spans, so `centroids * centroids_mpp` is in
            microns. None where no position or no resolution is recorded.
        tiles: Embeddings of the tiles read in the same pass, recording `purpose="cell_context"`, or
            None.
        backbone_digest: Digest of the UNI2-h model weights the embeddings were produced with, as
            `BundleInfo.backbone_digest` records it, or None where unknown.
    """

    cell_id: np.ndarray
    backbone: np.ndarray
    cell_space: np.ndarray | None
    alpha: np.ndarray
    tile_index: np.ndarray
    n_candidate_tiles: np.ndarray
    valid: np.ndarray
    encoder_id: str | None = None
    cell_space_id: str | None = None
    geometry: GeometryReport | None = None
    membership_source: str | None = None
    membership_approximate: bool | None = None
    n_cells_without_tokens: int | None = None
    centroids: np.ndarray | None = None
    centroids_frame: str | None = None
    centroids_mpp: float | None = None
    tiles: TileEmbeddings | None = None
    backbone_digest: str | None = None

    def __len__(self) -> int:
        return int(self.cell_id.shape[0])

    @property
    def mpp(self) -> float | None:
        """Microns per pixel the tiles were declared at, None where undeclared or unknown."""
        return None if self.geometry is None else self.geometry.mpp

    def to_anndata(self) -> AnnData:
        """Return the cells as an AnnData table, one row per cell.

        The layout is the one `alhena.io` describes, and `alhena.io.read_cells` reads it back.
        `alhena.io.add_annotation` adds the annotations made from the cells to it.

        Raises:
            AlhenaValueError: If the arrays disagree, or an identifier is neither integer nor text.
        """
        return _cell_table(self)

    def write_h5ad(self, path: str | Path) -> None:
        """Write the cells to an `.h5ad` file, with the provenance needed to annotate them later.

        Args:
            path: File to write. An existing file is replaced.

        Raises:
            AlhenaValueError: If the path does not end in `.h5ad`, or `to_anndata` refuses them.
        """
        target = _h5ad_path(path)
        self.to_anndata().write_h5ad(target)

    def to_spatialdata(
        self,
        sdata: Any,
        region: str,
        table_key: str | None = None,
        overwrite: bool = False,
    ) -> AnnData:
        """Write the cells into a SpatialData object, as the table of the element of their nuclei.

        `alhena.spatial.cells_to_spatialdata` with its default columns. The annotations of the
        cells are added to the same table later, by `AnnotationResult.to_spatialdata`.

        Args:
            sdata: The object the table is stored in, holding the element of the nuclei.
            region: Element the table annotates, whose instances the cell identifiers name.
            table_key: Name the table is stored under, or None for that call's default.
            overwrite: Replace a table of cells already stored under that name.

        Returns:
            The table of cells, stored in the object.

        Raises:
            AlhenaValueError: If the identifiers are not of the element, or the name is taken.
            MissingExtraError: If the spatial extra is not installed.
        """
        from alhena.spatial import cells_to_spatialdata

        return cells_to_spatialdata(sdata, self, region, table_key=table_key, overwrite=overwrite)

    def select(self, keep: Any) -> CellEmbeddings:
        """Return the cells `keep` selects, every array cut alike and the provenance as recorded.

        Args:
            keep: A boolean mask, one entry per cell, or the positions of the cells to keep, in the
                order they are kept in.

        Returns:
            The cells kept, with the tiles read and the provenance of the run that encoded them.

        Raises:
            AlhenaValueError: If a mask does not fit, or a position repeats or is out of range.
        """
        rows = _selected_rows(keep, len(self))
        return replace(
            self,
            cell_id=np.asarray(self.cell_id)[rows],
            backbone=np.asarray(self.backbone)[rows],
            cell_space=None if self.cell_space is None else np.asarray(self.cell_space)[rows],
            alpha=np.asarray(self.alpha)[rows],
            tile_index=np.asarray(self.tile_index)[rows],
            n_candidate_tiles=np.asarray(self.n_candidate_tiles)[rows],
            valid=np.asarray(self.valid)[rows],
            centroids=None if self.centroids is None else np.asarray(self.centroids)[rows],
        )


@dataclass
class AnnotationResult:
    """Cell-type annotations and their supporting quantities.

    Attributes:
        cell_id: Identifier of each cell, in the order of every other array here.
        labels: Predicted class name, empty string where no annotation was assigned.
        confidence: Mass the winning class received, between 1 / n_classes and 1; NaN where no
            annotation was assigned.
        mass: Per-class mass, shape (n_cells, n_classes). Each row sums to one, and is NaN where
            no annotation was assigned.
        class_order: Class names indexing the columns of `mass`.
        class_entropy: Entropy of the per-class mass.
        neighbor_entropy: Entropy of the neighbor weights.
        neighbor_similarity: Cosine similarity of each retrieved neighbor.
        valid: False where no annotation was assigned.
        axis: Class axis the reference was labeled on.
        meta: Provenance of the annotation, such as k, tau, the cell space identity, the class
            support of the reference and the median effective number of neighbors.
        cells: The cell embeddings the annotations were made from, in the same order, as
            `CellTyper.annotate` attaches them; None from `annotate_cells` until they are set.
    """

    cell_id: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray
    mass: np.ndarray
    class_order: tuple[str, ...]
    class_entropy: np.ndarray
    neighbor_entropy: np.ndarray
    neighbor_similarity: np.ndarray
    valid: np.ndarray
    axis: str
    meta: dict[str, Any] = field(default_factory=dict)
    cells: CellEmbeddings | None = None

    def __len__(self) -> int:
        return int(self.cell_id.shape[0])

    @property
    def missing_classes(self) -> tuple[str, ...]:
        """Classes of the axis the reference held no cell of, as `meta` records them, or none."""
        recorded = (self.meta or {}).get("unexpressed_classes")
        if recorded is None:
            return ()
        return tuple(str(name) for name in np.asarray(recorded, dtype=object).reshape(-1).tolist())

    def to_anndata(self, name: str | None = None) -> AnnData:
        """Return the annotations and the cells they were made from as an AnnData table.

        It is the table `CellEmbeddings.to_anndata` returns, with the annotation added under
        `name` as `alhena.io.add_annotation` adds it: on the default axis,
        `alhena_celltype_lineage_v1`. `alhena.io.read_annotations` reads it back.

        Args:
            name: Name the fields of the annotation carry, the axis of the annotations by default.

        Raises:
            AlhenaValueError: If no cells, or other cells, are attached, or the name is unusable.
        """
        if self.cells is None:
            raise AlhenaValueError(
                "These annotations carry no cell embeddings, so no table can hold them beside the "
                "annotations. CellTyper.annotate attaches them; for annotations made with "
                "annotate_cells, set cells to the CellEmbeddings the annotations were made from."
            )
        return _annotation_table(self, self.cells, name)

    def write_h5ad(self, path: str | Path, name: str | None = None) -> None:
        """Write the annotations, the cells they were made from and both provenances to `.h5ad`.

        Args:
            path: File to write. An existing file is replaced.
            name: Name the fields of the annotation carry, the axis of the annotations by default.

        Raises:
            AlhenaValueError: If the path does not end in `.h5ad`, or `to_anndata` refuses them.
        """
        target = _h5ad_path(path)
        self.to_anndata(name).write_h5ad(target)

    def to_spatialdata(
        self,
        sdata: Any,
        region: str,
        table_key: str | None = None,
        name: str | None = None,
        overwrite: bool = False,
    ) -> AnnData:
        """Write the annotations into the table of cells of the element their nuclei came from.

        `alhena.spatial.cells_to_spatialdata` with its default columns: the first call creates the
        table, and a later one adds its annotation to it.

        Args:
            sdata: The object the table is stored in, holding the element of the nuclei.
            region: Element the table annotates, whose instances the cell identifiers name.
            table_key: Name the table is stored under, or None for that call's default.
            name: Name the fields of the annotation carry. Defaults to the axis of the annotations.
            overwrite: Replace the annotation the table holds under `name`, and that one alone.

        Returns:
            The table of cells, stored in the object, with the annotation in it.

        Raises:
            AlhenaValueError: If the cells are missing or not of the element, or the name is taken.
            MissingExtraError: If the spatial extra is not installed.
        """
        from alhena.spatial import cells_to_spatialdata

        return cells_to_spatialdata(
            sdata, self, region, table_key=table_key, name=name, overwrite=overwrite
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the annotations as a table, one row per cell.

        The table holds the identifier, the predicted class, the confidence, both entropies, the
        validity flag and one `mass_<class>` column per class, and, where the cells are attached,
        what each records apart from its embeddings.

        Raises:
            AlhenaValueError: If the attached cells are not those the annotations describe.
        """
        import pandas as pd

        frame = pd.DataFrame(
            {
                "cell_id": self.cell_id,
                "label": self.labels,
                "confidence": self.confidence,
                "class_entropy": self.class_entropy,
                "neighbor_entropy": self.neighbor_entropy,
                "valid": self.valid,
            }
        )
        for position, name in enumerate(self.class_order):
            frame[f"mass_{name}"] = self.mass[:, position]
        if self.cells is not None:
            for name, values in _cell_columns(self.cells, np.asarray(self.cell_id)).items():
                frame[name] = values
        return frame

    def select(self, keep: Any) -> AnnotationResult:
        """Return the annotations of the cells `keep` selects, with those cells attached.

        Every per-cell array is cut alike. `meta` is kept as recorded, so its counts describe the
        run that made the annotations, not the cells kept.

        Args:
            keep: A boolean mask, one entry per cell, or the positions of the cells to keep, in the
                order they are kept in.

        Returns:
            The annotations kept, on the same axis and class order.

        Raises:
            AlhenaValueError: If a mask does not fit, or a position repeats or is out of range.
        """
        rows = _selected_rows(keep, len(self))
        return replace(
            self,
            cell_id=np.asarray(self.cell_id)[rows],
            labels=np.asarray(self.labels)[rows],
            confidence=np.asarray(self.confidence)[rows],
            mass=np.asarray(self.mass)[rows],
            class_entropy=np.asarray(self.class_entropy)[rows],
            neighbor_entropy=np.asarray(self.neighbor_entropy)[rows],
            neighbor_similarity=np.asarray(self.neighbor_similarity)[rows],
            valid=np.asarray(self.valid)[rows],
            meta=dict(self.meta),
            cells=None if self.cells is None else self.cells.select(rows),
        )


@dataclass(frozen=True)
class ReferenceInfo:
    """Name, cell space, provenance and class support of a reference shard, on one class axis.

    Attributes:
        name: Name of the shard.
        axis: Class axis the support is counted on.
        cell_space_id: Identity of the cell space the shard was projected into, or `undeclared`.
        n_cells: Rows of the shard, whether or not they carry a class of the axis.
        n_scored_cells: Rows carrying a class of the axis, which form the retrieval pool.
        class_order: Classes of the axis, in the order that fixes their codes.
        class_support: Reference cells of each class, in class order.
        source: Where the atlas comes from, as the shard or the registry beside it records it, or
            `undeclared`.
        license: Terms the atlas is distributed under, recorded likewise, or `undeclared`.
        scconcept_model: Single-cell model the embeddings of the shard come from, or `undeclared`.
        scconcept_version: Version of that model, or `undeclared`.
        coverage_verified: Whether the shard stamps the class support of the axis, found equal to
            the one its labels give; a stamp that differs is refused.
    """

    name: str
    axis: str
    cell_space_id: str
    n_cells: int
    n_scored_cells: int
    class_order: tuple[str, ...]
    class_support: dict[str, int]
    source: str
    license: str
    scconcept_model: str
    scconcept_version: str
    coverage_verified: bool = False

    @property
    def missing_classes(self) -> tuple[str, ...]:
        """Classes of the axis the reference holds no cell for."""
        return _missing_classes(self.class_support)

    @property
    def thin_classes(self) -> tuple[str, ...]:
        """Classes the reference supports too thinly for retrieval to be reliable."""
        return _thin_classes(self.class_support)


@dataclass(frozen=True)
class SelfCheckReport:
    """Leave-one-out retrieval of a reference against itself.

    A reference that cannot recover its own labels under the default operator cannot support
    annotation from images. A reference that can is not thereby shown to support it.

    Attributes:
        reference: Name of the reference checked.
        axis: Class axis its labels are read on.
        k: Neighbors each cell voted over, its own row removed.
        tau: Softmax temperature over the neighbor similarities.
        macro_f1: Mean F1 over the classes the scored cells carry.
        accuracy: Share of the cells scored whose own label was recovered.
        per_class_recall: Recall of every class of the axis, NaN for a class no scored cell carries.
        similarity_quantiles: Quantiles of the similarities to the neighbors voted over, keyed
            `p01` to `p99`.
        n_cells_evaluated: Cells scored: every cell of the reference, or a subsample of `max_cells`.
        warnings: Every reading the numbers depend on, such as a subsample or an absent class, one
            sentence or two each.
    """

    reference: str
    axis: str
    k: int
    tau: float
    macro_f1: float
    accuracy: float
    per_class_recall: dict[str, float]
    similarity_quantiles: dict[str, float]
    n_cells_evaluated: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoreReport:
    """Agreement between predicted and known labels.

    Two baselines travel with the headline, over the same cells and macro denominator: the
    commonest-class baseline, which gives every cell the commonest class, and the frequency-matched
    baseline, computed from its expected counts.

    Attributes:
        macro_f1: Mean F1 over the classes entering the macro average.
        balanced_accuracy: Mean recall over the classes entering the macro average.
        accuracy: Share of the scored cells predicted correctly. A cell without a prediction is an
            error.
        n_cells: Cells scored, whether or not a prediction was made for them.
        n_present_classes: Classes entering the macro average, the realized macro denominator.
        class_order: Classes of the axis, in the order indexing the confusion matrix.
        per_class: Support, predicted support, precision, recall and F1 of every class, and whether
            it enters the macro average.
        confusion: Cells of each known class (rows) predicted as each class (columns). A cell
            without a prediction enters no column.
        floor_macro_f1: Macro F1 of the commonest-class baseline.
        floor_accuracy: Accuracy of the commonest-class baseline.
        macro_min_support: Cells a class must carry to enter the macro average.
        frequency_matched_macro_f1: Macro F1 of the frequency-matched baseline: the mean share of
            the scored cells over the classes entering the average.
        missing_classes: Classes of the axis the reference held no cell of, as the annotations
            scored record them; empty where labels were scored without them.
    """

    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    n_cells: int
    n_present_classes: int
    class_order: tuple[str, ...]
    per_class: dict[str, dict[str, float]]
    confusion: np.ndarray
    floor_macro_f1: float
    floor_accuracy: float
    macro_min_support: int
    frequency_matched_macro_f1: float = float("nan")
    missing_classes: tuple[str, ...] = ()


def _selected_rows(keep: Any, n_rows: int, unit: str = "cell") -> np.ndarray:
    """Return the positions a mask or a list of positions selects, refusing anything else."""
    rows = np.asarray(keep)
    if rows.dtype == bool:
        if rows.shape != (n_rows,):
            raise AlhenaValueError(
                f"A mask selects {unit}s with one entry per {unit}, {n_rows}, and has shape "
                f"{rows.shape}."
            )
        return np.flatnonzero(rows)
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise AlhenaValueError(
            f"{unit.capitalize()}s are selected by a boolean mask or by integer positions, and an "
            f"array of dtype {rows.dtype} and shape {rows.shape} was given."
        )
    outside = rows[(rows < 0) | (rows >= n_rows)]
    if outside.size:
        raise AlhenaValueError(
            f"Positions of the {n_rows} {unit}s run from 0 to {n_rows - 1}, and "
            f"{sorted(set(outside.tolist()))[:5]} lie outside them."
        )
    if np.unique(rows).size != rows.size:
        raise AlhenaValueError(f"A position names a {unit} twice. Name each {unit} once.")
    return rows.astype(np.int64)


def _cell_positions(cells: Any) -> tuple[np.ndarray, str | None]:
    """Return the positions of cells, x then y, and the frame they record, refusing no position."""
    if isinstance(cells, AnnotationResult):
        if cells.cells is None:
            raise AlhenaValueError(
                "These annotations carry no cells, so they carry no positions. CellTyper.annotate "
                "attaches them; otherwise pass the CellEmbeddings the annotations were made from."
            )
        cells = cells.cells
    if isinstance(cells, CellEmbeddings):
        if cells.centroids is None:
            raise AlhenaValueError(
                "These cells record no position: their tiles shared no frame. Pass the positions "
                "of the cells as an array, in the frame of the tiles."
            )
        return np.asarray(cells.centroids, dtype=np.float64), cells.centroids_frame
    if hasattr(cells, "obsm") and hasattr(cells, "uns"):
        record = cells.uns.get(CELLS_PROVENANCE_KEY)
        if not isinstance(record, Mapping) or SPATIAL_KEY not in cells.obsm:
            raise AlhenaValueError(
                f"This table is not a table of cells ALHENA wrote with their positions: it lacks "
                f"{CELLS_PROVENANCE_KEY!r} in uns or {SPATIAL_KEY!r} in obsm. Pass the positions "
                "as an array, in the frame of the tiles."
            )
        frame = record.get("centroids_frame")
        return np.asarray(cells.obsm[SPATIAL_KEY], dtype=np.float64), (
            None if frame is None else str(frame)
        )
    try:
        positions = np.asarray(cells, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlhenaValueError(
            f"Cells are located from CellEmbeddings, an AnnotationResult, a table of cells or an "
            "(n_cells, 2) array of positions, and an object of type "
            f"{type(cells).__name__} was given."
        ) from error
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise AlhenaValueError(
            f"Positions have shape {positions.shape}, where an (n_cells, 2) array of x and y is "
            "expected."
        )
    return positions, None
