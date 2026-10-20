"""Load the released encoder, and annotate cells with it against a labeled reference.

The encoder is loaded only when tiles are given; an encoder or cells from another training run than
the reference are refused by their cell space identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from alhena._constants import (
    DEFAULT_AXIS,
    DEFAULT_K,
    DEFAULT_TAU,
    TISSUE_TO_SHARD,
    WEIGHTS_REPO,
)
from alhena._defaults import DEFAULT_BATCH_SIZE, _batch_size, _retrieval_workers
from alhena._errors import (
    AlhenaValueError,
    BundleError,
    CellSpaceMismatchError,
    ReferenceError,
)
from alhena._shards import _refuse_file_as_tissue
from alhena.annotate import annotate_cells
from alhena.cells._path import _encode_plan, _inspect_cells, _plan_cells
from alhena.encoder import AlhenaEncoder, load_bundle
from alhena.encoder._inference import _resolve_device
from alhena.types import (
    AnnotationResult,
    CellEmbeddings,
    GeometryReport,
    NucleiInput,
    TileInput,
)

if TYPE_CHECKING:
    from alhena.reference import Reference


# --- Loading --------------------------------------------------------------------------------


def load_encoder(
    *,
    bundle: str | Path | None = None,
    backbone: str | Path | None = None,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
    merge: bool = False,
    progress: bool | None = None,
) -> AlhenaEncoder:
    """Load the released encoder.

    Args:
        bundle: Directory holding the bundle files, or a repository `owner/name`. Defaults to the
            published repository.
        backbone: The UNI2-h weight file, or a directory holding one. Looked for in
            `ALHENA_BACKBONE` and the Hugging Face cache when omitted.
        device: Device to run on, such as `"cpu"`, `"cuda"`, or `"mps"` for the GPU of an Apple
            silicon Mac. Defaults to CUDA where available, the processor otherwise.
        cache_dir: Directory the bundle is downloaded into.
        merge: Fold the adapters into the fused weights instead of inserting them as modules; the
            forward pass then agrees with the published one to float32 rounding.
        progress: Whether to announce on standard error that the UNI2-h weights are being read. None
            follows `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        The encoder, on the resolved device.

    Raises:
        AlhenaValueError: If `device` names no device this machine has.
        BackboneNotFoundError: If the UNI2-h weights cannot be located.
        BundleError: If the bundle cannot be resolved, is incomplete or fails its digest checks.
    """
    from alhena.hub import ensure_bundle, find_backbone

    # Before anything is looked for or downloaded.
    _resolve_device(device)
    backbone_file = Path(find_backbone(backbone))
    directory = Path(ensure_bundle(bundle or WEIGHTS_REPO, cache_dir=cache_dir))
    return load_bundle(directory, backbone_file, device=device, merge=merge, progress=progress)


def _arguments(names: Sequence[str]) -> str:
    """Name arguments at the start of a sentence: `The argument a`, `The arguments a, b and c`."""
    if len(names) == 1:
        return f"The argument {names[0]}"
    return f"The arguments {', '.join(names[:-1])} and {names[-1]}"


def _load_reference(
    tissue: str | None,
    path: str | Path | None,
    axis: str,
    cache_dir: str | Path | None,
    progress: bool | None = None,
) -> Reference:
    """Load a reference shard through the reference layer, imported on use."""
    from alhena.reference import load_reference

    if path is not None:
        return load_reference(path=Path(path), axis=axis, cache_dir=cache_dir, progress=progress)
    return load_reference(tissue=tissue, axis=axis, cache_dir=cache_dir, progress=progress)


def _resolve_reference(
    tissue: str | None,
    reference: Reference | str | Path | None,
    axis: str | None,
    cache_dir: str | Path | None,
    progress: bool | None = None,
) -> Reference:
    """Return the reference to annotate against, loading it when it was named rather than given."""
    if isinstance(reference, (str, Path)):
        if tissue is not None:
            raise AlhenaValueError(
                f"Both a tissue, {tissue!r}, and a reference path, {reference!r}, were given. "
                "Name one of them: the shard a path points at is not looked up by tissue."
            )
        return _load_reference(None, reference, axis or DEFAULT_AXIS, cache_dir, progress)
    if reference is not None:
        if tissue is not None:
            raise AlhenaValueError(
                f"Both a tissue, {tissue!r}, and a loaded reference were given. Name one of them."
            )
        loaded_axis = getattr(getattr(reference, "info", None), "axis", None)
        if axis is not None and loaded_axis is not None and loaded_axis != axis:
            raise AlhenaValueError(
                f"The reference is labeled on {loaded_axis!r} and {axis!r} was asked for. Ask "
                "for the axis of the reference, or leave axis out: two axes describe different "
                "cell populations."
            )
        return reference
    if tissue is None:
        raise AlhenaValueError(
            "Name the tissue to annotate against, or pass a reference. Tissues with a prebuilt "
            "reference: "
            f"{sorted(TISSUE_TO_SHARD)}, each also accepted under the aliases "
            "alhena.TISSUE_ALIASES declares."
        )
    _refuse_file_as_tissue(tissue, "the reference")
    return _load_reference(tissue, None, axis or DEFAULT_AXIS, cache_dir, progress)


def load_celltyper(
    tissue: str | None = None,
    reference: Reference | str | Path | None = None,
    axis: str | None = None,
    *,
    encoder: AlhenaEncoder | None = None,
    bundle: str | Path | None = None,
    backbone: str | Path | None = None,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
    merge: bool = False,
    progress: bool | None = None,
) -> CellTyper:
    """Load a reference, and pair it with the encoder into a cell typer.

    The reference is loaded now, the encoder only when the typer is first given tiles, with the
    arguments `load_encoder` takes.

    Args:
        tissue: Tissue whose prebuilt reference is loaded: one of `TISSUE_TO_SHARD`, an alias
            `TISSUE_ALIASES` declares for one, or a shard name.
        reference: An already-loaded reference, or the path of a shard, instead of a tissue.
        axis: Class axis the cells are annotated on: a declared axis, or one a shard given as
            `reference` was built on. A loaded reference's own when None, else `DEFAULT_AXIS`.
        encoder: An encoder already loaded, checked against the reference at once, instead of
            `bundle`, `backbone`, `device` and `merge`.
        bundle: Directory holding the bundle files, or a repository `owner/name`, as `load_encoder`
            takes it.
        backbone: The UNI2-h weight file, or a directory holding one, as `load_encoder` takes it.
        device: Device the encoder runs on, as `load_encoder` takes it.
        cache_dir: Directory the reference and the bundle are downloaded into.
        merge: Fold the adapters into the fused weights, as `load_encoder` does.
        progress: Whether to report progress on standard error while the reference loads, and later
            while the encoder does. None follows `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        A cell typer holding the reference, and the encoder where one was given.

    Raises:
        AlhenaValueError: If no tissue or reference is given, or an argument is invalid.
        CellSpaceMismatchError: If the reference records no cell space, or not the encoder's.
        ReferenceError: If the reference cannot be loaded or does not fit the encoder.
    """
    if encoder is not None:
        building = [
            name
            for name, value in (("bundle", bundle), ("backbone", backbone), ("device", device))
            if value is not None
        ]
        if merge:
            building.append("merge")
        if building:
            verb = "describes" if len(building) == 1 else "describe"
            raise AlhenaValueError(
                f"{_arguments(building)} {verb} how to build an encoder, and an encoder was "
                "already given. Pass one or the other."
            )
    elif device is not None:
        _resolve_device(device)
    typer = CellTyper(encoder, _resolve_reference(tissue, reference, axis, cache_dir, progress))
    if encoder is None:
        typer._encoder_options = {
            "bundle": bundle,
            "backbone": backbone,
            "device": device,
            "cache_dir": cache_dir,
            "merge": bool(merge),
        }
        typer._progress = progress
    return typer


# --- The cell typer -------------------------------------------------------------------------


class CellTyper:
    """A labeled reference, and the encoder that projects cells into its space.

    `load_celltyper` builds it. A typer built without an encoder loads one when first given tiles;
    any encoder is checked against the reference's cell space identity before it is kept.

    Args:
        encoder: An encoder, or None to load one when the typer is first given tiles.
        reference: A reference as `load_reference` returns it.

    Attributes:
        encoder: The adapted backbone and its projection heads, or None until it is loaded.
        reference: The labeled reference cells the annotations are retrieved from.

    Raises:
        BundleError: If the encoder carries no cell head.
        CellSpaceMismatchError: If the reference records no cell space, or not the encoder's.
        ReferenceError: If the reference is malformed or does not fit the encoder.
    """

    def __init__(self, encoder: AlhenaEncoder | None, reference: Reference) -> None:
        for attribute in ("info", "embeddings", "label_ids", "class_order", "require_cell_space"):
            if not hasattr(reference, attribute):
                raise ReferenceError(
                    f"The reference carries no {attribute}. Load it with "
                    "alhena.reference.load_reference."
                )
        shape = getattr(reference.embeddings, "shape", ())
        if len(shape) != 2:
            raise ReferenceError(
                f"The reference embeddings have shape {tuple(shape)}, where an (n_cells, d) "
                "array is expected."
            )
        # Before any encoder is loaded: a reference recording no cell space passes no check.
        reference.require_cell_space()
        self.reference = reference
        self.encoder: AlhenaEncoder | None = None
        # How the encoder is built when first needed: as load_celltyper was told, else published.
        self._encoder_options: dict[str, Any] = {}
        self._progress: bool | None = None
        if encoder is not None:
            self._attach(encoder)

    def __repr__(self) -> str:
        return (
            f"CellTyper(reference={self.reference.info.name!r}, axis={self.axis!r}, "
            f"cell_space_id={self.cell_space_id!r})"
        )

    @property
    def axis(self) -> str:
        """Class axis the reference is labeled on, and the cells are annotated on."""
        return str(self.reference.info.axis)

    @property
    def class_order(self) -> tuple[str, ...]:
        """Class names, in the order that indexes the columns of the mass."""
        return tuple(self.reference.class_order)

    @property
    def cell_space_id(self) -> str:
        """Identity of the space the reference lies in, which the encoder and cells must share."""
        return str(self.reference.info.cell_space_id)

    def load_encoder(self, progress: bool | None = None) -> AlhenaEncoder:
        """Return the encoder, loading it the first time it is needed.

        It is loaded with the arguments `load_celltyper` was given, or as the published encoder, and
        checked against the reference before it is kept.

        Args:
            progress: Whether to announce on standard error that the UNI2-h weights are being read.
                None follows `load_celltyper`, then `ALHENA_DISABLE_PROGRESS_BARS`.

        Returns:
            The encoder.

        Raises:
            BackboneNotFoundError: If the UNI2-h weights cannot be located.
            BundleError: If the bundle cannot be resolved, or carries no cell head.
            CellSpaceMismatchError: If the encoder and the reference lie in different cell spaces.
            ReferenceError: If the reference does not fit the encoder.
        """
        if self.encoder is not None:
            return self.encoder
        chosen = self._progress if progress is None else progress
        encoder = load_encoder(**self._encoder_options, progress=chosen)
        self._attach(encoder)
        return encoder

    def _attach(self, encoder: AlhenaEncoder) -> None:
        """Check an encoder against the reference, and keep it."""
        for attribute in ("info", "encode_with_tokens", "project_cell"):
            if not hasattr(encoder, attribute):
                raise BundleError(
                    f"The encoder carries no {attribute}. Build it with load_encoder or with "
                    "alhena.encoder.load_bundle."
                )
        if not encoder.info.has_cell_proj:
            raise BundleError(
                "This bundle carries no cell head, so cells cannot be projected into the space "
                "the reference lives in. Cell-type annotation needs both image-side and "
                "reference-side cell projections."
            )
        self.reference.require_cell_space(encoder.info.cell_space_id)
        # A width check catches another model; another run of it needs the cell space identity.
        width = int(self.reference.embeddings.shape[1])
        if width != encoder.info.cell_dim:
            raise ReferenceError(
                f"The reference is {width}-dimensional and the encoder projects cells into "
                f"{encoder.info.cell_dim} dimensions."
            )
        self.encoder = encoder

    def _check_stored(self, cells: CellEmbeddings, given: dict[str, Any]) -> None:
        """Refuse arguments that describe tiles, and cells from another cell space."""
        passed = sorted(
            name for name, value in given.items() if value is not None and value is not False
        )
        if passed:
            verb = "describes" if len(passed) == 1 else "describe"
            raise AlhenaValueError(
                f"{_arguments(passed)} {verb} tiles to read, and these cells were read already: "
                "they record their nuclei, their resolution and their identifiers. Pass the cells "
                "alone, with k and tau."
            )
        if cells.cell_space_id is None:
            raise CellSpaceMismatchError(
                "These cells record no cell space identity, so nothing shows they lie in the "
                f"space of the {self.reference.info.name} reference. Annotate embeddings without "
                "one with alhena.annotate_cells, which takes the space on the caller's word."
            )
        self.reference.require_cell_space(str(cells.cell_space_id))

    def _record(self, cells: CellEmbeddings) -> dict[str, Any]:
        """Return the provenance a call records beside its operating point, read off its cells."""
        report = cells.geometry
        return {
            "axis": self.axis,
            "encoder_id": cells.encoder_id,
            "backbone_digest": cells.backbone_digest,
            "reference": str(self.reference.info.name),
            "mpp": cells.mpp,
            "geometry_declared": None if report is None else report.mpp is not None,
            "field_of_view_um": None if report is None else report.field_of_view_um,
            "pad_fraction": None if report is None else report.pad_fraction,
            "crop_fraction": None if report is None else report.crop_fraction,
            "geometry_warnings": () if report is None else tuple(report.warnings),
            "membership_source": cells.membership_source,
            "membership_approximate": cells.membership_approximate,
            "n_cells_without_tokens": cells.n_cells_without_tokens,
        }

    def preflight(
        self,
        tiles: TileInput,
        nuclei: NucleiInput,
        mpp: float | None = None,
    ) -> GeometryReport:
        """Describe the geometry the tiles and nuclei present, and warn about it.

        Built from shapes alone, with no encoder, it checks the nuclei as `annotate` does. Tiles
        whose size is known only once they are read, as image files are, are checked by `annotate`
        instead.

        Args:
            tiles: Raw uint8 RGB tiles, a tile source, or a batch already prepared for the model.
            nuclei: Label maps, the mapping an `alhena.io` reader of nuclei returns, or a membership
                table.
            mpp: Microns per pixel of the tiles.

        Returns:
            The field of view, the microns one token covers, and any geometry concern.

        Raises:
            AlhenaValueError: If `mpp` is out of range, or the tiles and the nuclei do not match.
            GeometryError: If the nuclei of a tile are drawn on another pixel grid than the tile.
            MembershipError: If a membership table is malformed or built at another resolution.
        """
        return _inspect_cells(tiles, nuclei, mpp).report

    def annotate(
        self,
        tiles: TileInput | CellEmbeddings,
        nuclei: NucleiInput | None = None,
        mpp: float | None = None,
        *,
        cell_id: Sequence[Any] | np.ndarray | None = None,
        all_tiles: bool = False,
        batch_size: int | None = None,
        k: int = DEFAULT_K,
        tau: float = DEFAULT_TAU,
        workers: int | None = None,
        progress: bool | None = None,
    ) -> AnnotationResult:
        """Annotate every nucleus of a set of tiles, or every cell encoded earlier.

        Given tiles and nuclei, it runs `AlhenaEncoder.encode_cells`, then retrieval, with the
        geometry and the nuclei checked before the encoder is loaded. Given the `CellEmbeddings` of
        an earlier run, it retrieves alone, against a reference in their cell space. The result
        carries the cells on `cells`; cells reaching no token are left out and counted on `meta`.

        Args:
            tiles: Raw uint8 RGB tiles, a tile source, or a batch already prepared for the model;
                or the `CellEmbeddings` of an earlier run.
            nuclei: Label maps, the mapping an `alhena.io` reader of nuclei returns, or a membership
                table. Needed with tiles, refused with cells.
            mpp: Microns per pixel of the tiles, declared once for the images and the nuclei alike.
                Refused with cells, which record it.
            cell_id: Unique identifier of each nucleus, for the builders that take positions rather
                than labels. Refused with cells.
            all_tiles: Read every tile, including those no cell is read from. Refused with cells.
            batch_size: Tiles per forward pass, which bounds the memory the pass holds; 32 when not
                given. Refused with cells.
            k: Number of reference neighbors each cell votes over.
            tau: Softmax temperature over neighbor similarities.
            workers: Threads retrieving neighbors at once. None uses the processors available, at
                most `MAX_DEFAULT_WORKERS`. The annotations are the same for any number.
            progress: Whether to report progress on standard error. None follows
                `ALHENA_DISABLE_PROGRESS_BARS`.

        Returns:
            The annotations, their mass, confidence and entropies, with the provenance of the run on
            `meta` and the cell embeddings on `cells`.

        Raises:
            AlhenaValueError: If an argument is invalid for the input, or the nuclei cannot be read.
            BackboneNotFoundError: If the UNI2-h weights are needed and cannot be located.
            BundleError: If the encoder has to be loaded and its bundle cannot be resolved.
            CellSpaceMismatchError: If the cells or encoder are not in the reference's cell space.
            GeometryError: If the nuclei of a tile are drawn on another pixel grid than the tile.
            MembershipError: If the nuclei and the tiles cannot be reconciled.
        """
        k = int(k)
        if k < 1:
            raise AlhenaValueError(f"k must be at least 1, got {k}.")
        tau = float(tau)
        if tau <= 0.0:
            raise AlhenaValueError(f"tau must be positive, got {tau}.")
        threads = _retrieval_workers(workers)

        if isinstance(tiles, CellEmbeddings):
            given = {
                "nuclei": nuclei,
                "mpp": mpp,
                "cell_id": cell_id,
                "all_tiles": all_tiles,
                "batch_size": batch_size,
            }
            self._check_stored(tiles, given)
            cells = tiles
        else:
            if nuclei is None:
                raise AlhenaValueError(
                    "Tiles are annotated through their nuclei, and none were given. Pass nuclei "
                    "beside the tiles, or pass the CellEmbeddings of an earlier run alone."
                )
            size = DEFAULT_BATCH_SIZE if batch_size is None else _batch_size(batch_size)
            plan = _plan_cells(
                tiles,
                nuclei,
                mpp,
                cell_id=cell_id,
                all_tiles=all_tiles,
                progress=progress,
            )
            encoder = self.load_encoder(progress)
            # Checked on every call too, since the reference a typer holds may have been replaced.
            self.reference.require_cell_space(encoder.info.cell_space_id)
            cells = _encode_plan(encoder, plan, batch_size=size, progress=progress)

        result = annotate_cells(
            cells,
            reference_embeddings=self.reference.embeddings,
            reference_labels=self.reference.label_ids,
            class_order=self.reference.class_order,
            k=k,
            tau=tau,
            axis=self.axis,
            cell_space_id=self.cell_space_id,
            progress=progress,
            workers=threads,
        )
        result.meta.update(self._record(cells))
        result.cells = cells
        return result
