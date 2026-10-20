"""Embedding tiles inside sopa's patch pipeline."""

from __future__ import annotations

import warnings
from typing import Any, Final

import torch
from torch import nn

from alhena._constants import INPUT_SIZE
from alhena._defaults import DEFAULT_BATCH_SIZE, _batch_size
from alhena._errors import (
    PACKAGE_PREFIXES,
    AlhenaValueError,
    BundleError,
    GeometryError,
    GeometryWarning,
)
from alhena._geometry import _checked_mpp, _format_mpp, _target_scale
from alhena.encoder import TILE_EMBEDDING_CONTRACT, AlhenaEncoder
from alhena.spatial._elements import _require

# A batch on the unit scale may round just above one; anything further is a batch of bytes.
_UNIT_SCALE_LIMIT: Final[float] = 1.0 + 1e-3


class Alhena(nn.Module):
    """ALHENA's backbone behind the tile-embedding hook of a patch pipeline.

    The pipeline hands over unnormalized tiles on the unit scale and never sets evaluation mode, so
    the module normalizes them, is in evaluation mode from construction, and moves each batch to the
    device of its weights. The pipeline stores the result under `TILE_EMBEDDINGS_KEY`, the class
    name followed by `_embeddings`; a subclass under its own name.

    Args:
        encoder: The encoder `load_encoder` returns, whose backbone is shared.

    Attributes:
        embed_dim: Width of the embedding one tile produces.
        input_size: Side, in pixels, of the canvas the model reads.
        reference_mpp: Microns per pixel that canvas is meant to cover.

    Raises:
        BundleError: If `encoder` is not one `load_encoder` returns.
    """

    def __init__(self, encoder: AlhenaEncoder) -> None:
        super().__init__()
        for attribute in ("info", "as_module"):
            if not hasattr(encoder, attribute):
                raise BundleError(
                    f"The encoder carries no {attribute}. Build it with alhena.load_encoder or "
                    "with alhena.encoder.load_bundle."
                )
        self.embedder = encoder.as_module()
        self.embed_dim = int(encoder.info.embed_dim)
        self.input_size = int(encoder.info.input_size)
        self.reference_mpp = float(encoder.info.reference_mpp)
        self.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Embed a batch of tiles given on the unit scale.

        Args:
            images: A `(B, 3, W, W)` float tensor with values in [0, 1], `W` the input size the
                model reads.

        Returns:
            A `(B, embed_dim)` float32 tensor, not normalized.

        Raises:
            GeometryError: If the batch is not RGB tiles of the input size on the unit scale.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise GeometryError(
                "Expected a (B, 3, W, W) tensor of RGB tiles on the unit scale, got shape "
                f"{tuple(images.shape)}."
            )
        height, width = int(images.shape[2]), int(images.shape[3])
        if (height, width) != (self.input_size, self.input_size):
            raise GeometryError(
                f"These patches are {height} by {width} pixels where the model reads "
                f"{self.input_size} by {self.input_size}. Set patch_width={self.input_size}, and "
                "pick the level that puts the field of view where you want it."
            )
        if images.numel() and float(images.max()) > _UNIT_SCALE_LIMIT:
            raise GeometryError(
                f"These patches reach {float(images.max()):.3f}, above the unit scale this module "
                "reads. Hand it bytes divided by 255: it normalizes them itself."
            )
        return self.embedder(images.to(device=self._weights_device()))

    def _weights_device(self) -> torch.device:
        """Return the device the weights are on, which a pipeline may have moved them to."""
        for tensor in self.embedder.parameters():
            return tensor.device
        for tensor in self.embedder.buffers():
            return tensor.device
        return torch.device("cpu")


TILE_EMBEDDINGS_KEY: Final[str] = f"{Alhena.__name__}_embeddings"
"""Key a patch pipeline stores the tile embeddings under, after the class name `Alhena`."""


def tile_embeddings(
    sdata: Any,
    encoder: AlhenaEncoder,
    patch_width: int = INPUT_SIZE,
    patch_overlap: int = 0,
    mpp: float | None = None,
    level: int | None = 0,
    magnification: int | None = None,
    image_key: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    roi_key: str | None = "region_of_interest",
    key_added: str | None = None,
) -> str:
    """Embed the patches of an image inside sopa's patch pipeline, and store them in the object.

    The pipeline reads the patches at the level or the magnification chosen, never resampled to
    the encoder's resolution, so that choice sets the field of view: `mpp` has it checked, else a
    `GeometryWarning` says it goes unchecked. For a `TileEmbeddings` result, read the image with
    `tile_source` and embed it with `AlhenaEncoder.encode_tiles`.

    Args:
        sdata: The object holding the image.
        encoder: The loaded encoder.
        patch_width: Side of a patch, in pixels: the input size the model reads.
        patch_overlap: Overlap between neighboring patches, in pixels.
        mpp: Microns per pixel of a patch as the model reads it, after any resizing to
            `magnification`.
        level: Image level the patches are read from.
        magnification: Target magnification, resolving the level when it is given instead.
        image_key: Image to read, needed only when the object holds several.
        batch_size: Patches per forward pass, run on the device the encoder was loaded on.
        roi_key: Shapes element patches must touch to be kept; every patch is kept when None.
        key_added: Name the embeddings are stored under. Defaults to `TILE_EMBEDDINGS_KEY`.

    Returns:
        The key the embeddings were stored under.

    Raises:
        AlhenaValueError: If the patch width is not the input size or an argument is out of range.
        MissingExtraError: If the spatial extra is not installed.
    """
    batch_size = _batch_size(batch_size)
    # The pipeline moves the module to the device it is given: the encoder's own, so it stays.
    loaded_on = getattr(encoder, "device", None)
    device = None if loaded_on is None else str(loaded_on)
    module = Alhena(encoder)
    patch_width = int(patch_width)
    if patch_width != module.input_size:
        raise AlhenaValueError(
            f"patch_width is {patch_width} where the model reads {module.input_size}. Patches "
            "of another width are read over another field of view."
        )

    if mpp is None:
        warnings.warn(
            "The resolution of the patches was not declared, so nothing here checks that "
            "they cover the field of view the encoder reads, "
            f"{module.input_size * module.reference_mpp:.1f} microns. Pass mpp= to have it "
            "checked.",
            GeometryWarning,
            skip_file_prefixes=PACKAGE_PREFIXES,
        )
    else:
        # Not warn=True: that warning tells of a resampling to the encoder's resolution.
        mpp = _checked_mpp(mpp, target_mpp=module.reference_mpp)
        ratio = _target_scale(mpp, module.reference_mpp)
        if ratio != 1.0:
            warnings.warn(
                f"Patches are at {_format_mpp(mpp)} micron per pixel against the "
                f"{_format_mpp(module.reference_mpp)} the "
                f"encoder reads, a factor of {ratio:.2f}. The patch pipeline never resamples "
                "patches to the encoder's resolution: read another level or magnification. "
                f"{TILE_EMBEDDING_CONTRACT}",
                GeometryWarning,
                skip_file_prefixes=PACKAGE_PREFIXES,
            )

    patches = _require("sopa.patches")
    return str(
        patches.compute_embeddings(
            sdata,
            module,
            patch_width=patch_width,
            patch_overlap=int(patch_overlap),
            level=level,
            magnification=magnification,
            image_key=image_key,
            batch_size=batch_size,
            device=device,
            roi_key=roi_key,
            key_added=TILE_EMBEDDINGS_KEY if key_added is None else key_added,
        )
    )
