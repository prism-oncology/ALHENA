"""Building the encoder from the backbone, the adapters and the heads of a bundle."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from alhena import _progress
from alhena._bundle import _check_stated_identity, _declared_property, bundle_identity
from alhena._constants import (
    ADAPTERS_FILE,
    BACKBONE_ACT_LAYER,
    BACKBONE_ARCHITECTURE,
    BACKBONE_DIM,
    BACKBONE_MLP_LAYER,
    BACKBONE_REPO,
    BACKBONE_TIMM_KWARGS,
    BUNDLE_CONFIG_FILE,
    CELL_SPACE_DIM,
    HEADS_FILE,
    IMAGE_INTERPOLATION,
    IMAGE_MEAN,
    IMAGE_STD,
    INPUT_SIZE,
    LORA_ALPHA,
    LORA_RANK,
    PATCH_SIZE,
    REFERENCE_MPP,
    SCCONCEPT_DIM,
    SCCONCEPT_MODEL,
    SCCONCEPT_VERSION,
    TOKEN_GRID,
)
from alhena._errors import BackboneNotFoundError, BundleError
from alhena._hub import _check_bundle_files, _is_repo_id, _normalize_digest, _read_json_object
from alhena._version import __version__
from alhena.encoder._adapters import insert_adapters, load_adapter_weights, merge_adapters
from alhena.encoder._inference import AlhenaEncoder, _resolve_device
from alhena.types import BundleInfo

# Quantities a bundle may state under `geometry`, checked against the constants of the package.
_DECLARED_GEOMETRY: tuple[tuple[str, Any], ...] = (
    ("input_size", INPUT_SIZE),
    ("patch_size", PATCH_SIZE),
    ("token_grid", TOKEN_GRID),
    ("reference_mpp", REFERENCE_MPP),
    ("image_mean", IMAGE_MEAN),
    ("image_std", IMAGE_STD),
    ("interpolation", IMAGE_INTERPOLATION),
)

# Relative tolerance on a declared number: a decimal round trip, and nothing wider.
_GEOMETRY_REL_TOL = 1e-9


def _backbone_file(backbone: str | Path) -> Path:
    """Return the UNI2-h weight file named, refusing a path that holds none."""
    path = Path(backbone)
    if not path.is_file():
        raise BackboneNotFoundError(
            f"No UNI2-h weight file at {path}. UNI2-h is gated and is not redistributed with "
            f"ALHENA: accept the conditions on {BACKBONE_REPO}, download its weight file, and "
            "pass its path."
        )
    return path


def _read_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Read a state dict from a safetensors file or a torch file."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise BundleError(
            f"{path} holds an object of type {type(loaded).__name__} where a state dict of "
            "tensors was expected."
        )
    return loaded


def build_backbone(
    backbone: str | Path,
    architecture: str = BACKBONE_ARCHITECTURE,
    timm_kwargs: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Instantiate the vision transformer and load the UNI2-h weights into it.

    The architecture is built locally and the weights read from the file named, with no network.

    Args:
        backbone: Path of the UNI2-h weight file, `pytorch_model.bin` or its safetensors form.
        architecture: timm architecture name.
        timm_kwargs: Constructor arguments; those of the released model by default.

    Returns:
        The transformer in float32, in evaluation mode, with gradients disabled.

    Raises:
        BackboneNotFoundError: If the weight file does not exist.
        BundleError: If the weights do not match the architecture.
    """
    import timm

    path = _backbone_file(backbone)
    kwargs = dict(BACKBONE_TIMM_KWARGS if timm_kwargs is None else timm_kwargs)
    kwargs.setdefault("mlp_layer", getattr(timm.layers, BACKBONE_MLP_LAYER))
    kwargs.setdefault("act_layer", getattr(nn, BACKBONE_ACT_LAYER))
    model = timm.create_model(architecture, pretrained=False, **kwargs)
    try:
        model.load_state_dict(_read_state_dict(path), strict=True)
    except RuntimeError as error:
        raise BundleError(
            f"The weights at {path} do not match {architecture} as ALHENA builds it. Check that "
            "the file is the UNI2-h checkpoint and not another vision transformer."
        ) from error

    model.float()
    model.eval()
    model.requires_grad_(False)
    return model


class _Heads(nn.Module):
    """The bundle's projection heads, any of which may be absent."""

    def __init__(self) -> None:
        super().__init__()
        self.image_proj: nn.Linear | None = None
        self.img_tok_proj: nn.Linear | None = None
        self.st_node_proj: nn.Linear | None = None


def build_heads(
    tensors: Mapping[str, torch.Tensor],
    embed_dim: int = BACKBONE_DIM,
    cell_dim: int = CELL_SPACE_DIM,
    scconcept_dim: int = SCCONCEPT_DIM,
) -> nn.Module:
    """Build the projection heads of a bundle from their weights.

    The heads are bias-free linear maps: `image_proj` and `img_tok_proj` read backbone features
    but were trained against different teachers, so they are not interchangeable, and
    `st_node_proj` reads single-cell embeddings. A head whose weight is absent is not built.

    Args:
        tensors: Head weights, keyed as in the bundle.
        embed_dim: Width the two image-side heads read.
        cell_dim: Width all three heads write.
        scconcept_dim: Width the reference head reads.

    Returns:
        A module holding the heads, in evaluation mode, with gradients disabled.

    Raises:
        BundleError: If a tensor is not one of the three head weights, or has the wrong shape.
    """
    shapes = {
        "image_proj": (cell_dim, embed_dim),
        "img_tok_proj": (cell_dim, embed_dim),
        "st_node_proj": (cell_dim, scconcept_dim),
    }
    unexpected = sorted(set(tensors) - {f"{name}.weight" for name in shapes})
    if unexpected:
        raise BundleError(
            f"Unexpected tensors among the head weights: {unexpected}. The heads are the three "
            f"bias-free maps {list(shapes)}."
        )

    heads = _Heads()
    for name, (out_features, in_features) in shapes.items():
        weight = tensors.get(f"{name}.weight")
        if weight is None:
            continue
        if tuple(weight.shape) != (out_features, in_features):
            raise BundleError(
                f"Head {name!r} has shape {tuple(weight.shape)}, expected "
                f"{(out_features, in_features)}."
            )
        layer = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            layer.weight.copy_(weight.to(dtype=torch.float32))
        setattr(heads, name, layer)

    heads.eval()
    heads.requires_grad_(False)
    return heads


def _weights_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    """Return the SHA-256 of weights, by name, shape and float32 values, whatever the file form."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        values = tensors[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(f"{name}|{tuple(values.shape)}|".encode())
        digest.update(values.numpy().reshape(-1).data)
    return digest.hexdigest()


def _declared_backbone_digest(directory: Path, config: Mapping[str, Any]) -> str | None:
    """Return the UNI2-h weights digest a bundle declares, or None, refusing any other value."""
    declared = _declared_property(directory, config, "backbone", "weights_digest", "")
    if not declared:
        return None
    return _normalize_digest(
        declared, f"the UNI2-h model weights of the bundle at {directory}", BundleError
    )


def _geometry_agrees(declared: Any, expected: Any) -> bool:
    """Report whether a declared quantity is the one the package is built around."""
    if isinstance(expected, str):
        return declared == expected
    if isinstance(expected, tuple):
        if not isinstance(declared, (list, tuple)) or len(declared) != len(expected):
            return False
        return all(
            _geometry_agrees(one, other) for one, other in zip(declared, expected, strict=True)
        )
    if not isinstance(declared, (int, float)):
        return False
    return math.isclose(float(declared), float(expected), rel_tol=_GEOMETRY_REL_TOL)


def _check_declared_geometry(directory: Path, config: Mapping[str, Any]) -> None:
    """Refuse a geometry a bundle states that differs from the package's; stating none is fine."""
    declared = config.get("geometry")
    if declared is None:
        return
    if not isinstance(declared, dict):
        raise BundleError(
            f"The bundle at {directory} states `geometry` as a value of type "
            f"{type(declared).__name__} where an object naming the canvas, the token grid, the "
            "resolution and the normalization is expected."
        )
    for name, expected in _DECLARED_GEOMETRY:
        if name not in declared:
            continue
        value = declared[name]
        if not _geometry_agrees(value, expected):
            raise BundleError(
                f"The bundle at {directory} declares geometry.{name} as {value!r} where this "
                f"installation of ALHENA reads {expected!r}. Install the version of ALHENA this "
                "bundle was published for, or load a bundle built for this version."
            )


def _resolve_bundle_directory(bundle: str | Path) -> Path:
    """Return the local directory of a bundle; a string `owner/name` that is no path is fetched."""
    directory = Path(bundle)
    if directory.is_dir():
        return directory
    name = str(bundle)
    if isinstance(bundle, str) and _is_repo_id(name) and not directory.exists():
        from alhena.hub import ensure_bundle

        return Path(ensure_bundle(name))
    raise BundleError(f"{name!r} is not a directory holding {ADAPTERS_FILE} and {HEADS_FILE}.")


def load_bundle(
    bundle: str | Path,
    backbone: str | Path,
    *,
    device: torch.device | str | None = None,
    merge: bool = False,
    progress: bool | None = None,
) -> AlhenaEncoder:
    """Build the encoder from a bundle and the user's UNI2-h weights.

    Args:
        bundle: Directory holding the bundle files, or the repository publishing them.
        backbone: Path of the UNI2-h weight file.
        device: Device to run on, such as `"cpu"`, `"cuda"`, or `"mps"` for the GPU of an Apple
            silicon Mac. A CUDA device is chosen when one is available, otherwise the processor.
        merge: Fold the adapters into the fused weights instead of inserting them, which agrees with
            the published forward pass to float32 rounding.
        progress: Whether to announce on standard error the reading of the UNI2-h weights. None
            follows `ALHENA_DISABLE_PROGRESS_BARS`.

    Returns:
        The encoder, on the resolved device.

    Raises:
        AlhenaValueError: If `device` names no device this machine has.
        BackboneNotFoundError: If the UNI2-h weight file does not exist.
        BundleError: If a file is missing or does not match what the bundle declares.
    """
    resolved_device = _resolve_device(device)
    directory = _resolve_bundle_directory(bundle)
    backbone_file = _backbone_file(backbone)
    _check_bundle_files(directory)
    adapters_file = directory / ADAPTERS_FILE
    heads_file = directory / HEADS_FILE

    config_file = directory / BUNDLE_CONFIG_FILE
    config = _read_json_object(config_file, BundleError) if config_file.is_file() else {}
    # Before any weight is read, so a bundle of another geometry costs a parse, not a load.
    _check_declared_geometry(directory, config)
    rank = _declared_property(directory, config, "adapters", "rank", LORA_RANK)
    alpha = _declared_property(directory, config, "adapters", "alpha", LORA_ALPHA)
    cell_dim = _declared_property(directory, config, "dims", "cell", CELL_SPACE_DIM)
    scconcept_model = _declared_property(
        directory, config, "teacher", "scconcept_model", SCCONCEPT_MODEL
    )
    scconcept_version = _declared_property(
        directory, config, "teacher", "scconcept_version", SCCONCEPT_VERSION
    )
    backbone_repository = _declared_property(
        directory, config, "backbone", "repository", BACKBONE_REPO
    )
    declared_digest = _declared_backbone_digest(directory, config)

    adapter_tensors = _read_state_dict(adapters_file)
    head_tensors = _read_state_dict(heads_file)
    encoder_id, cell_space_id = bundle_identity(
        adapters_file.read_bytes(), head_tensors, scconcept_model, scconcept_version
    )
    _check_stated_identity(directory, config, encoder_id, cell_space_id)

    _progress.step(f"Loading the UNI2-h weights from {backbone_file}", progress)
    vit = build_backbone(backbone_file)
    # Loaded strictly, so the state is the file's; read before the adapters enter it.
    backbone_digest = _weights_digest(vit.state_dict())
    if declared_digest is not None and backbone_digest != declared_digest:
        raise BundleError(
            f"The UNI2-h model weights at {backbone_file} are not those the bundle at {directory} "
            f"was trained on: their digest is {backbone_digest}, and the bundle declares "
            f"{declared_digest}. Download the weight file from {BACKBONE_REPO} again and pass "
            "its path, rather than a converted, fine-tuned or damaged copy."
        )
    insert_adapters(vit, rank=rank, alpha=alpha)
    load_adapter_weights(vit, adapter_tensors)
    if merge:
        merge_adapters(vit)

    embed_dim = int(vit.embed_dim)
    heads = build_heads(head_tensors, embed_dim=embed_dim, cell_dim=cell_dim)

    info = BundleInfo(
        encoder_id=encoder_id,
        cell_space_id=cell_space_id,
        backbone=backbone_repository,
        embed_dim=embed_dim,
        cell_dim=cell_dim,
        lora_rank=rank,
        lora_alpha=alpha,
        input_size=INPUT_SIZE,
        patch_size=PATCH_SIZE,
        reference_mpp=REFERENCE_MPP,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        interpolation=IMAGE_INTERPOLATION,
        has_image_proj=heads.image_proj is not None,
        has_cell_proj=heads.img_tok_proj is not None and heads.st_node_proj is not None,
        scconcept_model=scconcept_model,
        scconcept_version=scconcept_version,
        version=str(config.get("version", __version__)),
        backbone_digest=backbone_digest,
    )
    return AlhenaEncoder(vit, heads, info, device=resolved_device)
