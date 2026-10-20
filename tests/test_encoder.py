"""Tests for the model bundle: loading it, its low-rank adapters, its heads and its identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import timm
import torch
from safetensors.torch import save, save_file
from torch import nn

from alhena import (
    INPUT_SIZE,
    PATCH_SIZE,
    REFERENCE_MPP,
    AlhenaValueError,
    BackboneNotFoundError,
    BundleError,
    load_encoder,
    prepare_tiles,
)
from alhena._constants import BACKBONE_ARCHITECTURE, SCCONCEPT_DIM
from alhena.encoder import (
    ADAPTER_NAMES,
    adapter_key,
    build_heads,
    bundle_identity,
    insert_adapters,
    load_adapter_weights,
    load_bundle,
    merge_adapters,
)
from support import NUCLEI_PER_TILE, make_label_maps, make_tiles

TINY_DIM = 16
TINY_DEPTH = 2
TINY_RANK = 4
TINY_ALPHA = 2.0
TINY_CELL_DIM = 8
TEACHER_MODEL = "test-model"
TEACHER_VERSION = "v0"
BACKBONE_REPOSITORY = "test-owner/test-backbone"

HEAD_NAMES = ("image_proj", "img_tok_proj", "st_node_proj")


def tiny_vit(seed: int = 0) -> nn.Module:
    """Build a small transformer of the timm architecture UNI2-h uses, with its register tokens."""
    torch.manual_seed(seed)
    model = timm.create_model(
        BACKBONE_ARCHITECTURE,
        pretrained=False,
        img_size=INPUT_SIZE,
        patch_size=PATCH_SIZE,
        depth=TINY_DEPTH,
        num_heads=2,
        init_values=1e-5,
        embed_dim=TINY_DIM,
        mlp_ratio=2.0,
        num_classes=0,
        no_embed_class=True,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model.eval()
    model.requires_grad_(False)
    return model


def adapter_tensors(seed: int = 0) -> dict[str, torch.Tensor]:
    """Return the four adapter tensors of every block of the tiny transformer."""
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for index in range(TINY_DEPTH):
        for name in ADAPTER_NAMES:
            shape = (TINY_DIM, TINY_RANK) if name.endswith("downsample") else (TINY_RANK, TINY_DIM)
            tensors[adapter_key(index, name)] = torch.randn(shape, generator=generator) * 0.1
    return tensors


def head_tensors(seed: int = 1) -> dict[str, torch.Tensor]:
    """Return the weights of the three heads, the reference head reading single-cell widths."""
    generator = torch.Generator().manual_seed(seed)
    widths = {"image_proj": TINY_DIM, "img_tok_proj": TINY_DIM, "st_node_proj": SCCONCEPT_DIM}
    return {
        f"{name}.weight": torch.randn((TINY_CELL_DIM, widths[name]), generator=generator)
        for name in HEAD_NAMES
    }


def bundle_config() -> dict[str, Any]:
    """Return the configuration of the tiny bundle, in the published layout."""
    return {
        "adapters": {"rank": TINY_RANK, "alpha": TINY_ALPHA},
        "dims": {"embed": TINY_DIM, "cell": TINY_CELL_DIM},
        "teacher": {"scconcept_model": TEACHER_MODEL, "scconcept_version": TEACHER_VERSION},
        "backbone": {"architecture": BACKBONE_ARCHITECTURE, "repository": BACKBONE_REPOSITORY},
    }


def write_bundle(directory: Path, config: dict[str, Any] | None = None) -> Path:
    """Write a bundle of the tiny transformer: its configuration, adapters and heads."""
    directory.mkdir(parents=True)
    stated = bundle_config() if config is None else config
    (directory / "config.json").write_text(json.dumps(stated), encoding="utf-8")
    save_file(adapter_tensors(), str(directory / "adapters.safetensors"))
    save_file(head_tensors(), str(directory / "heads.safetensors"))
    return directory


def serve_tiny_backbone(monkeypatch: pytest.MonkeyPatch, directory: Path) -> Path:
    """Serve the tiny transformer in place of UNI2-h, and return a stand-in weight file."""
    monkeypatch.setattr("alhena.encoder._loading.build_backbone", lambda path: tiny_vit())
    path = directory / "pytorch_model.bin"
    path.write_bytes(b"")
    return path


def random_images(count: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    return torch.randn(count, 3, INPUT_SIZE, INPUT_SIZE, generator=generator)


def embed(vit: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return the class token a transformer gives images, as the encoder reads it."""
    with torch.inference_mode():
        return vit.forward_head(vit.forward_features(images), pre_logits=True)


# --- Loading a bundle -------------------------------------------------------------------------


def test_load_bundle_reads_the_identity_and_widths(tmp_path, monkeypatch):
    backbone = serve_tiny_backbone(monkeypatch, tmp_path)
    directory = write_bundle(tmp_path / "bundle")

    info = load_bundle(directory, backbone, device="cpu").info

    adapter_bytes = (directory / "adapters.safetensors").read_bytes()
    assert info.encoder_id == hashlib.sha256(adapter_bytes).hexdigest()[:16]
    identity = bundle_identity(adapter_bytes, head_tensors(), TEACHER_MODEL, TEACHER_VERSION)
    assert info.cell_space_id == identity[1]
    assert (info.embed_dim, info.cell_dim) == (TINY_DIM, TINY_CELL_DIM)
    assert (info.lora_rank, info.lora_alpha) == (TINY_RANK, TINY_ALPHA)
    assert (info.scconcept_model, info.scconcept_version) == (TEACHER_MODEL, TEACHER_VERSION)
    assert info.backbone == BACKBONE_REPOSITORY
    assert info.has_image_proj
    assert info.has_cell_proj


def test_load_bundle_encodes_with_the_adapters_of_the_bundle(tmp_path, monkeypatch):
    backbone = serve_tiny_backbone(monkeypatch, tmp_path)
    encoder = load_bundle(write_bundle(tmp_path / "bundle"), backbone, device="cpu")
    vit = tiny_vit()
    insert_adapters(vit, rank=TINY_RANK, alpha=TINY_ALPHA)
    load_adapter_weights(vit, adapter_tensors())
    tiles = make_tiles(n_tiles=2)
    batch, _ = prepare_tiles(tiles, mpp=REFERENCE_MPP)

    embeddings = encoder.encode_tiles(tiles, mpp=REFERENCE_MPP).embeddings
    cells = encoder.encode_cells(tiles, make_label_maps(n_tiles=2), mpp=REFERENCE_MPP)

    torch.testing.assert_close(torch.from_numpy(embeddings), embed(vit, batch))
    assert cells.backbone.shape == (2 * NUCLEI_PER_TILE, TINY_DIM)
    assert cells.cell_space.shape == (2 * NUCLEI_PER_TILE, TINY_CELL_DIM)
    assert np.isfinite(cells.cell_space).all()


def test_load_encoder_refuses_a_device_before_reading_anything():
    with pytest.raises(AlhenaValueError, match="'gpu' names no device"):
        load_encoder(device="gpu", backbone="absent.bin", bundle="absent")


def test_load_bundle_refuses_a_missing_backbone_file(tmp_path):
    directory = write_bundle(tmp_path / "bundle")
    with pytest.raises(BackboneNotFoundError, match="UNI2-h weight file"):
        load_bundle(directory, tmp_path / "absent.bin", device="cpu")


def test_load_bundle_refuses_a_bundle_missing_a_file(tmp_path, monkeypatch):
    backbone = serve_tiny_backbone(monkeypatch, tmp_path)
    directory = write_bundle(tmp_path / "bundle")
    (directory / "heads.safetensors").unlink()

    with pytest.raises(BundleError, match=r"missing heads\.safetensors"):
        load_bundle(directory, backbone, device="cpu")


@pytest.mark.parametrize(
    ("adapters", "match"),
    [
        pytest.param(TINY_RANK, "value of type int", id="section_not_an_object"),
        pytest.param({"rank": 4.5}, r"adapters\.rank as 4\.5", id="rank_not_an_integer"),
    ],
)
def test_load_bundle_refuses_a_malformed_config(tmp_path, monkeypatch, adapters, match):
    backbone = serve_tiny_backbone(monkeypatch, tmp_path)
    config = bundle_config()
    config["adapters"] = adapters

    with pytest.raises(BundleError, match=match):
        load_bundle(write_bundle(tmp_path / "bundle", config), backbone, device="cpu")


def test_load_bundle_refuses_an_identity_its_weights_do_not_produce(tmp_path, monkeypatch):
    backbone = serve_tiny_backbone(monkeypatch, tmp_path)
    config = {**bundle_config(), "cell_space_id": "0" * 16}

    with pytest.raises(BundleError, match="states cell_space_id"):
        load_bundle(write_bundle(tmp_path / "bundle", config), backbone, device="cpu")


# --- Adapters ---------------------------------------------------------------------------------


def test_insert_adapters_leaves_the_output_unchanged():
    vit = tiny_vit()
    images = random_images()
    before = embed(vit, images)

    assert insert_adapters(vit, rank=TINY_RANK, alpha=TINY_ALPHA) == TINY_DEPTH
    torch.testing.assert_close(embed(vit, images), before)


def test_load_adapter_weights_copies_the_tensors():
    vit = tiny_vit()
    insert_adapters(vit, rank=TINY_RANK, alpha=TINY_ALPHA)
    tensors = adapter_tensors()

    load_adapter_weights(vit, tensors)

    for key, tensor in tensors.items():
        torch.testing.assert_close(vit.get_parameter(key), tensor)


def test_merge_adapters_equals_the_inserted_adapters():
    vit = tiny_vit()
    fused = vit.blocks[0].attn.qkv.weight.detach().clone()
    insert_adapters(vit, rank=TINY_RANK, alpha=TINY_ALPHA)
    tensors = adapter_tensors()
    load_adapter_weights(vit, tensors)
    images = random_images()
    inserted = embed(vit, images)

    merge_adapters(vit)

    torch.testing.assert_close(embed(vit, images), inserted, rtol=1e-5, atol=1e-5)
    merged = vit.blocks[0].attn.qkv
    assert isinstance(merged, nn.Linear)
    # The closed form: the query and value rows move by alpha / rank times the low-rank product.
    block = {name: tensors[adapter_key(0, name)] for name in ADAPTER_NAMES}
    query = block["adapter_Q_downsample"] @ block["adapter_Q_upsample"]
    value = block["adapter_V_downsample"] @ block["adapter_V_upsample"]
    update = torch.cat([query.T, torch.zeros_like(query), value.T]) * (TINY_ALPHA / TINY_RANK)
    torch.testing.assert_close(merged.weight, fused + update)


def test_adapter_key_names_a_parameter_of_the_adapted_model():
    vit = tiny_vit()
    insert_adapters(vit, rank=TINY_RANK, alpha=TINY_ALPHA)

    keys = {adapter_key(index, name) for index in range(TINY_DEPTH) for name in ADAPTER_NAMES}

    assert keys == {name for name, _ in vit.named_parameters() if "adapter" in name}


# --- Heads and identity -----------------------------------------------------------------------


def test_build_heads_builds_bias_free_projections():
    tensors = head_tensors()

    heads = build_heads(
        tensors, embed_dim=TINY_DIM, cell_dim=TINY_CELL_DIM, scconcept_dim=SCCONCEPT_DIM
    )
    partial = build_heads(
        {"img_tok_proj.weight": tensors["img_tok_proj.weight"]},
        embed_dim=TINY_DIM,
        cell_dim=TINY_CELL_DIM,
    )

    for name in HEAD_NAMES:
        head = getattr(heads, name)
        assert head.bias is None
        torch.testing.assert_close(head.weight, tensors[f"{name}.weight"])
    assert partial.image_proj is None
    assert partial.st_node_proj is None


def test_bundle_identity_is_stable_for_the_same_weights():
    first = bundle_identity(save(adapter_tensors()), head_tensors())
    second = bundle_identity(save(adapter_tensors()), head_tensors())

    assert first == second
    assert first[0] == hashlib.sha256(save(adapter_tensors())).hexdigest()[:16]
    assert [len(identity) for identity in first] == [16, 16]


def test_bundle_identity_changes_with_other_weights():
    heads = head_tensors()
    encoder_id, cell_space_id = bundle_identity(save(adapter_tensors()), heads)
    moved = {**heads, "st_node_proj.weight": heads["st_node_proj.weight"] + 1.0}

    other_adapters = bundle_identity(save(adapter_tensors(seed=5)), heads)
    other_head = bundle_identity(save(adapter_tensors()), moved)

    assert other_adapters[0] != encoder_id
    assert other_adapters[1] != cell_space_id
    assert other_head[0] == encoder_id
    assert other_head[1] != cell_space_id
