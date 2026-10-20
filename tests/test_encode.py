"""Tests for encoding tiles and cells, and for preparing tiles for the encoder."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from alhena import (
    INPUT_SIZE,
    PATCH_SIZE,
    REFERENCE_MPP,
    AlhenaEncoder,
    AlhenaValueError,
    GeometryError,
    GeometryWarning,
    describe_geometry,
    membership_from_centroids,
    prepare_tiles,
)
from alhena.cells import POLYGONS_FRAME
from alhena.encoder import build_heads
from alhena.io import read_tiles
from support import (
    CELL_DIM,
    CELL_SPACE,
    EMBED_DIM,
    ENCODER_ID,
    N_TILES,
    NUCLEI_PER_TILE,
    NUCLEUS_SIDE,
    PixelBackbone,
    bundle_info,
    make_centroids,
    make_label_maps,
    make_polygons,
    make_tiles,
    tiny_encoder,
)

# ImageNet statistics, which the canvas is normalized with.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

N_CELLS = N_TILES * NUCLEI_PER_TILE


def normalized(tiles: np.ndarray) -> np.ndarray:
    """Return uint8 RGB tiles normalized by hand, channel first, as the encoder reads them."""
    values = (tiles / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return values.transpose(0, 3, 1, 2).astype(np.float32)


# --- Tiles ------------------------------------------------------------------------------------


def test_encode_tiles_returns_one_embedding_per_tile():
    result = tiny_encoder().encode_tiles(make_tiles(), mpp=REFERENCE_MPP)

    assert result.embeddings.shape == (N_TILES, EMBED_DIM)
    assert result.embeddings.dtype == np.float32
    assert result.tile_index.tolist() == list(range(N_TILES))
    assert result.tile_ids == tuple(str(index) for index in range(N_TILES))
    assert result.encoder_id == ENCODER_ID
    assert result.purpose == "tiles"
    assert result.bounds is None


def test_encode_tiles_returns_the_class_token():
    encoder = tiny_encoder()
    tiles = make_tiles()

    embeddings = encoder.encode_tiles(tiles, mpp=REFERENCE_MPP).embeddings

    with torch.no_grad():
        features = encoder.backbone.forward_features(torch.from_numpy(normalized(tiles)))
        expected = encoder.backbone.forward_head(features, pre_logits=True).numpy()
    np.testing.assert_allclose(embeddings, expected, rtol=1e-5, atol=1e-5)


def test_encode_tiles_records_the_bounds_of_a_source():
    origins = np.array([[0, 0], [INPUT_SIZE, 0], [0, INPUT_SIZE]])
    source = read_tiles(make_tiles(), mpp=REFERENCE_MPP, tile_ids=["a", "b", "c"], coords=origins)

    result = tiny_encoder().encode_tiles(source)

    np.testing.assert_array_equal(result.bounds, np.hstack([origins, origins + INPUT_SIZE]))
    # Pixel centers lie at whole coordinates, so a tile is centered on the mean of its pixels.
    np.testing.assert_array_equal(result.centers, origins + (INPUT_SIZE - 1) / 2)
    assert result.tile_ids == ("a", "b", "c")
    assert result.bounds_frame == source.coords_frame
    assert result.bounds_mpp == pytest.approx(REFERENCE_MPP)
    assert result.mpp == pytest.approx(REFERENCE_MPP)


def test_encode_tiles_is_independent_of_batch_size():
    encoder = tiny_encoder()
    tiles = make_tiles()
    whole = encoder.encode_tiles(tiles, mpp=REFERENCE_MPP, batch_size=N_TILES).embeddings

    for batch_size in (1, 2):
        result = encoder.encode_tiles(tiles, mpp=REFERENCE_MPP, batch_size=batch_size)
        np.testing.assert_allclose(result.embeddings, whole, rtol=1e-6, atol=1e-6)


def test_encode_tiles_is_deterministic():
    encoder = tiny_encoder()
    tiles = make_tiles()

    first = encoder.encode_tiles(tiles, mpp=REFERENCE_MPP).embeddings
    second = encoder.encode_tiles(tiles, mpp=REFERENCE_MPP).embeddings

    np.testing.assert_array_equal(first, second)


def test_encode_tiles_warns_when_tiles_are_cropped():
    side = 300
    with pytest.warns(GeometryWarning, match="cropped away"):
        result = tiny_encoder().encode_tiles(make_tiles(size=side), mpp=REFERENCE_MPP)

    assert result.crop_fraction == pytest.approx(1 - INPUT_SIZE**2 / side**2)
    assert result.pad_fraction == 0.0


def test_encode_tiles_warns_when_the_canvas_is_padded():
    side = INPUT_SIZE // 2
    with pytest.warns(GeometryWarning, match="white pad"):
        result = tiny_encoder().encode_tiles(make_tiles(size=side), mpp=REFERENCE_MPP)

    assert result.pad_fraction == pytest.approx(1 - side**2 / INPUT_SIZE**2)
    assert result.crop_fraction == 0.0


@pytest.mark.parametrize(
    ("tiles", "match"),
    [
        pytest.param(np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), np.float32), "uint8", id="float"),
        pytest.param(np.zeros((1, INPUT_SIZE, INPUT_SIZE, 4), np.uint8), "RGB", id="rgba"),
    ],
)
def test_encode_tiles_refuses_tiles_that_are_not_uint8_rgb(tiles, match):
    with pytest.raises(GeometryError, match=match):
        tiny_encoder().encode_tiles(tiles, mpp=REFERENCE_MPP)


@pytest.mark.parametrize("mpp", [0.0, 425.0])
def test_encode_tiles_refuses_an_implausible_mpp(mpp):
    with pytest.raises(AlhenaValueError, match="mpp"):
        tiny_encoder().encode_tiles(make_tiles(n_tiles=1), mpp=mpp)


def test_encode_returns_a_bare_array_from_raw_or_prepared_tiles():
    encoder = tiny_encoder()
    tiles = make_tiles()
    batch, _ = prepare_tiles(tiles, mpp=REFERENCE_MPP)

    embeddings = encoder.encode(tiles, mpp=REFERENCE_MPP)

    assert isinstance(embeddings, np.ndarray)
    expected = encoder.encode_tiles(tiles, mpp=REFERENCE_MPP).embeddings
    np.testing.assert_array_equal(embeddings, expected)
    np.testing.assert_array_equal(encoder.encode(batch), embeddings)


def test_encode_with_tokens_returns_the_token_grid():
    encoder = tiny_encoder()
    tiles = make_tiles()
    n_tokens = (INPUT_SIZE // PATCH_SIZE) ** 2

    pooled, tokens = encoder.encode_with_tokens(tiles, mpp=REFERENCE_MPP)

    assert tokens.shape == (N_TILES, n_tokens, EMBED_DIM)
    np.testing.assert_array_equal(pooled, encoder.encode(tiles, mpp=REFERENCE_MPP))
    with torch.no_grad():
        features = encoder.backbone.forward_features(torch.from_numpy(normalized(tiles)))
    np.testing.assert_allclose(tokens, features[:, -n_tokens:].numpy(), rtol=1e-5, atol=1e-5)


# --- Cells ------------------------------------------------------------------------------------


def test_encode_cells_from_label_maps():
    cells = tiny_encoder().encode_cells(make_tiles(), make_label_maps(), mpp=REFERENCE_MPP)

    assert cells.cell_id.tolist() == list(range(1, N_CELLS + 1))
    assert cells.backbone.shape == (N_CELLS, EMBED_DIM)
    assert cells.cell_space.shape == (N_CELLS, CELL_DIM)
    assert cells.valid.all()
    assert cells.tile_index.tolist() == np.repeat(np.arange(N_TILES), NUCLEI_PER_TILE).tolist()
    assert cells.alpha.tolist() == [NUCLEUS_SIDE**2] * N_CELLS
    assert cells.membership_source == "labels"
    assert cells.membership_approximate is False
    assert cells.n_cells_without_tokens == 0
    assert cells.cell_space_id == CELL_SPACE


def test_encode_cells_pools_the_tokens_under_each_nucleus():
    encoder = tiny_encoder()
    tiles, maps = make_tiles(), make_label_maps()
    grid = INPUT_SIZE // PATCH_SIZE

    cells = encoder.encode_cells(tiles, maps, mpp=REFERENCE_MPP)

    _, tokens = encoder.encode_with_tokens(tiles, mpp=REFERENCE_MPP)
    expected = []
    for cell in cells.cell_id:
        tile, rows, columns = np.nonzero(maps == cell)
        under = tokens[tile, (rows // PATCH_SIZE) * grid + columns // PATCH_SIZE]
        expected.append(under.astype(np.float64).mean(axis=0))
    np.testing.assert_allclose(cells.backbone, expected, rtol=1e-5, atol=1e-5)
    projected = encoder.project_cell(cells.backbone)
    np.testing.assert_allclose(cells.cell_space, projected, rtol=1e-5, atol=1e-5)


def test_encode_cells_from_centroids_through_a_membership():
    encoder = tiny_encoder()
    points, tile_indices = make_centroids()
    nuclei = {"centroids": points, "tile_indices": tile_indices, "radius_px": 8.0}

    with pytest.warns(GeometryWarning, match="approximated as circles"):
        cells = encoder.encode_cells(make_tiles(), nuclei, mpp=REFERENCE_MPP)
    with pytest.warns(GeometryWarning, match="approximated as circles"):
        membership = membership_from_centroids(
            points, tile_indices, radius_px=8.0, mpp=REFERENCE_MPP
        )
    from_table = encoder.encode_cells(make_tiles(), membership, mpp=REFERENCE_MPP)

    assert cells.cell_id.tolist() == list(range(N_CELLS))
    assert cells.membership_source == "centroids"
    assert cells.membership_approximate is True
    assert cells.valid.all()
    np.testing.assert_array_equal(from_table.backbone, cells.backbone)


def test_encode_cells_from_polygons():
    polygons, bounds = make_polygons()
    names = np.arange(100, 100 + N_CELLS)
    nuclei = {"polygons": polygons, "tile_bounds": bounds}

    cells = tiny_encoder().encode_cells(make_tiles(), nuclei, mpp=REFERENCE_MPP, cell_id=names)

    assert cells.cell_id.tolist() == names.tolist()
    assert cells.cell_space.shape == (N_CELLS, CELL_DIM)
    assert cells.membership_source == "polygons"
    assert cells.membership_approximate is False
    assert cells.n_cells_without_tokens == 0
    assert cells.tile_index.tolist() == np.repeat(np.arange(N_TILES), NUCLEI_PER_TILE).tolist()
    np.testing.assert_allclose(cells.centroids, [polygon.mean(axis=0) for polygon in polygons])
    assert cells.centroids_frame == POLYGONS_FRAME


def test_encode_cells_counts_nuclei_that_reach_no_token():
    side = 300
    maps = np.zeros((1, side, side), dtype=np.int32)
    maps[0, 140:160, 140:160] = 1
    # In the margin the center crop discards.
    maps[0, :NUCLEUS_SIDE, :NUCLEUS_SIDE] = 2

    with pytest.warns(GeometryWarning, match="center crop"):
        cells = tiny_encoder().encode_cells(
            make_tiles(n_tiles=1, size=side), maps, mpp=REFERENCE_MPP
        )

    assert cells.cell_id.tolist() == [1]
    assert cells.n_cells_without_tokens == 1
    assert cells.geometry.crop_fraction == pytest.approx(1 - INPUT_SIZE**2 / side**2)


def test_project_cell_applies_the_cell_head():
    encoder = tiny_encoder()
    pooled = np.random.default_rng(0).normal(size=(5, EMBED_DIM)).astype(np.float32)
    weight = encoder.heads.img_tok_proj.weight.detach().numpy()

    projected = encoder.project_cell(pooled)

    assert projected.shape == (5, CELL_DIM)
    np.testing.assert_allclose(projected, pooled @ weight.T, rtol=1e-5, atol=1e-5)


def test_project_tile_applies_the_tile_head():
    weight = torch.randn((CELL_DIM, EMBED_DIM), generator=torch.Generator().manual_seed(1))
    heads = build_heads({"image_proj.weight": weight}, embed_dim=EMBED_DIM, cell_dim=CELL_DIM)
    info = replace(bundle_info(), has_image_proj=True, has_cell_proj=False)
    encoder = AlhenaEncoder(PixelBackbone(), heads, info)
    embeddings = encoder.encode(make_tiles(), mpp=REFERENCE_MPP)

    projected = encoder.project_tile(embeddings)

    assert projected.shape == (N_TILES, CELL_DIM)
    np.testing.assert_allclose(projected, embeddings @ weight.numpy().T, rtol=1e-5, atol=1e-5)


# --- Preparing tiles --------------------------------------------------------------------------


def test_prepare_tiles_normalizes_tiles_onto_the_canvas():
    tiles = make_tiles()

    batch, info = prepare_tiles(tiles, mpp=REFERENCE_MPP)

    assert batch.shape == (N_TILES, 3, INPUT_SIZE, INPUT_SIZE)
    assert batch.dtype == torch.float32
    assert info.scale == 1.0
    assert not info.padded
    assert not info.cropped
    np.testing.assert_allclose(batch.numpy(), normalized(tiles), atol=1e-5)


def test_prepare_tiles_resamples_to_the_reference_mpp():
    tile = make_tiles(n_tiles=1)
    # Every pixel becomes a 2 by 2 block: the same field at half the microns per pixel.
    doubled = tile.repeat(2, axis=1).repeat(2, axis=2)

    batch, info = prepare_tiles(doubled, mpp=REFERENCE_MPP / 2)

    assert info.scale == pytest.approx(0.5)
    assert info.resampled_size == (INPUT_SIZE, INPUT_SIZE)
    np.testing.assert_allclose(batch.numpy(), normalized(tile), atol=1e-5)


def test_prepare_tiles_pads_small_fields_with_white():
    side = INPUT_SIZE // 2
    tiles = make_tiles(n_tiles=1, size=side)
    start = (INPUT_SIZE - side) // 2
    canvas = np.full((1, INPUT_SIZE, INPUT_SIZE, 3), 255, dtype=np.uint8)
    canvas[:, start : start + side, start : start + side] = tiles

    batch, info = prepare_tiles(tiles, mpp=REFERENCE_MPP)

    assert info.padded
    assert info.pad_fraction == pytest.approx(1 - side**2 / INPUT_SIZE**2)
    np.testing.assert_allclose(batch.numpy(), normalized(canvas), atol=1e-5)


def test_prepare_tiles_crops_large_fields_at_the_center():
    side = 300
    tiles = make_tiles(n_tiles=1, size=side)
    start = (side - INPUT_SIZE) // 2
    center = tiles[:, start : start + INPUT_SIZE, start : start + INPUT_SIZE]

    batch, info = prepare_tiles(tiles, mpp=REFERENCE_MPP)

    assert info.cropped
    assert info.crop_fraction == pytest.approx(1 - INPUT_SIZE**2 / side**2)
    np.testing.assert_allclose(batch.numpy(), normalized(center), atol=1e-5)


def test_prepare_tiles_warns_when_mpp_is_not_declared():
    with pytest.warns(GeometryWarning, match="not declared"):
        batch, info = prepare_tiles(make_tiles(n_tiles=1, size=300))

    assert batch.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert info.mpp_declared is False
    assert info.crop_fraction == 0.0


def test_describe_geometry_reports_the_physical_field():
    mpp, side = 0.25, 381
    token_um = PATCH_SIZE * REFERENCE_MPP

    report = describe_geometry(mpp, tile_size_px=side, median_nucleus_area_px=NUCLEUS_SIDE**2)

    assert report.field_of_view_um == pytest.approx(side * mpp)
    assert report.token_footprint_um == pytest.approx(token_um)
    assert report.mpp_ratio == pytest.approx(mpp / REFERENCE_MPP)
    assert report.median_nucleus_tokens == pytest.approx(NUCLEUS_SIDE * mpp / token_um)
    assert report.pad_fraction == 0.0
    assert report.crop_fraction == 0.0
    assert report.warnings == ()


@pytest.mark.parametrize(("side", "mpp"), [(112, REFERENCE_MPP), (300, REFERENCE_MPP), (448, 0.25)])
def test_describe_geometry_agrees_with_prepare_tiles(side, mpp):
    _, info = prepare_tiles(make_tiles(n_tiles=1, size=side), mpp=mpp)

    report = describe_geometry(mpp, tile_size_px=side)

    assert report.pad_fraction == pytest.approx(info.pad_fraction)
    assert report.crop_fraction == pytest.approx(info.crop_fraction)
