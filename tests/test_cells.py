"""Tests for membership tables built from nuclei, and for pooling tokens into cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alhena import (
    INPUT_SIZE,
    PATCH_SIZE,
    REFERENCE_MPP,
    GeometryError,
    GeometryWarning,
    MembershipError,
    prepare_tiles,
)
from alhena.cells import (
    GROUP_COLUMNS,
    ORIGINS_FRAME,
    POLYGONS_FRAME,
    fit_mask,
    membership_from_centroids,
    membership_from_labels,
    membership_from_polygons,
    membership_is_approximate,
    membership_source,
    missing_cells,
    pool_cells,
    pool_tokens,
    resolve_main_tile,
    token_index_map,
    validate_membership,
)
from alhena.io import read_label_maps, read_tiles
from alhena.types import MEMBERSHIP_COLUMNS
from support import (
    N_TILES,
    NUCLEI_PER_TILE,
    NUCLEUS_SIDE,
    make_centroids,
    make_label_maps,
    make_polygons,
    make_tiles,
    nucleus_corners,
    tiny_encoder,
)

GRID = INPUT_SIZE // PATCH_SIZE
N_TOKENS = GRID * GRID
TILE_INDICES = list(range(N_TILES))


def make_tokens(n_tiles: int, dim: int = 3, seed: int = 0) -> np.ndarray:
    """Return random patch tokens, one grid of them per tile."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_tiles, N_TOKENS, dim), dtype=np.float32)


def hand_table(*rows: tuple[int, object, int, int]) -> pd.DataFrame:
    """Return a membership table assembled by hand, one (tile, cell, token, pixels) per row."""
    return pd.DataFrame(list(rows), columns=list(MEMBERSHIP_COLUMNS))


def outline(x0: float, y0: float, side: int, along_edges: bool = False) -> np.ndarray:
    """Return a square around a nucleus `side` wide.

    It runs through the centers of the boundary pixels, or along their outer edges as a contour
    tracer draws it.
    """
    if along_edges:
        x0, y0, x1, y1 = x0 - 0.5, y0 - 0.5, x0 + side - 0.5, y0 + side - 0.5
    else:
        x1, y1 = x0 + side - 1, y0 + side - 1
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def rows_of(frame: pd.DataFrame) -> list[tuple]:
    """Return the rows of a table of integers, sorted, whatever their order in it."""
    return sorted(map(tuple, frame.to_numpy().tolist()))


# --- The canvas ---------------------------------------------------------------------------------


def test_token_index_map_numbers_tokens_row_major():
    grid = token_index_map()
    rows, columns = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE]

    assert grid.shape == (INPUT_SIZE, INPUT_SIZE)
    assert grid.dtype == np.int32
    assert np.array_equal(grid, (rows // PATCH_SIZE) * GRID + columns // PATCH_SIZE)
    assert grid[-1, -1] == N_TOKENS - 1


def test_fit_mask_keeps_the_center_crop_of_a_larger_mask():
    mask = np.arange(300 * 300, dtype=np.int64).reshape(300, 300) + 1
    top = (300 - INPUT_SIZE) // 2

    fitted = fit_mask(mask, mpp=REFERENCE_MPP)

    assert np.array_equal(fitted, mask[top : top + INPUT_SIZE, top : top + INPUT_SIZE])


@pytest.mark.parametrize(("side", "ratio"), [(201, 1.0), (200, 0.8), (150, 0.5)])
def test_fit_mask_covers_the_pixels_its_image_covers(side, ratio):
    mpp = REFERENCE_MPP * ratio
    batch, _ = prepare_tiles(np.zeros((1, side, side, 3), dtype=np.uint8), mpp=mpp)
    # Black tile pixels normalize below zero, the white pad around them above it.
    image = batch[0, 0].numpy() < 0.0

    mask = fit_mask(np.ones((side, side), dtype=np.int32), mpp=mpp)

    assert np.array_equal(mask > 0, image)
    assert mask[0, 0] == 0


# --- Membership tables --------------------------------------------------------------------------


def test_membership_from_labels_counts_pixels_per_token():
    maps = make_label_maps()
    frame = membership_from_labels(maps, TILE_INDICES, mpp=REFERENCE_MPP)

    tile, row, column = np.nonzero(maps)
    pixels = pd.DataFrame(
        {
            "tile_index": tile,
            "cell_id": maps[tile, row, column],
            "token_index": token_index_map()[row, column],
        }
    )
    counted = pixels.groupby(list(MEMBERSHIP_COLUMNS[:3])).size().reset_index()

    assert tuple(frame.columns) == MEMBERSHIP_COLUMNS
    assert rows_of(frame) == rows_of(counted)
    assert missing_cells(frame).size == 0


def test_membership_from_labels_reports_nuclei_lost_to_the_crop():
    label_map = np.zeros((300, 300), dtype=np.int64)
    label_map[0:5, 0:5] = 11
    label_map[150:160, 150:160] = 12
    label_map[295:300, 295:300] = 13

    frame = membership_from_labels([label_map], [0], mpp=REFERENCE_MPP)

    assert frame["cell_id"].unique().tolist() == [12]
    assert missing_cells(frame).tolist() == [11, 13]


def test_membership_from_centroids_paints_a_disk_of_the_radius():
    x, y, radius = 100.0, 120.0, 9.0
    with pytest.warns(GeometryWarning):
        frame = membership_from_centroids(
            np.array([[x, y]]), [0], radius_px=radius, mpp=REFERENCE_MPP
        )

    rows, columns = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE]
    disk = (columns - x) ** 2 + (rows - y) ** 2 <= radius**2
    counts = np.bincount(token_index_map()[disk], minlength=N_TOKENS)
    painted = dict(zip(frame["token_index"].tolist(), frame["nucleus_px"].tolist(), strict=True))

    assert painted == {token: int(count) for token, count in enumerate(counts) if count}


@pytest.mark.parametrize("along_edges", [False, True], ids=["through_centers", "along_edges"])
def test_membership_from_polygons_matches_the_label_maps(along_edges: bool):
    _, windows = make_polygons()
    polygons = [
        outline(tile * INPUT_SIZE + column, row, NUCLEUS_SIDE, along_edges)
        for tile in TILE_INDICES
        for row, column in nucleus_corners()
    ]
    cell_ids = np.arange(1, len(polygons) + 1)

    from_polygons = membership_from_polygons(
        polygons, windows, cell_ids=cell_ids, mpp=REFERENCE_MPP
    )
    from_labels = membership_from_labels(make_label_maps(), TILE_INDICES, mpp=REFERENCE_MPP)

    assert rows_of(from_polygons) == rows_of(from_labels)


def test_membership_source_and_approximation_by_builder():
    window = np.array([[0.0, 0.0, INPUT_SIZE, INPUT_SIZE]])
    labels = membership_from_labels(make_label_maps(n_tiles=1), [0])
    polygons = membership_from_polygons([outline(20.0, 20.0, 20)], window)
    with pytest.warns(GeometryWarning, match="approximated as circles"):
        centroids = membership_from_centroids(np.array([[30.0, 30.0]]), [0], radius_px=10.0)

    tables = (labels, centroids, polygons)
    assert [membership_source(table) for table in tables] == ["labels", "centroids", "polygons"]
    assert [membership_is_approximate(table) for table in tables] == [False, True, False]


def test_hand_built_table_records_no_provenance():
    table = hand_table((0, 1, 5, 20))

    assert membership_source(table) == "unknown"
    with pytest.raises(MembershipError, match="records no provenance"):
        membership_is_approximate(table)
    with pytest.raises(MembershipError, match="records no set of input cells"):
        missing_cells(table)
    cells = pool_cells(make_tokens(1), table)
    assert cells.membership_source == "unknown"
    assert cells.membership_approximate is None
    assert cells.n_cells_without_tokens is None


def test_validate_membership_checks_the_tile_count():
    table = membership_from_labels(make_label_maps(), TILE_INDICES, mpp=REFERENCE_MPP)

    assert validate_membership(table, n_tiles=3) is None
    with pytest.raises(MembershipError, match="outside the 2 tiles given"):
        validate_membership(table, n_tiles=2)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(0, 1, N_TOKENS, 5)], "outside the 256 tokens"),
        ([(0, 1, 4, 5), (0, 1, 4, 6)], "rows repeat a"),
    ],
    ids=["token off the grid", "repeated key"],
)
def test_validate_membership_refuses_a_malformed_table(rows, message):
    with pytest.raises(MembershipError, match=message):
        validate_membership(hand_table(*rows), n_tiles=1)


# --- Pooling ------------------------------------------------------------------------------------


def test_pool_tokens_matches_a_weighted_sum_by_hand():
    tokens = make_tokens(1, seed=1)

    pooled, groups = pool_tokens(tokens, hand_table((0, 1, 5, 30), (0, 1, 6, 10), (0, 2, 9, 40)))

    # A token weighs its pixels over alpha + 1e-6, alpha being the pixels of the cell in the tile.
    expected = np.stack(
        [
            (30 * tokens[0, 5] + 10 * tokens[0, 6]) / (40 + 1e-6),
            40 * tokens[0, 9] / (40 + 1e-6),
        ]
    )
    assert tuple(groups.columns) == GROUP_COLUMNS
    assert groups["cell_id"].tolist() == [1, 2]
    assert groups["alpha"].tolist() == [40.0, 40.0]
    assert pooled.dtype == np.float32
    np.testing.assert_allclose(pooled, expected, rtol=1e-6, atol=1e-7)


def test_pool_tokens_refuses_tokens_with_prefix_tokens():
    # The class token and the register tokens precede the patch tokens.
    tokens = np.zeros((1, 9 + N_TOKENS, 3), dtype=np.float32)
    with pytest.raises(GeometryError, match="265 positions per tile"):
        pool_tokens(tokens, hand_table((0, 1, 5, 20)))


def test_pool_cells_matches_a_pixel_weighted_mean():
    maps = make_label_maps()
    tokens = make_tokens(3, seed=2)

    cells = pool_cells(tokens, membership_from_labels(maps, TILE_INDICES, mpp=REFERENCE_MPP))

    grid = token_index_map()
    expected = []
    for tile, label_map in enumerate(maps):
        for label in np.unique(label_map[label_map > 0]).tolist():
            under = grid[label_map == label]
            expected.append(tokens[tile, under].sum(axis=0) / (under.size + 1e-6))
    assert cells.cell_id.tolist() == list(range(1, 13))
    np.testing.assert_allclose(cells.backbone, np.stack(expected), rtol=1e-5, atol=1e-6)
    assert cells.alpha.tolist() == [NUCLEUS_SIDE**2] * 12
    assert cells.tile_index.tolist() == np.repeat(TILE_INDICES, NUCLEI_PER_TILE).tolist()
    assert cells.n_candidate_tiles.tolist() == [1] * 12
    assert cells.valid.all()
    assert cells.cell_space is None


def test_pool_cells_projects_with_the_cell_head():
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)

    cells = pool_cells(
        make_tokens(1), hand_table((0, 1, 5, 20), (0, 2, 9, 10)), project=lambda x: x @ weight
    )

    assert cells.cell_space.shape == (2, 4)
    np.testing.assert_allclose(cells.cell_space, cells.backbone @ weight, rtol=1e-6)


def test_pool_cells_places_nuclei_at_their_tile_origins():
    origins = np.array([[1000, 2000], [1224, 2000], [1448, 2000]])
    frame = membership_from_labels(
        make_label_maps(), TILE_INDICES, mpp=REFERENCE_MPP, tile_origins=origins
    )

    cells = pool_cells(make_tokens(3), frame)

    points, tiles = make_centroids()
    np.testing.assert_allclose(cells.centroids, points + origins[tiles])
    assert cells.centroids_frame == ORIGINS_FRAME
    assert cells.centroids_mpp == pytest.approx(REFERENCE_MPP)


def test_pool_cells_reads_a_shared_nucleus_from_the_nearer_tile():
    windows = np.array(
        [[0, 0, INPUT_SIZE, INPUT_SIZE], [112, 0, 112 + INPUT_SIZE, INPUT_SIZE]], dtype=np.float64
    )
    frame = membership_from_polygons(
        [outline(200.0, 40.0, 20)], windows, cell_ids=["a"], mpp=REFERENCE_MPP
    )

    cells = pool_cells(make_tokens(2), frame)

    assert sorted(frame["tile_index"].unique().tolist()) == [0, 1]
    # Whole in both tiles, and nearer the center of the second.
    assert cells.tile_index.tolist() == [1]
    assert cells.n_candidate_tiles.tolist() == [2]
    assert cells.centroids.tolist() == [[209.5, 49.5]]
    assert cells.centroids_frame == POLYGONS_FRAME


def test_resolve_main_tile_prefers_pixels_then_the_nearer_center():
    groups = pd.DataFrame(
        {
            "tile_index": np.array([0, 3, 1, 0, 1], dtype=np.int32),
            "cell_id": [1, 1, 1, 2, 2],
            "alpha": np.array([10.0, 40.0, 25.0, 30.0, 30.0], dtype=np.float32),
        }
    )
    centers = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 0.0], [900.0, 0.0]])

    selected, candidates = resolve_main_tile(groups, centers, {1: (0.0, 0.0), 2: (90.0, 0.0)})

    assert groups["tile_index"].to_numpy()[selected].tolist() == [3, 1]
    assert candidates.tolist() == [3, 2]


# --- The cell path ------------------------------------------------------------------------------


def test_encode_cells_needs_a_shared_frame_for_shared_nuclei():
    region = np.zeros((INPUT_SIZE, INPUT_SIZE + 112), dtype=np.int64)
    region[40:60, 100:120] = 5
    region[80:100, 180:200] = 6
    maps = np.stack([region[:, x : x + INPUT_SIZE] for x in (0, 112)])
    tiles = make_tiles(n_tiles=2)
    encoder = tiny_encoder()

    with pytest.raises(MembershipError, match="shared coordinate frame"):
        encoder.encode_cells(tiles, maps, mpp=REFERENCE_MPP)
    source = read_tiles(tiles, mpp=REFERENCE_MPP, coords=np.array([[0, 0], [112, 0]]))
    cells = encoder.encode_cells(source, maps)

    assert cells.cell_id.tolist() == [5, 6]
    assert cells.n_candidate_tiles.tolist() == [2, 2]
    # Only the first tile holds all of cell 5; both hold cell 6, nearer the second's center.
    assert cells.tile_index.tolist() == [0, 1]


def test_encode_cells_warns_of_an_identifier_naming_nuclei_apart():
    # Two tiles numbering their nuclei from one, as a segmenter run tile by tile numbers them.
    maps = np.concatenate([make_label_maps(n_tiles=1)] * 2)
    origins = np.array([[0, 0], [400, 0]])
    source = read_tiles(make_tiles(n_tiles=2), mpp=REFERENCE_MPP, coords=origins)
    encoder = tiny_encoder()

    with pytest.warns(GeometryWarning, match="read_label_maps renumbers"):
        merged = encoder.encode_cells(source, maps)
    renumbered = encoder.encode_cells(source, read_label_maps(maps, tiles=source))

    assert len(merged) == NUCLEI_PER_TILE
    assert len(renumbered) == 2 * NUCLEI_PER_TILE


def test_encode_cells_refuses_label_maps_of_another_size():
    maps = make_label_maps(size=256)
    with pytest.raises(GeometryError, match="label map 256 by 256"):
        tiny_encoder().encode_cells(make_tiles(), maps, mpp=REFERENCE_MPP)
