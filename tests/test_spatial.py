"""Tests for reading inputs from SpatialData objects, writing results back, and the sopa hook."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from alhena import (
    INPUT_SIZE,
    REFERENCE_MPP,
    AlhenaValueError,
    AnnotationResult,
    CellEmbeddings,
    CellTyper,
    MembershipError,
    MissingExtraError,
    TileEmbeddings,
    load_reference,
    membership_from_polygons,
)
from alhena.io import (
    CELL_SPACE_KEY,
    CONFIDENCE_COLUMN,
    LABEL_COLUMN,
    SPATIAL_KEY,
    TILE_INDEX_COLUMN,
    read_annotations,
    read_tile_embeddings,
    read_tiles,
    tile_bounds,
)
from alhena.spatial import (
    CELL_TABLE_KEY,
    DEFAULT_COORDINATE_SYSTEM,
    INSTANCE_KEY,
    PATCHES_KEY,
    REGION_KEY,
    TILE_EMBEDDINGS_KEY,
    TILE_SHAPES_KEY,
    TILE_TABLE_KEY,
    Alhena,
    cells_from_shapes,
    cells_to_spatialdata,
    patch_bounds,
    tile_embeddings,
    tile_source,
    tiles_to_spatialdata,
)
from support import EMBED_DIM, make_polygons, make_shard, make_tiles, tiny_encoder

# The shared tiles side by side as one image, and the square outlines of their nuclei.
IMAGE = np.concatenate(list(make_tiles()), axis=1)
POLYGONS, BOUNDS = make_polygons()
CELL_IDS = np.arange(1, len(POLYGONS) + 1)
CENTERS = np.asarray([outline.mean(axis=0) for outline in POLYGONS])


def box_outline(window: np.ndarray) -> np.ndarray:
    """Return the corners of an x0, y0, x1, y1 window."""
    x0, y0, x1, y1 = window
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def scaled(factor: float) -> dict[str, Any]:
    """Return a placement in the global coordinate system by a scale of `factor`."""
    transformations = pytest.importorskip("spatialdata.transformations")
    return {"global": transformations.Scale([factor, factor], axes=("x", "y"))}


def image_element(pixels: np.ndarray, scale_factors: list[int] | None = None) -> Any:
    """Return an image element of `(height, width, 3)` pixels, multiscale with `scale_factors`."""
    models = pytest.importorskip("spatialdata.models")
    channels_first = np.moveaxis(pixels, -1, 0)
    return models.Image2DModel.parse(
        channels_first, dims=("c", "y", "x"), scale_factors=scale_factors
    )


def shapes_element(outlines: list[np.ndarray], index: Any = None, placed: Any = None) -> Any:
    """Return a shapes element holding one polygon per outline."""
    geopandas = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    models = pytest.importorskip("spatialdata.models")
    polygons = [shapely.Polygon(outline) for outline in outlines]
    frame = geopandas.GeoDataFrame(geometry=polygons, index=index)
    return models.ShapesModel.parse(frame, transformations=placed)


def slide(scale_factors: list[int] | None = None) -> Any:
    """Return an object holding the image and the outlines of its nuclei, indexed by cell."""
    spatialdata = pytest.importorskip("spatialdata")
    return spatialdata.SpatialData(
        images={"he": image_element(IMAGE, scale_factors)},
        shapes={"nuclei": shapes_element(POLYGONS, pd.Index(CELL_IDS))},
    )


@pytest.fixture
def sdata() -> Any:
    """Return the object of `slide`, its image a single array."""
    return slide()


@pytest.fixture
def cells(sdata: Any) -> CellEmbeddings:
    """Return the cells of the nuclei of the object, encoded from its image."""
    source = tile_source(sdata, tile_px=INPUT_SIZE, mpp=REFERENCE_MPP)
    membership = cells_from_shapes(
        sdata, "nuclei", patches=tile_bounds(source), image_key="he", mpp=REFERENCE_MPP
    )
    return tiny_encoder().encode_cells(source, membership, mpp=REFERENCE_MPP)


@pytest.fixture
def result(cells: CellEmbeddings, tmp_path: Path) -> AnnotationResult:
    """Return the annotations of the cells, against a reference in their cell space."""
    reference = load_reference(path=make_shard(tmp_path))
    return CellTyper(tiny_encoder(), reference).annotate(cells)


def encode_image_tiles(sdata: Any) -> TileEmbeddings:
    """Return the embeddings of the tiles of the image of an object."""
    return tiny_encoder().encode_tiles(tile_source(sdata, tile_px=INPUT_SIZE, mpp=REFERENCE_MPP))


def encode_array_tiles() -> TileEmbeddings:
    """Return the embeddings of the shared tiles, read from an array with their origins."""
    origins = BOUNDS[:, :2].astype(np.int64)
    return tiny_encoder().encode_tiles(read_tiles(make_tiles(), mpp=REFERENCE_MPP, coords=origins))


# --- Reading tiles and nuclei -----------------------------------------------------------------


@pytest.mark.parametrize("scale_factors", [None, [2]], ids=["array", "multiscale"])
def test_tile_source_reads_the_tiles_of_an_image_element(scale_factors: list[int] | None):
    source = tile_source(slide(scale_factors), tile_px=INPUT_SIZE, mpp=REFERENCE_MPP)
    tiles = make_tiles()
    assert len(source) == len(tiles)
    for index, tile in enumerate(tiles):
        np.testing.assert_array_equal(source[index], tile)
    np.testing.assert_array_equal(tile_bounds(source), BOUNDS)


def test_tile_source_cuts_the_windows_of_a_shapes_element(sdata: Any):
    windows = np.array([[100, 0, 324, 224], [300, 0, 524, 224]], dtype=np.float64)
    sdata.shapes["windows"] = shapes_element([box_outline(window) for window in windows])
    source = tile_source(sdata, patches="windows", mpp=REFERENCE_MPP)
    for index, (x0, y0, x1, y1) in enumerate(windows.astype(int)):
        np.testing.assert_array_equal(source[index], IMAGE[y0:y1, x0:x1])


def test_patch_bounds_reads_windows_in_the_pixels_of_the_image(sdata: Any):
    # Windows stored at half the pixel scale, which a scale of two places on the image.
    halved = [box_outline(window / 2.0) for window in BOUNDS]
    sdata.shapes["coarse"] = shapes_element(halved, placed=scaled(2.0))
    np.testing.assert_allclose(patch_bounds(sdata, patches="coarse", image_key="he"), BOUNDS)
    np.testing.assert_allclose(patch_bounds(sdata, patches="coarse"), BOUNDS / 2.0)


def test_cells_from_shapes_reads_the_outlines_of_the_element(sdata: Any):
    membership = cells_from_shapes(sdata, "nuclei", patches=BOUNDS, mpp=REFERENCE_MPP)
    expected = membership_from_polygons(POLYGONS, BOUNDS, cell_ids=CELL_IDS, mpp=REFERENCE_MPP)
    pd.testing.assert_frame_equal(membership, expected)


def test_cells_from_shapes_reads_microns_in_image_pixels(sdata: Any):
    # Nuclei stored in microns at half a micron per pixel, placed on the pixels by a scale of two.
    microns = [outline * 0.5 for outline in POLYGONS]
    sdata.shapes["microns"] = shapes_element(microns, pd.Index(CELL_IDS), placed=scaled(2.0))
    with pytest.raises(MembershipError, match="pass image_key"):
        cells_from_shapes(sdata, "microns", patches=BOUNDS, mpp=REFERENCE_MPP)

    membership = cells_from_shapes(
        sdata, "microns", patches=BOUNDS, image_key="he", mpp=REFERENCE_MPP
    )

    expected = membership_from_polygons(POLYGONS, BOUNDS, cell_ids=CELL_IDS, mpp=REFERENCE_MPP)
    pd.testing.assert_frame_equal(membership, expected)


# --- Writing cells and annotations ------------------------------------------------------------


def test_cells_to_spatialdata_writes_the_table_of_cells(sdata: Any, cells: CellEmbeddings):
    models = pytest.importorskip("spatialdata.models")

    table = cells_to_spatialdata(sdata, cells, "nuclei")

    assert sdata.tables[CELL_TABLE_KEY] is table
    assert models.get_table_keys(table) == ("nuclei", REGION_KEY, INSTANCE_KEY)
    assert table.obs[REGION_KEY].tolist() == ["nuclei"] * len(CELL_IDS)
    np.testing.assert_array_equal(table.obs[INSTANCE_KEY], CELL_IDS)
    np.testing.assert_allclose(table.obsm[SPATIAL_KEY], CENTERS)
    np.testing.assert_array_equal(table.obsm[CELL_SPACE_KEY], cells.cell_space)


def test_cells_to_spatialdata_adds_annotations_to_the_table(
    sdata: Any, cells: CellEmbeddings, result: AnnotationResult
):
    table = cells_to_spatialdata(sdata, cells, "nuclei")

    assert cells_to_spatialdata(sdata, result, "nuclei") is table
    assert list(sdata.tables) == [CELL_TABLE_KEY]
    assert table.obs[f"{LABEL_COLUMN}_lineage_v1"].tolist() == result.labels.tolist()
    np.testing.assert_allclose(table.obs[f"{CONFIDENCE_COLUMN}_lineage_v1"], result.confidence)


def test_cells_to_spatialdata_refuses_cells_of_another_element(sdata: Any, cells: CellEmbeddings):
    sdata.shapes["other"] = shapes_element(POLYGONS, pd.Index(CELL_IDS + 100))
    with pytest.raises(AlhenaValueError, match="12 of 12 cell identifiers name no instance"):
        cells_to_spatialdata(sdata, cells, "other")


# --- Writing tile embeddings ------------------------------------------------------------------


def test_tiles_to_spatialdata_writes_rectangles_and_a_table(sdata: Any):
    transformations = pytest.importorskip("spatialdata.transformations")
    models = pytest.importorskip("spatialdata.models")
    tiles = encode_image_tiles(sdata)
    placement = transformations.Scale([2.0, 2.0], axes=("x", "y"))
    transformations.set_transformation(sdata.images["he"], placement, to_coordinate_system="slide")

    assert tiles_to_spatialdata(sdata, tiles) is sdata

    rectangles = sdata.shapes[TILE_SHAPES_KEY]
    np.testing.assert_array_equal(rectangles.bounds, BOUNDS)
    placed = transformations.get_transformation(rectangles, get_all=True)
    assert placed == transformations.get_transformation(sdata.images["he"], get_all=True)
    table = sdata.tables[TILE_TABLE_KEY]
    assert models.get_table_keys(table) == (TILE_SHAPES_KEY, REGION_KEY, TILE_INDEX_COLUMN)
    np.testing.assert_array_equal(read_tile_embeddings(table).embeddings, tiles.embeddings)


def test_tiles_to_spatialdata_builds_an_object_of_its_own():
    transformations = pytest.importorskip("spatialdata.transformations")

    out = tiles_to_spatialdata(None, encode_array_tiles())

    assert list(out.images) == []
    assert list(out.shapes) == [TILE_SHAPES_KEY]
    assert list(out.tables) == [TILE_TABLE_KEY]
    placed = transformations.get_transformation(out.shapes[TILE_SHAPES_KEY], get_all=True)
    assert placed == {DEFAULT_COORDINATE_SYSTEM: transformations.Identity()}


def test_tiles_to_spatialdata_needs_overwrite_to_replace(sdata: Any):
    tiles = encode_image_tiles(sdata)
    tiles_to_spatialdata(sdata, tiles)
    with pytest.raises(AlhenaValueError, match="already name elements"):
        tiles_to_spatialdata(sdata, tiles)
    assert tiles_to_spatialdata(sdata, tiles, overwrite=True) is sdata


def test_spatialdata_round_trips_through_zarr(
    sdata: Any, cells: CellEmbeddings, result: AnnotationResult, tmp_path: Path
):
    spatialdata = pytest.importorskip("spatialdata")
    tiles = encode_image_tiles(sdata)
    tiles.to_spatialdata(sdata)
    cells.to_spatialdata(sdata, "nuclei")
    result.to_spatialdata(sdata, "nuclei")

    sdata.write(tmp_path / "slide.zarr")
    back = spatialdata.read_zarr(tmp_path / "slide.zarr")

    tiles_back = read_tile_embeddings(back.tables[TILE_TABLE_KEY])
    np.testing.assert_array_equal(tiles_back.embeddings, tiles.embeddings)
    np.testing.assert_array_equal(tiles_back.bounds, tiles.bounds)
    annotations = read_annotations(back.tables[CELL_TABLE_KEY])
    np.testing.assert_array_equal(annotations.cell_id, CELL_IDS)
    assert annotations.labels.tolist() == result.labels.tolist()
    np.testing.assert_array_equal(annotations.cells.cell_space, cells.cell_space)


# --- The sopa patch pipeline ------------------------------------------------------------------


# sopa's own table parse converts its region column to categories, with a warning.
@pytest.mark.filterwarnings("ignore:Converting `region_key")
def test_tile_embeddings_runs_the_encoder_in_sopa(sdata: Any):
    pytest.importorskip("sopa")
    encoder = tiny_encoder()

    key = tile_embeddings(sdata, encoder, mpp=REFERENCE_MPP)

    assert key == TILE_EMBEDDINGS_KEY
    np.testing.assert_array_equal(patch_bounds(sdata, patches=PATCHES_KEY, image_key="he"), BOUNDS)
    expected = encoder.encode_tiles(tile_source(sdata, patches=PATCHES_KEY, mpp=REFERENCE_MPP))
    np.testing.assert_allclose(sdata.tables[key].X, expected.embeddings, rtol=1e-5, atol=1e-5)


def test_alhena_module_is_frozen_and_embeds_each_tile():
    encoder = tiny_encoder()
    # Weights left trainable, as a training loop leaves them, are frozen by the module.
    encoder.backbone.train().requires_grad_(True)
    module = Alhena(encoder)
    tiles = make_tiles()

    embeddings = module(torch.from_numpy(tiles).permute(0, 3, 1, 2).float() / 255.0)

    assert not any(child.training for child in module.modules())
    assert not any(parameter.requires_grad for parameter in module.parameters())
    assert embeddings.shape == (len(tiles), EMBED_DIM)
    expected = encoder.encode(tiles, mpp=REFERENCE_MPP)
    np.testing.assert_allclose(embeddings.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_tile_embeddings_refuses_another_patch_width():
    # Refused before the object is read, so none is needed.
    with pytest.raises(AlhenaValueError, match=f"the model reads {INPUT_SIZE}"):
        tile_embeddings(None, tiny_encoder(), patch_width=2 * INPUT_SIZE, mpp=REFERENCE_MPP)


# --- Optional dependency ----------------------------------------------------------------------


def test_tiles_to_spatialdata_without_spatialdata_names_the_extra(monkeypatch: pytest.MonkeyPatch):
    tiles = encode_array_tiles()
    monkeypatch.setitem(sys.modules, "spatialdata", None)
    with pytest.raises(MissingExtraError, match=r"alhena\[spatial\]"):
        tiles_to_spatialdata(None, tiles)
