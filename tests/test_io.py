"""Tests for the readers of alhena.io: tiles, slides, images and nuclei."""

from __future__ import annotations

import json
import math
import sys
import types
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest

from alhena import INPUT_SIZE, REFERENCE_MPP, AlhenaValueError, GeometryError, MissingExtraError
from alhena.io import (
    CELL_ID_STRIDE,
    IMAGE_FRAME,
    PART_SEPARATOR,
    TileSource,
    as_tile_source,
    global_cell_ids,
    label_map_from_masks,
    read_centroids,
    read_image,
    read_label_image,
    read_label_maps,
    read_nuclei,
    read_polygons,
    read_region,
    read_slide,
    read_tile_origins,
    read_tiles,
    tile_bounds,
)
from support import (
    N_TILES,
    NUCLEI_PER_TILE,
    make_centroids,
    make_label_maps,
    make_polygons,
    make_tiles,
)

# Microns per pixel of the tiles nuclei are read against.
MPP = 0.25

N_NUCLEI = N_TILES * NUCLEI_PER_TILE

# Color of every pixel of the stand-in slide, its channels distinct so that a swap shows.
FILL = (10, 120, 230)


def make_image(height: int = 100, width: int = 140) -> np.ndarray:
    """Return an RGB image of random pixels, larger than the tiles cut from it."""
    return np.random.default_rng(3).integers(0, 256, (height, width, 3), dtype=np.uint8)


def write_png(path: Path, rgb: np.ndarray) -> Path:
    """Write an RGB image losslessly, in the channel order OpenCV writes."""
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write a table as its suffix names it: comma or tab separated, or Parquet."""
    if path.suffix == ".parquet":
        frame.to_parquet(path)
    else:
        frame.to_csv(path, sep="\t" if path.suffix == ".tsv" else ",", index=False)
    return path


def placed_tiles() -> TileSource:
    """Return the shared tiles at the origins of the windows `make_polygons` lays side by side."""
    _, bounds = make_polygons()
    return read_tiles(make_tiles(), mpp=MPP, coords=bounds[:, :2])


def cell_names(count: int) -> list[str]:
    return [f"n{index}" for index in range(count)]


def centroid_table(points: np.ndarray, **columns: Any) -> pd.DataFrame:
    """Return a table of centroids, x and y, with the further columns given."""
    return pd.DataFrame({"x": points[:, 0], "y": points[:, 1], **columns})


def ring(outline: np.ndarray) -> list[list[float]]:
    """Return an outline as a GeoJSON ring, closed by repeating its first vertex."""
    return [*outline.tolist(), outline[0].tolist()]


def polygon_feature(outline: np.ndarray, name: str | None = None) -> dict[str, Any]:
    """Return a GeoJSON feature outlining one nucleus, naming its cell where a name is given."""
    return {
        "type": "Feature",
        "properties": {} if name is None else {"cell_id": name},
        "geometry": {"type": "Polygon", "coordinates": [ring(outline)]},
    }


def write_features(path: Path, features: list[dict[str, Any]]) -> Path:
    """Write a GeoJSON feature collection."""
    collection = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(collection), encoding="utf-8")
    return path


def vertex_table(outlines: list[np.ndarray], names: list[str]) -> pd.DataFrame:
    """Return outlines as a table of vertices, one row per vertex, the rows of a cell together."""
    rows = [(name, x, y) for name, outline in zip(names, outlines, strict=True) for x, y in outline]
    return pd.DataFrame(rows, columns=["cell_id", "vertex_x", "vertex_y"])


def same_outlines(read: list[np.ndarray], given: list[np.ndarray]) -> bool:
    """Report whether the outlines read hold the vertices given, outline by outline."""
    return len(read) == len(given) and all(
        np.allclose(outline, expected) for outline, expected in zip(read, given, strict=True)
    )


# --- Tiles ----------------------------------------------------------------------------------------


def test_read_tiles_reads_a_stacked_npy_file(tmp_path: Path):
    tiles = make_tiles(size=16)
    np.save(tmp_path / "tiles.npy", tiles)

    source = read_tiles(tmp_path / "tiles.npy", mpp=0.5)

    assert len(source) == 3
    assert source.tile_ids == ("0", "1", "2")
    assert source.tile_shape == (16, 16)
    assert source.mpp == 0.5
    assert source[2].dtype == np.uint8
    assert np.array_equal(source[2], tiles[2])


def test_read_tiles_reads_a_directory_in_name_order(tmp_path: Path):
    tiles = make_tiles(size=8)
    for name, tile in zip(["b", "a", "c"], tiles, strict=True):
        write_png(tmp_path / f"{name}.png", tile)

    source = read_tiles(tmp_path)

    assert source.tile_ids == ("a", "b", "c")
    assert np.array_equal(source[0], tiles[1])
    assert np.array_equal(source[1], tiles[0])
    assert [path.name for path in source.paths] == ["a.png", "b.png", "c.png"]


def test_read_tiles_reads_an_hdf5_file_with_names_and_origins(tmp_path: Path):
    tiles = make_tiles(size=8)
    path = tmp_path / "tiles.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=tiles)
        names = np.array([["t0"], ["t1"], ["t2"]], dtype=object)
        handle.create_dataset("barcode", data=names, dtype=h5py.string_dtype("utf-8"))
        handle.create_dataset("coords", data=np.array([[0, 0], [8, 0], [16, 0]]))

    with read_tiles(path) as source:
        assert source.tile_ids == ("t0", "t1", "t2")
        assert source.coords.tolist() == [[0, 0], [8, 0], [16, 0]]
        assert source.coords_frame == "tiles.h5, coords"
        assert np.array_equal(source[2], tiles[2])
        assert source.path == path


def test_read_tiles_reads_an_array_in_batches():
    tiles = make_tiles(5, size=8)
    source = read_tiles(tiles)

    batches = list(source.batches(batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert np.array_equal(np.stack([tile for batch in batches for tile in batch]), tiles)
    assert source.tile_ids == ("0", "1", "2", "3", "4")
    assert source.mpp is None


def test_as_tile_source_wraps_tiles_and_keeps_a_source():
    tiles = make_tiles(2, size=8)

    source = as_tile_source(tiles, mpp=0.5)

    assert isinstance(source, TileSource)
    assert source.mpp == 0.5
    assert np.array_equal(source[1], tiles[1])
    assert as_tile_source(source) is source
    with pytest.raises(AlhenaValueError, match=r"0\.5 micron per pixel"):
        as_tile_source(source, mpp=2.0)


def test_tile_bounds_span_the_scaled_tile_size():
    tiles = np.zeros((2, 8, 6, 3), dtype=np.uint8)
    source = read_tiles(tiles, coords=np.array([[0, 0], [12, 0]]), coords_scale=2.0)
    # A tile 6 pixels wide and 8 high spans twice as many pixels of the frame of its origin.
    assert tile_bounds(source).tolist() == [[0, 0, 12, 16], [12, 0, 24, 16]]


def test_tile_bounds_refuse_tiles_without_origins():
    with pytest.raises(AlhenaValueError, match="declares no coordinates"):
        tile_bounds(read_tiles(make_tiles(2, size=8)))


def test_read_tile_origins_matches_rows_to_tiles_by_name(tmp_path: Path):
    folder = tmp_path / "tiles"
    folder.mkdir()
    for name, tile in zip(["a", "b", "c"], make_tiles(size=8), strict=True):
        write_png(folder / f"{name}.png", tile)
    table = pd.DataFrame({"tile_id": ["c", "a", "b"], "x": [16, 0, 8], "y": [0, 0, 0]})

    origins = read_tile_origins(
        write_table(table, tmp_path / "origins.csv"), tiles=read_tiles(folder)
    )

    assert origins.dtype == np.int64
    assert origins.tolist() == [[0, 0], [8, 0], [16, 0]]


# --- Slides ---------------------------------------------------------------------------------------


@pytest.fixture
def slide_file(tmp_path: Path) -> Path:
    """Return a file opening with a TIFF header, which the stand-in tiffslide reads as a slide."""
    path = tmp_path / "case.svs"
    path.write_bytes(b"II*\x00slide")
    return path


def fake_tiffslide(
    monkeypatch: pytest.MonkeyPatch,
    properties: dict[str, Any],
    dimensions: tuple[tuple[int, int], ...] = ((64, 48),),
    downsamples: tuple[float, ...] = (1.0,),
    ome_metadata: str | None = None,
) -> list[Any]:
    """Install a stand-in tiffslide opening a slide of one color; return the slides it opens."""
    opened: list[Any] = []

    class StandInSlide:
        """A slide of one color, laid out as the arguments say, recording the regions read."""

        def __init__(self, path: str) -> None:
            self.properties = dict(properties)
            self.level_dimensions = dimensions
            self.level_downsamples = downsamples
            self.ts_tifffile = types.SimpleNamespace(ome_metadata=ome_metadata)
            self.calls: list[tuple[Any, ...]] = []
            opened.append(self)

        def read_region(self, location: Any, level: int, size: Any) -> np.ndarray:
            self.calls.append((tuple(location), level, tuple(size)))
            region = np.empty((size[1], size[0], 3), dtype=np.uint8)
            region[...] = FILL
            # Past the edge of the level, tiffslide returns black.
            width, height = dimensions[level]
            left, top = (int(value / downsamples[level]) for value in location)
            region[:, max(width - left, 0) :] = 0
            region[max(height - top, 0) :] = 0
            return region

        def close(self) -> None:
            """Release nothing: the stand-in holds no file."""

    module = types.ModuleType("tiffslide")
    module.TiffSlide = StandInSlide
    monkeypatch.setitem(sys.modules, "tiffslide", module)
    return opened


def ome_xml(**pixels: Any) -> str:
    """Return OME metadata describing one image the size of the stand-in slide's level zero."""
    attributes = {"SizeX": 64, "SizeY": 48, **pixels}
    written = " ".join(f'{name}="{value}"' for name, value in attributes.items())
    return (
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image ID="Image:0"><Pixels ID="Pixels:0" {written}/></Image></OME>'
    )


def tag_properties(per_unit: float, unit: str) -> dict[str, Any]:
    """Return the properties tiffslide reports for a TIFF declaring its resolution in tags."""
    microns = {"INCH": 25_400.0, "CENTIMETER": 10_000.0}[unit]
    return {
        "tiff.XResolution": per_unit,
        "tiff.YResolution": per_unit,
        "tiff.ResolutionUnit": unit,
        "tiffslide.mpp-x": microns / per_unit,
        "tiffslide.mpp-y": microns / per_unit,
    }


def test_read_slide_cuts_a_grid_padded_white_past_the_edge(
    monkeypatch: pytest.MonkeyPatch, slide_file: Path
):
    fake_tiffslide(monkeypatch, {"aperio.MPP": 0.25})

    source = read_slide(slide_file, tile_px=32)

    # The 48 rows of the slide hold one row of tiles and half of a second.
    assert source.coords.tolist() == [[0, 0], [32, 0], [0, 32], [32, 32]]
    assert source.tile_ids == ("x0_y0", "x32_y0", "x0_y32", "x32_y32")
    assert source.tile_shape == (32, 32)
    assert source.mpp == 0.25
    assert source.path == slide_file
    assert (source[1] == FILL).all()
    assert (source[2][:16] == FILL).all()
    assert (source[2][16:] == 255).all()


def test_read_slide_places_a_coarser_level_in_level_zero_pixels(
    monkeypatch: pytest.MonkeyPatch, slide_file: Path
):
    opened = fake_tiffslide(
        monkeypatch, {"aperio.MPP": 0.25}, dimensions=((128, 96), (64, 48)), downsamples=(1.0, 2.0)
    )

    source = read_slide(slide_file, tile_px=32, level=1, edge="drop")

    assert source.mpp == pytest.approx(0.5)
    assert source.coords_scale == 2.0
    assert source.coords.tolist() == [[0, 0], [64, 0]]
    assert tile_bounds(source).tolist() == [[0, 0, 64, 64], [64, 0, 128, 64]]
    assert source[1].shape == (32, 32, 3)
    assert opened[-1].calls == [((64, 0), 1, (32, 32))]


def test_read_slide_reads_a_pyramidal_tiff(tmp_path: Path):
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("tiffslide")
    image = make_image(192, 256)
    path = tmp_path / "section.tif"
    # 40,000 pixels per centimeter are 0.25 micron per pixel.
    with tifffile.TiffWriter(path) as tiff:
        tiff.write(
            image,
            subifds=1,
            tile=(64, 64),
            photometric="rgb",
            resolution=(40_000, 40_000),
            resolutionunit="CENTIMETER",
        )
        tiff.write(image[::2, ::2], subfiletype=1, tile=(64, 64), photometric="rgb")

    with read_slide(path, tile_px=64) as source:
        assert source.mpp == pytest.approx(0.25)
        assert source.mpp_source == "TIFF resolution tags"
        assert len(source) == 12
        assert np.array_equal(source[5], image[64:128, 64:128])
    with read_slide(path, tile_px=32, level=1) as coarse:
        assert coarse.mpp == pytest.approx(0.5)
        assert coarse.coords[1].tolist() == [64, 0]
        assert np.array_equal(coarse[1], image[::2, ::2][:32, 32:64])


@pytest.mark.parametrize(
    ("properties", "ome", "mpp", "where"),
    [
        pytest.param({"aperio.MPP": 0.2527}, None, 0.2527, "aperio.MPP", id="aperio"),
        pytest.param(
            {},
            ome_xml(PhysicalSizeX=273.8, PhysicalSizeXUnit="nm"),
            0.2738,
            "OME PhysicalSizeX",
            id="ome",
        ),
        pytest.param(
            tag_properties(40_000, "CENTIMETER"), None, 0.25, "TIFF resolution tags", id="tags"
        ),
    ],
)
def test_read_slide_reads_the_resolution_the_slide_declares(
    monkeypatch: pytest.MonkeyPatch,
    slide_file: Path,
    properties: dict[str, Any],
    ome: str | None,
    mpp: float,
    where: str,
):
    fake_tiffslide(monkeypatch, properties, ome_metadata=ome)

    source = read_slide(slide_file, tile_px=32)

    assert source.mpp == pytest.approx(mpp)
    assert source.mpp_source == where


@pytest.mark.parametrize(
    ("properties", "refusal"),
    [
        pytest.param({}, "declares no resolution", id="none"),
        pytest.param(tag_properties(72, "INCH"), "XResolution 72 pixels per inch", id="screen"),
    ],
)
def test_read_slide_refuses_a_missing_or_implausible_resolution(
    monkeypatch: pytest.MonkeyPatch, slide_file: Path, properties: dict[str, Any], refusal: str
):
    fake_tiffslide(monkeypatch, properties)

    with pytest.raises(GeometryError, match=refusal):
        read_slide(slide_file, tile_px=32)
    declared = read_slide(slide_file, tile_px=32, mpp=0.25)
    assert declared.mpp == 0.25
    assert declared.mpp_source == "mpp argument"


def test_read_slide_names_the_extra_when_tiffslide_is_missing(
    monkeypatch: pytest.MonkeyPatch, slide_file: Path
):
    monkeypatch.setitem(sys.modules, "tiffslide", None)
    with pytest.raises(MissingExtraError, match=r"alhena\[wsi\]"):
        read_slide(slide_file)


# --- Images and regions ---------------------------------------------------------------------------


def test_read_image_cuts_tiles_filling_the_canvas():
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    source = read_image(image, mpp=0.25)

    side = math.ceil(INPUT_SIZE * REFERENCE_MPP / 0.25)
    assert source.tile_shape == (side, side)
    # The last column and row are moved inward to end at the edge of the image.
    end = 500 - side
    assert source.coords.tolist() == [[0, 0], [end, 0], [0, end], [end, end]]
    assert source.image_shape == (500, 500)
    assert source.coords_frame == IMAGE_FRAME
    assert source.mpp == 0.25


@pytest.mark.parametrize("form", ["array", "png", "npy"])
def test_read_image_tiles_hold_the_image_pixels(tmp_path: Path, form: str):
    image = make_image()
    source: Any = image
    if form == "png":
        source = write_png(tmp_path / "image.png", image)
    elif form == "npy":
        source = tmp_path / "image.npy"
        np.save(source, image)

    tiles = read_image(source, tile_px=48, edge="pad")

    assert len(tiles) == 9
    for index, (x, y) in enumerate(tiles.coords):
        inside = image[y : y + 48, x : x + 48]
        height, width = inside.shape[:2]
        tile = tiles[index]
        assert np.array_equal(tile[:height, :width], inside)
        assert (tile[height:] == 255).all()
        assert (tile[:, width:] == 255).all()
    assert tiles.path == (None if form == "array" else source)
    tiles.close()
    assert np.array_equal(tiles[0][:48, :48], image[:48, :48])


def test_read_image_needs_an_mpp_or_a_tile_side():
    with pytest.raises(AlhenaValueError, match="neither mpp nor tile_px"):
        read_image(make_image())


def test_read_region_reads_the_coarsest_level_within_downsample(
    monkeypatch: pytest.MonkeyPatch, slide_file: Path
):
    opened = fake_tiffslide(
        monkeypatch,
        {},
        dimensions=((128, 96), (64, 48), (32, 24)),
        downsamples=(1.0, 2.0, 4.0),
    )

    image, extent = read_region(slide_file, window=(32, 16, 96, 80), downsample=3)

    assert image.shape == (32, 32, 3)
    assert opened[-1].calls == [((32, 16), 1, (32, 32))]
    assert extent == (31.5, 95.5, 79.5, 15.5)


def test_read_region_decimates_an_image_array():
    image = make_image()

    picture, extent = read_region(image, window=(8, 4, 40, 36), downsample=4.5)

    assert np.array_equal(picture, image[4:36:4, 8:40:4])
    assert extent == (7.5, 39.5, 35.5, 3.5)


# --- Label maps -----------------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["npy", "directory", "hdf5"])
def test_read_label_maps_reads_npy_directory_and_hdf5(tmp_path: Path, form: str):
    maps = make_label_maps()
    source = tmp_path / {"npy": "maps.npy", "directory": "maps", "hdf5": "maps.h5"}[form]
    if form == "npy":
        np.save(source, maps)
    elif form == "directory":
        source.mkdir()
        for index, label_map in enumerate(maps):
            assert cv2.imwrite(str(source / f"t{index}.png"), label_map.astype(np.uint16))
    else:
        with h5py.File(source, "w") as handle:
            handle.create_dataset("labels", data=maps)

    tiles = read_tiles(make_tiles(size=8), tile_ids=["t0", "t1", "t2"])
    read = read_label_maps(source, tiles=tiles)

    assert read["tile_indices"].tolist() == [0, 1, 2]
    for tile, (label_map, given) in enumerate(zip(read["label_maps"], maps, strict=True)):
        assert np.array_equal(label_map, np.where(given > 0, tile * CELL_ID_STRIDE + given, 0))
    kept = read_label_maps(source, tiles=tiles, renumber=False)["label_maps"]
    assert all(np.array_equal(kept[tile], maps[tile]) for tile in range(N_TILES))


def test_read_label_maps_needs_the_tiles_its_files_are_named_after(tmp_path: Path):
    np.save(tmp_path / "3.npy", make_label_maps(1)[0])
    with pytest.raises(AlhenaValueError, match="tiles="):
        read_label_maps(tmp_path)


def test_read_label_maps_matches_files_to_tiles_by_name(tmp_path: Path):
    tiles = read_tiles(make_tiles(2), tile_ids=["left", "right"])
    np.save(tmp_path / "right.npy", make_label_maps(1)[0])

    read = read_label_maps(tmp_path, tiles=tiles)

    assert read["tile_indices"].tolist() == [1]
    identifiers = [CELL_ID_STRIDE + number for number in range(1, NUCLEI_PER_TILE + 1)]
    assert np.unique(read["label_maps"][0]).tolist() == [0, *identifiers]
    np.save(tmp_path / "elsewhere.npy", make_label_maps(1)[0])
    with pytest.raises(AlhenaValueError, match="names no tile"):
        read_label_maps(tmp_path, tiles=tiles)


def label_image() -> np.ndarray:
    """Return a label image of `make_image`: nuclei 7 and 3 across two tiles each, 9 in one."""
    labels = np.zeros((100, 140), dtype=np.int32)
    labels[10:20, 93:97] = 7
    labels[60:70, 10:20] = 3
    labels[97:100, 130:140] = 9
    return labels


@pytest.mark.parametrize("form", ["array", "npy", "tif"])
def test_read_label_image_cuts_the_map_under_each_tile(tmp_path: Path, form: str):
    tiles = read_image(make_image(), tile_px=48)
    labels = label_image()
    source: Any = labels
    if form == "npy":
        source = tmp_path / "labels.npy"
        np.save(source, labels)
    elif form == "tif":
        source = tmp_path / "labels.tif"
        assert cv2.imwrite(str(source), labels.astype(np.uint16))

    read = read_label_image(source, tiles=tiles)

    names = [tiles.tile_ids[index] for index in read["tile_indices"]]
    assert names == ["x48_y0", "x92_y0", "x0_y48", "x0_y52", "x92_y52"]
    for index, label_map in zip(read["tile_indices"], read["label_maps"], strict=True):
        x, y = tiles.coords[index]
        assert np.array_equal(label_map, labels[y : y + 48, x : x + 48])


def test_label_map_from_masks_favors_the_smaller_nucleus():
    masks = np.zeros((5, 8, 8), dtype=bool)
    masks[0, 0:6, 0:6] = True
    masks[1, 4:6, 4:6] = True
    # The third nucleus lies under the two smaller ones below, so it keeps no pixel.
    masks[2, 6:8, 0:2] = True
    masks[3, 6:7, 0:2] = True
    masks[4, 7:8, 0:2] = True

    label_map, lost = label_map_from_masks(masks, [11, 12, 13, 14, 15])

    expected = np.zeros((8, 8), dtype=np.int64)
    expected[0:6, 0:6] = 11
    expected[4:6, 4:6] = 12
    expected[6:7, 0:2] = 14
    expected[7:8, 0:2] = 15
    assert np.array_equal(label_map, expected)
    assert lost.tolist() == [13]


# --- Centroids ------------------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_read_centroids_reads_a_table_in_tile_pixels(tmp_path: Path, suffix: str):
    points, tile_index = make_centroids()
    names = cell_names(len(points))
    table = centroid_table(points, tile_index=tile_index, cell_id=names, radius_px=6.0)

    read = read_centroids(write_table(table, tmp_path / f"nuclei{suffix}"), tiles=make_tiles())

    assert np.array_equal(read["centroids"], points)
    assert read["tile_indices"].tolist() == tile_index.tolist()
    assert read["cell_ids"].tolist() == names
    assert read["radius_px"].tolist() == [6.0] * len(points)
    assert read["tile_shape"] == (INPUT_SIZE, INPUT_SIZE)


def test_read_centroids_places_rows_on_the_tiles_holding_them(tmp_path: Path):
    points, tile_index = make_centroids()
    tiles = placed_tiles()
    path = write_table(centroid_table(points + tiles.coords[tile_index]), tmp_path / "nuclei.csv")

    read = read_centroids(path, tiles=tiles, radius_px=6.0, frame="origins")

    assert read["tile_indices"].tolist() == tile_index.tolist()
    assert np.allclose(read["centroids"], points)
    assert read["cell_ids"].tolist() == list(range(len(points)))
    assert read["tile_shape"] == (INPUT_SIZE, INPUT_SIZE)


@pytest.mark.parametrize("frame", ["tile", "origins"])
def test_read_centroids_reads_microns_as_pixels(frame: str):
    points, tile_index = make_centroids()
    tiles = placed_tiles()
    if frame == "tile":
        table = centroid_table(points * MPP, tile_index=tile_index)
    else:
        table = centroid_table((points + tiles.coords[tile_index]) * MPP)

    read = read_centroids(table, tiles=tiles, radius_px=6.0, frame=frame, units="um")

    assert np.allclose(read["centroids"], points)
    assert read["tile_indices"].tolist() == tile_index.tolist()


def test_read_centroids_refuses_a_centroid_outside_its_tile():
    points, tile_index = make_centroids()
    tiles = placed_tiles()
    # Positions in the frame of the tile origins, read as positions in each tile.
    table = centroid_table(points + tiles.coords[tile_index], tile_index=tile_index)
    with pytest.raises(AlhenaValueError, match=r"8 of 12 centroids .* fall outside"):
        read_centroids(table, tiles=tiles, radius_px=6.0)


# --- Polygons -------------------------------------------------------------------------------------


def test_read_polygons_reads_geojson_against_the_tiles(tmp_path: Path):
    outlines, bounds = make_polygons()
    names = cell_names(len(outlines))
    features = list(map(polygon_feature, outlines, names))
    tiles = placed_tiles()

    read = read_polygons(write_features(tmp_path / "nuclei.geojson", features), tiles=tiles)

    assert read["cell_ids"].tolist() == names
    assert same_outlines(read["polygons"], outlines)
    assert np.array_equal(read["tile_bounds"], bounds)
    assert read["tile_scale"] == 1.0
    assert read["tile_frame"] == tiles.coords_frame


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".parquet"])
def test_read_polygons_reads_a_table_of_vertices(tmp_path: Path, suffix: str):
    outlines, _ = make_polygons()
    names = cell_names(len(outlines))
    path = write_table(vertex_table(outlines, names), tmp_path / f"vertices{suffix}")

    read = read_polygons(path)

    assert read["cell_ids"].tolist() == names
    assert same_outlines(read["polygons"], outlines)


def test_read_polygons_reads_geoparquet(tmp_path: Path):
    geopandas = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    outlines, _ = make_polygons()
    names = cell_names(len(outlines))
    geometry = [shapely.Polygon(outline) for outline in outlines]
    path = tmp_path / "nuclei.parquet"
    geopandas.GeoDataFrame({"cell_id": names}, geometry=geometry).to_parquet(path)

    read = read_polygons(path)

    assert read["cell_ids"].tolist() == names
    assert same_outlines(read["polygons"], outlines)
    assert read_nuclei(path)["cell_ids"].tolist() == names


def test_read_polygons_refuses_a_multipart_boundary_by_default(tmp_path: Path):
    outlines, _ = make_polygons(n_tiles=1, per_tile=2)
    geometry = {"type": "MultiPolygon", "coordinates": [[ring(outline)] for outline in outlines]}
    feature = {"type": "Feature", "properties": {"cell_id": "a"}, "geometry": geometry}
    path = write_features(tmp_path / "nuclei.geojson", [feature])

    with pytest.raises(AlhenaValueError, match="MultiPolygon of 2 parts"):
        read_polygons(path)
    split = read_polygons(path, multipart="split")
    assert split["cell_ids"].tolist() == [f"a{PART_SEPARATOR}0", f"a{PART_SEPARATOR}1"]
    assert same_outlines(split["polygons"], outlines)


# --- Nuclei from any source -----------------------------------------------------------------------


def write_nucleus_sources(folder: Path) -> None:
    """Write the shared nuclei as label maps, centroids, GeoJSON and a table of vertices."""
    maps = folder / "maps"
    maps.mkdir()
    for index, label_map in enumerate(make_label_maps()):
        np.save(maps / f"{index}.npy", label_map)
    points, tile_index = make_centroids()
    write_table(
        centroid_table(points, tile_index=tile_index, radius_px=6.0), folder / "centroids.csv"
    )
    outlines, _ = make_polygons()
    write_features(folder / "polygons.geojson", [polygon_feature(outline) for outline in outlines])
    write_table(vertex_table(outlines, cell_names(len(outlines))), folder / "vertices.csv")


@pytest.mark.parametrize(
    ("name", "kind", "key", "count"),
    [
        pytest.param("maps", None, "label_maps", N_TILES, id="label-maps"),
        pytest.param("centroids.csv", None, "centroids", N_NUCLEI, id="table"),
        pytest.param("polygons.geojson", None, "polygons", N_NUCLEI, id="geojson"),
        pytest.param("vertices.csv", "polygons", "polygons", N_NUCLEI, id="kind"),
    ],
)
def test_read_nuclei_chooses_the_reader_from_the_source(
    tmp_path: Path, name: str, kind: str | None, key: str, count: int
):
    write_nucleus_sources(tmp_path)

    tiles = read_tiles(make_tiles(), tile_ids=["0", "1", "2"]) if key == "label_maps" else None
    read = read_nuclei(tmp_path / name, tiles=tiles, kind=kind)

    assert len(read[key]) == count


def test_global_cell_ids_give_each_tile_its_own_block():
    assert global_cell_ids(0, [1, 2]).tolist() == [1, 2]
    assert global_cell_ids(1, [1, 2]).tolist() == [CELL_ID_STRIDE + 1, CELL_ID_STRIDE + 2]
    assert global_cell_ids([0, 2], [5, 5], stride=10).tolist() == [5, 25]
    with pytest.raises(AlhenaValueError, match="block of 10 identifiers"):
        global_cell_ids(0, [10], stride=10)


def test_transform_carries_nuclei_from_another_frame(tmp_path: Path):
    # Nuclei written on an image at half the resolution of the tiles, its origin at x 10, y 20.
    matrix = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 20.0]])
    np.savetxt(tmp_path / "transform.csv", matrix, delimiter=",")
    tiles = placed_tiles()
    points, tile_index = make_centroids()
    elsewhere = (points + tiles.coords[tile_index] - matrix[:, 2]) / 2.0
    outlines, _ = make_polygons()
    features = [polygon_feature((outline - matrix[:, 2]) / 2.0) for outline in outlines]

    centroids = read_centroids(
        centroid_table(elsewhere),
        tiles=tiles,
        radius_px=6.0,
        frame="origins",
        transform=tmp_path / "transform.csv",
    )
    polygons = read_polygons(
        write_features(tmp_path / "nuclei.geojson", features), tiles=tiles, transform=matrix
    )

    assert centroids["tile_indices"].tolist() == tile_index.tolist()
    assert np.allclose(centroids["centroids"], points)
    assert same_outlines(polygons["polygons"], outlines)


# --- Unknown files --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(read_tiles, id="read_tiles"),
        pytest.param(partial(read_image, tile_px=48), id="read_image"),
        pytest.param(read_centroids, id="read_centroids"),
        pytest.param(read_nuclei, id="read_nuclei"),
    ],
)
def test_readers_refuse_an_unknown_suffix(tmp_path: Path, reader: Callable[[Path], Any]):
    path = tmp_path / "data.zip"
    path.write_bytes(b"archive")
    with pytest.raises(AlhenaValueError, match=r"has suffix '\.zip'"):
        reader(path)
