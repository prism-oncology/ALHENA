"""Figures of tile embeddings, cell-type annotations and their evaluation, each drawn on an `Axes`.

Classes take the colors of `CLASS_COLORS` unless given a palette, and a class the reference cannot
express is marked and named, never left out.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import numpy as np
import pandas as pd

from alhena._constants import (
    CLASS_AXES,
    CLASS_COLORS,
    DEFAULT_AXIS,
    INPUT_SIZE,
    UNKNOWN_LABEL,
    label_order,
)
from alhena._errors import PACKAGE_PREFIXES, AlhenaValueError, GeometryWarning
from alhena._plotting import (
    _as_display_tile,
    _as_rgb,
    _axes,
    _category_palette,
    _check_unit_interval,
    _drawn_as_categories,
    _image_extent,
    _is_table,
    _missing_note,
    _region_bounds,
    _require,
    _shade,
    _source_tile,
    _table_column,
)
from alhena.cells._drawing import _RGBA, _cell_labels, _pixel_tint, _tile_nuclei, _token_tint
from alhena.io import TileSource
from alhena.io._saved import _as_annotations
from alhena.types import AnnotationResult, ScoreReport, TileEmbeddings

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes

RGB: TypeAlias = "tuple[float, float, float]"
"""Red, green and blue on the unit scale."""

NO_CALL_NAME: Final[str] = "no annotation"
"""Name given in a legend or a composition bar to the cells without an annotation."""

NOT_IN_REFERENCE: Final[str] = "not in reference"
"""Note attached to a class the reference holds no cell of."""

MISSING_HATCH: Final[str] = "///"
"""Hatch marking a bar or a patch standing for a class the reference cannot express."""

_OVERLAY_SIZE: Final[tuple[float, float]] = (5.0, 5.0)
_CHART_SIZE: Final[tuple[float, float]] = (6.0, 4.0)
_MATRIX_SIZE: Final[tuple[float, float]] = (5.5, 5.0)
_LEGEND_SIZE: Final[tuple[float, float]] = (3.0, 2.0)
_MAP_SIZE: Final[tuple[float, float]] = (6.0, 6.0)


# --- Palette --------------------------------------------------------------------------------


def class_color(name: str, colors: Mapping[str, Any] | None = None) -> RGB:
    """Return the palette color of one class.

    Args:
        name: Class name. The empty string, which a result carries where no annotation was
            assigned, takes the color of `NO_CALL_NAME`.
        colors: Palette of a custom taxonomy, replacing the default palette. Integer triples
            are read on the byte scale and everything else as a matplotlib color.

    Returns:
        Red, green and blue on the unit scale.

    Raises:
        AlhenaValueError: If the palette in force declares no color for the class.
    """
    palette: Mapping[str, Any] = CLASS_COLORS if colors is None else colors
    if name in ("", NO_CALL_NAME):
        return _as_rgb(palette.get(NO_CALL_NAME, CLASS_COLORS[UNKNOWN_LABEL]))
    if name not in palette:
        if colors is not None:
            raise AlhenaValueError(
                f"No color is declared for class {name!r} in the colors given, which cover "
                f"{sorted(str(key) for key in colors)}. Give colors= a color for every class drawn."
            )
        raise AlhenaValueError(
            f"No color is declared for class {name!r}. The default palette covers "
            f"{sorted(CLASS_COLORS)}; pass colors= to give the palette of a custom taxonomy."
        )
    return _as_rgb(palette[name])


def class_palette(names: Sequence[str], colors: Mapping[str, Any] | None = None) -> list[RGB]:
    """Return the palette colors of several classes, in the order given.

    Args:
        names: Class names.
        colors: Palette of a custom taxonomy, replacing the default palette.

    Returns:
        One RGB triple per name.

    Raises:
        AlhenaValueError: If the palette in force declares no color for one of the classes.
    """
    return [class_color(str(name), colors=colors) for name in names]


# --- Shared helpers -------------------------------------------------------------------------


def missing_class_names(source: Any, class_order: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return the classes a reference holds no cell of.

    Args:
        source: A `Reference`, a `ReferenceInfo`, an `AnnotationResult` or a `ScoreReport`, which
            record them, a sequence of class names, or None.
        class_order: Classes of the figure being drawn. When given, every name must be one of
            them, so a misspelling is caught rather than quietly marking nothing.

    Returns:
        The class names, in the order they were declared in.

    Raises:
        AlhenaValueError: If a name is given alone rather than in a sequence, or is not a class.
    """
    if source is None:
        return ()
    if isinstance(source, str):
        raise AlhenaValueError(
            f"Expected the classes a reference cannot express, got the single name {source!r}. "
            "Pass a sequence of names, or the reference itself."
        )
    names = getattr(getattr(source, "info", source), "missing_classes", None)
    if names is None:
        names = source
    resolved = tuple(str(name) for name in names)
    if class_order is not None:
        known = {str(name) for name in class_order}
        unknown = tuple(name for name in resolved if name not in known)
        if unknown:
            raise AlhenaValueError(
                f"{unknown} are named as classes the reference cannot express but are not "
                f"classes of this figure, whose axis is {tuple(class_order)}."
            )
    return resolved


# --- Overlay --------------------------------------------------------------------------------


def overlay(
    tile: TileSource | np.ndarray,
    nuclei: Any,
    result: AnnotationResult | AnnData,
    tile_index: int | None = None,
    ax: Axes | None = None,
    alpha: float = 0.6,
    shade_by_confidence: bool = False,
    show_no_call: bool = True,
    colors: Mapping[str, Any] | None = None,
    title: str | None = None,
    name: str | None = None,
) -> Axes:
    """Draw one tile with every nucleus tinted by the class it was annotated with.

    The nuclei, in any form `encode_cells` reads, meet their annotations by cell identifier; a
    nucleus without one keeps the color of `NO_CALL_NAME`. Outlines and label maps are tinted pixel
    for pixel, centroids over the disks the cell path pools, and a membership table token by token,
    each token to the cell holding most of its pixels, ties to the lowest identifier; a tile other
    than the canvas size is then warned about. A label map of instance numbers is numbered for the
    tile `tile_index` names, as `read_label_maps` numbers it, or tile 0 with a warning.

    Args:
        tile: The tile source the nuclei were read against, whose tile `tile_index` is drawn, or
            one `(H, W, 3)` tile, uint8 bytes or floats in [0, 1], as read rather than normalized.
        nuclei: The nuclei as an `alhena.io` reader returns them, label maps one per tile, an
            `(H, W)` label map of this tile, or a membership table with `MEMBERSHIP_COLUMNS`.
        result: Annotations covering the cells of this tile, or the table of cells holding them.
        tile_index: Tile drawn, by its index among the tiles the nuclei were read against. Tile 0
            when None, which a source must then hold alone.
        ax: Axes to draw on. A figure is created when none is given.
        alpha: Opacity of the tint, in (0, 1].
        shade_by_confidence: Scale each nucleus opacity by its confidence, from the confidence
            floor of the axis up to mass one.
        show_no_call: Draw the nuclei without an annotation in the color of `NO_CALL_NAME`.
        colors: Palette of a custom taxonomy, replacing the default palette.
        title: Title of the panel. The counts of annotated and unannotated nuclei are used when none
            is given.
        name: Annotation of a table to draw, the name it was stored under. May be left out where
            the table holds one.

    Returns:
        The axes drawn on, carrying the tile and the tint as two images.

    Raises:
        AlhenaValueError: If an input is not a form ALHENA reads, or the tile or `alpha` is invalid.
        GeometryError: If the tile or a label map is malformed, or of another size than the nuclei.
        MembershipError: If the membership table is malformed or holds no row of this tile.
    """
    opacity = _check_unit_interval(alpha, "alpha")
    result = _as_annotations(result, name)
    image = _as_display_tile(
        _source_tile(tile, tile_index) if isinstance(tile, TileSource) else tile
    )
    labels = np.asarray(result.labels).reshape(-1)
    confidence = np.asarray(result.confidence, dtype=np.float64).reshape(-1)
    valid = np.asarray(result.valid, dtype=bool).reshape(-1)
    floor = 1.0 / max(len(result.class_order), 1)
    # Identifiers are unique across the tiles, so a nucleus meets its annotation by identifier.
    lookup = {key: row for row, key in enumerate(np.asarray(result.cell_id).reshape(-1).tolist())}

    seen: dict[Any, bool] = {}
    tinted: dict[Any, _RGBA] = {}

    def rgba(key: Any) -> _RGBA:
        if key in tinted:
            return tinted[key]
        row = lookup.get(key)
        name = "" if row is None else str(labels[row])
        called = row is not None and bool(valid[row]) and name != ""
        if not called:
            color = (*class_color("", colors=colors), opacity if show_no_call else 0.0)
        else:
            weight = _shade(confidence[row], floor, opacity) if shade_by_confidence else opacity
            color = (*class_color(name, colors=colors), weight)
        seen[key] = called
        tinted[key] = color
        return color

    height, width = image.shape[:2]
    extent = (-0.5, width - 0.5, height - 0.5, -0.5)
    drawn = 0 if tile_index is None else int(tile_index)
    if isinstance(nuclei, pd.DataFrame):
        if (height, width) != (INPUT_SIZE, INPUT_SIZE):
            warnings.warn(
                f"The token grid is defined on the {INPUT_SIZE} by {INPUT_SIZE} canvas the "
                f"encoder reads, and this tile is {width} by {height}, so the tint is stretched "
                "over it and does not name a nucleus pixel for pixel. Pass the nuclei the table "
                "was built from for an exact overlay.",
                GeometryWarning,
                skip_file_prefixes=PACKAGE_PREFIXES,
            )
        tint = _token_tint(nuclei, drawn, rgba)
    elif isinstance(nuclei, np.ndarray) and nuclei.ndim == 2:
        tint = _pixel_tint(_cell_labels(nuclei, (height, width), result, tile_index), rgba)
    else:
        codes, keys = _tile_nuclei(nuclei, drawn, (height, width))
        if keys is None:
            tint = _pixel_tint(_cell_labels(codes, (height, width), result, tile_index), rgba)
        else:
            tint = _pixel_tint(codes, lambda code: rgba(keys[code - 1]))

    typed = sum(1 for called in seen.values() if called)
    noun = "nucleus" if typed == 1 else "nuclei"
    axes = _axes(ax, _OVERLAY_SIZE)
    axes.imshow(image, extent=extent, interpolation="nearest")
    axes.imshow(np.clip(tint, 0.0, 1.0), extent=extent, interpolation="nearest")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(
        f"{typed} {noun} annotated, {len(seen) - typed} without an annotation"
        if title is None
        else title
    )
    return axes


# --- Tile regions ---------------------------------------------------------------------------


def tile_regions(
    tiles: TileEmbeddings | AnnData | np.ndarray,
    values: Any,
    ax: Axes | None = None,
    image: np.ndarray | None = None,
    extent: Sequence[float] | None = None,
    colors: Mapping[Any, Any] | None = None,
    cmap: str = "viridis",
    alpha: float = 0.5,
    legend: bool = True,
    title: str | None = None,
) -> Axes:
    """Draw every tile region filled with the color of its value, over the image it was read from.

    Categories, such as tile clusters, take their colors from `colors`, else from a qualitative
    palette in the order of the categories, with a legend; numbers take a color from `cmap`, with a
    colorbar. A region runs from its first pixel to its last, pixel centers at integer coordinates,
    as `alhena.io.read_region` places its image. A tile without a value is left unfilled.

    Args:
        tiles: Tile embeddings carrying spatial bounds, their table as `TileEmbeddings.to_anndata`
            lays it out, or an `(n_tiles, 4)` array of x0, y0, x1, y1 bounds, upper bounds excluded.
        values: One value per tile, in the order of the tiles, or the name of a column of the table.
            A pandas categorical, text or booleans are drawn as categories, numbers on a scale.
        ax: Axes to draw on. A figure is created when none is given.
        image: The `(H, W, 3)` image under the tiles, drawn first.
        extent: Left, right, bottom and top edges of the image in the frame of the bounds, as
            `alhena.io.read_region` returns them. The pixels of the image itself when None.
        colors: Color of each category, replacing the qualitative palette. For a column of the
            table, the colors scanpy stores beside it in `uns` when none are given.
        cmap: Colormap of numeric values.
        alpha: Opacity of the fill, in (0, 1].
        legend: Draw the legend of the categories, or the colorbar of the numbers.
        title: Title of the panel.

    Returns:
        The axes drawn on, carrying the image and one patch per tile with a value.

    Raises:
        AlhenaValueError: If the tiles carry no bounds, or the values or a color do not fit them.
        GeometryError: If the image is not an `(H, W, 3)` image.
    """
    opacity = _check_unit_interval(alpha, "alpha")
    if isinstance(values, str):
        values, colors = _table_column(tiles, values, colors)
    bounds = _region_bounds(tiles)
    series = values.reset_index(drop=True) if isinstance(values, pd.Series) else pd.Series(values)
    if len(series) != bounds.shape[0]:
        raise AlhenaValueError(
            f"{len(series)} values were given for {bounds.shape[0]} tiles, where one value per "
            "tile, in the order of the tiles, is expected."
        )
    patches = _require("matplotlib.patches")
    collections = _require("matplotlib.collections")

    axes = _axes(ax, _MAP_SIZE)
    if image is None:
        left, top = bounds[:, :2].min(axis=0) - 0.5
        right, bottom = bounds[:, 2:].max(axis=0) - 0.5
    else:
        picture = _as_display_tile(image, "image")
        if extent is None:
            left, right, bottom, top = -0.5, picture.shape[1] - 0.5, picture.shape[0] - 0.5, -0.5
        else:
            left, right, bottom, top = _image_extent(extent)
        axes.imshow(picture, extent=(left, right, bottom, top))

    filled = series.notna().to_numpy()
    rectangles = [
        patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0)
        for x0, y0, x1, y1 in bounds[filled].tolist()
    ]
    name = series.name if isinstance(series.name, str) else None
    if _drawn_as_categories(series):
        categories = list(pd.Categorical(series).categories)
        palette = _category_palette(categories, colors)
        regions = collections.PatchCollection(
            rectangles,
            facecolor=[palette[value] for value in series[filled].tolist()],
            edgecolor="none",
            alpha=opacity,
        )
        axes.add_collection(regions)
        if legend:
            handles = [
                patches.Patch(facecolor=palette[value], alpha=opacity, label=str(value))
                for value in categories
            ]
            axes.legend(
                handles=handles,
                title=name,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=False,
            )
    else:
        numbers = series.to_numpy(dtype=np.float64)
        regions = collections.PatchCollection(
            rectangles, cmap=cmap, edgecolor="none", alpha=opacity
        )
        regions.set_array(numbers[filled])
        axes.add_collection(regions)
        if legend:
            axes.get_figure().colorbar(regions, ax=axes, label=name)

    axes.set_xlim(left, right)
    axes.set_ylim(bottom, top)
    axes.set_aspect("equal")
    axes.set_axis_off()
    if title is not None:
        axes.set_title(title)
    return axes


# --- Composition ----------------------------------------------------------------------------


def composition(
    result: AnnotationResult | AnnData,
    ax: Axes | None = None,
    normalize: bool = True,
    show_no_call: bool = True,
    missing_classes: Any = None,
    colors: Mapping[str, Any] | None = None,
    annotate: bool = True,
    name: str | None = None,
) -> Axes:
    """Draw the share of cells annotated with each class, as bars.

    Every class of the axis gets a bar, including the classes no cell was annotated with, so a
    class at zero reads as zero rather than as absent. A class the reference cannot express is
    hatched and named under the axis, because no cell could have been annotated with it.

    Args:
        result: Annotations to summarize, or the table of cells holding them.
        ax: Axes to draw on. A figure is created when none is given.
        normalize: Draw shares of the population rather than counts.
        show_no_call: Add a bar for the cells without an annotation.
        missing_classes: A `Reference`, or the names of the classes it holds no cell of. Read from
            the annotations when None; an empty sequence marks none.
        colors: Palette of a custom taxonomy, replacing the default palette.
        annotate: Write each bar's value above it, so a bar at zero stays legible.
        name: Annotation of a table to draw, the name it was stored under. May be left out where
            the table holds one.

    Returns:
        The axes drawn on, carrying one bar per class and, when asked for, one for the cells
        without an annotation.

    Raises:
        AlhenaValueError: If the annotations or colors are unusable, or a missing class is unknown.
    """
    result = _as_annotations(result, name)
    order = tuple(str(name) for name in result.class_order)
    missing = missing_class_names(result if missing_classes is None else missing_classes, order)
    labels = np.asarray(result.labels).reshape(-1)
    valid = np.asarray(result.valid, dtype=bool).reshape(-1)
    total = int(labels.shape[0])

    names = list(order)
    counts = [int(np.count_nonzero((labels == name) & valid)) for name in order]
    if show_no_call:
        names.append(NO_CALL_NAME)
        counts.append(total - int(sum(counts)))

    denominator = float(total) if normalize and total else 1.0
    heights = [count / denominator for count in counts]

    axes = _axes(ax, _CHART_SIZE)
    positions = np.arange(len(names))
    bars = axes.bar(
        positions,
        heights,
        color=class_palette(names, colors=colors),
        edgecolor="black",
        linewidth=0.6,
    )
    for drawn, bar in zip(names, bars, strict=True):
        if drawn in missing:
            bar.set_hatch(MISSING_HATCH)

    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=30, ha="right")
    axes.set_ylabel("fraction of all cells" if normalize else "cells")
    axes.set_ylim(bottom=0.0)
    note = _missing_note(missing)
    axes.set_xlabel(f"predicted class\n{note}" if note else "predicted class")
    if ax is None:
        axes.get_figure().set_layout_engine("constrained")

    if annotate:
        offset = 0.01 * max(max(heights, default=0.0), 1e-6)
        for position, drawn in enumerate(names):
            if drawn in missing:
                text = NOT_IN_REFERENCE
            elif normalize:
                text = f"{heights[position]:.3f}"
            else:
                text = str(counts[position])
            axes.text(
                position,
                heights[position] + offset,
                text,
                ha="center",
                va="bottom",
                fontsize="small",
            )
    return axes


# --- Confusion matrix -----------------------------------------------------------------------


def confusion(
    score_report: ScoreReport,
    ax: Axes | None = None,
    cmap: str = "Blues",
    colorbar: bool = True,
    missing_classes: Any = None,
    annotate: bool = True,
) -> Axes:
    """Draw a confusion matrix, normalizing each row over cells with a prediction.

    Each row shows how cells with predictions from one known class are distributed across predicted
    classes. The diagonal equals recall only when every cell of that known class received a
    prediction. Cells without a prediction count as errors in recall, totaled below the x-axis. A
    row with no predictions is marked n=0, with undefined fractions displayed as `-`.

    Args:
        score_report: Report to draw, supplying the matrix, the class order and the cell count.
        ax: Axes to draw on. A figure is created when none is given.
        cmap: Colormap of the shares.
        colorbar: Draw the scale beside the matrix.
        missing_classes: A `Reference`, or the names of the classes it holds no cell of. Read from
            the report when None; an empty sequence marks none.
        annotate: Write each share in its cell.

    Returns:
        The axes drawn on, carrying the matrix and, when asked for, the share written in each cell.

    Raises:
        AlhenaValueError: If a named missing class is not a class of the axis.
    """
    order = tuple(str(name) for name in score_report.class_order)
    chosen = score_report if missing_classes is None else missing_classes
    missing = missing_class_names(chosen, order)
    matrix = np.asarray(score_report.confusion, dtype=np.float64)
    support = matrix.sum(axis=1)
    shares = np.divide(
        matrix, support[:, None], out=np.full(matrix.shape, np.nan), where=support[:, None] > 0
    )

    axes = _axes(ax, _MATRIX_SIZE)
    drawn = axes.imshow(shares, vmin=0.0, vmax=1.0, cmap=cmap, interpolation="nearest")
    positions = np.arange(len(order))
    # Two-line labels clear their neighbors at 45 degrees; the caller's axes keep 30.
    steep = ax is None and bool(missing)
    axes.set_xticks(positions)
    axes.set_xticklabels(
        [f"{name}\n({NOT_IN_REFERENCE})" if name in missing else name for name in order],
        rotation=45 if steep else 30,
        ha="right",
    )
    axes.set_yticks(positions)
    axes.set_yticklabels(
        [f"{name} (n={int(count)})" for name, count in zip(order, support, strict=True)]
    )
    axes.set_ylabel("known class")

    no_call = int(score_report.n_cells) - int(matrix.sum())
    footnote = [
        f"{no_call} of {score_report.n_cells} cells carry no prediction\nand enter no column."
    ]
    note = _missing_note(missing)
    if note:
        footnote.append(note)
    axes.set_xlabel("predicted class\n" + "\n".join(footnote))

    if annotate:
        for row in range(len(order)):
            for column in range(len(order)):
                share = shares[row, column]
                defined = bool(np.isfinite(share))
                axes.text(
                    column,
                    row,
                    f"{share:.2f}" if defined else "-",
                    ha="center",
                    va="center",
                    color="white" if defined and share > 0.5 else "black",
                    fontsize="small",
                )
    # Before the colorbar: a constrained layout set after one fails when the figure is drawn.
    if ax is None:
        axes.get_figure().set_layout_engine("constrained")
        if steep:
            # Steeper labels need 5 points of padding, not 3, to keep the note inside the canvas.
            axes.get_figure().get_layout_engine().set(h_pad=5 / 72)
    if colorbar:
        # On two lines, so the label stays within a colorbar that long tick labels shorten.
        axes.get_figure().colorbar(drawn, ax=axes, label="fraction of predicted cells\nin each row")
    return axes


# --- Confidence sweep -----------------------------------------------------------------------


def confidence_curve(
    sweep: pd.DataFrame,
    ax: Axes | None = None,
    metric: str = "macro_f1",
    per_class: bool = True,
    show_retained: bool = True,
    missing_classes: Any = None,
    colors: Mapping[str, Any] | None = None,
) -> Axes:
    """Draw a confidence sweep, pooled and per class.

    Per-class recall is drawn beside the pooled curve, and the legend names every class whose recall
    falls. Each sampled threshold carries a marker, joined to the next by a straight segment,
    nothing interpolated; a threshold where a class retains no cell leaves a gap.

    Args:
        sweep: The table `metrics.confidence_sweep` returns.
        ax: Axes to draw on. A figure is created when none is given.
        metric: Pooled column drawn against the per-class recalls.
        per_class: Draw one recall curve per class.
        show_retained: Draw the retained fraction of the population on a right-hand axis, which a
            rising pooled curve is read against.
        missing_classes: A `Reference`, the annotations swept, or the names of the classes the
            reference holds no cell of. Read from the sweep when None.
        colors: Palette of a custom taxonomy, replacing the default palette.

    Returns:
        The axes drawn on, carrying the pooled curve and one line per class.

    Raises:
        AlhenaValueError: If the sweep is not a table, is empty, or lacks a column being drawn.
    """
    if not isinstance(sweep, pd.DataFrame):
        raise AlhenaValueError(
            f"Expected the table confidence_sweep returns, got {type(sweep).__name__}."
        )
    for column in ("threshold", metric):
        if column not in sweep.columns:
            raise AlhenaValueError(
                f"The sweep has no {column!r} column. Pass the table confidence_sweep returns, "
                f"whose columns are {list(sweep.columns)}."
            )
    if not len(sweep):
        raise AlhenaValueError("The sweep holds no threshold, so there is nothing to draw.")

    names = [column[len("recall_") :] for column in sweep.columns if column.startswith("recall_")]
    if missing_classes is None:
        # Those the sweep records, among the classes it still holds a recall column of.
        missing = tuple(name for name in sweep.attrs.get("missing_classes", ()) if name in names)
    else:
        missing = missing_class_names(missing_classes, names or None)
    thresholds = sweep["threshold"].to_numpy(dtype=float)

    axes = _axes(ax, _CHART_SIZE)
    axes.plot(
        thresholds,
        sweep[metric].to_numpy(dtype=float),
        color="black",
        marker="o",
        linewidth=2.0,
        label=metric.replace("_", " "),
    )
    if per_class:
        for name in names:
            recall = sweep[f"recall_{name}"].to_numpy(dtype=float)
            axes.plot(
                thresholds,
                recall,
                color=class_color(name, colors=colors),
                marker="o",
                markersize=4.0,
                linewidth=1.2,
                label=_recall_label(name, recall, name in missing),
            )
    axes.set_xlabel("confidence threshold")
    pooled = metric.replace("_", " ")
    axes.set_ylabel(f"{pooled} and per-class recall" if per_class and names else pooled)
    axes.set_ylim(0.0, 1.0)

    handles, labels = axes.get_legend_handles_labels()
    if show_retained and "retained_fraction" in sweep.columns:
        twin = axes.twinx()
        twin.plot(
            thresholds,
            sweep["retained_fraction"].to_numpy(dtype=float),
            color="0.45",
            linestyle="--",
            marker=".",
            label="cells retained",
        )
        twin.set_ylabel("fraction of cells retained")
        twin.set_ylim(0.0, 1.0)
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        handles = [*handles, *twin_handles]
        labels = [*labels, *twin_labels]
    axes.legend(handles, labels, fontsize="small", loc="best")
    return axes


def _recall_label(name: str, recall: np.ndarray, missing: bool) -> str:
    """Name a recall curve, saying where the class loses recall or has no reference cell."""
    if missing:
        return f"{name} ({NOT_IN_REFERENCE})"
    finite = recall[np.isfinite(recall)]
    if finite.size >= 2 and finite[-1] < finite[0]:
        return f"{name} (recall falls {finite[0]:.2f} to {finite[-1]:.2f})"
    return name


# --- Legend ---------------------------------------------------------------------------------


def legend(
    axis: str | Sequence[str] | AnnotationResult | AnnData = DEFAULT_AXIS,
    ax: Axes | None = None,
    missing_classes: Any = None,
    show_no_call: bool = True,
    colors: Mapping[str, Any] | None = None,
    ncols: int = 1,
    title: str | None = None,
    name: str | None = None,
) -> Axes:
    """Draw a standalone class legend, for the overlays that carry none.

    A class the reference cannot express is hatched and named.

    Args:
        axis: Name of a class axis, the class names themselves, or annotations or the table of
            cells holding them, whose classes and missing classes the legend names.
        ax: Axes to draw on. A figure is created when none is given.
        missing_classes: A `Reference`, or the names of the classes it holds no cell of. Read from
            annotations given as `axis` when None.
        show_no_call: Add an entry for the cells without an annotation.
        colors: Palette of a custom taxonomy, replacing the default palette.
        ncols: Columns of the legend.
        title: Title of the legend.
        name: Annotation of a table given as `axis`, the name it was stored under. May be left out
            where the table holds one.

    Returns:
        The axes drawn on, carrying the legend and no ticks.

    Raises:
        AlhenaValueError: If the axis or annotations are unusable, or a missing class is unknown.
    """
    patch = _require("matplotlib.patches").Patch
    if name is not None and not _is_table(axis):
        raise AlhenaValueError(
            "name picks one annotation of a table of cells, and no table was given. Leave name out."
        )

    if isinstance(axis, AnnotationResult) or _is_table(axis):
        axis = _as_annotations(axis, name)
        names = [str(name) for name in axis.class_order]
        missing_classes = axis if missing_classes is None else missing_classes
    elif isinstance(axis, str):
        if axis not in CLASS_AXES:
            raise AlhenaValueError(
                f"Unknown class axis {axis!r}. The declared axes are {sorted(CLASS_AXES)}; pass "
                "the class names themselves for a custom taxonomy."
            )
        names = [name for name in label_order(axis) if name != UNKNOWN_LABEL]
    else:
        names = [str(name) for name in axis]
    if not names:
        raise AlhenaValueError("No classes were given, so there is nothing to put in a legend.")
    missing = missing_class_names(missing_classes, names)

    handles = [
        patch(
            facecolor=class_color(name, colors=colors),
            edgecolor="black",
            linewidth=0.6,
            hatch=MISSING_HATCH if name in missing else None,
            label=f"{name} ({NOT_IN_REFERENCE})" if name in missing else name,
        )
        for name in names
    ]
    if show_no_call:
        handles.append(
            patch(
                facecolor=class_color("", colors=colors),
                edgecolor="black",
                linewidth=0.6,
                label=NO_CALL_NAME,
            )
        )

    axes = _axes(ax, _LEGEND_SIZE)
    axes.legend(handles=handles, loc="center", frameon=False, ncols=int(ncols), title=title)
    axes.set_axis_off()
    return axes
