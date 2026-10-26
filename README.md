<p align="center">
  <img src="docs/assets/alhena_banner.svg" alt="ALHENA" width="100%" height="168">
</p>

<p align="center">
  <b>ALHENA, an H&amp;E encoder distilled from transcriptomic foundation models for multi-scale slide analysis.</b>
</p>

<p align="center">
  <a href="https://prism-oncology.github.io/ALHENA/"><b>Documentation</b></a> |
  <a href="https://doi.org/10.64898/2026.10.20.760439"><b>Preprint</b></a> |
  <a href="https://github.com/prism-oncology/ALHENA-paper"><b>Paper code</b></a> |
  <a href="https://huggingface.co/prism-oncology/ALHENA"><b>Weights on Hugging Face</b></a> |
  <a href="https://huggingface.co/prism-oncology/ALHENA-references"><b>References on Hugging Face</b></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/alhena/"><img alt="PyPI" src="https://img.shields.io/pypi/v/alhena.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.23619657"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.23619657.svg"></a>
  <a href="https://github.com/prism-oncology/ALHENA/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/prism-oncology/ALHENA/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://github.com/prism-oncology/ALHENA/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/prism-oncology/ALHENA/actions/workflows/docs.yml/badge.svg"></a>
  <img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue.svg">
</p>

---

ALHENA reads an H&E section and returns two things: **embeddings**, per tile and per cell, and
**cell types** on an unlabeled section. No spatial transcriptomics is needed at inference, and
nothing is fitted at deployment.

Cell types come from retrieval against a labeled single-cell reference, so a different reference
gives a different taxonomy. ALHENA provides prebuilt references for fifteen human tissues, built
from fourteen public single-cell atlases, or you can build a custom reference.

When the inputs share a spatial frame, the embeddings and the cell types stay aligned with the
slide and can be analyzed together across scales.

ALHENA is gamma Geminorum, from the Arabic *al-han'ah*, a brand mark placed on something to make it
readable. To the eye, it appears as a single star, though it is in fact a binary system. The light
we see comes from a bright A-type subgiant. Its companion has never been imaged and reveals itself
only through the faint wobble it leaves in the motion of the visible star. The H&E section is that
visible star, inexpensive, familiar, and collected everywhere. The spatial transcriptome is the
unseen companion, physically present, shaping what we observe, yet never directly accessible at
inference.

## Install

ALHENA needs Python 3.12 or later, and runs with or without a GPU.

```bash
pip install alhena
```

Optional extras add plotting (`viz`), whole-slide reading (`wsi`) and SpatialData support
(`spatial`): `pip install "alhena[viz,wsi,spatial]"`.

The UNI2-h model weights are distributed separately by their authors and are not included with
ALHENA. Their repository is gated: request access at
[MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) and accept the license. Then log
in and download the weights, 2.7 GB, into the Hugging Face cache, where ALHENA finds them:

```bash
hf auth login
hf download MahmoodLab/UNI2-h pytorch_model.bin
```

For weights stored elsewhere, set `ALHENA_BACKBONE` to the file. ALHENA adds its own low-rank
adapters and projection heads on top of the UNI2-h weights, and downloads them from Hugging Face
on first use, with the prebuilt reference of each tissue it annotates.

## See it work

The demo needs no data of your own. It reads a few annotated breast H&E fields from the public
PanNuke dataset, annotates their nuclei with the prebuilt breast reference, and compares the
predicted cell types with the PanNuke labels. It needs the UNI2-h model weights.

```bash
alhena demo Breast
```

## Use it on your own data

```python
import alhena

source = alhena.io.read_slide("slide.svs")                        # an H&E slide, read tile by tile
nuclei = alhena.io.read_polygons("nuclei.geojson", tiles=source)  # nucleus outlines

encoder = alhena.load_encoder()
tiles = encoder.encode_tiles(source)                              # one embedding per tile
tiles.write_h5ad("tiles.h5ad")                                    # one row per tile, ready to analyze

cells = encoder.encode_cells(source, nuclei=nuclei)               # one pass, each tile once
cells.write_h5ad("cells.h5ad")

typer = alhena.load_celltyper(tissue="Breast")                    # loads the reference alone
result = typer.annotate(cells)                                    # retrieval, no encoder
result.to_frame()
```

`read_slide` needs the `wsi` extra and reads the resolution the slide declares. `source` can also
be an array of H&E tiles, `(n_tiles, height, width, 3)` uint8 RGB; `mpp=` then declares the
resolution it is already at. The nuclei come from a segmenter of your choice: ALHENA does not
segment. The documentation shows how to
[segment a slide with sopa and StarDist](https://prism-oncology.github.io/ALHENA/inputs/nuclei/#segment-a-slide-with-sopa-and-stardist).
`read_polygons` expects the outlines in the slide's full-resolution pixels. Outlines written in
microns need `units="um"`, and outlines segmented on another image need `transform=`. See
[Coordinates](https://prism-oncology.github.io/ALHENA/inputs/coordinates/).
With SpatialData, `alhena.spatial` reads the image and the nucleus outlines from the object.

Cells are encoded once and can be annotated again, against another reference or on another
annotation axis, without rerunning the image encoder: `alhena.io.read_cells("cells.h5ad")` reads
them back with the provenance that pairs them with a reference.
`typer.annotate(source, nuclei=nuclei)` runs both steps in one call and keeps the cells on
`result.cells`.

## Documentation

**[prism-oncology.github.io/ALHENA](https://prism-oncology.github.io/ALHENA/)**

Start with [Choose your task](https://prism-oncology.github.io/ALHENA/start/choose-your-task/):
tile embeddings, cell embeddings, cell-type annotation, or multi-scale slide analysis.
[Inputs](https://prism-oncology.github.io/ALHENA/inputs/) says how to provide your images and
nuclei, whatever form they take.

Six [tutorials](https://prism-oncology.github.io/ALHENA/tutorials/) cover these tasks, custom
references, and annotation evaluation. The [use case example](https://prism-oncology.github.io/ALHENA/use-case/crc_multiscale/) applies
ALHENA to two public colorectal cancer cohorts, from tissue contexts to patient-level analyses.

## Contributing

Report a problem or ask a question in the [issues](https://github.com/prism-oncology/ALHENA/issues);
[CONTRIBUTING.md](https://github.com/prism-oncology/ALHENA/blob/main/CONTRIBUTING.md) explains how to
install ALHENA for development and run its tests.

## License

The ALHENA software is released under the BSD 3-Clause License. The ALHENA model weights,
comprising the LoRA adapters and the projection heads used at inference, are released under the
Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

The precomputed single-cell reference representations are released under the CC BY-NC 4.0
license. They remain subject to the terms of their source atlases, and those derived from DISCO
are provided for academic research only, in accordance with the DISCO terms of use.

## Citation

If you use ALHENA, please cite the [preprint](https://doi.org/10.64898/2026.10.20.760439):

```bibtex
@article{alhena,
  title   = {{ALHENA}, an {H\&E} encoder distilled from transcriptomic foundation models
             for multi-scale slide analysis},
  author  = {Giraud-Sauveur, F{\'e}licie and Blampey, Quentin and Marza, Pierre and
             Benkirane, Hakim and Vakalopoulou, Maria and Courn{\`e}de, Paul-Henry and
             Christodoulidis, Stergios},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.20.760439}
}
```

Please also cite the models and datasets ALHENA builds on. The
[Cite page](https://prism-oncology.github.io/ALHENA/cite/) of the documentation lists them.
