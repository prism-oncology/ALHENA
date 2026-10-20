# Cite

## ALHENA

If you use ALHENA, please cite it using the metadata provided in
[`CITATION.cff`](https://github.com/prism-oncology/ALHENA/blob/main/CITATION.cff). On GitHub, the
same citation metadata is available through **Cite this repository**.

## Models and datasets used by ALHENA

ALHENA builds on external models and public single-cell datasets that should be cited separately
when they contribute to an analysis.

| Component | What it is |
|---|---|
| [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | Pathology foundation encoder used as the H&E backbone. Distributed separately by its authors, who ask that [Chen et al. (Nat Med, 2024)](https://doi.org/10.1038/s41591-024-02857-3) be cited. |
| [scConcept](https://github.com/theislab/scConcept) | Single-cell foundation model used as the cell-level teacher during contrastive distillation, presented in [Bahrami et al. (bioRxiv, 2025)](https://doi.org/10.1101/2025.10.14.682419). |
| [Novae](https://github.com/prism-oncology/novae) | Neighborhood-level spatial-transcriptomics teacher used during contrastive distillation, presented in [Blampey et al. (Nat Methods, 2025)](https://doi.org/10.1038/s41592-025-02899-6). |
| [DISCO](https://www.immunesinglecell.org/) | Source of eleven of the fourteen prebuilt single-cell reference atlases, presented in [Li et al. (Nucleic Acids Res, 2022)](https://doi.org/10.1093/nar/gkab1020). DISCO states: "All data and scripts in DISCO are free for academic research." |
| [Tabula Sapiens](https://cellxgene.cziscience.com/) | Source of the `prostate` and `uterus` prebuilt references through CELLxGENE. CC BY 4.0, with attribution to [Tabula Sapiens Consortium et al. (Science, 2022)](https://doi.org/10.1126/science.abl4896). |
| [Gene Expression Omnibus, GSE208653](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE208653) | Source of the `cervix` prebuilt reference. Deposited by [Guo et al. (Clin Transl Med, 2023)](https://doi.org/10.1002/ctm2.1219). No restriction stated by GEO. |
| [PanNuke](https://huggingface.co/datasets/RationAI/PanNuke) | Public nucleus dataset used by the demo and [Tutorial 6](tutorials/06_annotation_evaluation.ipynb), presented in [Gamper et al. (ECDP, 2019)](https://doi.org/10.1007/978-3-030-23937-4_2). CC BY-NC-SA 4.0, non-commercial. |
| [BioImage Archive S-BIAD1407](https://doi.org/10.6019/S-BIAD1407) | Colorectal cancer H&E slides used by the [use case example](use-case/crc_multiscale.ipynb). CC BY 4.0. |
| [Mendeley Data 9z48yfzr63](https://doi.org/10.17632/9z48yfzr63.1) | Nucleus coordinates and published cell labels used by the use case example. CC BY 4.0. |
| [Trahearn et al. (J Pathol, 2025)](https://doi.org/10.1002/path.6378) | Study that describes the BRAF and MISSONI colorectal cancer cohorts. Source of the microsatellite instability status and the 44.65 µm proximity threshold used by the use case example. CC BY 4.0. |

## Figures in this documentation

The example figures show a region of the H&E slide of
[FFPE Human Breast with Custom Add-on Panel](https://www.10xgenomics.com/datasets/ffpe-human-breast-with-custom-add-on-panel-1-standard),
the infiltrating ductal carcinoma sample, by 10x Genomics, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The image is cropped, downsampled and
overlaid with ALHENA results.

## Licenses

The ALHENA software is released under the BSD 3-Clause License.

The ALHENA model weights, comprising the LoRA adapters and the projection heads used at inference,
are released under the Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0). ALHENA requires the UNI2-h weights, which are not redistributed and must be
obtained from [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) under the license and
terms of use of their authors.

The precomputed single-cell reference representations are released under the CC BY-NC 4.0
license. They remain subject to the terms of their source atlases, and those derived from DISCO
are provided for academic research only, in accordance with the DISCO terms of use. No expression
values, gene names or cell barcodes from the source atlases are redistributed.

The table of models and datasets above gives the source of each prebuilt reference, its terms and
the study to credit.

Consult the corresponding source and license information before redistributing model or reference
artifacts.

---

**Back to:** [Home](index.md)
