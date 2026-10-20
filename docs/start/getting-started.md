# Getting started

Install ALHENA, make the UNI2-h model weights available, verify the setup, and run the demo on a
small public example.

## 1. Install

ALHENA requires Python 3.12 or later.

=== "Base installation"

    ```bash
    pip install alhena
    ```

=== "With optional extras"

    ```bash
    pip install "alhena[viz,wsi,spatial]"
    ```

`viz` adds plotting utilities, `wsi` enables whole-slide reading, and `spatial` adds SpatialData
support. See [Optional extras](../installation.md#optional-extras) for the packages included in
each extra.

Installation takes one to two minutes on macOS, longer on Linux, where PyTorch's default build
includes CUDA. See [Requirements](../installation.md#requirements) for the platforms and
versions tested.

## 2. Set up the UNI2-h model weights

ALHENA uses UNI2-h as its image encoder and adds low-rank adapters and projection heads. The
UNI2-h model weights are distributed separately by their authors and are not included with
ALHENA.

The UNI2-h repository is gated. Request access at <https://huggingface.co/MahmoodLab/UNI2-h> and
accept the license. Once access is granted, log in and download the weights, 2.7 GB, into the
Hugging Face cache, where ALHENA finds them:

```bash
hf auth login
hf download MahmoodLab/UNI2-h pytorch_model.bin
```

`hf auth login` asks for an access token of your Hugging Face account. To keep the weights
elsewhere, set `ALHENA_BACKBONE` to the file. See
[How ALHENA locates the UNI2-h weights](../installation.md#the-uni2-h-model-weights).

## 3. Verify the setup

```bash
alhena info
```

`alhena info` reports the ALHENA cache location, whether the UNI2-h weights are available, and
which ALHENA artifacts are already cached. If something is missing, it also indicates what still
needs to be downloaded.

## 4. Run the demo

```bash
alhena demo Breast
```

The demo uses a small set of annotated breast H&E fields from the public PanNuke dataset. ALHENA
encodes the nuclei, annotates them with the prebuilt breast reference, and compares the
predicted cell types with the PanNuke labels.

The UNI2-h weights must be available before the demo is run. On first use, ALHENA downloads and
caches the encoder bundle (45 MB) and the breast reference (179 MB). The PanNuke fields are read
from the dataset repository on each run and are not cached. The demo runs in one to two minutes
on a recent laptop CPU. It prints a report and does not write output files.

The [demo notebook](demo.ipynb) shows the same workflow step by step with the Python API.

## Next: choose your task

Choose the workflow that matches your goal: compute tile embeddings, compute cell embeddings,
annotate cells, or combine these outputs for multi-scale slide analysis.

---

**Next:** [Choose your task](choose-your-task.md)
