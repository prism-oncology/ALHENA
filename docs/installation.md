# Installation

This page describes the ALHENA installation, optional dependencies, UNI2-h model weights, and
local cache. For the shortest path to a working setup, see
[Getting started](start/getting-started.md).

```bash
pip install alhena
```

## Requirements

ALHENA requires Python 3.12 or later. Its continuous integration runs on Linux (Ubuntu): Python
3.12, 3.13 and 3.14 with the latest releases of the dependencies, and Python 3.12 with the oldest
releases `pyproject.toml` allows, such as PyTorch 2.2. The tests also run on macOS with Apple
silicon. A GPU is optional, and no special hardware is needed.

On macOS, installing ALHENA into a new environment takes one to two minutes with a broadband
connection: pip downloads about 280 MB, most of it PyTorch. On Linux, pip's default PyTorch
build includes CUDA libraries and downloads several gigabytes.

## What comes with the package

The base installation includes the dependencies required to compute tile and cell embeddings,
perform cell-type annotation, and read or write standard ALHENA result files.

Major dependencies include PyTorch, timm, NumPy, pandas, PyArrow, OpenCV, huggingface-hub,
safetensors, h5py, tqdm, and AnnData.

Optional functionality such as whole-slide reading, SpatialData integration, and plotting is
installed separately through extras.

!!! tip "A GPU is optional"
    ALHENA can run entirely on CPU. By default, `load_encoder()` uses CUDA when available and
    falls back to CPU otherwise. Set `device="cpu"` or `device="cuda"` explicitly when needed. On
    Apple silicon, `device="mps"` can be used to run the image encoder on the Apple GPU. Where
    pip's default PyTorch build has no CUDA, install a CUDA build first, following the
    [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Optional extras

| Extra | Installs | Use it for |
|---|---|---|
| `viz` | `matplotlib` | ALHENA plotting utilities |
| `spatial` | `spatialdata`, `sopa` | reading and writing SpatialData objects, GeoParquet nuclei, and the sopa patch workflow |
| `wsi` | `tiffslide >= 2.5.1` | reading supported whole-slide formats with `read_slide()` and `read_region()` |
| `dev` | `pytest`, `ruff` | development, testing, and linting |
| `docs` | MkDocs documentation dependencies | building this documentation from the repository |

Install one or more extras with:

```bash
pip install "alhena[viz,wsi]"
```

??? note "A missing extra is reported when it is needed"
    Each optional dependency is imported inside the function that needs it, so an installation
    without the extras imports and runs. A call that needs a missing extra raises a
    `MissingExtraError`, an `ImportError` naming the extra to install. The lower bound on
    tiffslide matters, and [Supported slide formats](inputs/images.md#supported-slide-formats)
    explains why.

## The repository

The installed Python package contains the ALHENA library. Tutorials, the reference-building
script, its dedicated environment, and the documentation sources are distributed in the
[ALHENA repository](https://github.com/prism-oncology/ALHENA).

Building a custom reference requires the repository because the atlas-embedding script and its
dedicated environment are not included in the Python wheel. See
[Building a custom reference](concepts/references.md#building-a-custom-reference).

## The UNI2-h model weights

ALHENA uses UNI2-h as its image encoder and adds its own low-rank adapters and projection heads.
The UNI2-h model weights are distributed separately by their authors and are not included with
ALHENA.

The UNI2-h repository on Hugging Face is gated. Request access on
[MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) with a Hugging Face account, and
accept the license. Once access is granted, log in with an access token of that account and
download `pytorch_model.bin`, 2.7 GB, into the Hugging Face cache:

```bash
hf auth login
hf download MahmoodLab/UNI2-h pytorch_model.bin
```

The `hf` command comes with huggingface-hub, a dependency of ALHENA.

ALHENA looks for the weights in this order:

1. the path passed explicitly through `backbone=`;
2. the path stored in `ALHENA_BACKBONE`;
3. the local Hugging Face model cache, where `hf download` puts them.

To use weights stored elsewhere, set `ALHENA_BACKBONE` to the file. An `export` lasts for the
session; add it to the shell's profile to keep it:

```bash
export ALHENA_BACKBONE=/path/to/UNI2-h/pytorch_model.bin
```

If the weights are already present in the Hugging Face cache, no environment variable is required.

If the weights cannot be found, ALHENA raises `BackboneNotFoundError` and reports the configured
location or Hugging Face cache that was searched. See [Errors and warnings](api/errors.md).

ALHENA checks the weights it loads. The published encoder bundle declares the digest of the UNI2-h
model weights its adapters were trained on, and a file holding other weights, such as a converted,
fine-tuned or damaged copy, is refused with `BundleError`. The digest is recorded as
`encoder.info.backbone_digest`, and in the tables of embeddings ALHENA writes.

??? example "A named location that holds nothing"
    If you explicitly name either of the first two locations and it does not contain the weights,
    ALHENA reports that location instead of silently falling through to the next lookup. This makes
    it clear which configured path is wrong.

    ```text
    ALHENA_BACKBONE: /data/uni2h does not exist.

    ALHENA_BACKBONE: directory /data/uni2h contains none of
    pytorch_model.bin, model.safetensors.
    ```

??? note "Advanced: Hugging Face cache lookup"
    ALHENA searches for `pytorch_model.bin` and then `model.safetensors` in the cache used by
    `huggingface_hub`.

    The cache location follows `HF_HUB_CACHE`, then `HF_HOME/hub`, and finally the default
    Hugging Face cache location. If these environment variables are changed after the weights were
    downloaded, the new configuration may point to a different cache.

    `alhena.hub.find_backbone()` reports the location that was searched when the weights cannot be
    found.

## The cache

The ALHENA cache stores resources downloaded by ALHENA itself, including the encoder bundle,
reference registry, and prebuilt reference files.

It is separate from the Hugging Face cache used for the UNI2-h model weights.

The location of the ALHENA cache is the first of these that is set:

1. the `cache_dir` argument, wherever a call takes one;
2. the environment variable **`ALHENA_CACHE`**;
3. `$XDG_CACHE_HOME/alhena`;
4. a per-user cache directory: `~/Library/Caches/alhena` on macOS,
   `%LOCALAPPDATA%\alhena\Cache` on Windows, `~/.cache/alhena` elsewhere.

```bash
export ALHENA_CACHE=/scratch/alhena
```

Read the resolved location back without creating it:

```python
from alhena.hub import cache_root

print(cache_root(create=False))
```

`alhena download` populates this cache using the standard Hugging Face repository layout. A cache
prepared on another machine can therefore be copied or shared for offline use.

!!! note "`ALHENA_CACHE` does not control the UNI2-h weights"
    `ALHENA_CACHE` configures resources downloaded by ALHENA. UNI2-h is located independently
    through `backbone=`, `ALHENA_BACKBONE`, or the Hugging Face model cache.

## Downloaded ALHENA resources

Two ALHENA resources may be downloaded when first needed.

**Encoder bundle.** The bundle contains `config.json`, `adapters.safetensors`, and
`heads.safetensors` and is shared across tissues.

**Prebuilt references.** Cell annotation downloads only the reference required for the selected
tissue, together with the small reference registry. Reference size varies substantially by
tissue.

UNI2-h is not part of either resource and is obtained separately from its authors.

??? note "Approximate download sizes"
    The encoder bundle is approximately 45 MB. Individual prebuilt reference files range from
    approximately 22 MB to 641 MB.

??? note "Nothing is fetched at import time"
    All network access occurs inside explicit function calls. Importing the package itself makes no
    network request.

??? note "Downloaded files are verified"
    ALHENA checks downloaded bundle and reference files against the digests declared by their
    metadata, and checks the bundle against the identities its configuration declares. Files that
    fail validation are refused.

??? tip "Pre-download resources when needed"
    Resources can be downloaded explicitly before a run:

    ```bash
    alhena download --bundle --tissue Breast Skin
    ```

## Verify the installation

```bash
alhena info
```

`alhena info` reports the ALHENA version, resolved cache, availability of the UNI2-h weights,
cached encoder bundle, and cached references.

When the encoder bundle is available, it also reports the encoder identity and `cell_space_id`.
The latter is used to verify compatibility between encoded cells and single-cell references.

See [`alhena info`](cli.md#alhena-info) for the command-line options.

---

**Next:** [Command line](cli.md)
