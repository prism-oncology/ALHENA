# References

A single-cell reference is required only for cell-type annotation. Tile and cell embeddings can be
computed without one.

Each reference contains labeled single-cell profiles projected into the same shared space used by
ALHENA's projected cell embeddings. During annotation, image-derived cells are compared with nearby
reference cells, whose labels are used to assign a cell type. Only classes represented in the
selected reference can be returned.

```python
import alhena

typer = alhena.load_celltyper(tissue="Breast")
result = typer.annotate(cells)
```

## Prebuilt references

ALHENA provides prebuilt references for fifteen human tissues: `Breast`, `BoneMarrow`, `Cervix`,
`Colon`, `Heart`, `Kidney`, `Liver`, `Lung`, `LymphNode`, `Ovary`, `Pancreas`, `Prostate`,
`Skin`, `Tonsil`, and `Uterus`. `LymphNode` and `Tonsil` use the same reference file, so fourteen
files cover the fifteen tissue names.

A reference is downloaded on first use, one tissue at a time, and cached. Annotating breast
tissue never downloads the heart reference.

```python
print(alhena.available_tissues())
```

Some tissues can also be requested through accepted aliases, including names of sampled anatomical
regions such as `Rectum` or `Endometrium`. Run `alhena reference list` to see all accepted names.
See [Tissue aliases](../concepts/references.md#tissue-aliases).

## Check class support

Before interpreting annotations, check which classes are represented in the selected reference. A
class with no reference cells cannot be predicted, while thinly supported classes may have
less robust neighbor support.

```python
info = alhena.describe_shard(tissue="Tonsil", axis="lineage_v1")
print(info.missing_classes, info.thin_classes)
```

```bash
alhena reference describe Tonsil
```

`describe_shard()` inspects the reference without loading the full embedding matrix. The same
class-support information is stored in annotation results as `meta["unexpressed_classes"]` and
`meta["thinly_supported_classes"]`. See [Class support](../concepts/references.md#class-support)
for the prebuilt references.

## The class axis

Cell annotations are produced on one class axis, selected when the reference is loaded:

| Axis | Classes |
|---|---|
| `lineage_v1`, the default | `B_Plasma`, `Epithelial_Melanocyte`, `Myeloid`, `Stromal_Vascular`, `T_NK` |
| `compartment_v1` | `Epithelial_Melanocyte`, `Immune`, `Stromal_Vascular` |

```python
typer = alhena.load_celltyper(tissue="Breast", axis="compartment_v1")
```

The two axes define different classification problems and their performance metrics should
therefore be interpreted separately. See
[The two class axes](../concepts/cell-typing.md#the-two-class-axes).

## Human tissue

The prebuilt ALHENA references are derived from human single-cell atlases. Using them to annotate
another species would still return the nearest available human-reference class, but that
prediction would be outside the scope represented by the reference. Computing H&E embeddings does
not require a reference, although ALHENA has not been validated on non-human tissue. See
[Can I use ALHENA on tissue from another species?](../faq.md#can-i-use-alhena-on-tissue-from-another-species).

## Working offline

To work without a network, fetch the reference beforehand, or pass a file on disk:

```bash
alhena download --tissue Breast
```

```python
typer = alhena.load_celltyper(reference="/shards/breast.safetensors")
```

Annotating previously computed `CellEmbeddings` requires only the reference; the UNI2-h weights
and encoder bundle are not used again. See [The cache](../installation.md#the-cache) for the local
resource locations.

## The encoder and the reference must match

The encoder and reference must use the same learned cell space. ALHENA records this compatibility
through `cell_space_id` and checks it before annotation. Previously saved `CellEmbeddings` also
retain their `cell_space_id`. A mismatch raises `CellSpaceMismatchError`.

Embedding dimensionality alone cannot establish compatibility because different releases may use
the same 512-dimensional representation. The `cell_space_id` check prevents cells and references
from different learned spaces from being compared accidentally. See
[Reference compatibility](../concepts/references.md#reference-compatibility).

## A tissue without a prebuilt reference

If no prebuilt reference is available for the tissue:

- **Compute embeddings normally.** Tile and cell embeddings do not require a reference.
- **Build a reference** from a single-cell atlas of that tissue. See
  [Build a custom reference](#build-a-custom-reference).
- **Do not substitute a neighboring organ solely because it has a reference.** The available cell
  populations may differ substantially. Tissue aliases should only be used when they correspond to
  regions explicitly covered by the reference.

## Build a custom reference

A custom reference allows cell annotation with your own labeled single-cell atlas and taxonomy.
Once the reference file has been built, it is loaded and used in the same way as a prebuilt
reference:

```python
typer = alhena.load_celltyper(reference="my_reference.safetensors", axis="my_taxonomy_v1")
result = typer.annotate(cells)
```

The same custom reference can be used from the command line by providing its file and annotation
axis:

```bash
alhena annotate cells.h5ad --reference my_reference.safetensors --axis my_taxonomy_v1 --out annotations.h5ad
```

Creating a new reference does not require re-encoding the H&E images. Previously computed
`CellEmbeddings` can be annotated with the custom reference directly. See
[Tutorial 5](../tutorials/05_custom_reference.ipynb) for the complete workflow and
[Building a custom reference](../concepts/references.md#building-a-custom-reference) for the
underlying concepts.

---

**Next:** [Outputs](../outputs/index.md)
